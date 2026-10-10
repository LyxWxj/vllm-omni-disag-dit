# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, identity-checked transport contract for queued PP edges."""

from __future__ import annotations

import queue
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from vllm_omni.diffusion.distributed.group_coordinator import TensorMetadata


@dataclass(frozen=True)
class PipelineMessage:
    batch_id: str
    step_index: int
    epoch: int
    payload: Any


class PipelineEdgeKind(StrEnum):
    ACTIVATION = "activation"
    FEEDBACK = "feedback"


def pipeline_payload_metadata(payload: dict[str, torch.Tensor | Any]) -> tuple[tuple[str, Any], ...]:
    metadata: list[tuple[str, Any]] = []

    def visit(values: dict[str, torch.Tensor | Any], prefix: str = "") -> None:
        for key, value in values.items():
            flattened_key = prefix + key
            if isinstance(value, torch.Tensor):
                metadata.append(
                    (
                        flattened_key,
                        TensorMetadata(value.device.type, value.dtype, tuple(value.size())),
                    )
                )
            elif isinstance(value, dict):
                if not value:
                    metadata.append((flattened_key, value))
                else:
                    visit(value, flattened_key + "%")
            else:
                metadata.append((flattened_key, value))

    visit(payload)
    return tuple(metadata)


@dataclass(frozen=True)
class PipelineTransferOffer:
    """Metadata-only readiness offer for one directed PP transfer."""

    batch_id: str
    step_index: int
    epoch: int
    edge_kind: PipelineEdgeKind
    src_rank: int
    dst_rank: int
    payload_metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.batch_id or self.step_index < 0 or self.epoch < 0:
            raise ValueError("invalid pipeline transfer identity")
        if not isinstance(self.edge_kind, PipelineEdgeKind):
            raise ValueError("pipeline transfer requires a valid edge kind")
        if self.src_rank < 0 or self.dst_rank < 0 or self.src_rank == self.dst_rank:
            raise ValueError("pipeline transfer endpoints must be distinct non-negative ranks")

    @property
    def identity(self) -> tuple[str, int, int, PipelineEdgeKind, int, int]:
        return (
            self.batch_id,
            self.step_index,
            self.epoch,
            self.edge_kind,
            self.src_rank,
            self.dst_rank,
        )


@dataclass
class PipelineTransportProgress:
    offers: list[PipelineTransferOffer] = field(default_factory=list)
    completions: list[tuple[Any, ...]] = field(default_factory=list)
    readiness: list[tuple[Any, ...]] = field(default_factory=list)


@dataclass
class TransferTicket:
    message: PipelineMessage
    started: bool = False
    completed: bool = False


@dataclass
class _PendingReceive:
    message: PipelineMessage
    identity: tuple[str, int, int]
    handles: list[Any]
    postprocess: list[Any]
    metadata_future: Future[Any] | None = None
    handles_verified: bool = False
    postprocess_index: int = 0
    failure: BaseException | None = None


class DistributedP2PTransport:
    """Granted tensor-dictionary P2P over the existing PP group primitives."""

    def __init__(
        self,
        *,
        group: Any,
        local_rank: int,
        edge_kind: PipelineEdgeKind,
        src_rank: int,
        dst_rank: int,
    ) -> None:
        if local_rank not in {src_rank, dst_rank}:
            raise ValueError("local rank must be a transport endpoint")
        group_ranks = tuple(getattr(group, "ranks", ()))
        if src_rank not in group_ranks or dst_rank not in group_ranks:
            raise ValueError("transport endpoints must belong to the supplied PP group")
        group_rank = getattr(group, "rank", None)
        rank_in_group = getattr(group, "rank_in_group", None)
        if group_rank is None and rank_in_group is not None:
            group_rank = group_ranks[rank_in_group]
        if group_rank is None:
            raise ValueError("supplied PP group does not expose its local physical rank")
        if group_rank != local_rank:
            raise ValueError("local rank does not match the supplied PP group rank")
        if rank_in_group is not None and group_ranks[rank_in_group] != group_rank:
            raise ValueError("supplied PP group rank metadata is inconsistent")
        self.group = group
        self.local_rank = local_rank
        self.edge_kind = edge_kind
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self._src_group_rank = group_ranks.index(src_rank)
        self._dst_group_rank = group_ranks.index(dst_rank)
        self._send_handles: dict[tuple[str, int, int], list[Any] | None] = {}
        self._completed_send_ids: set[tuple[str, int, int]] = set()
        self._pending_receives: deque[_PendingReceive] = deque()
        self._ready_receives: deque[PipelineMessage] = deque()
        self._active_receive_ids: set[tuple[str, int, int]] = set()
        self._completed_receive_ids: set[tuple[str, int, int]] = set()
        self._metadata_receives: queue.Queue[Future[Any] | None] = queue.Queue()
        self._metadata_thread = threading.Thread(
            target=self._receive_metadata_loop,
            daemon=True,
            name="DiffusionP2PMetadataReceive",
        )
        self._metadata_thread.start()
        self._closed = False

    def _receive_metadata_loop(self) -> None:
        while True:
            future = self._metadata_receives.get()
            if future is None:
                return
            try:
                device = getattr(self.group, "device", None)
                if device is not None:
                    from vllm_omni.platforms import current_omni_platform

                    current_omni_platform.set_device(device)
                future.set_result(self.group.irecv_tensor_dict(src=self._src_group_rank))
            except BaseException as exc:
                future.set_exception(exc)

    @property
    def has_outstanding_operations(self) -> bool:
        return bool(self._send_handles or self._pending_receives or self._ready_receives or self._active_receive_ids)

    def start_transfer(self, offer: PipelineTransferOffer, message: PipelineMessage | None = None) -> None:
        self._ensure_open()
        if offer.edge_kind is not self.edge_kind or offer.src_rank != self.src_rank or offer.dst_rank != self.dst_rank:
            raise ValueError("pipeline transfer offer does not match this P2P transport")
        if self.local_rank == self.src_rank:
            if message is None:
                raise ValueError("sender requires a pipeline message payload")
            self._validate_message_matches_offer(message, offer)
            self._start_send(message)
            return
        if message is not None:
            raise ValueError("receiver must not provide a sender payload")
        identity = (offer.batch_id, offer.step_index, offer.epoch)
        if identity in self._active_receive_ids or identity in self._completed_receive_ids:
            raise ValueError("duplicate distributed P2P receive")
        # Register before posting the receive. Metadata is blocking inside the
        # coordinator, so keep it off the StageEngine owner thread.
        self._active_receive_ids.add(identity)
        metadata_future: Future[Any] = Future()

        self._metadata_receives.put(metadata_future)
        self._pending_receives.append(
            _PendingReceive(
                message=PipelineMessage(
                    batch_id=offer.batch_id,
                    step_index=offer.step_index,
                    epoch=offer.epoch,
                    payload=None,
                ),
                identity=identity,
                handles=[],
                postprocess=[],
                metadata_future=metadata_future,
            )
        )

    def _start_send(self, message: PipelineMessage) -> None:
        self._ensure_open()
        if self.local_rank != self.src_rank:
            raise RuntimeError("only the source endpoint can send")
        if not isinstance(message.payload, dict):
            raise TypeError("distributed P2P payload must be a tensor dictionary")
        identity = self._message_identity(message)
        if identity in self._send_handles or identity in self._completed_send_ids:
            raise ValueError("duplicate distributed P2P send identity")
        # Register before entering the blocking metadata send. Ambiguous
        # backend failure retains ownership and prevents replay.
        self._send_handles[identity] = None
        # Always let the coordinator send metadata asynchronously. The offer's
        # metadata is a validation contract, not a second communication path.
        handles = self.group.isend_tensor_dict(message.payload, dst=self._dst_group_rank)
        self._send_handles[identity] = list(handles)

    def poll(self, limit: int | None = None) -> list[PipelineMessage]:
        self._ensure_open()
        if limit is None or type(limit) is not int or limit <= 0:
            raise ValueError("distributed P2P polling requires a positive limit")
        if self._ready_receives:
            return [self._ready_receives.popleft() for _ in range(min(limit, len(self._ready_receives)))]
        ready: list[PipelineMessage] = []
        while self._pending_receives and len(ready) < limit:
            pending = self._pending_receives[0]
            if pending.failure is not None:
                raise pending.failure
            if pending.metadata_future is not None:
                if not pending.metadata_future.done():
                    break
                try:
                    tensor_dict, handles, postprocess = pending.metadata_future.result()
                    pending.message = PipelineMessage(
                        batch_id=pending.message.batch_id,
                        step_index=pending.message.step_index,
                        epoch=pending.message.epoch,
                        payload=tensor_dict,
                    )
                    pending.handles = list(handles)
                    pending.postprocess = list(postprocess)
                    pending.metadata_future = None
                except BaseException as exc:
                    pending.failure = exc
                    raise
            if not all(handle.is_completed() for handle in pending.handles):
                break
            try:
                if not pending.handles_verified:
                    for handle in pending.handles:
                        if handle.wait() is False:
                            raise RuntimeError("distributed P2P receive Work reported unsuccessful completion")
                    pending.handles_verified = True
                while pending.postprocess_index < len(pending.postprocess):
                    pending.postprocess[pending.postprocess_index]()
                    pending.postprocess_index += 1
            except BaseException as exc:
                pending.failure = exc
                self._ready_receives.extend(ready)
                raise
            self._pending_receives.popleft()
            self._active_receive_ids.remove(pending.identity)
            self._completed_receive_ids.add(pending.identity)
            ready.append(pending.message)
        return ready

    def wait(self, ticket: TransferTicket) -> bool:
        self._ensure_open()
        handles = self._send_handles.get(self._message_identity(ticket.message))
        if handles is None:
            if self._message_identity(ticket.message) in self._send_handles:
                raise RuntimeError("distributed P2P send launch outcome is ambiguous")
            raise ValueError("unknown distributed P2P send ticket")
        for handle in handles:
            if handle.wait() is False:
                raise RuntimeError("distributed P2P send Work reported unsuccessful completion")
        identity = self._message_identity(ticket.message)
        self._send_handles.pop(identity)
        self._completed_send_ids.add(identity)
        return True

    def is_send_ready(self, ticket: TransferTicket) -> bool:
        """Return a nonblocking readiness hint for a known active send."""
        self._ensure_open()
        identity = self._message_identity(ticket.message)
        handles = self._send_handles.get(identity)
        if handles is None:
            if identity in self._send_handles:
                return False
            raise ValueError("unknown distributed P2P send ticket")
        tensor_handles = handles[1:] if handles and getattr(handles[0], "_is_metadata_handle", False) else handles
        return all(handle.is_completed() for handle in tensor_handles)

    def abort(self, ticket: TransferTicket) -> bool:
        # NCCL P2P has no safe per-operation cancellation. Discard therefore
        # drains the send before allowing its source tensor to be released.
        return self.wait(ticket)

    def close(self) -> None:
        if self._closed:
            return
        if self.has_outstanding_operations:
            raise RuntimeError("cannot close distributed P2P transport with outstanding operations")
        self._metadata_receives.put(None)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("distributed P2P transport is closed")

    @staticmethod
    def _message_identity(message: PipelineMessage) -> tuple[str, int, int]:
        return (message.batch_id, message.step_index, message.epoch)

    @staticmethod
    def _validate_message_matches_offer(message: PipelineMessage, offer: PipelineTransferOffer) -> None:
        if DistributedP2PTransport._message_identity(message) != (
            offer.batch_id,
            offer.step_index,
            offer.epoch,
        ):
            raise ValueError("pipeline message identity does not match its transfer grant")
        if offer.payload_metadata and (
            not isinstance(message.payload, dict)
            or pipeline_payload_metadata(message.payload) != offer.payload_metadata
        ):
            raise ValueError("pipeline message metadata does not match its transfer grant")


class PipelineStageConnector:
    """Own one directed edge's bounded send/receive leases.

    The connector does not choose requests, call model code, or create a
    background thread.  A Worker progress loop calls ``poll_received`` and
    explicitly retires tickets after consumer completion.
    """

    def __init__(self, *, edge: str, max_slots: int = 1, transport: DistributedP2PTransport) -> None:
        if not edge:
            raise ValueError("edge must be non-empty")
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError("max_slots must be a positive integer")
        self.edge = edge
        self.max_slots = max_slots
        self.transport = transport
        self._send_tickets: deque[TransferTicket] = deque()
        self._received: deque[PipelineMessage] = deque()
        self._received_leases: dict[int, PipelineMessage] = {}
        self._closed = False

    @property
    def send_in_use(self) -> int:
        return len(self._send_tickets)

    @property
    def receive_depth(self) -> int:
        return len(self._received) + len(self._received_leases)

    @property
    def closed(self) -> bool:
        return self._closed

    def enqueue_send(self, message: PipelineMessage) -> TransferTicket:
        self._ensure_open()
        self._validate_message(message)
        if self.send_in_use >= self.max_slots:
            raise RuntimeError(f"pipeline edge {self.edge!r} has no send credit")
        ticket = TransferTicket(message=message)
        self._send_tickets.append(ticket)
        return ticket

    def start_send(self, ticket: TransferTicket, offer: PipelineTransferOffer) -> None:
        """Launch one reserved send from the Worker-local FIFO."""
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if ticket.started:
            raise ValueError("transfer ticket has already started")
        if self._message_identity(ticket.message) != (
            offer.batch_id,
            offer.step_index,
            offer.epoch,
        ):
            raise ValueError("pipeline transfer offer does not match the reserved send")
        # Once control enters the backend, failure is ambiguous: metadata or
        # device work may already have started. Keep transport ownership until
        # the execution group is drained or torn down.
        ticket.started = True
        self.transport.start_transfer(offer, ticket.message)

    def start_receive(self, offer: PipelineTransferOffer) -> None:
        """Post a receive directly through the PP group coordinator."""
        self._ensure_open()
        if (
            offer.edge_kind is not self.transport.edge_kind
            or offer.src_rank != self.transport.src_rank
            or offer.dst_rank != self.transport.dst_rank
        ):
            raise ValueError("pipeline receive offer does not match this connector")
        self.transport.start_transfer(offer)

    def wait_send_completion(self, ticket: TransferTicket) -> None:
        """Verify backend completion while retaining connector ownership."""
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.started:
            raise RuntimeError("cannot wait for a transfer before its grant starts")
        if ticket.completed:
            return
        if not self.transport.wait(ticket):
            raise RuntimeError("transport wait did not complete transfer")
        ticket.completed = True

    def poll_send_completion(self, ticket: TransferTicket) -> bool:
        """Verify a ready backend send without blocking on unfinished Work."""
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.started:
            return False
        if ticket.completed:
            return True
        if not self.transport.is_send_ready(ticket):
            return False
        self.wait_send_completion(ticket)
        return True

    def poll_received(self, limit: int = 1) -> list[PipelineMessage]:
        self._ensure_open()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        available = self.max_slots - self.receive_depth
        if available > 0:
            poll_limit = min(limit, available)
            incoming = self.transport.poll(limit=poll_limit)
            if len(incoming) > poll_limit:
                raise RuntimeError("pipeline transport exceeded its bounded poll limit")
            incoming_keys = [self._message_identity(message) for message in incoming]
            held_keys = {
                self._message_identity(message)
                for message in (*self._received, *self._received_leases.values())
            }
            if len(set(incoming_keys)) != len(incoming_keys) or held_keys.intersection(incoming_keys):
                raise RuntimeError("pipeline transport returned a duplicate leased message")
            for message in incoming:
                self._validate_message(message)
            self._received.extend(incoming)
        messages = [self._received.popleft() for _ in range(min(limit, len(self._received)))]
        for message in messages:
            self._received_leases[id(message)] = message
        return messages

    def release_received(self, message: PipelineMessage) -> None:
        """Return receive credit after the consumer is finished reading."""
        self._ensure_open()
        if self._received_leases.pop(id(message), None) is None:
            raise ValueError("unknown or already released receive lease")

    def release_send(self, ticket: TransferTicket) -> None:
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if ticket.started and not ticket.completed:
            raise RuntimeError("cannot release a transfer before transport completion")
        self._send_tickets.remove(ticket)

    def retire_batch(self, batch_id: str, *, discard_results: bool = False) -> None:
        """Retire all local transport state for one batch after dependencies settle."""
        self._ensure_open()
        if not batch_id:
            raise ValueError("batch_id must be non-empty")
        matching = [ticket for ticket in self._send_tickets if ticket.message.batch_id == batch_id]
        leased = [message for message in self._received_leases.values() if message.batch_id == batch_id]
        if leased:
            raise RuntimeError("cannot retire a batch before receive consumers complete")
        if matching and not discard_results:
            unreleased = [ticket for ticket in matching if ticket.started and not ticket.completed]
            if unreleased:
                raise RuntimeError("cannot retire a batch before transport completion")
        for ticket in matching:
            if ticket.started and not ticket.completed:
                self._wait_or_abort(ticket, discard=True)
        for ticket in matching:
            self._send_tickets.remove(ticket)
        self._received = deque(message for message in self._received if message.batch_id != batch_id)

    def close(self, *, drain: bool = False) -> None:
        if self._closed:
            return
        if self._received_leases:
            raise RuntimeError("cannot close a connector with active receive consumers")
        if self._send_tickets and not drain:
            raise RuntimeError("cannot close a connector with outstanding transfers")
        if drain:
            for ticket in self._send_tickets:
                if ticket.started and not ticket.completed:
                    self._wait_or_abort(ticket, discard=False)
        self.transport.close()
        self._send_tickets.clear()
        self._received.clear()
        self._received_leases.clear()
        self._closed = True

    def _wait_or_abort(self, ticket: TransferTicket, *, discard: bool) -> None:
        completed = self.transport.abort(ticket) if discard else self.transport.wait(ticket)
        if not completed:
            operation = "abort" if discard else "wait"
            raise RuntimeError(f"transport {operation} did not complete transfer")
        ticket.completed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"pipeline edge {self.edge!r} is closed")

    def _validate_message(self, message: PipelineMessage) -> None:
        if not message.batch_id or message.step_index < 0 or message.epoch < 0:
            raise ValueError("invalid pipeline message identity")

    @staticmethod
    def _message_identity(message: PipelineMessage) -> tuple[str, int, int]:
        return (message.batch_id, message.step_index, message.epoch)
