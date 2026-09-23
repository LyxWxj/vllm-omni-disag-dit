# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, identity-checked transport contract for queued PP edges."""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


@dataclass(frozen=True)
class PipelineMessage:
    batch_id: str
    step_index: int
    epoch: int
    branch: str
    payload: Any


class PipelineEdgeKind(StrEnum):
    ACTIVATION = "activation"
    FEEDBACK = "feedback"


@dataclass(frozen=True)
class PipelineTransferOffer:
    """Metadata-only readiness offer for one directed PP transfer."""

    batch_id: str
    step_index: int
    epoch: int
    branch: str
    edge_kind: PipelineEdgeKind
    src_rank: int
    dst_rank: int

    def __post_init__(self) -> None:
        if not self.batch_id or self.step_index < 0 or self.epoch < 0:
            raise ValueError("invalid pipeline transfer identity")
        if self.branch != "conditional":
            raise ValueError("M2 supports only the conditional pipeline branch")
        if not isinstance(self.edge_kind, PipelineEdgeKind):
            raise ValueError("pipeline transfer requires a valid edge kind")
        if self.src_rank < 0 or self.dst_rank < 0 or self.src_rank == self.dst_rank:
            raise ValueError("pipeline transfer endpoints must be distinct non-negative ranks")

    @property
    def identity(self) -> tuple[str, int, int, str, PipelineEdgeKind, int, int]:
        return (
            self.batch_id,
            self.step_index,
            self.epoch,
            self.branch,
            self.edge_kind,
            self.src_rank,
            self.dst_rank,
        )


@dataclass
class PipelineTransferGrant:
    offer: PipelineTransferOffer
    completed_ranks: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class PipelineEndpointCompletion:
    identity: tuple[Any, ...]
    rank: int


@dataclass
class PipelineTransportProgress:
    rank: int
    offers: list[PipelineTransferOffer] = field(default_factory=list)
    completions: list[PipelineEndpointCompletion] = field(default_factory=list)


@dataclass
class PipelineCoordinatorProgress:
    grants: list[PipelineTransferGrant] = field(default_factory=list)
    completed: list[tuple[Any, ...]] = field(default_factory=list)


class PipelineTransferCoordinator:
    """FIFO control-plane grants for matched P2P endpoint readiness."""

    def __init__(
        self,
        *,
        activation_edges: set[tuple[int, int]],
        feedback_edges: set[tuple[int, int]],
    ) -> None:
        self._validate_topology(activation_edges, feedback_edges)
        self._valid_edges = {
            PipelineEdgeKind.ACTIVATION: frozenset(activation_edges),
            PipelineEdgeKind.FEEDBACK: frozenset(feedback_edges),
        }
        self._offers: dict[tuple[PipelineEdgeKind, int, int], deque[PipelineTransferOffer]] = {
            (edge_kind, src_rank, dst_rank): deque()
            for edge_kind, edges in self._valid_edges.items()
            for src_rank, dst_rank in edges
        }
        self._offer_ids: set[tuple[Any, ...]] = set()
        self._ready_ids: set[tuple[Any, ...]] = set()
        self._grants: dict[tuple[Any, ...], PipelineTransferGrant] = {}
        self._completed_ids: set[tuple[Any, ...]] = set()
        self._busy_ranks: set[int] = set()
        self._next_edge = PipelineEdgeKind.FEEDBACK
        self._edge_cursor = {
            PipelineEdgeKind.ACTIVATION: 0,
            PipelineEdgeKind.FEEDBACK: 0,
        }

    @property
    def endpoint_ranks(self) -> frozenset[int]:
        return frozenset(rank for edge in self._valid_edges[PipelineEdgeKind.ACTIVATION] for rank in edge)

    @property
    def stage_physical_ranks(self) -> dict[int, int]:
        """Return the physical rank for logical stages in the single M2 replica."""
        edges = self._valid_edges[PipelineEdgeKind.ACTIVATION]
        if len(edges) != 1:
            raise RuntimeError("M2 Engine submission requires exactly one configured PP replica")
        src_rank, dst_rank = next(iter(edges))
        return {0: src_rank, 1: dst_rank}

    def offer(self, offer: PipelineTransferOffer) -> None:
        identity = offer.identity
        if (offer.src_rank, offer.dst_rank) not in self._valid_edges[offer.edge_kind]:
            raise ValueError("pipeline transfer offer does not match the configured edge topology")
        if identity in self._offer_ids or identity in self._grants or identity in self._completed_ids:
            raise ValueError("duplicate pipeline transfer offer")
        self._offers[(offer.edge_kind, offer.src_rank, offer.dst_rank)].append(offer)
        self._offer_ids.add(identity)

    def mark_receive_ready(self, identity: tuple[Any, ...]) -> None:
        if identity not in self._offer_ids:
            raise KeyError("unknown pipeline transfer offer")
        self._ready_ids.add(identity)

    def pending_readiness_offers(self) -> list[PipelineTransferOffer]:
        """Return FIFO heads whose endpoint readiness has not been confirmed."""
        return [queue[0] for queue in self._offers.values() if queue and queue[0].identity not in self._ready_ids]

    def grant_ready(self, limit: int = 1) -> list[PipelineTransferGrant]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        grants: list[PipelineTransferGrant] = []
        while len(grants) < limit:
            edge_order = (
                self._next_edge,
                PipelineEdgeKind.ACTIVATION
                if self._next_edge is PipelineEdgeKind.FEEDBACK
                else PipelineEdgeKind.FEEDBACK,
            )
            selected: PipelineTransferOffer | None = None
            for edge_kind in edge_order:
                edge_keys = sorted(key for key in self._offers if key[0] is edge_kind)
                if not edge_keys:
                    continue
                cursor = self._edge_cursor[edge_kind] % len(edge_keys)
                for offset in range(len(edge_keys)):
                    edge_index = (cursor + offset) % len(edge_keys)
                    edge_key = edge_keys[edge_index]
                    queue = self._offers[edge_key]
                    if not queue:
                        continue
                    candidate = queue[0]
                    if candidate.identity not in self._ready_ids:
                        continue
                    if candidate.src_rank in self._busy_ranks or candidate.dst_rank in self._busy_ranks:
                        continue
                    selected = candidate
                    self._edge_cursor[edge_kind] = (edge_index + 1) % len(edge_keys)
                    break
                if selected is not None:
                    break
            if selected is None:
                break
            offer = selected
            identity = offer.identity
            self._offers[(offer.edge_kind, offer.src_rank, offer.dst_rank)].popleft()
            self._offer_ids.remove(identity)
            self._ready_ids.remove(identity)
            grant = PipelineTransferGrant(offer=offer)
            self._grants[identity] = grant
            self._busy_ranks.update((offer.src_rank, offer.dst_rank))
            grants.append(grant)
            self._next_edge = (
                PipelineEdgeKind.ACTIVATION
                if offer.edge_kind is PipelineEdgeKind.FEEDBACK
                else PipelineEdgeKind.FEEDBACK
            )
        return grants

    def complete(self, identity: tuple[Any, ...], rank: int) -> bool:
        grant = self._grants.get(identity)
        if grant is None:
            raise KeyError("unknown pipeline transfer grant")
        if rank not in {grant.offer.src_rank, grant.offer.dst_rank}:
            raise ValueError("completion rank is not a transfer endpoint")
        if rank in grant.completed_ranks:
            raise ValueError("duplicate pipeline transfer completion")
        grant.completed_ranks.add(rank)
        if grant.completed_ranks != {grant.offer.src_rank, grant.offer.dst_rank}:
            return False
        self._grants.pop(identity)
        self._completed_ids.add(identity)
        self._busy_ranks.difference_update((grant.offer.src_rank, grant.offer.dst_rank))
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "offers": sum(len(queue) for queue in self._offers.values()),
            "ready": len(self._ready_ids),
            "grants": len(self._grants),
            "completed": len(self._completed_ids),
            "busy_ranks": tuple(sorted(self._busy_ranks)),
        }

    @staticmethod
    def _validate_topology(
        activation_edges: set[tuple[int, int]],
        feedback_edges: set[tuple[int, int]],
    ) -> None:
        if not activation_edges or not feedback_edges:
            raise ValueError("pipeline transfer topology requires activation and feedback edges")
        expected_feedback = {(dst, src) for src, dst in activation_edges}
        if feedback_edges != expected_feedback:
            raise ValueError("feedback edges must exactly reverse the activation edges")
        endpoints: set[int] = set()
        for src_rank, dst_rank in activation_edges:
            if src_rank < 0 or dst_rank < 0 or src_rank == dst_rank:
                raise ValueError("pipeline topology endpoints must be distinct non-negative ranks")
            if src_rank in endpoints or dst_rank in endpoints:
                raise ValueError("queued PP=2 topology requires disjoint two-rank replicas")
            endpoints.update((src_rank, dst_rank))


@dataclass
class TransferTicket:
    message: PipelineMessage
    started: bool = False
    completed: bool = False
    released: bool = False


class PipelineTransport(Protocol):
    def start_granted_transfer(
        self,
        grant: PipelineTransferGrant,
        message: PipelineMessage | None = None,
    ) -> None: ...

    def poll(self, limit: int | None = None) -> list[PipelineMessage]: ...

    def wait(self, ticket: TransferTicket) -> bool: ...

    def abort(self, ticket: TransferTicket) -> bool: ...

    def close(self) -> None: ...


@dataclass
class _PendingReceive:
    message: PipelineMessage
    identity: tuple[str, int, int, str]
    handles: list[Any]
    postprocess: list[Any]
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
        self._send_handles: dict[tuple[str, int, int, str], list[Any] | None] = {}
        self._completed_send_ids: set[tuple[str, int, int, str]] = set()
        self._pending_receives: deque[_PendingReceive] = deque()
        self._ready_receives: deque[PipelineMessage] = deque()
        self._active_receive_ids: set[tuple[str, int, int, str]] = set()
        self._completed_receive_ids: set[tuple[str, int, int, str]] = set()
        self._closed = False

    @property
    def has_outstanding_operations(self) -> bool:
        return bool(self._send_handles or self._pending_receives or self._ready_receives or self._active_receive_ids)

    def start_granted_transfer(
        self,
        grant: PipelineTransferGrant,
        message: PipelineMessage | None = None,
    ) -> None:
        self._ensure_open()
        offer = grant.offer
        if offer.edge_kind is not self.edge_kind or offer.src_rank != self.src_rank or offer.dst_rank != self.dst_rank:
            raise ValueError("transfer grant does not match this P2P transport")
        if self.local_rank == self.src_rank:
            if message is None:
                raise ValueError("sender requires a pipeline message payload")
            self._validate_message_matches_offer(message, offer)
            self._start_send(message)
            return
        if message is not None:
            raise ValueError("receiver must not provide a sender payload")
        identity = (offer.batch_id, offer.step_index, offer.epoch, offer.branch)
        if identity in self._active_receive_ids or identity in self._completed_receive_ids:
            raise ValueError("duplicate distributed P2P receive grant")
        # Register before entering the blocking metadata receive. If the
        # backend raises after partially posting work, keep the identity active
        # so a replay cannot post an unmatched second receive.
        self._active_receive_ids.add(identity)
        tensor_dict, handles, postprocess = self.group.irecv_tensor_dict(src=self._src_group_rank)
        self._pending_receives.append(
            _PendingReceive(
                message=PipelineMessage(
                    batch_id=offer.batch_id,
                    step_index=offer.step_index,
                    epoch=offer.epoch,
                    branch=offer.branch,
                    payload=tensor_dict,
                ),
                identity=identity,
                handles=list(handles),
                postprocess=list(postprocess),
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
        self._send_handles[identity] = list(self.group.isend_tensor_dict(message.payload, dst=self._dst_group_rank))

    def poll(self, limit: int | None = None) -> list[PipelineMessage]:
        self._ensure_open()
        if limit is None or type(limit) is not int or limit <= 0:
            raise ValueError("distributed P2P polling requires a positive limit")
        if self._ready_receives:
            return [self._ready_receives.popleft() for _ in range(min(limit, len(self._ready_receives)))]
        while self._pending_receives and len(self._ready_receives) < limit:
            pending = self._pending_receives[0]
            if pending.failure is not None:
                raise pending.failure
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
                raise
            self._pending_receives.popleft()
            self._active_receive_ids.remove(pending.identity)
            self._completed_receive_ids.add(pending.identity)
            self._ready_receives.append(pending.message)
        return [self._ready_receives.popleft() for _ in range(min(limit, len(self._ready_receives)))]

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
        return all(handle.is_completed() for handle in handles)

    def abort(self, ticket: TransferTicket) -> bool:
        # NCCL P2P has no safe per-operation cancellation. Discard therefore
        # drains the send before allowing its source tensor to be released.
        return self.wait(ticket)

    def close(self) -> None:
        if self._closed:
            return
        if self.has_outstanding_operations:
            raise RuntimeError("cannot close distributed P2P transport with outstanding operations")
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("distributed P2P transport is closed")

    @staticmethod
    def _message_identity(message: PipelineMessage) -> tuple[str, int, int, str]:
        return (message.batch_id, message.step_index, message.epoch, message.branch)

    @staticmethod
    def _validate_message_matches_offer(message: PipelineMessage, offer: PipelineTransferOffer) -> None:
        if DistributedP2PTransport._message_identity(message) != (
            offer.batch_id,
            offer.step_index,
            offer.epoch,
            offer.branch,
        ):
            raise ValueError("pipeline message identity does not match its transfer grant")


class PipelineStageConnector:
    """Own one directed edge's bounded send/receive leases.

    The connector does not choose requests, call model code, or create a
    background thread.  A Worker progress loop calls ``poll_received`` and
    explicitly retires tickets after consumer completion.
    """

    def __init__(self, *, edge: str, max_slots: int = 1, transport: PipelineTransport | None = None) -> None:
        if not edge:
            raise ValueError("edge must be non-empty")
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError("max_slots must be a positive integer")
        self.edge = edge
        self.max_slots = max_slots
        self.transport = transport
        self._send_tickets: deque[TransferTicket] = deque()
        self._received: deque[PipelineMessage] = deque()
        self._transport_pending: deque[PipelineMessage] = deque()
        self._received_leases: dict[int, PipelineMessage] = {}
        self._closed = False

    @property
    def send_in_use(self) -> int:
        return len(self._send_tickets)

    @property
    def receive_depth(self) -> int:
        return len(self._received) + len(self._received_leases)

    @property
    def transport_pending(self) -> int:
        return len(self._transport_pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def enqueue_send(self, message: PipelineMessage) -> TransferTicket:
        self._ensure_open()
        self._validate_message(message)
        if self.send_in_use >= self.max_slots:
            raise RuntimeError(f"pipeline edge {self.edge!r} has no send credit")
        ticket = TransferTicket(message=message, started=self.transport is None)
        self._send_tickets.append(ticket)
        return ticket

    def start_granted_send(self, ticket: TransferTicket, grant: PipelineTransferGrant) -> None:
        """Launch one reserved send only after its coordinator grant arrives."""
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if ticket.started:
            raise ValueError("transfer ticket has already started")
        if self.transport is None:
            raise RuntimeError("connector has no transport for granted send")
        offer = grant.offer
        if self._message_identity(ticket.message) != (
            offer.batch_id,
            offer.step_index,
            offer.epoch,
            offer.branch,
        ):
            raise ValueError("transfer grant does not match the reserved send")
        # Once control enters the backend, failure is ambiguous: metadata or
        # device work may already have started. Keep transport ownership until
        # the execution group is drained or torn down.
        ticket.started = True
        self.transport.start_granted_transfer(grant, ticket.message)

    def mark_send_complete(self, ticket: TransferTicket) -> None:
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.started:
            raise RuntimeError("cannot complete a transfer before its grant starts")
        ticket.completed = True

    def wait_send_completion(self, ticket: TransferTicket) -> None:
        """Verify backend completion while retaining connector ownership."""
        self._ensure_open()
        if ticket not in self._send_tickets:
            raise ValueError("unknown transfer ticket")
        if not ticket.started:
            raise RuntimeError("cannot wait for a transfer before its grant starts")
        if ticket.completed:
            return
        if self.transport is None or not self.transport.wait(ticket):
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
        is_ready = getattr(self.transport, "is_send_ready", None)
        if not callable(is_ready) or not is_ready(ticket):
            return False
        self.wait_send_completion(ticket)
        return True

    def poll_received(self, limit: int = 1) -> list[PipelineMessage]:
        self._ensure_open()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        available = self.max_slots - self.receive_depth - self.transport_pending
        if self.transport is not None and available > 0 and not self._transport_pending:
            poll_limit = min(limit, available)
            parameters = inspect.signature(self.transport.poll).parameters.values()
            supports_limit = any(
                parameter.name == "limit" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
            )
            if not supports_limit:
                raise RuntimeError("pipeline transport must support bounded polling") from None
            incoming = self.transport.poll(limit=poll_limit)
            if len(incoming) > poll_limit:
                raise RuntimeError("pipeline transport exceeded its bounded poll limit")
            incoming_keys = [self._message_identity(message) for message in incoming]
            held_keys = {
                self._message_identity(message)
                for message in (*self._received, *self._transport_pending, *self._received_leases.values())
            }
            if len(set(incoming_keys)) != len(incoming_keys) or held_keys.intersection(incoming_keys):
                raise RuntimeError("pipeline transport returned a duplicate leased message")
            for message in incoming:
                self._validate_message(message)
            self._transport_pending.extend(incoming)
        while self._transport_pending and len(self._received) < self.max_slots:
            self._received.append(self._transport_pending.popleft())
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
        ticket.released = True

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
                if not discard_results:
                    raise RuntimeError("cannot retire a batch before transport completion")
                self._wait_or_abort(ticket, discard=True)
        for ticket in matching:
            self._send_tickets.remove(ticket)
            ticket.released = True
        self._received = deque(message for message in self._received if message.batch_id != batch_id)
        self._transport_pending = deque(message for message in self._transport_pending if message.batch_id != batch_id)

    def health(self) -> dict[str, Any]:
        health = {
            "edge": self.edge,
            "closed": self._closed,
            "send_in_use": self.send_in_use,
            "receive_depth": self.receive_depth,
            "receive_leases": len(self._received_leases),
            "transport_pending": self.transport_pending,
            "max_slots": self.max_slots,
        }
        transport_outstanding = getattr(self.transport, "has_outstanding_operations", False)
        health["transport_outstanding"] = bool(transport_outstanding)
        return health

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
        transport_close = getattr(self.transport, "close", None)
        if self.transport is not None and not callable(transport_close):
            raise RuntimeError("pipeline transport does not support close")
        if transport_close is not None:
            transport_close()
        for ticket in self._send_tickets:
            ticket.released = True
        self._send_tickets.clear()
        self._received.clear()
        self._received_leases.clear()
        self._transport_pending.clear()
        self._closed = True

    def _wait_or_abort(self, ticket: TransferTicket, *, discard: bool) -> None:
        operation = "abort" if discard else "wait"
        handler = getattr(self.transport, operation, None)
        if not callable(handler):
            raise RuntimeError(f"transport does not support {operation} for incomplete transfers")
        if not handler(ticket):
            raise RuntimeError(f"transport {operation} did not complete transfer")
        ticket.completed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"pipeline edge {self.edge!r} is closed")

    def _validate_message(self, message: PipelineMessage) -> None:
        if message.branch != "conditional":
            raise ValueError("M2 supports only the conditional pipeline branch")
        if not message.batch_id or message.step_index < 0 or message.epoch < 0:
            raise ValueError("invalid pipeline message identity")

    @staticmethod
    def _message_identity(message: PipelineMessage) -> tuple[str, int, int, str]:
        return (message.batch_id, message.step_index, message.epoch, message.branch)
