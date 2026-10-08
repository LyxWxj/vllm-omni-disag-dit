# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-owned lifecycle state for queued pipeline finalization."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    DistributedP2PTransport,
    PipelineEdgeKind,
    PipelineEndpointCompletion,
    PipelineMessage,
    PipelineStageConnector,
    PipelineTransferGrant,
    PipelineTransferOffer,
    PipelineTransportProgress,
)
from vllm_omni.diffusion.worker.pipeline_state import PipelineTask

PIPELINE_CONSUMER_EVENT_FAILED = object()


@dataclass
class PipelineTransportState:
    """Worker-owned transport reservations and consumer leases."""

    send_tickets: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    receive_reservations: dict[tuple[Any, ...], PipelineEdgeKind] = field(default_factory=dict)
    started_receive_ids: set[tuple[Any, ...]] = field(default_factory=set)
    receive_consumers: dict[tuple[Any, ...], tuple[PipelineEdgeKind, PipelineMessage, Any | None]] = field(
        default_factory=dict
    )
    pending_received: dict[PipelineEdgeKind, deque[PipelineMessage]] = field(
        default_factory=lambda: {
            PipelineEdgeKind.ACTIVATION: deque(),
            PipelineEdgeKind.FEEDBACK: deque(),
        }
    )


class PipelineTransportRuntime:
    """Adapt one Worker to the bounded queued-PP transport contract.

    The Worker still owns stage scheduling and model execution.  This object
    only owns the connector calls and the transport-side identity/lease
    bookkeeping that surrounds them.
    """

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    @property
    def state(self) -> PipelineTransportState:
        return self.worker._get_pipeline_transport_state()

    def require_connector(self, edge_kind: PipelineEdgeKind) -> PipelineStageConnector:
        connector = self.worker.pipeline_connectors.get(edge_kind)
        if connector is None:
            raise RuntimeError(f"pipeline connector {edge_kind.value!r} is not initialized")
        return connector

    def reserve_send(self, offer: PipelineTransferOffer, payload: dict[str, Any]) -> PipelineTransferOffer:
        """Reserve sender credit and payload without starting communication."""
        worker = self.worker
        if worker.rank != offer.src_rank:
            raise ValueError("only the transfer source can reserve a pipeline send")
        if not isinstance(payload, dict):
            raise TypeError("pipeline send payload must be a tensor dictionary")
        connector = self.require_connector(offer.edge_kind)
        if offer.identity in self.state.send_tickets:
            raise ValueError("pipeline transfer send is already reserved")
        message = PipelineMessage(
            batch_id=offer.batch_id,
            step_index=offer.step_index,
            epoch=offer.epoch,
            branch=offer.branch,
            payload=payload,
        )
        self.state.send_tickets[offer.identity] = connector.enqueue_send(message)
        return offer

    def accept_offer(self, offer: PipelineTransferOffer) -> bool:
        """Validate an endpoint and reserve receive credit when available."""
        worker = self.worker
        if worker.rank not in {offer.src_rank, offer.dst_rank}:
            return True
        connector = self.require_connector(offer.edge_kind)
        if worker.rank == offer.src_rank:
            ticket = self.state.send_tickets.get(offer.identity)
            if ticket is None:
                raise KeyError("pipeline transfer offer has no reserved sender ticket")
            if ticket.started or ticket.released:
                raise RuntimeError("pipeline sender ticket is not available for a new grant")
            message = ticket.message
            if not isinstance(message.payload, dict):
                raise TypeError("pipeline sender payload must be a tensor dictionary")
            if (message.batch_id, message.step_index, message.epoch, message.branch) != (
                offer.batch_id,
                offer.step_index,
                offer.epoch,
                offer.branch,
            ):
                raise ValueError("pipeline sender ticket identity does not match the transfer offer")
        else:
            if offer.identity in self.state.receive_reservations:
                if self.state.receive_reservations[offer.identity] is not offer.edge_kind:
                    raise ValueError("pipeline receive reservation has a different edge kind")
                return True
            if offer.identity in self.state.receive_consumers:
                raise ValueError("pipeline receive transfer is still owned by its stage consumer")
            reserved = sum(edge_kind is offer.edge_kind for edge_kind in self.state.receive_reservations.values())
            # A reservation covers only the transport receive slot. Once the
            # message is handed to the stage, its tensor is tracked separately
            # by the compute lease and no longer blocks another receive.
            if reserved >= connector.max_slots:
                return False
            self.state.receive_reservations[offer.identity] = offer.edge_kind
        return True

    def start_transfer(self, grant: PipelineTransferGrant) -> bool:
        """Start this Worker's endpoint after the coordinator grants it."""
        worker = self.worker
        offer = grant.offer
        if worker.rank not in {offer.src_rank, offer.dst_rank}:
            return True
        connector = self.require_connector(offer.edge_kind)
        if worker.rank == offer.src_rank:
            ticket = self.state.send_tickets.get(offer.identity)
            if ticket is None:
                raise KeyError("pipeline transfer grant has no reserved sender ticket")
            connector.start_granted_send(ticket, grant)
        else:
            if offer.identity not in self.state.receive_reservations:
                raise KeyError("pipeline transfer grant has no reserved receive credit")
            if offer.identity in self.state.started_receive_ids:
                raise ValueError("pipeline transfer receive has already started")
            transport = connector.transport
            if not isinstance(transport, DistributedP2PTransport):
                raise RuntimeError("pipeline destination does not use distributed P2P transport")
            self.state.started_receive_ids.add(offer.identity)
            transport.start_granted_transfer(grant)
        return True

    def retire_send(self, identity: tuple[Any, ...]) -> bool:
        """Wait for and release one completed send reservation."""
        ticket = self.state.send_tickets.get(identity)
        if ticket is None:
            raise KeyError("unknown pipeline send reservation")
        if len(identity) < 5 or not isinstance(identity[4], PipelineEdgeKind):
            raise ValueError("invalid pipeline transfer identity")
        connector = self.require_connector(identity[4])
        connector.wait_send_completion(ticket)
        connector.release_send(ticket)
        self.state.send_tickets.pop(identity)
        return True

    def poll_completed_sends(self, progress: PipelineTransportProgress) -> None:
        for identity, ticket in list(self.state.send_tickets.items()):
            connector = self.require_connector(identity[4])
            if connector.poll_send_completion(ticket):
                connector.release_send(ticket)
                self.state.send_tickets.pop(identity)
                progress.completions.append(PipelineEndpointCompletion(identity=identity, rank=self.worker.rank))

    def poll_received_into_pending(self, edge_kind: PipelineEdgeKind) -> None:
        connector = self.require_connector(edge_kind)
        for message in connector.poll_received(limit=1):
            self.state.pending_received[edge_kind].append(message)

    def release_completed_consumers(self) -> None:
        for identity, (_edge_kind, _message, event) in list(self.state.receive_consumers.items()):
            if event is PIPELINE_CONSUMER_EVENT_FAILED:
                continue
            if event is not None and not event.query():
                continue
            self.state.receive_consumers.pop(identity)

    def find_receive_reservation(
        self,
        edge_kind: PipelineEdgeKind,
        message: PipelineMessage,
    ) -> tuple[Any, ...]:
        identity = (message.batch_id, message.step_index, message.epoch, message.branch)
        matching = [
            key
            for key, reserved_edge in self.state.receive_reservations.items()
            if key[:4] == identity and reserved_edge is edge_kind
        ]
        if len(matching) != 1:
            raise RuntimeError("pipeline receive message has no unique reservation")
        return matching[0]

    def poll_received(self, edge_kind: PipelineEdgeKind, limit: int = 1) -> list[PipelineMessage]:
        return self.require_connector(edge_kind).poll_received(limit=limit)

    def release_received(self, edge_kind: PipelineEdgeKind, message: PipelineMessage) -> None:
        connector = self.require_connector(edge_kind)
        identity = (message.batch_id, message.step_index, message.epoch, message.branch)
        matching = [key for key in self.state.receive_reservations if key[:4] == identity]
        if len(matching) != 1 or self.state.receive_reservations[matching[0]] is not edge_kind:
            raise RuntimeError("pipeline receive reservation does not match released message")
        connector.release_received(message)
        self.state.receive_reservations.pop(matching[0])
        self.state.started_receive_ids.discard(matching[0])

    def release_unstarted_batch_transfers(self, task: PipelineTask) -> None:
        for identity, ticket in list(self.state.send_tickets.items()):
            if identity[0] != task.batch_id or identity[2] != task.epoch or ticket.started:
                continue
            connector = self.require_connector(identity[4])
            connector.release_send(ticket)
            self.state.send_tickets.pop(identity)
        for identity in list(self.state.receive_reservations):
            if (
                identity[0] == task.batch_id
                and identity[2] == task.epoch
                and identity not in self.state.started_receive_ids
            ):
                self.state.receive_reservations.pop(identity)
@dataclass
class PipelineFinalizationState:
    """Own finalization futures and device-completion metadata for one Worker."""

    futures: dict[str, Future[Any]] = field(default_factory=dict)
    executor: ThreadPoolExecutor | None = None
    published: set[str] = field(default_factory=set)
    device_events: dict[str, Any] = field(default_factory=dict)
    stream: Any | None = None

    def ensure_executor(self, rank: int) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"WanFinalDecode-rank{rank}",
            )
        return self.executor

    def clear_batch(self, batch_id: str) -> None:
        self.futures.pop(batch_id, None)
        self.published.discard(batch_id)
        self.device_events.pop(batch_id, None)

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None
