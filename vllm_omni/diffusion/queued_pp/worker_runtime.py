# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-owned lifecycle state for queued pipeline finalization."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pp_group, get_world_group
from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    DistributedP2PTransport,
    PipelineEdgeKind,
    PipelineEndpointCompletion,
    PipelineMessage,
    PipelineStageConnector,
    PipelineTransferGrant,
    PipelineTransferOffer,
    PipelineTransportProgress,
    TransferTicket,
    pipeline_payload_metadata,
)
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEvent,
    PipelineEventType,
    PipelineFinalizationUpdate,
    PipelineProgress,
    PipelineStageSpec,
    PipelineStageState,
    PipelineTask,
    PipelineTaskStatus,
    PipelineWorkerUpdate,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput
from vllm_omni.platforms import current_omni_platform

PIPELINE_CONSUMER_EVENT_FAILED = object()
_PIPELINE_CONSUMER_EVENT_FAILED = PIPELINE_CONSUMER_EVENT_FAILED
logger = init_logger(__name__)


def _all_gather_rank_values(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    control_group = get_world_group().cpu_group
    values: list[Any] = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(values, value, group=control_group)
    return values


def _run_and_gather_rank_values(operation: str, func: Any) -> list[Any]:
    try:
        local_result = (True, func())
    except Exception as exc:
        logger.exception("%s failed on this Worker rank", operation)
        local_result = (False, f"{type(exc).__name__}: {exc}")
    rank_results = _all_gather_rank_values(local_result)
    failures = [f"rank {rank}: {result}" for rank, (ok, result) in enumerate(rank_results) if not ok]
    if failures:
        raise RuntimeError(f"{operation} failed on " + "; ".join(failures))
    return [result for _, result in rank_results]


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


class QueuedWorkerRuntime:
    """Own queued stage lifecycle while delegating generic Worker operations."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
        else:
            setattr(self.owner, name, value)

    def _run_and_gather_rank_values(self, operation: str, func: Any) -> list[Any]:
        helper = getattr(self.owner, "_run_and_gather_rank_values", None)
        if callable(helper):
            return helper(operation, func)
        return _run_and_gather_rank_values(operation, func)

    def _all_gather_rank_values(self, value: Any) -> list[Any]:
        helper = getattr(self.owner, "_all_gather_rank_values", None)
        if callable(helper):
            return helper(value)
        return _all_gather_rank_values(value)

    def _get_pp_group(self) -> Any:
        helper = getattr(self.owner, "_get_pp_group", None)
        if callable(helper):
            return helper()
        return get_pp_group()

    def _require_pipeline_stage(self, pp_stage_id: int) -> PipelineStageState:
        stage = self.pipeline_stages.get(pp_stage_id)
        if stage is None:
            raise KeyError(f"Unknown pipeline stage {pp_stage_id}.")
        return stage

    def _owner_method(self, name: str, fallback: Any) -> Any:
        method = getattr(self.owner, name, None)
        return method if callable(method) else fallback
    @property
    def pipeline_stages(self) -> dict[int, PipelineStageState]:
        if not hasattr(self, "_pipeline_stages"):
            self._pipeline_stages = {}
        return self._pipeline_stages

    @property
    def pipeline_events(self) -> list[PipelineEvent]:
        if not hasattr(self, "_pipeline_events"):
            self._pipeline_events = []
        return self._pipeline_events

    @property
    def pipeline_connectors(self) -> dict[PipelineEdgeKind, PipelineStageConnector]:
        if not hasattr(self, "_pipeline_connectors"):
            self._pipeline_connectors = {}
        return self._pipeline_connectors

    @property
    def pipeline_send_tickets(self) -> dict[tuple[Any, ...], TransferTicket]:
        return self._get_pipeline_transport_state().send_tickets

    @property
    def pipeline_receive_reservations(self) -> dict[tuple[Any, ...], PipelineEdgeKind]:
        return self._get_pipeline_transport_state().receive_reservations

    @property
    def pipeline_started_receive_ids(self) -> set[tuple[Any, ...]]:
        return self._get_pipeline_transport_state().started_receive_ids

    @property
    def pipeline_receive_consumers(
        self,
    ) -> dict[tuple[Any, ...], tuple[PipelineEdgeKind, PipelineMessage, Any | None]]:
        return self._get_pipeline_transport_state().receive_consumers

    @property
    def pipeline_pending_received(self) -> dict[PipelineEdgeKind, deque[PipelineMessage]]:
        return self._get_pipeline_transport_state().pending_received

    def initialize_pipeline_transports(self, max_slots: int = 1) -> dict[str, Any]:
        """Build this Worker's granted activation and feedback P2P endpoints."""
        if self.pipeline_connectors:
            raise RuntimeError("pipeline transports are already initialized")
        pp_group = self._get_pp_group()
        if pp_group.world_size != 2:
            raise ValueError("queued v1 transport requires PP world size 2")
        src_rank, dst_rank = pp_group.ranks
        for edge_kind, edge_src, edge_dst in (
            (PipelineEdgeKind.ACTIVATION, src_rank, dst_rank),
            (PipelineEdgeKind.FEEDBACK, dst_rank, src_rank),
        ):
            transport = DistributedP2PTransport(
                group=pp_group,
                local_rank=self.rank,
                edge_kind=edge_kind,
                src_rank=edge_src,
                dst_rank=edge_dst,
            )
            self.pipeline_connectors[edge_kind] = PipelineStageConnector(
                edge=f"{edge_src}->{edge_dst}:{edge_kind.value}",
                max_slots=max_slots,
                transport=transport,
            )
        return {
            "rank": self.rank,
            "activation_edge": (src_rank, dst_rank),
            "feedback_edge": (dst_rank, src_rank),
        }

    def initialize_pipeline_transports_all_ranks(self, max_slots: int = 1) -> list[dict[str, Any]]:
        """Initialize locally and report every Worker's actual PP topology."""
        return self._run_and_gather_rank_values(
            "queued pipeline transport initialization",
            lambda: self.initialize_pipeline_transports(max_slots),
        )

    def reserve_pipeline_send(self, offer: PipelineTransferOffer, payload: dict[str, Any]) -> PipelineTransferOffer:
        return self._get_pipeline_transport_runtime().reserve_send(offer, payload)

    def accept_pipeline_transfer_offer(self, offer: PipelineTransferOffer) -> bool:
        return self._get_pipeline_transport_runtime().accept_offer(offer)

    def accept_pipeline_transfer_offer_all_ranks(self, offer: PipelineTransferOffer) -> bool:
        """Return false for temporary receive backpressure; raise on invalid readiness."""
        accept_offer = self._owner_method("accept_pipeline_transfer_offer", self.accept_pipeline_transfer_offer)
        rank_results = self._run_and_gather_rank_values(
            "queued pipeline transfer readiness",
            lambda: (self.rank, accept_offer(offer)),
        )
        endpoint_results: dict[int, bool] = {}
        for rank, ready in rank_results:
            if rank not in {offer.src_rank, offer.dst_rank}:
                continue
            if rank in endpoint_results or type(ready) is not bool:
                raise RuntimeError("pipeline transfer readiness returned invalid endpoint reports")
            endpoint_results[rank] = ready
        if set(endpoint_results) != {offer.src_rank, offer.dst_rank}:
            raise RuntimeError("pipeline transfer readiness did not report both endpoints")
        return all(endpoint_results.values())

    def accept_pipeline_transfer_offer_rank_local(self, offer: PipelineTransferOffer) -> dict[str, Any]:
        """Report this Worker's readiness without an in-Worker rank collective."""
        return {"rank": self.rank, "ready": self.accept_pipeline_transfer_offer(offer)}

    def accept_pipeline_transfer_offers_rank_local(
        self,
        offers: tuple[PipelineTransferOffer, ...] | list[PipelineTransferOffer],
    ) -> dict[str, Any]:
        """Report readiness for a batch of offers in one rank-local RPC."""
        if not isinstance(offers, (tuple, list)):
            raise TypeError("pipeline transfer offers must be a tuple or list")
        accept_offer = self._owner_method("accept_pipeline_transfer_offer", self.accept_pipeline_transfer_offer)
        readiness = [(offer.identity, accept_offer(offer)) for offer in offers]
        return {"rank": self.rank, "readiness": readiness}

    def start_pipeline_transfer(self, grant: PipelineTransferGrant) -> bool:
        return self._get_pipeline_transport_runtime().start_transfer(grant)

    def retire_pipeline_send(self, identity: tuple[Any, ...]) -> bool:
        return self._get_pipeline_transport_runtime().retire_send(identity)

    def progress_pipeline_transfers(self) -> PipelineTransportProgress:
        """Advance one local FIFO compute task and bounded transport work."""
        progress = PipelineTransportProgress(rank=self.rank)
        transport_runtime = self._get_pipeline_transport_runtime()

        transport_runtime.poll_completed_sends(progress)

        self._release_completed_pipeline_consumers(progress)

        for edge_kind in (PipelineEdgeKind.FEEDBACK, PipelineEdgeKind.ACTIVATION):
            transport_runtime.poll_received_into_pending(edge_kind)
            self._consume_ready_pipeline_message(edge_kind, progress)

        self._release_completed_pipeline_consumers(progress)

        first_stage = self.pipeline_stages.get(0)
        if first_stage is not None and first_stage.spec.is_first:
            activation_connector = self._require_pipeline_connector(PipelineEdgeKind.ACTIVATION)
            if (
                activation_connector.send_in_use < activation_connector.max_slots
                and not self._pipeline_stage_engine_has_unstarted_send(PipelineEdgeKind.ACTIVATION)
            ):
                stage_progress = self.progress_pipeline(0)
                if stage_progress is not None:
                    if not isinstance(stage_progress.output, PipelineTransferOffer):
                        raise RuntimeError("first pipeline stage did not reserve an activation transfer")
                    progress.offers.append(stage_progress.output)
        return progress

    def pipeline_stage_engine_tick(self) -> PipelineWorkerUpdate | None:
        """Advance one Worker-local turn and publish only meaningful metadata."""
        if set(self.pipeline_connectors) != {
            PipelineEdgeKind.ACTIVATION,
            PipelineEdgeKind.FEEDBACK,
        }:
            return None
        progress = self.progress_pipeline_transfers()
        self._reserve_expected_pipeline_receives(progress)
        events = tuple(self.poll_pipeline_events())
        finalizations = self._collect_completed_pipeline_finalizations()
        if not (progress.offers or progress.completions or progress.readiness or events or finalizations):
            return None
        return PipelineWorkerUpdate(
            worker_id=self.rank,
            progress=progress,
            events=events,
            finalizations=finalizations,
        )

    def pipeline_stage_engine_needs_progress(self) -> bool:
        """Whether local transport polling or an admitted task can make progress."""
        if set(self.pipeline_connectors) != {
            PipelineEdgeKind.ACTIVATION,
            PipelineEdgeKind.FEEDBACK,
        }:
            return False
        if self.pipeline_receive_consumers:
            return True
        if any(not ticket.started for ticket in self.pipeline_send_tickets.values()):
            return True
        for batch_id, future in self._pipeline_finalization_futures.items():
            if not future.done():
                continue
            if batch_id not in self._pipeline_finalization_published:
                return True
            event = self._pipeline_finalization_device_events.get(batch_id)
            if event is not None and callable(getattr(event, "query", None)) and not event.query():
                return True
        for connector in self.pipeline_connectors.values():
            transport = connector.transport
            if transport is not None and transport.has_outstanding_operations:
                return True
        for edge_kind, messages in self.pipeline_pending_received.items():
            if messages:
                return True
        return self._pipeline_stage_engine_has_runnable_forward() or self._pipeline_stage_engine_has_expected_receive()

    def _reserve_expected_pipeline_receives(self, progress: PipelineTransportProgress) -> None:
        """Reserve identity-specific credit for stage work already admitted locally."""
        candidates: list[tuple[PipelineEdgeKind, PipelineTask]] = []
        last_stage = self.pipeline_stages.get(1)
        if (
            last_stage is not None
            and last_stage.spec.is_last
            and last_stage.active_task is None
            and last_stage.pending_tasks
        ):
            task = last_stage.pending_tasks[0]
            if task.batch_id in last_stage.authorized_batches:
                candidates.append((PipelineEdgeKind.ACTIVATION, task))

        first_stage = self.pipeline_stages.get(0)
        if first_stage is not None and first_stage.spec.is_first:
            candidates.extend((PipelineEdgeKind.FEEDBACK, task) for task in first_stage.awaiting_feedback.values())

        for edge_kind in (PipelineEdgeKind.ACTIVATION, PipelineEdgeKind.FEEDBACK):
            connector = self._require_pipeline_connector(edge_kind)
            reserved = sum(kind is edge_kind for kind in self.pipeline_receive_reservations.values())
            available = connector.max_slots - reserved
            if available <= 0:
                continue
            for candidate_edge, task in candidates:
                if candidate_edge is not edge_kind or available <= 0:
                    continue
                offer = self._make_pipeline_transfer_offer(task, edge_kind)
                if offer.identity in self.pipeline_receive_reservations:
                    continue
                self.pipeline_receive_reservations[offer.identity] = edge_kind
                progress.readiness.append((offer.identity, True))
                available -= 1

    def _collect_completed_pipeline_finalizations(self) -> tuple[PipelineFinalizationUpdate, ...]:
        updates: list[PipelineFinalizationUpdate] = []
        for batch_id, future in self._pipeline_finalization_futures.items():
            if batch_id in self._pipeline_finalization_published or not future.done():
                continue
            try:
                output = future.result()
                device_event = self._pipeline_finalization_device_events.get(batch_id)
                if output is None and device_event is not None and not device_event.query():
                    continue
                self._pipeline_finalization_published.add(batch_id)
                updates.append(
                    PipelineFinalizationUpdate(
                        batch_id=batch_id,
                        output=output,
                        device_event=device_event,
                    )
                )
            except BaseException as exc:
                self._pipeline_finalization_published.add(batch_id)
                updates.append(
                    PipelineFinalizationUpdate(
                        batch_id=batch_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(updates)

    def _pipeline_stage_engine_has_runnable_forward(self) -> bool:
        stage = self.pipeline_stages.get(0)
        if stage is None or stage.active_task is not None or not stage.pending_tasks:
            return False
        task = stage.pending_tasks[0]
        connector = self.pipeline_connectors[PipelineEdgeKind.ACTIVATION]
        return (
            task.batch_id in stage.authorized_batches
            and connector.send_in_use < connector.max_slots
            and not self._pipeline_stage_engine_has_unstarted_send(PipelineEdgeKind.ACTIVATION)
        )

    def _pipeline_stage_engine_has_unstarted_send(self, edge_kind: PipelineEdgeKind) -> bool:
        return any(
            identity[4] is edge_kind and not ticket.started
            for identity, ticket in self.pipeline_send_tickets.items()
        )

    def _pipeline_stage_engine_has_expected_receive(self) -> bool:
        """Keep ticking until each admitted receive identity has announced credit."""
        stage = self.pipeline_stages.get(1)
        if (
            stage is not None
            and stage.spec.is_last
            and stage.active_task is None
            and stage.pending_tasks
            and stage.pending_tasks[0].batch_id in stage.authorized_batches
        ):
            task = stage.pending_tasks[0]
            offer = self._make_pipeline_transfer_offer(task, PipelineEdgeKind.ACTIVATION)
            if offer.identity not in self.pipeline_receive_reservations:
                return True
        stage = self.pipeline_stages.get(0)
        if stage is not None and stage.spec.is_first:
            return any(
                self._make_pipeline_transfer_offer(task, PipelineEdgeKind.FEEDBACK).identity
                not in self.pipeline_receive_reservations
                for task in stage.awaiting_feedback.values()
            )
        return False

    def progress_pipeline_transfers_and_poll_events_all_ranks(
        self,
    ) -> list[tuple[PipelineTransportProgress, list[Any]]]:
        """Advance each Worker and collect its events in the same rank agreement."""

        progress_fn = self._owner_method("progress_pipeline_transfers", self.progress_pipeline_transfers)
        poll_events = self._owner_method("poll_pipeline_events", self.poll_pipeline_events)

        def progress_and_poll() -> tuple[PipelineTransportProgress, list[Any]]:
            progress = progress_fn()
            return progress, poll_events()

        return self._run_and_gather_rank_values("queued pipeline progress snapshot", progress_and_poll)

    def progress_pipeline_transfers_and_poll_events(
        self,
        pending_offers: tuple[PipelineTransferOffer, ...] = (),
    ) -> tuple[PipelineTransportProgress, list[Any]]:
        """Advance one local Worker and return only its progress and events."""
        accept_offer = self._owner_method("accept_pipeline_transfer_offer", self.accept_pipeline_transfer_offer)
        progress_fn = self._owner_method("progress_pipeline_transfers", self.progress_pipeline_transfers)
        poll_events = self._owner_method("poll_pipeline_events", self.poll_pipeline_events)
        readiness = [(offer.identity, accept_offer(offer)) for offer in pending_offers]
        progress = progress_fn()
        progress.readiness = readiness
        return progress, poll_events()

    def _consume_ready_pipeline_message(
        self,
        edge_kind: PipelineEdgeKind,
        progress: PipelineTransportProgress,
    ) -> None:
        pending = self.pipeline_pending_received[edge_kind]
        if not pending:
            return
        message = pending[0]
        cancelled_context = self._cancelled_pipeline_message_context(edge_kind, message)
        if cancelled_context is not None:
            self._validate_pipeline_message_task_identity(cancelled_context.task, message)
            pending.popleft()
            reservation = self._find_pipeline_receive_reservation(edge_kind, message)
            self.release_pipeline_received(edge_kind, message)
            progress.completions.append(PipelineEndpointCompletion(identity=reservation, rank=self.rank))
            return
        if not self._pipeline_message_is_runnable(edge_kind, message):
            return
        reservation = self._find_pipeline_receive_reservation(edge_kind, message)
        if edge_kind is PipelineEdgeKind.ACTIVATION:
            # Do not consume or clone an activation unless its feedback
            # reservation can be created in the same turn.
            feedback_connector = self._require_pipeline_connector(PipelineEdgeKind.FEEDBACK)
            if feedback_connector.send_in_use >= feedback_connector.max_slots:
                return
        # The transport work is already complete when this message is
        # returned by poll_received(). Release only that ownership now so the
        # next transfer can use the bounded communication slot. Keep the
        # message in pipeline_receive_consumers as a separate compute lease;
        # its tensor remains alive until the stage has consumed it.
        self.release_pipeline_received(edge_kind, message)
        progress.completions.append(PipelineEndpointCompletion(identity=reservation, rank=self.rank))
        try:
            if edge_kind is PipelineEdgeKind.ACTIVATION:
                stage_progress = self.progress_pipeline(
                    1,
                    intermediate_tensors=IntermediateTensors(message.payload),
                )
                if stage_progress is None or not isinstance(stage_progress.output, PipelineTransferOffer):
                    raise RuntimeError("runnable pipeline activation made no local stage progress")
                progress.offers.append(stage_progress.output)
            else:
                latents = message.payload.get("latents") if isinstance(message.payload, dict) else None
                if not isinstance(latents, torch.Tensor):
                    raise RuntimeError("pipeline feedback message has no latent tensor")
                self.complete_pipeline_feedback(0, message.batch_id, latents)
        except BaseException:
            self.pipeline_receive_consumers[reservation] = (
                edge_kind,
                message,
                _PIPELINE_CONSUMER_EVENT_FAILED,
            )
            raise
        pending.popleft()
        consumer_event = current_omni_platform.record_device_event()
        if consumer_event is None and current_omni_platform.is_available():
            self.pipeline_receive_consumers[reservation] = (
                edge_kind,
                message,
                _PIPELINE_CONSUMER_EVENT_FAILED,
            )
            raise RuntimeError("failed to record pipeline receive consumer completion event")
        self.pipeline_receive_consumers[reservation] = (edge_kind, message, consumer_event)

    def _cancelled_pipeline_message_context(
        self,
        edge_kind: PipelineEdgeKind,
        message: PipelineMessage,
    ) -> Any | None:
        stage_id = 1 if edge_kind is PipelineEdgeKind.ACTIVATION else 0
        context = self.model_runner.pipeline_batch_contexts.get((stage_id, message.batch_id))
        if context is not None and context.status is PipelineTaskStatus.CANCELLED:
            return context
        return None

    def _pipeline_message_is_runnable(
        self,
        edge_kind: PipelineEdgeKind,
        message: PipelineMessage,
    ) -> bool:
        stage_id = 1 if edge_kind is PipelineEdgeKind.ACTIVATION else 0
        stage = self._require_pipeline_stage(stage_id)
        if edge_kind is PipelineEdgeKind.FEEDBACK:
            task = stage.awaiting_feedback.get(message.batch_id)
            if task is None:
                return False
            self._validate_pipeline_message_task_identity(task, message)
            return True
        if stage.active_task is not None or not stage.pending_tasks:
            return False
        head = stage.pending_tasks[0]
        if head.batch_id != message.batch_id:
            return False
        self._validate_pipeline_message_task_identity(head, message)
        if message.batch_id not in stage.authorized_batches:
            return False
        feedback_connector = self._require_pipeline_connector(PipelineEdgeKind.FEEDBACK)
        return feedback_connector.send_in_use < feedback_connector.max_slots

    @staticmethod
    def _validate_pipeline_message_task_identity(task: PipelineTask, message: PipelineMessage) -> None:
        task_identity = (task.batch_id, task.step_index, task.epoch, task.branch)
        message_identity = (message.batch_id, message.step_index, message.epoch, message.branch)
        if task_identity != message_identity:
            raise RuntimeError(
                f"Stale pipeline message identity {message_identity!r} does not match task {task_identity!r}."
            )

    def _release_completed_pipeline_consumers(self, progress: PipelineTransportProgress) -> None:
        del progress
        self._get_pipeline_transport_runtime().release_completed_consumers()

    def _find_pipeline_receive_reservation(
        self,
        edge_kind: PipelineEdgeKind,
        message: PipelineMessage,
    ) -> tuple[Any, ...]:
        return self._get_pipeline_transport_runtime().find_receive_reservation(edge_kind, message)

    def poll_pipeline_received(
        self,
        edge_kind: PipelineEdgeKind,
        limit: int = 1,
    ) -> list[PipelineMessage]:
        return self._get_pipeline_transport_runtime().poll_received(edge_kind, limit)

    def release_pipeline_received(self, edge_kind: PipelineEdgeKind, message: PipelineMessage) -> None:
        self._get_pipeline_transport_runtime().release_received(edge_kind, message)

    def _require_pipeline_connector(self, edge_kind: PipelineEdgeKind) -> PipelineStageConnector:
        return self._get_pipeline_transport_runtime().require_connector(edge_kind)

    def _pipeline_event(
        self,
        event_type: PipelineEventType,
        task: PipelineTask,
        pp_stage_id: int,
    ) -> PipelineEvent:
        return PipelineEvent(
            event_type=event_type,
            task=task,
            pp_stage_id=pp_stage_id,
            physical_rank=self.rank,
        )

    def _record_pipeline_event(self, event: PipelineEvent) -> PipelineEvent:
        self.pipeline_events.append(event)
        return event

    def _pipeline_stage(self, pp_stage_spec: PipelineStageSpec) -> PipelineStageState:
        stage = self.pipeline_stages.get(pp_stage_spec.pp_stage_id)
        if stage is None:
            stage = PipelineStageState(pp_stage_spec)
            self.pipeline_stages[pp_stage_spec.pp_stage_id] = stage
        elif stage.spec != pp_stage_spec:
            raise ValueError("Pipeline stage specification changed after Worker initialization.")
        return stage

    def enqueue_pipeline_batch(
        self,
        task: PipelineTask,
        pp_stage_spec: PipelineStageSpec | dict[int, PipelineStageSpec],
    ) -> PipelineEvent:
        """Resolve local request state and queue a metadata-only descriptor."""
        assert self.model_runner is not None, "Model runner not initialized"
        if isinstance(pp_stage_spec, dict):
            pp_stage_spec = self._select_rank_value(pp_stage_spec)
        states = []
        for request_id in task.request_ids:
            state = self.model_runner.state_cache.get(request_id)
            if state is None:
                raise ValueError(f"Missing prepared pipeline state for request {request_id!r}.")
            states.append(state)
        stage_was_new = pp_stage_spec.pp_stage_id not in self.pipeline_stages
        stage = self._pipeline_stage(pp_stage_spec)
        stage.enqueue(task)
        try:
            self.model_runner.prepare_pipeline_batch(task, pp_stage_spec, states)
        except BaseException:
            stage.rollback_pending(task.batch_id)
            if stage_was_new:
                self.pipeline_stages.pop(pp_stage_spec.pp_stage_id, None)
            raise
        return self._record_pipeline_event(
            self._pipeline_event(PipelineEventType.ACCEPTED, task, pp_stage_spec.pp_stage_id)
        )

    def prepare_pipeline_requests(self, scheduler_output: DiffusionSchedulerOutput) -> dict[str, Any]:
        """Prepare this Worker's request-local state without running a stage."""
        assert self.model_runner is not None, "Model runner not initialized"
        request_ids = self.model_runner.prepare_pipeline_requests(scheduler_output)
        return {"rank": self.rank, "request_ids": request_ids}

    def prepare_pipeline_requests_all_ranks(self, scheduler_output: DiffusionSchedulerOutput) -> list[dict[str, Any]]:
        """Prepare every rank at the same drained collective boundary."""
        return self._run_and_gather_rank_values(
            "queued pipeline request preparation",
            lambda: self.prepare_pipeline_requests(scheduler_output),
        )

    def authorize_pipeline_batch(self, pp_stage_id: int | dict[int, int], batch_id: str) -> PipelineEvent:
        """Apply the all-Worker acceptance gate's EXECUTE authorization."""
        if isinstance(pp_stage_id, dict):
            pp_stage_id = self._select_rank_value(pp_stage_id)
        stage = self._require_pipeline_stage(pp_stage_id)
        task = stage.authorize(batch_id)
        return self._record_pipeline_event(self._pipeline_event(PipelineEventType.AUTHORIZED, task, pp_stage_id))

    def authorize_pipeline_batches(
        self,
        authorizations: list[tuple[int | dict[int, int], str]],
    ) -> list[PipelineEvent]:
        """Authorize several already-enqueued tasks in one control call."""
        events = []
        for pp_stage_id, batch_id in authorizations:
            events.append(self.authorize_pipeline_batch(pp_stage_id, batch_id))
        return events

    def admit_pipeline_batch(
        self,
        task: PipelineTask,
        pp_stage_spec: PipelineStageSpec | dict[int, PipelineStageSpec],
    ) -> tuple[PipelineEvent, PipelineEvent]:
        accepted = self.enqueue_pipeline_batch(task, pp_stage_spec)
        authorized = self.authorize_pipeline_batch(accepted.pp_stage_id, task.batch_id)
        return accepted, authorized

    def admit_pipeline_batches(
        self,
        admissions: list[tuple[PipelineTask, PipelineStageSpec | dict[int, PipelineStageSpec]]],
    ) -> tuple[PipelineEvent, ...]:
        events: list[PipelineEvent] = []
        for task, pp_stage_spec in admissions:
            events.extend(self.admit_pipeline_batch(task, pp_stage_spec))
        return tuple(events)

    @staticmethod
    def _select_rank_value(values: dict[int, Any]) -> Any:
        from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_rank

        rank = get_pipeline_parallel_rank()
        if rank not in values:
            raise KeyError(f"No pipeline descriptor for local PP rank {rank}.")
        return values[rank]

    def progress_pipeline(
        self,
        pp_stage_id: int,
        intermediate_tensors: Any | None = None,
    ) -> PipelineProgress | None:
        """Run at most one authorized FIFO head through the local model stage."""
        assert self.model_runner is not None, "Model runner not initialized"
        if set(self.pipeline_connectors) != {
            PipelineEdgeKind.ACTIVATION,
            PipelineEdgeKind.FEEDBACK,
        }:
            raise RuntimeError("pipeline transports must be initialized before queued progression")
        stage = self._require_pipeline_stage(pp_stage_id)
        task = stage.start_next()
        if task is None:
            return None
        context = self.model_runner.pipeline_batch_contexts.get((pp_stage_id, task.batch_id))
        if context is None:
            stage.fail_active()
            raise RuntimeError(f"Pipeline batch {task.batch_id!r} has no ModelRunner context.")
        try:
            result = self.model_runner.execute_pipeline_stage(context, stage.spec, intermediate_tensors)
            output = result
            if stage.spec.is_last:
                output = self.model_runner.complete_pipeline_step(context, stage.spec)
            if self.pipeline_connectors:
                if stage.spec.is_first:
                    tensors = getattr(output, "tensors", None)
                    if not isinstance(tensors, dict):
                        raise RuntimeError("first pipeline stage did not produce intermediate tensors")
                    offer = self._make_pipeline_transfer_offer(task, PipelineEdgeKind.ACTIVATION, tensors)
                    self.reserve_pipeline_send(offer, tensors)
                    output = offer
                elif stage.spec.is_last:
                    if not isinstance(output, torch.Tensor):
                        raise RuntimeError("last pipeline stage did not produce latent feedback")
                    offer = self._make_pipeline_transfer_offer(task, PipelineEdgeKind.FEEDBACK, {"latents": output})
                    self.reserve_pipeline_send(offer, {"latents": output})
                    output = offer
            if stage.spec.is_last:
                stage.complete_active()
            else:
                stage.await_feedback()
        except BaseException:
            context.status = PipelineTaskStatus.FAILED
            if stage.active_task is not None and stage.active_task.batch_id == task.batch_id:
                stage.fail_active()
            raise
        return PipelineProgress(
            event=self._record_pipeline_event(
                self._pipeline_event(PipelineEventType.STAGE_COMPLETED, task, pp_stage_id)
            ),
            output=output,
        )

    def _make_pipeline_transfer_offer(
        self,
        task: PipelineTask,
        edge_kind: PipelineEdgeKind,
        payload: dict[str, Any] | None = None,
    ) -> PipelineTransferOffer:
        connector = self._require_pipeline_connector(edge_kind)
        transport = connector.transport
        if not isinstance(transport, DistributedP2PTransport):
            raise RuntimeError("queued pipeline stage does not use distributed P2P transport")
        return PipelineTransferOffer(
            batch_id=task.batch_id,
            step_index=task.step_index,
            epoch=task.epoch,
            branch=task.branch,
            edge_kind=edge_kind,
            src_rank=transport.src_rank,
            dst_rank=transport.dst_rank,
            payload_metadata=() if payload is None else pipeline_payload_metadata(payload),
        )

    def complete_pipeline_feedback(
        self,
        pp_stage_id: int,
        batch_id: str,
        latents: torch.Tensor,
    ) -> PipelineEvent:
        """Adopt feedback on stage 0 and emit the sole step-completion event."""
        assert self.model_runner is not None, "Model runner not initialized"
        stage = self._require_pipeline_stage(pp_stage_id)
        if not stage.spec.is_first:
            raise ValueError("Only the first pipeline stage can complete feedback adoption.")
        context = self.model_runner.pipeline_batch_contexts.get((pp_stage_id, batch_id))
        if context is None:
            raise RuntimeError(f"Pipeline batch {batch_id!r} has no ModelRunner context.")
        if stage.terminal_statuses.get(batch_id) is PipelineTaskStatus.CANCELLED:
            if context.status is not PipelineTaskStatus.CANCELLED:
                raise RuntimeError("Cancelled pipeline stage and ModelRunner context disagree.")
            return self._record_pipeline_event(
                self._pipeline_event(PipelineEventType.CANCELLED, context.task, pp_stage_id)
            )
        task = stage.awaiting_feedback.get(batch_id)
        if task is None:
            raise RuntimeError(f"Pipeline batch {batch_id!r} is not awaiting feedback on stage {pp_stage_id}.")
        try:
            self.model_runner.adopt_pipeline_feedback(context, stage.spec, latents)
            stage.complete_feedback(batch_id)
        except BaseException:
            if batch_id in stage.awaiting_feedback:
                stage.fail_feedback(batch_id)
            raise
        return self._record_pipeline_event(self._pipeline_event(PipelineEventType.STEP_COMPLETED, task, pp_stage_id))

    def cancel_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineEvent:
        """Make a pending or active local batch terminal without releasing it."""
        assert self.model_runner is not None, "Model runner not initialized"
        stage = self._require_pipeline_stage(pp_stage_id)
        context = self.model_runner.pipeline_batch_contexts.get((pp_stage_id, batch_id))
        if context is None:
            raise KeyError(f"Unknown pipeline batch context {(pp_stage_id, batch_id)!r}.")
        finalization = getattr(self, "_pipeline_finalization_futures", {}).get(batch_id)
        if finalization is not None and not finalization.done():
            finalization.result()
        self.model_runner.cancel_pipeline_batch(pp_stage_id, batch_id)
        if not stage.cancel(batch_id):
            raise RuntimeError(f"Pipeline batch {batch_id!r} was not cancellable on stage {pp_stage_id}.")
        self._release_unstarted_pipeline_batch_transfers(context.task)
        return self._record_pipeline_event(self._pipeline_event(PipelineEventType.CANCELLED, context.task, pp_stage_id))

    def _release_unstarted_pipeline_batch_transfers(self, task: PipelineTask) -> None:
        self._get_pipeline_transport_runtime().release_unstarted_batch_transfers(task)

    def release_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineEvent:
        """Release one terminal ModelRunner context after dependent work retires."""
        assert self.model_runner is not None, "Model runner not initialized"
        stage = self._require_pipeline_stage(pp_stage_id)
        if batch_id in stage.retired_batches:
            raise ValueError(f"batch {batch_id!r} is already retired")
        if batch_id not in stage.terminal_statuses:
            raise RuntimeError(f"Pipeline batch {batch_id!r} is not terminal on stage {pp_stage_id}.")
        retained_transfers = [identity for identity in self.pipeline_send_tickets if identity[0] == batch_id]
        if retained_transfers:
            raise RuntimeError("Cannot release a pipeline batch with retained transfer ownership.")
        retained_receives = [identity for identity in self.pipeline_receive_reservations if identity[0] == batch_id]
        if retained_receives:
            raise RuntimeError("Cannot release a pipeline batch with retained receive ownership.")
        retained_consumers = [identity for identity in self.pipeline_receive_consumers if identity[0] == batch_id]
        if retained_consumers:
            raise RuntimeError("Cannot release a pipeline batch with active receive consumers.")
        context = self.model_runner.release_pipeline_batch(pp_stage_id, batch_id)
        stage.retire(batch_id)
        if hasattr(self, "_pipeline_finalization_futures") and batch_id in self._pipeline_finalization_futures:
            self._get_pipeline_finalization_state().clear_batch(batch_id)
        return self._record_pipeline_event(self._pipeline_event(PipelineEventType.RELEASED, context.task, pp_stage_id))

    def pipeline_batch_release_ready(self, pp_stage_id: int | dict[int, int], batch_id: str) -> bool:
        """Check whether a terminal batch has no retained transport ownership."""
        if isinstance(pp_stage_id, dict):
            pp_stage_id = self._select_rank_value(pp_stage_id)
        stage = self._require_pipeline_stage(pp_stage_id)
        if batch_id not in stage.terminal_statuses:
            raise RuntimeError(f"Pipeline batch {batch_id!r} is not terminal on stage {pp_stage_id}.")
        finalization = getattr(self, "_pipeline_finalization_futures", {}).get(batch_id)
        if finalization is not None and not finalization.done():
            return False
        device_event = getattr(self, "_pipeline_finalization_device_events", {}).get(batch_id)
        if device_event is not None and callable(getattr(device_event, "query", None)) and not device_event.query():
            return False
        return not any(
            identity[0] == batch_id
            for identity in (
                *self.pipeline_send_tickets,
                *self.pipeline_receive_reservations,
                *self.pipeline_receive_consumers,
            )
        )

    def pipeline_batch_release_ready_all_ranks(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
    ) -> bool:
        """Agree on release readiness without mutating batch ownership."""
        results = self._run_and_gather_rank_values(
            "queued pipeline batch release readiness",
            lambda: self.pipeline_batch_release_ready(pp_stage_id, batch_id),
        )
        if not all(type(result) is bool for result in results):
            raise RuntimeError("queued pipeline batch release readiness returned invalid reports")
        return all(results)

    def finalize_pipeline_batch(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
        output_rank: int | None = None,
    ) -> str | None:
        """Submit local decode on its selected owner or join distributed VAE decode."""
        if isinstance(pp_stage_id, dict):
            pp_stage_id = self._select_rank_value(pp_stage_id)

        parallel = getattr(self.od_config, "parallel_config", None)
        distributed_vae_requested = int(getattr(parallel, "vae_patch_parallel_size", 1) or 1) > 1
        if distributed_vae_requested:
            local: dict[str, Any] = {}
            pp_group = self._get_pp_group()
            expected_stage_ranks = dict(enumerate(pp_group.ranks))
            if output_rank is None:
                output_rank = expected_stage_ranks[0]
            if output_rank != expected_stage_ranks[0]:
                raise ValueError("distributed VAE finalization output must stay on the first PP stage")

            def validate_local_decode() -> dict[str, Any]:
                stage = self._require_pipeline_stage(pp_stage_id)
                context = self.model_runner.pipeline_batch_contexts.get((pp_stage_id, batch_id))
                if context is None:
                    raise KeyError(f"Unknown pipeline batch context {(pp_stage_id, batch_id)!r}.")
                state = self.model_runner.validate_pipeline_finalization(context, stage.spec)
                if not self.model_runner.pipeline_has_distributed_vae():
                    raise RuntimeError("distributed VAE decode was requested but is not enabled locally")
                local.update(stage=stage, context=context)
                return {
                    "rank": self.rank,
                    "pp_stage_id": stage.spec.pp_stage_id,
                    "batch_id": batch_id,
                    "shape": tuple(state.latents.shape),
                    "dtype": str(state.latents.dtype),
                }

            reports = self._run_and_gather_rank_values(
                "queued distributed VAE finalization readiness",
                validate_local_decode,
            )
            reports_are_mappings = all(isinstance(report, dict) for report in reports)
            reported_stage_ranks = (
                {report.get("pp_stage_id"): report.get("rank") for report in reports} if reports_are_mappings else {}
            )
            if (
                len(reports) != len(expected_stage_ranks)
                or reported_stage_ranks != expected_stage_ranks
                or any(report.get("batch_id") != batch_id for report in reports if isinstance(report, dict))
                or not reports_are_mappings
                or len({(report.get("shape"), report.get("dtype")) for report in reports if isinstance(report, dict)})
                != 1
            ):
                raise RuntimeError("distributed VAE finalization readiness did not match across PP ranks")
            stage = local["stage"]
            context = local["context"]
            should_finalize = True
            return_handle = True
        else:
            stage = self._require_pipeline_stage(pp_stage_id)
            context = self.model_runner.pipeline_batch_contexts.get((pp_stage_id, batch_id))
            if context is None:
                raise KeyError(f"Unknown pipeline batch context {(pp_stage_id, batch_id)!r}.")
            self.model_runner.validate_pipeline_finalization(context, stage.spec)
            if output_rank is None:
                should_finalize = stage.spec.is_first
                return_handle = should_finalize
                output_rank = self.rank if should_finalize else None
            else:
                pp_group = self._get_pp_group()
                if type(output_rank) is not int or output_rank not in pp_group.ranks:
                    raise ValueError("queued finalization output rank must belong to the PP group")
                should_finalize = self.rank == output_rank
                return_handle = True
        if not should_finalize:
            return batch_id if return_handle else None

        if batch_id not in self._pipeline_finalization_futures:
            finalization_executor = self._get_pipeline_finalization_state().ensure_executor(self.rank)

            def finalize() -> BatchRunnerOutput | None:
                started_at = time.perf_counter()
                logger.info("Queued pipeline final decode started batch=%s", batch_id)
                device = getattr(self, "device", None)
                if device is not None:
                    current_omni_platform.set_device(device)
                stream_context = nullcontext()
                if device is not None and torch.device(device).type == "cuda":
                    if self._pipeline_finalization_stream is None:
                        self._pipeline_finalization_stream = torch.cuda.Stream(device=device)
                    self._pipeline_finalization_stream.wait_stream(torch.cuda.current_stream(device))
                    stream_context = torch.cuda.stream(self._pipeline_finalization_stream)
                with stream_context:
                    result = self.model_runner.finalize_pipeline_batch(
                        context,
                        stage.spec,
                        output_owner=self.rank == output_rank,
                    )
                    device_event = current_omni_platform.record_device_event()
                # A missing native event on an accelerator still needs a
                # synchronous completion barrier before the result is
                # published.  CPU test workers do not have a device and the
                # unspecified platform intentionally has no synchronize().
                if (
                    device_event is None
                    and device is not None
                    and torch.device(device).type != "cpu"
                    and current_omni_platform.is_available()
                ):
                    current_omni_platform.synchronize()
                self._pipeline_finalization_device_events[batch_id] = device_event
                logger.info(
                    "Queued pipeline final decode finished batch=%s elapsed_ms=%.3f",
                    batch_id,
                    (time.perf_counter() - started_at) * 1000,
                )
                return result

            future = finalization_executor.submit(finalize)
            self._pipeline_finalization_futures[batch_id] = future
            wake_stage_engine = getattr(self, "_pipeline_stage_engine_wake", None)
            if callable(wake_stage_engine):
                future.add_done_callback(lambda _completed: wake_stage_engine())
        return batch_id if return_handle else None

    def poll_pipeline_finalization(self, batch_id: str) -> BatchRunnerOutput | None:
        """Return a completed decode result without blocking the Worker RPC loop."""
        future = self._pipeline_finalization_futures.get(batch_id)
        if future is None:
            raise KeyError(f"Unknown queued pipeline finalization {batch_id!r}.")
        if not future.done():
            return None
        return future.result()

    def release_pipeline_batch_all_ranks(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
    ) -> list[PipelineEvent]:
        """Release every rank-local context and gather acknowledgements."""
        local_event: PipelineEvent | None = None

        def release_local() -> PipelineEvent:
            nonlocal local_event
            local_event = self.release_pipeline_batch(
                self._select_rank_value(pp_stage_id) if isinstance(pp_stage_id, dict) else pp_stage_id,
                batch_id,
            )
            return local_event

        acknowledgements = self._run_and_gather_rank_values(
            "queued pipeline batch release",
            release_local,
        )
        if local_event is not None:
            self._pipeline_events = [event for event in self.pipeline_events if event is not local_event]
        return acknowledgements

    def cleanup_finalized_pipeline_request(self, request_id: str) -> bool:
        """Drop persistent request tensors after final decode and retirement."""
        assert self.model_runner is not None, "Model runner not initialized"
        self.model_runner.state_cache.pop(request_id, None)
        self.model_runner.input_batch = None
        remove_kv = getattr(self.model_runner, "remove_diffusion_kv_requests", None)
        if callable(remove_kv):
            remove_kv([request_id])
        return True

    def cleanup_finalized_pipeline_request_all_ranks(self, request_id: str) -> list[bool]:
        return self._run_and_gather_rank_values(
            "queued finalized request cleanup",
            lambda: self.cleanup_finalized_pipeline_request(request_id),
        )

    def poll_pipeline_events(self) -> list[PipelineEvent]:
        """Return and clear buffered metadata-only pipeline events."""
        events, self._pipeline_events = self.pipeline_events, []
        return events

    def pipeline_stage_memory_budget_bytes(self) -> list[dict[str, int]]:
        local_report = {"rank": self.rank, "free_bytes": int(current_omni_platform.get_free_memory(self.device))}
        pp_group = self._get_pp_group()
        if pp_group.world_size == 1:
            return [local_report]
        reports: list[dict[str, int] | None] = [None] * pp_group.world_size
        dist.all_gather_object(reports, local_report, group=pp_group.cpu_group)
        return [report for report in reports if report is not None]

    def poll_pipeline_events_all_ranks(self) -> list[PipelineEvent]:
        """Clear every rank's queue and return all events on the reply rank."""
        rank_events = self._all_gather_rank_values(self.poll_pipeline_events())
        return [event for events in rank_events for event in events]

    def cancel_pipeline_requests(self, request_generations: Any) -> list[PipelineEvent]:
        """Cancel all local contexts matching request IDs or (ID, generation)."""
        is_single_pair = (
            isinstance(request_generations, (tuple, list))
            and len(request_generations) == 2
            and isinstance(request_generations[0], str)
            and type(request_generations[1]) is int
        )
        requested = (
            [request_generations]
            if is_single_pair or not isinstance(request_generations, (list, tuple, set))
            else request_generations
        )
        request_ids: set[str] = set()
        request_epochs: set[tuple[str, int]] = set()
        for item in requested:
            if isinstance(item, (tuple, list)):
                if len(item) != 2 or not isinstance(item[0], str) or type(item[1]) is not int:
                    raise ValueError("generation-scoped cancellation requires (request_id, epoch)")
                request_epochs.add((item[0], item[1]))
            elif isinstance(item, str):
                request_ids.add(item)
            else:
                raise ValueError("pipeline cancellation selectors must be request IDs or (request_id, epoch)")
        events: list[PipelineEvent] = []
        for (pp_stage_id, batch_id), context in list(self.model_runner.pipeline_batch_contexts.items()):
            matches_request = bool(request_ids.intersection(context.request_state_ids))
            matches_generation = any(
                (request_id, context.task.epoch) in request_epochs for request_id in context.request_state_ids
            )
            if matches_request or matches_generation:
                events.append(self.cancel_pipeline_batch(pp_stage_id, batch_id))
        return events

    def cancel_pipeline_requests_all_ranks(self, request_generations: Any) -> list[PipelineEvent]:
        """Cancel matching local contexts and gather acknowledgements from every rank."""
        local_events: list[PipelineEvent] = []

        def cancel_local() -> list[PipelineEvent]:
            nonlocal local_events
            local_events = self.cancel_pipeline_requests(request_generations)
            return local_events

        try:
            rank_events = self._run_and_gather_rank_values(
                "queued pipeline cancellation",
                cancel_local,
            )
        finally:
            if local_events:
                local_event_ids = {id(event) for event in local_events}
                self._pipeline_events = [event for event in self.pipeline_events if id(event) not in local_event_ids]
        return [event for events in rank_events for event in events]

    def drain_pipeline(self, deadline: float | None = None) -> list[PipelineEvent]:
        """Require all local pipeline contexts to be retired before shutdown."""
        del deadline
        if self.model_runner.pipeline_batch_contexts:
            raise RuntimeError("cannot drain pipeline with unreleased batch contexts")
        retained_tickets = [ticket for ticket in self.pipeline_send_tickets.values() if not ticket.released]
        if retained_tickets:
            raise RuntimeError("cannot drain pipeline with retained send tickets")
        if self.pipeline_receive_reservations:
            raise RuntimeError("cannot drain pipeline with reserved receive credit")
        if self.pipeline_receive_consumers:
            raise RuntimeError("cannot drain pipeline with active receive consumers")
        busy_connectors = {
            edge_kind.value: health
            for edge_kind, connector in self.pipeline_connectors.items()
            if (
                (health := connector.health())["send_in_use"]
                or health["receive_depth"]
                or health["transport_pending"]
                or health["transport_outstanding"]
            )
        }
        if busy_connectors:
            raise RuntimeError(f"cannot drain pipeline with active connector ownership: {busy_connectors}")
        return []

    def drain_pipeline_all_ranks(self, deadline: float | None = None) -> list[PipelineEvent]:
        """Coordinate the drain guard, then gather terminal events from all ranks."""
        self._run_and_gather_rank_values("queued pipeline drain", lambda: self.drain_pipeline(deadline))
        return self.poll_pipeline_events_all_ranks()
