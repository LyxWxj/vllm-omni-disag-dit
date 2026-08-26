# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clock, credit, and P2P transport primitives for interleaved diffusion PP.

``PipelineClockSimulator`` is the CPU-only reference for the retained-state
clock. ``PipelineP2PChannel`` applies the same bounded-slot contract to one
real ``torch.distributed`` PP edge: receive buffers belong to the downstream
stage, credits return only after local consumption, and an explicit send
sequence restores FIFO delivery when distinct physical slots complete out of
order.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from threading import Event, Thread
from typing import Any, ClassVar

from vllm_omni.diffusion.distributed import pp_trace


@dataclass(frozen=True, slots=True)
class PipelineToken:
    """One homogeneous microbatch moving through the PP clock.

    ``slot_id`` is assigned by the edge that currently owns the token.  The
    remaining fields are the small control header needed by the later device
    transport; model tensors and static per-request state intentionally do not
    belong in this object.
    """

    request_ids: tuple[str, ...]
    row_map: tuple[int, ...]
    step_idx: int
    cfg_branch: str
    microbatch_id: int
    token_id: str
    slot_id: int | None
    compatibility_key: Hashable
    model_phase: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ids", tuple(self.request_ids))
        object.__setattr__(self, "row_map", tuple(self.row_map))
        if not self.request_ids:
            raise ValueError("PipelineToken.request_ids must not be empty")
        if len(self.request_ids) != len(self.row_map):
            raise ValueError("PipelineToken.row_map must have one entry per request_id")
        if self.step_idx < 0:
            raise ValueError("PipelineToken.step_idx must be non-negative")
        if self.microbatch_id < 0:
            raise ValueError("PipelineToken.microbatch_id must be non-negative")
        if not self.token_id:
            raise ValueError("PipelineToken.token_id must not be empty")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("PipelineToken.slot_id must be non-negative when assigned")

    def for_slot(self, slot_id: int) -> PipelineToken:
        """Return this token with the physical edge slot recorded in its header."""
        return replace(self, slot_id=slot_id)


class PipelineSlotState(Enum):
    """State of one fixed-capacity receive slot on a PP edge."""

    FREE = auto()
    RECV_POSTED = auto()
    READY = auto()
    COMPUTING = auto()
    SEND_PENDING = auto()


def pipeline_edge_pairs(pp_ranks: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return adjacent forward edges plus the last-to-first feedback edge."""
    ranks = tuple(pp_ranks)
    if len(ranks) < 2:
        raise ValueError("interleaved pipeline parallelism requires at least two ranks")
    return (*zip(ranks[:-1], ranks[1:], strict=True), (ranks[-1], ranks[0]))


@dataclass(frozen=True, slots=True)
class PipelineTransportHeader:
    """Fixed-size control header carried with one PP tensor payload.

    ``cfg_branch`` is intentionally an integer transport value. Model code
    owns the mapping from a branch name to that value, keeping P2P metadata
    fixed length and avoiding object collectives in the clock hot path.
    ``slot_id`` and ``send_sequence`` are assigned by the outgoing edge.
    """

    token_id: int
    step_idx: int
    cfg_branch: int
    flags: int = 0
    slot_id: int | None = None
    send_sequence: int | None = None

    FIELD_COUNT: ClassVar[int] = 6
    SHUTDOWN_FLAG: ClassVar[int] = 1
    CANCELLED_FLAG: ClassVar[int] = 2
    INT64_MAX: ClassVar[int] = 2**63 - 1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("token_id", self.token_id),
            ("step_idx", self.step_idx),
            ("cfg_branch", self.cfg_branch),
            ("flags", self.flags),
        ):
            if not isinstance(value, int) or not 0 <= value <= self.INT64_MAX:
                raise ValueError(f"PipelineTransportHeader.{field_name} must fit in signed int64")
        for field_name, value in (("slot_id", self.slot_id), ("send_sequence", self.send_sequence)):
            if value is not None and (not isinstance(value, int) or not 0 <= value <= self.INT64_MAX):
                raise ValueError(f"PipelineTransportHeader.{field_name} must fit in signed int64 when assigned")

    def for_slot(self, slot_id: int, send_sequence: int) -> PipelineTransportHeader:
        """Return this header stamped with one physical slot and FIFO sequence."""
        return replace(self, slot_id=slot_id, send_sequence=send_sequence)

    def encode_into(self, buffer: Any) -> None:
        """Write the fixed transport representation into a preallocated tensor."""
        if self.slot_id is None or self.send_sequence is None:
            raise ValueError("slot_id and send_sequence must be assigned before transport")
        if buffer.numel() != self.FIELD_COUNT:
            raise ValueError(f"header buffer must contain exactly {self.FIELD_COUNT} values")
        buffer.copy_(
            buffer.new_tensor(
                (
                    self.token_id,
                    self.slot_id,
                    self.step_idx,
                    self.cfg_branch,
                    self.flags,
                    self.send_sequence,
                )
            )
        )

    @classmethod
    def decode(cls, buffer: Any) -> PipelineTransportHeader:
        """Decode a fixed transport header after its receive work completes."""
        if buffer.numel() != cls.FIELD_COUNT:
            raise ValueError(f"header buffer must contain exactly {cls.FIELD_COUNT} values")
        token_id, slot_id, step_idx, cfg_branch, flags, send_sequence = (int(value) for value in buffer.tolist())
        return cls(
            token_id=token_id,
            slot_id=slot_id,
            step_idx=step_idx,
            cfg_branch=cfg_branch,
            flags=flags,
            send_sequence=send_sequence,
        )


@dataclass(frozen=True, slots=True)
class PipelineTransportMessage:
    """A received transport header and its receiver-owned payload buffer."""

    header: PipelineTransportHeader
    payload: Any


class _P2PSendSlot:
    """A sender-owned output buffer that cannot be overwritten while in use."""

    def __init__(self, slot_id: int, header_buffer: Any, payload_buffer: Any) -> None:
        self.slot_id = slot_id
        self.header_buffer = header_buffer
        self.payload_buffer = payload_buffer
        self.header_work: Any | None = None
        self.payload_work: Any | None = None
        self.state = PipelineSlotState.FREE
        self.history = [self.state]

    @property
    def is_send_pending(self) -> bool:
        return self.state is PipelineSlotState.SEND_PENDING

    def begin_send(self) -> None:
        if self.state is not PipelineSlotState.FREE:
            raise RuntimeError(f"send slot {self.slot_id} is not free")
        self.state = PipelineSlotState.SEND_PENDING
        self.history.append(self.state)

    def finish_send(self) -> None:
        if self.state is not PipelineSlotState.SEND_PENDING:
            raise RuntimeError(f"send slot {self.slot_id} has no pending transfer")
        self.header_work = None
        self.payload_work = None
        self.state = PipelineSlotState.FREE
        self.history.append(self.state)


class _P2PReceiveSlot:
    """A receiver-owned input buffer with an explicit reuse lifecycle."""

    def __init__(self, slot_id: int, header_buffer: Any, payload_buffer: Any) -> None:
        self.slot_id = slot_id
        self.header_buffer = header_buffer
        self.payload_buffer = payload_buffer
        self.header_work: Any | None = None
        self.payload_work: Any | None = None
        self.header: PipelineTransportHeader | None = None
        self.state = PipelineSlotState.FREE
        self.history = [self.state]

    def transition(self, expected: PipelineSlotState, new: PipelineSlotState) -> None:
        if self.state is not expected:
            raise RuntimeError(
                f"receive slot {self.slot_id} transition {self.state.name}->{new.name} requires {expected.name}"
            )
        self.state = new
        self.history.append(new)


class _TrackedWork:
    """Gloo work plus the event/error state owned by its progress thread."""

    def __init__(self, work: Any) -> None:
        self.work = work
        self.completed = Event()
        self.error: BaseException | None = None


class PipelineP2PChannel:
    """Retained-state P2P runtime for one directed, fixed-shape PP edge.

    The destination owns the receive ring. It pre-posts one header and tensor
    receive per physical slot, then returns a slot credit only after
    :meth:`release_after_compute`. The source can send only after it has
    received that credit, and copies each payload into its own per-slot output
    buffer before starting ``isend``. This gives bounded memory and protects
    both input and output buffer lifetimes without a blocking metadata path.
    """

    _TAGS_PER_SLOT = 3

    def __init__(
        self,
        *,
        source_rank: int,
        destination_rank: int,
        tensor_shape: tuple[int, ...],
        tensor_dtype: Any,
        device: Any,
        slots_per_edge: int = 2,
        tag_base: int = 0,
        group: Any = None,
    ) -> None:
        if source_rank < 0 or destination_rank < 0 or source_rank == destination_rank:
            raise ValueError("source_rank and destination_rank must be distinct non-negative ranks")
        if slots_per_edge < 1:
            raise ValueError("slots_per_edge must be at least one")
        if tag_base < 0:
            raise ValueError("tag_base must be non-negative")
        if any(not isinstance(dim, int) or dim < 0 for dim in tensor_shape):
            raise ValueError("tensor_shape dimensions must be non-negative ints")

        import torch
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("PipelineP2PChannel requires an initialized torch.distributed process group")

        self._torch = torch
        self._dist = dist
        self._group = group
        self._requires_wait_thread = dist.get_backend(group) == "gloo"
        self.source_rank = source_rank
        self.destination_rank = destination_rank
        self.slots_per_edge = slots_per_edge
        self.tag_base = tag_base
        self.tensor_shape = tuple(tensor_shape)
        self.tensor_dtype = tensor_dtype
        self.device = torch.device(device)
        self.rank = dist.get_rank()
        if self.rank not in (source_rank, destination_rank):
            raise ValueError(f"rank {self.rank} is not an endpoint of channel {source_rank}->{destination_rank}")

        self._is_source = self.rank == source_rank
        self._peer_rank = destination_rank if self._is_source else source_rank
        self._next_send_sequence = 0
        self._next_receive_sequence = 0
        self._ready_slot_ids: deque[int] = deque()
        self._received_by_sequence: dict[int, _P2PReceiveSlot] = {}
        self._max_occupied = 0
        self._shutdown_requested = False
        self._shutdown_slot_ids: set[int] = set()
        self._closed = False

        if self._is_source:
            self._send_slots = [
                _P2PSendSlot(
                    slot_id,
                    torch.empty(PipelineTransportHeader.FIELD_COUNT, dtype=torch.int64, device=self.device),
                    torch.empty(self.tensor_shape, dtype=tensor_dtype, device=self.device),
                )
                for slot_id in range(slots_per_edge)
            ]
            self._credit_buffers = [
                torch.empty(1, dtype=torch.int64, device=self.device) for _ in range(slots_per_edge)
            ]
            self._credit_works: list[Any | None] = [None] * slots_per_edge
            self._credit_slot_ids: deque[int] = deque()
            self._credit_slot_id_set: set[int] = set()
            for slot_id in range(slots_per_edge):
                self._post_credit_receive(slot_id)
        else:
            self._receive_slots = [
                _P2PReceiveSlot(
                    slot_id,
                    torch.empty(PipelineTransportHeader.FIELD_COUNT, dtype=torch.int64, device=self.device),
                    torch.empty(self.tensor_shape, dtype=tensor_dtype, device=self.device),
                )
                for slot_id in range(slots_per_edge)
            ]
            self._credit_buffers = [
                torch.empty(1, dtype=torch.int64, device=self.device) for _ in range(slots_per_edge)
            ]
            self._credit_works = [None] * slots_per_edge
            for slot in self._receive_slots:
                self._post_receive(slot)
                self._send_credit(slot.slot_id)

    @property
    def is_source(self) -> bool:
        return self._is_source

    @property
    def is_destination(self) -> bool:
        return not self._is_source

    @property
    def is_closed(self) -> bool:
        """Whether this edge has completed its explicit shutdown handshake."""
        return self._closed

    @property
    def available_credits(self) -> int:
        """Credits visible to the source after it has polled incoming messages."""
        self._require_source()
        return len(self._credit_slot_ids)

    @property
    def can_send(self) -> bool:
        """Whether the source can begin one new payload transfer now."""
        self._require_source()
        self.poll()
        return any(self._send_slots[slot_id].state is PipelineSlotState.FREE for slot_id in self._credit_slot_ids)

    @property
    def has_ready_message(self) -> bool:
        self._require_destination()
        self.poll()
        return bool(self._ready_slot_ids)

    @property
    def num_ready_messages(self) -> int:
        self._require_destination()
        self.poll()
        return len(self._ready_slot_ids)

    @property
    def max_occupied(self) -> int:
        self._require_destination()
        return self._max_occupied

    @property
    def receive_slot_state_histories(self) -> tuple[tuple[PipelineSlotState, ...], ...]:
        self._require_destination()
        return tuple(tuple(slot.history) for slot in self._receive_slots)

    @property
    def send_slot_state_histories(self) -> tuple[tuple[PipelineSlotState, ...], ...]:
        self._require_source()
        return tuple(tuple(slot.history) for slot in self._send_slots)

    @property
    def in_flight_send_count(self) -> int:
        self._require_source()
        self.poll()
        return sum(slot.is_send_pending for slot in self._send_slots)

    @property
    def pending_work_count(self) -> int:
        """Number of live send/receive Work handles retained by this edge."""
        if self._is_source:
            return sum(
                work is not None for slot in self._send_slots for work in (slot.header_work, slot.payload_work)
            ) + sum(work is not None for work in self._credit_works)
        return sum(
            work is not None for slot in self._receive_slots for work in (slot.header_work, slot.payload_work)
        ) + sum(work is not None for work in self._credit_works)

    def poll(self) -> None:
        """Advance local non-blocking work without waiting for a peer."""
        if self._is_source:
            self._poll_source()
        else:
            self._poll_destination()

    def send(self, header: PipelineTransportHeader, payload: Any) -> PipelineTransportHeader:
        """Copy and send one payload after reserving a downstream receive credit."""
        self._require_source()
        self.poll()
        self._validate_payload(payload)
        # Prepare all local data before mutating credit, sequence, or Work
        # state. A failed header conversion or payload copy is retryable.
        slot_id = self._peek_send_credit()
        slot = self._send_slots[slot_id]
        if slot.is_send_pending:
            raise RuntimeError(f"slot {slot_id} credit returned before its previous send completed")

        transport_header = header.for_slot(slot_id, self._next_send_sequence)
        transport_header.encode_into(slot.header_buffer)
        slot.payload_buffer.copy_(payload)

        self._consume_send_credit(slot_id)
        slot.begin_send()
        self._next_send_sequence += 1
        if not self._shutdown_requested:
            self._post_credit_receive(slot_id)
        slot.header_work = self._start_work(
            self._dist.isend(
                slot.header_buffer,
                dst=self._peer_rank,
                group=self._group,
                tag=self._header_tag(slot_id),
            )
        )
        slot.payload_work = self._start_work(
            self._dist.isend(
                slot.payload_buffer,
                dst=self._peer_rank,
                group=self._group,
                tag=self._payload_tag(slot_id),
            )
        )
        return transport_header

    def begin_compute(self) -> PipelineTransportMessage:
        """Reserve the oldest FIFO-ready input buffer for local stage compute."""
        self._require_destination()
        self.poll()
        if not self._ready_slot_ids:
            raise RuntimeError("attempted to compute without a READY pipeline message")
        slot = self._receive_slots[self._ready_slot_ids.popleft()]
        slot.transition(PipelineSlotState.READY, PipelineSlotState.COMPUTING)
        assert slot.header is not None
        return PipelineTransportMessage(header=slot.header, payload=slot.payload_buffer)

    def release_after_compute(self, message: PipelineTransportMessage) -> None:
        """Release a consumed input and return its physical slot credit upstream."""
        self._require_destination()
        slot_id = message.header.slot_id
        if slot_id is None:
            raise ValueError("received PipelineTransportMessage has no slot_id")
        slot = self._receive_slots[slot_id]
        if slot.state is not PipelineSlotState.COMPUTING or slot.header != message.header:
            raise RuntimeError("attempted to release a message that is not currently computing")
        slot.transition(PipelineSlotState.COMPUTING, PipelineSlotState.FREE)
        slot.header = None
        if message.header.flags & PipelineTransportHeader.SHUTDOWN_FLAG:
            self._shutdown_slot_ids.add(slot_id)
            self._closed = len(self._shutdown_slot_ids) == self.slots_per_edge
            return
        self._post_receive(slot)
        self._send_credit(slot_id)

    def send_shutdown(self) -> None:
        """Send one tombstone per physical slot and finish the edge cleanly."""
        self._require_source()
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.wait_for_sends()
        payload = self._torch.zeros(self.tensor_shape, dtype=self.tensor_dtype, device=self.device)
        for _ in range(self.slots_per_edge):
            while not self.can_send:
                self.poll()
            self.send(
                PipelineTransportHeader(
                    token_id=0,
                    step_idx=0,
                    cfg_branch=0,
                    flags=PipelineTransportHeader.SHUTDOWN_FLAG,
                ),
                payload,
            )
        self.wait_for_sends()
        self._closed = True

    def wait_for_sends(self) -> None:
        """Wait for locally-issued sends; receive work remains non-blocking."""
        if self._is_source:
            for slot in self._send_slots:
                self._wait_for_works(slot.header_work, slot.payload_work)
            self._poll_source()
            return
        for slot_id, work in enumerate(self._credit_works):
            self._wait_for_works(work)
            self._credit_works[slot_id] = None
        self._poll_destination()

    def wait_for_shutdown(self) -> None:
        """Wait for all source-side Work after an explicit shutdown handshake.

        Normal progression must not block on a future credit because an idle
        edge intentionally has no reposted ``irecv``.  During shutdown every
        normal payload has either returned its credit or has been superseded
        by a tombstone on the same slot, so retaining such Work would leak the
        Gloo waiter thread into the next process-group lifetime.
        """
        self._require_source()
        self.wait_for_sends()
        for slot_id, work in enumerate(self._credit_works):
            self._wait_for_works(work)
            self._credit_works[slot_id] = None
        self._poll_source()

    def _poll_source(self) -> None:
        for slot in self._send_slots:
            if slot.is_send_pending and self._works_complete(slot.header_work, slot.payload_work):
                self._wait_for_works(slot.header_work, slot.payload_work)
                slot.finish_send()

        for expected_slot_id, work in enumerate(self._credit_works):
            if not self._works_complete(work):
                continue
            self._wait_for_works(work)
            self._credit_works[expected_slot_id] = None
            credit_slot_id = int(self._credit_buffers[expected_slot_id].item())
            if credit_slot_id != expected_slot_id:
                raise RuntimeError(
                    f"credit tag for slot {expected_slot_id} carried slot {credit_slot_id}; PP edge is desynchronized"
                )
            if credit_slot_id in self._credit_slot_id_set:
                raise RuntimeError(f"received duplicate credit for slot {credit_slot_id}")
            self._credit_slot_ids.append(credit_slot_id)
            self._credit_slot_id_set.add(credit_slot_id)

    def _poll_destination(self) -> None:
        for slot in self._receive_slots:
            if slot.state is not PipelineSlotState.RECV_POSTED or slot.header is not None:
                continue
            if not self._works_complete(slot.header_work, slot.payload_work):
                continue
            self._wait_for_works(slot.header_work, slot.payload_work)
            slot.header_work = None
            slot.payload_work = None
            header = PipelineTransportHeader.decode(slot.header_buffer)
            if header.slot_id != slot.slot_id:
                raise RuntimeError(f"header for receive slot {slot.slot_id} carries mismatched slot {header.slot_id}")
            assert header.send_sequence is not None
            if header.send_sequence < self._next_receive_sequence:
                raise RuntimeError(f"received stale sequence {header.send_sequence} on PP edge")
            if header.send_sequence in self._received_by_sequence:
                raise RuntimeError(f"received duplicate sequence {header.send_sequence} on PP edge")
            slot.header = header
            self._received_by_sequence[header.send_sequence] = slot
            self._max_occupied = max(
                self._max_occupied,
                sum(receive_slot.header is not None for receive_slot in self._receive_slots),
            )

        while (slot := self._received_by_sequence.pop(self._next_receive_sequence, None)) is not None:
            slot.transition(PipelineSlotState.RECV_POSTED, PipelineSlotState.READY)
            self._ready_slot_ids.append(slot.slot_id)
            self._next_receive_sequence += 1

    def _post_credit_receive(self, slot_id: int) -> None:
        self._credit_works[slot_id] = self._start_work(
            self._dist.irecv(
                self._credit_buffers[slot_id],
                src=self._peer_rank,
                group=self._group,
                tag=self._credit_tag(slot_id),
            )
        )

    def _post_receive(self, slot: _P2PReceiveSlot) -> None:
        slot.transition(PipelineSlotState.FREE, PipelineSlotState.RECV_POSTED)
        slot.header_work = self._start_work(
            self._dist.irecv(
                slot.header_buffer,
                src=self._peer_rank,
                group=self._group,
                tag=self._header_tag(slot.slot_id),
            )
        )
        slot.payload_work = self._start_work(
            self._dist.irecv(
                slot.payload_buffer,
                src=self._peer_rank,
                group=self._group,
                tag=self._payload_tag(slot.slot_id),
            )
        )

    def _send_credit(self, slot_id: int) -> None:
        previous_work = self._credit_works[slot_id]
        self._wait_for_works(previous_work)
        self._credit_buffers[slot_id].fill_(slot_id)
        self._credit_works[slot_id] = self._start_work(
            self._dist.isend(
                self._credit_buffers[slot_id],
                dst=self._peer_rank,
                group=self._group,
                tag=self._credit_tag(slot_id),
            )
        )

    def _peek_send_credit(self) -> int:
        for slot_id in self._credit_slot_ids:
            if self._send_slots[slot_id].state is PipelineSlotState.FREE:
                return slot_id
        raise RuntimeError("attempted to send without a downstream credit")

    def _consume_send_credit(self, slot_id: int) -> None:
        if slot_id not in self._credit_slot_id_set:
            raise RuntimeError(f"credit for slot {slot_id} is no longer available")
        self._credit_slot_ids.remove(slot_id)
        self._credit_slot_id_set.remove(slot_id)

    def _validate_payload(self, payload: Any) -> None:
        if tuple(payload.shape) != self.tensor_shape:
            raise ValueError(f"payload shape {tuple(payload.shape)} does not match fixed shape {self.tensor_shape}")
        if payload.dtype != self.tensor_dtype:
            raise ValueError(f"payload dtype {payload.dtype} does not match channel dtype {self.tensor_dtype}")
        if payload.device != self.device:
            raise ValueError(f"payload device {payload.device} does not match channel device {self.device}")

    def _header_tag(self, slot_id: int) -> int:
        return self.tag_base + slot_id * self._TAGS_PER_SLOT

    def _payload_tag(self, slot_id: int) -> int:
        return self._header_tag(slot_id) + 1

    def _credit_tag(self, slot_id: int) -> int:
        return self._header_tag(slot_id) + 2

    def _start_work(self, work: Any) -> Any:
        if not self._requires_wait_thread:
            return work

        tracked_work = _TrackedWork(work)

        def wait_for_gloo_work() -> None:
            try:
                tracked_work.work.wait()
            except BaseException as exc:  # pragma: no cover - backend failure path.
                tracked_work.error = exc
            finally:
                tracked_work.completed.set()

        Thread(target=wait_for_gloo_work, daemon=True).start()
        return tracked_work

    def _works_complete(self, *works: Any | None) -> bool:
        for work in works:
            if work is None:
                return False
            if isinstance(work, _TrackedWork):
                if not work.completed.is_set():
                    return False
            elif not work.is_completed():
                return False
        return True

    def _wait_for_works(self, *works: Any | None) -> None:
        for work in works:
            if work is None:
                continue
            if isinstance(work, _TrackedWork):
                work.completed.wait()
                if work.error is not None:
                    raise work.error
            else:
                work.wait()

    def _require_source(self) -> None:
        if not self._is_source:
            raise RuntimeError("operation is only valid on the source rank of a PipelineP2PChannel")

    def _require_destination(self) -> None:
        if self._is_source:
            raise RuntimeError("operation is only valid on the destination rank of a PipelineP2PChannel")


@dataclass(frozen=True, slots=True)
class PipelineTensorSpec:
    """Fixed tensor layouts for one homogeneous PP microbatch."""

    intermediate_shape: tuple[int, ...]
    intermediate_dtype: Any
    feedback_shape: tuple[int, ...]
    feedback_dtype: Any

    def __post_init__(self) -> None:
        for field_name, shape in (
            ("intermediate_shape", self.intermediate_shape),
            ("feedback_shape", self.feedback_shape),
        ):
            normalized = tuple(shape)
            if not normalized or any(not isinstance(dim, int) or dim < 1 for dim in normalized):
                raise ValueError(f"PipelineTensorSpec.{field_name} must contain positive integer dimensions")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True, slots=True)
class PipelineTickCompletion:
    """One request-local scheduler result received by stage 0 in a tick."""

    request_ids: tuple[str, ...]
    step_idx: int


@dataclass(frozen=True, slots=True)
class _PipelineMicrobatch:
    """Static metadata shared by all PP ranks for one admitted microbatch."""

    token_id_by_branch: Mapping[int, int]
    request_ids: tuple[str, ...]
    row_counts: tuple[int, ...]
    step_idx: int
    microbatch_id: int
    spec: PipelineTensorSpec
    model_phase: str
    do_true_cfg: bool

    @property
    def row_map(self) -> tuple[int, ...]:
        return tuple(range(len(self.request_ids)))


class PipelineTickRuntime:
    """Persistent one-clock PP runtime used by ``execute_pipeline_tick``.

    Request-local state stays on each rank. Only fixed tensor payloads cross
    the data path, so headers can be resolved from token metadata registered
    during the collective scheduler admission RPC.
    """

    _BRANCH_POSITIVE = 0
    _BRANCH_NEGATIVE = 1
    _BRANCH_FEEDBACK = 2
    _TAG_BASE = 10_000
    _MODEL_PHASE_SHIFT = 8

    def __init__(
        self,
        *,
        pipeline: Any,
        state_cache: dict[str, Any],
        pp_ranks: Sequence[int],
        global_rank: int,
        device: Any,
        edge_groups: Mapping[tuple[int, int], Any],
        slots_per_edge: int = 2,
    ) -> None:
        if len(pp_ranks) < 2:
            raise ValueError("PipelineTickRuntime requires at least two PP ranks")
        if global_rank not in pp_ranks:
            raise ValueError("global_rank must belong to pp_ranks")
        if slots_per_edge < 1:
            raise ValueError("slots_per_edge must be positive")
        required = (
            "build_microbatches",
            "pipeline_transport_spec",
            "pipeline_model_phase",
            "pipeline_forward_local_stage",
            "pipeline_finish_microbatch",
        )
        missing = [name for name in required if not callable(getattr(pipeline, name, None))]
        if missing:
            raise ValueError(f"{type(pipeline).__name__} does not implement interleaved PP hooks: {', '.join(missing)}")

        self.pipeline = pipeline
        self.state_cache = state_cache
        self.pp_ranks = tuple(pp_ranks)
        self.global_rank = global_rank
        self.stage = self.pp_ranks.index(global_rank)
        self.world_size = len(self.pp_ranks)
        self.device = device
        self.edge_groups = edge_groups
        self.slots_per_edge = slots_per_edge
        self.clock = 0

        self._next_token_id = 1
        self._next_microbatch_id = 0
        self._waiting: deque[int] = deque()
        self._microbatches: dict[int, _PipelineMicrobatch] = {}
        self._tokens: dict[int, PipelineToken] = {}
        self._cancelled_request_ids: set[str] = set()
        self._positive_noise: dict[int, Any] = {}
        self._model_phase_ids: dict[str, int] = {}
        self._spec_ids: dict[PipelineTensorSpec, int] = {}
        self._forward_channels: dict[tuple[int, PipelineTensorSpec], PipelineP2PChannel] = {}
        self._feedback_channels: dict[PipelineTensorSpec, PipelineP2PChannel] = {}

    @property
    def is_first_stage(self) -> bool:
        return self.stage == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage == self.world_size - 1

    @property
    def has_in_flight_work(self) -> bool:
        """Whether rank 0 still owns a token that must reach feedback.

        A terminal scheduler status does not imply physical PP work is gone:
        a cancelled token must still drain its receiver-owned slots.  Every
        non-cancelled microbatch is retained here until rank 0 consumes its
        feedback, while a cancelled one is retained until its fixed-shape
        feedback tombstone arrives.
        """
        return bool(self._waiting or self._microbatches)

    def cancel(self, request_ids: Sequence[str]) -> None:
        """Mark cancelled work for discard after any physical transfer drains."""
        self._cancelled_request_ids.update(request_ids)
        for token_id, plan in tuple(self._microbatches.items()):
            if self._is_fully_cancelled(plan):
                self._positive_noise.pop(token_id, None)

    def admit(self, states: Sequence[Any]) -> None:
        """Register scheduler-admitted state and queue stage-0 branches.

        Every PP rank calls this for the same scheduler RPC. Only stage 0 owns
        the injection queue, but all ranks retain the token registry needed to
        decode a later fixed-size P2P header without an object collective.
        """
        for microbatch_states in self.pipeline.build_microbatches(states):
            if not microbatch_states:
                continue
            spec = self.pipeline.pipeline_transport_spec(microbatch_states)
            if not isinstance(spec, PipelineTensorSpec):
                raise TypeError("pipeline_transport_spec() must return PipelineTensorSpec")
            self._ensure_channels(spec)

            step_idx = self._step_index(microbatch_states)
            model_phase = self.pipeline.pipeline_model_phase(microbatch_states)
            if not isinstance(model_phase, str) or not model_phase:
                raise ValueError("pipeline_model_phase() must return a non-empty string")
            self._register_model_phase(model_phase)
            do_true_cfg = bool(getattr(microbatch_states[0], "do_true_cfg", False))
            if any(bool(getattr(state, "do_true_cfg", False)) != do_true_cfg for state in microbatch_states[1:]):
                raise ValueError("one pipeline microbatch must have one CFG policy")

            branches = (self._BRANCH_POSITIVE, self._BRANCH_NEGATIVE) if do_true_cfg else (self._BRANCH_POSITIVE,)
            token_id_by_branch = {branch: self._allocate_token_id() for branch in branches}
            plan = _PipelineMicrobatch(
                token_id_by_branch=token_id_by_branch,
                request_ids=tuple(state.request_id for state in microbatch_states),
                row_counts=tuple(self._state_row_count(state) for state in microbatch_states),
                step_idx=step_idx,
                microbatch_id=self._next_microbatch_id,
                spec=spec,
                model_phase=model_phase,
                do_true_cfg=do_true_cfg,
            )
            self._next_microbatch_id += 1
            for branch, token_id in token_id_by_branch.items():
                token = PipelineToken(
                    request_ids=plan.request_ids,
                    row_map=plan.row_map,
                    step_idx=step_idx,
                    cfg_branch=self._branch_name(branch),
                    microbatch_id=plan.microbatch_id,
                    token_id=str(token_id),
                    slot_id=None,
                    compatibility_key=spec,
                    model_phase=model_phase,
                )
                self._tokens[token_id] = token
                self._microbatches[token_id] = plan
                if self.is_first_stage:
                    self._waiting.append(token_id)
                pp_trace.event(
                    "token_registered",
                    pp_rank=self.stage,
                    pp_size=self.world_size,
                    clock=self.clock,
                    token_id=token_id,
                    request_ids=plan.request_ids,
                    microbatch_id=plan.microbatch_id,
                    step_idx=plan.step_idx,
                    cfg_branch=self._branch_name(branch),
                    model_phase=plan.model_phase,
                )

    def progress_one_clock(self) -> tuple[PipelineTickCompletion, ...]:
        """Advance at most one local DiT stage and consume ready feedback."""
        self._poll_channels()
        completions = self._consume_feedback() if self.is_first_stage else []
        self._run_one_local_stage()
        self.clock += 1
        return tuple(completions)

    def close(self) -> None:
        """Close an already-drained lane without retaining P2P Work handles.

        Worker shutdown invokes this only after the engine has stopped issuing
        clocks.  Every PP worker executes the method, so each source endpoint
        can emit its per-slot tombstones while each destination polls and
        releases them.  It intentionally refuses to close stage-0 queued work
        or an unmatched positive CFG branch: those are runtime bugs, not safe
        teardown states.
        """
        if self._waiting or self._positive_noise:
            raise RuntimeError("cannot close PipelineTickRuntime with queued or partial CFG tokens")

        channels = tuple({id(channel): channel for channel in self._all_channels()}.values())
        for channel in channels:
            if channel.is_source:
                channel.send_shutdown()

        destinations = [channel for channel in channels if channel.is_destination]
        shutdown_deadline = time.monotonic() + 10.0
        while True:
            for channel in destinations:
                channel.poll()
                while channel.has_ready_message:
                    message = channel.begin_compute()
                    channel.release_after_compute(message)
            if all(channel.is_closed for channel in destinations):
                break
            if time.monotonic() >= shutdown_deadline:
                raise RuntimeError("pipeline channel shutdown did not receive every tombstone")
            # Gloo's background Work waiters need an opportunity to publish
            # their completion between non-blocking polling rounds.
            time.sleep(0.001)

        for channel in channels:
            if channel.is_source:
                channel.wait_for_shutdown()
            else:
                channel.wait_for_sends()
            if channel.pending_work_count != 0:
                raise RuntimeError("pipeline channel retained pending Work after shutdown")

    def _all_channels(self) -> Sequence[PipelineP2PChannel]:
        return (*self._forward_channels.values(), *self._feedback_channels.values())

    def _run_one_local_stage(self) -> None:
        token_id: int | None = None
        incoming: PipelineP2PChannel | None = None
        message: PipelineTransportMessage | None = None

        if self.is_first_stage:
            if not self._waiting:
                return
            token_id = self._waiting[0]
            plan = self._microbatches[token_id]
            if not self._forward_channel(self.stage, plan.spec).can_send:
                pp_trace.event(
                    "credit_wait",
                    pp_rank=self.stage,
                    pp_size=self.world_size,
                    clock=self.clock,
                    token_id=self._tokens[token_id].token_id,
                    request_ids=plan.request_ids,
                    microbatch_id=plan.microbatch_id,
                    step_idx=plan.step_idx,
                    cfg_branch=self._tokens[token_id].cfg_branch,
                    model_phase=plan.model_phase,
                    slot_id=None,
                )
                return
            self._waiting.popleft()
        else:
            candidate = self._next_ready_input()
            if candidate is None:
                return
            incoming, spec = candidate
            if not self.is_last_stage and not self._forward_channel(self.stage, spec).can_send:
                pp_trace.event(
                    "credit_wait",
                    pp_rank=self.stage,
                    pp_size=self.world_size,
                    clock=self.clock,
                    token_id=None,
                    request_ids=(),
                    microbatch_id=None,
                    step_idx=None,
                    cfg_branch=None,
                    model_phase=None,
                    slot_id=None,
                )
                return
            if self.is_last_stage and not self._feedback_channel(spec).can_send:
                pp_trace.event(
                    "credit_wait",
                    pp_rank=self.stage,
                    pp_size=self.world_size,
                    clock=self.clock,
                    token_id=None,
                    request_ids=(),
                    microbatch_id=None,
                    step_idx=None,
                    cfg_branch=None,
                    model_phase=None,
                    slot_id=None,
                )
                return
            message = incoming.begin_compute()
            token_id = message.header.token_id
            self._validate_received_header(message.header, spec)

        assert token_id is not None
        plan = self._microbatches[token_id]
        token = self._tokens[token_id]
        trace_fields = {
            "clock": self.clock,
            "token_id": token_id,
            "request_ids": plan.request_ids,
            "microbatch_id": plan.microbatch_id,
            "step_idx": token.step_idx,
            "cfg_branch": token.cfg_branch,
            "model_phase": token.model_phase,
            "slot_id": token.slot_id if token.slot_id is not None else (message.header.slot_id if message else None),
        }
        if message is not None:
            pp_trace.event("recv_ready", pp_rank=self.stage, pp_size=self.world_size, **trace_fields)
        try:
            cancelled = self._is_fully_cancelled(plan) or (
                message is not None and bool(message.header.flags & PipelineTransportHeader.CANCELLED_FLAG)
            )
            if cancelled:
                self._forward_cancelled_token(token_id, token, plan)
                return
            states = self._states_for(plan)
            self._set_state_step(states, token.step_idx)
            input_batch = self._make_input_batch(states)
            with pp_trace.span(
                "stage_forward",
                pp_rank=self.stage,
                pp_size=self.world_size,
                device=self.device,
                **trace_fields,
            ):
                result = self.pipeline.pipeline_forward_local_stage(
                    input_batch,
                    states=states,
                    cfg_branch=token.cfg_branch,
                    intermediate_hidden_states=None if message is None else message.payload,
                )

            if not self.is_last_stage:
                payload = self._extract_tensor(result, "pipeline_forward_local_stage")
                self._validate_tensor(
                    payload, plan.spec.intermediate_shape, plan.spec.intermediate_dtype, "intermediate"
                )
                self._forward_channel(self.stage, plan.spec).send(self._header_for(token_id, token), payload)
                if not self.is_first_stage:
                    self._advance_nonfirst_state(states, token.step_idx)
                    if self._is_feedback_token(plan, token_id):
                        self._discard_cancelled_states(plan)
                    self._retire_token(token_id)
                elif plan.do_true_cfg and self._branch_code(token.cfg_branch) == self._BRANCH_POSITIVE:
                    # Only the negative branch produces the feedback token.
                    # Stage 0 no longer needs the positive branch after send.
                    self._retire_token(token_id)
                elif self._is_feedback_token(plan, token_id):
                    self._discard_cancelled_states(plan)
                return

            self._finish_last_stage(token_id, token, plan, states, result)
        finally:
            if message is not None and incoming is not None:
                incoming.release_after_compute(message)

    def _forward_cancelled_token(
        self,
        token_id: int,
        token: PipelineToken,
        plan: _PipelineMicrobatch,
    ) -> None:
        """Drain a fully cancelled token without reusing its model buffers.

        Stage 0 injects a fixed-shape tombstone for queued work; middle
        stages relay it; the last stage returns one feedback tombstone for the
        canonical branch.  This preserves P2P order and releases every credit
        before rank 0 reports the lane idle.
        """
        feedback_token_id = self._feedback_token_id(plan)
        if self.is_last_stage:
            if token_id == feedback_token_id:
                import torch

                payload = torch.zeros(
                    plan.spec.feedback_shape,
                    dtype=plan.spec.feedback_dtype,
                    device=self.device,
                )
                self._feedback_channel(plan.spec).send(
                    PipelineTransportHeader(
                        token_id=feedback_token_id,
                        step_idx=token.step_idx,
                        cfg_branch=self._BRANCH_FEEDBACK,
                        flags=self._model_phase_flags(token.model_phase) | PipelineTransportHeader.CANCELLED_FLAG,
                    ),
                    payload,
                )
                self._discard_cancelled_states(plan)
            self._retire_token(token_id)
            return

        import torch

        payload = torch.zeros(
            plan.spec.intermediate_shape,
            dtype=plan.spec.intermediate_dtype,
            device=self.device,
        )
        self._forward_channel(self.stage, plan.spec).send(
            self._header_for(token_id, token, cancelled=True),
            payload,
        )
        if token_id == feedback_token_id:
            self._discard_cancelled_states(plan)
        if not self.is_first_stage or token_id != feedback_token_id:
            self._retire_token(token_id)

    def _finish_last_stage(
        self,
        token_id: int,
        token: PipelineToken,
        plan: _PipelineMicrobatch,
        states: Sequence[Any],
        result: Any,
    ) -> None:
        noise_pred = self._extract_tensor(result, "pipeline_forward_local_stage")
        branch = self._branch_code(token.cfg_branch)
        if plan.do_true_cfg and branch == self._BRANCH_POSITIVE:
            self._positive_noise[token_id] = noise_pred
            self._advance_nonfirst_state(states, token.step_idx)
            self._retire_token(token_id, retain_positive_noise=True)
            return

        positive_noise = None
        if plan.do_true_cfg:
            positive_token_id = plan.token_id_by_branch[self._BRANCH_POSITIVE]
            positive_noise = self._positive_noise.pop(positive_token_id, None)
            if positive_noise is None:
                raise RuntimeError("negative CFG token reached the last stage before its positive token")

        latents = self._extract_tensor(
            self.pipeline.pipeline_finish_microbatch(states, noise_pred, positive_noise_pred=positive_noise),
            "pipeline_finish_microbatch",
        )
        self._validate_tensor(latents, plan.spec.feedback_shape, plan.spec.feedback_dtype, "feedback")
        self._feedback_channel(plan.spec).send(
            PipelineTransportHeader(
                token_id=token_id,
                step_idx=token.step_idx,
                cfg_branch=self._BRANCH_FEEDBACK,
                flags=self._model_phase_flags(token.model_phase),
            ),
            latents,
        )
        self._advance_nonfirst_state(states, token.step_idx)
        self._discard_cancelled_states(plan)
        self._retire_token(token_id)

    def _consume_feedback(self) -> list[PipelineTickCompletion]:
        completions: list[PipelineTickCompletion] = []
        for spec, channel in self._feedback_channels.items():
            channel.poll()
            while channel.has_ready_message:
                message = channel.begin_compute()
                try:
                    header = message.header
                    if header.cfg_branch != self._BRANCH_FEEDBACK:
                        raise RuntimeError("feedback edge received a non-feedback token")
                    plan = self._microbatches.get(header.token_id)
                    token = self._tokens.get(header.token_id)
                    if plan is None or token is None:
                        raise RuntimeError(f"feedback for unknown pipeline token {header.token_id}")
                    if (
                        plan.spec != spec
                        or header.step_idx != token.step_idx
                        or (header.flags & ~PipelineTransportHeader.CANCELLED_FLAG)
                        != self._model_phase_flags(token.model_phase)
                    ):
                        raise RuntimeError("feedback header does not match its registered token")
                    if header.flags & PipelineTransportHeader.CANCELLED_FLAG:
                        if not self._is_fully_cancelled(plan):
                            raise RuntimeError("received a cancellation tombstone for a live pipeline token")
                        self._retire_feedback_plan(header.token_id, plan)
                        continue
                    self._validate_tensor(message.payload, spec.feedback_shape, spec.feedback_dtype, "feedback")
                    self._apply_feedback(plan, message.payload, header.step_idx)
                    active_request_ids = self._active_request_ids(plan)
                    if active_request_ids:
                        completions.append(PipelineTickCompletion(active_request_ids, header.step_idx))
                    self._retire_feedback_plan(header.token_id, plan)
                finally:
                    channel.release_after_compute(message)
        return completions

    def _retire_feedback_plan(self, token_id: int, plan: _PipelineMicrobatch) -> None:
        """Drop rank-0 metadata once a feedback message has been consumed."""
        self._discard_cancelled_states(plan)
        self._retire_token(token_id)
        if plan.do_true_cfg:
            self._retire_token(plan.token_id_by_branch[self._BRANCH_POSITIVE])

    def _retire_token(self, token_id: int, *, retain_positive_noise: bool = False) -> None:
        """Release local metadata once this rank will not see the token again."""
        self._tokens.pop(token_id, None)
        self._microbatches.pop(token_id, None)
        if not retain_positive_noise:
            self._positive_noise.pop(token_id, None)
        active_request_ids = {request_id for plan in self._microbatches.values() for request_id in plan.request_ids}
        self._cancelled_request_ids.intersection_update(active_request_ids)

    def _apply_feedback(self, plan: _PipelineMicrobatch, latents: Any, step_idx: int) -> None:
        offset = 0
        for request_id, row_count in zip(plan.request_ids, plan.row_counts, strict=True):
            next_offset = offset + row_count
            state = self.state_cache.get(request_id)
            if state is None:
                if request_id not in self._cancelled_request_ids:
                    raise RuntimeError(f"pipeline feedback references unknown request state {request_id!r}")
                offset = next_offset
                continue
            state.latents = latents[offset:next_offset].clone()
            state.step_index = step_idx + 1
            offset = next_offset
        if offset != latents.shape[0]:
            raise RuntimeError("feedback row count does not match its request state")

    def _next_ready_input(self) -> tuple[PipelineP2PChannel, PipelineTensorSpec] | None:
        candidates: list[tuple[int, PipelineP2PChannel, PipelineTensorSpec]] = []
        for spec, spec_id in self._spec_ids.items():
            channel = self._forward_channel(self.stage - 1, spec)
            channel.poll()
            if channel.has_ready_message:
                candidates.append((spec_id, channel, spec))
        if not candidates:
            return None
        _, channel, spec = min(candidates, key=lambda candidate: candidate[0])
        return channel, spec

    def _ensure_channels(self, spec: PipelineTensorSpec) -> None:
        if spec in self._spec_ids:
            return
        spec_id = len(self._spec_ids)
        self._spec_ids[spec] = spec_id
        tag_stride = (self.world_size + 1) * self.slots_per_edge * PipelineP2PChannel._TAGS_PER_SLOT
        tag_base = self._TAG_BASE + spec_id * tag_stride
        for edge in range(self.world_size - 1):
            if self.stage not in (edge, edge + 1):
                continue
            self._forward_channels[(edge, spec)] = PipelineP2PChannel(
                source_rank=self.pp_ranks[edge],
                destination_rank=self.pp_ranks[edge + 1],
                tensor_shape=spec.intermediate_shape,
                tensor_dtype=spec.intermediate_dtype,
                device=self.device,
                slots_per_edge=self.slots_per_edge,
                tag_base=tag_base + edge * self.slots_per_edge * PipelineP2PChannel._TAGS_PER_SLOT,
                group=self._edge_group(self.pp_ranks[edge], self.pp_ranks[edge + 1]),
            )
        if self.stage in (0, self.world_size - 1):
            self._feedback_channels[spec] = PipelineP2PChannel(
                source_rank=self.pp_ranks[-1],
                destination_rank=self.pp_ranks[0],
                tensor_shape=spec.feedback_shape,
                tensor_dtype=spec.feedback_dtype,
                device=self.device,
                slots_per_edge=self.slots_per_edge,
                tag_base=tag_base + (self.world_size - 1) * self.slots_per_edge * PipelineP2PChannel._TAGS_PER_SLOT,
                group=self._edge_group(self.pp_ranks[-1], self.pp_ranks[0]),
            )

    def _poll_channels(self) -> None:
        for channel in self._forward_channels.values():
            channel.poll()
        for channel in self._feedback_channels.values():
            channel.poll()

    def _forward_channel(self, edge: int, spec: PipelineTensorSpec) -> PipelineP2PChannel:
        try:
            return self._forward_channels[(edge, spec)]
        except KeyError as exc:
            raise RuntimeError(f"missing local pipeline edge {edge} for tensor spec") from exc

    def _feedback_channel(self, spec: PipelineTensorSpec) -> PipelineP2PChannel:
        try:
            return self._feedback_channels[spec]
        except KeyError as exc:
            raise RuntimeError("missing local pipeline feedback edge") from exc

    def _edge_group(self, source_rank: int, destination_rank: int) -> Any:
        try:
            return self.edge_groups[(source_rank, destination_rank)]
        except KeyError as exc:
            raise RuntimeError(f"missing process group for pipeline edge {source_rank}->{destination_rank}") from exc

    def _states_for(self, plan: _PipelineMicrobatch) -> tuple[Any, ...]:
        states: list[Any] = []
        for request_id in plan.request_ids:
            state = self.state_cache.get(request_id)
            if state is None:
                raise RuntimeError(f"pipeline token references unknown request state {request_id!r}")
            states.append(state)
        return tuple(states)

    def _is_fully_cancelled(self, plan: _PipelineMicrobatch) -> bool:
        return all(request_id in self._cancelled_request_ids for request_id in plan.request_ids)

    def _active_request_ids(self, plan: _PipelineMicrobatch) -> tuple[str, ...]:
        return tuple(request_id for request_id in plan.request_ids if request_id not in self._cancelled_request_ids)

    def _feedback_token_id(self, plan: _PipelineMicrobatch) -> int:
        branch = self._BRANCH_NEGATIVE if plan.do_true_cfg else self._BRANCH_POSITIVE
        return plan.token_id_by_branch[branch]

    def _is_feedback_token(self, plan: _PipelineMicrobatch, token_id: int) -> bool:
        return token_id == self._feedback_token_id(plan)

    def _discard_cancelled_states(self, plan: _PipelineMicrobatch) -> None:
        for request_id in plan.request_ids:
            if request_id in self._cancelled_request_ids:
                self.state_cache.pop(request_id, None)

    @staticmethod
    def _make_input_batch(states: Sequence[Any]) -> Any:
        from vllm_omni.diffusion.worker.input_batch import InputBatch

        return InputBatch.make_batch(states)

    @staticmethod
    def _state_row_count(state: Any) -> int:
        latents = getattr(state, "latents", None)
        if latents is None:
            raise RuntimeError(f"pipeline request state {state.request_id!r} has no latents")
        return int(latents.shape[0])

    @staticmethod
    def _step_index(states: Sequence[Any]) -> int:
        step_idx = int(states[0].step_index)
        if any(int(state.step_index) != step_idx for state in states[1:]):
            raise ValueError("one pipeline microbatch must have one step index")
        return step_idx

    @staticmethod
    def _set_state_step(states: Sequence[Any], step_idx: int) -> None:
        for state in states:
            state.step_index = step_idx

    @staticmethod
    def _advance_nonfirst_state(states: Sequence[Any], step_idx: int) -> None:
        for state in states:
            state.step_index = step_idx + 1

    @staticmethod
    def _extract_tensor(value: Any, operation: str) -> Any:
        if isinstance(value, tuple):
            if len(value) != 1:
                raise TypeError(f"{operation} returned {len(value)} tensors; interleaved PP currently needs one tensor")
            value = value[0]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{operation} must return one tensor, got {type(value).__name__}")
        return value

    @staticmethod
    def _validate_tensor(value: Any, shape: tuple[int, ...], dtype: Any, label: str) -> None:
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"pipeline {label} payload expected shape={shape}, dtype={dtype}; "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}"
            )

    def _validate_received_header(self, header: PipelineTransportHeader, spec: PipelineTensorSpec) -> None:
        token = self._tokens.get(header.token_id)
        plan = self._microbatches.get(header.token_id)
        if token is None or plan is None:
            raise RuntimeError(f"received unknown pipeline token {header.token_id}")
        if plan.spec != spec:
            raise RuntimeError("received token on an edge with the wrong tensor spec")
        if header.step_idx != token.step_idx or header.cfg_branch != self._branch_code(token.cfg_branch):
            raise RuntimeError("received pipeline header does not match registered token metadata")
        if (header.flags & ~PipelineTransportHeader.CANCELLED_FLAG) != self._model_phase_flags(token.model_phase):
            raise RuntimeError("received pipeline header does not match registered model phase")

    def _header_for(
        self,
        token_id: int,
        token: PipelineToken,
        *,
        cancelled: bool = False,
    ) -> PipelineTransportHeader:
        return PipelineTransportHeader(
            token_id=token_id,
            step_idx=token.step_idx,
            cfg_branch=self._branch_code(token.cfg_branch),
            flags=(
                self._model_phase_flags(token.model_phase)
                | (PipelineTransportHeader.CANCELLED_FLAG if cancelled else 0)
            ),
        )

    def _register_model_phase(self, model_phase: str) -> None:
        if model_phase in self._model_phase_ids:
            return
        phase_id = len(self._model_phase_ids)
        max_phase_id = PipelineTransportHeader.INT64_MAX >> self._MODEL_PHASE_SHIFT
        if phase_id > max_phase_id:
            raise ValueError("too many pipeline model phases for the fixed transport header")
        self._model_phase_ids[model_phase] = phase_id

    def _model_phase_flags(self, model_phase: str) -> int:
        try:
            return self._model_phase_ids[model_phase] << self._MODEL_PHASE_SHIFT
        except KeyError as exc:
            raise RuntimeError(f"pipeline token references unknown model phase {model_phase!r}") from exc

    def _allocate_token_id(self) -> int:
        token_id = self._next_token_id
        self._next_token_id += 1
        return token_id

    @classmethod
    def _branch_name(cls, branch: int) -> str:
        if branch == cls._BRANCH_POSITIVE:
            return "positive"
        if branch == cls._BRANCH_NEGATIVE:
            return "negative"
        raise ValueError(f"unsupported pipeline branch {branch}")

    @classmethod
    def _branch_code(cls, branch: str) -> int:
        if branch == "positive":
            return cls._BRANCH_POSITIVE
        if branch == "negative":
            return cls._BRANCH_NEGATIVE
        raise ValueError(f"unsupported pipeline branch {branch!r}")


class PipelineDeadlockError(RuntimeError):
    """Raised when pending work cannot make progress in the clock model."""


@dataclass(frozen=True, slots=True)
class PipelineClockRecord:
    """Observable result of one global PP clock."""

    clock: int
    active_stages: tuple[int, ...]
    completed_token_ids: tuple[str, ...]
    edge_credits: tuple[int, ...]


class _PipelineSlot:
    """A slot whose transition history makes unsafe reuse observable in tests."""

    def __init__(self, slot_id: int) -> None:
        self.slot_id = slot_id
        self.state = PipelineSlotState.FREE
        self.token: PipelineToken | None = None
        self.send_sequence: int | None = None
        self.history = [self.state]

    def _transition(self, expected: PipelineSlotState, new: PipelineSlotState) -> None:
        if self.state is not expected:
            raise RuntimeError(f"slot {self.slot_id} transition {self.state.name}->{new.name} requires {expected.name}")
        self.state = new
        self.history.append(new)

    def post_receive(self) -> None:
        self._transition(PipelineSlotState.FREE, PipelineSlotState.RECV_POSTED)

    def begin_send(self, token: PipelineToken, send_sequence: int) -> None:
        self._transition(PipelineSlotState.RECV_POSTED, PipelineSlotState.SEND_PENDING)
        self.token = token.for_slot(self.slot_id)
        self.send_sequence = send_sequence

    def complete_send(self) -> None:
        self._transition(PipelineSlotState.SEND_PENDING, PipelineSlotState.READY)

    def begin_compute(self) -> PipelineToken:
        self._transition(PipelineSlotState.READY, PipelineSlotState.COMPUTING)
        assert self.token is not None
        return self.token

    def release_after_compute(self) -> None:
        self._transition(PipelineSlotState.COMPUTING, PipelineSlotState.FREE)
        self.token = None
        self.send_sequence = None


class _PipelineEdge:
    """A bounded ring of receiver-owned slots between adjacent PP stages."""

    def __init__(self, slot_count: int) -> None:
        self.slots = [_PipelineSlot(slot_id) for slot_id in range(slot_count)]
        self._next_send_sequence = 0
        self._ready_slot_ids: deque[int] = deque()
        self.max_occupied = 0

    @property
    def credits(self) -> int:
        return sum(slot.state is PipelineSlotState.RECV_POSTED for slot in self.slots)

    @property
    def has_ready_token(self) -> bool:
        return any(slot.state is PipelineSlotState.READY for slot in self.slots)

    def peek_ready_token(self) -> PipelineToken | None:
        slot = self._peek_ready_slot()
        return slot.token if slot is not None else None

    @property
    def has_work(self) -> bool:
        return any(slot.token is not None for slot in self.slots)

    def complete_sends(self) -> None:
        pending_slots = sorted(
            (slot for slot in self.slots if slot.state is PipelineSlotState.SEND_PENDING),
            key=lambda slot: slot.send_sequence,
        )
        for slot in pending_slots:
            slot.complete_send()
            self._ready_slot_ids.append(slot.slot_id)

    def post_receives(self) -> None:
        for slot in self.slots:
            if slot.state is PipelineSlotState.FREE:
                slot.post_receive()

    def can_accept_after_consume(self, consumer_will_run: bool) -> bool:
        """Whether an upstream result can occupy this edge at clock end.

        A receiver that is consuming a READY token in this same clock returns
        that slot before the upstream send is posted.  This models the required
        completion ordering without allowing the sender to overwrite an input
        still used by the downstream local forward.
        """
        return self.credits > 0 or (consumer_will_run and self.has_ready_token)

    def begin_compute(self) -> PipelineToken:
        slot = self._peek_ready_slot()
        if slot is None:
            raise RuntimeError("attempted to consume an edge with no READY token")
        self._ready_slot_ids.popleft()
        return slot.begin_compute()

    def release_after_compute(self, token_id: str) -> None:
        for slot in self.slots:
            if slot.state is PipelineSlotState.COMPUTING and slot.token is not None:
                if slot.token.token_id != token_id:
                    continue
                slot.release_after_compute()
                return
        raise RuntimeError(f"attempted to release missing COMPUTING token {token_id!r}")

    def send(self, token: PipelineToken) -> None:
        for slot in self.slots:
            if slot.state is PipelineSlotState.RECV_POSTED:
                slot.begin_send(token, self._next_send_sequence)
                self._next_send_sequence += 1
                self.max_occupied = max(self.max_occupied, self._occupied_count())
                return
        raise RuntimeError("attempted to send without receiver credit")

    def state_histories(self) -> tuple[tuple[PipelineSlotState, ...], ...]:
        return tuple(tuple(slot.history) for slot in self.slots)

    def _occupied_count(self) -> int:
        return sum(slot.token is not None for slot in self.slots)

    def _peek_ready_slot(self) -> _PipelineSlot | None:
        if not self._ready_slot_ids:
            return None
        slot = self.slots[self._ready_slot_ids[0]]
        if slot.state is not PipelineSlotState.READY:
            raise RuntimeError("READY queue references a slot that is not READY")
        return slot

    def assert_invariants(self) -> None:
        ready_slot_ids = tuple(slot.slot_id for slot in self.slots if slot.state is PipelineSlotState.READY)
        if len(self._ready_slot_ids) != len(set(self._ready_slot_ids)):
            raise RuntimeError("READY queue contains a duplicate slot")
        if set(self._ready_slot_ids) != set(ready_slot_ids):
            raise RuntimeError("READY queue and slot states disagree")
        for slot in self.slots:
            if slot.state is PipelineSlotState.SEND_PENDING and slot.send_sequence is None:
                raise RuntimeError("SEND_PENDING slot has no send sequence")


@dataclass(frozen=True, slots=True)
class _StagePlan:
    stage: int
    token: PipelineToken


class PipelineClockSimulator:
    """Offline reference implementation of a retained-state PP clock.

    The simulator admits one compatibility cohort at a time.  New tokens may
    be submitted between clocks; stage 0 injects them only when the first PP
    edge has credit.  The model is intentionally synchronous at clock
    boundaries, matching the proposed first ``execute_pipeline_tick()`` RPC.
    """

    def __init__(self, num_stages: int, slots_per_edge: int = 2) -> None:
        if num_stages < 1:
            raise ValueError("num_stages must be at least one")
        if slots_per_edge < 1:
            raise ValueError("slots_per_edge must be at least one")

        self.num_stages = num_stages
        self._edges = [_PipelineEdge(slots_per_edge) for _ in range(num_stages - 1)]
        self._waiting: deque[PipelineToken] = deque()
        self._admitted_token_ids: set[str] = set()
        self._completed: list[PipelineToken] = []
        self._active_compatibility_key: Hashable | None = None
        self._stage_ready = [True] * num_stages
        self._clock = 0
        self.records: list[PipelineClockRecord] = []

    @property
    def completed_tokens(self) -> tuple[PipelineToken, ...]:
        return tuple(self._completed)

    @property
    def is_idle(self) -> bool:
        return not self._waiting and not any(edge.has_work for edge in self._edges)

    @property
    def edge_credits(self) -> tuple[int, ...]:
        return tuple(edge.credits for edge in self._edges)

    @property
    def edge_max_occupancy(self) -> tuple[int, ...]:
        return tuple(edge.max_occupied for edge in self._edges)

    @property
    def slot_state_histories(self) -> tuple[tuple[tuple[PipelineSlotState, ...], ...], ...]:
        return tuple(edge.state_histories() for edge in self._edges)

    @property
    def stage_readiness(self) -> tuple[bool, ...]:
        return tuple(self._stage_ready)

    def set_stage_ready(self, stage: int, ready: bool) -> None:
        """Inject stage readiness for deterministic backpressure simulations."""
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"stage {stage} is outside [0, {self.num_stages})")
        if not isinstance(ready, bool):
            raise TypeError("stage readiness must be a bool")
        self._stage_ready[stage] = ready

    def submit(self, token: PipelineToken) -> None:
        """Queue a token for stage-0 admission, rejecting duplicates and mixed shapes."""
        if token.slot_id is not None:
            raise ValueError("stage-0 PipelineToken must not have an assigned slot_id")
        if token.token_id in self._admitted_token_ids:
            raise ValueError(f"PipelineToken {token.token_id!r} was already admitted")

        if self.is_idle:
            self._active_compatibility_key = token.compatibility_key
        elif token.compatibility_key != self._active_compatibility_key:
            raise ValueError("only one homogeneous compatibility cohort may be active")

        self._admitted_token_ids.add(token.token_id)
        self._waiting.append(token)
        self._assert_invariants()

    def progress_one_clock(self) -> PipelineClockRecord:
        """Advance all stages by one microbatch, subject to receiver credits."""
        for edge in self._edges:
            edge.complete_sends()
            edge.post_receives()

        plans: list[_StagePlan | None] = [None] * self.num_stages
        for stage in range(self.num_stages - 1, -1, -1):
            if not self._stage_ready[stage]:
                continue
            if stage == 0:
                token = self._waiting[0] if self._waiting else None
            else:
                # Planning must not mutate the slot.  The actual receive is
                # consumed only after every stage has found a viable plan.
                token = self._edges[stage - 1].peek_ready_token()

            if token is None:
                continue
            if stage == self.num_stages - 1:
                plans[stage] = _StagePlan(stage, token)
                continue

            downstream_will_consume = plans[stage + 1] is not None
            if self._edges[stage].can_accept_after_consume(downstream_will_consume):
                plans[stage] = _StagePlan(stage, token)

        active_plans = [plan for plan in plans if plan is not None]
        for plan in active_plans:
            if plan.stage == 0:
                admitted = self._waiting.popleft()
                if admitted.token_id != plan.token.token_id:
                    raise RuntimeError("stage 0 admission order changed during a clock")
            else:
                consumed = self._edges[plan.stage - 1].begin_compute()
                if consumed.token_id != plan.token.token_id:
                    raise RuntimeError("edge READY token changed during a clock")

        # A receiver owns its input until its local forward completes.  Only
        # after every local forward has completed may upstream stages reuse the
        # released slots and consume the returned credits.
        for plan in active_plans:
            if plan.stage > 0:
                self._edges[plan.stage - 1].release_after_compute(plan.token.token_id)
        for edge in self._edges:
            edge.post_receives()

        completed_token_ids: list[str] = []
        for plan in active_plans:
            if plan.stage == self.num_stages - 1:
                self._completed.append(plan.token)
                completed_token_ids.append(plan.token.token_id)
            else:
                self._edges[plan.stage].send(plan.token)

        record = PipelineClockRecord(
            clock=self._clock,
            active_stages=tuple(plan.stage for plan in active_plans),
            completed_token_ids=tuple(completed_token_ids),
            edge_credits=self.edge_credits,
        )
        self._clock += 1
        self.records.append(record)
        self._assert_invariants()
        return record

    def run_until_idle(self, max_clocks: int) -> tuple[PipelineClockRecord, ...]:
        """Drain all queued tokens, failing deterministically on a stalled lane."""
        if max_clocks < 1:
            raise ValueError("max_clocks must be positive")

        start = len(self.records)
        for _ in range(max_clocks):
            if self.is_idle:
                return tuple(self.records[start:])
            record = self.progress_one_clock()
            if not record.active_stages and not self.is_idle:
                raise PipelineDeadlockError("pipeline has pending tokens but no stage can progress")

        if not self.is_idle:
            raise PipelineDeadlockError(f"pipeline did not drain after {max_clocks} clocks")
        return tuple(self.records[start:])

    def _assert_invariants(self) -> None:
        resident_ids = [token.token_id for token in self._waiting]
        for edge in self._edges:
            edge.assert_invariants()
            for slot in edge.slots:
                if slot.state in (PipelineSlotState.FREE, PipelineSlotState.RECV_POSTED):
                    assert slot.token is None
                else:
                    assert slot.token is not None
                    resident_ids.append(slot.token.token_id)

        completed_ids = [token.token_id for token in self._completed]
        all_ids = resident_ids + completed_ids
        if len(all_ids) != len(set(all_ids)):
            raise RuntimeError("PipelineToken was duplicated across queue, slots, or completion")
        if set(all_ids) != self._admitted_token_ids:
            raise RuntimeError("PipelineToken was lost from queue, slots, or completion")
