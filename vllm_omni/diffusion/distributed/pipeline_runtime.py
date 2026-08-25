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

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, replace
from enum import Enum, auto
from threading import Condition, Event, Thread
from typing import Any, ClassVar


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

    def __post_init__(self) -> None:
        for field_name, value in (
            ("token_id", self.token_id),
            ("step_idx", self.step_idx),
            ("cfg_branch", self.cfg_branch),
            ("flags", self.flags),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"PipelineTransportHeader.{field_name} must be a non-negative int")
        for field_name, value in (("slot_id", self.slot_id), ("send_sequence", self.send_sequence)):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"PipelineTransportHeader.{field_name} must be a non-negative int when assigned")

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
        self._work_wait_queue: deque[_TrackedWork] = deque()
        self._work_wait_condition = Condition()
        if self._requires_wait_thread:
            Thread(target=self._gloo_wait_loop, daemon=True).start()
        self.source_rank = source_rank
        self.destination_rank = destination_rank
        self.slots_per_edge = slots_per_edge
        self.tag_base = tag_base
        self.tensor_shape = tuple(tensor_shape)
        self.tensor_dtype = tensor_dtype
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
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
        slot_id = self._take_send_credit()
        slot = self._send_slots[slot_id]
        if slot.is_send_pending:
            raise RuntimeError(f"slot {slot_id} credit returned before its previous send completed")

        transport_header = header.for_slot(slot_id, self._next_send_sequence)
        self._next_send_sequence += 1
        transport_header.encode_into(slot.header_buffer)
        slot.payload_buffer.copy_(payload)
        slot.begin_send()
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
        for work in self._credit_works:
            self._wait_for_works(work)
        self._poll_destination()

    def _poll_source(self) -> None:
        for slot in self._send_slots:
            if slot.is_send_pending and self._works_complete(slot.header_work, slot.payload_work):
                self._wait_for_works(slot.header_work, slot.payload_work)
                slot.finish_send()

        for expected_slot_id, work in enumerate(self._credit_works):
            if not self._works_complete(work):
                continue
            self._wait_for_works(work)
            credit_slot_id = int(self._credit_buffers[expected_slot_id].item())
            if credit_slot_id != expected_slot_id:
                raise RuntimeError(
                    f"credit tag for slot {expected_slot_id} carried slot {credit_slot_id}; PP edge is desynchronized"
                )
            if credit_slot_id in self._credit_slot_id_set:
                raise RuntimeError(f"received duplicate credit for slot {credit_slot_id}")
            self._credit_slot_ids.append(credit_slot_id)
            self._credit_slot_id_set.add(credit_slot_id)
            if not self._shutdown_requested:
                self._post_credit_receive(expected_slot_id)

    def _poll_destination(self) -> None:
        for slot in self._receive_slots:
            if slot.state is not PipelineSlotState.RECV_POSTED or slot.header is not None:
                continue
            if not self._works_complete(slot.header_work, slot.payload_work):
                continue
            self._wait_for_works(slot.header_work, slot.payload_work)
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

    def _take_send_credit(self) -> int:
        for _ in range(len(self._credit_slot_ids)):
            slot_id = self._credit_slot_ids.popleft()
            self._credit_slot_id_set.remove(slot_id)
            if self._send_slots[slot_id].state is PipelineSlotState.FREE:
                return slot_id
            self._credit_slot_ids.append(slot_id)
            self._credit_slot_id_set.add(slot_id)
        raise RuntimeError("attempted to send without a downstream credit")

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
        with self._work_wait_condition:
            self._work_wait_queue.append(tracked_work)
            self._work_wait_condition.notify()
        return tracked_work

    def _gloo_wait_loop(self) -> None:
        """Serialize Gloo ``Work.wait`` calls for one edge.

        Gloo's CPU backend does not advance a nonblocking work from
        ``is_completed()`` alone, and concurrent waits on several P2P works in
        one process are not reliable on all supported builds. A single waiter
        preserves the nonblocking public ``poll`` API while keeping native
        progress serialized per edge.
        """
        while True:
            with self._work_wait_condition:
                while not self._work_wait_queue:
                    self._work_wait_condition.wait()
                tracked_work = self._work_wait_queue.popleft()
            try:
                tracked_work.work.wait()
            except BaseException as exc:  # pragma: no cover - backend failure path.
                tracked_work.error = exc
            finally:
                tracked_work.completed.set()

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
