# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference clock and credit model for interleaved diffusion PP.

This module deliberately has no torch or distributed dependency.  It fixes the
ordering and buffer-lifetime contract before the device transport is added:
each call to :meth:`PipelineClockSimulator.progress_one_clock` advances at
most one microbatch at every stage, while edge credits bound the number of
tokens that may be in flight.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, replace
from enum import Enum, auto


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
        self.history = [self.state]

    def _transition(self, expected: PipelineSlotState, new: PipelineSlotState) -> None:
        if self.state is not expected:
            raise RuntimeError(
                f"slot {self.slot_id} transition {self.state.name}->{new.name} "
                f"requires {expected.name}"
            )
        self.state = new
        self.history.append(new)

    def post_receive(self) -> None:
        self._transition(PipelineSlotState.FREE, PipelineSlotState.RECV_POSTED)

    def begin_send(self, token: PipelineToken) -> None:
        self._transition(PipelineSlotState.RECV_POSTED, PipelineSlotState.SEND_PENDING)
        self.token = token.for_slot(self.slot_id)

    def complete_send(self) -> None:
        self._transition(PipelineSlotState.SEND_PENDING, PipelineSlotState.READY)

    def begin_compute(self) -> PipelineToken:
        self._transition(PipelineSlotState.READY, PipelineSlotState.COMPUTING)
        assert self.token is not None
        return self.token

    def release_after_compute(self) -> None:
        self._transition(PipelineSlotState.COMPUTING, PipelineSlotState.FREE)
        self.token = None


class _PipelineEdge:
    """A bounded ring of receiver-owned slots between adjacent PP stages."""

    def __init__(self, slot_count: int) -> None:
        self.slots = [_PipelineSlot(slot_id) for slot_id in range(slot_count)]
        self.max_occupied = 0

    @property
    def credits(self) -> int:
        return sum(slot.state is PipelineSlotState.RECV_POSTED for slot in self.slots)

    @property
    def has_ready_token(self) -> bool:
        return any(slot.state is PipelineSlotState.READY for slot in self.slots)

    def peek_ready_token(self) -> PipelineToken | None:
        for slot in self.slots:
            if slot.state is PipelineSlotState.READY:
                return slot.token
        return None

    @property
    def has_work(self) -> bool:
        return any(slot.token is not None for slot in self.slots)

    def complete_sends(self) -> None:
        for slot in self.slots:
            if slot.state is PipelineSlotState.SEND_PENDING:
                slot.complete_send()

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
        for slot in self.slots:
            if slot.state is PipelineSlotState.READY:
                return slot.begin_compute()
        raise RuntimeError("attempted to consume an edge with no READY token")

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
                slot.begin_send(token)
                self.max_occupied = max(self.max_occupied, self._occupied_count())
                return
        raise RuntimeError("attempted to send without receiver credit")

    def state_histories(self) -> tuple[tuple[PipelineSlotState, ...], ...]:
        return tuple(tuple(slot.history) for slot in self.slots)

    def _occupied_count(self) -> int:
        return sum(slot.token is not None for slot in self.slots)


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
