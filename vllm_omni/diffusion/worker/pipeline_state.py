# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small ownership records for queued diffusion pipeline execution.

These records intentionally contain no scheduling policy or model calls.  The
Worker owns stage/task lifecycle; the ModelRunner owns tensor-bearing contexts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PipelineTaskStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineStageSpec:
    """Immutable logical-stage placement used by one Worker."""

    pp_stage_id: int
    world_size: int
    is_first: bool
    is_last: bool

    def __post_init__(self) -> None:
        if self.pp_stage_id < 0 or self.pp_stage_id >= self.world_size:
            raise ValueError("pp_stage_id must be within the pipeline world")
        if self.world_size < 2:
            raise ValueError("queued pipeline execution requires at least two stages")
        if self.is_first and self.is_last:
            raise ValueError("a queued stage cannot be both first and last")


@dataclass(frozen=True)
class PipelineTask:
    """One conditional denoise task in the v1 queued contract."""

    batch_id: str
    request_ids: tuple[str, ...]
    step_index: int
    epoch: int
    branch: str = "conditional"

    def __post_init__(self) -> None:
        if not self.batch_id or not self.request_ids:
            raise ValueError("pipeline tasks require a batch id and request ids")
        if self.step_index < 0 or self.epoch < 0:
            raise ValueError("step_index and epoch must be non-negative")
        if self.branch != "conditional":
            raise ValueError("M2 supports only the conditional pipeline branch")


@dataclass
class PipelineBatchContext:
    """ModelRunner-owned context reference for one stage/batch execution."""

    task: PipelineTask
    stage_spec: PipelineStageSpec
    request_state_ids: tuple[str, ...]
    states: tuple[Any, ...]
    input_batch: Any
    tensors: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    status: PipelineTaskStatus = PipelineTaskStatus.PENDING

    @property
    def pp_stage_id(self) -> int:
        return self.stage_spec.pp_stage_id

    def __post_init__(self) -> None:
        if self.pp_stage_id < 0:
            raise ValueError("pp_stage_id must be non-negative")
        if not self.request_state_ids:
            raise ValueError("pipeline context requires request state ids")
        if self.request_state_ids != tuple(state.request_id for state in self.states):
            raise ValueError("pipeline context request ids must match its states")


@dataclass
class PipelineStageState:
    """Worker-owned FIFO and active-task bookkeeping for one logical stage."""

    spec: PipelineStageSpec
    pending_tasks: deque[PipelineTask] = field(default_factory=deque)
    active_task: PipelineTask | None = None
    completed_batches: set[str] = field(default_factory=set)

    def enqueue(self, task: PipelineTask) -> None:
        if task.batch_id in self.completed_batches:
            raise ValueError(f"batch {task.batch_id!r} has already completed")
        if self.active_task is not None and task.batch_id == self.active_task.batch_id:
            raise ValueError(f"batch {task.batch_id!r} is already active")
        if any(item.batch_id == task.batch_id for item in self.pending_tasks):
            raise ValueError(f"batch {task.batch_id!r} is already pending")
        self.pending_tasks.append(task)

    def start_next(self) -> PipelineTask | None:
        if self.active_task is not None or not self.pending_tasks:
            return None
        self.active_task = self.pending_tasks.popleft()
        return self.active_task

    def complete_active(self) -> PipelineTask:
        if self.active_task is None:
            raise RuntimeError("cannot complete a stage without an active task")
        task = self.active_task
        self.active_task = None
        self.completed_batches.add(task.batch_id)
        return task

    def cancel(self, batch_id: str) -> bool:
        if self.active_task is not None and self.active_task.batch_id == batch_id:
            self.active_task = None
            return True
        retained = deque(task for task in self.pending_tasks if task.batch_id != batch_id)
        changed = len(retained) != len(self.pending_tasks)
        self.pending_tasks = retained
        return changed
