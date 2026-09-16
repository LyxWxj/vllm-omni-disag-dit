# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.distributed.pipeline_stage_connector import PipelineMessage, PipelineStageConnector
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineStageSpec,
    PipelineStageState,
    PipelineTask,
    PipelineTaskStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _task(batch_id: str = "batch-a") -> PipelineTask:
    return PipelineTask(batch_id=batch_id, request_ids=("req-a",), step_index=0, epoch=1)


def test_stage_state_preserves_fifo_and_one_active_task() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    first, second = _task(), _task("batch-b")

    state.enqueue(first)
    state.enqueue(second)
    state.authorize(first.batch_id)
    state.authorize(second.batch_id)
    assert state.start_next() == first
    assert state.start_next() is None
    assert state.complete_active() == first
    assert state.start_next() == second


def test_stage_state_rejects_reenqueue_after_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)
    state.start_next()
    state.complete_active()

    with pytest.raises(ValueError, match="already terminal"):
        state.enqueue(task)


def test_stage_state_rejects_duplicate_and_unknown_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True))
    task = _task()
    state.enqueue(task)
    with pytest.raises(ValueError, match="already pending"):
        state.enqueue(task)
    with pytest.raises(RuntimeError, match="active task"):
        PipelineStageState(state.spec).complete_active()


def test_stage_state_requires_authorization_without_bypassing_fifo_head() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    first, second = _task(), _task("batch-b")
    state.enqueue(first)
    state.enqueue(second)
    state.authorize(second.batch_id)

    assert state.start_next() is None
    state.authorize(first.batch_id)
    assert state.start_next() == first


def test_stage_state_cancel_and_retire_preserve_terminal_tombstone() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)

    assert state.cancel(task.batch_id)
    state.retire(task.batch_id)
    with pytest.raises(ValueError, match="already terminal"):
        state.enqueue(task)


def test_stage_state_cancellation_overrides_local_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)
    state.start_next()
    state.complete_active()

    assert state.cancel(task.batch_id)
    assert state.terminal_statuses[task.batch_id] is PipelineTaskStatus.CANCELLED
    assert task.batch_id not in state.completed_batches


def test_connector_enforces_credit_and_completion_before_release() -> None:
    connector = PipelineStageConnector(edge="0->1", max_slots=1)
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    with pytest.raises(RuntimeError, match="no send credit"):
        connector.enqueue_send(message)
    with pytest.raises(RuntimeError, match="before transport completion"):
        connector.release_send(ticket)
    connector.mark_send_complete(ticket)
    connector.release_send(ticket)
    assert connector.send_in_use == 0


def test_connector_rejects_non_conditional_messages() -> None:
    connector = PipelineStageConnector(edge="0->1")
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="negative", payload=None)
    with pytest.raises(ValueError, match="conditional"):
        connector.enqueue_send(message)
