# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.diffusion_engine import (
    DiffusionEngine,
    _QueuedPipelineBatchPhase,
)
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.pipeline_state import PipelineEvent, PipelineEventType, PipelineTask
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _scheduler_output(request_id: str = "req-a") -> DiffusionSchedulerOutput:
    request = SimpleNamespace(
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(step_index=0, num_inference_steps=2),
    )
    return DiffusionSchedulerOutput(
        step_id=7,
        scheduled_new_reqs=[NewRequestData(request_id=request_id, req=request)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _engine(mocker, scheduler_output: DiffusionSchedulerOutput) -> DiffusionEngine:
    engine = object.__new__(DiffusionEngine)
    request = scheduler_output.scheduled_new_reqs[0].req
    engine.scheduler = SimpleNamespace(
        get_request_state=mocker.Mock(return_value=SimpleNamespace(req=request)),
    )
    engine.executor = mocker.Mock()
    engine.executor.pipeline_stage_physical_ranks.return_value = {0: 0, 1: 1}
    engine._queued_pipeline_batches = {}
    engine._queued_pipeline_epoch = 3
    return engine


def test_queued_submission_records_ownership_before_control_dispatch(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    calls: list[str] = []

    def observe_prepare(_scheduler_output):
        assert len(engine._queued_pipeline_batches) == 1
        calls.append("prepare")

    engine.executor.prepare_pipeline_requests.side_effect = observe_prepare
    engine.executor.submit_pipeline_batch.side_effect = lambda *_args: calls.append("submit")
    engine.executor.authorize_pipeline_batch.side_effect = lambda *_args: calls.append("authorize")

    batch = engine._submit_queued_pipeline_batch(scheduler_output)

    assert calls == ["prepare", "submit", "authorize"]
    assert batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED
    assert batch.task.batch_id == "pp-3-7"
    assert batch.task.request_ids == ("req-a",)
    assert batch.task.step_index == 0
    assert engine._queued_pipeline_batches[batch.task.batch_id] is batch
    assert set(batch.stage_specs) == {0, 1}
    engine.executor.submit_pipeline_batch.assert_called_once_with(batch.task, batch.stage_specs)
    engine.executor.authorize_pipeline_batch.assert_called_once_with({0: 0, 1: 1}, batch.task.batch_id)


def test_queued_reservation_rejects_second_retained_batch(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine._submit_queued_pipeline_batch(scheduler_output)

    with pytest.raises(RuntimeError, match="only one retained pipeline batch"):
        engine._reserve_queued_pipeline_batch(_scheduler_output("req-b"))


def test_queued_progress_accepts_only_matching_first_stage_completion(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.reset_mock()
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(
            event_type=PipelineEventType.STEP_COMPLETED,
            task=batch.task,
            pp_stage_id=0,
            physical_rank=0,
        )
    ]

    assert engine._progress_queued_pipeline_batch(batch)
    assert batch.phase is _QueuedPipelineBatchPhase.STEP_COMPLETED
    engine.executor.progress_pipeline.assert_called_once_with()
    engine.executor.poll_pipeline_events.assert_called_once_with()


def test_queued_progress_rejects_unknown_task_event(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    other_task = PipelineTask(
        batch_id="other-batch",
        request_ids=("req-b",),
        step_index=0,
        epoch=99,
    )
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(
            event_type=PipelineEventType.STEP_COMPLETED,
            task=other_task,
            pp_stage_id=0,
            physical_rank=0,
        )
    ]

    with pytest.raises(RuntimeError, match="unknown queued pipeline task"):
        engine._progress_queued_pipeline_batch(batch)

    assert batch.phase is _QueuedPipelineBatchPhase.FAILED


def test_queued_progress_rejects_completion_from_wrong_physical_rank(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(
            event_type=PipelineEventType.STEP_COMPLETED,
            task=batch.task,
            pp_stage_id=0,
            physical_rank=1,
        )
    ]

    with pytest.raises(RuntimeError, match="physical rank"):
        engine._progress_queued_pipeline_batch(batch)

    assert batch.phase is _QueuedPipelineBatchPhase.FAILED


def test_queued_completion_uses_nonzero_physical_topology(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.executor.pipeline_stage_physical_ranks.return_value = {0: 2, 1: 3}
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(
            event_type=PipelineEventType.STEP_COMPLETED,
            task=batch.task,
            pp_stage_id=0,
            physical_rank=2,
        )
    ]

    assert engine._progress_queued_pipeline_batch(batch)
    assert batch.stage_specs[0].is_first
    assert batch.stage_physical_ranks == {0: 2, 1: 3}
    engine.executor.submit_pipeline_batch.assert_called_once_with(batch.task, batch.stage_specs)
    engine.executor.authorize_pipeline_batch.assert_called_once_with({0: 0, 1: 1}, batch.task.batch_id)
    assert batch.phase is _QueuedPipelineBatchPhase.STEP_COMPLETED
