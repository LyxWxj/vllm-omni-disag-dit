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
    DiffusionRequestStatus,
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
    engine.executor.pipeline_stage_memory_budget_bytes.return_value = 1 << 30
    engine.od_config = SimpleNamespace(max_inflight_batches=2)
    engine._queued_pipeline_batches = {}
    engine._queued_reserved_bytes = 0
    engine._queued_stage_buffer_budget_bytes = None
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


def test_queued_reservation_keeps_distinct_request_ownership(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    first = engine._submit_queued_pipeline_batch(scheduler_output)

    second = engine._reserve_queued_pipeline_batch(_scheduler_output("req-b"))

    assert first.task.request_ids == ("req-a",)
    assert second.task.request_ids == ("req-b",)
    assert len(engine._queued_pipeline_batches) == 2


def test_queued_byte_reservation_defers_after_budget_is_consumed(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine._queued_stage_buffer_budget_bytes = engine._estimate_queued_request_bytes(_scheduler_output("req-a"))

    first = engine._reserve_queued_pipeline_batch(_scheduler_output("req-a"))

    with pytest.raises(RuntimeError, match="stage buffer capacity is exhausted"):
        engine._reserve_queued_pipeline_batch(_scheduler_output("req-b"))

    assert engine._queued_reserved_bytes == first.reserved_bytes
    assert len(engine._queued_pipeline_batches) == 1


def test_queued_oversize_request_is_rejected_without_retry_loop(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine._queued_stage_buffer_budget_bytes = 1
    engine.scheduler.finish_requests = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    scheduler_output = _scheduler_output("req-too-large")

    with pytest.raises(RuntimeError, match="exceeds stage buffer budget"):
        engine._reserve_queued_pipeline_batch(scheduler_output)

    engine._reject_queued_admission(scheduler_output, RuntimeError("request exceeds stage buffer budget"))

    engine.scheduler.finish_requests.assert_called_once_with(["req-too-large"], DiffusionRequestStatus.FINISHED_ERROR)
    engine._emit_finished_outputs.assert_called_once()
    assert engine._queued_pipeline_batches == {}


def test_queued_failure_targets_matching_descriptor_batch(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second_output = _scheduler_output("req-b")
    second = engine._submit_queued_pipeline_batch(second_output)
    engine._cancel_queued_pipeline_batch = mocker.Mock()
    engine._retire_queued_pipeline_batch = mocker.Mock()
    engine._finish_failed_queued_batch = mocker.Mock()

    engine._handle_queued_iteration_failure(second_output, RuntimeError("second failed"))

    assert first.failure is None
    assert second.failure is not None
    engine._cancel_queued_pipeline_batch.assert_called_once_with(second)
    engine._retire_queued_pipeline_batch.assert_called_once_with(second)
    engine._finish_failed_queued_batch.assert_called_once_with(second)


def test_failed_queued_batch_retries_cancellation_before_retirement(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.failure = RuntimeError("progress failed")
    batch.phase = _QueuedPipelineBatchPhase.FAILED
    cancel = mocker.patch.object(engine, "_cancel_queued_pipeline_batch")
    retire = mocker.patch.object(engine, "_retire_queued_pipeline_batch")
    finish = mocker.patch.object(engine, "_finish_failed_queued_batch")

    engine._run_queued_pipeline_iteration(scheduler_output)

    cancel.assert_called_once_with(batch)
    retire.assert_called_once_with(batch)
    finish.assert_called_once_with(batch)


def test_unhandled_retained_batch_failure_uses_descriptor_cleanup(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second_output = _scheduler_output("req-b")
    second = engine._submit_queued_pipeline_batch(second_output)
    advance = mocker.patch.object(engine, "_advance_queued_pipeline_batch", side_effect=RuntimeError("progress failed"))
    cleanup = mocker.patch.object(engine, "_handle_queued_iteration_failure")

    engine._advance_unhandled_queued_batches({"req-a"})

    advance.assert_called_once_with(second)
    cleanup.assert_called_once_with(second_output, advance.side_effect)


def test_unhandled_retained_batches_progress_independently(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"))
    first.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    second.phase = _QueuedPipelineBatchPhase.AUTHORIZED
    advance = mocker.patch.object(engine, "_advance_queued_pipeline_batch", return_value=None)

    engine._advance_unhandled_queued_batches({"req-a"})

    advance.assert_called_once_with(second)


def test_cleanup_retry_preserves_failure_and_skips_repeat_cancellation(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    output = _scheduler_output("req-a")
    batch = engine._submit_queued_pipeline_batch(output)
    original = RuntimeError("original progress failure")
    batch.failure = original
    batch.cancelled = True
    cancel = mocker.patch.object(engine, "_cancel_queued_pipeline_batch")
    mocker.patch.object(engine, "_retire_queued_pipeline_batch", side_effect=RuntimeError("release retry failed"))

    engine._handle_queued_iteration_failure(output, RuntimeError("cleanup failure"))

    assert batch.failure is original
    cancel.assert_not_called()


def test_abort_retains_batch_until_cancel_and_release_complete(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.scheduler.finish_requests = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    batch = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    acknowledgements = [
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 1, 1),
    ]
    released = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cancel_pipeline_requests.side_effect = [RuntimeError("cancel timeout"), acknowledgements]
    engine.executor.release_pipeline_batch.return_value = released

    engine._abort_requests("req-a")
    assert batch.abort_requested
    assert batch.task.batch_id in engine._queued_pipeline_batches

    engine._advance_unhandled_queued_batches(set())

    assert engine._queued_pipeline_batches == {}
    engine._emit_finished_outputs.assert_called_once_with({"req-a"}, None)


@pytest.mark.parametrize("finalizing", [False, True])
def test_abort_overrides_failure_or_finalizing_success(mocker, finalizing: bool) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.scheduler.finish_requests = mocker.Mock()
    engine.scheduler.complete_pipeline_request = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    batch = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    if finalizing:
        batch.phase = _QueuedPipelineBatchPhase.FINALIZING
        batch.finalizing_request_ids = frozenset({"req-a"})
        engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]
    else:
        batch.phase = _QueuedPipelineBatchPhase.FAILED
        batch.failure = RuntimeError("execution failed")
    engine.executor.cancel_pipeline_requests.return_value = [
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 1, 1),
    ]
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]

    engine._abort_requests("req-a")

    engine.scheduler.complete_pipeline_request.assert_not_called()
    engine.scheduler.finish_requests.assert_called_once_with("req-a", DiffusionRequestStatus.FINISHED_ABORTED)
    engine._emit_finished_outputs.assert_called_once_with({"req-a"}, None)
    assert engine._queued_pipeline_batches == {}


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


def test_queued_step_commit_advances_scheduler_once_without_finishing_request(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step = mocker.Mock(return_value=set())
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMPLETED

    assert engine._commit_queued_pipeline_step(batch) == frozenset()
    engine.scheduler.commit_pipeline_step.assert_called_once_with(
        scheduler_output,
        {"req-a": 1},
    )
    assert batch.phase is _QueuedPipelineBatchPhase.STEP_COMMITTED

    with pytest.raises(RuntimeError, match="has not completed"):
        engine._commit_queued_pipeline_step(batch)


def test_queued_final_step_enters_finalizing_without_client_completion(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step = mocker.Mock(return_value={"req-a"})
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMPLETED

    assert engine._commit_queued_pipeline_step(batch) == frozenset({"req-a"})
    assert batch.finalizing_request_ids == frozenset({"req-a"})
    assert batch.phase is _QueuedPipelineBatchPhase.FINALIZING
