# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.diffusion_engine import (
    DiffusionEngine,
    _QueuedAdmissionDeferredError,
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
    engine.od_config = SimpleNamespace(mode="queued", max_inflight_batches=2)
    engine._queued_pipeline_batches = {}
    engine._queued_pipeline_epoch = 3

    submit = DiffusionEngine._submit_queued_pipeline_batch

    def submit_with_admission_updates(output, *, authorize=True):
        batch = submit(engine, output, authorize=authorize)
        pending = [
            item
            for item in engine._queued_pipeline_batches.values()
            if item.phase is _QueuedPipelineBatchPhase.ADMISSION_PENDING
        ]
        for item in pending:
            item.stage_enqueued = True
            item.phase = _QueuedPipelineBatchPhase.AUTHORIZED
            item.admission_acknowledgements = {
                (event_type, stage_id, rank)
                for event_type in (PipelineEventType.ACCEPTED, PipelineEventType.AUTHORIZED)
                for stage_id, rank in item.stage_physical_ranks.items()
            }

        return batch

    engine._submit_queued_pipeline_batch = submit_with_admission_updates
    return engine


def test_autonomous_admission_waits_for_both_stage_acknowledgements(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.executor.submit_pipeline_admissions.side_effect = lambda *_args: [True, True]

    batch = DiffusionEngine._submit_queued_pipeline_batch(engine, scheduler_output)

    assert batch.phase is _QueuedPipelineBatchPhase.ADMISSION_PENDING
    assert batch.stage_enqueued is True
    assert engine._queued_denoise_batch_count() == 1
    events = [
        PipelineEvent(PipelineEventType.ACCEPTED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.ACCEPTED, batch.task, 1, 1),
        PipelineEvent(PipelineEventType.AUTHORIZED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.AUTHORIZED, batch.task, 1, 1),
    ]
    engine.executor.poll_pipeline_events.return_value = events

    grouped = engine._collect_queued_pipeline_events()

    assert set(grouped) == {batch.task.batch_id}
    assert batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED
    engine.executor.submit_pipeline_admissions.assert_called_once_with([(batch.task, batch.stage_specs)])


def test_autonomous_admission_batches_prepared_fifo(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"), authorize=False)
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"), authorize=False)

    engine._authorize_waiting_queued_batches()

    assert first.phase is _QueuedPipelineBatchPhase.ADMISSION_PENDING
    assert second.phase is _QueuedPipelineBatchPhase.ADMISSION_PENDING
    engine.executor.submit_pipeline_admissions.assert_called_once_with(
        [(first.task, first.stage_specs), (second.task, second.stage_specs)]
    )


def test_queued_reservation_keeps_distinct_request_ownership(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    first = engine._submit_queued_pipeline_batch(scheduler_output)

    second = engine._reserve_queued_pipeline_batch(_scheduler_output("req-b"))

    assert first.task.request_id == "req-a"
    assert second.task.request_id == "req-b"
    assert len(engine._queued_pipeline_batches) == 2


def test_finalizing_batch_releases_denoise_admission_capacity(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.od_config.max_inflight_batches = 1
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    first.phase = _QueuedPipelineBatchPhase.FINALIZING

    assert engine._queued_denoise_batch_count() == 0
    second = engine._reserve_queued_pipeline_batch(_scheduler_output("req-b"))

    assert second.task.request_id == "req-b"


def test_prepared_fifo_prevents_continuation_starvation(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.od_config.max_inflight_batches = 1
    engine.scheduler.max_num_running_reqs = 3

    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"))
    assert first.phase is _QueuedPipelineBatchPhase.AUTHORIZED
    assert second.phase is _QueuedPipelineBatchPhase.PREPARED

    first.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine._queued_pipeline_batches.pop(first.task.batch_id)
    engine.scheduler.get_request_state.return_value.req.sampling_params.step_index = 1
    continuation = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))

    assert second.phase is _QueuedPipelineBatchPhase.AUTHORIZED
    assert continuation.phase is _QueuedPipelineBatchPhase.PREPARED


def test_prepared_only_cancellation_cleans_request_without_stage_release(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.od_config.max_inflight_batches = 1
    engine.scheduler.max_num_running_reqs = 2
    engine.scheduler.finish_requests = mocker.Mock()
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"))
    assert first.phase is _QueuedPipelineBatchPhase.AUTHORIZED
    assert second.phase is _QueuedPipelineBatchPhase.PREPARED
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    engine._cancel_queued_pipeline_batch(second)
    engine._retire_queued_pipeline_batch(second)

    engine.executor.cancel_pipeline_requests.assert_not_called()
    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-b")
    assert second.task.batch_id not in engine._queued_pipeline_batches
    engine.scheduler.finish_requests.assert_called_once_with("req-b", DiffusionRequestStatus.FINISHED_ABORTED)


def test_prepared_only_failure_retires_before_finishing_request(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.od_config.max_inflight_batches = 1
    engine.scheduler.max_num_running_reqs = 2
    engine.scheduler.finish_requests = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"))
    second.failure = RuntimeError("preparation failed")
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    engine._handle_queued_iteration_failure(second.scheduler_output, second.failure)

    engine.executor.cancel_pipeline_requests.assert_not_called()
    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-b")
    assert second.task.batch_id not in engine._queued_pipeline_batches
    engine.scheduler.finish_requests.assert_called_once_with("req-b", DiffusionRequestStatus.FINISHED_ERROR)


def test_partial_preparation_failure_still_cleans_every_worker(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.scheduler.finish_requests = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    engine.executor.prepare_pipeline_requests.side_effect = RuntimeError("rank 1 preparation failed")
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    with pytest.raises(RuntimeError, match="rank 1 preparation failed"):
        engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    batch = next(iter(engine._queued_pipeline_batches.values()))
    engine._handle_queued_iteration_failure(batch.scheduler_output, batch.failure or RuntimeError("failed"))

    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
    engine.executor.cancel_pipeline_requests.assert_not_called()
    assert engine._queued_pipeline_batches == {}
    engine.scheduler.finish_requests.assert_called_once_with("req-a", DiffusionRequestStatus.FINISHED_ERROR)


def test_split_queued_scheduler_output_processes_cached_before_new(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    mixed = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[_scheduler_output("req-b").scheduled_new_reqs[0]],
        scheduled_cached_reqs=CachedRequestData(request_ids=["req-a"]),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )

    descriptors = engine._split_queued_scheduler_output(mixed)

    assert [tuple(descriptor.scheduled_request_ids) for descriptor in descriptors] == [("req-a",), ("req-b",)]


def test_queued_iteration_does_not_reuse_previous_request_step(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    engine.scheduler.get_request_state.return_value.req.sampling_params.step_index = 1

    current_step_output = _scheduler_output("req-a")
    current_step_output.step_id = 8
    engine._run_queued_pipeline_iteration(current_step_output, submit_only=True)

    batches = list(engine._queued_pipeline_batches.values())
    assert [batch.task.step_index for batch in batches] == [first.task.step_index, 1]
    assert engine._queued_pipeline_batch_for_scheduler_output(current_step_output) is batches[-1]
    assert engine.executor.prepare_pipeline_requests.call_count == 2


def test_failure_cleanup_finds_original_step_after_scheduler_advance(mocker) -> None:
    scheduler_output = _scheduler_output("req-a")
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.finish_requests = mocker.Mock()
    engine._emit_finished_outputs = mocker.Mock()
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.stage_enqueued = False
    batch.request_prepared = True
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    engine.scheduler.get_request_state.return_value.req.sampling_params.step_index = 1
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    engine._handle_queued_iteration_failure(scheduler_output, RuntimeError("decode failed"))

    assert batch.task.batch_id not in engine._queued_pipeline_batches
    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
    engine.scheduler.finish_requests.assert_called_once_with("req-a", DiffusionRequestStatus.FINISHED_ERROR)


def test_queued_admission_deferral_preserves_new_and_cached_identity(mocker) -> None:
    scheduler_output = _scheduler_output("req-a")
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.defer_request = mocker.Mock(return_value=True)
    engine.scheduler.preempt_request = mocker.Mock(return_value=True)

    engine._defer_queued_admission(scheduler_output)

    engine.scheduler.defer_request.assert_called_once_with("req-a")
    engine.scheduler.preempt_request.assert_not_called()

    cached_output = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=["req-b"]),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )
    engine.scheduler.defer_request.reset_mock()

    engine._defer_queued_admission(cached_output)

    engine.scheduler.defer_request.assert_not_called()
    engine.scheduler.preempt_request.assert_called_once_with("req-b")


def test_queued_admission_deferral_keeps_owned_cached_request_running(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.scheduler.preempt_request = mocker.Mock(return_value=True)
    engine._queued_pipeline_batches = {
        "owned": SimpleNamespace(task=SimpleNamespace(request_id="req-b")),
    }
    cached_output = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=["req-b"]),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )

    engine._defer_queued_admission(cached_output)

    engine.scheduler.preempt_request.assert_not_called()


def test_queued_failure_targets_matching_descriptor_batch(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second_output = _scheduler_output("req-b")
    second = engine._submit_queued_pipeline_batch(second_output)
    engine._cancel_queued_pipeline_batch = mocker.Mock()
    engine._retire_queued_pipeline_batch = mocker.Mock(
        side_effect=lambda batch: engine._queued_pipeline_batches.pop(batch.task.batch_id)
    )
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
    retire = mocker.patch.object(
        engine,
        "_retire_queued_pipeline_batch",
        side_effect=lambda candidate: engine._queued_pipeline_batches.pop(candidate.task.batch_id),
    )
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

    advance.assert_called_once_with(second, pipeline_events=None)
    cleanup.assert_called_once_with(second_output, advance.side_effect)


def test_unhandled_retained_batches_progress_independently(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second = engine._submit_queued_pipeline_batch(_scheduler_output("req-b"))
    first.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    second.phase = _QueuedPipelineBatchPhase.AUTHORIZED
    advance = mocker.patch.object(engine, "_advance_queued_pipeline_batch", return_value=None)

    engine._advance_unhandled_queued_batches({"req-a"})

    advance.assert_called_once_with(second, pipeline_events=None)


def test_finalizing_batch_does_not_sleep_while_other_requests_are_schedulable(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    batch = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalization_handle = "decode-handle"
    engine.scheduler.has_queued_admission_candidate = mocker.Mock(return_value=True)
    engine.executor.pipeline_updates_pending.return_value = False

    assert not engine._should_wait_for_queued_pipeline_update()

    engine.scheduler.has_queued_admission_candidate.return_value = False
    assert engine._should_wait_for_queued_pipeline_update()


def test_full_queued_capacity_waits_for_worker_update_before_rescheduling(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.od_config.max_inflight_batches = 1
    batch = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    batch.phase = _QueuedPipelineBatchPhase.AUTHORIZED
    engine.scheduler.has_queued_admission_candidate = mocker.Mock(return_value=True)
    engine.scheduler.has_queued_waiting_request = mocker.Mock(return_value=True)
    engine.executor.pipeline_updates_pending.return_value = False

    assert engine._should_wait_for_queued_pipeline_update()

    engine.executor.pipeline_updates_pending.return_value = True
    assert not engine._should_wait_for_queued_pipeline_update()

    batch.failure = RuntimeError("cleanup pending")
    engine.executor.pipeline_updates_pending.return_value = False
    assert not engine._should_wait_for_queued_pipeline_update()


def test_shared_progress_delivers_events_to_retained_batch(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    first_output = _scheduler_output("req-a")
    second_output = _scheduler_output("req-b")
    first = engine._submit_queued_pipeline_batch(first_output)
    second = engine._submit_queued_pipeline_batch(second_output)
    engine.executor.progress_pipeline.reset_mock()
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(PipelineEventType.STEP_COMPLETED, first.task, 0, 0),
        PipelineEvent(PipelineEventType.STEP_COMPLETED, second.task, 0, 0),
    ]

    events_by_batch = engine._collect_queued_pipeline_events()
    received: dict[str, list[PipelineEvent] | None] = {}

    def advance(batch, pipeline_events=None):
        received[batch.task.batch_id] = pipeline_events
        return None

    mocker.patch.object(engine, "_advance_queued_pipeline_batch", side_effect=advance)
    engine._run_queued_pipeline_iteration(
        first_output,
        pipeline_events=events_by_batch[first.task.batch_id],
    )
    engine._advance_unhandled_queued_batches({"req-a"}, events_by_batch)

    assert received == {
        first.task.batch_id: events_by_batch[first.task.batch_id],
        second.task.batch_id: events_by_batch[second.task.batch_id],
    }
    engine.executor.progress_pipeline.assert_called_once_with()
    engine.executor.poll_pipeline_events.assert_called_once_with()


def test_retained_authorized_batch_forces_progress_when_all_admission_deferred(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    retained_output = _scheduler_output("req-a")
    retained = engine._submit_queued_pipeline_batch(retained_output)
    engine.executor.progress_pipeline.reset_mock()
    event = PipelineEvent(PipelineEventType.STEP_COMPLETED, retained.task, 0, 0)
    engine.executor.poll_pipeline_events.return_value = [event]
    handled_request_ids = {"req-b"}

    assert engine._has_unhandled_authorized_queued_batch(handled_request_ids)
    events_by_batch = engine._collect_queued_pipeline_events()
    received: list[PipelineEvent] = []
    mocker.patch.object(
        engine,
        "_advance_queued_pipeline_batch",
        side_effect=lambda batch, pipeline_events=None: received.extend(pipeline_events or []),
    )

    engine._advance_unhandled_queued_batches(handled_request_ids, events_by_batch)

    assert received == [event]
    engine.executor.progress_pipeline.assert_called_once_with()
    engine.executor.poll_pipeline_events.assert_called_once_with()


def test_busy_loop_progresses_retained_batch_when_scheduler_snapshot_is_empty(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.od_config.mode = "queued"
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition()
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    retained = SimpleNamespace(
        phase=_QueuedPipelineBatchPhase.AUTHORIZED,
        failure=None,
        abort_requested=False,
        task=SimpleNamespace(request_id="req-a"),
    )
    engine._queued_pipeline_batches = {"retained": retained}
    empty = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=0,
        num_waiting_reqs=0,
    )
    engine.scheduler.has_requests = mocker.Mock(return_value=False)
    engine.scheduler.schedule = mocker.Mock(return_value=empty)
    engine._process_aborts_queue = mocker.Mock()
    engine._process_rpc_queue = mocker.Mock()
    engine._advance_unhandled_queued_batches = mocker.Mock()

    def collect_once():
        engine.stop_event.set()
        return {}

    engine._collect_queued_pipeline_events = mocker.Mock(side_effect=collect_once)

    engine._busy_loop()

    engine._collect_queued_pipeline_events.assert_called_once_with()
    engine._advance_unhandled_queued_batches.assert_called_once_with(set(), {})


def test_busy_loop_skips_scheduler_for_owned_queued_work(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.od_config.mode = "queued"
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition()
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    retained = SimpleNamespace(
        phase=_QueuedPipelineBatchPhase.AUTHORIZED,
        failure=None,
        abort_requested=False,
        task=SimpleNamespace(request_id="req-a"),
    )
    engine._queued_pipeline_batches = {"retained": retained}
    engine.scheduler.has_requests = mocker.Mock(return_value=True)
    engine.scheduler.has_queued_admission_candidate = mocker.Mock(return_value=False)
    engine.scheduler.num_waiting_requests = mocker.Mock(return_value=1)
    engine.scheduler.schedule = mocker.Mock(side_effect=AssertionError("owned work must not be rescheduled"))
    engine._process_aborts_queue = mocker.Mock()
    engine._process_rpc_queue = mocker.Mock()
    engine._advance_unhandled_queued_batches = mocker.Mock()

    def collect_once():
        engine.stop_event.set()
        return {}

    engine._collect_queued_pipeline_events = mocker.Mock(side_effect=collect_once)

    engine._busy_loop()

    engine.scheduler.schedule.assert_not_called()
    assert engine._scheduler_num_waiting_reqs == 1
    assert engine.scheduler.has_queued_admission_candidate.call_count == 2
    engine.scheduler.has_queued_admission_candidate.assert_has_calls(
        [
            mocker.call({"req-a"}, admission_capacity_available=True),
            mocker.call({"req-a"}, admission_capacity_available=True),
        ]
    )
    engine._collect_queued_pipeline_events.assert_called_once_with()
    engine._advance_unhandled_queued_batches.assert_called_once_with(set(), {})


def test_busy_loop_progresses_worker_update_between_admissions(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.od_config.mode = "queued"
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition()
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    first = _scheduler_output("req-a").scheduled_new_reqs[0]
    second = _scheduler_output("req-b").scheduled_new_reqs[0]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[first, second],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )
    engine.scheduler.has_requests = mocker.Mock(return_value=True)
    engine.scheduler.schedule = mocker.Mock(return_value=scheduler_output)
    engine._wait_for_admission_if_needed_locked = mocker.Mock()
    engine._process_aborts_queue = mocker.Mock()
    engine._process_rpc_queue = mocker.Mock()
    update_pending = [True]
    engine.executor.pipeline_updates_pending.side_effect = lambda: update_pending[0]
    order: list[str] = []

    def progress_pipeline():
        order.append("progress")
        update_pending[0] = False

    engine.executor.progress_pipeline.side_effect = progress_pipeline
    engine.executor.poll_pipeline_events.return_value = []

    def run_iteration(output, *, submit_only=False, pipeline_events=None):
        request_id = output.scheduled_request_ids[0]
        order.append(f"{'admit' if submit_only else 'advance'}:{request_id}")

    engine._run_queued_pipeline_iteration = mocker.Mock(side_effect=run_iteration)
    engine._advance_unhandled_queued_batches = mocker.Mock(side_effect=lambda *_args: engine.stop_event.set())

    engine._busy_loop()

    assert order.index("progress") < order.index("admit:req-b")
    assert order[:3] == ["admit:req-a", "progress", "admit:req-b"]
    engine.executor.poll_pipeline_events.assert_called_once_with()


def test_admission_progress_does_not_wait_for_future_worker_update(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    engine.executor.pipeline_updates_pending.return_value = False

    engine._progress_autonomous_updates_between_admissions()

    engine.executor.progress_pipeline.assert_not_called()


def test_admission_progress_drains_fast_followup_worker_update(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    update_pending = [True]
    engine.executor.pipeline_updates_pending.side_effect = lambda: update_pending[0]
    progress_calls = 0

    def progress_pipeline():
        nonlocal progress_calls
        progress_calls += 1
        if progress_calls == 2:
            update_pending[0] = False

    engine.executor.progress_pipeline.side_effect = progress_pipeline

    engine._progress_autonomous_updates_between_admissions()

    assert progress_calls == 2


def test_busy_loop_progresses_cached_retained_batch_after_deferred_new_tail(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    engine.od_config.mode = "queued"
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition()
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    retained = SimpleNamespace(
        phase=_QueuedPipelineBatchPhase.AUTHORIZED,
        failure=None,
        abort_requested=False,
        task=SimpleNamespace(request_id="req-a"),
    )
    engine._queued_pipeline_batches = {"retained": retained}
    mixed = DiffusionSchedulerOutput(
        step_id=8,
        scheduled_new_reqs=[_scheduler_output("req-b").scheduled_new_reqs[0]],
        scheduled_cached_reqs=CachedRequestData(request_ids=["req-a"]),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )
    engine.scheduler.has_requests = mocker.Mock(return_value=True)
    engine.scheduler.schedule = mocker.Mock(return_value=mixed)
    engine._wait_for_admission_if_needed_locked = mocker.Mock()
    engine._process_aborts_queue = mocker.Mock()
    engine._process_rpc_queue = mocker.Mock()
    engine._run_queued_pipeline_iteration = mocker.Mock(side_effect=_QueuedAdmissionDeferredError())
    engine._defer_queued_admission_tail = mocker.Mock()
    engine._advance_unhandled_queued_batches = mocker.Mock()

    def collect_once():
        engine.stop_event.set()
        return {}

    engine._collect_queued_pipeline_events = mocker.Mock(side_effect=collect_once)

    engine._busy_loop()

    engine._collect_queued_pipeline_events.assert_called_once_with()
    engine._advance_unhandled_queued_batches.assert_called_once_with(set(), {})


def test_malformed_shared_snapshot_does_not_repoll_retained_batch(mocker) -> None:
    engine = _engine(mocker, _scheduler_output("req-a"))
    admitted_output = _scheduler_output("req-a")
    retained_output = _scheduler_output("req-b")
    admitted = engine._submit_queued_pipeline_batch(admitted_output)
    retained = engine._submit_queued_pipeline_batch(retained_output)
    unknown_task = PipelineTask("unknown", "req-c", step_index=0, epoch=99)
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(PipelineEventType.STEP_COMPLETED, unknown_task, 0, 0)
    ]
    failure_cleanup = mocker.patch.object(engine, "_handle_queued_iteration_failure")
    retained_progress: list[tuple[object, list[PipelineEvent] | None]] = []
    mocker.patch.object(
        engine,
        "_advance_queued_pipeline_batch",
        side_effect=lambda batch, pipeline_events=None: retained_progress.append((batch, pipeline_events)),
    )

    with pytest.raises(RuntimeError, match="unknown queued pipeline task"):
        engine._collect_queued_pipeline_events()
    engine._handle_queued_progress_snapshot_failure(
        [admitted_output],
        {"req-a"},
        RuntimeError("malformed progress snapshot"),
    )

    failure_cleanup.assert_called_once_with(admitted_output, mocker.ANY)
    assert retained_progress == [(retained, [])]
    assert admitted.task.batch_id in engine._queued_pipeline_batches
    assert retained.task.batch_id in engine._queued_pipeline_batches
    engine.executor.progress_pipeline.assert_called_once_with()
    engine.executor.poll_pipeline_events.assert_called_once_with()


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
        batch.finalizing = True
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
        request_id="req-b",
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
    engine.executor.submit_pipeline_admissions.assert_called_once_with([(batch.task, batch.stage_specs)])
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
    assert batch.finalizing
    assert batch.phase is _QueuedPipelineBatchPhase.FINALIZING
