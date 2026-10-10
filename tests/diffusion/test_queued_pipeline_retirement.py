# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from tests.diffusion.test_queued_pipeline_engine import _engine, _scheduler_output
from vllm_omni.diffusion.diffusion_engine import _QueuedPipelineBatchPhase
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.pipeline_state import PipelineEvent, PipelineEventType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_final_decode_completes_scheduler_only_after_successful_output(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    engine.executor.finalize_pipeline_batch.return_value = SimpleNamespace(
        get_request_output=lambda request_id: SimpleNamespace(
            result=SimpleNamespace(error=None),
        )
    )

    output = engine._finalize_queued_pipeline_batch(batch)

    assert output is engine.executor.finalize_pipeline_batch.return_value
    engine.scheduler.complete_pipeline_request.assert_not_called()
    assert batch.decoded_output is output


def test_async_final_decode_keeps_batch_finalizing_until_poll_completes(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    output = SimpleNamespace(get_request_output=lambda request_id: SimpleNamespace(result=SimpleNamespace(error=None)))
    engine.executor.finalize_pipeline_batch.return_value = "decode-handle"
    engine.executor.poll_pipeline_finalization.side_effect = [None, output]
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})

    assert engine._advance_queued_pipeline_batch(batch) is None
    assert batch.phase is _QueuedPipelineBatchPhase.FINALIZING
    assert batch.task.batch_id in engine._queued_pipeline_batches
    engine.scheduler.complete_pipeline_request.assert_not_called()

    assert engine._advance_queued_pipeline_batch(batch) is output
    assert engine._queued_pipeline_batches == {}
    engine.executor.finalize_pipeline_batch.assert_called_once_with(
        {0: 0, 1: 1}, batch.task.batch_id, batch.stage_physical_ranks[0]
    )
    assert engine.executor.poll_pipeline_finalization.call_count == 2


def test_final_decode_waits_until_step_transfers_retire(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    engine.executor.pipeline_batch_release_ready.side_effect = [False, True]
    engine.executor.finalize_pipeline_batch.return_value = "decode-handle"
    engine.executor.poll_pipeline_finalization.return_value = None

    assert engine._finalize_queued_pipeline_batch(batch) is None
    assert batch.finalization_handle is None
    engine.executor.finalize_pipeline_batch.assert_not_called()

    assert engine._finalize_queued_pipeline_batch(batch) is None
    assert batch.finalization_handle == "decode-handle"
    engine.executor.finalize_pipeline_batch.assert_called_once_with(
        {0: 0, 1: 1}, batch.task.batch_id, batch.stage_physical_ranks[0]
    )


def test_local_final_decode_balances_outstanding_requests_across_stages(mocker) -> None:
    engine = _engine(mocker, _scheduler_output())
    first = engine._submit_queued_pipeline_batch(_scheduler_output("req-a"))
    second_output = _scheduler_output("req-b")
    second_output.step_id = 8
    second = engine._submit_queued_pipeline_batch(second_output)
    first.phase = second.phase = _QueuedPipelineBatchPhase.FINALIZING

    assert engine._select_queued_pipeline_output_rank(first) == 0
    first.finalization_output_rank = 0
    first.finalization_handle = "decode-a"

    assert engine._select_queued_pipeline_output_rank(second) == 1


def test_retirement_requires_two_matching_released_events(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine.executor.reset_mock()
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]

    engine._retire_queued_pipeline_batch(batch)

    assert engine._queued_pipeline_batches == {}


def test_retirement_ticks_transport_before_releasing_context(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine.executor.reset_mock()
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]

    engine._retire_queued_pipeline_batch(batch)

    assert engine.executor.method_calls[:2] == [
        mocker.call.progress_pipeline(),
        mocker.call.pipeline_batch_release_ready({0: 0, 1: 1}, batch.task.batch_id),
    ]
    assert engine.executor.method_calls[2] == mocker.call.release_pipeline_batch({0: 0, 1: 1}, batch.task.batch_id)


def test_retirement_keeps_finalizing_batch_until_receive_ownership_retires(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.decoded_output = object()
    batch.finalizing = True
    engine.executor.pipeline_batch_release_ready.return_value = False

    engine._retire_queued_pipeline_batch(batch)

    assert not batch.release_acknowledged
    assert batch.task.batch_id in engine._queued_pipeline_batches
    engine.executor.release_pipeline_batch.assert_not_called()


def test_final_retirement_cleans_persistent_worker_state(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    engine._retire_queued_pipeline_batch(batch)

    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert engine._queued_pipeline_batches == {}


def test_retirement_rejects_partial_acknowledgement(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine.executor.release_pipeline_batch.return_value = []

    with pytest.raises(RuntimeError, match="both stages"):
        engine._retire_queued_pipeline_batch(batch)

    assert batch.task.batch_id in engine._queued_pipeline_batches


def test_advance_queued_batch_commits_and_retires_intermediate_step(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step = mocker.Mock(return_value=set())
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(PipelineEventType.STEP_COMPLETED, batch.task, 0, 0)
    ]
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]

    assert engine._advance_queued_pipeline_batch(batch) is None
    assert engine._queued_pipeline_batches == {}
    engine.scheduler.commit_pipeline_step.assert_called_once_with(scheduler_output, {"req-a": 1})


def test_advance_queued_batch_decodes_final_output_before_retirement(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step = mocker.Mock(return_value={"req-a"})
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.poll_pipeline_events.return_value = [
        PipelineEvent(PipelineEventType.STEP_COMPLETED, batch.task, 0, 0)
    ]
    engine.executor.finalize_pipeline_batch.return_value = SimpleNamespace(
        get_request_output=lambda request_id: SimpleNamespace(result=SimpleNamespace(error=None))
    )
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    output = engine._advance_queued_pipeline_batch(batch)

    assert output is engine.executor.finalize_pipeline_batch.return_value
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert engine._queued_pipeline_batches == {}


def test_final_retirement_failure_preserves_decoded_output_and_scheduler_finalizing(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step = mocker.Mock(return_value={"req-a"})
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    output = SimpleNamespace(get_request_output=lambda request_id: SimpleNamespace(result=SimpleNamespace(error=None)))
    engine.executor.finalize_pipeline_batch.return_value = output
    engine.executor.release_pipeline_batch.return_value = []

    with pytest.raises(RuntimeError, match="both stages"):
        engine._advance_queued_pipeline_batch(batch)

    assert batch.decoded_output is output
    assert batch.task.batch_id in engine._queued_pipeline_batches
    engine.scheduler.complete_pipeline_request.assert_not_called()

    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]
    engine._advance_queued_pipeline_batch(batch)

    engine.executor.finalize_pipeline_batch.assert_called_once()
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert engine._queued_pipeline_batches == {}


def test_retirement_retry_skips_successful_release_after_cleanup_failure(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing = True
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.side_effect = [
        RuntimeError("cleanup failed"),
        [True, True],
    ]

    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine._retire_queued_pipeline_batch(batch)

    assert batch.release_acknowledged
    assert batch.task.batch_id in engine._queued_pipeline_batches
    engine._retire_queued_pipeline_batch(batch)

    engine.executor.release_pipeline_batch.assert_called_once()
    engine.executor.cleanup_finalized_pipeline_request.assert_called_with("req-a")
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert engine._queued_pipeline_batches == {}


def test_cancellation_keeps_scheduler_capacity_until_retirement(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.finish_requests = mocker.Mock()
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    engine.executor.cancel_pipeline_requests.return_value = [
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 1, 1),
    ]

    engine._cancel_queued_pipeline_batch(batch)

    assert batch.cancelled
    assert batch.phase is _QueuedPipelineBatchPhase.CANCELLING
    engine.scheduler.finish_requests.assert_not_called()

    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine._retire_queued_pipeline_batch(batch)

    engine.scheduler.finish_requests.assert_called_once_with("req-a", DiffusionRequestStatus.FINISHED_ABORTED)
    assert engine._queued_pipeline_batches == {}


def test_partial_cancellation_acknowledgement_is_retryable(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.finish_requests = mocker.Mock()
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    partial = PipelineEvent(PipelineEventType.CANCELLED, batch.task, 0, 0)
    complete = [
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.CANCELLED, batch.task, 1, 1),
    ]
    engine.executor.cancel_pipeline_requests.side_effect = [[partial], complete]

    with pytest.raises(RuntimeError, match="do not match topology"):
        engine._cancel_queued_pipeline_batch(batch)

    assert not batch.cancelled
    assert batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED

    engine._cancel_queued_pipeline_batch(batch)

    assert batch.cancelled
    assert batch.phase is _QueuedPipelineBatchPhase.CANCELLING
    assert engine.executor.cancel_pipeline_requests.call_count == 2


@pytest.mark.parametrize(
    "events",
    [
        pytest.param(
            lambda batch: [
                PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
                PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
            ],
            id="duplicate-stage",
        ),
        pytest.param(
            lambda batch: [
                PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 9),
                PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
            ],
            id="wrong-physical-rank",
        ),
    ],
)
def test_retirement_rejects_mismatched_stage_topology(mocker, events) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine.executor.release_pipeline_batch.return_value = events(batch)

    with pytest.raises(RuntimeError, match="do not match topology"):
        engine._retire_queued_pipeline_batch(batch)

    assert batch.task.batch_id in engine._queued_pipeline_batches
