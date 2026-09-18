# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from tests.diffusion.test_queued_pipeline_engine import _engine, _scheduler_output
from vllm_omni.diffusion.diffusion_engine import _QueuedPipelineBatchPhase
from vllm_omni.diffusion.worker.pipeline_state import PipelineEvent, PipelineEventType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_final_decode_completes_scheduler_only_after_successful_output(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.complete_pipeline_request = mocker.Mock(return_value={"req-a"})
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing_request_ids = frozenset({"req-a"})
    engine.executor.finalize_pipeline_batch.return_value = SimpleNamespace(
        get_request_output=lambda request_id: SimpleNamespace(
            result=SimpleNamespace(error=None),
        )
    )

    output = engine._finalize_queued_pipeline_batch(batch)

    assert output is engine.executor.finalize_pipeline_batch.return_value
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert batch.phase is _QueuedPipelineBatchPhase.FINALIZING


def test_retirement_requires_two_matching_released_events(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]

    engine._retire_queued_pipeline_batch(batch)

    assert engine._queued_pipeline_batches == {}


def test_final_retirement_cleans_persistent_worker_state(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    batch = engine._submit_queued_pipeline_batch(scheduler_output)
    batch.phase = _QueuedPipelineBatchPhase.FINALIZING
    batch.finalizing_request_ids = frozenset({"req-a"})
    engine.executor.release_pipeline_batch.return_value = [
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 0, 0),
        PipelineEvent(PipelineEventType.RELEASED, batch.task, 1, 1),
    ]
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]

    engine._retire_queued_pipeline_batch(batch)

    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
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
