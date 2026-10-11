# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _QueuedPipelineBatchPhase
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
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
    request = scheduler_output.scheduled_new_reqs[0].req
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = SimpleNamespace(
        get_request_state=mocker.Mock(return_value=SimpleNamespace(req=request)),
        max_num_running_reqs=2,
        commit_pipeline_step=mocker.Mock(return_value=set()),
        complete_pipeline_request=mocker.Mock(),
        finish_requests=mocker.Mock(),
    )
    engine.executor = mocker.Mock()
    engine.executor.cleanup_finalized_pipeline_request.return_value = [True, True]
    engine.od_config = SimpleNamespace(
        mode="queued",
        max_inflight_batches=2,
        max_num_seqs=2,
        num_gpus=2,
    )
    engine._queued_pipeline_batches = {}
    engine._queued_pipeline_epoch = 0
    engine._cv = threading.Condition()
    return engine


def _step_output(request_id: str, *, finished: bool = False) -> BatchRunnerOutput:
    return BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=request_id,
                step_index=1,
                finished=finished,
                result=DiffusionOutput() if finished else None,
            )
        ]
    )


def test_queued_step_commits_from_nonblocking_worker_future(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    future = Future()
    engine.executor.execute_step.return_value = future
    batch = engine._submit_queued_pipeline_batch(scheduler_output, authorize=False)

    engine._authorize_waiting_queued_batches()
    engine.executor.execute_step.assert_called_once_with(scheduler_output, non_block=True)
    assert batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED

    future.set_result(_step_output("req-a"))
    assert engine._advance_queued_pipeline_batch(batch) is None

    engine.scheduler.commit_pipeline_step.assert_called_once_with(scheduler_output, {"req-a": 1})
    assert batch.batch_id not in engine._queued_pipeline_batches
    engine.executor.cleanup_finalized_pipeline_request.assert_not_called()


def test_final_step_uses_worker_output_then_cleans_request_state(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    engine.scheduler.commit_pipeline_step.return_value = {"req-a"}
    future = Future()
    engine.executor.execute_step.return_value = future
    batch = engine._submit_queued_pipeline_batch(scheduler_output, authorize=False)
    engine._authorize_waiting_queued_batches()
    future.set_result(_step_output("req-a", finished=True))

    output = engine._advance_queued_pipeline_batch(batch)

    assert output is not None
    assert output.get_request_output("req-a").result is not None
    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
    engine.scheduler.complete_pipeline_request.assert_called_once_with("req-a")
    assert batch.batch_id not in engine._queued_pipeline_batches


def test_abort_waits_for_step_and_cleans_prepared_worker_state(mocker) -> None:
    scheduler_output = _scheduler_output()
    engine = _engine(mocker, scheduler_output)
    future = Future()
    engine.executor.execute_step.return_value = future
    batch = engine._submit_queued_pipeline_batch(scheduler_output, authorize=False)
    engine._authorize_waiting_queued_batches()
    batch.abort_requested = True
    future.set_result(_step_output("req-a"))

    engine._cancel_queued_pipeline_batch(batch)
    engine._retire_queued_pipeline_batch(batch)

    engine.executor.cleanup_finalized_pipeline_request.assert_called_once_with("req-a")
    engine.scheduler.finish_requests.assert_called_once()
    assert batch.batch_id not in engine._queued_pipeline_batches
