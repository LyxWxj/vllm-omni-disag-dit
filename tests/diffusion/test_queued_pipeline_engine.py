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
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, vae_patch_parallel_size=1),
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
    engine.executor.execute_step.assert_called_once_with(scheduler_output, non_block=True, output_rank=0)
    assert batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED

    future.set_result(_step_output("req-a"))
    assert engine._advance_queued_pipeline_batch(batch) is None

    engine.scheduler.commit_pipeline_step.assert_called_once_with(scheduler_output, {"req-a": 1})
    assert batch.batch_id not in engine._queued_pipeline_batches
    engine.executor.cleanup_finalized_pipeline_request.assert_not_called()


def test_scheduler_cycle_requests_share_one_worker_future(mocker) -> None:
    request_a = _scheduler_output("req-a").scheduled_new_reqs[0]
    request_b = _scheduler_output("req-b").scheduled_new_reqs[0]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=7,
        scheduled_new_reqs=[request_a, request_b],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )
    engine = _engine(mocker, scheduler_output)
    future = Future()
    engine.executor.execute_step.return_value = future
    batches = [
        engine._submit_queued_pipeline_batch(task, authorize=False)
        for task in engine._split_queued_scheduler_output(scheduler_output)
    ]

    engine._authorize_waiting_queued_batches()

    engine.executor.execute_step.assert_called_once()
    combined_output = engine.executor.execute_step.call_args.args[0]
    assert combined_output.scheduled_request_ids == ["req-a", "req-b"]
    assert engine.executor.execute_step.call_args.kwargs == {"non_block": True, "output_rank": 0}
    assert batches[0].step_future is batches[1].step_future is future
    assert engine._queued_denoise_batch_count() == 1


def test_scheduler_cycle_requests_rotate_decode_owners(mocker) -> None:
    request_a = _scheduler_output("req-a").scheduled_new_reqs[0]
    request_b = _scheduler_output("req-b").scheduled_new_reqs[0]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=7,
        scheduled_new_reqs=[request_a, request_b],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )
    engine = _engine(mocker, scheduler_output)
    future = Future()
    engine.executor.execute_step.return_value = future
    batches = [
        engine._submit_queued_pipeline_batch(task, authorize=False)
        for task in engine._split_queued_scheduler_output(scheduler_output)
    ]

    engine._authorize_waiting_queued_batches()

    assert [batch.output_rank for batch in batches] == [0, 1]
    engine.executor.execute_step.assert_called_once_with(
        mocker.ANY,
        non_block=True,
        output_rank=0,
    )


def test_prepared_compatible_cycles_share_one_worker_future(mocker) -> None:
    scheduler_outputs = [_scheduler_output(request_id) for request_id in ("req-a", "req-b", "req-c")]
    engine = _engine(mocker, scheduler_outputs[0])
    engine.scheduler.max_num_running_reqs = 3
    future = Future()
    engine.executor.execute_step.return_value = future
    batches = [
        engine._submit_queued_pipeline_batch(scheduler_output, authorize=False)
        for scheduler_output in scheduler_outputs
    ]

    engine._authorize_waiting_queued_batches()

    engine.executor.execute_step.assert_called_once()
    combined_output = engine.executor.execute_step.call_args.args[0]
    assert combined_output.scheduled_request_ids == ["req-a", "req-b", "req-c"]
    assert combined_output.finished_req_ids == set()
    assert all(batch.step_future is future for batch in batches)
    assert all(batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED for batch in batches)


def test_shared_wave_waits_for_all_commits_before_authorizing_next_wave(mocker) -> None:
    scheduler_output = DiffusionSchedulerOutput(
        step_id=7,
        scheduled_new_reqs=[
            _scheduler_output("req-a").scheduled_new_reqs[0],
            _scheduler_output("req-b").scheduled_new_reqs[0],
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )
    engine = _engine(mocker, scheduler_output)
    batches = [
        engine._submit_queued_pipeline_batch(task, authorize=False)
        for task in engine._split_queued_scheduler_output(scheduler_output)
    ]
    future = Future()
    for batch in batches:
        batch.step_future = future
        batch.phase = _QueuedPipelineBatchPhase.STEP_COMPLETED
    authorize = mocker.patch.object(engine, "_authorize_waiting_queued_batches")

    engine._commit_queued_pipeline_step(batches[0])
    authorize.assert_not_called()
    engine._commit_queued_pipeline_step(batches[1])
    authorize.assert_called_once()


def test_future_ready_mid_iteration_commits_whole_wave_before_next_schedule(mocker) -> None:
    output = _scheduler_output("req-a")
    output.scheduled_new_reqs.append(_scheduler_output("req-b").scheduled_new_reqs[0])
    engine = _engine(mocker, output)
    future = Future()
    engine.executor.execute_step.return_value = future
    batches = [
        engine._submit_queued_pipeline_batch(task, authorize=False)
        for task in engine._split_queued_scheduler_output(output)
    ]
    engine._authorize_waiting_queued_batches()

    # Engine has already visited A when the Future becomes ready at B.
    assert engine._advance_queued_pipeline_batch(batches[0]) is None
    future.set_result(BatchRunnerOutput.from_list([
        RunnerOutput(request_id="req-a", step_index=1),
        RunnerOutput(request_id="req-b", step_index=1),
    ]))
    engine._advance_queued_pipeline_batch(batches[1])

    assert engine.scheduler.commit_pipeline_step.call_count == 2
    assert batches[0].phase is _QueuedPipelineBatchPhase.STEP_COMMITTED
    assert engine._queued_pipeline_can_advance_without_schedule()
    engine._advance_queued_pipeline_batch(batches[0])
    assert not engine._queued_pipeline_batches


def test_prepared_incompatible_cycles_keep_separate_worker_futures(mocker) -> None:
    scheduler_outputs = [_scheduler_output(request_id) for request_id in ("req-a", "req-b")]
    engine = _engine(mocker, scheduler_outputs[0])
    keys = {request_id: object() for request_id in ("req-a", "req-b")}
    engine.scheduler.get_request_state.side_effect = lambda request_id: SimpleNamespace(
        req=_scheduler_output(request_id).scheduled_new_reqs[0].req,
        sampling_params_key=keys[request_id],
    )
    futures = [Future(), Future()]
    engine.executor.execute_step.side_effect = futures
    batches = [
        engine._submit_queued_pipeline_batch(scheduler_output, authorize=False)
        for scheduler_output in scheduler_outputs
    ]

    engine._authorize_waiting_queued_batches()

    assert engine.executor.execute_step.call_count == 2
    assert batches[0].step_future is futures[0]
    assert batches[1].step_future is futures[1]


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
