# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from concurrent.futures import Future

import pytest

from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEvent,
    PipelineEventType,
    PipelineTask,
    PipelineWorkerUpdate,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _executor(mocker) -> MultiprocDiffusionExecutor:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._closed = False
    executor._result_mq = object()
    executor._broadcast_mq = object()
    executor._pipeline_update_error = None
    executor._pipeline_update_lock = threading.Lock()
    executor._pipeline_update_cursor = 0
    executor._pipeline_update_buffers = {0: queue.Queue(), 1: queue.Queue()}
    executor._pipeline_cached_events = []
    executor._pipeline_progress_lock = threading.Lock()
    executor._pipeline_stage_ranks = {0: 0, 1: 1}
    executor._futures_lock = threading.RLock()
    executor._pipeline_step_futures = {}
    executor._uses_autonomous_pipeline_stages = mocker.Mock(return_value=True)
    executor._pipeline_update_callback = None
    return executor


def test_autonomous_progress_collects_worker_events_without_transfer_grants(mocker) -> None:
    executor = _executor(mocker)
    task = PipelineTask(batch_id="batch-a", request_id="req-a", step_index=0, epoch=1)
    executor._pipeline_update_buffers[0].put(
        PipelineWorkerUpdate(
            0,
            None,
            (PipelineEvent(PipelineEventType.STEP_COMPLETED, task, 0, 0),),
        )
    )

    executor.progress_pipeline()

    events = executor.poll_pipeline_events()
    assert events[0].task == task


def test_step_future_resolves_from_autonomous_worker_event(mocker) -> None:
    executor = _executor(mocker)
    task = PipelineTask(batch_id="batch-a", request_id="req-a", step_index=2, epoch=1)
    future = Future()
    executor._pipeline_step_futures[task.batch_id] = future
    executor._pipeline_update_buffers[0].put(
        PipelineWorkerUpdate(
            0,
            None,
            (PipelineEvent(PipelineEventType.STEP_COMPLETED, task, 0, 0),),
        )
    )

    executor.progress_pipeline()

    assert future.done()
    assert future.result().get_request_output("req-a").step_index == 3


def test_sparse_update_drain_is_fair_and_bounded(mocker) -> None:
    executor = _executor(mocker)
    for index in range(40):
        executor._pipeline_update_buffers[0].put(PipelineWorkerUpdate(0, None, (index,)))
    executor._pipeline_update_buffers[1].put(PipelineWorkerUpdate(1, None, ("rank-1",)))

    executor.progress_pipeline()

    assert len(executor._pipeline_cached_events) == 33
    assert executor._pipeline_update_buffers[0].qsize() == 8
    assert executor._pipeline_update_buffers[1].empty()
