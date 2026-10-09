# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading

import pytest

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineTransferCoordinator,
    PipelineTransferOffer,
    PipelineTransportProgress,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.worker.pipeline_state import PipelineWorkerUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _executor(mocker) -> MultiprocDiffusionExecutor:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._pipeline_update_error = None
    executor._pipeline_update_lock = threading.Lock()
    executor._pipeline_update_cursor = 0
    executor._pipeline_update_buffers = {0: queue.Queue(), 1: queue.Queue()}
    executor._pipeline_cached_events = []
    executor._pipeline_progress_lock = threading.Lock()
    executor._pipeline_transfer_coordinator = PipelineTransferCoordinator(
        activation_edges={(0, 1)},
        feedback_edges={(1, 0)},
    )
    executor.enqueue_pipeline_transfer_start = mocker.Mock()
    return executor


def _offer(batch_id: str = "batch-a") -> PipelineTransferOffer:
    return PipelineTransferOffer(
        batch_id=batch_id,
        step_index=0,
        epoch=1,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )


def test_sparse_readiness_grants_transfer_without_worker_control_rpc(mocker) -> None:
    executor = _executor(mocker)
    executor.collective_rpc = mocker.Mock(side_effect=AssertionError("unexpected fallback RPC"))
    offer = _offer()
    executor._pipeline_update_buffers[0].put(
        PipelineWorkerUpdate(0, PipelineTransportProgress(offers=[offer]), ())
    )
    executor._pipeline_update_buffers[1].put(
        PipelineWorkerUpdate(1, PipelineTransportProgress(readiness=[offer.identity]), ())
    )

    progress = executor.progress_pipeline()

    assert [grant.offer for grant in progress.grants] == [offer]
    executor.enqueue_pipeline_transfer_start.assert_called_once_with(progress.grants[0])
    executor.collective_rpc.assert_not_called()


def test_transfer_completion_waits_for_both_endpoint_updates(mocker) -> None:
    executor = _executor(mocker)
    coordinator = executor._pipeline_transfer_coordinator
    offer = _offer()
    coordinator.offer(offer)
    coordinator.mark_receive_ready(offer.identity, rank=1)
    grant = coordinator.grant_ready()[0]

    executor._pipeline_update_buffers[0].put(
        PipelineWorkerUpdate(
            0,
            PipelineTransportProgress(
                completions=[offer.identity],
            ),
            (),
        )
    )
    executor.progress_pipeline()
    assert grant.completed_ranks == {0}

    executor._pipeline_update_buffers[1].put(
        PipelineWorkerUpdate(
            1,
            PipelineTransportProgress(
                completions=[offer.identity],
            ),
            (),
        )
    )
    executor.progress_pipeline()
    assert grant.completed_ranks == {0, 1}


def test_sparse_update_drain_is_fair_and_bounded(mocker) -> None:
    executor = _executor(mocker)
    for index in range(40):
        executor._pipeline_update_buffers[0].put(PipelineWorkerUpdate(0, None, (index,)))
    executor._pipeline_update_buffers[1].put(
        PipelineWorkerUpdate(1, None, ("rank-1",))
    )

    executor.progress_pipeline()

    assert len(executor._pipeline_cached_events) == 33
    assert executor._pipeline_update_buffers[0].qsize() == 8
    assert executor._pipeline_update_buffers[1].empty()


def test_transfer_start_is_queued_for_both_workers(mocker) -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._ensure_open = mocker.Mock()
    executor._broadcast_mq = mocker.Mock()
    grant = object()

    executor.enqueue_pipeline_transfer_start(grant)

    executor._broadcast_mq.enqueue.assert_called_once_with(
        {
            "type": "rpc",
            "method": "start_pipeline_transfer",
            "args": (grant,),
            "kwargs": {},
            "output_rank": -1,
            "exec_all_ranks": True,
            "collect_rank_status": False,
            "reply_all_ranks": False,
        }
    )
