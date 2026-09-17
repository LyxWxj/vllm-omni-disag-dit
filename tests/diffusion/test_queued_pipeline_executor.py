# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=[MultiprocDiffusionExecutor, UniProcDiffusionExecutor])
def executor(request):
    instance = object.__new__(request.param)
    instance._ensure_open = Mock()
    instance.collective_rpc = Mock()
    instance._is_failed = False
    instance._failure_callbacks = []
    instance.shutdown = Mock()
    return instance


def test_submit_pipeline_batch_passes_rank_local_maps_through_aggregated_rpc(executor) -> None:
    specs = {0: "stage-0", 1: "stage-1"}
    executor.collective_rpc.return_value = ["accepted"]

    assert executor.submit_pipeline_batch("task", specs) == ["accepted"]

    executor.collective_rpc.assert_called_once_with(
        "enqueue_pipeline_batch",
        args=("task", specs),
    )


def test_submission_failure_does_not_issue_authorization(executor) -> None:
    executor.collective_rpc.side_effect = RuntimeError("rank 1 rejected")

    with pytest.raises(RuntimeError, match="rank 1 rejected"):
        executor.submit_pipeline_batch("task", {0: "stage-0", 1: "stage-1"})

    assert executor.collective_rpc.call_count == 1
    assert executor.collective_rpc.call_args.args[0] == "enqueue_pipeline_batch"
    assert executor._is_failed
    executor.collective_rpc.reset_mock()
    with pytest.raises(EngineDeadError):
        executor.authorize_pipeline_batch({0: 0, 1: 1}, "batch-a")
    executor.collective_rpc.assert_not_called()


def test_authorization_passes_rank_local_stage_ids(executor) -> None:
    executor.authorize_pipeline_batch({0: 0, 1: 1}, "batch-a")

    executor.collective_rpc.assert_called_once_with(
        "authorize_pipeline_batch",
        args=({0: 0, 1: 1}, "batch-a"),
    )


def test_event_poll_uses_all_rank_gather_and_flattens_reply(executor) -> None:
    executor.collective_rpc.return_value = [["rank-0-event", "rank-1-event"]]

    assert executor.poll_pipeline_events() == ["rank-0-event", "rank-1-event"]
    executor.collective_rpc.assert_called_once_with("poll_pipeline_events_all_ranks")


def test_drain_aggregates_nonzero_rank_events(executor) -> None:
    executor.collective_rpc.return_value = [["rank-0-released", "rank-1-released"]]

    assert executor.drain_pipeline(deadline=3.0) == ["rank-0-released", "rank-1-released"]
    executor.collective_rpc.assert_called_once_with("drain_pipeline_all_ranks", args=(3.0,))
