# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineEndpointCompletion,
    PipelineTransferOffer,
    PipelineTransportProgress,
)
from vllm_omni.diffusion.executor.abstract import PIPELINE_GRANT_START_TIMEOUT_S
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _topology_reports():
    return [
        [
            {"rank": 0, "activation_edge": (0, 1), "feedback_edge": (1, 0)},
            {"rank": 1, "activation_edge": (0, 1), "feedback_edge": (1, 0)},
        ]
    ]


@pytest.fixture(params=[MultiprocDiffusionExecutor, UniProcDiffusionExecutor])
def executor(request, mocker):
    instance = object.__new__(request.param)
    instance._ensure_open = mocker.Mock()
    instance.collective_rpc = mocker.Mock()
    instance._is_failed = False
    instance._failure_callbacks = []
    instance.shutdown = mocker.Mock()
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


def test_multiproc_cleanup_failure_preserves_first_worker_error(mocker) -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._ensure_open = mocker.Mock()
    worker_error = RuntimeError("rank 1: activation metadata rejected")
    executor.collective_rpc = mocker.Mock(side_effect=worker_error)
    executor._is_failed = False
    executor._failure_callbacks = []
    executor.shutdown = mocker.Mock(side_effect=RuntimeError("worker shutdown failed"))
    failure_callback = mocker.Mock()
    executor._failure_callbacks.append(failure_callback)

    with pytest.raises(RuntimeError, match="activation metadata rejected") as exc_info:
        executor.submit_pipeline_batch("task", {0: "stage-0", 1: "stage-1"})

    assert exc_info.value is worker_error
    assert executor._queued_control_failure is worker_error
    executor.shutdown.assert_called_once()
    failure_callback.assert_called_once()


def test_authorization_passes_rank_local_stage_ids(executor) -> None:
    executor.authorize_pipeline_batch({0: 0, 1: 1}, "batch-a")

    executor.collective_rpc.assert_called_once_with(
        "authorize_pipeline_batch",
        args=({0: 0, 1: 1}, "batch-a"),
    )


def test_prepare_pipeline_requests_requires_matching_all_rank_reports(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id="req-a")],
    )
    reports = [
        {"rank": 0, "request_ids": ("req-a",)},
        {"rank": 1, "request_ids": ("req-a",)},
    ]
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [reports]

    assert executor.prepare_pipeline_requests(scheduler_output) == reports
    executor.collective_rpc.assert_called_once_with(
        "prepare_pipeline_requests_all_ranks",
        args=(scheduler_output,),
    )


def test_prepare_pipeline_requests_rejects_missing_rank(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id="req-a")],
    )
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [[{"rank": 0, "request_ids": ("req-a",)}]]

    with pytest.raises(RuntimeError, match="do not cover every configured endpoint"):
        executor.prepare_pipeline_requests(scheduler_output)

    assert executor._is_failed


def test_memory_budget_queries_all_endpoints_and_takes_minimum(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [
        [
            {"rank": 0, "free_bytes": 200},
            {"rank": 1, "free_bytes": 100},
        ]
    ]

    assert executor.pipeline_stage_memory_budget_bytes() == 100
    executor.collective_rpc.assert_called_once_with(
        "pipeline_stage_memory_budget_bytes",
        args=(),
        exec_all_ranks=True,
    )


@pytest.mark.parametrize(
    "reports",
    [
        [[{"rank": 0, "free_bytes": 100}]],
        [[{"rank": 0, "free_bytes": 100}, {"rank": 0, "free_bytes": 90}]],
        [[{"rank": 0, "free_bytes": 100}, {"rank": 1, "free_bytes": 0}]],
    ],
)
def test_malformed_memory_budget_reports_fail_executor(executor, reports) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = reports

    with pytest.raises(RuntimeError):
        executor.pipeline_stage_memory_budget_bytes()

    assert executor._is_failed


def test_event_poll_uses_all_rank_gather_and_flattens_reply(executor) -> None:
    executor.collective_rpc.return_value = [["rank-0-event", "rank-1-event"]]

    assert executor.poll_pipeline_events() == ["rank-0-event", "rank-1-event"]
    executor.collective_rpc.assert_called_once_with("poll_pipeline_events_all_ranks")


def test_cancellation_uses_all_rank_gather_and_flattens_reply(executor) -> None:
    executor.collective_rpc.return_value = [["rank-0-cancelled", "rank-1-cancelled"]]

    assert executor.cancel_pipeline_requests([("req-a", 3)]) == [
        "rank-0-cancelled",
        "rank-1-cancelled",
    ]
    executor.collective_rpc.assert_called_once_with(
        "cancel_pipeline_requests_all_ranks",
        args=([("req-a", 3)],),
    )


def test_drain_aggregates_nonzero_rank_events(executor) -> None:
    executor.collective_rpc.return_value = [["rank-0-released", "rank-1-released"]]

    assert executor.drain_pipeline(deadline=3.0) == ["rank-0-released", "rank-1-released"]
    executor.collective_rpc.assert_called_once_with("drain_pipeline_all_ranks", args=(3.0,))


def test_executor_coordinates_ready_offer_and_dispatches_grant(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)}, max_slots=1)
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [True]
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )

    grants = executor.coordinate_pipeline_transfer(offer)

    assert grants[0].offer is offer
    assert executor.collective_rpc.call_count == 2
    assert executor.collective_rpc.call_args_list[0].args == ("accept_pipeline_transfer_offer_all_ranks",)
    assert executor.collective_rpc.call_args_list[0].kwargs == {"args": (offer,)}
    assert executor.collective_rpc.call_args_list[1].args == ("start_pipeline_transfer",)
    assert executor.collective_rpc.call_args_list[1].kwargs == {
        "args": (grants[0],),
        "timeout": PIPELINE_GRANT_START_TIMEOUT_S,
    }


def test_executor_defers_offer_until_receive_credit_is_available(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)}, max_slots=1)
    executor.collective_rpc.reset_mock()
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )

    executor.collective_rpc.return_value = [False]
    assert executor.coordinate_pipeline_transfer(offer) == []
    assert executor._pipeline_transfer_coordinator.snapshot()["offers"] == 1
    assert executor._pipeline_transfer_coordinator.snapshot()["ready"] == 0
    assert not executor._is_failed

    executor.collective_rpc.reset_mock()
    executor.collective_rpc.side_effect = [
        [[PipelineTransportProgress(rank=0), PipelineTransportProgress(rank=1)]],
        [True],
        [True],
    ]
    progress = executor.progress_pipeline()

    assert len(progress.grants) == 1
    assert progress.grants[0].offer == offer
    assert executor._pipeline_transfer_coordinator.snapshot()["offers"] == 0
    assert executor.collective_rpc.call_args_list[1].args == ("accept_pipeline_transfer_offer_all_ranks",)
    assert executor.collective_rpc.call_args_list[2].args == ("start_pipeline_transfer",)


def test_executor_readiness_rejection_never_dispatches_grant(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.side_effect = RuntimeError("sender reservation missing")
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )

    with pytest.raises(RuntimeError, match="sender reservation missing"):
        executor.coordinate_pipeline_transfer(offer)

    executor.collective_rpc.assert_called_once_with("accept_pipeline_transfer_offer_all_ranks", args=(offer,))
    assert executor._is_failed


def test_executor_rejects_worker_topology_mismatch_and_fails_closed(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()

    with pytest.raises(ValueError, match="does not match Worker PP groups"):
        executor.initialize_pipeline_transfers({(2, 3)}, {(3, 2)})

    assert executor._is_failed


def test_progress_retires_endpoints_before_granting_reverse_rank_offer(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    activation = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    executor.collective_rpc.return_value = [True]
    activation_grant = executor.coordinate_pipeline_transfer(activation)[0]
    feedback = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.side_effect = [
        [
            [
                PipelineTransportProgress(
                    rank=1,
                    offers=[feedback],
                    completions=[PipelineEndpointCompletion(activation.identity, 1)],
                ),
                PipelineTransportProgress(
                    rank=0,
                    completions=[PipelineEndpointCompletion(activation.identity, 0)],
                ),
            ]
        ],
        [True],
        [True],
    ]

    progress = executor.progress_pipeline()

    assert progress.completed == [activation.identity]
    assert len(progress.grants) == 1
    assert progress.grants[0].offer is feedback
    assert activation_grant.completed_ranks == {0, 1}
    assert executor.collective_rpc.call_args_list[0].args == ("progress_pipeline_transfers_all_ranks",)
    assert executor.collective_rpc.call_args_list[1].args == ("accept_pipeline_transfer_offer_all_ranks",)
    assert executor.collective_rpc.call_args_list[2].args == ("start_pipeline_transfer",)


def test_progress_retries_ready_offer_after_active_edge_completes(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    activation = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    executor.collective_rpc.return_value = [True]
    executor.coordinate_pipeline_transfer(activation)
    feedback = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    coordinator = executor._pipeline_transfer_coordinator
    coordinator.offer(feedback)
    coordinator.mark_receive_ready(feedback.identity)
    assert coordinator.grant_ready() == []
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.side_effect = [
        [
            PipelineTransportProgress(
                rank=0,
                completions=[PipelineEndpointCompletion(activation.identity, 0)],
            ),
            PipelineTransportProgress(
                rank=1,
                completions=[PipelineEndpointCompletion(activation.identity, 1)],
            ),
        ],
        [True],
    ]

    progress = executor.progress_pipeline()

    assert progress.completed == [activation.identity]
    assert len(progress.grants) == 1
    assert progress.grants[0].offer is feedback
    assert executor.collective_rpc.call_args_list[1].args == ("start_pipeline_transfer",)


def test_progress_invalid_completion_fails_executor_closed(executor) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [
        [
            PipelineTransportProgress(
                rank=0,
                completions=[PipelineEndpointCompletion(("unknown",), 0)],
            ),
            PipelineTransportProgress(rank=1),
        ]
    ]

    with pytest.raises(KeyError, match="unknown pipeline transfer grant"):
        executor.progress_pipeline()

    assert executor._is_failed


@pytest.mark.parametrize(
    "reports",
    [
        pytest.param([], id="empty"),
        pytest.param([PipelineTransportProgress(rank=0)], id="missing-rank"),
    ],
)
def test_progress_rejects_incomplete_rank_coverage(executor, reports) -> None:
    executor.collective_rpc.return_value = _topology_reports()
    executor.initialize_pipeline_transfers({(0, 1)}, {(1, 0)})
    executor.collective_rpc.reset_mock()
    executor.collective_rpc.return_value = [reports]

    with pytest.raises(RuntimeError, match="does not cover every configured endpoint"):
        executor.progress_pipeline()

    assert executor._is_failed
