# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.diffusion.request_pipeline import RequestPipelineTransport, build_request_pipeline_tasks


def test_request_pipeline_tasks_are_request_step_transposed() -> None:
    tasks = build_request_pipeline_tasks(["A", "B"], 3, window=2)

    assert [(task.request_id, task.step_index) for task in tasks] == [
        ("A", 0),
        ("B", 0),
        ("A", 1),
        ("B", 1),
        ("A", 2),
        ("B", 2),
    ]
    assert [task.transport_slot for task in tasks] == [0, 1, 0, 1, 0, 1]


def test_request_pipeline_tasks_reject_duplicate_request_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_request_pipeline_tasks(["A", "A"], 2)


def test_request_pipeline_tasks_emit_tombstones_for_short_requests() -> None:
    tasks = build_request_pipeline_tasks(["A", "B"], [1, 3], window=2)

    assert [(task.request_id, task.step_index, task.is_tombstone) for task in tasks] == [
        ("A", 0, False),
        ("B", 0, False),
        ("A", 1, True),
        ("B", 1, False),
        ("A", 2, True),
        ("B", 2, False),
    ]
    assert [task.transport_slot for task in tasks] == [0, 1, 0, 1, 0, 1]


def test_request_pipeline_tasks_reject_mismatched_step_counts() -> None:
    with pytest.raises(ValueError, match="one positive step count"):
        build_request_pipeline_tasks(["A", "B"], [2])


def test_request_pipeline_transport_bounds_active_window() -> None:
    class FakePipeline:
        _request_pipeline_transport = None

        def __init__(self) -> None:
            self.sync_count = 0

        def _sync_pp_send(self) -> None:
            self.sync_count += 1

    pipeline = FakePipeline()
    tasks = build_request_pipeline_tasks(["A"], 3, window=1)
    with RequestPipelineTransport(pipeline, window=1) as transport:
        transport.begin_task(tasks[0])
        transport.begin_task(tasks[1])
        assert len(transport.active_tasks) == 1
    assert pipeline.sync_count == 2
