# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
import time

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.queued_pp.executor_adapter import (
    QueuedExecutorAdapter,
    normalize_pipeline_preparation_reports,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _MessageQueue:
    def __init__(self) -> None:
        self.items: queue.Queue = queue.Queue()

    def enqueue(self, value) -> None:
        self.items.put(value)

    def dequeue(self, timeout: float):
        try:
            return self.items.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc


def _executor_stub() -> MultiprocDiffusionExecutor:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._pump_running = False
    executor._pump_stop = threading.Event()
    executor._result_mq = _MessageQueue()
    executor._result_mqs = [executor._result_mq]
    executor._broadcast_mq = _MessageQueue()
    executor._closed = False
    executor._is_failed = False
    executor._rpc_id_counter = 0
    executor._rpc_id_lock = threading.Lock()
    executor._rpc_futures = {}
    executor._output_futures = {}
    executor._completed_outputs = {}
    executor._batch_split_map = {}
    executor._futures_lock = threading.RLock()
    executor._sync_result_buffer = queue.Queue()
    return executor


def test_nonblocking_queued_rpc_returns_correlated_future() -> None:
    raw_executor = _executor_stub()
    adapter = QueuedExecutorAdapter(raw_executor)
    try:
        future = adapter._collective_rpc_future(
            "execute_prepared_stepwise",
            args=("scheduler-output", 1),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        request = raw_executor._broadcast_mq.items.get(timeout=1.0)
        assert request["method"] == "execute_prepared_stepwise"
        assert request["args"] == ("scheduler-output", 1)
        assert request["output_rank"] == 0
        assert request["exec_all_ranks"] is True

        raw_executor._result_mq.enqueue(
            AsyncDiffusionOutput(
                kind=AsyncOutputKind.RPC_RESULT,
                rpc_id=request["rpc_id"],
                result=("req-a",),
            )
        )
        assert future.result(timeout=2.0) == ("req-a",)
    finally:
        raw_executor._pump_stop.set()
        for thread in raw_executor._result_pump_threads:
            thread.join(timeout=2.0)


def test_queued_rpc_future_waits_for_background_output_ready() -> None:
    raw_executor = _executor_stub()
    adapter = QueuedExecutorAdapter(raw_executor)
    try:
        future = adapter._collective_rpc_future(
            "execute_prepared_decode",
            args=("req-a",),
            unique_reply_rank=1,
            exec_all_ranks=False,
        )
        request = raw_executor._broadcast_mq.items.get(timeout=1.0)
        output = DiffusionOutput(output="decoded")
        raw_executor._result_mq.enqueue(
            AsyncDiffusionOutput(
                kind=AsyncOutputKind.COMPUTE_DONE,
                rpc_id=request["rpc_id"],
                async_output_id="output-1",
            )
        )
        raw_executor._result_mq.enqueue(
            AsyncDiffusionOutput(
                kind=AsyncOutputKind.OUTPUT_READY,
                async_output_id="output-1",
                output=output,
            )
        )
        assert future.result(timeout=2.0).output == "decoded"
    finally:
        raw_executor._pump_stop.set()
        for thread in raw_executor._result_pump_threads:
            thread.join(timeout=2.0)


def test_step_rpc_reads_sync_result_buffer_when_queued_pump_owns_queue() -> None:
    raw_executor = _executor_stub()
    _adapter = QueuedExecutorAdapter(raw_executor)
    raw_executor.od_config = type("Config", (), {"step_execution": True})()
    try:
        expected = DiffusionOutput(output="dummy-warmup")
        raw_executor._sync_result_buffer.put(expected)
        result = raw_executor._dequeue_one_with_failure_polling(
            time.monotonic() + 1.0,
            "execute_step",
        )
        assert result is expected
    finally:
        raw_executor._pump_stop.set()
        for thread in raw_executor._result_pump_threads:
            thread.join(timeout=2.0)


def test_preparation_reports_require_each_rank_and_same_request() -> None:
    reports = [
        {"rank": 0, "request_ids": ("req-a",)},
        {"rank": 1, "request_ids": ("req-a",)},
    ]
    assert normalize_pipeline_preparation_reports(reports, frozenset({0, 1}), ("req-a",)) == reports
    with pytest.raises(RuntimeError, match="every configured endpoint"):
        normalize_pipeline_preparation_reports(reports[:1], frozenset({0, 1}), ("req-a",))
    with pytest.raises(RuntimeError, match="disagree on request identity"):
        normalize_pipeline_preparation_reports(
            [reports[0], {"rank": 1, "request_ids": ("req-b",)}],
            frozenset({0, 1}),
            ("req-a",),
        )
