# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
import time
from concurrent.futures import Future

import pytest

from vllm_omni.diffusion.worker.pipeline_stage_engine import (
    PipelineStageEngine,
    _StageAsyncCommand,
    _StageCommand,
    _StageWake,
)
from vllm_omni.diffusion.worker.pipeline_state import PipelineWorkerUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StageWorker:
    def __init__(self, *, fail_tick: bool = False) -> None:
        self.tick_started = threading.Event()
        self.release_tick = threading.Event()
        self.fail_tick = fail_tick
        self.tick_count = 0

    def execute_method(self, method: str, *args, **kwargs):
        if method == "rpc":
            return args[0]
        if method == "pipeline_stage_engine_tick":
            self.tick_count += 1
            self.tick_started.set()
            if self.fail_tick:
                self.release_tick.wait(timeout=2)
                raise RuntimeError("local progress failed")
            return None
        if method == "pipeline_stage_engine_needs_progress":
            return False
        raise AssertionError(f"unexpected Worker method: {method}")


def test_stage_engine_waits_for_executor_wake_after_rpc() -> None:
    worker = _StageWorker()
    updates: list[PipelineWorkerUpdate] = []
    engine = PipelineStageEngine(worker, worker_id=0, device=None, publish_update=updates.append)
    try:
        assert engine.call("rpc", "accepted") == "accepted"
        assert not worker.tick_started.wait(timeout=0.05)

        engine.notify_progress()

        assert worker.tick_started.wait(timeout=1)
        assert updates == []
    finally:
        engine.shutdown()


def test_stage_engine_submit_returns_before_command_completion() -> None:
    worker = _StageWorker()
    updates: list[PipelineWorkerUpdate] = []
    engine = PipelineStageEngine(worker, worker_id=0, device=None, publish_update=updates.append)
    try:
        engine.submit("rpc", "accepted")
        deadline = time.monotonic() + 1
        while engine._commands.qsize() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not updates
    finally:
        engine.shutdown()


def test_stage_engine_rearms_progress_after_prequeued_wake() -> None:
    worker = _StageWorker()
    updates: list[PipelineWorkerUpdate] = []
    engine = PipelineStageEngine(worker, worker_id=0, device=None, publish_update=updates.append)
    result: Future[str] = Future()
    try:
        with engine._state_lock:
            engine._commands.put_nowait(_StageWake())
            engine._wake_pending = True
            engine._commands.put_nowait(_StageAsyncCommand("rpc", ("accepted",), {}, result))

        assert result.result(timeout=1) == "accepted"
        deadline = time.monotonic() + 1
        while worker.tick_count < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert worker.tick_count >= 2
    finally:
        engine.shutdown()


def test_stage_engine_fails_queued_rpc_after_local_progress_failure() -> None:
    worker = _StageWorker(fail_tick=True)
    updates: list[PipelineWorkerUpdate] = []
    engine = PipelineStageEngine(worker, worker_id=0, device=None, publish_update=updates.append)
    call_errors: list[BaseException] = []

    def call_during_tick() -> None:
        try:
            engine.call("rpc", "queued")
        except BaseException as exc:
            call_errors.append(exc)

    call_thread: threading.Thread | None = None
    try:
        engine.notify_progress()
        assert worker.tick_started.wait(timeout=1)

        call_thread = threading.Thread(target=call_during_tick)
        call_thread.start()
        deadline = time.monotonic() + 1
        while engine._commands.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert engine._commands.qsize() == 1

        worker.release_tick.set()
        call_thread.join(timeout=1)

        assert not call_thread.is_alive()
        assert len(call_errors) == 1
        assert isinstance(call_errors[0], RuntimeError)
        assert "local progress failed" in str(call_errors[0].__cause__)
        assert len(updates) == 1
        assert updates[0].error == "RuntimeError: local progress failed"
        with pytest.raises(RuntimeError, match="Pipeline StageEngine failed"):
            engine.call("rpc", "after-failure")
    finally:
        worker.release_tick.set()
        if call_thread is not None:
            call_thread.join(timeout=1)
        engine.shutdown()
