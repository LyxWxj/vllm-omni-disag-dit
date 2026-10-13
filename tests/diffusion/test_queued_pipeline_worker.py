# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind
from vllm_omni.diffusion.queued_pp.worker_runtime import (
    QueuedWorkerRuntime,
    return_queued_rpc_result,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_small_queued_rpc_result_uses_async_future_envelope() -> None:
    messages = []
    worker = SimpleNamespace(
        od_config=SimpleNamespace(mode="queued"),
        _enqueue_result=messages.append,
    )

    assert return_queued_rpc_result(worker, ("req-a",), "rpc-1")
    assert len(messages) == 1
    assert isinstance(messages[0], AsyncDiffusionOutput)
    assert messages[0].kind is AsyncOutputKind.RPC_RESULT
    assert messages[0].rpc_id == "rpc-1"
    assert messages[0].result == ("req-a",)


def test_static_worker_does_not_route_rpc_through_queued_runtime() -> None:
    worker = SimpleNamespace(od_config=SimpleNamespace(mode="static"))

    assert not return_queued_rpc_result(worker, ("req-a",), "rpc-1")


def test_step_worker_sends_tensor_output_through_async_d2h_queue(monkeypatch) -> None:
    messages = []
    worker = object.__new__(DiffusionWorker)
    worker.result_mq = object()
    worker.od_config = SimpleNamespace(step_execution=True, mode="queued")
    worker._async_output_queue = queue.Queue()
    worker._async_output_done = threading.Condition()
    worker._async_output_pending = 0
    worker._enqueue_result = messages.append
    monkeypatch.setattr(WorkerProc, "_generate_async_output_id", staticmethod(lambda: "output-1"))
    monkeypatch.setattr(current_omni_platform, "record_device_event", lambda: "device-event")

    worker._return_result(BatchRunnerOutput.from_list([]), rpc_id="rpc-1")

    item = worker._async_output_queue.get_nowait()
    assert item[1:] == ("output-1", "device-event")
    assert worker._async_output_pending == 1
    assert messages[0].kind is AsyncOutputKind.COMPUTE_DONE
    assert messages[0].rpc_id == "rpc-1"


def test_step_worker_sends_metadata_rpc_result_inline() -> None:
    messages = []
    worker = object.__new__(DiffusionWorker)
    worker.result_mq = object()
    worker.od_config = SimpleNamespace(step_execution=True, mode="queued")
    worker._enqueue_result = messages.append

    worker._return_result(("req-a",), rpc_id="rpc-2")

    assert messages[0].kind is AsyncOutputKind.RPC_RESULT
    assert messages[0].rpc_id == "rpc-2"
    assert messages[0].result == ("req-a",)


def test_cleanup_drops_request_model_and_lora_state() -> None:
    removed = []
    state = object()
    model_runner = SimpleNamespace(
        state_cache={"req-a": state},
        input_batch=object(),
        remove_diffusion_kv_requests=removed.append,
    )
    worker = SimpleNamespace(
        model_runner=model_runner,
        _step_lora_state={"req-a": (None, 1.0)},
    )

    assert QueuedWorkerRuntime(worker).cleanup_finalized_pipeline_request("req-a")
    assert "req-a" not in model_runner.state_cache
    assert model_runner.input_batch is None
    assert "req-a" not in worker._step_lora_state
    assert removed == [["req-a"]]
