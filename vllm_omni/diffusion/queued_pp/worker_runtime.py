# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch.distributed as dist

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput


def return_queued_rpc_result(worker: Any, output: Any, rpc_id: str | None) -> bool:
    """Return small queued RPC metadata through the executor's Future pump.

    Tensor-bearing diffusion outputs use the Worker D2H output thread; small
    readiness tuples and preparation reports can be delivered inline.
    """
    if rpc_id is None or getattr(worker.od_config, "mode", "static") != "queued":
        return False
    worker._enqueue_result(
        AsyncDiffusionOutput(
            kind=AsyncOutputKind.RPC_RESULT,
            rpc_id=rpc_id,
            result=output,
        )
    )
    return True


def _all_gather_rank_values(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=get_world_group().cpu_group)
    return values


def _run_and_gather_rank_values(operation: str, func: Callable[[], Any]) -> list[Any]:
    try:
        local_result = (True, func())
    except Exception as exc:
        local_result = (False, f"{type(exc).__name__}: {exc}")
    rank_results = _all_gather_rank_values(local_result)
    failures = [f"rank {rank}: {result}" for rank, (ok, result) in enumerate(rank_results) if not ok]
    if failures:
        raise RuntimeError(f"{operation} failed on " + "; ".join(failures))
    return [result for _, result in rank_results]


class QueuedWorkerRuntime:
    """Worker-local lifecycle calls for queued PP, kept behind a delegate."""

    def __init__(self, worker: Any):
        object.__setattr__(self, "_worker", worker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._worker, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_worker" or name.startswith("_queued_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._worker, name, value)

    def execute_prepared_stepwise(self, scheduler_output: Any, output_rank: int = 0) -> BaseRunnerOutput:
        assert self.model_runner is not None, "Model runner not initialized"
        self._activate_step_lora(scheduler_output)
        profiler = self._get_profiler()
        context = profiler.annotate_context_manager("diffusion_step") if profiler else nullcontext()
        with context:
            output = self.model_runner.execute_prepared_stepwise(scheduler_output, output_rank=output_rank)
        if profiler:
            profiler.step()
        return output

    def prepare_queued_decode(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.prepare_queued_decode(request_ids)

    def execute_prepared_decode(self, request_id: str) -> BaseRunnerOutput:
        assert self.model_runner is not None, "Model runner not initialized"
        profiler = self._get_profiler()
        context = profiler.annotate_context_manager("diffusion_decode") if profiler else nullcontext()
        with context:
            output = self.model_runner.execute_prepared_decode(request_id)
        if profiler:
            profiler.step()
        return output

    def prepare_pipeline_requests(self, scheduler_output: Any) -> dict[str, Any]:
        assert self.model_runner is not None, "Model runner not initialized"
        request_ids = self.model_runner.prepare_pipeline_requests(scheduler_output)
        return {"rank": self.rank, "request_ids": request_ids}

    def prepare_pipeline_requests_all_ranks(self, scheduler_output: Any) -> list[dict[str, Any]]:
        return _run_and_gather_rank_values(
            "queued pipeline request preparation",
            lambda: self.prepare_pipeline_requests(scheduler_output),
        )

    def cleanup_finalized_pipeline_request(self, request_id: str) -> bool:
        assert self.model_runner is not None, "Model runner not initialized"
        self._step_lora_state.pop(request_id, None)
        self.model_runner.state_cache.pop(request_id, None)
        self.model_runner.input_batch = None
        remove_kv = getattr(self.model_runner, "remove_diffusion_kv_requests", None)
        if callable(remove_kv):
            remove_kv([request_id])
        return True

    def cleanup_finalized_pipeline_request_all_ranks(self, request_id: str) -> list[bool]:
        return _run_and_gather_rank_values(
            "queued finalized request cleanup",
            lambda: self.cleanup_finalized_pipeline_request(request_id),
        )
