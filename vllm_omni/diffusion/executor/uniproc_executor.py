# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""In-process diffusion executor for single-GPU deployments.

Counterpart to :class:`MultiprocDiffusionExecutor` for ``num_gpus == 1``.
The worker is constructed and driven in the engine process, so none of the
multiproc machinery is created:

* no spawned worker subprocess (and no second model load / CUDA context)
* no ``MessageQueue`` shared-memory ring buffers or zmq ``ipc://`` sockets
* no POSIX ``/dev/shm`` segments for output tensors (``ipc.py`` pack/unpack)

Mirrors vLLM's ``UniProcExecutor`` (``vllm/v1/executor/uniproc_executor.py``).

``timeout`` on :meth:`collective_rpc` is accepted for interface parity but is
not enforced: the call runs on the calling thread, so a hung worker blocks it.
The multiproc executor enforces deadlines via its result-queue dequeue.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineCoordinatorProgress,
    PipelineTransferCoordinator,
    PipelineTransferOffer,
)
from vllm_omni.diffusion.executor.abstract import (
    PIPELINE_GRANT_START_TIMEOUT_S,
    DiffusionExecutor,
    normalize_pipeline_preparation_reports,
    normalize_pipeline_transfer_readiness,
    normalize_pipeline_transport_progress,
    validate_pipeline_topology_reports,
)
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput

logger = init_logger(__name__)


class UniProcDiffusionExecutor(DiffusionExecutor):
    """Runs a single diffusion worker inline, with no IPC of any kind."""

    def _init_executor(self) -> None:
        from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

        self._closed = False
        self._is_failed = False
        self.driver_worker = None
        self._failure_callbacks: list[Callable[[], None]] = []
        self._warned_about_timeout = False

        num_gpus = self.od_config.num_gpus or 1
        if num_gpus != 1:
            raise ValueError(
                f"UniProcDiffusionExecutor supports a single GPU only, got num_gpus={num_gpus}. "
                'Use distributed_executor_backend="mp" for multi-GPU deployments.'
            )

        # The worker still initializes a one-rank distributed environment so
        # that group lookups (`get_dp_group()` and friends) resolve. At world
        # size 1, vLLM's GroupCoordinator skips the device communicator and
        # the shared-memory MessageQueue, and every collective short-circuits.
        worker_cls_path = current_omni_platform.get_diffusion_worker_cls()
        self.driver_worker = WorkerWrapperBase(
            gpu_id=0,
            od_config=self.od_config,
            worker_extension_cls=self.od_config.worker_extension_cls,
            custom_pipeline_args=getattr(self.od_config, "custom_pipeline_args", None),
            base_worker_class=resolve_obj_by_qualname(worker_cls_path),
        )
        logger.info("Diffusion worker initialized in-process (uniproc executor)")

    @property
    def is_dead(self) -> bool:
        return self._closed or self._is_failed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if self.driver_worker is None:
            raise RuntimeError("Diffusion worker is not initialized.")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        """Invoke ``method`` on the single in-process worker.

        ``unique_reply_rank`` / ``exec_all_ranks`` only select which ranks
        participate, which is degenerate at world size 1. The return shape
        still matches :class:`MultiprocDiffusionExecutor` (a bare result when a
        reply rank is named, a one-element list otherwise).
        """
        self._ensure_open()
        if timeout is not None and not self._warned_about_timeout:
            self._warned_about_timeout = True
            logger.info(
                "Diffusion RPC deadlines are not enforced by the in-process executor: "
                "the worker runs on the calling thread, so a hung call blocks it."
            )
        try:
            result = self.driver_worker.execute_method(method, *(args or ()), **(kwargs or {}))
        except Exception:
            if not self._device_is_usable():
                self._mark_failed()
            raise
        return result if unique_reply_rank is not None else [result]

    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.sched.interface import validate_new_request_data_identity
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        runner_outputs: list[RunnerOutput] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            try:
                args: tuple = (new_req.req, self.od_config, scheduler_output.kv_prefetch_job)
                if new_req.diffusion_kv_metadata is not None:
                    args += (new_req.diffusion_kv_metadata,)
                result = self.collective_rpc(
                    "execute_model",
                    args=args,
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                )
                if not isinstance(result, DiffusionOutput):
                    raise RuntimeError(f"Unexpected response type: {type(result)!r}")
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=result,
                    )
                )
            except Exception as exc:
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )
                )
        return BatchRunnerOutput.from_list(runner_outputs)

    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

        self._ensure_open()
        if len(scheduler_output.scheduled_new_reqs) <= 1:
            return self.execute_request(scheduler_output)

        result = self.collective_rpc(
            "execute_model_batch",
            args=(scheduler_output, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, BatchRunnerOutput):
            raise RuntimeError(f"Unexpected response type for execute_batch: {type(result)!r}")
        return result

    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if isinstance(result, BaseRunnerOutput):
            return result
        raise RuntimeError(f"Unexpected response type for execute_step: {type(result)!r}")

    def _queued_control_rpc(
        self,
        method: str,
        *,
        args: tuple = (),
        timeout: float | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        if self._is_failed:
            raise EngineDeadError()
        self._ensure_open()
        try:
            kwargs: dict[str, Any] = {"args": args}
            if timeout is not None:
                kwargs["timeout"] = timeout
            if exec_all_ranks:
                kwargs["exec_all_ranks"] = True
            return self.collective_rpc(method, **kwargs)
        except BaseException:
            self._mark_failed()
            raise

    def submit_pipeline_batch(self, task: Any, pp_stage_spec: Any) -> Any:
        return self._queued_control_rpc("enqueue_pipeline_batch", args=(task, pp_stage_spec))

    def prepare_pipeline_requests(self, scheduler_output: DiffusionSchedulerOutput) -> list[dict[str, Any]]:
        coordinator = getattr(self, "_pipeline_transfer_coordinator", None)
        if coordinator is None:
            raise RuntimeError("pipeline transfer coordinator is not initialized")
        expected_request_ids = tuple(request.request_id for request in scheduler_output.scheduled_new_reqs)
        expected_request_ids += tuple(
            getattr(getattr(scheduler_output, "scheduled_cached_reqs", None), "request_ids", ())
        )
        if len(expected_request_ids) != 1:
            raise ValueError("M2 queued preparation requires exactly one new request.")
        try:
            result = self._queued_control_rpc("prepare_pipeline_requests_all_ranks", args=(scheduler_output,))
            return normalize_pipeline_preparation_reports(
                result,
                coordinator.endpoint_ranks,
                expected_request_ids,
            )
        except BaseException:
            self._mark_failed()
            raise

    def finalize_pipeline_batch(
        self,
        pp_stage_id: dict[int, int],
        batch_id: str,
        output_rank: int,
    ) -> BaseRunnerOutput:
        try:
            result = self.collective_rpc(
                "finalize_pipeline_batch",
                args=(pp_stage_id, batch_id),
                unique_reply_rank=output_rank,
                exec_all_ranks=True,
            )
            if not isinstance(result, BaseRunnerOutput):
                raise RuntimeError("Queued pipeline final decode returned an invalid output.")
            return result
        except BaseException:
            self._mark_failed()
            raise

    def release_pipeline_batch(self, pp_stage_id: dict[int, int], batch_id: str) -> Any:
        result = self._queued_control_rpc("release_pipeline_batch_all_ranks", args=(pp_stage_id, batch_id))
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result

    def cleanup_finalized_pipeline_request(self, request_id: str) -> Any:
        result = self._queued_control_rpc(
            "cleanup_finalized_pipeline_request_all_ranks",
            args=(request_id,),
        )
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result

    def authorize_pipeline_batch(self, pp_stage_id: int | dict[int, int], batch_id: str) -> Any:
        return self._queued_control_rpc("authorize_pipeline_batch", args=(pp_stage_id, batch_id))

    def poll_pipeline_events(self) -> list[Any]:
        self._ensure_open()
        result = self.collective_rpc("poll_pipeline_events_all_ranks")
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result if isinstance(result, list) else [result]

    def cancel_pipeline_requests(self, request_generations: Any) -> Any:
        self._ensure_open()
        result = self.collective_rpc(
            "cancel_pipeline_requests_all_ranks",
            args=(request_generations,),
        )
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result

    def drain_pipeline(self, deadline: float | None = None) -> Any:
        self._ensure_open()
        result = self.collective_rpc("drain_pipeline_all_ranks", args=(deadline,))
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result

    def initialize_pipeline_transfers(
        self,
        activation_edges: set[tuple[int, int]],
        feedback_edges: set[tuple[int, int]],
        max_slots: int = 1,
    ) -> Any:
        if hasattr(self, "_pipeline_transfer_coordinator"):
            raise RuntimeError("pipeline transfer coordinator is already initialized")
        coordinator = PipelineTransferCoordinator(
            activation_edges=activation_edges,
            feedback_edges=feedback_edges,
        )
        result = self._queued_control_rpc("initialize_pipeline_transports_all_ranks", args=(max_slots,))
        try:
            validate_pipeline_topology_reports(result, activation_edges, feedback_edges)
        except BaseException:
            self._mark_failed()
            raise
        self._pipeline_transfer_coordinator = coordinator
        self._pipeline_pending_readiness: dict[tuple[Any, ...], PipelineTransferOffer] = {}
        return result

    def coordinate_pipeline_transfer(self, offer: PipelineTransferOffer) -> list[Any]:
        coordinator = getattr(self, "_pipeline_transfer_coordinator", None)
        if coordinator is None:
            raise RuntimeError("pipeline transfer coordinator is not initialized")
        coordinator.offer(offer)
        self._pipeline_pending_readiness[offer.identity] = offer
        return self._retry_pipeline_transfer_readiness()

    def _retry_pipeline_transfer_readiness(self) -> list[Any]:
        coordinator = self._pipeline_transfer_coordinator
        attempted: set[tuple[Any, ...]] = set()
        while True:
            candidates = [offer for offer in coordinator.pending_readiness_offers() if offer.identity not in attempted]
            if not candidates:
                break
            for offer in candidates:
                attempted.add(offer.identity)
                ready = normalize_pipeline_transfer_readiness(
                    self._queued_control_rpc("accept_pipeline_transfer_offer_all_ranks", args=(offer,))
                )
                if ready:
                    coordinator.mark_receive_ready(offer.identity)
                    self._pipeline_pending_readiness.pop(offer.identity, None)
        return self._start_ready_pipeline_transfers()

    def _start_ready_pipeline_transfers(self) -> list[Any]:
        coordinator = self._pipeline_transfer_coordinator
        grants = coordinator.grant_ready()
        for grant in grants:
            self._queued_control_rpc(
                "start_pipeline_transfer",
                args=(grant,),
                timeout=PIPELINE_GRANT_START_TIMEOUT_S,
            )
        return grants

    def progress_pipeline(self) -> PipelineCoordinatorProgress:
        coordinator = getattr(self, "_pipeline_transfer_coordinator", None)
        if coordinator is None:
            raise RuntimeError("pipeline transfer coordinator is not initialized")
        try:
            result = self._queued_control_rpc("progress_pipeline_transfers_all_ranks")
            worker_progress = normalize_pipeline_transport_progress(result, coordinator.endpoint_ranks)
            progress = PipelineCoordinatorProgress()
            for rank_progress in worker_progress:
                for completion in rank_progress.completions:
                    if coordinator.complete(completion.identity, completion.rank):
                        progress.completed.append(completion.identity)
            for rank_progress in worker_progress:
                for offer in rank_progress.offers:
                    coordinator.offer(offer)
                    self._pipeline_pending_readiness[offer.identity] = offer
            progress.grants.extend(self._retry_pipeline_transfer_readiness())
            return progress
        except BaseException:
            self._mark_failed()
            raise

    def pipeline_stage_physical_ranks(self) -> dict[int, int]:
        coordinator = getattr(self, "_pipeline_transfer_coordinator", None)
        if coordinator is None:
            raise RuntimeError("pipeline transfer coordinator is not initialized")
        return coordinator.stage_physical_ranks

    def pipeline_stage_memory_budget_bytes(self) -> int:
        coordinator = getattr(self, "_pipeline_transfer_coordinator", None)
        if coordinator is None:
            raise RuntimeError("pipeline transfer coordinator is not initialized")
        try:
            result = self._queued_control_rpc("pipeline_stage_memory_budget_bytes", exec_all_ranks=True)
            while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                result = result[0]
            if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
                raise RuntimeError("Workers returned invalid pipeline memory budget reports")
            expected = coordinator.endpoint_ranks
            ranks = {item.get("rank") for item in result}
            if len(result) != len(expected) or ranks != expected:
                raise RuntimeError("pipeline memory budget reports do not cover every configured endpoint")
            budgets = [item.get("free_bytes") for item in result]
            if any(type(value) is not int or value <= 0 for value in budgets):
                raise RuntimeError("pipeline memory budget reports must contain positive integer free_bytes")
            return min(budgets)
        except BaseException:
            self._mark_failed()
            raise

    def _device_is_usable(self) -> bool:
        """Whether the accelerator context survived the failure we just caught.

        Running the worker inline means a worker fault no longer kills a
        process, so nothing else notices it. Recoverable faults (OOM, an
        invalid request) leave the context intact and must not take the
        deployment down. A sticky fault (illegal memory access, ECC error)
        poisons the context: every later launch fails, so the engine needs
        to be restarted. ``synchronize()`` re-raises a pending sticky error
        on the registered accelerator (CUDA, ROCm, NPU).
        """
        if not torch.accelerator.is_available():
            return True
        accelerator = torch.accelerator.current_accelerator()
        device_mod = getattr(torch, accelerator.type, None) if accelerator is not None else None
        is_initialized = getattr(device_mod, "is_initialized", None)
        if callable(is_initialized) and not is_initialized():
            return True
        try:
            torch.accelerator.synchronize()
        except Exception as exc:
            logger.error("Device context is unusable after a worker failure: %s", exc)
            return False
        return True

    def _mark_failed(self) -> None:
        if self._is_failed:
            return
        self._is_failed = True
        logger.error("Diffusion worker failed unrecoverably in-process; marking the executor dead.")
        for callback in self._failure_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Diffusion executor failure callback raised")

    def register_failure_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when the inline worker fails fatally.

        The multiproc executor fires these from its process monitor. There is
        no process to monitor here, so they are fired from ``collective_rpc``
        instead — without them a poisoned device context would leave
        ``check_health`` reporting healthy while every request fails.
        """
        self._failure_callbacks.append(callback)

    def check_health(self) -> None:
        if self._is_failed:
            raise EngineDeadError()
        self._ensure_open()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        worker = self.driver_worker
        self.driver_worker = None
        self._failure_callbacks.clear()
        if worker is not None:
            try:
                worker.shutdown()
            except Exception as exc:
                logger.warning("Diffusion worker shutdown encountered an error: %s", exc)
        # Unlike the multiprocess executor, tearing down an inline worker does
        # not exit its process. Drop the final model reference and return cached
        # allocations to the driver so a subsequent engine can use the device.
        del worker
        gc.collect()
        try:
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()
        except Exception as exc:
            logger.warning("Failed to release accelerator cache during diffusion worker shutdown: %s", exc)
