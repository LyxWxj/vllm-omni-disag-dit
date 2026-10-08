# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Diffusion Worker for vLLM-Omni.

Handles GPU infrastructure initialization and delegates model operations
to DiffusionModelRunner.
"""

import gc
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
import zmq
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.logger import init_logger
from vllm.profiler.wrapper import CudaProfilerWrapper, WorkerProfiler
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.mem_utils import GiB_bytes, MemorySnapshot, format_gib, memory_profiling
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import (
    AsyncDiffusionOutput,
    AsyncOutputKind,
    DiffusionOutput,
    OmniACK,
    OmniDiffusionConfig,
    OmniSleepTask,
    OmniWakeTask,
)
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    init_worker_kv_connector,
    shutdown_kv_connector,
)
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_cfg_group,
    get_dp_group,
    get_fs_group,
    get_hsdp_replicate_group,
    get_pp_group,
    get_sp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineMessage,
    PipelineStageConnector,
    PipelineTransferGrant,
    PipelineTransferOffer,
    PipelineTransportProgress,
    TransferTicket,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.ipc import (
    DIFFUSION_RPC_RESULT_ENVELOPE,
    pack_diffusion_output_shm,
    payload_carries_typed_media,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, LoRABackend
from vllm_omni.diffusion.queued_pp.worker_runtime import (
    PipelineFinalizationState,
    PipelineTransportRuntime,
    PipelineTransportState,
    QueuedWorkerRuntime,
)
from vllm_omni.diffusion.registry import get_diffusion_ir_op_priority_func
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
    validate_new_request_data_identity,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.pipeline_stage_engine import PipelineStageEngine
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEvent,
    PipelineEventType,
    PipelineFinalizationUpdate,
    PipelineProgress,
    PipelineStageSpec,
    PipelineStageState,
    PipelineTask,
    PipelineWorkerUpdate,
)
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, BatchRunnerOutput
from vllm_omni.engine.stage_init_utils import set_death_signal
from vllm_omni.inputs.data import OmniInteractionPrompt
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform
from vllm_omni.profiler import OmniTorchProfilerWrapper, create_omni_profiler
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)

_ASYNC_OUTPUT_THREAD_JOIN_TIMEOUT_S = 10.0
# Maximum time (in seconds) to wait for pending background D2H / SHM packing
# to drain before the worker executes memory-releasing lifecycle tasks
# (e.g. during sleep/wake transitions). This barrier prevents device tensors
# from being freed while the side CUDA stream is still actively reading them.
_ASYNC_OUTPUT_DRAIN_TIMEOUT_S = 10.0

# Worker entry points that release device memory. Background D2H/SHM packing
# still reads model output tensors, so it must finish before these run.
_MEMORY_RELEASING_METHODS = frozenset({"sleep", "handle_sleep_task"})
_PIPELINE_PREPARATION_METHODS = frozenset({"prepare_pipeline_requests_all_ranks"})


def _cleanup_after_execution_error(exc: Exception) -> None:
    """Release device tensors retained by a failed execution traceback."""
    exc.__traceback__ = None
    try:
        gc.collect()
        current_omni_platform.empty_cache()
    except Exception:
        logger.warning("Failed to release device memory after an execution error", exc_info=True)


def _all_gather_rank_values(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    control_group = get_world_group().cpu_group
    values: list[Any] = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(values, value, group=control_group)
    return values


def _run_and_gather_rank_values(operation: str, func: Callable[[], Any]) -> list[Any]:
    """Run one rank-local probe without stranding peers on local failure."""

    try:
        local_result = (True, func())
    except Exception as exc:
        logger.exception("%s failed on this Worker rank", operation)
        local_result = (False, f"{type(exc).__name__}: {exc}")

    rank_results = _all_gather_rank_values(local_result)
    failures = [f"rank {rank}: {result}" for rank, (ok, result) in enumerate(rank_results) if not ok]
    if failures:
        raise RuntimeError(f"{operation} failed on " + "; ".join(failures))
    return [result for _, result in rank_results]


def _run_and_agree_rank_status(operation: str, func: Callable[[], Any]) -> Any:
    """Run locally, agree on failures, and retain the local result in place."""
    local_result: Any = None
    try:
        local_result = func()
        local_status = (True, None)
    except Exception as exc:
        logger.exception("%s failed on this Worker rank", operation)
        local_status = (False, f"{type(exc).__name__}: {exc}")
    rank_statuses = _all_gather_rank_values(local_status)
    failures = [f"rank {rank}: {error}" for rank, (ok, error) in enumerate(rank_statuses) if not ok]
    if failures:
        raise RuntimeError(f"{operation} failed on " + "; ".join(failures))
    return local_result


def _setup_diffusion_worker_proc_title_and_log_prefix(
    enable_ep: bool,
    use_hsdp: bool,
    hsdp_replicate_size: int = 1,
) -> None:
    """Set the worker process title and log prefix from initialized groups."""
    process_name = "DiffusionWorker"
    if model_parallel_is_initialized():
        dp_group = get_dp_group()
        pp_group = get_pp_group()
        sp_group = get_sp_group()
        cfg_group = get_cfg_group()
        tp_group = get_tp_group()

        if dp_group.world_size > 1:
            process_name += f"_DP{dp_group.rank_in_group}"
        if pp_group.world_size > 1:
            process_name += f"_PP{pp_group.rank_in_group}"
        if sp_group.world_size > 1:
            process_name += f"_SP{sp_group.rank_in_group}"
        if cfg_group.world_size > 1:
            process_name += f"_CFG{cfg_group.rank_in_group}"
        if tp_group.world_size > 1:
            process_name += f"_TP{tp_group.rank_in_group}"
        if use_hsdp:
            fs_group = get_fs_group()
            if fs_group.world_size > 1:
                process_name += f"_FS{fs_group.rank_in_group}"
            if hsdp_replicate_size > 1:
                replicate_group = get_hsdp_replicate_group()
                if replicate_group.world_size > 1:
                    process_name += f"_RP{replicate_group.rank_in_group}"
        if enable_ep:
            ep_group = get_ep_group()
            if ep_group.world_size > 1:
                process_name += f"_EP{ep_group.rank_in_group}"

    set_process_title(name=process_name, prefix="vLLM-Omni")
    decorate_logs(process_name)


@contextmanager
def _force_cutlass_fp8_linear_kernel(quant_config: object | None) -> Iterator[None]:
    import vllm.model_executor.layers.quantization.modelopt as vllm_modelopt

    linear_method_cls = getattr(quant_config, "LinearMethodCls", None)
    if linear_method_cls in {
        vllm_modelopt.ModelOptFp8LinearMethod,
        vllm_modelopt.ModelOptFp8PcPtLinearMethod,
    }:
        from vllm.platforms import current_platform

        if current_platform.is_cuda() and current_platform.has_device_capability(89):
            from vllm.model_executor.kernels.linear import CutlassFP8ScaledMMLinearKernel

            original_init_fp8_linear_kernel = vllm_modelopt.init_fp8_linear_kernel

            def init_fp8_linear_kernel_with_cutlass(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("force_kernel", CutlassFP8ScaledMMLinearKernel)
                return original_init_fp8_linear_kernel(*args, **kwargs)

            vllm_modelopt.init_fp8_linear_kernel = init_fp8_linear_kernel_with_cutlass
            logger.info("Using CUTLASS FP8 linear kernels for this ModelOpt FP8 diffusion stage.")
            try:
                yield
            finally:
                vllm_modelopt.init_fp8_linear_kernel = original_init_fp8_linear_kernel
            return

    yield


def _get_cumem_allocator_class() -> type:
    from vllm.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator


def _resolve_ir_op_priority(od_config: OmniDiffusionConfig, vllm_config: VllmConfig) -> Any:
    ir_op_priority = current_omni_platform.get_default_ir_op_priority(vllm_config)
    ir_op_priority_func = get_diffusion_ir_op_priority_func(od_config)
    if ir_op_priority_func is not None:
        ir_op_priority = ir_op_priority_func(ir_op_priority, vllm_config=vllm_config)
    return ir_op_priority


class DiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to DiffusionModelRunner.
    """

    def _get_pipeline_finalization_state(self) -> PipelineFinalizationState:
        state = getattr(self, "_pipeline_finalization_state", None)
        if state is None:
            state = PipelineFinalizationState()
            self._pipeline_finalization_state = state
        return state

    def _get_pipeline_transport_state(self) -> PipelineTransportState:
        state = getattr(self, "_pipeline_transport_state", None)
        if state is None:
            state = PipelineTransportState()
            self._pipeline_transport_state = state
        return state

    def _get_pipeline_transport_runtime(self) -> PipelineTransportRuntime:
        runtime = getattr(self, "_pipeline_transport_runtime", None)
        if runtime is None:
            runtime = PipelineTransportRuntime(self)
            self._pipeline_transport_runtime = runtime
        return runtime

    def _get_queued_worker_runtime(self) -> QueuedWorkerRuntime:
        runtime = getattr(self, "_queued_worker_runtime", None)
        if runtime is None:
            runtime = QueuedWorkerRuntime(self)
            self._queued_worker_runtime = runtime
        return runtime

    def _run_and_gather_rank_values(self, operation: str, func: Callable[[], Any]) -> list[Any]:
        return _run_and_gather_rank_values(operation, func)

    def _get_pp_group(self) -> Any:
        return get_pp_group()

    @property
    def _pipeline_finalization_futures(self) -> dict[str, Future[Any]]:
        return self._get_pipeline_finalization_state().futures

    @_pipeline_finalization_futures.setter
    def _pipeline_finalization_futures(self, value: dict[str, Future[Any]]) -> None:
        self._get_pipeline_finalization_state().futures = value

    @property
    def _pipeline_finalization_executor(self) -> ThreadPoolExecutor | None:
        return self._get_pipeline_finalization_state().executor

    @_pipeline_finalization_executor.setter
    def _pipeline_finalization_executor(self, value: ThreadPoolExecutor | None) -> None:
        self._get_pipeline_finalization_state().executor = value

    @property
    def _pipeline_finalization_published(self) -> set[str]:
        return self._get_pipeline_finalization_state().published

    @_pipeline_finalization_published.setter
    def _pipeline_finalization_published(self, value: set[str]) -> None:
        self._get_pipeline_finalization_state().published = value

    @property
    def _pipeline_finalization_device_events(self) -> dict[str, Any]:
        return self._get_pipeline_finalization_state().device_events

    @_pipeline_finalization_device_events.setter
    def _pipeline_finalization_device_events(self, value: dict[str, Any]) -> None:
        self._get_pipeline_finalization_state().device_events = value

    @property
    def _pipeline_finalization_stream(self) -> Any | None:
        return self._get_pipeline_finalization_state().stream

    @_pipeline_finalization_stream.setter
    def _pipeline_finalization_stream(self, value: Any | None) -> None:
        self._get_pipeline_finalization_state().stream = value

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
        skip_load_model: bool = False,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionModelRunner | None = None
        self.init_snapshot: MemorySnapshot | None = None
        self.requested_memory: int | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        # Worker-side cache of (lora_request, lora_scale) per scheduled
        # request id. Used by step mode to recover LoRA identity for cached
        # requests, which only carry their request_id in subsequent ticks.
        self._step_lora_state: dict[str, tuple[LoRARequest | None, float]] = {}
        self._pipeline_stages: dict[int, PipelineStageState] = {}
        self._pipeline_events: list[PipelineEvent] = []
        self._pipeline_connectors: dict[PipelineEdgeKind, PipelineStageConnector] = {}
        self._pipeline_transport_state = PipelineTransportState()
        self._pipeline_finalization_state = PipelineFinalizationState()
        self.stage_id = getattr(od_config, "stage_id", 0)
        self.init_device()
        # Create model runner — one decision chain, in precedence order:
        #   1. explicit od_config.diffusion_model_runner_cls (user override),
        #   2. the runner declared by the engine class that engine_backend
        #      selects (e.g. ARDiffusionEngine -> ARDiffusionModelRunner),
        #   3. the platform default.
        # Routing policy therefore lives on the engine class / config surface;
        # engines never mutate od_config. Overrides must be import-path
        # strings — guard with isinstance so a non-string (e.g. a Mock
        # od_config in tests) doesn't shadow the platform hook.
        runner_override = getattr(self.od_config, "diffusion_model_runner_cls", None)
        engine_runner = None
        if not (isinstance(runner_override, str) and runner_override):
            try:
                from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

                engine_cls = DiffusionEngine.resolve_engine_class(self.od_config)
                engine_runner = getattr(engine_cls, "default_diffusion_model_runner_cls", None)
            except Exception:
                logger.warning("Worker %s: engine_backend resolution failed; using platform runner", self.rank)
                engine_runner = None
        if isinstance(runner_override, str) and runner_override:
            model_runner_cls_path = runner_override
        elif isinstance(engine_runner, str) and engine_runner:
            model_runner_cls_path = engine_runner
        else:
            model_runner_cls_path = current_omni_platform.get_diffusion_model_runner_cls()
        model_runner_cls = resolve_obj_by_qualname(model_runner_cls_path)
        self.model_runner = model_runner_cls(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.profiler: WorkerProfiler | None = self._create_profiler()
        if not skip_load_model:
            self.load_model(load_format=self.od_config.diffusion_load_format)
            self.init_lora_manager()
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def init_device(self) -> None:
        """Initialize the device and distributed environment."""
        world_size = self.od_config.num_gpus
        rank = self.rank

        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Setup device
        self.device = current_omni_platform.get_torch_device(rank)
        current_omni_platform.set_device(self.device)

        # Create vllm_config for parallel configuration. Pass explicit device_config
        # so DeviceConfig does not rely on current_platform in worker subprocesses.
        vllm_config = create_diffusion_vllm_config(self.device, self.od_config)
        # Since vLLM v0.20.0, IR wraps GPU ops. Set IR op priority preference to enforce GPU op fusion during wrapping.
        # Also need to log, because vLLM internally logs another line in VllmConfig.__post_init__. Avoid confusion.
        vllm_config.kernel_config.ir_op_priority = _resolve_ir_op_priority(self.od_config, vllm_config)
        if self.od_config.moe_backend != "auto":
            logger.warning(
                "Overriding MoE backend from default 'auto' to '%s' per deploy config.",
                self.od_config.moe_backend,
            )
            vllm_config.kernel_config.moe_backend = self.od_config.moe_backend
        logger.info(
            "Final IR op priority after setting vLLM-Omni overrides: %s", vllm_config.kernel_config.ir_op_priority
        )
        self.vllm_config = vllm_config
        current_omni_platform.init_diffusion_worker_vllm_config(vllm_config)

        # Initialize distributed environment
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                allgather_degree=parallel_config.allgather_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
                enable_expert_parallel=parallel_config.enable_expert_parallel,
                use_hsdp=parallel_config.use_hsdp,
            )
            _setup_diffusion_worker_proc_title_and_log_prefix(
                enable_ep=parallel_config.enable_expert_parallel,
                use_hsdp=parallel_config.use_hsdp,
                hsdp_replicate_size=parallel_config.hsdp_replicate_size,
            )
            if (
                getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
                is DiffusionKVCacheMode.PAGED_SCHEDULER
            ):
                gc.collect()
                current_omni_platform.empty_cache()
                self.init_snapshot = MemorySnapshot(device=self.device)
                self.requested_memory = request_memory(
                    self.init_snapshot,
                    vllm_config.cache_config,
                )
                logger.debug(
                    "Worker %d: Diffusion KV initial memory snapshot: %r; requested=%s GiB",
                    self.rank,
                    self.init_snapshot,
                    format_gib(self.requested_memory),
                )
            init_workspace_manager(self.device)

    def _create_profiler(self) -> WorkerProfiler | None:
        profiler_config = self.od_config.profiler_config
        profiler_type = getattr(profiler_config, "profiler", None)
        if profiler_type == "torch":
            return create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=f"diffusion_rank{self.rank}",
                local_rank=self.local_rank,
            )
        if profiler_type == "cuda":
            return CudaProfilerWrapper(profiler_config)
        if profiler_type is not None:
            logger.warning("Unknown profiler backend %r on diffusion worker %s", profiler_type, self.rank)
        return None

    def _get_profiler(self) -> WorkerProfiler | None:
        return getattr(self, "profiler", None)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None, **kwargs) -> None:
        """Load the diffusion model using DiffusionModelRunner."""
        load_format = kwargs.get("load_format", load_format)
        custom_pipeline_name = kwargs.get("custom_pipeline_name", custom_pipeline_name)
        cutlass_fp8_context = (
            _force_cutlass_fp8_linear_kernel(self.od_config.quantization_config)
            if getattr(self.od_config, "force_cutlass_fp8", False)
            else nullcontext()
        )
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
            cutlass_fp8_context,
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )
            current_omni_platform.synchronize()
            gc.collect()
        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "Worker %d: Process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

        # When load_format is "dummy", pipeline will init with custom pipeline later
        if load_format != "dummy":
            assert self.model_runner.pipeline is not None

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """Return native rank-local specs for every diffusion Worker."""

        assert self.model_runner is not None
        return _run_and_gather_rank_values(
            "Diffusion KV cache spec discovery",
            self.model_runner.get_kv_cache_spec,
        )

    def determine_available_kv_memory(self, profile_requests: list[OmniDiffusionRequest]) -> list[int]:
        """Profile and return each rank's safe Diffusion KV memory budget."""

        def determine_local_memory() -> int:
            assert self.vllm_config is not None
            assert self.model_runner is not None
            if self.init_snapshot is None or self.requested_memory is None:
                raise RuntimeError("Diffusion KV memory snapshot was not captured before model loading")
            override = self.vllm_config.cache_config.kv_cache_memory_bytes
            if override:
                # Match native vLLM: an explicit cache budget skips automatic
                # capacity derivation, but still runs the maximum-shape model
                # request so lazy kernels and communication buffers initialize.
                self.model_runner.profile_run(profile_requests)
                logger.info(
                    "Worker %d: Initial free memory %s GiB, reserved %s GiB memory for "
                    "Diffusion KV Cache as specified by kv_cache_memory_bytes config and "
                    "skipped automatic memory profiling. This does not respect the "
                    "gpu_memory_utilization config. A profile warmup was still executed.",
                    self.rank,
                    format_gib(self.init_snapshot.free_memory),
                    format_gib(int(override)),
                )
                return int(override)

            with memory_profiling(
                self.init_snapshot,
                weights_memory=self.model_runner.model_memory_usage,
            ) as profile_result:
                self.model_runner.profile_run(profile_requests)

            available_memory = self.requested_memory - profile_result.non_kv_cache_memory
            if available_memory <= 0:
                raise RuntimeError(
                    "No memory remains for Diffusion KV cache after profiling: "
                    f"requested_memory={self.requested_memory} bytes, "
                    f"non_kv_cache_memory={profile_result.non_kv_cache_memory} bytes. "
                    "Increase gpu_memory_utilization or reduce the maximum profile request shape."
                )
            free_gpu_memory = profile_result.after_profile.free_memory
            unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
            logger.debug(
                "Worker %d: Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
                self.rank,
                format_gib(self.init_snapshot.free_memory),
                self.vllm_config.cache_config.gpu_memory_utilization,
                format_gib(self.requested_memory),
            )
            logger.debug(
                "Worker %d: Free memory after profiling: %s GiB (total), %s GiB (within requested)",
                self.rank,
                format_gib(free_gpu_memory),
                format_gib(free_gpu_memory - unrequested_memory),
            )
            logger.debug("Worker %d: %r", self.rank, profile_result)
            logger.info(
                "Worker %d: Available Diffusion KV cache memory: %s GiB",
                self.rank,
                format_gib(available_memory),
            )
            return int(available_memory)

        return [
            int(value)
            for value in _run_and_gather_rank_values(
                "Diffusion KV memory discovery",
                determine_local_memory,
            )
        ]

    def set_kv_cache_configs(
        self,
        kv_cache_configs: list[KVCacheConfig],
        resolved_max_model_len: int,
    ) -> None:
        """Select this rank's config and initialize its physical KV pages."""

        assert self.model_runner is not None
        assert self.vllm_config is not None
        if len(kv_cache_configs) != self.od_config.num_gpus:
            raise ValueError(
                "Diffusion KVCacheConfig rank count mismatch: "
                f"expected={self.od_config.num_gpus}, got={len(kv_cache_configs)}"
            )
        if not 0 <= self.rank < len(kv_cache_configs):
            raise ValueError(f"Diffusion Worker rank {self.rank} has no rank-local KVCacheConfig")
        if type(resolved_max_model_len) is not int or resolved_max_model_len <= 0:
            raise ValueError("resolved Diffusion KV max_model_len must be a positive integer")

        # Native cache sizing may resolve an explicit ``-1`` model-length
        # sentinel to the capacity that actually fits the profiled pool.
        self.vllm_config.model_config.max_model_len = resolved_max_model_len
        kv_cache_config = kv_cache_configs[self.rank]
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        init_worker_kv_connector(self.vllm_config, kv_cache_config)
        with self._maybe_get_memory_pool_context("kv_cache"):
            self.model_runner.set_kv_cache_config(kv_cache_config)

    def remove_diffusion_kv_requests(self, request_ids: list[str]) -> int:
        """Clear Worker-local rows without freeing Scheduler-owned blocks."""

        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.remove_diffusion_kv_requests(request_ids)

    def init_lora_manager(self) -> None:
        """Initialize the LoRA manager for this worker."""
        pipeline = self.model_runner.pipeline
        if pipeline is None:
            return

        # A release whose weights cannot be expressed as switchable LoRA layers
        # is fused into the checkpoint while the pipeline loads. There is then
        # no adapter left to register, and handing the same path to the manager
        # would only fail on a format it does not accept.
        if getattr(pipeline, "lora_is_fused", False):
            logger.info("LoRA was fused into the checkpoint at load time; skipping the dynamic LoRA manager.")
            return

        lora_path = self.od_config.lora_path
        if isinstance(lora_path, list) and len(lora_path) == 1:
            lora_path = lora_path[0]

        lora_backend = self.od_config.lora_backend
        if lora_backend == LoRABackend.PEFT:
            self.lora_manager = DiffusionLoRAManager(
                pipeline=self.model_runner.pipeline,
                device=self.device,
                dtype=self.od_config.dtype,
                max_cached_adapters=self.od_config.max_cpu_loras,
                lora_path=lora_path,
                lora_scale=self.od_config.lora_scale,
            )
        elif lora_backend == LoRABackend.DISTILL:
            pipeline = self.model_runner.pipeline
            if hasattr(pipeline, "load_lora_weights"):
                if self.od_config.lora_scale > 1.0:
                    logger.warning("lora_scale > 1.0 may not take any effect when using distilled LoRA backend.")
                pipeline.load_lora_weights(lora_path)
                pipeline.lora_is_fused = True
            else:
                logger.warning("Pipeline does not support loading distilled LoRA weights for now.")
        else:
            raise ValueError(f"Unknown LoRA backend: {lora_backend}. Available choices: {LoRABackend.__members__}")

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop profiling for this GPU worker.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename.
        """
        profiler = self._get_profiler()
        if profiler is None:
            return

        if is_start:
            if isinstance(profiler, OmniTorchProfilerWrapper):
                import time

                filename = profile_prefix or f"diffusion_rank{self.rank}_{int(time.time())}"
                profiler.set_trace_filename(filename)
            profiler.start()
        else:
            profiler.stop()

    def _run_ar_diffusion_session_lifecycle(self, method: str, session_id: str) -> bool:
        """Delegate an optional AR session lifecycle call to the model runner."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        assert self.model_runner is not None, "Model runner not initialized"
        lifecycle_method = getattr(self.model_runner, method, None)
        if not callable(lifecycle_method):
            return False
        lifecycle_method(session_id)
        return True

    def reset_ar_diffusion_session(self, session_id: str) -> bool:
        """Reset runner-owned AR state through the collective RPC boundary."""
        return self._run_ar_diffusion_session_lifecycle("reset_session", session_id)

    def close_ar_diffusion_session(self, session_id: str) -> bool:
        """Close runner-owned AR state through the collective RPC boundary."""
        return self._run_ar_diffusion_session_lifecycle("close_session", session_id)

    def execute_model(
        self,
        req: OmniDiffusionRequest | list[NewRequestData],
        od_config: OmniDiffusionConfig,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner.

        If *req* is a list (DP multi-concurrency), each rank picks one complete
        NewRequestData envelope based on its distributed rank. AllGather in
        the layerwise offload only gathers weight shards (request-independent),
        so all ranks stay synchronised at each AllGather call while computing
        different activations. Selecting the envelope keeps Scheduler-issued
        KV metadata bound to the request that owns its block tables.

        Each rank returns its OWN DiffusionOutput (no gather). The executor
        collects N responses via the per-worker result queues.
        """
        assert self.model_runner is not None, "Model runner not initialized"

        # DP multi-concurrency: pick one request per DP rank.
        # Use rank_in_group (DP rank) not global rank, so that SP/TP/CFG
        # ranks within the same DP replica select the same request.
        is_batch = isinstance(req, list)
        if is_batch:
            from vllm_omni.diffusion.distributed.parallel_state import get_data_parallel_rank

            dp_rank = get_data_parallel_rank()
            idx = dp_rank % len(req)
            new_req = req[idx]
            validate_new_request_data_identity(new_req)
            req = new_req.req
            diffusion_kv_metadata = new_req.diffusion_kv_metadata

        if self.lora_manager is not None:
            try:
                self.lora_manager.set_active_adapter(req.sampling_params.lora_request, req.sampling_params.lora_scale)
            except Exception as exc:
                if req.sampling_params.lora_request is not None:
                    raise
                logger.warning("LoRA activation skipped: %s", exc)
        profiler = self._get_profiler()
        ctx = profiler.annotate_context_manager("diffusion_forward") if profiler else nullcontext()
        with ctx:
            kwargs: dict[str, Any] = {"kv_prefetch_job": kv_prefetch_job}
            if diffusion_kv_metadata is not None:
                kwargs["diffusion_kv_metadata"] = diffusion_kv_metadata
            output = self.model_runner.execute_model(req, **kwargs)
        if profiler:
            profiler.step()

        # Each primary rank returns its own output tagged with its DP rank
        # so the executor can match results to requests by dp_rank.
        # Non-primary ranks (SP/TP > 0) do not reply.
        if is_batch:
            from vllm_omni.diffusion.distributed.parallel_state import get_data_parallel_rank

            return {"dp_rank": get_data_parallel_rank(), "output": output}
        return output

    def execute_model_batch(
        self, scheduler_output: DiffusionSchedulerOutput, od_config: OmniDiffusionConfig
    ) -> BatchRunnerOutput:
        """Batch forward: LoRA activate once, delegate to model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        # LoRA: same adapter/scale within batch guaranteed by RequestBatchSamplingParamsKey.
        if self.lora_manager is not None and scheduler_output.scheduled_new_reqs:
            sp = scheduler_output.scheduled_new_reqs[0].req.sampling_params
            try:
                self.lora_manager.set_active_adapter(sp.lora_request, sp.lora_scale)
            except Exception as exc:
                if sp.lora_request is not None:
                    raise
                logger.warning("LoRA activation skipped: %s", exc)
        profiler = self._get_profiler()
        ctx = profiler.annotate_context_manager("diffusion_forward_batch") if profiler else nullcontext()
        with ctx:
            output = self.model_runner.execute_model_batch(scheduler_output, od_config)
        if profiler:
            profiler.step()
        return output

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute one diffusion step by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        self._activate_step_lora(scheduler_output)
        profiler = self._get_profiler()
        ctx = profiler.annotate_context_manager("diffusion_step") if profiler else nullcontext()
        with ctx:
            output = self.model_runner.execute_stepwise(scheduler_output)
        if profiler:
            profiler.step()
        return output


    @property
    def pipeline_stages(self) -> dict[int, PipelineStageState]:
        return self._get_queued_worker_runtime().pipeline_stages

    @property
    def pipeline_events(self) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().pipeline_events

    @property
    def pipeline_connectors(self) -> dict[PipelineEdgeKind, PipelineStageConnector]:
        return self._get_queued_worker_runtime().pipeline_connectors

    @property
    def pipeline_send_tickets(self) -> dict[tuple[Any, ...], TransferTicket]:
        return self._get_queued_worker_runtime().pipeline_send_tickets

    @property
    def pipeline_receive_reservations(self) -> dict[tuple[Any, ...], PipelineEdgeKind]:
        return self._get_queued_worker_runtime().pipeline_receive_reservations

    @property
    def pipeline_started_receive_ids(self) -> set[tuple[Any, ...]]:
        return self._get_queued_worker_runtime().pipeline_started_receive_ids

    @property
    def pipeline_receive_consumers(
        self,
    ) -> dict[tuple[Any, ...], tuple[PipelineEdgeKind, PipelineMessage, Any | None]]:
        return self._get_queued_worker_runtime().pipeline_receive_consumers

    @property
    def pipeline_pending_received(self) -> dict[PipelineEdgeKind, deque[PipelineMessage]]:
        return self._get_queued_worker_runtime().pipeline_pending_received

    def _pipeline_stage(self, pp_stage_spec: PipelineStageSpec) -> PipelineStageState:
        return self._get_queued_worker_runtime()._pipeline_stage(pp_stage_spec)

    def _pipeline_event(
        self,
        event_type: PipelineEventType,
        task: PipelineTask,
        pp_stage_id: int,
    ) -> PipelineEvent:
        return self._get_queued_worker_runtime()._pipeline_event(event_type, task, pp_stage_id)

    def initialize_pipeline_transports(self, max_slots: int = 1) -> dict[str, Any]:
        return self._get_queued_worker_runtime().initialize_pipeline_transports(max_slots)

    def initialize_pipeline_transports_all_ranks(self, max_slots: int = 1) -> list[dict[str, Any]]:
        return self._get_queued_worker_runtime().initialize_pipeline_transports_all_ranks(max_slots)

    def reserve_pipeline_send(self, offer: PipelineTransferOffer, payload: dict[str, Any]) -> PipelineTransferOffer:
        return self._get_queued_worker_runtime().reserve_pipeline_send(offer, payload)

    def accept_pipeline_transfer_offer(self, offer: PipelineTransferOffer) -> bool:
        return self._get_queued_worker_runtime().accept_pipeline_transfer_offer(offer)

    def accept_pipeline_transfer_offer_all_ranks(self, offer: PipelineTransferOffer) -> bool:
        return self._get_queued_worker_runtime().accept_pipeline_transfer_offer_all_ranks(offer)

    def accept_pipeline_transfer_offer_rank_local(self, offer: PipelineTransferOffer) -> dict[str, Any]:
        return self._get_queued_worker_runtime().accept_pipeline_transfer_offer_rank_local(offer)

    def accept_pipeline_transfer_offers_rank_local(
        self,
        offers: tuple[PipelineTransferOffer, ...] | list[PipelineTransferOffer],
    ) -> dict[str, Any]:
        return self._get_queued_worker_runtime().accept_pipeline_transfer_offers_rank_local(offers)

    def start_pipeline_transfer(self, grant: PipelineTransferGrant) -> bool:
        return self._get_queued_worker_runtime().start_pipeline_transfer(grant)

    def retire_pipeline_send(self, identity: tuple[Any, ...]) -> bool:
        return self._get_queued_worker_runtime().retire_pipeline_send(identity)

    def progress_pipeline_transfers(self) -> PipelineTransportProgress:
        return self._get_queued_worker_runtime().progress_pipeline_transfers()

    def pipeline_stage_engine_tick(self) -> PipelineWorkerUpdate | None:
        return self._get_queued_worker_runtime().pipeline_stage_engine_tick()

    def pipeline_stage_engine_needs_progress(self) -> bool:
        return self._get_queued_worker_runtime().pipeline_stage_engine_needs_progress()

    def progress_pipeline_transfers_and_poll_events_all_ranks(
        self,
    ) -> list[tuple[PipelineTransportProgress, list[Any]]]:
        return self._get_queued_worker_runtime().progress_pipeline_transfers_and_poll_events_all_ranks()

    def progress_pipeline_transfers_and_poll_events(
        self,
        pending_offers: tuple[PipelineTransferOffer, ...] = (),
    ) -> tuple[PipelineTransportProgress, list[Any]]:
        return self._get_queued_worker_runtime().progress_pipeline_transfers_and_poll_events(pending_offers)

    def enqueue_pipeline_batch(
        self,
        task: PipelineTask,
        pp_stage_spec: PipelineStageSpec | dict[int, PipelineStageSpec],
    ) -> PipelineEvent:
        return self._get_queued_worker_runtime().enqueue_pipeline_batch(task, pp_stage_spec)

    def prepare_pipeline_requests(self, scheduler_output: DiffusionSchedulerOutput) -> dict[str, Any]:
        return self._get_queued_worker_runtime().prepare_pipeline_requests(scheduler_output)

    def prepare_pipeline_requests_all_ranks(self, scheduler_output: DiffusionSchedulerOutput) -> list[dict[str, Any]]:
        return self._get_queued_worker_runtime().prepare_pipeline_requests_all_ranks(scheduler_output)

    def authorize_pipeline_batch(self, pp_stage_id: int | dict[int, int], batch_id: str) -> PipelineEvent:
        return self._get_queued_worker_runtime().authorize_pipeline_batch(pp_stage_id, batch_id)

    def authorize_pipeline_batches(
        self,
        authorizations: list[tuple[int | dict[int, int], str]],
    ) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().authorize_pipeline_batches(authorizations)

    def admit_pipeline_batch(
        self,
        task: PipelineTask,
        pp_stage_spec: PipelineStageSpec | dict[int, PipelineStageSpec],
    ) -> tuple[PipelineEvent, PipelineEvent]:
        return self._get_queued_worker_runtime().admit_pipeline_batch(task, pp_stage_spec)

    def admit_pipeline_batches(
        self,
        admissions: list[tuple[PipelineTask, PipelineStageSpec | dict[int, PipelineStageSpec]]],
    ) -> tuple[PipelineEvent, ...]:
        return self._get_queued_worker_runtime().admit_pipeline_batches(admissions)

    def progress_pipeline(
        self,
        pp_stage_id: int,
        intermediate_tensors: Any | None = None,
    ) -> PipelineProgress | None:
        return self._get_queued_worker_runtime().progress_pipeline(pp_stage_id, intermediate_tensors)

    def complete_pipeline_feedback(self, pp_stage_id: int, batch_id: str, latents: torch.Tensor) -> PipelineEvent:
        return self._get_queued_worker_runtime().complete_pipeline_feedback(pp_stage_id, batch_id, latents)

    def cancel_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineEvent:
        return self._get_queued_worker_runtime().cancel_pipeline_batch(pp_stage_id, batch_id)

    def release_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineEvent:
        return self._get_queued_worker_runtime().release_pipeline_batch(pp_stage_id, batch_id)

    def pipeline_batch_release_ready(self, pp_stage_id: int | dict[int, int], batch_id: str) -> bool:
        return self._get_queued_worker_runtime().pipeline_batch_release_ready(pp_stage_id, batch_id)

    def pipeline_batch_release_ready_all_ranks(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
    ) -> bool:
        return self._get_queued_worker_runtime().pipeline_batch_release_ready_all_ranks(pp_stage_id, batch_id)

    def finalize_pipeline_batch(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
        output_rank: int | None = None,
    ) -> str | None:
        return self._get_queued_worker_runtime().finalize_pipeline_batch(pp_stage_id, batch_id, output_rank)

    def poll_pipeline_finalization(self, batch_id: str) -> BatchRunnerOutput | None:
        return self._get_queued_worker_runtime().poll_pipeline_finalization(batch_id)

    def release_pipeline_batch_all_ranks(
        self,
        pp_stage_id: int | dict[int, int],
        batch_id: str,
    ) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().release_pipeline_batch_all_ranks(pp_stage_id, batch_id)

    def cleanup_finalized_pipeline_request(self, request_id: str) -> bool:
        return self._get_queued_worker_runtime().cleanup_finalized_pipeline_request(request_id)

    def cleanup_finalized_pipeline_request_all_ranks(self, request_id: str) -> list[bool]:
        return self._get_queued_worker_runtime().cleanup_finalized_pipeline_request_all_ranks(request_id)

    def poll_pipeline_events(self) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().poll_pipeline_events()

    def pipeline_stage_memory_budget_bytes(self) -> list[dict[str, int]]:
        return self._get_queued_worker_runtime().pipeline_stage_memory_budget_bytes()

    def poll_pipeline_events_all_ranks(self) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().poll_pipeline_events_all_ranks()

    def cancel_pipeline_requests(self, request_generations: Any) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().cancel_pipeline_requests(request_generations)

    def cancel_pipeline_requests_all_ranks(self, request_generations: Any) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().cancel_pipeline_requests_all_ranks(request_generations)

    def drain_pipeline(self, deadline: float | None = None) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().drain_pipeline(deadline)

    def drain_pipeline_all_ranks(self, deadline: float | None = None) -> list[PipelineEvent]:
        return self._get_queued_worker_runtime().drain_pipeline_all_ranks(deadline)

    def _activate_step_lora(self, scheduler_output: DiffusionSchedulerOutput) -> None:
        """Activate the LoRA adapter for the scheduled step batch.

        Newly scheduled requests register their (lora_request, lora_scale)
        in ``_step_lora_state`` so cached requests can resolve to the same
        adapter on later ticks. Finished requests are evicted. Batch
        homogeneity is enforced by the scheduler via ``StepBatchSamplingParamsKey``,
        so any scheduled request id resolves to the active LoRA identity.
        """
        for request_id in scheduler_output.finished_req_ids:
            self._step_lora_state.pop(request_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            sampling = new_req.req.sampling_params
            self._step_lora_state[new_req.request_id] = (
                sampling.lora_request,
                sampling.lora_scale,
            )

        if self.lora_manager is None:
            return

        lora_request: LoRARequest | None = None
        lora_scale = 1.0
        for request_id in scheduler_output.scheduled_request_ids:
            entry = self._step_lora_state.get(request_id)
            if entry is not None:
                lora_request, lora_scale = entry
                break

        try:
            self.lora_manager.set_active_adapter(lora_request, lora_scale)
        except Exception as exc:
            if lora_request is not None:
                raise
            logger.warning("LoRA activation skipped: %s", exc)

    def remove_lora(self, adapter_id: int) -> bool:
        if self.lora_manager is None:
            return False
        return self.lora_manager.remove_adapter(adapter_id)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # NOTE (Alex): We have not implemented the API routing
        # for the frontend server yet.
        if self.lora_manager is None:
            return False
        return self.lora_manager.add_adapter(lora_request)

    def submit_interaction(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Apply a midway interaction to an active stepwise request."""
        assert self.model_runner is not None, "Model runner not initialized"
        self.model_runner.submit_interaction(request_id, interaction)

    def list_loras(self) -> list[int]:
        if self.lora_manager is None:
            return []
        return self.lora_manager.list_adapters()

    def pin_lora(self, adapter_id: int) -> bool:
        if self.lora_manager is None:
            return False
        return self.lora_manager.pin_adapter(adapter_id)

    def sleep(self, level: int = 1) -> int:
        """
        Put the worker to sleep, offloading model weights.

        Args:
            level: Sleep level. Level 1 offloads weights, level 2 also saves buffers.
        """
        CuMemAllocator = _get_cumem_allocator_class()
        allocator = CuMemAllocator.get_instance()

        usage_before = allocator.get_current_usage()

        if level == 2 and self.model_runner is not None:
            self.model_runner.release_captured_graphs()
            logger.info(f"[Worker {self.rank}] CUDA Graphs cleared.")
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        free_mem_before = current_omni_platform.get_free_memory()

        # Level 1: Offload weights; Level 2: Total Discard
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator.sleep(offload_tags=offload_tags)

        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()

        free_mem_after = current_omni_platform.get_free_memory()
        try:
            total_mem = current_omni_platform.get_device_total_memory()
        except (NotImplementedError, AttributeError):
            total_mem = torch.cuda.get_device_properties(self.device).total_memory

        phys_freed_bytes = max(0, free_mem_after - free_mem_before)
        phys_used_bytes = total_mem - free_mem_after

        if usage_before > 0:
            logger.info(
                f"[Diffusion Worker {self.rank}] Sleep Level {level}: "
                f"physically freed {phys_freed_bytes / GiB_bytes:.2f} GiB, "
                f"{phys_used_bytes / GiB_bytes:.2f} GiB is still in use."
            )
        else:
            logger.info(f"[Worker {self.rank}] Sleep Level {level} completed (GPU was already empty).")
        logger.info(f"[Worker {self.rank}] Memory usage before sleep: {usage_before / GiB_bytes:.2f} GiB.")
        return usage_before

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode.

        Re-activates the memory allocator for the specified tags and restores
        model buffers from CPU back to GPU if they were saved during Level 2 sleep.

        Args:
            tags: List of memory pool tags to re-activate (e.g., ["weights"]
                  to match Level 1 sleep). If None, all pools are re-activated.
        """
        CuMemAllocator = _get_cumem_allocator_class()
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        if self.model_runner is not None and (tags is None or "kv_cache" in tags):
            self.model_runner.refresh_diffusion_kv_block_table_layout()
        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
            logger.info(f"[Worker {self.rank}] Buffers restored from CPU.")
        logger.info(f"[Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK | None:
        from vllm_omni.platforms import current_omni_platform

        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[Worker {self.rank}] Handshake Received: Task {task.task_id}")

            current_omni_platform.synchronize()
            free_before = current_omni_platform.get_free_memory(self.device)
            allocator_freed = self.sleep(level=task.level)
            current_omni_platform.synchronize()
            free_after = current_omni_platform.get_free_memory(self.device)
            phys_freed = max(0, free_after - free_before)
            real_freed = max(int(allocator_freed), phys_freed)
            logger.info(f"[Worker {self.rank}] Preparing ACK: freed_bytes={real_freed / GiB_bytes:.2f} GiB.")

            # Ensure all ranks have completed sleep before measuring memory and sending ACK
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(real_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                real_freed = int(t_freed.item())

            if self.rank != 0:
                return None

            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            residual_gib = (total_mem - free_after) / GiB_bytes
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=self.stage_id,
                rank=self.rank,
                freed_bytes=real_freed,
                # return RL need metadata
                metadata={
                    "source": f"Platform_{current_omni_platform.get_device_name()}",
                    "total_freed_gib": f"{real_freed / GiB_bytes:.2f}",
                    "allocator_freed_gib": f"{allocator_freed / GiB_bytes:.2f}",
                    "physical_freed_gib": f"{phys_freed / GiB_bytes:.2f}",
                    "rank_residual_gib": f"{residual_gib:.2f}",
                },
            )
            logger.info(f"[Worker {self.rank}] ACK emitted. Freed {real_freed / GiB_bytes:.2f} GiB.")
            return ack
        except Exception as e:
            logger.error(f"Sleep failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK | None:
        from vllm_omni.platforms import current_omni_platform

        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            logger.info(f"[Worker {self.rank}] Responding to Wake-up Task: {task.task_id}")
            self.wake_up(tags=task.tags)

            logger.info(f"[Worker {self.rank}] wake_up logic finished, entering barrier...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            current_omni_platform.synchronize()
            free_now = current_omni_platform.get_free_memory(self.device)
            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            current_used_gib = (total_mem - free_now) / (1024**3)

            if self.rank != 0:
                return None
            logger.info(f"[Worker {self.rank}] PASSED barrier, about to return to loop.")

            return OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=self.stage_id,
                rank=self.rank,
                metadata={
                    "state": "WARM",
                    "source": f"Platform_{current_omni_platform.get_device_name()}",
                    "current_vram_gib": f"{current_used_gib:.2f}",
                },
            )
        except Exception as e:
            logger.error(f"Wake-up failed on Rank {self.rank}: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get memory pool context for sleep mode support."""
        is_sleep_enabled = getattr(self.od_config, "enable_sleep_mode", False)
        if is_sleep_enabled:
            current_omni_platform.synchronize()
            gc.collect()
            CuMemAllocator = _get_cumem_allocator_class()
            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            logger.info(f"[Worker {self.rank}] Activating Diffusion CuMem pool for tag: {tag}")
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup distributed environment."""
        try:
            self._get_pipeline_finalization_state().shutdown()
            if self.model_runner is not None:
                mgr = getattr(self.model_runner, "kv_transfer_manager", None)
                try:
                    offload_backend = getattr(self.model_runner, "offload_backend", None)
                    if offload_backend is not None:
                        offload_backend.disable()
                finally:
                    if mgr is not None:
                        mgr.shutdown_prefetch()
        finally:
            try:
                shutdown_kv_connector()
            finally:
                try:
                    a2a_permute = sys.modules.get("vllm_omni.diffusion.distributed.a2a_permute")
                    if a2a_permute is not None:
                        a2a_permute.clear_a2a_permute_workspaces()
                except Exception:
                    logger.exception("Failed to release fused Ulysses symmetric-memory workspaces")
                finally:
                    destroy_distributed_env()


class CustomPipelineWorkerExtension:
    def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]) -> None:
        """
        Re-initialize the pipeline with custom arguments.

        Args:
            custom_pipeline_args: Dictionary of arguments for custom pipeline initialization
        """

        # Clean up old pipeline
        if self.model_runner.pipeline is not None:
            del self.model_runner.pipeline
            gc.collect()
            torch.accelerator.empty_cache()

        # Get custom pipeline class name
        custom_pipeline_name = custom_pipeline_args["pipeline_class"]

        # Use the DiffusionWorker's load_model method which handles the forward context
        self.load_model(
            load_format="custom_pipeline",
            custom_pipeline_name=custom_pipeline_name,
        )
        self.init_lora_manager()


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
        wake_event: mp.Event,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        self.od_config = od_config
        self.gpu_id = gpu_id
        self.wake_event = wake_event

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize MessageQueue reader from handle
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        self.result_mq = None
        self.result_mq_handle = None

        # Each worker creates its own result MessageQueue (as writer).
        # The executor creates one reader per worker to collect responses.
        # This supports DP multi-concurrency where all ranks reply independently.
        self.result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
        self.result_mq_handle = self.result_mq.export_handle()
        logger.info(f"Worker {gpu_id} created result MessageQueue")

        assert od_config.master_port is not None

        # Create worker using WorkerWrapperBase for extension support
        self.worker = self._create_worker(gpu_id, od_config, worker_extension_cls, custom_pipeline_args)
        self._running = True

        self._async_output_queue: queue.Queue | None = None
        self._async_output_thread: threading.Thread | None = None
        self._async_output_done = threading.Condition()
        self._async_output_pending = 0
        # MessageQueue.acquire_write() is single-writer: it reads current_idx,
        # writes that block, then advances the index. The async output thread
        # enqueues OUTPUT_READY while the main loop enqueues COMPUTE_DONE, so
        # unsynchronized writers can target the same block and drop a message.
        self._result_mq_lock = threading.Lock()
        self._stage_engine: PipelineStageEngine | None = None
        self._pipeline_prepare_executor: ThreadPoolExecutor | None = None
        parallel = self.od_config.parallel_config
        if (
            self.od_config.mode == "queued"
            and self.od_config.step_execution
            and parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 2
            and parallel.tensor_parallel_size == 1
            and parallel.sequence_parallel_size == 1
            and parallel.cfg_parallel_size == 1
        ):
            worker_device = getattr(self.worker.worker, "device", None)
            self._stage_engine = PipelineStageEngine(
                worker=self.worker,
                worker_id=gpu_id,
                device=worker_device,
                publish_update=self._publish_pipeline_update,
            )
            self.worker.worker._pipeline_stage_engine_wake = self._stage_engine.notify_progress
            self._pipeline_prepare_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"DiffusionPipelinePrep-rank{gpu_id}",
            )
        if not self.od_config.step_execution or self._stage_engine is not None:
            self._async_output_queue = queue.Queue()
            self._async_output_thread = threading.Thread(
                target=self._async_output_loop,
                daemon=True,
                name="DiffusionAsyncOutput",
            )
            self._async_output_thread.start()

    @staticmethod
    def _generate_async_output_id() -> str:
        return uuid.uuid4().hex

    def _create_worker(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        worker_extension_cls: str | None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> "WorkerWrapperBase":
        """Create a worker instance. Override in subclasses for different worker types."""
        worker_cls_path = current_omni_platform.get_diffusion_worker_cls()
        base_worker_class = resolve_obj_by_qualname(worker_cls_path)
        wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            od_config=od_config,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
            base_worker_class=base_worker_class,
        )
        return wrapper

    def _enqueue_result(self, msg: Any) -> None:
        """Serialize writes to the single-writer result queue."""
        with self._result_mq_lock:
            self.result_mq.enqueue(msg)

    def _publish_pipeline_update(self, update: PipelineWorkerUpdate) -> None:
        finalization_updates = tuple(
            PipelineFinalizationUpdate(batch_id=finalization.batch_id, error=finalization.error)
            for finalization in update.finalizations
        )
        for finalization in update.finalizations:
            if finalization.error is not None:
                self._enqueue_result(
                    AsyncDiffusionOutput(
                        kind=AsyncOutputKind.PIPELINE_FINALIZED,
                        async_output_id=finalization.batch_id,
                        error=finalization.error,
                    )
                )
            elif finalization.output is not None:
                self._queue_pipeline_finalization_output(
                    finalization.batch_id,
                    finalization.output,
                    finalization.device_event,
                )

        if update.error is not None:
            self._enqueue_result(update)
            return
        if update.progress is None:
            return
        if not (
            update.progress.offers
            or update.progress.completions
            or update.progress.readiness
            or update.events
            or update.error
            or finalization_updates
        ):
            return
        self._enqueue_result(
            PipelineWorkerUpdate(
                worker_id=update.worker_id,
                progress=update.progress,
                events=update.events,
                finalizations=finalization_updates,
                error=update.error,
            )
        )

    def _queue_pipeline_finalization_output(self, batch_id: str, output: Any, gpu_event: Any | None) -> None:
        if self._async_output_queue is None:
            raise RuntimeError("Queued pipeline finalization output queue is not initialized")
        if gpu_event is None:
            gpu_event = current_omni_platform.record_device_event()
        with self._async_output_done:
            self._async_output_pending += 1
        self._async_output_queue.put((output, batch_id, gpu_event, AsyncOutputKind.PIPELINE_FINALIZED))

    def _return_result(self, output: Any, rpc_id: str | None = None) -> None:
        """Reply to client, only on rank 0."""
        if self.result_mq is None:
            return
        if isinstance(output, OmniACK):
            self._enqueue_result(output)
            return

        # Async path: enqueue compute_done immediately, bg thread does D2H+SHM.
        if not self.od_config.step_execution and isinstance(output, (DiffusionOutput, BatchRunnerOutput)):
            async_output_id = WorkerProc._generate_async_output_id()
            gpu_event = current_omni_platform.record_device_event()
            with self._async_output_done:
                self._async_output_pending += 1
            self._async_output_queue.put((output, async_output_id, gpu_event))
            msg = AsyncDiffusionOutput(
                kind=AsyncOutputKind.COMPUTE_DONE,
                rpc_id=rpc_id,
                async_output_id=async_output_id,
            )
            self._enqueue_result(msg)
            return

        # Sync path (original, or async fallback).
        try:
            pack_diffusion_output_shm(output)
        except (TypeError, ValueError):
            raise
        except Exception as e:
            # Typed media that fails to pack is left unprepared/on device (the
            # pack is failure-atomic and does not mutate it), so enqueueing it
            # would ship a broken payload. Re-raise memory/packing failures here
            # instead of swallowing them for typed media.
            if payload_carries_typed_media(output):
                raise
            if hasattr(output, "output"):
                logger.warning("SHM pack failed for model output: %s", e)
        self._enqueue_result(output)

    def _async_output_loop(self):
        """Background thread: D2H + SHM packing for async diffusion output.

        Uses a side stream so the D2H transfer does not block the default
        stream where the next forward runs.
        """
        device = torch.device(torch.accelerator.current_accelerator().type, self.gpu_id)
        d2h_stream = torch.Stream(device=device)
        while True:
            item = self._async_output_queue.get()
            if item is None:
                break
            if len(item) == 3:
                output, async_output_id, gpu_event = item
                output_kind = AsyncOutputKind.OUTPUT_READY
            else:
                output, async_output_id, gpu_event, output_kind = item
            try:
                # Cross-stream ordering: wait for default stream to finish
                # writing the output tensors before the side stream reads.
                if gpu_event is not None:
                    d2h_stream.wait_event(gpu_event)
                pack_diffusion_output_shm(output, d2h_stream=d2h_stream)
                d2h_stream.synchronize()

                self._enqueue_result(
                    AsyncDiffusionOutput(
                        kind=output_kind,
                        async_output_id=async_output_id,
                        output=output if output_kind is AsyncOutputKind.OUTPUT_READY else None,
                        result=output if output_kind is AsyncOutputKind.PIPELINE_FINALIZED else None,
                    )
                )
            except Exception:
                logger.exception(
                    "Async output packing failed for id '%s'; sending error",
                    async_output_id,
                )
                self._enqueue_result(
                    AsyncDiffusionOutput(
                        kind=output_kind,
                        async_output_id=async_output_id,
                        error="Background D2H/SHM packing failed",
                    )
                )
            finally:
                with self._async_output_done:
                    # Clamped: only explicit output-queue submissions are counted.
                    self._async_output_pending = max(0, self._async_output_pending - 1)
                    self._async_output_done.notify_all()

    def drain_async_outputs(self, timeout: float = _ASYNC_OUTPUT_DRAIN_TIMEOUT_S) -> bool:
        """Block until background D2H/SHM packing has no work left.

        Returns False if outputs are still in flight when ``timeout`` expires.
        """
        with self._async_output_done:
            if self._async_output_pending == 0:
                return True
            drained = self._async_output_done.wait_for(lambda: self._async_output_pending == 0, timeout=timeout)
            pending = self._async_output_pending
        if not drained:
            logger.warning(
                "Worker %d: %d async output(s) still in flight after %.1fs; "
                "releasing device memory now may drop async output messages",
                self.gpu_id,
                pending,
                timeout,
            )
        return drained

    def shutdown(self) -> None:
        """Stop background work and release worker-owned IPC resources."""
        self._running = False

        if self._pipeline_prepare_executor is not None:
            self._pipeline_prepare_executor.shutdown(wait=True, cancel_futures=True)
            self._pipeline_prepare_executor = None

        if self._stage_engine is not None:
            self._stage_engine.shutdown()
            self._stage_engine = None

        if self._async_output_queue is not None:
            self._async_output_queue.put(None)
        if self._async_output_thread is not None:
            self._async_output_thread.join(timeout=_ASYNC_OUTPUT_THREAD_JOIN_TIMEOUT_S)
            if self._async_output_thread.is_alive():
                logger.warning(
                    "Worker %d: Async output thread did not stop before shutdown",
                    self.gpu_id,
                )

        try:
            self.worker.shutdown()
        finally:
            if self.mq is not None:
                self.mq.shutdown()
                self.mq = None

            # The worker creates this queue's shared-memory ring buffer.
            # Dropping the final creator reference invokes
            # ShmRingBuffer.__del__, which closes and unlinks it before
            # multiprocessing.resource_tracker exits.
            if self.result_mq is not None and (
                self._async_output_thread is None or not self._async_output_thread.is_alive()
            ):
                result_mq = self.result_mq
                self.result_mq = None
                result_mq.shutdown()
                del result_mq
                gc.collect()

    def _gather_rpc_rank_statuses(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        if not torch.distributed.is_initialized():
            return [status]

        control_group = get_world_group().cpu_group
        world_size = torch.distributed.get_world_size(group=control_group)
        statuses: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(statuses, status, group=control_group)
        missing_ranks = [rank for rank, rank_status in enumerate(statuses) if rank_status is None]
        if missing_ranks:
            logger.warning("RPC rank status gather returned missing entries for ranks: %s", missing_ranks)
        return [s for s in statuses if s is not None]

    def _execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""
        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)
        collect_rank_status = rpc_request.get("collect_rank_status", False)
        reply_all_ranks = rpc_request.get("reply_all_ranks", False)
        wave_id = rpc_request.get("wave_id")

        if collect_rank_status and not exec_all_ranks:
            raise ValueError("collect_rank_status requires exec_all_ranks=True so all ranks enter the status gather")
        if reply_all_ranks and not exec_all_ranks:
            raise ValueError("reply_all_ranks requires exec_all_ranks=True")

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        # For DP multi-concurrency (output_rank=None), only the primary rank
        # within each DP replica should reply.  This prevents SP/TP/CFG/PP
        # ranks from enqueuing extra replies that the executor doesn't drain.
        if reply_all_ranks:
            should_reply = self.result_mq is not None
        elif output_rank is None and exec_all_ranks:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            from vllm_omni.diffusion.distributed.parallel_state import (
                get_classifier_free_guidance_rank,
                get_pipeline_parallel_rank,
                get_sequence_parallel_rank,
            )

            is_primary_in_replica = (
                get_sequence_parallel_rank() == 0
                and get_classifier_free_guidance_rank() == 0
                and get_tensor_model_parallel_rank() == 0
                and get_pipeline_parallel_rank() == 0
            )
            should_reply = is_primary_in_replica and self.result_mq is not None
        else:
            should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None

        if not should_execute:
            return None, False

        result = None
        status: dict[str, Any] = {
            "rank": self.gpu_id,
            "ok": True,
            "error": None,
            "error_type": None,
            "traceback": None,
            "bool_result": None,
        }

        try:
            if method in _MEMORY_RELEASING_METHODS:
                self.drain_async_outputs()
            # Use execute_method from WorkerWrapperBase for consistent method resolution
            pipeline = getattr(getattr(self.worker, "worker", None), "pipeline", None)
            profiler_enabled = bool(getattr(pipeline, "enable_diffusion_pipeline_profiler", False))
            stage_engine = getattr(self, "_stage_engine", None)
            preparation_executor = getattr(self, "_pipeline_prepare_executor", None)
            if stage_engine is not None and method in _PIPELINE_PREPARATION_METHODS and not profiler_enabled:
                if preparation_executor is None:
                    raise RuntimeError("Queued pipeline preparation executor is not initialized")

                def prepare_pipeline_request() -> Any:
                    device = getattr(getattr(self.worker, "worker", None), "device", None)
                    if device is not None:
                        current_omni_platform.set_device(device)
                    return self.worker.execute_method(method, *args, **kwargs)

                result = preparation_executor.submit(prepare_pipeline_request).result()
            elif stage_engine is not None and method in {
                "start_pipeline_transfer",
                "admit_pipeline_batch",
                "admit_pipeline_batches",
            }:
                stage_engine.submit(
                    method,
                    *args,
                    publish_result_events=method.startswith("admit_pipeline"),
                    **kwargs,
                )
                result = True
            elif stage_engine is not None:
                result = stage_engine.call(method, *args, **kwargs)
            else:
                result = self.worker.execute_method(method, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            status.update(
                {
                    "ok": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
            if not collect_rank_status:
                raise
            _cleanup_after_execution_error(e)

        if isinstance(result, bool):
            status["bool_result"] = result

        if collect_rank_status:
            rank_statuses = self._gather_rpc_rank_statuses(status)
            if should_reply:
                return (
                    {
                        "type": DIFFUSION_RPC_RESULT_ENVELOPE,
                        "method": method,
                        "result": result,
                        "rank_statuses": rank_statuses,
                        "wave_id": wave_id,
                    },
                    True,
                )
            return None, False

        if reply_all_ranks:
            return (
                {
                    "rank_local_rpc": True,
                    "worker_id": self.gpu_id,
                    "status": "ok",
                    "result": result,
                    "wave_id": wave_id,
                },
                should_reply,
            )

        if isinstance(result, dict) and wave_id is not None:
            result["wave_id"] = wave_id
        if not should_reply:
            # A rank that will not reply must not hand the result back: the busy
            # loop binds it to a local that stays alive until the next request
            # overwrites it, so a device-resident output -- for diffusion, an
            # entire decoded video -- would occupy accelerator memory for the
            # whole idle period on every rank that did not produce the reply.
            # The `collect_rank_status` branch above already returns None here.
            return None, False
        return result, should_reply

    def recv_message(self) -> Any:
        """Receive one complete broadcast message without dropping overflow data."""
        return self.mq.dequeue(indefinite=True)

    def _worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers."""
        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            msg = None
            try:
                msg = self.recv_message()
            except Exception:
                if self.wake_event and self.wake_event.is_set():
                    self.wake_event.clear()
                    logger.info(f"Worker {self.gpu_id} caught OOB POKE, forcing wake-up sequence.")
                    msg = {"type": "wake_up", "task_id": "recovery-task", "tags": None}
                else:
                    continue
            if msg is None:
                continue

            if msg is None or len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            if isinstance(msg, dict) and msg.get("type") == "pipeline_wake":
                if self._stage_engine is not None:
                    self._stage_engine.notify_progress()
            elif isinstance(msg, dict) and msg.get("type") == "sleep":
                self.drain_async_outputs()
                task = OmniSleepTask(level=msg.get("level", 2), task_id=msg.get("task_id", "local"))
                ack = (
                    self._stage_engine.call("handle_sleep_task", task)
                    if self._stage_engine is not None
                    else self.worker.handle_sleep_task(task)
                )
                self._return_result(ack)
            elif isinstance(msg, dict) and msg.get("type") == "wake_up":
                task = OmniWakeTask(tags=msg.get("tags"), task_id=msg.get("task_id", "local"))
                ack = (
                    self._stage_engine.call("handle_wake_task", task)
                    if self._stage_engine is not None
                    else self.worker.handle_wake_task(task)
                )
                if self._stage_engine is not None:
                    self._stage_engine.notify_progress()
                self._return_result(ack)
            # Route message based on type
            elif isinstance(msg, dict) and msg.get("type") == "rpc":
                try:
                    rpc_id = msg.get("rpc_id")
                    result, should_reply = self._execute_rpc(msg)
                    if should_reply:
                        reply_start = time.perf_counter()
                        self._return_result(result, rpc_id=rpc_id)
                        if msg.get("method") == "poll_pipeline_finalization" and result is not None:
                            logger.info(
                                "Queued pipeline final decode reply packed batch=%s elapsed_ms=%.3f",
                                msg.get("args", (None,))[0],
                                (time.perf_counter() - reply_start) * 1000,
                            )
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    error = str(e)
                    _cleanup_after_execution_error(e)
                    # Apply the same reply gate as the success path so
                    # non-output ranks don't enqueue stale error replies
                    # that compete with the expected responder's message.
                    output_rank = msg.get("output_rank")
                    exec_all_ranks = msg.get("exec_all_ranks", False)
                    reply_all_ranks = msg.get("reply_all_ranks", False)
                    wave_id = msg.get("wave_id")
                    if self.result_mq is not None:
                        if rpc_id is not None:
                            # Async RPC: must complete the executor's pending
                            # future so collective_rpc() doesn't hang.
                            self._enqueue_result(
                                AsyncDiffusionOutput(
                                    kind=AsyncOutputKind.RPC_RESULT,
                                    rpc_id=rpc_id,
                                    error=error,
                                )
                            )
                        elif reply_all_ranks:
                            self._return_result(
                                {
                                    "rank_local_rpc": True,
                                    "worker_id": self.gpu_id,
                                    "status": "error",
                                    "error": error,
                                    "wave_id": wave_id,
                                }
                            )
                        elif output_rank is None and exec_all_ranks:
                            # DP multi-concurrency: primary ranks reply, tagged
                            from vllm.distributed.parallel_state import (
                                get_tensor_model_parallel_rank,
                            )

                            from vllm_omni.diffusion.distributed.parallel_state import (
                                get_classifier_free_guidance_rank,
                                get_data_parallel_rank,
                                get_pipeline_parallel_rank,
                                get_sequence_parallel_rank,
                            )

                            is_primary = (
                                get_sequence_parallel_rank() == 0
                                and get_classifier_free_guidance_rank() == 0
                                and get_tensor_model_parallel_rank() == 0
                                and get_pipeline_parallel_rank() == 0
                            )
                            if is_primary:
                                try:
                                    dp_rank = get_data_parallel_rank()
                                except Exception:
                                    dp_rank = self.gpu_id
                                self._return_result(
                                    {"status": "error", "error": error, "dp_rank": dp_rank, "wave_id": wave_id}
                                )
                        elif output_rank is None or output_rank == self.gpu_id:
                            # Normal RPC: only the expected rank replies
                            self._return_result({"status": "error", "error": error, "wave_id": wave_id})

            elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                continue

            else:
                # Handle direct generation requests.
                try:
                    if self._stage_engine is not None:
                        output = self._stage_engine.call("execute_model", msg, self.od_config)
                    else:
                        output = self.worker.execute_model(msg, self.od_config)
                except Exception as e:
                    logger.error(
                        f"Error executing forward in event loop: {e}",
                        exc_info=True,
                    )
                    output = DiffusionOutput.from_exception(e)
                    _cleanup_after_execution_error(e)

                try:
                    self._return_result(output)
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error sending reply: {e}")
                    continue

        logger.info("event loop terminated.")

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
        wake_event: mp.Event,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> None:
        """Worker initialization and execution loops."""
        from vllm_omni.plugins import load_omni_general_plugins

        shutdown_triggered = False

        def signal_handler(signum: int, frame) -> None:
            nonlocal shutdown_triggered
            if not shutdown_triggered:
                shutdown_triggered = True
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        set_death_signal(signal.SIGTERM)

        _setup_diffusion_worker_proc_title_and_log_prefix(
            enable_ep=od_config.parallel_config.enable_expert_parallel,
            use_hsdp=od_config.parallel_config.use_hsdp,
            hsdp_replicate_size=od_config.parallel_config.hsdp_replicate_size,
        )

        load_omni_general_plugins()
        worker_proc = None
        try:
            worker_proc = WorkerProc(
                od_config,
                gpu_id=rank,
                broadcast_handle=broadcast_handle,
                wake_event=wake_event,
                worker_extension_cls=worker_extension_cls,
                custom_pipeline_args=custom_pipeline_args,
            )
            logger.info(f"Worker {rank}: Scheduler loop started.")
            pipe_writer.send(
                {
                    "status": "ready",
                    "result_handle": worker_proc.result_mq_handle,
                }
            )
            worker_proc._worker_busy_loop()
        except SystemExit:
            logger.info("Worker %d: Shutdown signal received, starting cleanup.", rank)
            raise
        finally:
            if worker_proc is not None:
                try:
                    worker_proc.shutdown()
                except Exception as exc:
                    logger.warning("Worker %d: Shutdown encountered an error: %s", rank, exc)
                worker_proc.context.term()
            else:
                # In case of signal interrupting worker_proc initialization
                # where distributed env is initialized but worker_proc is still None
                destroy_distributed_env()

        logger.info("Worker %d: Shutdown complete.", rank)


class WorkerWrapperBase:
    """
    Wrapper base class that creates DiffusionWorker with optional worker_extension_cls support.
    This enables dynamic inheritance for DiffusionWorker to extend with custom functionality.
    """

    def __init__(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        base_worker_class: type = DiffusionWorker,
        wake_event: mp.Event = None,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        """
        Initialize WorkerWrapperBase with support for worker extensions.

        Args:
            gpu_id: GPU device ID
            od_config: OmniDiffusionConfig configuration
            worker_extension_cls: Optional qualified name of worker extension class
            custom_pipeline_args: Optional arguments for custom pipeline initialization
        """
        self.gpu_id = gpu_id
        self.od_config = od_config
        self.base_worker_class = base_worker_class
        self.worker_extension_cls = worker_extension_cls
        self.custom_pipeline_args = custom_pipeline_args

        # Prepare worker class with extension support
        worker_class = self._prepare_worker_class()

        # Create the actual worker instance
        # When custom_pipeline_args is provided, skip initial model loading
        # since re_init_pipeline will handle it. This avoids allocating memory
        # through CuMemAllocator twice, which causes assertion failures in
        # sleep mode.
        self.worker = worker_class(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
            skip_load_model=(self.custom_pipeline_args is not None),
        )
        self._execute_method_lock = threading.RLock()

        # Re-initialize pipeline with custom pipeline if provided
        if self.custom_pipeline_args is not None:
            self.worker.re_init_pipeline(self.custom_pipeline_args)

    def _prepare_worker_class(self) -> type:
        """
        Prepare the worker class with optional extension.
        Dynamically extends GPUWorker with worker_extension_cls if provided.

        Returns:
            The worker class (potentially extended)
        """
        worker_class = self.base_worker_class

        # If custom_pipeline_args is provided, use CustomPipelineWorkerExtension
        if self.custom_pipeline_args is not None:
            # Set worker_extension_cls to CustomPipelineWorkerExtension if not already set
            if self.worker_extension_cls is None:
                self.worker_extension_cls = CustomPipelineWorkerExtension

        if self.worker_extension_cls:
            if isinstance(self.worker_extension_cls, str):
                worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)
            else:
                worker_extension_cls = self.worker_extension_cls
            extended_calls = []

            if worker_extension_cls not in worker_class.__bases__:
                # Check for conflicts between worker and extension
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    if hasattr(worker_class, attr):
                        logger.warning(
                            f"Worker class {worker_class} already has attribute "
                            f"{attr}, which may conflict with worker extension "
                            f"class {worker_extension_cls}."
                        )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)

                # Dynamically inherit the worker extension class
                class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
                worker_class = type(class_name, (worker_extension_cls, worker_class), {})
                logger.info(
                    "Created extended worker class %s from %s for extended calls %s",
                    class_name,
                    worker_extension_cls,
                    extended_calls,
                )

        return worker_class

    def execute_model(
        self,
        req: OmniDiffusionRequest | list[NewRequestData],
        od_config: OmniDiffusionConfig,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        """
        Execute a forward pass.

        Args:
            req: Diffusion request.
            od_config: OmniDiffusionConfig configuration
            kv_prefetch_job: Optional next-request KV prefetch descriptor.

        Returns:
            DiffusionOutput with generated results
        """
        kwargs: dict[str, Any] = {"kv_prefetch_job": kv_prefetch_job}
        if diffusion_kv_metadata is not None:
            kwargs["diffusion_kv_metadata"] = diffusion_kv_metadata
        return self.worker.execute_model(req, od_config, **kwargs)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute one diffusion step."""
        return self.worker.execute_stepwise(scheduler_output)

    def sleep(self, level: int = 1) -> int:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache. Level 2 also saves buffers.
                Currently only support level 1 and level 2.

        Returns:
            Bytes held by the allocator before sleeping.
        """
        return self.worker.sleep(level)

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.

        Returns:
            True on success
        """
        return self.worker.wake_up(tags)

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK | None:
        return self.worker.handle_sleep_task(task)

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK | None:
        return self.worker.handle_wake_task(task)

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        return self.worker.shutdown()

    def execute_method(self, method: str | bytes, *args, **kwargs) -> Any:
        """
        Execute a method on the worker.

        Args:
            method: Method name (str) or serialized callable (bytes)

        Returns:
            Result of the method execution (type depends on the method)

        Raises:
            Exception: If method execution fails
        """
        try:
            # Method resolution order:
            # 1. If method is defined in this class, it will be called directly
            # 2. Otherwise, since we define `__getattr__` and redirect attribute
            #    query to `self.worker`, the method will be called on the worker
            assert isinstance(method, str), "Method must be str"
            with self._execute_method_lock:
                func = getattr(self.worker, method)
                return func(*args, **kwargs)

        except Exception as e:
            msg = f"Error executing method {method!r}. This might cause issues in distributed execution."
            logger.exception(msg)
            raise e

    def __getattr__(self, attr: str):
        """Delegate attribute access to the wrapped worker."""
        return getattr(self.worker, attr)
