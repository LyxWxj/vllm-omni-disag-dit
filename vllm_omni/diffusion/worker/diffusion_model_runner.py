# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import copy
import gc
import math
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes, format_gib
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend, cache_summary
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.cancellation import check_request_cancellation, request_cancellation_scope
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionMetadata,
    DiffusionPagedAttentionRow,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_pp_group,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.types import InteractionPayload
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportsInteractionApply,
    adopt_request_scoped_cache_dit,
    is_request_scoped_cache_dit_enabled,
    supports_interaction_apply,
    supports_request_pipeline,
    supports_step_execution,
)
from vllm_omni.diffusion.offloader import enable_offload_backend
from vllm_omni.diffusion.offloader.config import (
    TEXT_ENCODER_COMPONENT,
    OffloadStrategy,
    offload_enabled,
    resolve_offload,
    resolve_offload_strategy,
)
from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.request_pipeline import (
    RequestPipelineBatchState,
    RequestPipelineTransport,
    build_request_pipeline_tasks,
    trace_request_pipeline_event,
)
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
    validate_new_request_data_identity,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.stage_payload import DiffusionStagePayloadMixin
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    RunnerOutput,
    StepRequestState,
    attach_stage_durations,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniInteractionPrompt

logger = init_logger(__name__)


def _dit_any_rank_failed(local_failed: bool) -> bool:
    """All-reduce a per-request failure flag across the DiT process group.

    Every DiT rank must reach this point in lockstep; the caller wraps
    ``pipeline.prepare_encode`` so both the success and the exception paths
    call the helper. In single-rank execution this collapses to the local
    flag with no collectives issued.
    """
    if not torch.distributed.is_initialized():
        return local_failed
    try:
        from vllm_omni.diffusion.distributed import parallel_state

        get_dit_group = getattr(parallel_state, "get_dit_group", None)
        group = get_dit_group() if get_dit_group is not None else None
    except (AssertionError, ImportError):
        group = None
    if group is None:
        return local_failed
    device = current_omni_platform.device_type if current_omni_platform.is_available() else "cpu"
    signal = torch.tensor(1 if local_failed else 0, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(signal, op=torch.distributed.ReduceOp.MAX, group=group)
    return bool(signal.item())


def _normalize_pipeline_outputs(
    outputs: object,
    *,
    expected_count: int,
    allow_single_output: bool,
    pipeline_name: str,
) -> list[DiffusionOutput]:
    if isinstance(outputs, DiffusionOutput):
        if allow_single_output and expected_count == 1:
            return [outputs]
        raise RuntimeError(
            f"{pipeline_name}.forward returned a single DiffusionOutput; "
            "request-batch forward must return list[DiffusionOutput]."
        )

    if not isinstance(outputs, list):
        raise RuntimeError(
            f"{pipeline_name}.forward returned {type(outputs).__name__}; "
            "expected DiffusionOutput or list[DiffusionOutput]."
        )

    if len(outputs) != expected_count:
        raise RuntimeError(
            f"{pipeline_name}.forward returned {len(outputs)} outputs for {expected_count} requests; "
            "expected exactly one DiffusionOutput per request."
        )

    bad_index = next((idx for idx, output in enumerate(outputs) if not isinstance(output, DiffusionOutput)), None)
    if bad_index is not None:
        raise RuntimeError(
            f"{pipeline_name}.forward returned list item {bad_index} with type "
            f"{type(outputs[bad_index]).__name__}; expected DiffusionOutput."
        )

    return outputs


class DiffusionModelRunner(DiffusionStagePayloadMixin):
    """
    Model runner that handles model loading and execution for diffusion models.

    This class follows the AR pattern where the Runner handles all model-related
    operations including loading, compilation, offloading, caching, and execution.
    The Worker only handles infrastructure (device, distributed env).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ):
        """
        Initialize the diffusion model runner.

        Args:
            vllm_config: vLLM configuration.
            od_config: OmniDiffusion configuration.
            device: The device to run on.
        """
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline: Any | None = None
        self.cache_backend: Any | None = None
        self.offload_backend: Any | None = None
        self.prompt_embed_cache: Any | None = None
        self.input_batch: InputBatch | None = None
        self._interaction_coordinator: InteractionCoordinator | None = None
        self.model_memory_usage = 0
        self.diffusion_kv_backend = DiffusionKVModelRunnerBackend(
            vllm_config=vllm_config,
            od_config=od_config,
            device=device,
        )
        # Compatibility view for callers that inspect the installed config;
        # physical ownership remains in ``diffusion_kv_backend``.
        self.kv_cache_config: KVCacheConfig | None = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, StepRequestState] = {}

        # Initialize KV cache manager for connector management.
        payload_transfer_manager = OmniKVTransferManager.from_od_config(od_config)
        self.kv_transfer_manager = (
            payload_transfer_manager if getattr(od_config, "kv_transfer_config", None) is None else None
        )
        self.init_omni_connectors(od_config, payload_transfer_manager, synchronous=True)  # type: ignore[arg-type]
        self._kv_connector = None
        from vllm_omni.diffusion.diffusion_kv.kv_connector import KVReceiveProgress, native_prefetch_enabled

        self._kv_receive_progress = KVReceiveProgress() if native_prefetch_enabled(od_config) else None

        # Prefetch covers TP / SP / CFG-Parallel / HSDP.  Disabled when a CFG
        # companion KV collector is set (that KV is not backgrounded).
        has_cfg_companion_kv = getattr(od_config, "cfg_kv_collect_func", None) is not None

        self._kv_prefetch_enabled = (
            self.kv_transfer_manager is not None
            and bool(self.kv_transfer_manager.config.enable_kv_async_prefetch)
            and not has_cfg_companion_kv
            and self.kv_transfer_manager.config.need_recv_cache
        )

    @property
    def _target_device(self) -> torch.device | None:
        return getattr(self.pipeline, "device", None)

    def _validate_diffusion_kv_metadata(
        self,
        *,
        request_id: str,
        metadata: DiffusionKVMetadata | None,
    ) -> None:
        cache_mode = getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        if cache_mode is DiffusionKVCacheMode.PAGED_SCHEDULER and metadata is None:
            raise ValueError(f"paged_scheduler request {request_id!r} requires Diffusion KV metadata")
        if cache_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER and metadata is not None:
            raise ValueError(f"{cache_mode.value} request {request_id!r} must not carry Diffusion KV metadata")

        if metadata is not None and metadata.request_id != request_id:
            raise ValueError(
                f"Diffusion KV metadata request mismatch: expected={request_id!r}, got={metadata.request_id!r}"
            )

    def _compile_transformer(self, attr_name: str) -> None:
        """Compile a transformer attribute on the pipeline with torch.compile."""
        model = getattr(self.pipeline, attr_name, None)
        if model is None:
            return

        compile_kwargs: dict[str, Any] = {"dynamic": self.od_config.diffusion_compile_dynamic}
        if getattr(model, "enable_cuda_graph_decode", False):
            # Decode-graph models still compile their blocks: graph capture
            # records the compiled (fused) kernels. Scope inductor cudagraphs
            # to this compile so the two graph layers never stack, without
            # changing torch.compile for every other model in the process.
            compile_kwargs["options"] = {
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            }
            logger.info("Model runner: %s combines CUDA graph decode with torch.compile.", attr_name)

        compile_granularity = self.od_config.diffusion_compile_granularity
        try:
            if compile_granularity == "full":
                model.compile(**compile_kwargs)
                compiled_model = model
            else:
                compiled_model = regionally_compile(model, **compile_kwargs)
            setattr(self.pipeline, attr_name, compiled_model)
        except Exception as e:
            logger.warning(
                "Model runner: %s torch.compile setup for %s failed before activation: %s. "
                "Continuing with the uncompiled model; lazy compilation errors can still "
                "surface on the first request.",
                compile_granularity,
                attr_name,
                e,
            )
            return

        logger.info(
            "Model runner: %s configured for lazy %s torch.compile with dynamic=%s; "
            "compilation errors may surface on the first request.",
            attr_name,
            compile_granularity,
            compile_kwargs["dynamic"],
        )

    def load_model(
        self,
        memory_pool_context_fn: Callable[[str], AbstractContextManager[Any]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
        initialize_lora: Callable[[], None] | None = None,
    ) -> None:
        """
        Launch the diffusion pipeline, applying compilation, offloading, and caching.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
            load_format: Format for loading model weights. Supported formats:
                - "default" (default): Automatically detect and use the default format based on configuration
                - "custom_pipeline": Init model from a custom pipeline class specified by `custom_pipeline_name`
                - "dummy": Skip actual weight loading, useful for testing and custom pipelines that
                    don't require default weights.
            custom_pipeline_name: Optional custom pipeline class name to use.
        """

        if load_format == "dummy":
            return

        # Resolve environment overrides before model loading. Rank-local
        # prompt-cache hits cannot safely skip text-encoder collectives.
        enable_pec, pec_size = resolve_prompt_embed_cache_config(
            enable=getattr(self.od_config, "enable_prompt_embed_cache", False),
            max_size=getattr(self.od_config, "prompt_embed_cache_size", 32),
        )
        resolved_offload = resolve_offload(self.od_config)
        dp_size = int(getattr(getattr(self.od_config, "parallel_config", None), "data_parallel_size", 1))
        if (
            enable_pec
            and dp_size > 1
            and resolved_offload.offloads(TEXT_ENCODER_COMPONENT)
            and resolved_offload.uses_allgather(TEXT_ENCODER_COMPONENT)
        ):
            raise ValueError(
                "Prompt embedding cache cannot be combined with text_encoder "
                "AllGather across data-parallel ranks; disable the cache or use "
                "rank-local text_encoder transfer."
            )

        current_omni_platform.init_diffusion_model_runner_runtime(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )

        load_device = "cpu" if offload_enabled(self.od_config) else str(self.device)

        def get_memory_context() -> AbstractContextManager[Any]:
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn("weights")
            return nullcontext()

        # Load model within forward context
        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        time_before_load = time.perf_counter()

        with get_memory_context():
            with DeviceMemoryProfiler() as m:
                self.pipeline = model_loader.load_model(
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )
                if initialize_lora is not None and getattr(self.pipeline, "requires_early_lora_init", False) is True:
                    initialize_lora()
        time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        self.model_memory_usage = int(m.consumed_memory)
        logger.info("Model runner: Model loaded successfully.")

        if self.od_config.streaming_output and not getattr(self.od_config, "step_execution", False):
            logger.warning("streaming_output=True requires step_execution=True; enabling step execution.")
            self.od_config.step_execution = True

        if getattr(self.od_config, "step_execution", False) and not self._supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing "
                "prepare_encode(), denoise_step(), step_scheduler(), and post_decode(); "
                f"{self.od_config.model_class_name} does not support that contract."
            )
        if (
            getattr(self.od_config, "step_execution", False)
            and getattr(self.od_config.parallel_config, "pipeline_parallel_size", 1) > 1
            and not getattr(self.pipeline, "supports_step_pipeline_parallel", False)
        ):
            raise ValueError(
                "step-execution pipeline parallelism currently requires a pipeline implementing "
                "the request-local step PP contract; "
                f"{self.od_config.model_class_name} does not support it."
            )
        if self.od_config.streaming_output and not self._supports_step_mode():
            raise ValueError(
                "streaming_output=True requires step execution support; "
                f"{self.od_config.model_class_name} does not support that contract."
            )
        if getattr(self.od_config, "request_pipeline_parallel", False) and not supports_request_pipeline(self.pipeline):
            raise ValueError(
                "request_pipeline_parallel=True requires a pipeline implementing the request-pipeline contract; "
                f"{self.od_config.model_class_name} does not support it."
            )

        # The offloader owns loader-plan handoff and startup recovery. The
        # runner only receives the pipeline/backend pair that is ready to use.
        self.pipeline, self.offload_backend = enable_offload_backend(
            self.od_config,
            self.pipeline,
            device=self.device,
        )

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                if hasattr(self.pipeline, "setup_compile"):
                    try:
                        self.pipeline.setup_compile()
                    except Exception as exc:
                        logger.warning(
                            "Model runner: setup_compile() failed (%s); running without compile.",
                            exc,
                        )
                else:
                    transformer_attrs = getattr(self.pipeline, "_dit_modules", None)
                    if not transformer_attrs:
                        transformer_attrs = ("transformer", "transformer_2")
                    for attr_name in transformer_attrs:
                        self._compile_transformer(attr_name)
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            if self.od_config.model_class_name in _NO_CACHE_ACCELERATION:
                logger.warning(
                    "Cache backend '%s' is not supported for %s; disabling cache acceleration.",
                    self.od_config.cache_backend,
                    self.od_config.model_class_name,
                )
                self.cache_backend = None
                self.od_config.cache_backend = None
            else:
                # Install configured cache capability once at startup. A model
                # may explicitly adopt the enabled Cache-DiT backend and then
                # own all later request-boundary enable/disable transitions.
                self.cache_backend.enable(self.pipeline)
                if isinstance(self.cache_backend, CacheDiTBackend) and adopt_request_scoped_cache_dit(
                    self.pipeline,
                    self.cache_backend,
                ):
                    logger.info(
                        "Pipeline %s owns request-scoped Cache-DiT transitions.",
                        type(self.pipeline).__name__,
                    )
                    self.cache_backend = None

        # Install prompt-embedding cache (transparent wrapper around
        # ``pipeline.encode_prompt``). Enabled via config or env var; a no-op
        # when the pipeline does not expose ``encode_prompt``.
        if enable_pec:
            self.prompt_embed_cache = install_prompt_embed_cache(
                self.pipeline,
                max_size=pec_size,
                enabled=True,
                model_tag=self.od_config.model_class_name,
            )

        self._interaction_coordinator = InteractionCoordinator.build(self.pipeline, self.od_config)
        if hasattr(self.pipeline, "_interaction_coordinator"):
            self.pipeline._interaction_coordinator = self._interaction_coordinator

        logger.info("Model runner: Initialization complete.")

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Collect native specs from cache-enabled loaded attention modules."""

        if self.od_config.diffusion_kv_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER:
            return {}
        if self.pipeline is None:
            raise RuntimeError("Model must be loaded before collecting Diffusion KV cache specs")

        cache_layers: dict[str, tuple[Attention, KVCacheSpec]] = {}
        for module_path, module in self.pipeline.named_modules():
            if not isinstance(module, Attention):
                continue
            spec = module.get_kv_cache_spec(self.vllm_config)
            if spec is not None:
                layer_name = module.prefix
                if not isinstance(layer_name, str) or not layer_name:
                    raise RuntimeError(
                        "Paged Diffusion Attention must expose a non-empty canonical prefix; "
                        f"module_path={module_path!r}"
                    )
                if layer_name in cache_layers:
                    raise RuntimeError(f"Duplicate canonical paged Diffusion Attention prefix {layer_name!r}")
                cache_layers[layer_name] = (module, spec)
        if not cache_layers:
            raise RuntimeError(
                "paged_scheduler Diffusion KV found no cache-enabled Attention modules "
                f"in {type(self.pipeline).__name__}"
            )
        return self.diffusion_kv_backend.register_kv_cache_layers(cache_layers)

    def set_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Physically initialize the Engine-generated rank-local config."""

        self.diffusion_kv_backend.initialize_kv_cache(kv_cache_config)
        self.kv_cache_config = self.diffusion_kv_backend.kv_cache_config
        if getattr(self.od_config, "kv_transfer_config", None) is not None:
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

            self._kv_connector = ActiveKVConnector(self.vllm_config, self.diffusion_kv_backend.kv_caches_by_layer)

    def prepare_kv_for_forward(self, scheduler_output: DiffusionSchedulerOutput):
        from vllm_omni.diffusion.diffusion_kv.kv_connector import wait_for_kv_load

        assert self.od_config.kv_transfer_config is not None
        timeout = self.od_config.kv_transfer_config.kv_connector_extra_config.get("transfer_timeout", 60.0)
        if self._kv_receive_progress is not None:
            return self._kv_receive_progress.prepare(self._kv_connector, scheduler_output, timeout)
        return wait_for_kv_load(self._kv_connector, scheduler_output, timeout)

    def install_diffusion_kv_metadata(self, metadata: DiffusionKVMetadata) -> bool:
        return self.diffusion_kv_backend.install_diffusion_kv_metadata(metadata)

    def get_diffusion_kv_row(
        self,
        request_id: str,
        sequence_id: int | None,
        context_id: str | None = None,
    ) -> int:
        return self.diffusion_kv_backend.get_diffusion_kv_row(request_id, sequence_id, context_id)

    def remove_diffusion_kv_requests(self, request_ids: Sequence[str | tuple[str, int]]) -> int:
        return self.diffusion_kv_backend.remove_diffusion_kv_requests(request_ids)

    def refresh_diffusion_kv_block_table_layout(self) -> None:
        self.diffusion_kv_backend.refresh_block_table_layout()

    def _build_paged_attention_metadata(
        self,
        metadata: list[DiffusionKVMetadata],
    ) -> DiffusionPagedAttentionMetadata:
        """Translate Scheduler sequence state into ordered request-level rows."""

        cfg_size = int(getattr(self.od_config.parallel_config, "cfg_parallel_size", 1) or 1)
        cfg_rank = get_classifier_free_guidance_rank() if cfg_size > 1 else None
        prefill_rows: list[DiffusionPagedAttentionRow] = []
        denoise_rows: list[DiffusionPagedAttentionRow] = []
        for request_metadata in metadata:
            sequences = request_metadata.sequences
            if cfg_rank is not None and len(sequences) > 1:
                if len(sequences) != cfg_size:
                    raise ValueError(
                        "Paged CFG parallel execution requires one Scheduler sequence per CFG rank: "
                        f"request={request_metadata.request_id!r}, cfg_size={cfg_size}, rows={len(sequences)}"
                    )
                sequences = (sequences[cfg_rank],)
            for sequence in sequences:
                active_seq_len = sequence.prefix_len + sequence.target_len
                if active_seq_len > sequence.seq_len:
                    raise ValueError(
                        "Paged denoise span exceeds its Scheduler allocation: "
                        f"request={request_metadata.request_id!r}, sequence={sequence.sequence_id}, "
                        f"active={active_seq_len}, allocated={sequence.seq_len}"
                    )
                # Imported AR KV and local hits have separate owners.
                kv_start_pos = (
                    sequence.num_computed_tokens
                    if getattr(self.od_config, "kv_transfer_config", None) is not None
                    else sequence.cached_prefix_len
                )
                prefill_rows.append(
                    DiffusionPagedAttentionRow(
                        request_id=request_metadata.request_id,
                        sequence_id=sequence.sequence_id,
                        kv_start_pos=kv_start_pos,
                        query_len=sequence.seq_len - kv_start_pos,
                        seq_len=sequence.seq_len,
                    )
                )
                denoise_rows.append(
                    DiffusionPagedAttentionRow(
                        request_id=request_metadata.request_id,
                        sequence_id=sequence.sequence_id,
                        query_len=sequence.target_len,
                        seq_len=active_seq_len,
                        kv_start_pos=sequence.prefix_len,
                    )
                )
        return DiffusionPagedAttentionMetadata(tuple(prefill_rows), tuple(denoise_rows))

    def clear_prompt_embed_cache(self) -> None:
        """Evict all cached text-encoder outputs (e.g. between training epochs).

        Kept primarily for extension purposes.
        """
        if self.prompt_embed_cache is not None:
            self.prompt_embed_cache.clear()

    def release_captured_graphs(self) -> None:
        """Drop every CUDA graph held for this model, wherever it is kept.

        Sleep level 2 discards the memory a capture was recorded against, so a
        graph that outlives it replays over freed storage. The runner owns
        compilation and execution resources, so it owns the release: a pipeline
        that captures graphs of its own implements ``release_captured_graphs``
        and is collected here, instead of every caller having to know which
        pipelines have one.
        """
        runners = getattr(self, "graph_runners", None)
        if runners is not None:
            runners.clear()
        release = getattr(self.pipeline, "release_captured_graphs", None)
        if callable(release):
            release()

    def get_prompt_embed_cache_stats(self) -> dict | None:
        """Return hit/miss statistics for the prompt-embedding cache, if enabled.

        Kept primarily for extension purposes.
        """
        if self.prompt_embed_cache is None:
            return None
        return self.prompt_embed_cache.stats()

    def _sample_peak_memory_mb(self) -> float:
        """Return peak reserved GPU memory for the current forward pass in MiB.

        Must be called immediately after the measured forward/step work, with
        reset_peak_memory_stats() called just before it, so the measurement
        reflects the current execution slice and not the global historical
        maximum.

        Uses max_memory_reserved (CUDA memory pool high-water mark) rather than
        max_memory_allocated so that allocator fragmentation is also visible.
        See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html

        Named ``_mb`` to match the public ``peak_memory_mb`` field.
        """
        peak_reserved_bytes = current_omni_platform.max_memory_reserved()
        peak_allocated_bytes = current_omni_platform.max_memory_allocated()

        pool_overhead_bytes = peak_reserved_bytes - peak_allocated_bytes

        peak_memory_mb = peak_reserved_bytes / (1024**2)

        logger.debug(
            "Peak GPU memory (this request): %s GiB reserved, %s GiB allocated, %s GiB pool overhead (%.1f%%)",
            format_gib(peak_reserved_bytes),
            format_gib(peak_allocated_bytes),
            format_gib(pool_overhead_bytes),
            pool_overhead_bytes / peak_reserved_bytes * 100 if peak_reserved_bytes > 0 else 0.0,
        )
        return peak_memory_mb

    def _prepare_request_for_forward(
        self,
        req: OmniDiffusionRequest,
        *,
        od_config: OmniDiffusionConfig,
        kv_prefetch_job: KVPrefetchJob | None = None,
        use_prefetch: bool = False,
    ) -> None:
        # Fetch upstream conditioning before anything else: the pipeline reads
        # it out of the prompt during the forward below.
        self._maybe_recv_stage_payload(req)

        if self.kv_transfer_manager is None:
            self._initialize_generator(req.sampling_params)
            return
        # Receive AR KV. Single-request execution can use the prefetch path:
        # consume prior-forward payload, sync-fallback on miss; request-batch
        # execution keeps the synchronous per-request receive path.
        kv_recv_t0 = time.perf_counter()
        if use_prefetch and self._kv_prefetch_enabled:
            self.kv_transfer_manager.consume_and_distribute_kv_cache(
                req,
                target_device=self._target_device,
            )
        else:
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                req,
                cfg_kv_collect_func=getattr(od_config, "cfg_kv_collect_func", None),
                target_device=self._target_device if use_prefetch else getattr(self.pipeline, "device", None),
            )
        kv_recv_ms = (time.perf_counter() - kv_recv_t0) * 1000
        req.kv_recv_ms = kv_recv_ms
        logger.debug("KV recv for %s %.1fms", req.request_id, kv_recv_ms)

        # Kick off the next request's prefetch (+ H2D) to overlap this forward.
        if use_prefetch and self._kv_prefetch_enabled and kv_prefetch_job is not None:
            self.kv_transfer_manager.start_prefetch(kv_prefetch_job, self._target_device)

        self._initialize_generator(req.sampling_params)

    def _initialize_generator(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        if sampling_params.generator is None and sampling_params.seed is not None:
            if sampling_params.generator_device is not None:
                gen_device = sampling_params.generator_device
            elif self.device.type == "cpu":
                gen_device = "cpu"
            else:
                gen_device = self.device
            sampling_params.generator = torch.Generator(device=gen_device).manual_seed(sampling_params.seed)

    def _refresh_cache_for_requests(
        self,
        reqs: list[OmniDiffusionRequest],
        *,
        od_config: OmniDiffusionConfig,
    ) -> None:
        first_req = reqs[0]
        if self.cache_backend is None or not self.cache_backend.is_enabled():
            return

        # Refresh cache context if needed. Batch admission groups requests by
        # RequestBatchSamplingParamsKey, so the first request's num_inference_steps applies
        # to the whole runner batch.
        num_inference_steps = first_req.sampling_params.num_inference_steps
        if num_inference_steps is None and first_req.sampling_params.timesteps is not None:
            num_inference_steps = len(first_req.sampling_params.timesteps)
        if num_inference_steps is None and first_req.sampling_params.sigmas is not None:
            num_inference_steps = len(first_req.sampling_params.sigmas)
        if num_inference_steps is None:
            num_inference_steps = getattr(self.pipeline, "default_num_inference_steps", None)
        if num_inference_steps is None and od_config.cache_backend in (
            "tea_cache",
            "sea_cache",
            "step_cache",
        ):
            # When num_inference_steps is None, some pipelines defer to their
            # own defaults. These backends use refresh to reset request state;
            # runtime step metadata is either unused or resolved in the
            # pipeline. Use the pipeline default when available to keep refresh
            # behavior aligned with single-request execution.
            num_inference_steps = getattr(self.pipeline, "num_inference_steps", 0) or 0

        if num_inference_steps is not None:
            self.cache_backend.refresh(self.pipeline, num_inference_steps)
        else:
            logger.warning(
                "Failed to refresh the diffusion transformer cache; backend %s "
                "currently requires num_inference_steps to be passed explicitly",
                od_config.cache_backend,
            )

    def _prepare_output_for_transport(
        self,
        output: DiffusionOutput,
        sampling_params: OmniDiffusionSamplingParams,
    ) -> DiffusionOutput:
        if output.media is not None:
            if output.output is not None:
                raise ValueError("DiffusionOutput cannot contain both media and legacy output")
            output.media = prepare_diffusion_media_for_transport(
                output.media,
                od_config=self.od_config,
                sampling_params=sampling_params,
            )
        return output

    @staticmethod
    def _mark_pp_video_output(output: DiffusionOutput, sampling: OmniDiffusionSamplingParams) -> None:
        if (
            isinstance(output.output, torch.Tensor)
            and output.output.ndim == 5
            and sampling.output_type != "latent"
            and sampling.num_frames > 1
        ):
            output.video_output_index = 0

    def _runner_output_from_outputs(
        self,
        reqs: list[OmniDiffusionRequest],
        outputs: list[DiffusionOutput],
    ) -> BatchRunnerOutput:
        for i in range(len(reqs)):
            # Carry the runner-measured KV-recv timing onto the output so the
            # engine's step_streaming can surface it as diffusion_kv_load_s.
            outputs[i].kv_recv_ms = reqs[i].kv_recv_ms
        return BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    request_id=reqs[i].request_id,
                    step_index=None,
                    finished=True,
                    result=outputs[i],
                )
                for i in range(len(reqs))
            ]
        )

    def _execute_request_list(
        self,
        reqs: list[OmniDiffusionRequest],
        *,
        od_config: OmniDiffusionConfig,
        allow_single_output: bool,
        require_request_batch_support: bool,
        kv_prefetch_job: KVPrefetchJob | None = None,
        record_name: str,
        record_output_peak_memory: bool = True,
        in_diffusion_kv_memory_profile: bool = False,
        diffusion_kv_metadata: list[DiffusionKVMetadata] | None = None,
    ) -> BatchRunnerOutput:
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not reqs:
            return BatchRunnerOutput.from_list([])
        for req in reqs:
            if req.prompt is None:
                raise ValueError("Cannot execute model with empty prompt")
        if require_request_batch_support and not getattr(self.pipeline, "supports_request_batch", False):
            raise RuntimeError(f"{type(self.pipeline).__name__} does not support request-batch forward.")

        # Use no_grad() for HSDP compatibility, inference_mode() otherwise for
        # better perf. HSDP2's fully_shard pre-forward hooks need tensor version
        # counters, which inference tensors do not track.
        use_hsdp = od_config.parallel_config.use_hsdp
        use_distributed_offload = resolve_offload_strategy(self.od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        grad_context = torch.no_grad() if (use_hsdp or use_distributed_offload) else torch.inference_mode()
        with grad_context:
            for req in reqs:
                self._prepare_request_for_forward(
                    req,
                    od_config=od_config,
                    kv_prefetch_job=kv_prefetch_job,
                    use_prefetch=allow_single_output,
                )

            self._refresh_cache_for_requests(reqs, od_config=od_config)

            batch = DiffusionRequestBatch(requests=reqs)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary and record_output_peak_memory:
                current_omni_platform.reset_peak_memory_stats()

            paged_kv_runtime = None
            paged_kv_context: AbstractContextManager[Any] = nullcontext()
            if diffusion_kv_metadata is not None:
                if len(diffusion_kv_metadata) != len(reqs):
                    raise ValueError(
                        "Diffusion KV metadata count must match the request batch: "
                        f"metadata={len(diffusion_kv_metadata)}, requests={len(reqs)}"
                    )
                native_kv_transfer = getattr(self.od_config, "kv_transfer_config", None) is not None
                if native_kv_transfer:
                    for req, metadata in zip(reqs, diffusion_kv_metadata, strict=True):
                        # This field is consumed by Hunyuan only for native
                        # AR->DiT transfer. Local prefix hits must still run
                        # their VAE/ViT conditioning path.
                        req.kv_computed_tokens = tuple(seq.num_computed_tokens for seq in metadata.sequences)
                paged_metadata = self._build_paged_attention_metadata(diffusion_kv_metadata)
                paged_kv_cached_prefix_len = 0
                if not native_kv_transfer:
                    cached_prefix_lens = {row.kv_start_pos for row in paged_metadata.prefill_rows}
                    if len(cached_prefix_lens) != 1:
                        raise ValueError(
                            "One paged request-level forward requires a uniform cached prefix boundary; "
                            f"got {sorted(cached_prefix_lens)}"
                        )
                    paged_kv_cached_prefix_len = next(iter(cached_prefix_lens))
                paged_kv_runtime, paged_kv_context = self.diffusion_kv_backend.activate_paged_attention_metadata(
                    paged_metadata
                )
                if is_primary:
                    # Trace the boundary actually passed to the model, not a
                    # speculative lookup. Useful for warm-cache regressions.
                    for request_metadata in diffusion_kv_metadata:
                        for sequence in request_metadata.sequences:
                            logger.debug(
                                "Diffusion prefix prefill: request_id=%s sequence_id=%d "
                                "cached_prefix_len=%d prefix_len=%d query_len=%d",
                                request_metadata.request_id,
                                sequence.sequence_id,
                                sequence.cached_prefix_len,
                                sequence.prefix_len,
                                sequence.seq_len - sequence.cached_prefix_len,
                            )
            else:
                paged_kv_cached_prefix_len = 0
            with (
                set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=od_config,
                    paged_kv_runtime=paged_kv_runtime,
                    paged_kv_cached_prefix_len=paged_kv_cached_prefix_len,
                    in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
                ),
                paged_kv_context,
                request_cancellation_scope(
                    [getattr(req, "cancellation_signal", None) for req in reqs],
                    enabled=getattr(self.pipeline, "supports_request_cancellation", False) is True,
                ),
            ):
                with record_function(record_name):
                    try:
                        check_request_cancellation()
                        run_stage = getattr(self.pipeline, "run_stage", None)
                        if callable(run_stage):
                            raw_outputs = run_stage(batch)
                        else:
                            raw_outputs = self.pipeline.forward(batch)
                        outputs = _normalize_pipeline_outputs(
                            raw_outputs,
                            expected_count=len(reqs),
                            allow_single_output=allow_single_output,
                            pipeline_name=type(self.pipeline).__name__,
                        )
                    except DiffusionRequestAbortedError as exc:
                        # The checkpoint aborts only a fully cancelled wave;
                        # a mixed batch must keep running for its live peers.
                        logger.info(
                            "Stopped cancelled diffusion request(s) %s at a model execution boundary",
                            [req.request_id for req in reqs],
                        )
                        outputs = [DiffusionOutput(aborted=True, abort_message=str(exc)) for _ in reqs]
                with record_function("prepare_output_for_transport"):
                    outputs = [
                        self._prepare_output_for_transport(output, req.sampling_params)
                        for req, output in zip(reqs, outputs, strict=True)
                    ]

            if is_primary and outputs and record_output_peak_memory:
                batch_peak_memory_mb = self._sample_peak_memory_mb()
                for output in outputs:
                    output.peak_memory_mb = max(output.peak_memory_mb, batch_peak_memory_mb)

            # Log prompt-embed cache activity; hits/misses accumulate across requests.
            prompt_embed_cache = getattr(self, "prompt_embed_cache", None)
            if is_primary and prompt_embed_cache is not None:
                logger.debug("prompt-embed cache: %s", prompt_embed_cache.stats())

            runner_cache_dit_enabled = self.cache_backend is not None and self.cache_backend.is_enabled()
            if (
                od_config.cache_backend == "cache_dit"
                and od_config.enable_cache_dit_summary
                and (runner_cache_dit_enabled or is_request_scoped_cache_dit_enabled(self.pipeline))
            ):
                cache_summary(self.pipeline, details=True)

        self._maybe_send_stage_payload(reqs, outputs)

        return self._runner_output_from_outputs(reqs, outputs)

    def _attach_stepwise_metadata(
        self,
        state: StepRequestState,
        output: DiffusionOutput,
    ) -> None:
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )
        attach_stage_durations(state, output)

        # In streaming output mode with interaction, acknowledge which events are handled in this chunk.
        meta = state.interaction_chunk_metadata
        state.interaction_chunk_metadata = None
        if meta is not None:
            output.started_event_ids = list(meta.started_event_ids)
            output.active_event_ids = list(meta.active_event_ids)
            output.completed_event_ids = list(meta.completed_event_ids)
            output.coalesced_event_ids = list(meta.coalesced_event_ids)

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            req: A diffusion request containing a list of prompts to process.

        Returns:
            DiffusionOutput with generated results.

        Note:
            We use torch.no_grad() for HSDP because HSDP2's fully_shard requires access
            to tensor version counters in pre_forward hooks, which inference tensors do
            not track. For non-HSDP inference, we use torch.inference_mode() for better
            performance.
        """
        self._validate_diffusion_kv_metadata(
            request_id=req.request_id,
            metadata=diffusion_kv_metadata,
        )
        installed_request = False
        if diffusion_kv_metadata is not None:
            self.install_diffusion_kv_metadata(diffusion_kv_metadata)
            installed_request = True
        try:
            runner_output = self._execute_request_list(
                [req],
                od_config=self.od_config,
                allow_single_output=True,
                require_request_batch_support=False,
                kv_prefetch_job=kv_prefetch_job,
                record_name="pipeline_forward",
                diffusion_kv_metadata=[diffusion_kv_metadata] if diffusion_kv_metadata is not None else None,
            )
            output = runner_output.runner_outputs[0].result
            assert output is not None
            return output
        except OmniClientError as exc:
            # Preserve request errors before the worker RPC serializes them.
            return DiffusionOutput.from_exception(exc)
        finally:
            if installed_request:
                self.remove_diffusion_kv_requests([req.request_id])

    def profile_run(self, requests: list[OmniDiffusionRequest]) -> None:
        """Run the maximum per-rank request batch for memory profiling.

        This deliberately bypasses Scheduler admission and Diffusion KV
        metadata validation because cache capacity has not been sized yet.
        It otherwise uses the normal execution-mode path so model inputs,
        collective communication, backend workspaces, denoising, and decode
        allocations contribute to the observed peak. Step-PP profiling keeps
        one fused ``InputBatch`` as a conservative memory bound; runtime
        forwards are request-local.
        """

        if not requests:
            raise ValueError("Diffusion memory profiling requires at least one request.")

        runner_output: BatchRunnerOutput | None = None
        request_ids = [request.request_id for request in requests]
        try:
            if getattr(self.od_config, "step_execution", False):
                scheduler_output = DiffusionSchedulerOutput(
                    step_id=0,
                    scheduled_new_reqs=[
                        NewRequestData(request_id=request.request_id, req=request) for request in requests
                    ],
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                    finished_req_ids=set(),
                    num_running_reqs=len(requests),
                    num_waiting_reqs=0,
                )
                runner_output = self._execute_stepwise(
                    scheduler_output,
                    validate_kv_metadata=False,
                    record_output_peak_memory=False,
                    in_diffusion_kv_memory_profile=True,
                )
            else:
                runner_output = self._execute_request_list(
                    requests,
                    od_config=self.od_config,
                    allow_single_output=len(requests) == 1,
                    require_request_batch_support=len(requests) > 1,
                    record_name="pipeline_memory_profile",
                    # The enclosing native memory_profiling context owns the
                    # peak counters. Resetting them here would discard request
                    # preparation allocations and understate the budget.
                    record_output_peak_memory=False,
                    in_diffusion_kv_memory_profile=True,
                )
            current_omni_platform.synchronize()
        finally:
            for request_id in request_ids:
                self.state_cache.pop(request_id, None)
            self.input_batch = None
            del runner_output
            gc.collect()

    def execute_model_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        od_config: OmniDiffusionConfig,
    ) -> BatchRunnerOutput:
        """Execute scheduled request-mode requests through the batch forward path.

        Builds a ``DiffusionRequestBatch`` from scheduled new requests, runs
        per-request setup, and calls ``pipeline.forward(batch)``. The pipeline
        must declare ``supports_request_batch = True``.
        """
        if getattr(od_config, "request_pipeline_parallel", False):
            return self.execute_request_pipeline_batch(scheduler_output, od_config)

        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            self._validate_diffusion_kv_metadata(
                request_id=new_req.req.request_id,
                metadata=new_req.diffusion_kv_metadata,
            )
        installed_request_ids: list[str] = []
        reqs = [nr.req for nr in scheduler_output.scheduled_new_reqs]
        try:
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.diffusion_kv_metadata is not None:
                    self.install_diffusion_kv_metadata(new_req.diffusion_kv_metadata)
                    installed_request_ids.append(new_req.request_id)
            return self._execute_request_list(
                reqs,
                od_config=od_config,
                allow_single_output=False,
                require_request_batch_support=True,
                record_name="pipeline_forward_batch",
                diffusion_kv_metadata=[
                    new_req.diffusion_kv_metadata
                    for new_req in scheduler_output.scheduled_new_reqs
                    if new_req.diffusion_kv_metadata is not None
                ]
                or None,
            )
        except OmniClientError as exc:
            return self._runner_output_from_outputs(reqs, [DiffusionOutput.from_exception(exc) for _ in reqs])
        finally:
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)

    def _validate_request_pipeline_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        od_config: OmniDiffusionConfig,
    ) -> list[OmniDiffusionRequest]:
        """Validate the closed request-pipeline contract.

        Requests are prepared and executed one at a time, so spatial/temporal
        shape, denoise step count, and Wan scheduler settings are request-local.
        The remaining fields below still describe batch-wide execution policy.
        """

        if scheduler_output.scheduled_cached_reqs.request_ids:
            raise ValueError("request_pipeline_parallel does not accept cached step requests")
        requests = [new_req.req for new_req in scheduler_output.scheduled_new_reqs]
        if not requests:
            return []
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if od_config.cache_backend not in (None, "none"):
            raise ValueError("request_pipeline_parallel does not support cache_backend")
        if (
            getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        ):
            raise ValueError("request_pipeline_parallel does not support paged diffusion KV metadata yet")

        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            self._validate_diffusion_kv_metadata(
                request_id=new_req.request_id,
                metadata=new_req.diffusion_kv_metadata,
            )
            if new_req.req.prompt is None:
                raise ValueError(f"request_pipeline_parallel request {new_req.request_id!r} has no prompt")

        first = requests[0].sampling_params
        comparable_fields = (
            "output_type",
            "num_outputs_per_prompt",
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "cfg_normalize",
        )
        for request in requests[1:]:
            sampling = request.sampling_params
            for field_name in comparable_fields:
                if getattr(sampling, field_name, None) != getattr(first, field_name, None):
                    raise ValueError(
                        "request_pipeline_parallel requires homogeneous sampling fields; "
                        f"{field_name!r} differs between requests"
                    )
        allowed_request_local_extra_args = {"sample_solver", "flow_shift"}
        allowed_legacy_wan_extra_args = {"cfg_img_scale", "cfg_text_scale"}
        for request in requests:
            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            unsupported_extra_args = set(extra_args) - allowed_request_local_extra_args - allowed_legacy_wan_extra_args
            if unsupported_extra_args:
                raise ValueError(
                    "request_pipeline_parallel only supports Wan scheduler and legacy CFG extra_args; "
                    f"unsupported keys for {request.request_id!r}: {sorted(unsupported_extra_args)}"
                )
        return requests

    def _prepare_request_pipeline_states(
        self,
        scheduler_output: DiffusionSchedulerOutput,
    ) -> tuple[list[StepRequestState], list[RunnerOutput]]:
        """Build and prepare all request-local states before any task starts."""

        states, new_request_ids = self._update_states(scheduler_output)
        # Request-pipeline tasks are request-local.  In particular, do not
        # collate different latent/prompt shapes into one InputBatch during
        # preparation; each task builds its one-request view later.
        prepared_states, _, preparation_errors = self._prepare_batch_inputs(
            states,
            new_request_ids,
            build_input_batch=False,
        )
        self.input_batch = None
        return prepared_states, preparation_errors

    def _execute_request_step(
        self,
        state: StepRequestState,
        *,
        od_config: OmniDiffusionConfig,
        attn_metadata: dict[str, Any],
        paged_kv_runtime: Any | None,
    ) -> None:
        """Run one request-step forward and its rank-local scheduler hook."""
        pipeline = self.pipeline
        assert pipeline is not None
        input_batch = InputBatch.make_batch((state,))
        clear_pipeline_stage_durations(pipeline)
        with set_forward_context(
            vllm_config=self.vllm_config,
            omni_diffusion_config=od_config,
            attn_metadata=attn_metadata,
            paged_kv_runtime=paged_kv_runtime,
        ):
            noise_pred = pipeline.denoise_step(input_batch)
        merge_stage_durations(state, consume_pipeline_stage_durations(pipeline))
        pipeline.step_scheduler(state, noise_pred)

    def _build_request_pipeline_batch(
        self,
        states: list[StepRequestState],
        *,
        window: int,
    ) -> RequestPipelineBatchState:
        steps = tuple(state.total_steps for state in states)
        if any(state.do_true_cfg for state in states):
            raise ValueError("request_pipeline_parallel currently does not support true CFG")
        tasks = build_request_pipeline_tasks(
            [state.request_id for state in states],
            steps,
            window=window,
        )
        transport_factory = getattr(self.pipeline, "request_pipeline_transport", None)
        transport = (
            transport_factory(window=window)
            if callable(transport_factory)
            else RequestPipelineTransport(self.pipeline, window=window)
        )
        return RequestPipelineBatchState(tuple(states), tasks, transport)

    def _decode_pipeline_request(
        self, state: StepRequestState, request_index: int
    ) -> tuple[DiffusionOutput, BaseException | None]:
        """Decode on the request owner and gather its output to PP rank 0."""
        pipeline = self.pipeline
        assert pipeline is not None
        distributed = torch.distributed.is_initialized()
        pp_rank = get_pipeline_parallel_rank() if distributed else 0
        pp_size = get_pipeline_parallel_world_size() if distributed else 1
        owner = request_index % pp_size
        output: DiffusionOutput | None = None
        error: BaseException | None = None
        if pp_rank == owner:
            previous_owner = getattr(pipeline, "_request_pipeline_decode_owner", None)
            pipeline._request_pipeline_decode_owner = True
            try:
                use_hsdp = self.od_config.parallel_config.use_hsdp
                distributed_offload = resolve_offload_strategy(self.od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE
                with torch.no_grad() if (use_hsdp or distributed_offload) else torch.inference_mode():
                    clear_pipeline_stage_durations(pipeline)
                    output = pipeline.post_decode(state) or DiffusionOutput()
                    output = self._prepare_output_for_transport(output, state.sampling)
                if output.error is not None:
                    error = RuntimeError(output.error)
                elif owner != 0 and not isinstance(output.output, torch.Tensor):
                    raise TypeError(f"Pipeline decode for {state.request_id} on rank {owner} requires tensor outputs")
            except BaseException as exc:
                error = exc
                output = DiffusionOutput.from_exception(exc)
            finally:
                if previous_owner is None:
                    delattr(pipeline, "_request_pipeline_decode_owner")
                else:
                    pipeline._request_pipeline_decode_owner = previous_owner

        if owner != 0:
            pp_group = get_pp_group()
            if pp_rank == owner:
                assert output is not None
                payload = {"error": output.error} if error is not None else {"output": output.output}
                for work in pp_group.isend_tensor_dict(payload, dst=0):
                    work.wait()
                output = None
            elif pp_rank == 0:
                received, handles, callbacks = pp_group.irecv_tensor_dict(src=owner)
                for handle in handles:
                    handle.wait()
                for callback in callbacks:
                    callback()
                assert received is not None
                if received.get("error") is not None:
                    error = RuntimeError(received["error"])
                    output = DiffusionOutput.from_exception(error)
                elif not isinstance(received.get("output"), torch.Tensor):
                    error = TypeError(f"Pipeline decode owner {owner} returned no tensor for {state.request_id}")
                    output = DiffusionOutput.from_exception(error)
                else:
                    output = DiffusionOutput(output=received["output"])

        output = output or DiffusionOutput()
        if pp_rank == 0:
            self._mark_pp_video_output(output, state.sampling)
            attach_stage_durations(state, output)
        return output, error

    def execute_request_pipeline_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        od_config: OmniDiffusionConfig,
    ) -> BatchRunnerOutput:
        """Execute a closed request batch as an interleaved PP microtask plan."""

        requests = self._validate_request_pipeline_batch(scheduler_output, od_config)
        if not requests:
            return BatchRunnerOutput.from_list([])

        installed_request_ids: list[str] = []
        request_ids = [request.request_id for request in requests]
        batch: RequestPipelineBatchState | None = None
        outputs: list[DiffusionOutput] = []
        try:
            use_hsdp = od_config.parallel_config.use_hsdp
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.diffusion_kv_metadata is not None:
                    self.install_diffusion_kv_metadata(new_req.diffusion_kv_metadata)
                    installed_request_ids.append(new_req.request_id)

            with torch.no_grad() if use_hsdp else torch.inference_mode():
                states, preparation_errors = self._prepare_request_pipeline_states(scheduler_output)
            if preparation_errors:
                first_error = preparation_errors[0].result
                if first_error is None or first_error.error is None:
                    raise RuntimeError("request-pipeline preparation failed")
                raise RuntimeError(first_error.error)
            if len(states) != len(requests):
                raise RuntimeError("request-pipeline request/state count mismatch")
            batch = self._build_request_pipeline_batch(
                states,
                window=getattr(od_config, "request_pipeline_parallel_window", 1),
            )

            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary and current_omni_platform.is_available():
                current_omni_platform.reset_peak_memory_stats()
            grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
            kv_backend = getattr(self, "diffusion_kv_backend", None)
            paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None
            attn_metadata: dict[str, Any] = {}
            state_by_id = {state.request_id: state for state in states}
            with (
                grad_context,
                request_cancellation_scope(
                    [getattr(request, "cancellation_signal", None) for request in requests],
                    enabled=getattr(self.pipeline, "supports_request_cancellation", False) is True,
                ),
                batch.transport,
            ):
                for task in batch.tasks:
                    state = state_by_id[task.request_id]
                    task_error: BaseException | None = None
                    try:
                        check_request_cancellation()
                        batch.transport.begin_task(task)
                        if task.is_tombstone:
                            if state.step_index != state.total_steps:
                                raise RuntimeError(
                                    f"request-pipeline tombstone arrived before {task.request_id} completed: "
                                    f"state step {state.step_index}, total steps {state.total_steps}"
                                )
                        elif state.step_index != task.step_index:
                            raise RuntimeError(
                                f"request-pipeline task order mismatch for {task.request_id}: "
                                f"expected state step {state.step_index}, task step {task.step_index}"
                            )
                    except BaseException as exc:
                        task_error = exc
                    if _dit_any_rank_failed(task_error is not None):
                        if task_error is None:
                            task_error = RuntimeError(
                                f"request-pipeline task {task.task_index} failed on another DiT rank"
                            )
                        batch.failed = task_error
                        raise task_error
                    if task.is_tombstone:
                        # Keep every rank on the same task/control sequence.
                        continue
                    trace_request_pipeline_event(
                        "request_pipeline_execute",
                        phase="enter",
                        request_id=task.request_id,
                        step_index=task.step_index,
                        task_index=task.task_index,
                    )
                    local_error: BaseException | None = None
                    try:
                        if state.latents is not None:
                            state.latents = batch.transport.consume_feedback(state.latents)
                        # Every rank invokes the scheduler hook.  Rank 0 posts
                        # the feedback receive while the last rank owns the
                        # actual solver update and send.
                        self._execute_request_step(
                            state,
                            od_config=od_config,
                            attn_metadata=attn_metadata,
                            paged_kv_runtime=paged_kv_runtime,
                        )
                        if state.step_index != task.step_index + 1:
                            raise RuntimeError(
                                f"request-pipeline scheduler did not advance {task.request_id} "
                                f"from step {task.step_index}"
                            )
                    except BaseException as exc:
                        local_error = exc
                    if _dit_any_rank_failed(local_error is not None):
                        if local_error is None:
                            local_error = RuntimeError(
                                f"request-pipeline task {task.task_index} failed on another DiT rank"
                            )
                        batch.failed = local_error
                        raise local_error
                    trace_request_pipeline_event(
                        "request_pipeline_execute",
                        phase="return",
                        request_id=task.request_id,
                        step_index=task.step_index,
                        task_index=task.task_index,
                    )

                # The context exits through RequestPipelineTransport.__exit__,
                # which drains both activation sends and lazy feedback receives.

            decode_errors: list[BaseException] = []
            for request_index, state in enumerate(batch.states):
                output, error = self._decode_pipeline_request(state, request_index)
                if error is not None:
                    decode_errors.append(error)
                outputs.append(output)
                if is_primary:
                    output.peak_memory_mb = max(output.peak_memory_mb, self._sample_peak_memory_mb())

            if _dit_any_rank_failed(bool(decode_errors)):
                decode_error = (
                    decode_errors[0]
                    if decode_errors
                    else RuntimeError("request-pipeline decode failed on another rank")
                )
                batch.failed = decode_error
                raise decode_error

            self._maybe_send_stage_payload(requests, outputs)
            return self._runner_output_from_outputs(requests, outputs)
        except OmniClientError as exc:
            error_outputs = [DiffusionOutput.from_exception(exc) for _ in requests]
            return self._runner_output_from_outputs(requests, error_outputs)
        finally:
            if batch is not None and batch.failed is not None:
                logger.error("Request-pipeline batch failed: %s", batch.failed, exc_info=True)
            self.release_step_requests(request_ids)
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def _supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def release_step_requests(self, request_ids: Sequence[str]) -> None:
        """Drop request-owned state after execution/output preparation has returned.

        The cached batch also owns states, including model-private tensors in
        ``extra``. Invalidate it when its composition contains a retired request;
        surviving requests retain their own state and are repacked on the next step.
        """
        for request_id in request_ids:
            state = self.state_cache.pop(request_id, None)
            release = getattr(getattr(self, "pipeline", None), "release_step_request", None)
            if state is not None and callable(release):
                release(state)
        batch = getattr(self, "input_batch", None)
        if batch is not None and any(request_id in batch.request_ids for request_id in request_ids):
            self.input_batch = None

    def _cleanup_finished_step_requests(self, scheduler_output: DiffusionSchedulerOutput) -> None:
        """Retire state and paged-KV rows released by the scheduler wave."""
        finished_req_ids = scheduler_output.finished_req_ids
        self.release_step_requests(list(finished_req_ids))

        if (
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
            and finished_req_ids
        ):
            self.remove_diffusion_kv_requests(list(finished_req_ids))

    def _update_states(self, scheduler_output: DiffusionSchedulerOutput) -> tuple[list[StepRequestState], list[str]]:
        """Resolve cached state and create state for newly admitted requests."""

        resolved: list[StepRequestState] = []
        new_request_ids: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                self._maybe_recv_stage_payload(sched_new_req.req)
                new_state = StepRequestState(
                    request_id=request_id,
                    sampling=copy.deepcopy(sched_new_req.req.sampling_params),
                    prompt=sched_new_req.req.prompt,
                    kv_sender_info=sched_new_req.req.kv_sender_info,
                    prepared_layout=getattr(sched_new_req.req, "prepared_layout", None),
                    external_req_id=getattr(sched_new_req.req, "external_req_id", None),
                )
                if (
                    sched_new_req.diffusion_kv_metadata is not None
                    and getattr(self.od_config, "kv_transfer_config", None) is not None
                ):
                    new_state.extra["kv_computed_tokens"] = tuple(
                        seq.num_computed_tokens for seq in sched_new_req.diffusion_kv_metadata.sequences
                    )
                state_req = copy.copy(sched_new_req.req)
                state_req.sampling_params = new_state.sampling
                if self.kv_transfer_manager is not None:
                    self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                        state_req,
                        cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                        target_device=self._target_device,
                    )
                self.state_cache[request_id] = new_state
                resolved.append(new_state)

            # process cached requests
            for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
                state = self.state_cache.get(request_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {request_id}.")
                resolved.append(state)
            for state in resolved:
                context = scheduler_output.interaction_boundaries.get(state.request_id)
                if context is not None:
                    state.interaction_boundary = context
        except Exception:
            for request_id in new_request_ids:
                self.state_cache.pop(request_id, None)
            raise

        return resolved, new_request_ids

    def _prepare_batch_inputs(
        self,
        states: list[StepRequestState],
        new_request_ids: list[str],
        *,
        build_input_batch: bool = True,
    ) -> tuple[list[StepRequestState], InputBatch | None, list[RunnerOutput]]:
        # process new reqs
        pipeline = self.pipeline
        assert pipeline is not None, "Model not loaded. Call load_model() first."
        prepared_states: list[StepRequestState] = []
        error_outputs: list[RunnerOutput] = []
        for state in states:
            if state.request_id in new_request_ids or state.pending_chunk_boundary:
                # Everything that requires rank-synchronization must be called
                # inside a try, record the exception and handle with `_dit_any_rank_failed`.
                # Reason (example): An exception in ``_initialize_generator`` or
                # ``clear_pipeline_stage_durations`` on one rank would skip the
                # all-reduce here while every peer proceeds into it, and the
                # peers then hang on the NCCL collective until timeout.
                def _abort_prep_failure(per_req_exc: BaseException | None) -> None:
                    self.state_cache.pop(state.request_id, None)
                    if per_req_exc is None:
                        per_req_exc = RuntimeError(
                            f"Stepwise preparation failed on another DiT rank for {state.request_id}"
                        )
                    logger.error(
                        "Stepwise request preparation failed for %s: %s",
                        state.request_id,
                        per_req_exc,
                        exc_info=isinstance(per_req_exc, Exception),
                    )
                    error_outputs.append(
                        RunnerOutput(
                            request_id=state.request_id,
                            step_index=state.step_index,
                            finished=True,
                            result=DiffusionOutput.from_exception(per_req_exc),
                        )
                    )

                per_req_exc: BaseException | None = None
                try:
                    clear_pipeline_stage_durations(pipeline)
                    if state.request_id in new_request_ids:
                        if getattr(
                            state.sampling, "streaming_buffer_seconds", None
                        ) is not None and not supports_interaction_apply(pipeline):
                            raise ValueError("playback feedback requires an interactive chunked pipeline")
                        self._initialize_generator(state.sampling)
                        pipeline.prepare_encode(state)
                except Exception as exc:
                    per_req_exc = exc
                # Pipelines that do rank-0-only work (e.g. MiniMax H3
                # reference-video prep) must broadcast per-request failures
                # internally so downstream collectives stay in step; even so,
                # cross-check that every DiT rank agrees so a rank-local error
                # (or a future pipeline that omits the guard) does not leave
                # the process group half-way through a new request.
                if _dit_any_rank_failed(per_req_exc is not None):
                    _abort_prep_failure(per_req_exc)
                    continue
                # If the pipeline supports interaction, the interaction session initialization also needs to call
                # synchronized_monotonic_time(). Wrap in another try-block to not block on prepare_encode failures.
                try:
                    if supports_interaction_apply(pipeline):
                        pipe = cast(SupportsInteractionApply, pipeline)
                        if state.pending_chunk_boundary:
                            # The engine drains control RPCs (and waits for
                            # playback, when enabled) before this next call.
                            pipe.apply_interaction_at_chunk_boundary(state)
                            state.pending_chunk_boundary = False
                        elif state.chunk_index == 0:
                            assert self._interaction_coordinator is not None, (
                                "Model not loaded. Call load_model() first."
                            )
                            state.interaction_chunk_metadata = (
                                self._interaction_coordinator.maybe_prepare_initial_session(state, pipe)
                            )
                        pipe.prepare_next_chunk(state)
                        buffer_seconds = getattr(state.sampling, "streaming_buffer_seconds", None)
                        if buffer_seconds is not None and pipe.peek_chunk_media(state).duration_s > buffer_seconds:
                            raise ValueError("streaming_buffer_seconds must fit a complete media chunk")
                    merge_stage_durations(
                        state,
                        consume_pipeline_stage_durations(pipeline),
                    )
                except Exception as exc:
                    per_req_exc = exc
                if _dit_any_rank_failed(per_req_exc is not None):
                    _abort_prep_failure(per_req_exc)
                    continue
            prepared_states.append(state)

        if not prepared_states or not build_input_batch:
            return prepared_states, None, error_outputs
        input_batch = InputBatch.make_batch(
            prepared_states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return prepared_states, input_batch, error_outputs

    def _update_states_after(
        self,
        states: list[StepRequestState],
        input_batch: InputBatch,
        interrupted: bool = False,
    ) -> None:
        """Step-after update: clear cached state for completed request."""
        gathered_latents = torch.cat([state.latents for state in states], dim=0)
        if (
            input_batch.latents.size() == gathered_latents.size()
            and input_batch.latents.dtype == gathered_latents.dtype
            and input_batch.latents.device == gathered_latents.device
        ):
            input_batch.latents.copy_(gathered_latents)
        else:
            input_batch.latents = gathered_latents.clone()

        self.input_batch = input_batch
        scatter_latents(states, input_batch)

        # Include per-request errors, whose cache entries were already removed.
        self.release_step_requests(
            [
                state.request_id
                for state in states
                if interrupted or state.request_denoise_completed or state.request_id not in self.state_cache
            ]
        )

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step for the scheduled request batch and return runner output."""
        if (
            getattr(self.od_config, "step_execution", False)
            and getattr(self.od_config.parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            try:
                return self._execute_stepwise_pipeline(scheduler_output)
            except Exception:
                self.release_step_requests(scheduler_output.scheduled_request_ids)
                raise
        return self._execute_stepwise(
            scheduler_output,
            validate_kv_metadata=True,
            record_output_peak_memory=True,
        )

    def _execute_stepwise_pipeline(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Run one request-step forward per scheduled state on every PP rank."""
        pipeline = self.pipeline
        if pipeline is None or not getattr(pipeline, "supports_step_pipeline_parallel", False):
            raise ValueError("The loaded pipeline does not support Step-PP")
        if self.od_config.cache_backend not in (None, "none") or self.od_config.streaming_output:
            raise ValueError("Step-PP does not support cache backends or streaming output")
        omni_kv = getattr(self.od_config, "omni_kv_config", None) or {}
        if getattr(self.od_config, "kv_transfer_config", None) is not None or any(
            omni_kv.get(key, False) for key in ("need_send_cache", "need_recv_cache")
        ):
            raise ValueError("Step-PP does not support diffusion KV transfer")
        if (
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        ):
            raise ValueError("Step-PP does not support paged diffusion KV")

        self._cleanup_finished_step_requests(scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
            if not new_req.req.use_step_execution:
                raise ValueError("Step-PP does not support full-forward fallback requests")
            self._validate_diffusion_kv_metadata(
                request_id=new_req.request_id,
                metadata=new_req.diffusion_kv_metadata,
            )

        states, new_request_ids = self._update_states(scheduler_output)
        new_request_id_set = set(new_request_ids)
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        outputs: list[RunnerOutput] = []
        kv_backend = getattr(self, "diffusion_kv_backend", None)
        paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None

        with grad_context:
            for state in states:
                if state.request_id in new_request_id_set:
                    if state.sampling.generator is not None:
                        raise ValueError("Step-PP requires a seed-based generator, not a custom Generator")
                    self._initialize_generator(state.sampling)
                    clear_pipeline_stage_durations(pipeline)
                    pipeline.prepare_encode(state)
                    merge_stage_durations(state, consume_pipeline_stage_durations(pipeline))
            self.input_batch = None
            for state in states:
                if isinstance(state.latents, AsyncLatents):
                    state.latents = state.latents.resolve()
                self._execute_request_step(
                    state,
                    od_config=self.od_config,
                    attn_metadata={},
                    paged_kv_runtime=paged_kv_runtime,
                )
                outputs.append(RunnerOutput(request_id=state.request_id, step_index=state.step_index, finished=False))

        return BatchRunnerOutput.from_list(outputs)

    def decode_pipeline_requests(self, request_ids: list[str]) -> BatchRunnerOutput:
        """Drain the PP edge, then distribute a decode cohort across PP ranks."""
        pipeline = self.pipeline
        if pipeline is None or not getattr(pipeline, "supports_step_pipeline_parallel", False):
            raise ValueError("The loaded pipeline does not support Step-PP")
        pp_group = get_pp_group()
        pp_rank = get_pipeline_parallel_rank()
        states = [self.state_cache[request_id] for request_id in request_ids]
        use_hsdp = self.od_config.parallel_config.use_hsdp
        use_distributed_offload = resolve_offload_strategy(self.od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        grad_context = torch.no_grad() if (use_hsdp or use_distributed_offload) else torch.inference_mode()

        with grad_context:
            pipeline._sync_pp_send()
            if pp_rank == 0:
                for state in states:
                    if isinstance(state.latents, AsyncLatents):
                        state.latents = state.latents.resolve()
            pp_group.barrier()

            outputs: list[RunnerOutput] = []
            for index, state in enumerate(states):
                output, error = self._decode_pipeline_request(state, index)
                if error is not None and not isinstance(error, Exception):
                    raise error
                if pp_rank == 0:
                    if output.error is None and not isinstance(output.output, torch.Tensor):
                        output = DiffusionOutput(error="Step-PP decode requires tensor outputs")
                    outputs.append(
                        RunnerOutput(
                            request_id=state.request_id,
                            step_index=state.step_index,
                            finished=True,
                            result=output,
                        )
                    )

        self.release_step_requests(request_ids)
        return BatchRunnerOutput.from_list(outputs) if pp_rank == 0 else BatchRunnerOutput.from_list([])

    def _execute_non_step_requests(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Run a complete legacy forward for requests excluded from step mode."""
        if scheduler_output.scheduled_cached_reqs.request_ids:
            raise ValueError("A non-step fallback batch cannot contain cached stepwise requests.")

        runner_outputs: list[RunnerOutput] = []
        for index, new_req in enumerate(scheduler_output.scheduled_new_reqs):
            try:
                result = self.execute_model(
                    new_req.req,
                    kv_prefetch_job=getattr(scheduler_output, "kv_prefetch_job", None) if index == 0 else None,
                    diffusion_kv_metadata=new_req.diffusion_kv_metadata,
                )
            except Exception as exc:
                logger.error(
                    "Non-step fallback execution failed for %s",
                    new_req.request_id,
                    exc_info=True,
                )
                # Keep client-error metadata, matching the stepwise path below.
                result = DiffusionOutput.from_exception(exc)

            step_index = getattr(new_req.req.sampling_params, "step_index", None)
            runner_outputs.append(
                RunnerOutput(
                    request_id=new_req.request_id,
                    step_index=0 if step_index is None else step_index,
                    finished=True,
                    result=result,
                )
            )
        return BatchRunnerOutput.from_list(runner_outputs)

    def _execute_stepwise(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        *,
        validate_kv_metadata: bool,
        record_output_peak_memory: bool,
        in_diffusion_kv_memory_profile: bool = False,
    ) -> BatchRunnerOutput:
        """Execute one step with explicit validation and profiling policy."""

        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        # A scheduler wave can release a previous stepwise request while it
        # admits a full-forward fallback request. Do this before dispatch so
        # the fallback does not bypass normal state/KV retirement.
        self._cleanup_finished_step_requests(scheduler_output)
        if not scheduler_output.scheduled_request_ids:
            return BatchRunnerOutput.from_list([])
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_new_request_data_identity(new_req)
        non_step_requests = [
            new_req
            for new_req in scheduler_output.scheduled_new_reqs
            if not getattr(new_req.req, "use_step_execution", True)
        ]
        if non_step_requests:
            scheduled_request_count = len(scheduler_output.scheduled_new_reqs) + len(
                scheduler_output.scheduled_cached_reqs.request_ids
            )
            if len(non_step_requests) != scheduled_request_count:
                raise ValueError("Cannot mix stepwise and non-step fallback requests in one scheduler batch.")
            # This wave has no stepwise requests, so the previous batch cannot
            # be reused and must not keep its step tensors alive.
            self.input_batch = None
            return self._execute_non_step_requests(scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            if validate_kv_metadata:
                self._validate_diffusion_kv_metadata(
                    request_id=new_req.req.request_id,
                    metadata=new_req.diffusion_kv_metadata,
                )
        if not self._supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        # Stepwise mode only supports the basic state-driven denoise path for now.
        # Request-mode extras such as cache backends, editing inputs, and
        # similar features are not supported here yet.
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        # Scheduler metadata is installed only for newly admitted requests;
        # cached requests continue to use the row installed on their first
        # step. Finished-state retirement is shared with full-forward
        # fallback dispatch above.
        installed_request_ids: list[str] = []
        try:
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.diffusion_kv_metadata is not None:
                    self.install_diffusion_kv_metadata(new_req.diffusion_kv_metadata)
                    installed_request_ids.append(new_req.request_id)
            return self._execute_stepwise_core(
                scheduler_output,
                record_output_peak_memory=record_output_peak_memory,
                in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
            )
        except Exception:
            self.release_step_requests(scheduler_output.scheduled_request_ids)
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)
            raise

    def _execute_stepwise_core(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        *,
        record_output_peak_memory: bool,
        in_diffusion_kv_memory_profile: bool = False,
    ) -> BatchRunnerOutput:
        """Run the denoise step after metadata admission and row installation."""

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            pipeline = self.pipeline
            assert pipeline is not None, "Model not loaded. Call load_model() first."
            had_active_states = bool(self.state_cache)
            states, new_request_ids = self._update_states(scheduler_output)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if (
                record_output_peak_memory
                and new_request_ids
                and not had_active_states
                and is_primary
                and current_omni_platform.is_available()
            ):
                current_omni_platform.reset_peak_memory_stats()
            states, input_batch, runner_output_list = self._prepare_batch_inputs(states, new_request_ids)
            if input_batch is None:
                return BatchRunnerOutput.from_list(runner_output_list)
            attn_metadata: dict[str, Any] = {}

            kv_backend = getattr(self, "diffusion_kv_backend", None)
            paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None
            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
                paged_kv_runtime=paged_kv_runtime,
                in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
            ):
                clear_pipeline_stage_durations(pipeline)
                noise_pred = pipeline.denoise_step(input_batch, states=states)
                denoise_stage_durations = consume_pipeline_stage_durations(pipeline)
                for state in states:
                    merge_stage_durations(
                        state,
                        denoise_stage_durations,
                    )

                pipeline_interrupted = getattr(pipeline, "interrupt", False)
                if noise_pred is None and pipeline_interrupted:
                    for state in states:
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=state.request_id,
                                step_index=state.step_index,
                                finished=True,
                                result=DiffusionOutput(error="stepwise denoise interrupted"),
                            )
                        )

                else:
                    offset = 0
                    for req in states:
                        if req.latents is None:
                            raise RuntimeError(f"Stepwise request {req.request_id} has no latent state.")
                        row_num = req.latents.shape[0]
                        try:
                            pipeline.step_scheduler(
                                req, noise_pred[offset : offset + row_num] if noise_pred is not None else None
                            )
                            if self.od_config.streaming_output:
                                should_decode = req.chunk_denoise_completed or req.request_denoise_completed
                            else:
                                should_decode = req.denoise_completed

                            streaming_media_duration = 0.0
                            streaming_next_media_duration = 0.0
                            if should_decode:
                                if getattr(req.sampling, "streaming_buffer_seconds", None) is not None:
                                    streaming_media_duration = (
                                        cast(SupportsInteractionApply, pipeline).peek_chunk_media(req).duration_s
                                    )
                                clear_pipeline_stage_durations(pipeline)
                                result = pipeline.post_decode(req)
                                if result is not None:
                                    result = self._prepare_output_for_transport(result, req.sampling)
                                    self._attach_stepwise_metadata(
                                        req,
                                        result,
                                    )
                                    if supports_interaction_apply(pipeline) and not req.request_denoise_completed:
                                        if getattr(req.sampling, "streaming_buffer_seconds", None) is not None:
                                            req.pending_chunk_boundary = True
                                            pipe = cast(SupportsInteractionApply, pipeline)
                                            streaming_next_media_duration = pipe.peek_chunk_media(req).duration_s
                                        else:
                                            pipe = cast(SupportsInteractionApply, pipeline)
                                            pipe.apply_interaction_at_chunk_boundary(req)
                                            pipe.prepare_next_chunk(req)
                            else:
                                result = None
                            # finished should be computed after post_decode() advanced chunk_index
                            finished = (
                                req.request_denoise_completed
                                if self.od_config.streaming_output
                                else req.denoise_completed
                            )
                            if finished and result is not None:
                                self._maybe_send_stage_payload([req], [result])  # type: ignore[list-item]
                            runner_output_list.append(
                                RunnerOutput(
                                    request_id=req.request_id,
                                    step_index=req.step_index,
                                    finished=finished,
                                    result=result,
                                    streaming_media_duration=streaming_media_duration,
                                    streaming_next_media_duration=streaming_next_media_duration,
                                )
                            )
                            offset = offset + row_num
                        except Exception as per_req_exc:
                            offset = offset + row_num
                            self.state_cache.pop(req.request_id, None)
                            logger.error(
                                "Stepwise per-request error for %s: %s",
                                req.request_id,
                                per_req_exc,
                                exc_info=True,
                            )
                            runner_output_list.append(
                                RunnerOutput(
                                    request_id=req.request_id,
                                    step_index=req.step_index,
                                    finished=True,
                                    result=DiffusionOutput.from_exception(per_req_exc),
                                )
                            )

                    if noise_pred is not None and offset != noise_pred.shape[0]:
                        raise ValueError(
                            f"Stepwise noise_pred consumed {offset} rows, "
                            f"but batched noise_pred has {noise_pred.shape[0]} rows."
                        )

                if is_primary and record_output_peak_memory:
                    batch_peak_memory_mb = self._sample_peak_memory_mb()
                    states_by_id = {state.request_id: state for state in states}
                    for state in states:
                        state.peak_memory_mb = max(state.peak_memory_mb, batch_peak_memory_mb)
                    for runner_output in runner_output_list:
                        if runner_output.result is None:
                            continue
                        matched_state = states_by_id.get(runner_output.request_id)
                        if matched_state is None:
                            continue
                        runner_output.result.peak_memory_mb = max(
                            runner_output.result.peak_memory_mb,
                            matched_state.peak_memory_mb,
                        )

                terminal_request_ids = [
                    runner_output.request_id for runner_output in runner_output_list if runner_output.finished
                ]
                self._update_states_after(states, input_batch, pipeline_interrupted)
                if (
                    getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
                    is DiffusionKVCacheMode.PAGED_SCHEDULER
                    and terminal_request_ids
                ):
                    self.remove_diffusion_kv_requests(terminal_request_ids)

                return BatchRunnerOutput.from_list(runner_output_list)

    def submit_interaction(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Route a midway interaction through the pipeline interaction coordinator."""
        assert self.pipeline is not None and self._interaction_coordinator is not None, (
            "Model not loaded. Call load_model() first."
        )
        if not self.od_config.streaming_output:
            raise ValueError("submit_interaction requires streaming_output=True")
        if not self._supports_step_mode():
            raise ValueError("submit_interaction requires step execution support")

        state = self.state_cache.get(request_id)
        if state is None:
            raise ValueError(f"No active request state for interaction: {request_id!r}")

        event = interaction["event"]
        parts: list[tuple[str, InteractionPayload]] = []
        prompt = event.get("prompt")
        if prompt is not None:
            parts.append(("prompt", {"prompt": prompt}))
        multi_modal_data = event.get("multi_modal_data")
        if multi_modal_data:
            parts.extend(
                (str(modality), cast(InteractionPayload, payload)) for modality, payload in multi_modal_data.items()
            )

        received_at = interaction.get("received_at", time.monotonic())
        if (
            isinstance(received_at, bool)
            or not isinstance(received_at, (int, float))
            or not math.isfinite(received_at)
            or received_at < 0
        ):
            raise ValueError("interaction received_at must be finite and non-negative")
        self._interaction_coordinator.enqueue_parts(
            state,
            parts=parts,
            event_id=interaction["event_id"],
            received_at=float(received_at),
            transition_chunks=interaction.get("transition_chunks"),
        )
