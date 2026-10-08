# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Queued pipeline model-runner lifecycle and stage execution."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed import parallel_state
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.interface import supports_pipeline_stage_execution
from vllm_omni.diffusion.sched.interface import (
    DiffusionSchedulerOutput,
    NewRequestData,
    validate_new_request_data_identity,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineBatchContext,
    PipelineStageSpec,
    PipelineTask,
    PipelineTaskStatus,
)
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    RunnerOutput,
    StepRequestState,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def dit_any_rank_failed(local_failed: bool) -> bool:
    """Agree on queued preparation failures across the DiT process group."""
    if not torch.distributed.is_initialized():
        return local_failed
    try:
        get_dit_group = getattr(parallel_state, "get_dit_group", None)
        group = get_dit_group() if get_dit_group is not None else None
    except (AssertionError, ImportError):
        group = None
    if group is None:
        return local_failed
    signal = torch.tensor(1 if local_failed else 0, dtype=torch.int32)
    if current_omni_platform.is_available():
        signal = signal.to(device=current_omni_platform.device_type)
    torch.distributed.all_reduce(signal, op=torch.distributed.ReduceOp.MAX, group=group)
    return bool(signal.item())


class QueuedModelRuntime:
    """Own queued PP model state while delegating generic model operations."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
        else:
            setattr(self.owner, name, value)

    @property
    def pipeline_batch_contexts(self) -> dict[tuple[int, str], PipelineBatchContext]:
        contexts = getattr(self.owner, "_pipeline_batch_contexts", None)
        if contexts is None:
            contexts = {}
            self.owner._pipeline_batch_contexts = contexts
        return contexts

    @property
    def pipeline_request_owners(self) -> dict[tuple[str, int], tuple[int, str]]:
        owners = getattr(self.owner, "_pipeline_request_owners", None)
        if owners is None:
            owners = {}
            self.owner._pipeline_request_owners = owners
        return owners

    def _pipeline_inference_context(self) -> AbstractContextManager[Any]:
        use_hsdp = bool(getattr(getattr(self.od_config, "parallel_config", None), "use_hsdp", False))
        return torch.no_grad() if use_hsdp else torch.inference_mode()

    def _run_queued_preparation_phase(
        self,
        request_id: str,
        phase: str,
        operation: Callable[[], Any],
    ) -> Any:
        local_error: Exception | None = None
        result: Any = None
        try:
            result = operation()
        except Exception as exc:
            local_error = exc
        rank_failed = getattr(self.owner, "_dit_any_rank_failed", dit_any_rank_failed)
        if rank_failed(local_error is not None):
            if local_error is None:
                local_error = RuntimeError(f"Queued pipeline {phase} failed on another rank for {request_id}")
            raise local_error
        return result

    def _prepare_cached_pipeline_request(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        request_id: str,
    ) -> tuple[str, ...]:
        def prepare() -> tuple[str, ...]:
            with self._pipeline_inference_context():
                states, _ = self._update_states(scheduler_output)
            if len(states) != 1 or states[0].request_id != request_id:
                raise RuntimeError("Queued cached preparation did not retain the expected request state.")
            input_batch = InputBatch.make_batch(states, cached_batch=self.input_batch)
            if input_batch is None:
                raise RuntimeError("Queued cached batch construction produced no input batch.")
            self.input_batch = input_batch
            return (request_id,)

        return self._run_queued_preparation_phase(request_id, "cached preparation", prepare)

    def _prepare_queued_new_preamble(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        new_request: NewRequestData,
        installed_request_ids: list[str],
    ) -> list[StepRequestState]:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not self._supports_step_mode() or not supports_pipeline_stage_execution(self.pipeline):
            raise ValueError("The loaded diffusion pipeline does not support queued local-stage execution.")
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Queued pipeline preparation does not support cache_backend yet.")
        self._cleanup_finished_step_requests(scheduler_output)
        if new_request.diffusion_kv_metadata is not None:
            self._validate_diffusion_kv_metadata(
                request_id=new_request.request_id,
                metadata=new_request.diffusion_kv_metadata,
            )
            installed_request_ids.append(new_request.request_id)
            self.install_diffusion_kv_metadata(new_request.diffusion_kv_metadata)
        with self._pipeline_inference_context():
            states, _ = self._update_states(scheduler_output)
        return states

    def _prepare_queued_new_setup(self, states: list[StepRequestState]) -> None:
        for state in states:
            self._initialize_generator(state.sampling)
            clear_pipeline_stage_durations(self.pipeline)

    def _prepare_queued_new_encode(self, states: list[StepRequestState]) -> None:
        with self._pipeline_inference_context():
            for state in states:
                self.pipeline.prepare_encode(state)
                merge_stage_durations(state, consume_pipeline_stage_durations(self.pipeline))

    def _prepare_queued_new_batch(
        self,
        states: list[StepRequestState],
        expected_request_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        prepared_request_ids = tuple(state.request_id for state in states)
        if prepared_request_ids != expected_request_ids:
            raise RuntimeError("Queued pipeline preparation did not retain the expected request state.")
        input_batch = InputBatch.make_batch(states, cached_batch=None)
        if input_batch is None:
            raise RuntimeError("Queued pipeline batch construction produced no input batch.")
        self.input_batch = input_batch
        return prepared_request_ids

    def prepare_pipeline_requests(self, scheduler_output: DiffusionSchedulerOutput) -> tuple[str, ...]:
        """Prepare one queued request coherently without executing a denoise step."""
        new_requests = list(scheduler_output.scheduled_new_reqs)
        cached_request_ids = list(scheduler_output.scheduled_cached_reqs.request_ids)
        if len(new_requests) + len(cached_request_ids) != 1 or (new_requests and cached_request_ids):
            raise ValueError("M2 queued preparation requires exactly one new or cached request.")
        if cached_request_ids:
            return self._prepare_cached_pipeline_request(scheduler_output, cached_request_ids[0])
        new_request = new_requests[0]
        validate_new_request_data_identity(new_request)
        if not getattr(new_request.req, "use_step_execution", True):
            raise ValueError("Queued pipeline preparation requires step execution.")

        expected_request_ids = (new_request.request_id,)
        installed_request_ids: list[str] = []
        try:
            states = self._run_queued_preparation_phase(
                new_request.request_id,
                "pipeline preamble",
                lambda: self._prepare_queued_new_preamble(scheduler_output, new_request, installed_request_ids),
            )
            self._run_queued_preparation_phase(
                new_request.request_id,
                "local setup",
                lambda: self._prepare_queued_new_setup(states),
            )
            self._run_queued_preparation_phase(
                new_request.request_id,
                "encoding",
                lambda: self._prepare_queued_new_encode(states),
            )
            return self._run_queued_preparation_phase(
                new_request.request_id,
                "batch construction",
                lambda: self._prepare_queued_new_batch(states, expected_request_ids),
            )
        except Exception:
            self.state_cache.pop(new_request.request_id, None)
            self.input_batch = None
            if installed_request_ids:
                self.remove_diffusion_kv_requests(installed_request_ids)
            raise

    def prepare_pipeline_batch(
        self,
        task: PipelineTask,
        pp_stage_spec: PipelineStageSpec,
        states: list[StepRequestState],
    ) -> PipelineBatchContext:
        """Create an independently owned context from coherently prepared states."""
        if self.pipeline is None or not supports_pipeline_stage_execution(self.pipeline):
            raise ValueError("The loaded diffusion pipeline does not support queued local-stage execution.")
        self.pipeline.validate_pipeline_stage_execution(pp_stage_spec)
        request_state_ids = tuple(state.request_id for state in states)
        if request_state_ids != task.request_ids:
            raise ValueError(
                f"Pipeline task request ids {task.request_ids!r} do not match prepared states {request_state_ids!r}."
            )
        if len(states) != 1:
            raise ValueError("M2 queued pipeline execution requires exactly one request state per batch.")
        if states[0].step_index != task.step_index:
            raise ValueError(
                f"Pipeline task step {task.step_index} does not match request state step {states[0].step_index}."
            )
        key = (pp_stage_spec.pp_stage_id, task.batch_id)
        if key in self.pipeline_batch_contexts:
            raise ValueError(f"Pipeline batch context {key!r} already exists.")
        owner_key = (states[0].request_id, task.step_index)
        existing_owner = self.pipeline_request_owners.get(owner_key)
        if existing_owner is not None:
            raise ValueError(f"Request step {owner_key!r} is already owned by pipeline batch {existing_owner!r}.")
        with self._pipeline_inference_context():
            context = PipelineBatchContext(
                task=task,
                stage_spec=pp_stage_spec,
                request_state_ids=request_state_ids,
                states=tuple(states),
                input_batch=InputBatch.make_batch(states),
            )
        self.pipeline_batch_contexts[key] = context
        self.pipeline_request_owners[owner_key] = key
        return context

    def execute_pipeline_stage(
        self,
        context: PipelineBatchContext,
        pp_stage_spec: PipelineStageSpec,
        intermediate_tensors: Any | None,
    ) -> Any:
        """Execute one local partition without transport or numerical update."""
        self._require_pipeline_context(context, pp_stage_spec)
        if context.status is not PipelineTaskStatus.PENDING:
            raise RuntimeError(f"Pipeline batch {context.task.batch_id!r} is not pending.")
        try:
            self._validate_pipeline_context_progress(context)
            context.status = PipelineTaskStatus.ACTIVE
            kv_backend = getattr(self, "diffusion_kv_backend", None)
            paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None
            with (
                self._pipeline_inference_context(),
                set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=self.od_config,
                    attn_metadata={},
                    paged_kv_runtime=paged_kv_runtime,
                    denoise_step_idx=context.task.step_index,
                ),
            ):
                context.result = self.pipeline.forward_pipeline_stage(
                    context.input_batch,
                    pp_stage_spec=pp_stage_spec,
                    intermediate_tensors=intermediate_tensors,
                    states=context.states,
                )
        except BaseException:
            context.status = PipelineTaskStatus.FAILED
            raise
        return context.result

    def complete_pipeline_step(self, context: PipelineBatchContext, pp_stage_spec: PipelineStageSpec) -> torch.Tensor:
        """Apply the one authoritative numerical update on the last stage."""
        self._require_pipeline_context(context, pp_stage_spec)
        if not pp_stage_spec.is_last:
            raise ValueError("Only the last pipeline stage can complete the numerical step.")
        if context.status is not PipelineTaskStatus.ACTIVE:
            raise RuntimeError("Pipeline batch must be active before numerical completion.")
        if not isinstance(context.result, torch.Tensor):
            context.status = PipelineTaskStatus.FAILED
            raise RuntimeError("Pipeline batch produced a non-tensor result for numerical completion.")
        state = context.states[0]
        try:
            self._validate_pipeline_context_progress(context)
            kv_backend = getattr(self, "diffusion_kv_backend", None)
            paged_kv_runtime = kv_backend if getattr(kv_backend, "paged_attention_adapter", None) is not None else None
            with (
                self._pipeline_inference_context(),
                set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=self.od_config,
                    attn_metadata={},
                    paged_kv_runtime=paged_kv_runtime,
                    denoise_step_idx=context.task.step_index,
                ),
            ):
                self.pipeline.step_scheduler_pipeline_stage(state, context.result)
                if state.latents is None:
                    raise RuntimeError("Pipeline numerical completion produced no latents.")
        except BaseException:
            context.status = PipelineTaskStatus.FAILED
            raise
        context.status = PipelineTaskStatus.COMPLETED
        return state.latents

    def adopt_pipeline_feedback(
        self,
        context: PipelineBatchContext,
        pp_stage_spec: PipelineStageSpec,
        latents: torch.Tensor,
    ) -> None:
        """Adopt last-stage feedback on stage 0 without rerunning the solver."""
        self._require_pipeline_context(context, pp_stage_spec)
        if not pp_stage_spec.is_first:
            raise ValueError("Only the first pipeline stage can adopt latent feedback.")
        if context.status is not PipelineTaskStatus.ACTIVE:
            raise RuntimeError("Pipeline batch must be active before feedback adoption.")
        state = context.states[0]
        try:
            self._validate_pipeline_context_progress(context)
            if state.latents is None:
                raise RuntimeError("First-stage pipeline state has no latent mirror.")
            if (
                state.latents.shape != latents.shape
                or state.latents.dtype != latents.dtype
                or state.latents.device != latents.device
            ):
                raise ValueError("Pipeline feedback latents do not match the first-stage latent mirror.")
            with self._pipeline_inference_context():
                state.latents.copy_(latents)
                state.step_index = context.task.step_index + 1
                context.input_batch.latents = state.latents
        except BaseException:
            context.status = PipelineTaskStatus.FAILED
            raise
        context.status = PipelineTaskStatus.COMPLETED

    def pipeline_has_distributed_vae(self) -> bool:
        vae = getattr(self.pipeline, "vae", None)
        is_distributed_enabled = getattr(vae, "is_distributed_enabled", None)
        return callable(is_distributed_enabled) and bool(is_distributed_enabled())

    def validate_pipeline_finalization(
        self,
        context: PipelineBatchContext,
        pp_stage_spec: PipelineStageSpec,
    ) -> StepRequestState:
        self._require_pipeline_context(context, pp_stage_spec)
        if context.status is not PipelineTaskStatus.COMPLETED:
            raise RuntimeError("Pipeline batch must be completed before final decode.")
        state = context.states[0]
        if not state.request_denoise_completed:
            raise RuntimeError("Pipeline request has not completed its denoise schedule.")
        if state.latents is None:
            raise RuntimeError("Pipeline final decode has no latents to decode.")
        return state

    def finalize_pipeline_batch(
        self,
        context: PipelineBatchContext,
        pp_stage_spec: PipelineStageSpec,
        output_owner: bool | None = None,
    ) -> BatchRunnerOutput | None:
        """Decode on the assigned output owner, joining distributed VAE work when enabled."""
        state = self.validate_pipeline_finalization(context, pp_stage_spec)
        distributed_vae = self.pipeline_has_distributed_vae()
        if output_owner is None:
            output_owner = pp_stage_spec.is_first
        if type(output_owner) is not bool:
            raise TypeError("queued pipeline output_owner must be a bool")
        if not output_owner and not distributed_vae:
            raise ValueError("Only the assigned output owner can finalize queued output.")
        decode_owner_override = output_owner and not pp_stage_spec.is_first and not distributed_vae
        previous_decode_owner = getattr(self.pipeline, "_queued_pipeline_decode_owner", False)
        try:
            decode_start = time.perf_counter()
            if decode_owner_override:
                self.pipeline._queued_pipeline_decode_owner = True
            try:
                with (
                    self._pipeline_inference_context(),
                    set_forward_context(
                        vllm_config=self.vllm_config,
                        omni_diffusion_config=self.od_config,
                        attn_metadata={},
                        denoise_step_idx=context.task.step_index,
                    ),
                ):
                    result = self.pipeline.post_decode(state, queued_pipeline=True)
            finally:
                if decode_owner_override:
                    self.pipeline._queued_pipeline_decode_owner = previous_decode_owner
            decode_ms = (time.perf_counter() - decode_start) * 1000
            if not isinstance(result, DiffusionOutput):
                raise RuntimeError("Pipeline final decode produced no DiffusionOutput.")
            if distributed_vae and not pp_stage_spec.is_first:
                return None
            transport_start = time.perf_counter()
            result = self._prepare_output_for_transport(result, state.sampling)
            transport_ms = (time.perf_counter() - transport_start) * 1000
            logger.info(
                "Queued pipeline finalization batch=%s decode_ms=%.3f transport_prepare_ms=%.3f",
                context.task.batch_id,
                decode_ms,
                transport_ms,
            )
            self._attach_stepwise_metadata(state, result)
            return BatchRunnerOutput.from_list(
                [RunnerOutput(request_id=state.request_id, step_index=state.step_index, finished=True, result=result)]
            )
        except BaseException:
            context.status = PipelineTaskStatus.FAILED
            raise

    def release_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineBatchContext:
        key = (pp_stage_id, batch_id)
        context = self.pipeline_batch_contexts.get(key)
        if context is None:
            raise KeyError(f"Unknown pipeline batch context {key!r}.")
        if context.status not in {
            PipelineTaskStatus.COMPLETED,
            PipelineTaskStatus.CANCELLED,
            PipelineTaskStatus.FAILED,
        }:
            raise RuntimeError("Cannot release a non-terminal pipeline batch context.")
        context = self.pipeline_batch_contexts.pop(key)
        for request_id in context.request_state_ids:
            owner_key = (request_id, context.task.step_index)
            if self.pipeline_request_owners.get(owner_key) == key:
                self.pipeline_request_owners.pop(owner_key)
        return context

    def cancel_pipeline_batch(self, pp_stage_id: int, batch_id: str) -> PipelineBatchContext:
        key = (pp_stage_id, batch_id)
        context = self.pipeline_batch_contexts.get(key)
        if context is None:
            raise KeyError(f"Unknown pipeline batch context {key!r}.")
        if context.status is PipelineTaskStatus.CANCELLED:
            return context
        if context.status is PipelineTaskStatus.FAILED:
            raise RuntimeError("Cannot cancel a terminal pipeline batch context.")
        context.status = PipelineTaskStatus.CANCELLED
        return context

    def _require_pipeline_context(self, context: PipelineBatchContext, pp_stage_spec: PipelineStageSpec) -> None:
        key = (pp_stage_spec.pp_stage_id, context.task.batch_id)
        if self.pipeline_batch_contexts.get(key) is not context:
            raise ValueError(f"Pipeline batch context {key!r} is not owned by this ModelRunner.")
        if context.stage_spec != pp_stage_spec:
            raise ValueError("Pipeline batch context stage specification changed after preparation.")

    @staticmethod
    def _validate_pipeline_context_progress(context: PipelineBatchContext) -> None:
        mismatched = [
            (state.request_id, state.step_index)
            for state in context.states
            if state.step_index != context.task.step_index
        ]
        if mismatched:
            raise RuntimeError(
                f"Pipeline batch {context.task.batch_id!r} was prepared for step {context.task.step_index}, "
                f"but request progress changed: {mismatched!r}."
            )
