# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, cast

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.interface import SupportsInteractionApply, supports_interaction_apply
from vllm_omni.diffusion.sched.interface import (
    DiffusionSchedulerOutput,
    NewRequestData,
    validate_new_request_data_identity,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import _dit_any_rank_failed
from vllm_omni.diffusion.worker.input_batch import InputBatch
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


class QueuedModelRuntime:
    """Prepare, step, and decode request-local state for queued PP."""

    def __init__(self, model_runner: Any):
        object.__setattr__(self, "_runner", model_runner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_runner" or name.startswith("_queued_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._runner, name, value)

    def _pipeline_inference_context(self) -> AbstractContextManager[Any]:
        use_hsdp = bool(getattr(getattr(self.od_config, "parallel_config", None), "use_hsdp", False))
        return torch.no_grad() if use_hsdp else torch.inference_mode()

    def _run_queued_preparation_phase(
        self,
        request_id: str,
        phase: str,
        operation: Callable[[], Any],
    ) -> Any:
        """Run one rank-local preparation phase and agree on its outcome."""
        local_error: Exception | None = None
        result: Any = None
        try:
            result = operation()
        except Exception as exc:
            local_error = exc
        if _dit_any_rank_failed(local_error is not None):
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
        if not self._supports_step_mode():
            raise ValueError("The loaded diffusion pipeline does not support step execution.")
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
                merge_stage_durations(
                    state,
                    consume_pipeline_stage_durations(self.pipeline),
                )

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
        if self.pipeline is None or not getattr(self.pipeline, "supports_queued_pipeline", False):
            raise ValueError("The loaded diffusion pipeline does not support queued pipeline execution.")
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

    def execute_prepared_stepwise(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        output_rank: int = 0,
    ) -> BatchRunnerOutput:
        """Execute a queued microbatch from request state prepared at admission."""
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        states = [self.state_cache.get(request_id) for request_id in request_ids]
        if not request_ids or any(state is None for state in states):
            missing = [request_id for request_id, state in zip(request_ids, states, strict=True) if state is None]
            raise RuntimeError(f"Prepared step requests have no Worker state: {missing}")
        states = cast(list[StepRequestState], states)
        if int(getattr(self.od_config.parallel_config, "pipeline_parallel_size", 1)) <= 1:
            input_batch = InputBatch.make_batch(states, cached_batch=None)
            if input_batch is None:
                raise RuntimeError("Prepared step execution produced no input batch.")
        else:
            input_batch = None
        return self._execute_stepwise_core(
            scheduler_output,
            record_output_peak_memory=True,
            prepared_states=states,
            prepared_input_batch=input_batch,
            output_rank=output_rank,
        )

    def prepare_queued_decode(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Confirm completion on this rank without issuing new P2P operations."""
        from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents

        for request_id in request_ids:
            state = self.state_cache.get(request_id)
            if state is None or not state.denoise_completed:
                raise RuntimeError(f"Queued decode request {request_id!r} has not completed denoise.")
        # Also resolve feedback of paused, unfinished requests before draining
        # shared sends: a threshold can interrupt an otherwise active window.
        for state in self.state_cache.values():
            if isinstance(state.latents, AsyncLatents):
                state.latents = state.latents.resolve()
        sync_send = getattr(self.pipeline, "_sync_pp_send", None)
        if callable(sync_send):
            sync_send()
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
        return request_ids

    def execute_prepared_decode(self, request_id: str) -> BatchRunnerOutput:
        """Decode one completed queued request on the RPC-selected PP rank."""
        pipeline = self.pipeline
        assert pipeline is not None, "Model not loaded. Call load_model() first."
        state = self.state_cache.get(request_id)
        if state is None:
            raise RuntimeError(f"Queued decode request {request_id!r} has no Worker state.")

        from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_rank

        try:
            local_pp_rank = int(get_pipeline_parallel_rank())
        except AssertionError:
            local_pp_rank = 0
        previous_decode_owner = getattr(pipeline, "_queued_pipeline_decode_owner", False)
        if local_pp_rank != 0:
            pipeline._queued_pipeline_decode_owner = True
        try:
            with self._pipeline_inference_context(), set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata={},
            ):
                clear_pipeline_stage_durations(pipeline)
                result = pipeline.post_decode(state, queued_pipeline=True)
            if result is None:
                raise RuntimeError(f"Queued decode request {request_id!r} produced no output.")
            result = self._prepare_output_for_transport(result, state.sampling)
            self._attach_stepwise_metadata(state, result)
            self._maybe_send_stage_payload([state], [result])
            return BatchRunnerOutput.from_list(
                [
                    RunnerOutput(
                        request_id=request_id,
                        step_index=state.step_index,
                        finished=True,
                        result=result,
                    )
                ]
            )
        finally:
            if local_pp_rank != 0:
                pipeline._queued_pipeline_decode_owner = previous_decode_owner

    def _execute_stepwise_core(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        *,
        record_output_peak_memory: bool,
        in_diffusion_kv_memory_profile: bool = False,
        prepared_states: list[StepRequestState] | None = None,
        prepared_input_batch: InputBatch | None = None,
        output_rank: int | None = None,
    ) -> BatchRunnerOutput:
        """Run the denoise step after metadata admission and row installation."""

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            pipeline = self.pipeline
            assert pipeline is not None, "Model not loaded. Call load_model() first."
            queued_microbatches = (
                prepared_states is not None
                and int(getattr(self.od_config.parallel_config, "pipeline_parallel_size", 1)) > 1
            )
            had_active_states = bool(self.state_cache)
            if prepared_states is None:
                states, new_request_ids = self._update_states(scheduler_output)
            else:
                states, new_request_ids = prepared_states, []
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if (
                record_output_peak_memory
                and new_request_ids
                and not had_active_states
                and is_primary
                and current_omni_platform.is_available()
            ):
                current_omni_platform.reset_peak_memory_stats()
            if queued_microbatches:
                input_batch = None
                runner_output_list = []
            elif prepared_input_batch is None:
                states, input_batch, runner_output_list = self._prepare_batch_inputs(states, new_request_ids)
            else:
                input_batch = prepared_input_batch
                runner_output_list = []
            if input_batch is None and not queued_microbatches:
                return BatchRunnerOutput.from_list(runner_output_list)
            row_counts = (
                []
                if queued_microbatches
                else [state.latents.shape[0] if state.latents is not None else 0 for state in states]
            )
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
                def post_decode_request(req: StepRequestState) -> DiffusionOutput | None:
                    """Decode one request while preserving the queued PP owner."""
                    vae = getattr(pipeline, "vae", None)
                    is_distributed_vae = bool(
                        callable(getattr(vae, "is_distributed_enabled", None))
                        and vae.is_distributed_enabled()
                    )
                    from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_rank

                    try:
                        local_pp_rank = int(get_pipeline_parallel_rank())
                    except AssertionError:
                        local_pp_rank = 0
                    should_decode_here = (
                        output_rank is None
                        or is_distributed_vae
                        or local_pp_rank == output_rank
                    )
                    result = None
                    if should_decode_here:
                        clear_pipeline_stage_durations(pipeline)
                        previous_decode_owner = getattr(pipeline, "_queued_pipeline_decode_owner", False)
                        use_nonfirst_owner = (
                            output_rank is not None and local_pp_rank == output_rank and output_rank != 0
                        )
                        if use_nonfirst_owner:
                            pipeline._queued_pipeline_decode_owner = True
                        try:
                            result = pipeline.post_decode(
                                req,
                                queued_pipeline=queued_microbatches,
                            )
                        finally:
                            if use_nonfirst_owner:
                                pipeline._queued_pipeline_decode_owner = previous_decode_owner
                    if result is not None:
                        result = self._prepare_output_for_transport(result, req.sampling)
                        self._attach_stepwise_metadata(req, result)
                        if supports_interaction_apply(pipeline) and not req.request_denoise_completed:
                            pipe = cast(SupportsInteractionApply, pipeline)
                            pipe.apply_interaction_at_chunk_boundary(req)
                            pipe.prepare_next_chunk(req)
                    return result

                scheduler_errors: dict[str, Exception] = {}
                if queued_microbatches:
                    noise_pred = None
                    previous_async_send_mode = getattr(pipeline, "_queued_pipeline_async_sends", False)
                    pipeline._queued_pipeline_async_sends = True
                    try:
                        for state in states:
                            microbatch = InputBatch.make_batch([state], cached_batch=None)
                            if microbatch is None or microbatch.latents is None:
                                raise RuntimeError(f"Queued step request {state.request_id} produced no latent batch.")
                            row_counts.append(microbatch.latents.shape[0])
                            clear_pipeline_stage_durations(pipeline)
                            req_noise_pred = pipeline.denoise_step(microbatch, states=[state])
                            merge_stage_durations(state, consume_pipeline_stage_durations(pipeline))
                            if getattr(pipeline, "interrupt", False):
                                break
                            try:
                                advance_step = getattr(pipeline, "advance_step_for_pipeline_rank", None)
                                if callable(advance_step):
                                    advance_step(state, req_noise_pred)
                                else:
                                    pipeline.step_scheduler(state, req_noise_pred)
                            except Exception as exc:
                                scheduler_errors[state.request_id] = exc
                    finally:
                        pipeline._queued_pipeline_async_sends = previous_async_send_mode
                else:
                    clear_pipeline_stage_durations(pipeline)
                    noise_pred = pipeline.denoise_step(input_batch, states=states)
                    denoise_stage_durations = consume_pipeline_stage_durations(pipeline)
                    for state in states:
                        merge_stage_durations(state, denoise_stage_durations)

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
                    for req_index, req in enumerate(states):
                        if req.latents is None:
                            raise RuntimeError(f"Stepwise request {req.request_id} has no latent state.")
                        row_num = row_counts[req_index]
                        try:
                            if not queued_microbatches:
                                req_noise_pred = (
                                    noise_pred[offset : offset + row_num] if noise_pred is not None else None
                                )
                                pipeline.step_scheduler(req, req_noise_pred)
                            elif req.request_id in scheduler_errors:
                                raise scheduler_errors[req.request_id]
                            if self.od_config.streaming_output:
                                should_decode = req.chunk_denoise_completed or req.request_denoise_completed
                            else:
                                should_decode = req.denoise_completed

                            if should_decode:
                                # Final queued decode is submitted by Engine as a
                                # request-local RPC so different output owners can
                                # decode on different PP ranks concurrently.
                                result = (
                                    None
                                    if queued_microbatches and not self.od_config.streaming_output
                                    else post_decode_request(req)
                                )
                            else:
                                result = None
                            # finished should be computed after post_decode() advanced chunk_index
                            finished = (
                                req.request_denoise_completed
                                if self.od_config.streaming_output
                                else req.denoise_completed
                            )
                            if finished and result is not None:
                                self._maybe_send_stage_payload([req], [result])
                            runner_output_list.append(
                                RunnerOutput(
                                    request_id=req.request_id,
                                    step_index=req.step_index,
                                    finished=finished,
                                    result=result,
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

                    if not queued_microbatches and noise_pred is not None and offset != noise_pred.shape[0]:
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
                if prepared_states is None:
                    self._update_states_after(states, input_batch, pipeline_interrupted)
                if (
                    getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
                    is DiffusionKVCacheMode.PAGED_SCHEDULER
                    and terminal_request_ids
                ):
                    self.remove_diffusion_kv_requests(terminal_request_ids)

                return BatchRunnerOutput.from_list(runner_output_list)
