# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineStageSpec,
    PipelineTask,
    PipelineTaskStatus,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _QueuedPipeline:
    supports_pipeline_stage_execution = True

    def __init__(self) -> None:
        self.scheduler_calls = 0
        self.forward_context_seen = None

    def validate_pipeline_stage_execution(self, pp_stage_spec: PipelineStageSpec) -> None:
        if pp_stage_spec.world_size != 2:
            raise ValueError("bad topology")

    def forward_pipeline_stage(self, input_batch, *, pp_stage_spec, intermediate_tensors, states):
        del intermediate_tensors, states
        self.forward_context_seen = get_forward_context()
        if pp_stage_spec.is_last:
            return torch.ones_like(input_batch.latents)
        return SimpleNamespace(tensors={"hidden_states": input_batch.latents + 1})

    def step_scheduler_pipeline_stage(self, state, noise_pred) -> None:
        self.scheduler_calls += 1
        state.latents = state.latents + noise_pred
        state.step_index += 1


def _state(request_id: str = "req-a") -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(guidance_scale=1.0, num_inference_steps=2),
        prompt="prompt",
    )
    state.latents = torch.zeros(1, 4, 1, 2, 2)
    state.timesteps = torch.tensor([2.0, 1.0])
    state.prompt_embeds = torch.zeros(1, 4, 8)
    return state


def _runner() -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = _QueuedPipeline()
    runner.vllm_config = None
    runner.od_config = SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False))
    runner.diffusion_kv_backend = None
    runner.input_batch = SimpleNamespace(marker="shared-static-batch")
    runner._pipeline_batch_contexts = {}
    runner._pipeline_request_owners = {}
    return runner


def _task(batch_id: str = "batch-a") -> PipelineTask:
    return PipelineTask(batch_id=batch_id, request_ids=("req-a",), step_index=0, epoch=1)


def test_prepare_pipeline_batch_owns_independent_context() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)

    context = runner.prepare_pipeline_batch(_task(), spec, [state])

    assert context.input_batch is not runner.input_batch
    assert context.states == (state,)
    assert runner.pipeline_batch_contexts[(0, "batch-a")] is context
    with pytest.raises(ValueError, match="already exists"):
        runner.prepare_pipeline_batch(_task(), spec, [state])


def test_prepare_pipeline_batch_rejects_stale_step_identity() -> None:
    runner = _runner()
    state = _state()
    state.step_index = 1
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)

    with pytest.raises(ValueError, match="does not match request state step"):
        runner.prepare_pipeline_batch(_task(), spec, [state])


def test_prepare_pipeline_batch_rejects_overlapping_request_step_owner() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    runner.prepare_pipeline_batch(_task(), spec, [state])

    with pytest.raises(ValueError, match="already owned"):
        runner.prepare_pipeline_batch(_task("batch-b"), spec, [state])


def test_last_stage_executes_and_updates_scheduler_exactly_once() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])

    result = runner.execute_pipeline_stage(context, spec, intermediate_tensors=object())
    feedback = runner.complete_pipeline_step(context, spec)

    torch.testing.assert_close(result, torch.ones_like(state.latents))
    torch.testing.assert_close(feedback, torch.ones_like(feedback))
    assert runner.pipeline.scheduler_calls == 1
    assert state.step_index == 1
    assert context.status is PipelineTaskStatus.COMPLETED
    assert runner.release_pipeline_batch(1, "batch-a") is context
    assert runner.pipeline_request_owners == {}


def test_local_stage_executes_with_real_forward_context() -> None:
    runner = _runner()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    context = runner.prepare_pipeline_batch(_task(), spec, [_state()])

    assert not is_forward_context_available()
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=None)

    seen = runner.pipeline.forward_context_seen
    assert seen is not None
    assert seen.vllm_config is runner.vllm_config
    assert seen.omni_diffusion_config is runner.od_config
    assert seen.denoise_step_idx == 0
    assert not is_forward_context_available()


def test_last_stage_empty_latent_postcondition_marks_context_failed() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=object())

    def clear_latents(state, noise_pred) -> None:
        del noise_pred
        state.latents = None

    runner.pipeline.step_scheduler_pipeline_stage = clear_latents

    with pytest.raises(RuntimeError, match="produced no latents"):
        runner.complete_pipeline_step(context, spec)

    assert context.status is PipelineTaskStatus.FAILED
    assert runner.release_pipeline_batch(1, "batch-a") is context


def test_last_stage_non_tensor_result_marks_context_failed() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    context.status = PipelineTaskStatus.ACTIVE
    context.result = object()

    with pytest.raises(RuntimeError, match="non-tensor result"):
        runner.complete_pipeline_step(context, spec)

    assert context.status is PipelineTaskStatus.FAILED
    assert runner.release_pipeline_batch(1, "batch-a") is context


def test_first_stage_adopts_feedback_without_scheduler_update() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=None)
    feedback = torch.full_like(state.latents, 7.0)

    runner.adopt_pipeline_feedback(context, spec, feedback)

    torch.testing.assert_close(state.latents, feedback)
    assert context.input_batch.latents is state.latents
    assert state.step_index == 1
    assert runner.pipeline.scheduler_calls == 0
    assert context.status is PipelineTaskStatus.COMPLETED


def test_feedback_adoption_mutates_inference_tensor_inside_inference_scope() -> None:
    runner = _runner()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    with torch.inference_mode():
        state = _state()
        feedback = torch.full_like(state.latents, 9.0)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=None)

    runner.adopt_pipeline_feedback(context, spec, feedback)

    torch.testing.assert_close(state.latents, feedback)
    assert context.status is PipelineTaskStatus.COMPLETED


def test_release_rejects_non_terminal_context() -> None:
    runner = _runner()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    runner.prepare_pipeline_batch(_task(), spec, [_state()])

    with pytest.raises(RuntimeError, match="non-terminal"):
        runner.release_pipeline_batch(0, "batch-a")


def test_context_rejects_changed_stage_specification() -> None:
    runner = _runner()
    first = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    context = runner.prepare_pipeline_batch(_task(), first, [_state()])
    forged = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=False, is_last=True)

    with pytest.raises(ValueError, match="changed after preparation"):
        runner.execute_pipeline_stage(context, forged, intermediate_tensors=None)


def test_feedback_rejects_mismatched_latent_shape() -> None:
    runner = _runner()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    context = runner.prepare_pipeline_batch(_task(), spec, [_state()])
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=None)

    with pytest.raises(ValueError, match="do not match"):
        runner.adopt_pipeline_feedback(context, spec, torch.zeros(1, 4, 1, 3, 3))

    assert context.status is PipelineTaskStatus.FAILED
    assert runner.release_pipeline_batch(0, "batch-a") is context


@pytest.mark.parametrize("phase", ["forward", "completion", "feedback"])
def test_context_revalidates_request_progress_before_each_phase(phase: str) -> None:
    runner = _runner()
    is_last = phase == "completion"
    spec = PipelineStageSpec(pp_stage_id=int(is_last), world_size=2, is_first=not is_last, is_last=is_last)
    state = _state()
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    if phase != "forward":
        runner.execute_pipeline_stage(context, spec, intermediate_tensors=None if not is_last else object())
    state.step_index = 1

    with pytest.raises(RuntimeError, match="request progress changed"):
        if phase == "forward":
            runner.execute_pipeline_stage(context, spec, intermediate_tensors=None)
        elif phase == "completion":
            runner.complete_pipeline_step(context, spec)
        else:
            runner.adopt_pipeline_feedback(context, spec, torch.zeros_like(state.latents))

    assert context.status is PipelineTaskStatus.FAILED
