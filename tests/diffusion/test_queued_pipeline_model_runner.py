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
    supports_step_execution = True
    supports_pipeline_stage_execution = True

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.forward_context_seen = None

    def prepare_encode(self, state) -> None:
        del state
        self.prepare_calls += 1

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        self.denoise_calls += 1
        raise AssertionError("queued preparation must not denoise")

    def step_scheduler(self, state, noise_pred) -> None:
        del state, noise_pred

    def post_decode(self, state, **kwargs):
        del state, kwargs
        return None

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
    runner.device = torch.device("cpu")
    runner.vllm_config = None
    runner.od_config = SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False))
    runner.diffusion_kv_backend = None
    runner.input_batch = SimpleNamespace(marker="shared-static-batch")
    runner._pipeline_batch_contexts = {}
    runner._pipeline_request_owners = {}
    return runner


def _task(batch_id: str = "batch-a") -> PipelineTask:
    return PipelineTask(batch_id=batch_id, request_ids=("req-a",), step_index=0, epoch=1)


def test_prepare_pipeline_requests_retains_local_state_without_denoising(mocker) -> None:
    runner = _runner()
    runner.od_config.cache_backend = None
    runner.state_cache = {}
    runner.input_batch = None
    state = _state()
    new_request = SimpleNamespace(
        request_id="req-a",
        req=SimpleNamespace(request_id="req-a", use_step_execution=True),
        diffusion_kv_metadata=None,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        finished_req_ids=set(),
    )

    def update_states(_scheduler_output):
        runner.state_cache[state.request_id] = state
        return [state], [state.request_id]

    mocker.patch.object(runner, "_update_states", side_effect=update_states)

    request_ids = runner.prepare_pipeline_requests(scheduler_output)

    assert request_ids == ("req-a",)
    assert runner.state_cache == {"req-a": state}
    assert runner.pipeline.prepare_calls == 1
    assert runner.pipeline.denoise_calls == 0
    assert runner.pipeline.scheduler_calls == 0


def test_prepare_pipeline_requests_rolls_back_local_state_failure(mocker) -> None:
    runner = _runner()
    runner.od_config.cache_backend = None
    runner.state_cache = {}
    new_request = SimpleNamespace(
        request_id="req-a",
        req=SimpleNamespace(request_id="req-a", use_step_execution=True),
        diffusion_kv_metadata=None,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        finished_req_ids=set(),
    )

    def fail_update(_scheduler_output):
        runner.state_cache["req-a"] = _state()
        raise RuntimeError("local state allocation failed")

    mocker.patch.object(runner, "_update_states", side_effect=fail_update)
    failure_agreement = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner._dit_any_rank_failed",
        side_effect=lambda failed: failed,
    )

    with pytest.raises(RuntimeError, match="local state allocation failed"):
        runner.prepare_pipeline_requests(scheduler_output)

    failure_agreement.assert_called_once_with(True)
    assert runner.state_cache == {}
    assert runner.input_batch is None
    assert runner.pipeline.prepare_calls == 0
    assert runner.pipeline.denoise_calls == 0


def test_prepare_pipeline_requests_agrees_and_rolls_back_metadata_install_failure(mocker) -> None:
    runner = _runner()
    runner.od_config.cache_backend = None
    runner.state_cache = {}
    metadata = SimpleNamespace(request_id="req-a")
    new_request = SimpleNamespace(
        request_id="req-a",
        req=SimpleNamespace(request_id="req-a", use_step_execution=True),
        diffusion_kv_metadata=metadata,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        finished_req_ids=set(),
    )
    mocker.patch.object(runner, "_validate_diffusion_kv_metadata")
    install = mocker.patch.object(
        runner,
        "install_diffusion_kv_metadata",
        side_effect=RuntimeError("metadata install failed"),
    )
    remove = mocker.patch.object(runner, "remove_diffusion_kv_requests", return_value=0)
    update_states = mocker.patch.object(runner, "_update_states")
    failure_agreement = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner._dit_any_rank_failed",
        side_effect=lambda failed: failed,
    )

    with pytest.raises(RuntimeError, match="metadata install failed"):
        runner.prepare_pipeline_requests(scheduler_output)

    install.assert_called_once_with(metadata)
    failure_agreement.assert_called_once_with(True)
    update_states.assert_not_called()
    remove.assert_called_once_with(["req-a"])
    assert runner.state_cache == {}
    assert runner.input_batch is None
    assert runner.pipeline.prepare_calls == 0
    assert runner.pipeline.denoise_calls == 0


def test_prepare_pipeline_requests_agrees_before_encode_on_generator_failure(mocker) -> None:
    runner = _runner()
    runner.od_config.cache_backend = None
    runner.state_cache = {}
    runner.input_batch = None
    state = _state()
    new_request = SimpleNamespace(
        request_id="req-a",
        req=SimpleNamespace(request_id="req-a", use_step_execution=True),
        diffusion_kv_metadata=None,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        finished_req_ids=set(),
    )

    def update_states(_scheduler_output):
        runner.state_cache[state.request_id] = state
        return [state], [state.request_id]

    mocker.patch.object(runner, "_update_states", side_effect=update_states)
    mocker.patch.object(
        runner,
        "_initialize_generator",
        side_effect=RuntimeError("generator setup failed"),
    )
    failure_agreement = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner._dit_any_rank_failed",
        side_effect=lambda failed: failed,
    )

    with pytest.raises(RuntimeError, match="generator setup failed"):
        runner.prepare_pipeline_requests(scheduler_output)

    assert [item.args for item in failure_agreement.call_args_list] == [(False,), (True,)]
    assert runner.state_cache == {}
    assert runner.input_batch is None
    assert runner.pipeline.prepare_calls == 0
    assert runner.pipeline.denoise_calls == 0


def test_prepare_pipeline_requests_rolls_back_successful_peer_on_late_remote_failure(mocker) -> None:
    runner = _runner()
    runner.od_config.cache_backend = None
    runner.state_cache = {}
    runner.input_batch = None
    state = _state()
    metadata = SimpleNamespace(request_id="req-a")
    new_request = SimpleNamespace(
        request_id="req-a",
        req=SimpleNamespace(request_id="req-a", use_step_execution=True),
        diffusion_kv_metadata=metadata,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        finished_req_ids=set(),
    )

    def update_states(_scheduler_output):
        runner.state_cache[state.request_id] = state
        return [state], [state.request_id]

    mocker.patch.object(runner, "_update_states", side_effect=update_states)
    mocker.patch.object(runner, "_validate_diffusion_kv_metadata")
    install = mocker.patch.object(runner, "install_diffusion_kv_metadata", return_value=True)
    remove = mocker.patch.object(runner, "remove_diffusion_kv_requests", return_value=1)
    failure_agreement = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner._dit_any_rank_failed",
        side_effect=[False, False, False, True],
    )

    with pytest.raises(RuntimeError, match="batch construction failed on another rank"):
        runner.prepare_pipeline_requests(scheduler_output)

    assert failure_agreement.call_count == 4
    install.assert_called_once_with(metadata)
    remove.assert_called_once_with(["req-a"])
    assert runner.state_cache == {}
    assert runner.input_batch is None
    assert runner.pipeline.prepare_calls == 1
    assert runner.pipeline.denoise_calls == 0


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


def test_cancel_pipeline_batch_requires_explicit_release() -> None:
    runner = _runner()
    spec = PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False)
    context = runner.prepare_pipeline_batch(_task(), spec, [_state()])

    assert runner.cancel_pipeline_batch(0, "batch-a") is context
    assert context.status is PipelineTaskStatus.CANCELLED
    assert runner.pipeline_request_owners == {("req-a", 0): (0, "batch-a")}
    assert runner.release_pipeline_batch(0, "batch-a") is context
    assert runner.pipeline_request_owners == {}


def test_cancel_pipeline_batch_overrides_completed_local_work() -> None:
    runner = _runner()
    state = _state()
    spec = PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True)
    context = runner.prepare_pipeline_batch(_task(), spec, [state])
    runner.execute_pipeline_stage(context, spec, intermediate_tensors=object())
    runner.complete_pipeline_step(context, spec)

    assert context.status is PipelineTaskStatus.COMPLETED
    assert runner.cancel_pipeline_batch(1, "batch-a") is context
    assert context.status is PipelineTaskStatus.CANCELLED
    assert runner.cancel_pipeline_batch(1, "batch-a") is context
    assert runner.release_pipeline_batch(1, "batch-a") is context


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
