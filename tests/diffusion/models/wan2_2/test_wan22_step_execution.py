# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.distributed.pipeline_parallel as pp_module
import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.models.interface import (
    SupportsPipelineStageExecution,
    SupportsStepExecution,
    supports_pipeline_stage_execution,
    supports_step_execution,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.pipeline_state import PipelineStageSpec
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Transformer(nn.Module):
    config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=4, out_channels=4)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _Scheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.timesteps = torch.empty(0)
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.float32, device=device)

    def step(self, noise_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        self.step_calls.append((noise_pred, timestep, latents))
        return (latents + noise_pred,)


class _VAE:
    dtype = torch.float32
    config = SimpleNamespace(
        z_dim=4,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )

    def decode(self, latents, return_dict=False):
        assert return_dict is False
        return (latents[:, :3],)


class _PPGroup:
    world_size = 2

    def __init__(self, is_first_rank: bool, broadcast_fn) -> None:
        self.is_first_rank = is_first_rank
        self.cpu_group = object()
        self._broadcast_fn = broadcast_fn

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        return self._broadcast_fn(tensor, src)


class _UnpickleableWork:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True

    def __reduce__(self):
        raise TypeError("distributed work handles cannot be pickled")


def _pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _Transformer()
    pipeline.transformer_2 = None
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline.vae = _VAE()
    pipeline.vae_scale_factor_temporal = pipeline.vae.config.scale_factor_temporal
    pipeline.vae_scale_factor_spatial = pipeline.vae.config.scale_factor_spatial
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline.scheduler = _Scheduler()
    pipeline.boundary_ratio = None
    pipeline.expand_timesteps = True
    pipeline.has_transformer_2 = False
    pipeline.is_dmd = False
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **_kwargs: None

    def encode_prompt(**kwargs):
        shape = (kwargs["num_videos_per_prompt"], kwargs["max_sequence_length"], 8)
        negative = torch.full(shape, -2.0) if kwargs["do_classifier_free_guidance"] else None
        return torch.full(shape, 2.0), negative

    pipeline.encode_prompt = encode_prompt
    pipeline.prepare_latents = lambda **kwargs: torch.randn(
        kwargs["batch_size"],
        kwargs["num_channels_latents"],
        (kwargs["num_frames"] - 1) // 4 + 1,
        kwargs["height"] // 8,
        kwargs["width"] // 8,
        generator=kwargs["generator"],
    )
    return pipeline


def _state(
    *,
    request_id: str = "req",
    seed: int | None = 7,
    output_type: str = "latent",
) -> StepRequestState:
    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    sampling = OmniDiffusionSamplingParams(
        height=16,
        width=16,
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        max_sequence_length=4,
        output_type=output_type,
        seed=seed,
        generator=generator,
    )
    return StepRequestState(request_id=request_id, sampling=sampling, prompt="a quiet lake")


def _patch_scheduler(monkeypatch) -> None:
    monkeypatch.setattr(wan22_module, "build_wan_scheduler", lambda *_args, **_kwargs: _Scheduler())


def _patch_pp_topology(monkeypatch, *, rank: int, world_size: int = 2) -> None:
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: world_size)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_rank", lambda: rank)
    monkeypatch.setattr(wan22_module, "is_pipeline_first_stage", lambda: rank == 0)
    monkeypatch.setattr(wan22_module, "is_pipeline_last_stage", lambda: rank == world_size - 1)


def test_wan22_declares_step_execution_capability() -> None:
    pipeline = _pipeline()

    assert isinstance(pipeline, SupportsStepExecution)
    assert supports_step_execution(pipeline)
    assert isinstance(pipeline, SupportsPipelineStageExecution)
    assert supports_pipeline_stage_execution(pipeline)

    prepare_only = SimpleNamespace(supports_step_execution=True, prepare_encode=lambda *_args, **_kwargs: None)
    assert not isinstance(prepare_only, SupportsStepExecution)
    assert not supports_step_execution(prepare_only)


def test_prepare_encode_creates_request_local_unipc_state(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    first = _state(request_id="first")
    second = _state(request_id="second")

    pipeline.prepare_encode(first)
    pipeline.prepare_encode(second)

    assert first.scheduler is not second.scheduler
    assert first.scheduler is not pipeline.scheduler
    assert first.step_index == 0
    assert first.do_true_cfg is False
    assert first.timesteps.tolist() == [2.0, 1.0]
    assert first.latents.shape == (1, 4, 1, 2, 2)
    assert first.prompt_embeds.shape == (1, 4, 8)
    assert first.extra["wan_boundary_timestep"] == pytest.approx(875.0)


def test_prepare_encode_prefers_precomputed_conditioning_with_real_validator(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    pipeline.check_inputs = Wan22Pipeline.check_inputs.__get__(pipeline)
    pipeline.encode_prompt = lambda **_kwargs: pytest.fail("precomputed conditioning must skip text encoding")
    prompt_embeds = torch.full((4, 8), 3.0)
    negative_prompt_embeds = torch.full((4, 8), -3.0)
    state = _state()
    state.sampling.guidance_scale = 4.0
    state.prompt = {
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }

    pipeline.prepare_encode(state)

    torch.testing.assert_close(state.prompt_embeds, prompt_embeds.unsqueeze(0))
    torch.testing.assert_close(state.negative_prompt_embeds, negative_prompt_embeds.unsqueeze(0))
    assert state.do_true_cfg is True


def test_prepare_encode_normalizes_2d_positive_embeds_before_encoding_negative_prompt(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    pipeline.check_inputs = Wan22Pipeline.check_inputs.__get__(pipeline)
    encode_batch_sizes: list[int] = []

    def encode_prompt(**kwargs):
        prompt = kwargs["prompt"]
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        encode_batch_sizes.append(batch_size)
        shape = (batch_size * kwargs["num_videos_per_prompt"], kwargs["max_sequence_length"], 8)
        return torch.zeros(shape), torch.full(shape, -2.0)

    pipeline.encode_prompt = encode_prompt
    state = _state()
    state.sampling.guidance_scale = 4.0
    state.prompt = {"prompt": "a cat", "prompt_embeds": torch.full((4, 8), 3.0)}

    pipeline.prepare_encode(state)

    assert encode_batch_sizes == [1]
    assert state.prompt_embeds.shape == (1, 4, 8)
    assert state.negative_prompt_embeds.shape == (1, 4, 8)


def test_prepare_encode_preserves_supplied_latents(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    pipeline.prepare_latents = Wan22Pipeline.prepare_latents.__get__(pipeline)
    supplied = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)
    state = _state()
    state.sampling.latents = supplied

    pipeline.prepare_encode(state)

    assert state.latents is supplied


@pytest.mark.parametrize("seed", [7, None], ids=["seeded", "unseeded"])
def test_prepare_encode_broadcasts_stage_zero_initial_latents(monkeypatch, seed) -> None:
    _patch_scheduler(monkeypatch)
    source_latents: list[torch.Tensor] = []

    def gather_statuses(statuses, local_status, *, group) -> None:
        del local_status, group
        statuses[:] = [None, None]

    monkeypatch.setattr(wan22_module.torch.distributed, "all_gather_object", gather_statuses)

    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 2)
    first_pipeline = _pipeline()

    def first_broadcast(tensor: torch.Tensor, src: int) -> torch.Tensor:
        assert src == 0
        source_latents.append(tensor.clone())
        return tensor

    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: _PPGroup(True, first_broadcast))
    first = _state(request_id="first", seed=seed)
    first_pipeline.prepare_encode(first)

    last_pipeline = _pipeline()
    last_pipeline.prepare_latents = lambda **_kwargs: pytest.fail("non-first rank must not sample initial latents")

    def last_broadcast(tensor: torch.Tensor, src: int) -> torch.Tensor:
        assert src == 0
        tensor.copy_(source_latents[0])
        return tensor

    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: _PPGroup(False, last_broadcast))
    last = _state(request_id="last", seed=seed)
    last_pipeline.prepare_encode(last)

    torch.testing.assert_close(last.latents, first.latents)
    assert len(source_latents) == 1


@pytest.mark.parametrize("is_first_rank", [True, False], ids=["first-rank", "non-first-rank"])
def test_prepare_encode_coordinates_generator_mismatch_before_broadcast(monkeypatch, is_first_rank) -> None:
    _patch_scheduler(monkeypatch)
    pipeline = _pipeline()
    state = _state()
    state.sampling.generator = [torch.Generator(), torch.Generator()]
    broadcasts: list[torch.Tensor] = []
    group = _PPGroup(is_first_rank, lambda tensor, _src: broadcasts.append(tensor) or tensor)
    gathered: list[str | None] = []

    def gather_statuses(statuses, local_status, *, group: object) -> None:
        gathered.append(local_status)
        statuses[:] = [local_status, None] if is_first_rank else [None, local_status]

    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(wan22_module.torch.distributed, "all_gather_object", gather_statuses)

    with pytest.raises(RuntimeError, match="Generator list length 2 does not match batch size 1"):
        pipeline.prepare_encode(state)

    assert gathered and gathered[0].startswith("ValueError:")
    assert broadcasts == []


def test_prepare_encode_coordinates_rank_local_allocation_failure_before_broadcast(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    pipeline = _pipeline()
    state = _state()
    broadcasts: list[torch.Tensor] = []
    group = _PPGroup(False, lambda tensor, _src: broadcasts.append(tensor) or tensor)
    gathered: list[str | None] = []

    def gather_statuses(statuses, local_status, *, group: object) -> None:
        gathered.append(local_status)
        statuses[:] = [None, local_status]

    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(wan22_module.torch.distributed, "all_gather_object", gather_statuses)
    monkeypatch.setattr(
        wan22_module.torch, "empty", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("oom"))
    )

    with pytest.raises(RuntimeError, match="RuntimeError: oom"):
        pipeline.prepare_encode(state)

    assert gathered == ["RuntimeError: oom"]
    assert broadcasts == []


def test_prepare_encode_rejects_deferred_modes(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)

    image_pipeline = _pipeline()
    image_state = _state()
    image_state.prompt = {"prompt": "animate", "multi_modal_data": {"image": torch.zeros(3, 16, 16)}}
    with pytest.raises(ValueError, match="text-only T2V"):
        image_pipeline.prepare_encode(image_state)

    cascade_pipeline = _pipeline()
    cascade_pipeline.has_transformer_2 = True
    with pytest.raises(ValueError, match="single-transformer"):
        cascade_pipeline.prepare_encode(_state())


def test_prepare_encode_rejects_cfg_with_pipeline_parallelism(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    broadcasts: list[torch.Tensor] = []
    group = _PPGroup(True, lambda tensor, _src: broadcasts.append(tensor) or tensor)

    def gather_statuses(statuses, local_status, *, group: object) -> None:
        statuses[:] = [local_status, None]

    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(wan22_module.torch.distributed, "all_gather_object", gather_statuses)
    pipeline = _pipeline()
    state = _state()
    state.sampling.guidance_scale = 4.0

    with pytest.raises(RuntimeError, match="classifier-free guidance with PP>1"):
        pipeline.prepare_encode(state)

    assert broadcasts == []


def test_denoise_and_scheduler_use_request_local_state_once(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    state = _state()
    pipeline.prepare_encode(state)
    batch = InputBatch.make_batch([state])
    captured: dict[str, object] = {}

    def predict_noise_maybe_with_cfg(**kwargs):
        captured.update(kwargs)
        return torch.ones_like(batch.latents)

    pipeline.predict_noise_maybe_with_cfg = predict_noise_maybe_with_cfg
    noise_pred = pipeline.denoise_step(batch, states=[state])
    assert noise_pred is not None
    pipeline.step_scheduler(state, noise_pred)

    assert captured["do_true_cfg"] is False
    assert captured["negative_kwargs"] is None
    positive_kwargs = captured["positive_kwargs"]
    assert positive_kwargs["current_model"] is pipeline.transformer
    torch.testing.assert_close(positive_kwargs["hidden_states"], batch.latents)
    assert len(state.scheduler.step_calls) == 1
    assert len(pipeline.scheduler.step_calls) == 0
    assert state.step_index == 1


@pytest.mark.parametrize("rank", [0, 1], ids=["first-stage", "last-stage"])
def test_forward_pipeline_stage_runs_local_partition_without_pp_wrapper(monkeypatch, rank) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    state = _state()
    pipeline.prepare_encode(state)
    batch = InputBatch.make_batch([state])
    intermediate = None if rank == 0 else object()
    captured: dict[str, object] = {}
    pipeline.predict_noise_maybe_with_cfg = lambda **_kwargs: pytest.fail("queued local stage must bypass PP wrapper")

    def predict_noise(**kwargs):
        captured.update(kwargs)
        return torch.ones_like(batch.latents)

    pipeline.predict_noise = predict_noise
    _patch_pp_topology(monkeypatch, rank=rank)

    output = pipeline.forward_pipeline_stage(
        batch,
        pp_stage_spec=PipelineStageSpec(
            pp_stage_id=rank,
            world_size=2,
            is_first=rank == 0,
            is_last=rank == 1,
        ),
        intermediate_tensors=intermediate,
        states=[state],
    )

    torch.testing.assert_close(output, torch.ones_like(batch.latents))
    assert captured["intermediate_tensors"] is intermediate
    assert captured["current_model"] is pipeline.transformer


def test_forward_pipeline_stage_rejects_cfg_and_non_m2_topology(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    cfg_state = _state()
    cfg_state.sampling.guidance_scale = 4.0
    pipeline.prepare_encode(cfg_state)
    cfg_batch = InputBatch.make_batch([cfg_state])
    _patch_pp_topology(monkeypatch, rank=0)

    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        pipeline.forward_pipeline_stage(
            cfg_batch,
            pp_stage_spec=PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False),
            intermediate_tensors=None,
            states=[cfg_state],
        )


@pytest.mark.parametrize(
    ("actual_rank", "actual_world_size", "spec", "message"),
    [
        (0, 2, PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True), "does not match PP rank"),
        (
            0,
            2,
            PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=False, is_last=True),
            "endpoint flags",
        ),
        (0, 3, PipelineStageSpec(pp_stage_id=0, world_size=3, is_first=True, is_last=False), "exactly two stages"),
    ],
)
def test_pipeline_stage_validation_rejects_topology_mismatch(
    monkeypatch,
    actual_rank,
    actual_world_size,
    spec,
    message,
) -> None:
    pipeline = _pipeline()
    _patch_pp_topology(monkeypatch, rank=actual_rank, world_size=actual_world_size)

    with pytest.raises(ValueError, match=message):
        pipeline.validate_pipeline_stage_execution(spec)


def test_pp1_step_cfg_matches_request_denoise_math(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    state = _state()
    state.sampling.guidance_scale = 4.0
    pipeline.prepare_encode(state)

    def predict_noise(**kwargs):
        value = kwargs["encoder_hidden_states"].mean()
        return torch.full_like(kwargs["hidden_states"], value)

    pipeline.predict_noise = predict_noise
    initial_latents = state.latents.clone()
    reference_scheduler = _Scheduler()
    pipeline.scheduler = reference_scheduler
    reference = pipeline.diffuse(
        latents=initial_latents.clone(),
        timesteps=state.timesteps[:1],
        prompt_embeds=state.prompt_embeds,
        negative_prompt_embeds=state.negative_prompt_embeds,
        guidance_low=4.0,
        guidance_high=4.0,
        boundary_timestep=state.extra["wan_boundary_timestep"],
        dtype=torch.float32,
        attention_kwargs={},
    )

    batch = InputBatch.make_batch([state])
    noise_pred = pipeline.denoise_step(batch, states=[state])
    pipeline.step_scheduler(state, noise_pred)

    torch.testing.assert_close(state.latents, reference)
    assert state.do_true_cfg is True
    assert len(state.scheduler.step_calls) == 1


def test_denoise_cfg_uses_per_row_guidance_across_timestep_boundary(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    high_noise = _state(request_id="high-noise")
    low_noise = _state(request_id="low-noise")
    for state in (high_noise, low_noise):
        state.sampling.guidance_scale = 2.0
        state.sampling.guidance_scale_2 = 5.0
        state.sampling.guidance_scale_2_provided = True
        pipeline.prepare_encode(state)
    high_noise.timesteps = torch.tensor([900.0])
    low_noise.timesteps = torch.tensor([100.0])
    batch = InputBatch.make_batch([high_noise, low_noise])
    captured: dict[str, object] = {}

    def predict_noise_maybe_with_cfg(**kwargs):
        captured.update(kwargs)
        return torch.ones_like(batch.latents)

    pipeline.predict_noise_maybe_with_cfg = predict_noise_maybe_with_cfg

    pipeline.denoise_step(batch, states=[high_noise, low_noise])

    assert captured["do_true_cfg"] is True
    assert captured["negative_kwargs"] is not None
    scale = captured["true_cfg_scale"]
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(scale.flatten(), torch.tensor([2.0, 5.0]))


def test_mixed_cfg_batch_matches_independent_positive_and_cfg_predictions(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    high_noise = _state(request_id="positive-only")
    low_noise = _state(request_id="cfg")
    for state in (high_noise, low_noise):
        state.sampling.guidance_scale = 0.0
        state.sampling.guidance_scale_2 = 5.0
        state.sampling.guidance_scale_2_provided = True
        pipeline.prepare_encode(state)
    high_noise.timesteps = torch.tensor([900.0])
    low_noise.timesteps = torch.tensor([100.0])

    def predict_noise(**kwargs):
        value = kwargs["encoder_hidden_states"].mean()
        return torch.full_like(kwargs["hidden_states"], value)

    pipeline.predict_noise = predict_noise
    independent = torch.cat(
        [pipeline.denoise_step(InputBatch.make_batch([state]), states=[state]) for state in (high_noise, low_noise)]
    )
    batched = pipeline.denoise_step(
        InputBatch.make_batch([high_noise, low_noise]),
        states=[high_noise, low_noise],
    )

    torch.testing.assert_close(batched, independent)
    torch.testing.assert_close(batched[:, 0, 0, 0, 0], torch.tensor([2.0, 18.0]))


def test_post_decode_matches_latent_and_video_output_contract(monkeypatch) -> None:
    monkeypatch.setattr(wan22_module.current_omni_platform, "is_available", lambda: False)
    pipeline = _pipeline()
    latent_state = _state(output_type="latent")
    latent_state.latents = torch.ones(1, 4, 1, 2, 2)
    latent_state.extra["wan_output_type"] = "latent"

    latent_output = pipeline.post_decode(latent_state)

    assert latent_output.output is latent_state.latents
    video_state = _state(output_type="np")
    video_state.latents = torch.ones(1, 4, 1, 2, 2)
    video_state.extra["wan_output_type"] = "np"

    video_output = pipeline.post_decode(video_state)

    assert video_output.output is None
    assert video_output.media is not None
    assert video_output.media.video.tensor.shape == (1, 3, 1, 2, 2)


def test_post_decode_materializes_async_latents_before_runner_ipc(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module.current_omni_platform, "is_available", lambda: False)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
    pipeline = _pipeline()
    state = _state(output_type="latent")
    pipeline.prepare_encode(state)
    tensor = torch.arange(300_000, dtype=torch.float32).reshape(1, 4, 1, 300, 250)
    work = _UnpickleableWork()
    state.latents = AsyncLatents({"latents": tensor}, [work], [])
    state.extra["wan_output_type"] = "latent"

    output = pipeline.post_decode(state)

    assert work.waited is True
    assert state.latents is tensor
    assert output.output is tensor

    input_batch = InputBatch.make_batch([state])
    runner = object.__new__(DiffusionModelRunner)
    DiffusionModelRunner._update_states_after(runner, [state], input_batch)
    payload = BatchRunnerOutput.from_list([RunnerOutput(request_id=state.request_id, finished=True, result=output)])

    pack_diffusion_output_shm(payload)
    try:
        assert payload.runner_outputs[0].result.output["__tensor_shm__"] is True
        pickle.dumps(payload)
    finally:
        unpack_diffusion_output_shm(payload)


def test_post_decode_returns_empty_output_on_non_output_pp_rank(monkeypatch) -> None:
    monkeypatch.setattr(wan22_module.current_omni_platform, "is_available", lambda: False)
    pipeline = _pipeline()
    pipeline.vae.decode = lambda *_args, **_kwargs: (None,)
    state = _state(output_type="np")
    state.latents = torch.ones(1, 4, 1, 2, 2)
    state.extra["wan_output_type"] = "np"

    output = pipeline.post_decode(state)

    assert output.output is None
    assert output.media is None
