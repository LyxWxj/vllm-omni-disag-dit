# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.models.interface import SupportsStepExecution, supports_step_execution
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
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


def _pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _Transformer()
    pipeline.transformer_2 = None
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline.vae = _VAE()
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline.scheduler = _Scheduler()
    pipeline.boundary_ratio = None
    pipeline.expand_timesteps = True
    pipeline.has_transformer_2 = False
    pipeline.is_dmd = False
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **_kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (
        torch.full((kwargs["num_videos_per_prompt"], kwargs["max_sequence_length"], 8), 2.0),
        None,
    )
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


def test_wan22_declares_step_execution_capability() -> None:
    pipeline = _pipeline()

    assert isinstance(pipeline, SupportsStepExecution)
    assert supports_step_execution(pipeline)


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


@pytest.mark.parametrize("seed", [7, None], ids=["seeded", "unseeded"])
def test_prepare_encode_broadcasts_stage_zero_initial_latents(monkeypatch, seed) -> None:
    _patch_scheduler(monkeypatch)
    source_latents: list[torch.Tensor] = []

    class _Group:
        def __init__(self, is_first_rank: bool) -> None:
            self.is_first_rank = is_first_rank

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            assert src == 0
            if self.is_first_rank:
                source_latents.append(tensor.clone())
                return tensor
            tensor.copy_(source_latents[0])
            return tensor

    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 2)
    first_pipeline = _pipeline()
    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: _Group(True))
    first = _state(request_id="first", seed=seed)
    first_pipeline.prepare_encode(first)

    last_pipeline = _pipeline()
    last_pipeline.prepare_latents = lambda **_kwargs: pytest.fail("non-first rank must not sample initial latents")
    monkeypatch.setattr(wan22_module, "get_pp_group", lambda: _Group(False))
    last = _state(request_id="last", seed=seed)
    last_pipeline.prepare_encode(last)

    torch.testing.assert_close(last.latents, first.latents)
    assert len(source_latents) == 1


def test_prepare_encode_rejects_deferred_modes(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)

    cfg_pipeline = _pipeline()
    cfg_state = _state()
    cfg_state.sampling.guidance_scale = 4.0
    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        cfg_pipeline.prepare_encode(cfg_state)

    image_pipeline = _pipeline()
    image_state = _state()
    image_state.prompt = {"prompt": "animate", "multi_modal_data": {"image": torch.zeros(3, 16, 16)}}
    with pytest.raises(ValueError, match="text-only T2V"):
        image_pipeline.prepare_encode(image_state)

    cascade_pipeline = _pipeline()
    cascade_pipeline.has_transformer_2 = True
    with pytest.raises(ValueError, match="single-transformer"):
        cascade_pipeline.prepare_encode(_state())


def test_denoise_and_scheduler_use_request_local_state_once(monkeypatch) -> None:
    _patch_scheduler(monkeypatch)
    monkeypatch.setattr(wan22_module, "get_pipeline_parallel_world_size", lambda: 1)
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
