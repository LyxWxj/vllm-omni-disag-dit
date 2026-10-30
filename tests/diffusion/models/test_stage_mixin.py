# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import DiffusionStageRole
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.stage_mixin import DiffusionStageMixin
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion]


def _request_batch() -> DiffusionRequestBatch:
    request = SimpleNamespace(
        request_id="request-0",
        is_dummy_run_request_id=lambda request_id: False,
        prompt="prompt",
        sampling_params=SimpleNamespace(num_outputs_per_prompt=1, output_type="latent"),
    )
    return DiffusionRequestBatch(requests=[request])


class _RequestPipeline(DiffusionStageMixin):
    def __init__(self, stage_role: DiffusionStageRole | str):
        self.stage_role = stage_role
        self.calls: list[str] = []

    def encode_batch(self, batch):
        del batch
        self.calls.append("encode")
        return [{"payload": "encode"}]

    def decode_batch(self, batch):
        del batch
        self.calls.append("decode")
        return "decode"

    def diffuse(self, batch):
        del batch
        self.calls.append("diffuse")
        return torch.zeros(1)


@pytest.mark.parametrize(
    ("stage_role", "expected_call", "expected_result"),
    [
        (DiffusionStageRole.ENCODE, ["encode"], "encode"),
        (DiffusionStageRole.DENOISE, ["diffuse"], "latents"),
        (DiffusionStageRole.DENOISE_DECODE, ["diffuse", "decode"], "decode"),
        (DiffusionStageRole.DECODE, ["decode"], "decode"),
        (DiffusionStageRole.FULL, ["diffuse", "decode"], "decode"),
    ],
)
def test_run_stage_dispatches_request_mode_by_role(stage_role, expected_call, expected_result):
    pipeline = _RequestPipeline(stage_role)

    result = pipeline.run_stage(_request_batch())

    assert pipeline.calls == expected_call
    if stage_role == DiffusionStageRole.ENCODE:
        assert result[0].custom_output == {"payload": "encode"}
    elif stage_role == DiffusionStageRole.DENOISE:
        assert result[0].custom_output["latents"].shape == (1,)
    else:
        assert result[0].output == expected_result


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.ENCODE, DiffusionStageRole.DECODE])
def test_request_only_roles_bypass_step_execution(stage_role):
    pipeline = _RequestPipeline(stage_role)
    pipeline.od_config = SimpleNamespace(step_execution=True)

    result = pipeline.run_stage(_request_batch())

    expected_call = ["encode"] if stage_role == DiffusionStageRole.ENCODE else ["decode"]
    assert pipeline.calls == expected_call
    if stage_role == DiffusionStageRole.ENCODE:
        assert result[0].output is None
    else:
        assert result[0].output == "decode"


def test_decode_wrapping_preserves_pipeline_stage_durations():
    pipeline = _RequestPipeline(DiffusionStageRole.DECODE)
    pipeline.stage_durations = {"vae.decode": 1.25}

    result = pipeline.run_stage(_request_batch())

    assert result[0].stage_durations == {"vae.decode": 1.25}


class _StepPipeline(DiffusionStageMixin):
    supports_step_execution = True

    def __init__(self):
        self.calls = 0
        self.seen_states = None

    def prepare_encode(self, state, **kwargs):
        del kwargs
        return state

    def denoise_step(self, batch, *, states=None, **kwargs):
        del batch, kwargs
        self.calls += 1
        self.seen_states = states
        return torch.ones(1)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs

    def post_decode(self, state, **kwargs):
        del state, kwargs
        return None


def test_run_stage_uses_one_denoise_step_when_enabled():
    pipeline = _StepPipeline()
    batch = object.__new__(InputBatch)
    batch.states = ("batch-state",)
    pipeline.od_config = SimpleNamespace(step_execution=True)

    result = pipeline.run_stage(batch)

    assert isinstance(result, torch.Tensor)
    assert pipeline.calls == 1
    assert pipeline.seen_states == batch.states


def test_forward_uses_the_same_role_dispatch():
    pipeline = _RequestPipeline(DiffusionStageRole.DENOISE_DECODE)

    assert pipeline.forward(_request_batch())[0].output == "decode"
    assert pipeline.calls == ["diffuse", "decode"]


@pytest.mark.parametrize("completed", [False, True])
def test_denoise_decode_step_switches_to_decode_after_denoising(completed):
    class DenoiseDecodeStepPipeline(_StepPipeline):
        stage_role = DiffusionStageRole.DENOISE_DECODE

        def decode_batch(self, batch):
            del batch
            self.decode_calls = getattr(self, "decode_calls", 0) + 1
            return ["decoded"]

    state = SimpleNamespace(
        request_id="request-0",
        prompt="prompt",
        sampling=SimpleNamespace(output_type="latent", num_outputs_per_prompt=1),
        latents=torch.zeros(1),
        request_denoise_completed=completed,
    )
    pipeline = DenoiseDecodeStepPipeline()
    pipeline.od_config = SimpleNamespace(step_execution=True)
    batch = object.__new__(InputBatch)
    batch.states = (state,)

    result = pipeline.run_stage(batch)

    if completed:
        assert result[0].output == "decoded"
        assert pipeline.decode_calls == 1
        assert pipeline.calls == 0
    else:
        assert isinstance(result, torch.Tensor)
        assert pipeline.calls == 1


def test_run_stage_rejects_pipeline_without_step_protocol():
    class UnsupportedStepPipeline(DiffusionStageMixin):
        stage_role = DiffusionStageRole.DENOISE
        supports_step_execution = True

    pipeline = UnsupportedStepPipeline()
    batch = object.__new__(InputBatch)
    batch.states = ()

    assert not supports_step_execution(pipeline)
    with pytest.raises(RuntimeError, match="does not support step execution"):
        pipeline.run_stage(batch, step_execution=True)


def test_supports_step_execution_requires_the_runtime_protocol():
    assert not supports_step_execution(SimpleNamespace(supports_step_execution=True))
    assert supports_step_execution(_StepPipeline())


def test_legacy_stage_rejects_explicit_step_execution_for_denoise():
    class LegacyPipeline(_RequestPipeline):
        _uses_stage_primitives = False

    pipeline = LegacyPipeline(DiffusionStageRole.DENOISE)

    with pytest.raises(RuntimeError, match="does not support step execution"):
        pipeline.run_stage(_request_batch(), step_execution=True)
