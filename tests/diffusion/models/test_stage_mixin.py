# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import DiffusionStageOperator, DiffusionStageRole
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.stage_mixin import DiffusionStageMixin
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("stage_role", "expected_operators"),
    [
        (
            DiffusionStageRole.FULL,
            {
                DiffusionStageOperator.PREPARE_ENCODE,
                DiffusionStageOperator.DENOISE_STEP,
                DiffusionStageOperator.STEP_SCHEDULER,
                DiffusionStageOperator.POST_DECODE,
            },
        ),
        (
            DiffusionStageRole.DENOISE_DECODE,
            {
                DiffusionStageOperator.PREPARE_ENCODE,
                DiffusionStageOperator.DENOISE_STEP,
                DiffusionStageOperator.STEP_SCHEDULER,
                DiffusionStageOperator.POST_DECODE,
            },
        ),
        (
            DiffusionStageRole.DENOISE,
            {
                DiffusionStageOperator.PREPARE_ENCODE,
                DiffusionStageOperator.DENOISE_STEP,
                DiffusionStageOperator.STEP_SCHEDULER,
                DiffusionStageOperator.STEP_OUTPUT,
            },
        ),
        (DiffusionStageRole.ENCODE, {DiffusionStageOperator.ENCODE}),
        (DiffusionStageRole.DECODE, {DiffusionStageOperator.DECODE}),
    ],
)
def test_required_operators_are_derived_from_role_stages(stage_role, expected_operators):
    assert stage_role.required_operators(step_execution=True) == expected_operators


@pytest.mark.parametrize(
    "stage_role",
    list(DiffusionStageRole),
)
def test_request_mode_requires_only_the_default_operator(stage_role):
    assert stage_role.required_operators(step_execution=False) == {stage_role.default_request_operator}


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
        self.initialize_stage_execution(step_execution=False)

    def stage_forward(self, batch):
        del batch
        self.calls.append("forward")
        return [DiffusionOutput(output="forward")]

    def stage_encode(self, batch):
        del batch
        self.calls.append("encode")
        return [DiffusionOutput(custom_output={"payload": "encode"})]

    def stage_diffuse(self, batch):
        del batch
        self.calls.append("diffuse")
        return [DiffusionOutput(output="decode", custom_output={"latents": torch.zeros(1)})]

    def stage_decode(self, batch):
        del batch
        self.calls.append("decode")
        return [DiffusionOutput(output="decode", stage_durations=getattr(self, "stage_durations", {}))]

    def forward(self, batch):
        return self.run_stage(batch)


@pytest.mark.parametrize(
    ("stage_role", "expected_call", "expected_result"),
    [
        (DiffusionStageRole.ENCODE, ["encode"], "encode"),
        (DiffusionStageRole.DENOISE, ["diffuse"], "latents"),
        (DiffusionStageRole.DENOISE_DECODE, ["diffuse"], "decode"),
        (DiffusionStageRole.DECODE, ["decode"], "decode"),
        (DiffusionStageRole.FULL, ["forward"], "forward"),
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
    stage_role = DiffusionStageRole.DENOISE
    supports_step_execution = True

    def __init__(self):
        self.calls = 0
        self.seen_states = None
        self.initialize_stage_execution(step_execution=True)

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

    def step_output(self, state, **kwargs):
        del state, kwargs
        return None

    def post_decode(self, state, **kwargs):
        del state, kwargs
        return None


def test_run_stage_uses_one_denoise_step_when_enabled():
    pipeline = _StepPipeline()
    batch = object.__new__(InputBatch)
    batch.states = ("batch-state",)

    result = pipeline.run_stage(
        batch,
        operator=DiffusionStageOperator.DENOISE_STEP,
        states=batch.states,
    )

    assert isinstance(result, torch.Tensor)
    assert pipeline.calls == 1
    assert pipeline.seen_states == batch.states


def test_forward_uses_the_same_role_dispatch():
    pipeline = _RequestPipeline(DiffusionStageRole.DENOISE_DECODE)

    assert pipeline.forward(_request_batch())[0].output == "decode"
    assert pipeline.calls == ["diffuse"]


@pytest.mark.parametrize("completed", [False, True])
def test_denoise_decode_step_switches_to_decode_after_denoising(completed):
    class DenoiseDecodeStepPipeline(_StepPipeline):
        stage_role = DiffusionStageRole.DENOISE_DECODE

        def post_decode(self, state, **kwargs):
            del state, kwargs
            self.decode_calls = getattr(self, "decode_calls", 0) + 1
            return DiffusionOutput(output="decoded")

    state = SimpleNamespace(
        request_id="request-0",
        prompt="prompt",
        sampling=SimpleNamespace(output_type="latent", num_outputs_per_prompt=1),
        latents=torch.zeros(1),
        request_denoise_completed=completed,
    )
    pipeline = DenoiseDecodeStepPipeline()
    batch = object.__new__(InputBatch)
    batch.states = (state,)

    operator = DiffusionStageOperator.POST_DECODE if completed else DiffusionStageOperator.DENOISE_STEP
    result = pipeline.run_stage(batch, operator=operator, states=batch.states)

    if completed:
        assert result.output == "decoded"
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

    assert not supports_step_execution(pipeline)
    with pytest.raises(TypeError, match="required step-execution primitives"):
        pipeline.initialize_stage_execution(step_execution=True)


def test_supports_step_execution_requires_the_runtime_protocol():
    assert not supports_step_execution(SimpleNamespace(supports_step_execution=True))
    assert supports_step_execution(_StepPipeline())


def test_legacy_stage_rejects_explicit_step_execution_for_denoise():
    class LegacyPipeline(_RequestPipeline):
        _uses_stage_primitives = False

    pipeline = LegacyPipeline(DiffusionStageRole.DENOISE)

    with pytest.raises(TypeError, match="required step-execution primitives"):
        pipeline.initialize_stage_execution(step_execution=True)
