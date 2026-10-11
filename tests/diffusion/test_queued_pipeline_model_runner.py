# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _QueuedPipeline:
    supports_step_execution = True

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.events = []
        self.forward_context_seen = None
        self.scheduler_forward_context_seen = None

    def prepare_encode(self, state) -> None:
        del state
        self.prepare_calls += 1

    def denoise_step(self, input_batch, **kwargs):
        del input_batch
        self.denoise_calls += 1
        states = kwargs.get("states")
        if states:
            self.events.append(("denoise", states[0].request_id))
        raise AssertionError("queued preparation must not denoise")

    def step_scheduler(self, state, noise_pred) -> None:
        del state, noise_pred

    def post_decode(self, state, **kwargs):
        del kwargs
        self.events.append(("decode", state.request_id))
        from vllm_omni.diffusion.data import DiffusionOutput

        return DiffusionOutput()

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
    return runner


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


def test_execute_prepared_stepwise_uses_request_local_batch_and_advances_state(mocker) -> None:
    runner = _runner()
    state = _state()
    runner.state_cache = {state.request_id: state}
    runner.od_config.streaming_output = False
    runner.od_config.diffusion_kv_mode = None
    runner._sample_peak_memory_mb = mocker.Mock(return_value=0.0)
    update_states_after = mocker.patch.object(runner, "_update_states_after")
    shared_batch = runner.input_batch
    runner.pipeline.denoise_step = mocker.Mock(return_value=torch.ones_like(state.latents))

    def step_scheduler(request_state, noise_pred):
        request_state.latents = request_state.latents + noise_pred
        request_state.step_index += 1

    runner.pipeline.step_scheduler = step_scheduler
    scheduler_output = SimpleNamespace(
        scheduled_request_ids=[state.request_id],
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[state.request_id]),
        finished_req_ids=set(),
    )

    result = runner.execute_prepared_stepwise(scheduler_output)

    output = result.get_request_output(state.request_id)
    assert output is not None and output.finished is False and output.result is None
    assert state.step_index == 1
    torch.testing.assert_close(state.latents, torch.ones_like(state.latents))
    assert runner.input_batch is shared_batch
    assert runner.state_cache[state.request_id] is state
    update_states_after.assert_not_called()
    runner.pipeline.denoise_step.assert_called_once()


def test_execute_prepared_stepwise_sends_one_microbatch_per_pipeline_request(mocker) -> None:
    runner = _runner()
    runner.od_config.parallel_config.pipeline_parallel_size = 2
    states = [_state("req-a"), _state("req-b")]
    for state in states:
        state.timesteps = torch.tensor([2.0])
    runner.state_cache = {state.request_id: state for state in states}
    runner.od_config.streaming_output = False
    runner.od_config.diffusion_kv_mode = None
    runner._sample_peak_memory_mb = mocker.Mock(return_value=0.0)
    runner.pipeline.denoise_step = mocker.Mock(
        side_effect=lambda _batch, states: (
            runner.pipeline.events.append(("denoise", states[0].request_id))
            or torch.ones_like(states[0].latents)
        )
    )

    def step_scheduler(state, noise_pred):
        runner.pipeline.events.append(("scheduler", state.request_id))
        state.latents = state.latents + noise_pred
        state.step_index += 1

    runner.pipeline.step_scheduler = step_scheduler
    scheduler_output = SimpleNamespace(
        scheduled_request_ids=["req-a", "req-b"],
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(request_ids=["req-a", "req-b"]),
        finished_req_ids=set(),
    )

    result = runner.execute_prepared_stepwise(scheduler_output)

    assert runner.pipeline.denoise_step.call_count == 2
    assert all(result.get_request_output(state.request_id) is not None for state in states)
    assert all(state.step_index == 1 for state in states)
    for state in states:
        torch.testing.assert_close(state.latents, torch.ones_like(state.latents))
    assert runner.pipeline.events == [
        ("denoise", "req-a"),
        ("denoise", "req-b"),
        ("scheduler", "req-a"),
        ("scheduler", "req-b"),
    ]


def test_execute_prepared_decode_runs_after_queued_denoise(mocker) -> None:
    runner = _runner()
    state = _state("req-a")
    state.step_index = state.total_steps
    runner.state_cache = {state.request_id: state}
    runner.od_config.diffusion_kv_mode = None
    runner._sample_peak_memory_mb = mocker.Mock(return_value=0.0)

    result = runner.execute_prepared_decode(state.request_id)

    output = result.get_request_output(state.request_id)
    assert output is not None and output.finished and output.result is not None
    assert runner.pipeline.events == [("decode", "req-a")]


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
