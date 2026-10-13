# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.queued_pp.scheduler import QueuedStepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(request_id: str, steps: int = 2) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=f"prompt-{request_id}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=steps),
        request_id=request_id,
    )


def _scheduler(max_num_seqs: int = 2) -> QueuedStepScheduler:
    scheduler = QueuedStepScheduler()
    scheduler.initialize(SimpleNamespace(mode="queued", max_num_seqs=max_num_seqs))
    return scheduler


def test_queued_steps_commit_separately_from_decode_completion() -> None:
    scheduler = _scheduler()
    request_id = scheduler.add_request(_request("req-a"))

    first_step = scheduler.schedule()
    assert first_step.scheduled_request_ids == [request_id]
    assert scheduler.commit_pipeline_step(first_step, {request_id: 1}) == set()
    assert scheduler.get_request_state(request_id).req.sampling_params.step_index == 1

    final_step = scheduler.schedule()
    assert final_step.scheduled_cached_reqs.request_ids == [request_id]
    assert scheduler.commit_pipeline_step(final_step, {request_id: 2}) == {request_id}
    assert scheduler.is_pipeline_finalizing(request_id)

    while_finalizing = scheduler.schedule()
    assert request_id not in while_finalizing.scheduled_request_ids
    assert scheduler.has_requests()

    assert scheduler.complete_pipeline_request(request_id) == {request_id}
    assert scheduler.get_request_state(request_id).status is DiffusionRequestStatus.FINISHED_COMPLETED
    assert scheduler.pending_finished_request_ids() == {request_id}


def test_queued_wave_commit_validates_all_members_before_mutating_state() -> None:
    scheduler = _scheduler()
    first_id = scheduler.add_request(_request("req-a"))
    second_id = scheduler.add_request(_request("req-b"))
    wave = scheduler.schedule()

    with pytest.raises(ValueError, match="exactly cover"):
        scheduler.commit_pipeline_step(wave, {first_id: 1})
    assert scheduler.get_request_state(first_id).req.sampling_params.step_index == 0
    assert scheduler.get_request_state(second_id).req.sampling_params.step_index == 0


def test_queued_admission_deferral_returns_request_to_waiting() -> None:
    scheduler = _scheduler()
    first_id = scheduler.add_request(_request("req-a"))
    second_id = scheduler.add_request(_request("req-b"))
    scheduler.schedule()

    assert scheduler.defer_request(second_id)
    assert scheduler.get_request_state(second_id).status is DiffusionRequestStatus.WAITING
    assert scheduler.schedule().scheduled_cached_reqs.request_ids == [first_id]
