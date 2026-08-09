# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    KVPrefetchJob,
    NewRequestData,
)
from vllm_omni.diffusion.sched.step_cost import (
    RequestStepCost,
    StepCostCandidate,
    order_step_cost_candidates,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int
    cost: RequestStepCost


class StepScheduler(BaseScheduler):
    """Scheduler that advances each request by one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}
        self._request_age_ticks: dict[str, int] = {}
        self._serial_cache_execution = False
        self._aging_credit_ms_per_tick = 1.0

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()
        self._request_age_ticks.clear()
        cache_backend = str(getattr(self.od_config, "cache_backend", "none") or "none").lower()
        self._serial_cache_execution = cache_backend != "none"
        self._aging_credit_ms_per_tick = float(getattr(self.od_config, "step_schedule_aging_credit_ms_per_tick", 1.0))

    def add_request(self, request: OmniDiffusionRequest) -> str:
        request_id = request.request_id
        total_steps = self._get_total_steps(request)
        if total_steps <= 0:
            raise ValueError(f"Diffusion request {request_id} must have positive total_steps, got {total_steps}")

        current_step = request.sampling_params.step_index or 0
        if current_step < 0 or current_step >= total_steps:
            raise ValueError(
                f"Diffusion request {request_id} has invalid initial step_index {current_step} "
                f"for total_steps={total_steps}"
            )

        request.sampling_params.step_index = current_step
        request_id = self._add_request_with_request_id(request_id, request)
        self._request_progress[request_id] = _StepProgress(
            current_step=current_step,
            total_steps=total_steps,
            cost=RequestStepCost(),
        )
        self._request_age_ticks[request_id] = 0
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            request_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return request_id

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._serial_cache_execution:
            return super().schedule()
        return self._schedule_serial_cache_request()

    def _schedule_serial_cache_request(self) -> DiffusionSchedulerOutput:
        while self._waiting and self._waiting[0] not in self._request_states:
            self._waiting.popleft()

        waiting_candidate: str | None = None
        if self._waiting and len(self._running) < self.max_num_running_reqs:
            request_id = self._waiting[0]
            state = self._request_states[request_id]
            if self._can_schedule_waiting(state):
                waiting_candidate = request_id

        candidate_ids = [
            *(request_id for request_id in (waiting_candidate,) if request_id is not None),
            *(
                request_id
                for request_id in self._running
                if (state := self._request_states.get(request_id)) is not None and not state.is_finished()
            ),
        ]
        selected_request_id: str | None = None
        if candidate_ids:
            selected_request_id = order_step_cost_candidates(
                [
                    StepCostCandidate(
                        request_id=request_id,
                        cost=self._request_progress[request_id].cost,
                        age_ticks=self._request_age_ticks.get(request_id, 0),
                    )
                    for request_id in candidate_ids
                ],
                aging_credit_ms_per_tick=self._aging_credit_ms_per_tick,
            )[0]

        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_request_ids: list[str] = []
        if selected_request_id is not None:
            state = self._request_states[selected_request_id]
            if selected_request_id == waiting_candidate:
                self._waiting.popleft()
                was_new_request = state.status == DiffusionRequestStatus.WAITING
                if not self._running:
                    self._running_sampling_params_key = state.sampling_params_key
                state.status = DiffusionRequestStatus.RUNNING
                self._running.append(selected_request_id)
                if was_new_request:
                    scheduled_new_reqs.append(NewRequestData.from_state(state))
                else:
                    scheduled_cached_request_ids.append(selected_request_id)
            else:
                scheduled_cached_request_ids.append(selected_request_id)

        for request_id in (*self._running, *self._waiting):
            if request_id == selected_request_id:
                self._request_age_ticks[request_id] = 0
            else:
                self._request_age_ticks[request_id] = self._request_age_ticks.get(request_id, 0) + 1

        kv_prefetch_job: KVPrefetchJob | None = None
        if self._prefetch_enabled and self._waiting:
            next_state = self._request_states.get(self._waiting[0])
            if next_state is not None and not next_state.is_finished():
                sender_info = getattr(next_state.req, "kv_sender_info", None)
                if sender_info:
                    kv_prefetch_job = {
                        "request_id": next_state.request_id,
                        "kv_sender_info": sender_info,
                    }

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(request_ids=scheduled_cached_request_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            kv_prefetch_job=kv_prefetch_job,
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            progress = self._request_progress.get(request_id)
            if state is None or progress is None or state.is_finished():
                continue
            req_output = output.get_request_output(request_id)
            if req_output is None:
                logger.warning(
                    "No RunnerOutput for request %s, treating as error",
                    request_id,
                )
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = "No output for request"
                continue

            req_result = req_output.result
            if req_result is not None and req_result.aborted:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ABORTED
                terminal_errors[request_id] = None
                continue
            output_error = req_result.error if req_result is not None else None
            if output_error is not None:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = output_error
                continue

            if req_output.step_index is None:
                logger.warning(
                    "Received RunnerOutput with no step_index for request %s, treating as error",
                    request_id,
                )
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = "Missing step_index in RunnerOutput"
                continue

            # We assume that the decoding stage is executed immediately after the denoising stage completes.
            progress.current_step = req_output.step_index
            state.req.sampling_params.step_index = req_output.step_index
            if req_output.step_cost_observation is not None:
                progress.cost.observe(req_output.step_cost_observation)
            if req_output.finished:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None
            else:
                state.error = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._request_progress.pop(request_id, None)
        self._request_age_ticks.pop(request_id, None)

    def get_step_cost(self, request_id: str) -> RequestStepCost | None:
        progress = self._request_progress.get(request_id)
        return progress.cost if progress is not None else None

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        return int(sampling.num_inference_steps)

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)
