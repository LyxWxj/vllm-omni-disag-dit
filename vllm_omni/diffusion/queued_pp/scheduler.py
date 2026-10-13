# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus, DiffusionSchedulerOutput, SchedulerRequestState
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler


class QueuedStepScheduler(StepScheduler):
    """Step scheduler that separates denoise commit from final decode retirement."""

    def __init__(self) -> None:
        super().__init__()
        self._pipeline_finalizing: set[str] = set()
        self.grouped_queued_decode = False

    def _reset_scheduler_state(self) -> None:
        super()._reset_scheduler_state()
        self._pipeline_finalizing.clear()

    def _can_schedule_waiting(self, _state: SchedulerRequestState) -> bool:
        return True

    def schedule(self) -> DiffusionSchedulerOutput:
        original_capacity = self.max_num_running_reqs
        if not self.grouped_queued_decode:
            self.max_num_running_reqs += len(self._pipeline_finalizing)
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = original_capacity
        if self._pipeline_finalizing:
            scheduler_output.scheduled_cached_reqs.request_ids = [
                request_id
                for request_id in scheduler_output.scheduled_cached_reqs.request_ids
                if request_id not in self._pipeline_finalizing
            ]
        return scheduler_output

    def has_queued_admission_candidate(
        self,
        owned_request_ids: set[str],
        *,
        admission_capacity_available: bool,
    ) -> bool:
        if self._finished_req_ids:
            return True
        if any(
            request_id not in owned_request_ids and request_id not in self._pipeline_finalizing
            for request_id in self._running
        ):
            return True
        active_running = len(self._running)
        if not self.grouped_queued_decode:
            active_running -= len(self._pipeline_finalizing)
        return admission_capacity_available and bool(self._waiting) and active_running < self.max_num_running_reqs

    def has_queued_waiting_request(self) -> bool:
        return bool(self._waiting)

    def has_queued_denoise_request(self) -> bool:
        return bool(self._waiting) or any(request_id not in self._pipeline_finalizing for request_id in self._running)

    def commit_pipeline_step(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        resulting_steps: dict[str, int],
    ) -> set[str]:
        scheduled_request_ids = tuple(scheduler_output.scheduled_request_ids)
        if len(scheduled_request_ids) != len(set(scheduled_request_ids)):
            raise ValueError("Queued pipeline step contains duplicate scheduled request IDs.")
        if not scheduled_request_ids or set(resulting_steps) != set(scheduled_request_ids):
            raise ValueError("Queued pipeline step results must exactly cover the scheduled requests.")

        commits: list[tuple[str, SchedulerRequestState, Any, int]] = []
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            progress = self._request_progress.get(request_id)
            if state is None or progress is None:
                raise KeyError(f"Queued pipeline request {request_id!r} is not owned by this Scheduler.")
            if state.status is not DiffusionRequestStatus.RUNNING:
                raise RuntimeError(f"Queued pipeline request {request_id!r} is not running.")
            if request_id in self._pipeline_finalizing:
                raise RuntimeError(f"Queued pipeline request {request_id!r} is already finalizing.")
            resulting_step = resulting_steps[request_id]
            if type(resulting_step) is not int or resulting_step != progress.current_step + 1:
                raise ValueError(
                    f"Queued pipeline request {request_id!r} expected step {progress.current_step + 1}, "
                    f"got {resulting_step!r}."
                )
            if resulting_step > progress.total_steps:
                raise ValueError(f"Queued pipeline request {request_id!r} advanced past its denoise schedule.")
            commits.append((request_id, state, progress, resulting_step))

        finalizing: set[str] = set()
        for request_id, state, progress, resulting_step in commits:
            progress.current_step = resulting_step
            state.req.sampling_params.step_index = resulting_step
            state.error = None
            if resulting_step == progress.total_steps:
                self._pipeline_finalizing.add(request_id)
                finalizing.add(request_id)
        return finalizing

    def is_pipeline_finalizing(self, request_id: str) -> bool:
        return request_id in self._pipeline_finalizing

    def complete_pipeline_request(self, request_id: str) -> set[str]:
        if request_id not in self._pipeline_finalizing:
            raise RuntimeError(f"Queued pipeline request {request_id!r} is not finalizing.")
        finished = self._finish_requests({request_id: DiffusionRequestStatus.FINISHED_COMPLETED})
        if finished != {request_id}:
            raise RuntimeError(f"Queued pipeline request {request_id!r} could not be completed.")
        self._pipeline_finalizing.remove(request_id)
        return finished

    def defer_request(self, request_id: str) -> bool:
        state = self._request_states.get(request_id)
        if state is None or request_id not in self._running or state.status is not DiffusionRequestStatus.RUNNING:
            return False
        self._running.remove(request_id)
        if not self._running:
            self._running_sampling_params_key = None
        self._waiting.appendleft(request_id)
        state.status = DiffusionRequestStatus.WAITING
        return True

    def _pop_extra_request_state(self, request_id: str) -> None:
        super()._pop_extra_request_state(request_id)
        self._pipeline_finalizing.discard(request_id)
