# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    SchedulerRequestState,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int


class StepScheduler(BaseScheduler):
    """Scheduler that advances each request by one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}
        self._pipeline_inflight: set[str] = set()
        self._pipeline_decode_pending: set[str] = set()
        self._pipeline_parallel = False
        self._pipeline_batch_size = 1

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()
        self._pipeline_inflight.clear()
        self._pipeline_decode_pending.clear()
        parallel = getattr(self.od_config, "parallel_config", None)
        self._pipeline_parallel = bool(
            getattr(self.od_config, "step_execution", False) and getattr(parallel, "pipeline_parallel_size", 1) > 1
        )
        self._pipeline_batch_size = self.max_num_running_reqs
        if self._pipeline_parallel:
            # Keep three batches per rank resident so stage 0 can stay ahead of
            # solver completion on the last rank.
            self.max_num_running_reqs *= 3 * parallel.pipeline_parallel_size

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._pipeline_parallel:
            return super().schedule()
        ready_ids = [
            request_id
            for request_id in self._running
            if request_id not in self._pipeline_inflight and request_id not in self._pipeline_decode_pending
        ][: self._pipeline_batch_size]
        return self._schedule(ready_ids, batch_limit=self._pipeline_batch_size)

    def mark_pipeline_inflight(self, request_ids: list[str]) -> None:
        self._pipeline_inflight.update(request_ids)

    def pipeline_step_completed(self, request_ids: list[str]) -> None:
        self._pipeline_inflight.difference_update(request_ids)

    @property
    def has_pipeline_inflight(self) -> bool:
        return bool(self._pipeline_inflight)

    @property
    def has_pipeline_ready(self) -> bool:
        return any(
            request_id not in self._pipeline_inflight and request_id not in self._pipeline_decode_pending
            for request_id in self._running
        )

    def pipeline_decode_pending(self) -> list[str]:
        return [request_id for request_id in self._running if request_id in self._pipeline_decode_pending]

    def should_decode_pipeline(self, *, scheduling_exhausted: bool = False) -> bool:
        """Request a decode barrier for a full cohort or an underfilled tail."""
        assert self.od_config is not None
        return bool(self._pipeline_decode_pending) and (
            scheduling_exhausted
            or (
                len(self._pipeline_decode_pending) >= self.od_config.parallel_config.pipeline_parallel_size
                and not self.has_pipeline_ready
            )
        )

    def finish_pipeline_decode(self, output: BaseRunnerOutput) -> set[str]:
        request_ids = self.pipeline_decode_pending()
        if not request_ids:
            return set()
        sched_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(request_ids),
            finished_req_ids=set(),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._pipeline_decode_pending.difference_update(request_ids)
        return self.update_from_output(sched_output, output)

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
        self._request_progress[request_id] = _StepProgress(current_step=current_step, total_steps=total_steps)
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            request_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return request_id

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not super()._can_schedule_waiting(state):
            return False

        if self._pipeline_parallel:
            return True

        is_first_step = self._request_progress[state.request_id].current_step == 0
        for request_id in self._running:
            running = self._request_states[request_id]
            if state.req.allow_mixed_step_phases and running.req.allow_mixed_step_phases:
                continue
            if (self._request_progress[request_id].current_step == 0) != is_first_step:
                return False
        return True

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids and not sched_output.finished_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            self._pipeline_inflight.discard(request_id)
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
            if req_output.finished:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None
            else:
                state.error = None
                if self._pipeline_parallel and progress.current_step >= progress.total_steps:
                    self._pipeline_decode_pending.add(request_id)

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._request_progress.pop(request_id, None)
        self._pipeline_decode_pending.discard(request_id)

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        assert sampling.num_inference_steps is not None
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
