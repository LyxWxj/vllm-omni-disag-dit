# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SchedulerRequestState,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

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
        self._pipeline_finalizing: set[str] = set()

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()
        self._pipeline_finalizing.clear()

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

    def schedule(self) -> DiffusionSchedulerOutput:
        """Exclude decode-owned requests while retaining their running capacity."""
        scheduler_output = super().schedule()
        if self._pipeline_finalizing:
            scheduler_output.scheduled_cached_reqs.request_ids = [
                request_id
                for request_id in scheduler_output.scheduled_cached_reqs.request_ids
                if request_id not in self._pipeline_finalizing
            ]
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
            if req_output.finished:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None
            else:
                state.error = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def commit_pipeline_step(
        self,
        sched_output: DiffusionSchedulerOutput,
        resulting_steps: dict[str, int],
    ) -> set[str]:
        """Commit validated queued denoise progress without completing decode."""
        scheduled_request_ids = tuple(sched_output.scheduled_request_ids)
        if len(scheduled_request_ids) != len(set(scheduled_request_ids)):
            raise ValueError("Queued pipeline step contains duplicate scheduled request IDs.")
        if not scheduled_request_ids or set(resulting_steps) != set(scheduled_request_ids):
            raise ValueError("Queued pipeline step results must exactly cover the scheduled requests.")

        commits: list[tuple[str, SchedulerRequestState, _StepProgress, int]] = []
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
        """Mark a decoded queued request complete and release scheduler capacity."""
        if request_id not in self._pipeline_finalizing:
            raise RuntimeError(f"Queued pipeline request {request_id!r} is not finalizing.")
        finished = self._finish_requests({request_id: DiffusionRequestStatus.FINISHED_COMPLETED})
        if finished != {request_id}:
            raise RuntimeError(f"Queued pipeline request {request_id!r} could not be completed.")
        self._pipeline_finalizing.remove(request_id)
        return finished

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._request_progress.pop(request_id, None)
        self._pipeline_finalizing.discard(request_id)

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
