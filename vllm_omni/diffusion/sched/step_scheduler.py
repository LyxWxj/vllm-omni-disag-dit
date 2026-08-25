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
        self._in_flight: set[str] = set()

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()
        self._in_flight.clear()

    @property
    def num_in_flight_requests(self) -> int:
        """Number of step requests submitted to a tick but not yet completed."""
        return len(self._in_flight)

    def mark_in_flight(self, sched_output: DiffusionSchedulerOutput) -> None:
        """Retain submitted work for capacity without scheduling it twice.

        A future interleaved PP clock can return before each submitted token
        reaches the final stage. Keeping the request in ``_running`` preserves
        its active-slot ownership and compatibility cohort, while this method
        excludes it from the next local tick until ``update_from_output``.
        """
        for request_id in sched_output.scheduled_request_ids:
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            if state.status != DiffusionRequestStatus.RUNNING:
                raise RuntimeError(f"Cannot mark request {request_id!r} in flight from status {state.status.name}.")
            state.status = DiffusionRequestStatus.IN_FLIGHT
            self._in_flight.add(request_id)

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

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        completed_request_ids = output.completed_request_ids
        if not completed_request_ids:
            # An empty pipeline clock is valid while tokens are moving through
            # downstream PP stages. It can still follow an abort that happened
            # after schedule(), so preserve terminal notifications recorded by
            # the scheduler even when this clock has no rank-0 completion.
            return self._finalize_update_from_output(sched_output, {}, {})

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in completed_request_ids:
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
                state.status = DiffusionRequestStatus.RUNNING

        self._in_flight.difference_update(completed_request_ids)
        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        finished_request_ids = super()._finish_requests(statuses, errors)
        self._in_flight.difference_update(finished_request_ids)
        return finished_request_ids

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._request_progress.pop(request_id, None)
        self._in_flight.discard(request_id)

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
