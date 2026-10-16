# SPDX-License-Identifier: Apache-2.0
"""Task and lifetime records for request-mode PP microtasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import StepRequestState


@dataclass(frozen=True, slots=True)
class RequestPipelineTask:
    """One request-local denoise step shared by every PP rank."""

    request_id: str
    step_index: int
    task_index: int
    transport_slot: int
    # A tombstone keeps both PP ranks on the same task/control sequence after
    # a shorter request has completed.  It owns no model or P2P work.
    is_tombstone: bool = False


@dataclass(slots=True)
class RequestPipelineBatchState:
    """All state owned by one ``execute_model_batch`` invocation."""

    states: tuple[StepRequestState, ...]
    tasks: tuple[RequestPipelineTask, ...]
    transport: Any
    failed: BaseException | None = None


def build_request_pipeline_tasks(
    request_ids: list[str] | tuple[str, ...],
    num_steps: int | Sequence[int],
    *,
    window: int = 1,
) -> tuple[RequestPipelineTask, ...]:
    """Build a deterministic request-major transpose with tombstones.

    ``transport_slot`` is deliberately derived from the global task index.  It
    gives both PP ranks the same bounded FIFO slot without sending request ids
    through the device transport.  When ``num_steps`` is a sequence, shorter
    requests receive no-op tasks through the longest request's final step so
    both ranks retain the same control/transport sequence.
    """

    if not request_ids:
        return ()
    if type(window) is not int or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_ids must be unique within a request pipeline batch")

    if type(num_steps) is int:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}")
        request_steps = (num_steps,) * len(request_ids)
    else:
        request_steps = tuple(num_steps)
        if len(request_steps) != len(request_ids):
            raise ValueError("num_steps must contain one positive step count per request")
        if any(type(steps) is not int or steps <= 0 for steps in request_steps):
            raise ValueError(f"num_steps must contain positive integers, got {request_steps!r}")

    tasks: list[RequestPipelineTask] = []
    task_index = 0
    for step_index in range(max(request_steps)):
        for request_index, request_id in enumerate(request_ids):
            tasks.append(
                RequestPipelineTask(
                    request_id=request_id,
                    step_index=step_index,
                    task_index=task_index,
                    transport_slot=task_index % window,
                    is_tombstone=step_index >= request_steps[request_index],
                )
            )
            task_index += 1
    return tuple(tasks)
