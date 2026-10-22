# SPDX-License-Identifier: Apache-2.0
"""Bounded transport context for request-mode PP microtasks."""

from __future__ import annotations

from collections import deque
from typing import Any

from vllm_omni.diffusion.request_pipeline.task import RequestPipelineTask
from vllm_omni.diffusion.request_pipeline.trace import trace_request_pipeline_event


class RequestPipelineTransport:
    """Own PP work lifetime for one closed request batch.

    The existing :class:`PipelineParallelMixin` performs the actual tensor
    sends and receives.  This object supplies the request-mode lifetime and
    bounded window around those calls, so the ModelRunner does not need to
    know about ``torch.distributed.Work`` details.
    """

    def __init__(self, pipeline: Any, *, window: int = 1):
        if type(window) is not int or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.pipeline = pipeline
        self.window = window
        self._active: deque[tuple[RequestPipelineTask, list[Any]]] = deque()
        self._pending_latents: list[Any] = []
        self._closed = False

    @property
    def active_tasks(self) -> tuple[RequestPipelineTask, ...]:
        return tuple(task for task, _ in self._active)

    def __enter__(self) -> RequestPipelineTransport:
        if getattr(self.pipeline, "_request_pipeline_transport", None) is not None:
            raise RuntimeError("A request pipeline transport is already active on this pipeline")
        self.pipeline._request_pipeline_transport = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.drain()
        except BaseException:
            if exc_type is None:
                raise
        finally:
            if getattr(self.pipeline, "_request_pipeline_transport", None) is self:
                self.pipeline._request_pipeline_transport = None

    def begin_task(self, task: RequestPipelineTask) -> None:
        if self._closed:
            raise RuntimeError("Request pipeline transport is closed")
        if not 0 <= task.transport_slot < self.window:
            raise ValueError(f"task transport slot {task.transport_slot} is outside window {self.window}")
        if self._active and task.task_index <= self._active[-1][0].task_index:
            raise ValueError("request pipeline tasks must be admitted in task-index order")
        self.wait_oldest_slot_if_full()
        self._active.append((task, []))
        trace_request_pipeline_event(
            "request_pipeline_task",
            phase="begin",
            request_id=task.request_id,
            step_index=task.step_index,
            task_index=task.task_index,
            transport_slot=task.transport_slot,
            is_tombstone=task.is_tombstone,
            window=self.window,
        )

    def register_send_work(self, work: list[Any]) -> None:
        if not self._active:
            raise RuntimeError("request pipeline send work has no active task")
        self._active[-1][1].extend(work)

    def register_feedback(self, latents: Any) -> None:
        self._pending_latents.append(latents)

    def consume_feedback(self, latents: Any) -> Any:
        """Resolve and release one feedback receive at its next task use."""
        resolve = getattr(latents, "resolve", None)
        value = resolve() if callable(resolve) else latents
        for index, pending in enumerate(self._pending_latents):
            if pending is latents:
                del self._pending_latents[index]
                break
        return value

    def wait_oldest_slot_if_full(self) -> None:
        if len(self._active) < self.window:
            return
        _, work = self._active[0]
        if work:
            for handle in work:
                handle.wait()
        else:
            self.pipeline._sync_pp_send()
        self._active.popleft()

    def drain(self) -> None:
        if self._closed:
            return
        for _, work in self._active:
            for handle in work:
                handle.wait()
        self._active.clear()
        self.pipeline._sync_pp_send()
        for latent in self._pending_latents:
            resolve = getattr(latent, "resolve", None)
            if callable(resolve):
                resolve()
        self._pending_latents.clear()
        self._closed = True
