# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Private queued-PP capacity adapters.

The public diffusion configuration keeps the historical
``max_inflight_batches`` name. Queued PP owns its interpretation here so the
field only controls the number of outstanding step Futures; transport slots,
stage memory estimates, and retirement state are not coupled to it.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


def resolve_queued_queue_depth(config: object, vllm_config: object | None = None) -> int:
    """Return the bounded Future queue depth for queued diffusion execution."""
    configured = getattr(config, "max_inflight_batches", None)
    if configured is None:
        configured = getattr(vllm_config, "max_concurrent_batches", None)
    if configured is None:
        configured = getattr(
            getattr(config, "parallel_config", None),
            "pipeline_parallel_size",
            1,
        )
    if type(configured) is not int or configured <= 0:
        raise ValueError(f"max_inflight_batches must be a positive integer, got {configured!r}")
    return configured


@dataclass
class _StepFutureQueue:
    pool: ThreadPoolExecutor
    depth: int
    lock: Lock = field(default_factory=Lock)
    pending: int = 0

    def submit(self, executor: Any, scheduler_output: Any) -> Future:
        with self.lock:
            if self.pending >= self.depth:
                raise RuntimeError(
                    "Queued diffusion step Future queue is full: "
                    f"max_inflight_batches={self.depth}"
                )
            self.pending += 1
        try:
            future = self.pool.submit(executor.execute_step, scheduler_output)
        except BaseException:
            with self.lock:
                self.pending -= 1
            raise

        def release(_: Future) -> None:
            with self.lock:
                self.pending -= 1

        future.add_done_callback(release)
        return future


def submit_step_future(executor: Any, scheduler_output: Any) -> Future:
    """Submit one step to a serialized executor-owned Future queue.

    Worker state is still request-local, but the current executor transports
    one RPC wave at a time. A single queue worker preserves that ordering while
    exposing the same Future boundary used by vLLM's non-blocking executor.
    """
    queue = getattr(executor, "_queued_step_future_queue", None)
    if queue is None:
        lock = getattr(executor, "_queued_step_future_init_lock", None)
        if lock is None:
            lock = Lock()
            setattr(executor, "_queued_step_future_init_lock", lock)
        with lock:
            queue = getattr(executor, "_queued_step_future_queue", None)
            if queue is None:
                config = getattr(executor, "od_config", executor)
                queue = _StepFutureQueue(
                    pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion-step-queue"),
                    depth=resolve_queued_queue_depth(config),
                )
                setattr(executor, "_queued_step_future_queue", queue)
    return queue.submit(executor, scheduler_output)


def execute_model(executor: Any, scheduler_output: Any, *, non_block: bool = False) -> Any:
    """Private vLLM-shaped adapter for queued step execution.

    Keeping this entry in ``queued_pp`` avoids widening the common diffusion
    executor interface while the native Future path is introduced gradually.
    """
    if non_block:
        ensure_open = getattr(executor, "_ensure_open", None)
        if callable(ensure_open):
            ensure_open()
        return submit_step_future(executor, scheduler_output)
    return executor.execute_step(scheduler_output)


def shutdown_step_futures(executor: Any) -> None:
    """Drain and close the private step Future queue during executor shutdown."""
    queue = getattr(executor, "_queued_step_future_queue", None)
    if queue is not None:
        queue.pool.shutdown(wait=True, cancel_futures=True)
        setattr(executor, "_queued_step_future_queue", None)
