# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-owner execution loop for one queued diffusion PP Worker."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.worker.pipeline_state import PipelineWorkerUpdate

logger = logging.getLogger(__name__)

_COMMAND_QUEUE_CAPACITY = 64
_INITIAL_PROGRESS_INTERVAL_S = 0.001
# Transport Work completion is polled non-blockingly.  Keep the first poll
# responsive after a command/update, then back off far enough that a long
# device transfer does not turn the StageEngine into a host-side spin loop.
_MAX_PROGRESS_INTERVAL_S = 0.016


@dataclass
class _StageCommand:
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Future[Any]


@dataclass(frozen=True)
class _StageWake:
    pass


class PipelineStageEngine:
    """Serialize Worker RPCs with autonomous local PP and transport progress."""

    def __init__(
        self,
        worker: Any,
        worker_id: int,
        device: Any,
        publish_update: Callable[[PipelineWorkerUpdate], None],
    ) -> None:
        self._worker = worker
        self._worker_id = worker_id
        self._device = device
        self._publish_update = publish_update
        self._commands: queue.Queue[_StageCommand | _StageWake | None] = queue.Queue(maxsize=_COMMAND_QUEUE_CAPACITY)
        self._closed = threading.Event()
        self._state_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._wake_pending = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"DiffusionStageEngine-{worker_id}",
            daemon=True,
        )
        self._thread.start()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        result: Future[Any] = Future()
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("Pipeline StageEngine is closed")
            if self._fatal_error is not None:
                raise RuntimeError("Pipeline StageEngine failed") from self._fatal_error
            try:
                self._commands.put_nowait(_StageCommand(method, args, kwargs, result))
            except queue.Full as exc:
                raise RuntimeError("Pipeline StageEngine command queue is full") from exc
        return result.result()

    def submit(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Queue a command and return once it is owned by the StageEngine."""
        result: Future[Any] = Future()
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("Pipeline StageEngine is closed")
            if self._fatal_error is not None:
                raise RuntimeError("Pipeline StageEngine failed") from self._fatal_error
            try:
                self._commands.put_nowait(_StageCommand(method, args, kwargs, result))
            except queue.Full as exc:
                raise RuntimeError("Pipeline StageEngine command queue is full") from exc

        def report_failure(completed: Future[Any]) -> None:
            try:
                completed.result()
            except BaseException as exc:
                try:
                    self._publish_update(
                        PipelineWorkerUpdate(
                            worker_id=self._worker_id,
                            progress=None,
                            events=(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                except Exception:
                    logger.exception("Failed to publish asynchronous StageEngine command failure")

        result.add_done_callback(report_failure)
        self.notify_progress()

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
        self._commands.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError("Pipeline StageEngine did not stop before Worker shutdown")

    def notify_progress(self) -> None:
        with self._state_lock:
            if self._closed.is_set() or self._fatal_error is not None or self._wake_pending:
                return
            try:
                self._commands.put_nowait(_StageWake())
            except queue.Full as exc:
                raise RuntimeError("Pipeline StageEngine command queue is full while queuing progress") from exc
            self._wake_pending = True

    def _fail_pending_commands(self, error: BaseException) -> None:
        pending_commands: list[_StageCommand] = []
        with self._state_lock:
            self._fatal_error = error
            self._wake_pending = False
            while True:
                try:
                    pending = self._commands.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, _StageCommand):
                    pending_commands.append(pending)

        for command in pending_commands:
            if not command.result.done():
                failure = RuntimeError("Pipeline StageEngine failed")
                failure.__cause__ = error
                command.result.set_exception(failure)

    def _set_device(self) -> None:
        if self._device is None:
            return
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.set_device(self._device)

    def _run(self) -> None:
        try:
            self._set_device()
        except Exception as exc:
            logger.exception("Failed to select the device for pipeline StageEngine %s", self._worker_id)
            fatal_error = RuntimeError("Pipeline StageEngine device initialization failed")
            fatal_error.__cause__ = exc
            self._fail_pending_commands(fatal_error)
            try:
                self._publish_update(
                    PipelineWorkerUpdate(
                        worker_id=self._worker_id,
                        progress=None,
                        events=(),
                        error=f"{type(fatal_error).__name__}: {fatal_error}",
                    )
                )
            except Exception:
                logger.exception("Failed to publish Pipeline StageEngine initialization failure")
            return

        interval = _INITIAL_PROGRESS_INTERVAL_S
        needs_progress = False
        awaiting_rpc_completion = False
        while not self._closed.is_set():
            command: _StageCommand | _StageWake | None = None
            try:
                timeout = interval if needs_progress and not awaiting_rpc_completion else None
                command = self._commands.get(timeout=timeout)
            except queue.Empty:
                pass

            if command is None and self._closed.is_set():
                break
            if isinstance(command, _StageCommand):
                command_value: Any = None
                command_error: BaseException | None = None
                try:
                    command_value = self._worker.execute_method(command.method, *command.args, **command.kwargs)
                except BaseException as exc:
                    command_error = exc
                error = command_error
                if error is None:
                    command.result.set_result(command_value)
                else:
                    command.result.set_exception(error)
                awaiting_rpc_completion = True
                continue
            if isinstance(command, _StageWake):
                with self._state_lock:
                    self._wake_pending = False
                awaiting_rpc_completion = False

            fatal_error: BaseException | None = None
            try:
                update = self._worker.execute_method("pipeline_stage_engine_tick")
                if update is not None:
                    self._publish_update(update)
                needs_progress = bool(self._worker.execute_method("pipeline_stage_engine_needs_progress"))
            except BaseException as exc:
                fatal_error = exc
                logger.exception("Pipeline StageEngine %s failed", self._worker_id)
                self._fail_pending_commands(fatal_error)
                try:
                    self._publish_update(
                        PipelineWorkerUpdate(
                            worker_id=self._worker_id,
                            progress=None,
                            events=(),
                            error=f"{type(fatal_error).__name__}: {fatal_error}",
                        )
                    )
                except Exception:
                    logger.exception("Failed to publish Pipeline StageEngine failure")

            if fatal_error is not None:
                return

            if update is not None or not needs_progress:
                interval = _INITIAL_PROGRESS_INTERVAL_S
            else:
                interval = min(interval * 2, _MAX_PROGRESS_INTERVAL_S)

        while True:
            try:
                pending = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(pending, _StageCommand) and not pending.result.done():
                pending.result.set_exception(RuntimeError("Pipeline StageEngine is closed"))
