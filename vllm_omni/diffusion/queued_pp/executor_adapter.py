# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

import zmq
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm
from vllm_omni.diffusion.utils.future_utils import try_set_exception, try_set_result

logger = init_logger(__name__)


def normalize_pipeline_preparation_reports(
    result: Any,
    expected_ranks: frozenset[int],
    expected_request_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise RuntimeError("Workers returned invalid pipeline preparation reports")
    reporting_ranks = [item.get("rank") for item in result]
    if len(reporting_ranks) != len(expected_ranks) or set(reporting_ranks) != expected_ranks:
        raise RuntimeError("pipeline preparation reports do not cover every configured endpoint")
    if any(tuple(item.get("request_ids", ())) != expected_request_ids for item in result):
        raise RuntimeError("pipeline preparation reports disagree on request identity")
    return result


class QueuedExecutorAdapter:
    """Add private queued RPCs around the existing multiprocess executor."""

    def __init__(self, executor: Any):
        if not isinstance(executor, MultiprocDiffusionExecutor):
            raise ValueError("queued pipeline execution currently requires the multiprocess diffusion executor")
        object.__setattr__(self, "_executor", executor)
        object.__setattr__(self, "_queued_rpc_output_ids", {})
        self._start_queued_result_pumps()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_executor" or name.startswith("_queued_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._executor, name, value)

    def _start_queued_result_pumps(self) -> None:
        executor = self._executor
        if executor._pump_running:
            raise RuntimeError("queued executor adapter must own the result queues before starting")
        if not hasattr(executor, "_collective_rpc_lock"):
            executor._collective_rpc_lock = threading.RLock()
        result_mqs = executor._result_mqs or ([executor._result_mq] if executor._result_mq is not None else [])
        if not result_mqs:
            raise RuntimeError("queued executor has no Worker result queues")
        executor._pump_running = True
        executor._pump_stop.clear()
        threads = [
            threading.Thread(
                target=self._result_pump,
                args=(result_mq,),
                daemon=True,
                name=f"QueuedDiffusionResultPump-{rank}",
            )
            for rank, result_mq in enumerate(result_mqs)
        ]
        executor._result_pump_threads = threads
        executor._result_pump_thread = threads[0]
        for thread in threads:
            thread.start()

    def _result_pump(self, result_mq: Any) -> None:
        while not self._pump_stop.is_set():
            try:
                message = result_mq.dequeue(timeout=1.0)
            except (TimeoutError, zmq.error.Again):
                if self._is_failed:
                    break
                continue
            except Exception:
                logger.exception("Queued result pump dequeue failed")
                if self._is_failed:
                    break
                continue

            if not isinstance(message, AsyncDiffusionOutput):
                # Synchronous control RPCs still use the executor's normal
                # result path. Queued non-blocking RPCs arrive as tagged async
                # envelopes and are completed below.
                self._sync_result_buffer.put(message)
                continue

            if message.kind in (AsyncOutputKind.RPC_RESULT, AsyncOutputKind.COMPUTE_DONE):
                if message.kind is AsyncOutputKind.COMPUTE_DONE and message.rpc_id and message.async_output_id:
                    if message.error:
                        with self._futures_lock:
                            future = self._rpc_futures.pop(message.rpc_id, None)
                        if future is not None:
                            try_set_exception(future, RuntimeError(message.error))
                    else:
                        self._queued_rpc_output_ids[message.async_output_id] = message.rpc_id
                    continue
                with self._futures_lock:
                    future = self._rpc_futures.pop(message.rpc_id, None) if message.rpc_id else None
                if future is None or future.done():
                    continue
                if message.error:
                    try_set_exception(future, RuntimeError(message.error))
                elif message.kind is AsyncOutputKind.RPC_RESULT:
                    try:
                        unpack_diffusion_output_shm(message.result)
                    except Exception as exc:
                        try_set_exception(future, exc)
                    else:
                        try_set_result(future, message.result)
                else:
                    try_set_result(future, message)
            elif message.kind is AsyncOutputKind.OUTPUT_READY:
                batch_id = message.async_output_id
                rpc_id = self._queued_rpc_output_ids.pop(batch_id, None) if batch_id else None
                if rpc_id is not None:
                    with self._futures_lock:
                        future = self._rpc_futures.pop(rpc_id, None)
                    if future is not None and not future.done():
                        if message.error:
                            try_set_exception(future, RuntimeError(message.error))
                        else:
                            try:
                                unpack_diffusion_output_shm(message.output)
                            except Exception as exc:
                                try_set_exception(future, exc)
                            else:
                                try_set_result(future, message.output)
                    continue
                with self._futures_lock:
                    per_request_map = self._batch_split_map.pop(batch_id, None) if batch_id else None
                if per_request_map is not None:
                    try:
                        unpack_diffusion_output_shm(message.output)
                    except Exception:
                        logger.exception("SHM unpack failed for queued batch %s", batch_id)
                    self._deliver_batch_split(per_request_map, message.output, message.error)
                elif batch_id:
                    output: Any = None
                    error: Exception | None = RuntimeError(message.error) if message.error else None
                    if error is None:
                        try:
                            unpack_diffusion_output_shm(message.output)
                            output = message.output
                        except Exception as exc:
                            error = exc
                    with self._futures_lock:
                        pending = self._output_futures.pop(batch_id, None)
                        if pending is not None and not pending.done():
                            if error is not None:
                                try_set_exception(pending, error)
                            else:
                                try_set_result(pending, output)
                        else:
                            completed: concurrent.futures.Future = concurrent.futures.Future()
                            if error is not None:
                                completed.set_exception(error)
                            else:
                                completed.set_result(output)
                            self._completed_outputs[batch_id] = completed

    def _collective_rpc_future(
        self,
        method: str,
        *,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None,
        exec_all_ranks: bool,
    ) -> concurrent.futures.Future[Any]:
        if unique_reply_rank is None:
            raise ValueError("queued RPC requires one unique reply rank")
        self._ensure_open()
        with self._collective_rpc_lock:
            rpc_id = self._next_rpc_id()
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            request = {
                "type": "rpc",
                "method": method,
                "args": args,
                "kwargs": kwargs or {},
                "output_rank": unique_reply_rank,
                "exec_all_ranks": exec_all_ranks,
                "collect_rank_status": False,
                "rpc_id": rpc_id,
                "non_block": True,
            }
            with self._futures_lock:
                self._rpc_futures[rpc_id] = future
            try:
                self._broadcast_mq.enqueue(request)
            except BaseException:
                with self._futures_lock:
                    self._rpc_futures.pop(rpc_id, None)
                raise
            return future

    def execute_step(
        self,
        scheduler_output: Any,
        non_block: bool = False,
        output_rank: int = 0,
        reply_rank: int | None = None,
    ) -> Any:
        if not non_block:
            return self._executor.execute_step(scheduler_output)
        return self._collective_rpc_future(
            "execute_prepared_stepwise",
            args=(scheduler_output, output_rank),
            unique_reply_rank=output_rank if reply_rank is None else reply_rank,
            exec_all_ranks=True,
        )

    def prepare_queued_decode(self, request_ids: tuple[str, ...], output_rank: int) -> concurrent.futures.Future[Any]:
        return self._collective_rpc_future(
            "prepare_queued_decode",
            args=(request_ids,),
            unique_reply_rank=output_rank,
            exec_all_ranks=False,
        )

    def execute_decode(
        self,
        request_id: str,
        output_rank: int = 0,
        non_block: bool = False,
    ) -> Any:
        if non_block:
            return self._collective_rpc_future(
                "execute_prepared_decode",
                args=(request_id,),
                unique_reply_rank=output_rank,
                exec_all_ranks=False,
            )
        return self.collective_rpc(
            "execute_prepared_decode",
            args=(request_id,),
            unique_reply_rank=output_rank,
            exec_all_ranks=False,
        )

    def _fail_queued_control(self, operation: str, exc: BaseException) -> None:
        if getattr(self, "_queued_control_failure", None) is None:
            self._queued_control_failure = exc
        failure = self._queued_control_failure
        logger.error("Queued pipeline %s failed after dispatch: %s", operation, failure)
        self._is_failed = True
        try:
            self.shutdown()
        except BaseException:
            logger.exception("Worker-group cleanup failed after queued pipeline failure: %s", failure)
        for callback in self._failure_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("failure_callback raised")

    def _queued_control_rpc(
        self,
        method: str,
        *,
        args: tuple = (),
        exec_all_ranks: bool = False,
    ) -> Any:
        if self._is_failed:
            raise EngineDeadError()
        self._ensure_open()
        try:
            should_execute_all_ranks = exec_all_ranks or method.endswith("_all_ranks")
            return self.collective_rpc(method, args=args, exec_all_ranks=should_execute_all_ranks)
        except BaseException as exc:
            self._fail_queued_control(method, exc)
            raise

    def prepare_pipeline_requests(self, scheduler_output: Any) -> list[dict[str, Any]]:
        expected_request_ids = tuple(request.request_id for request in scheduler_output.scheduled_new_reqs)
        expected_request_ids += tuple(scheduler_output.scheduled_cached_reqs.request_ids)
        if len(expected_request_ids) != 1:
            raise ValueError("Queued preparation requires exactly one new or cached request.")
        try:
            result = self._queued_control_rpc("prepare_pipeline_requests_all_ranks", args=(scheduler_output,))
            return normalize_pipeline_preparation_reports(
                result,
                frozenset(range(self.od_config.num_gpus)),
                expected_request_ids,
            )
        except BaseException as exc:
            if not self._is_failed:
                self._fail_queued_control("pipeline request preparation", exc)
            raise

    def cleanup_finalized_pipeline_request(self, request_id: str) -> Any:
        result = self._queued_control_rpc("cleanup_finalized_pipeline_request_all_ranks", args=(request_id,))
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result

    def collective_rpc(self, method: str, **kwargs: Any) -> Any:
        with self._collective_rpc_lock:
            return self._executor.collective_rpc(method, **kwargs)
