# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.queued_pp.runtime import (
    _QueuedAdmissionDeferredError,
    _QueuedDecodeRound,
    _QueuedPipelineBatch,
    _QueuedPipelineBatchPhase,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, BatchRunnerOutput, RunnerOutput

logger = init_logger(__name__)


def _max_num_seqs(config: Any) -> int:
    try:
        return max(1, int(getattr(config, "max_num_seqs", 1)))
    except (TypeError, ValueError):
        return 1


class QueuedPipelineEngineQueue:
    """Own queued request-step Futures and Decode rounds for one Engine."""

    def __init__(self, engine: Any, executor_adapter: Any):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_executor_adapter", executor_adapter)
        self._queued_pipeline_batches: dict[str, _QueuedPipelineBatch] = {}
        self._queued_pipeline_epoch = 0
        self._queued_decode_round: _QueuedDecodeRound | None = None
        self._queued_round = 0
        self.scheduler.grouped_queued_decode = self._uses_grouped_queued_decode()

    @property
    def executor(self) -> Any:
        return self._executor_adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_engine", "_executor_adapter"} or name.startswith("_queued_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._engine, name, value)

    def has_work(self) -> bool:
        return self._has_queued_pipeline_work()

    def can_advance_without_schedule(self) -> bool:
        return self._queued_pipeline_can_advance_without_schedule()

    def should_wait_for_update(self) -> bool:
        return self._should_wait_for_queued_pipeline_update()

    def abort_requests(self, request_ids: str | list[str]) -> None:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        request_ids = list(dict.fromkeys(request_ids))
        queued_request_ids = set(request_ids)
        queued_batches = self._queued_pipeline_batches
        owned_queued_request_ids = {batch.request_id for batch in queued_batches.values()}
        for batch in list(queued_batches.values()):
            if batch.request_id not in queued_request_ids:
                continue
            batch.abort_requested = True
            try:
                if not batch.cancelled:
                    self._cancel_queued_pipeline_batch(batch)
                self._retire_queued_pipeline_batch(batch)
                if batch.batch_id in self._queued_pipeline_batches:
                    continue
                self._emit_finished_outputs({batch.request_id}, None)
            except Exception:
                logger.error("Queued abort cleanup is pending; retaining ownership", exc_info=True)

        remaining_request_ids = [request_id for request_id in request_ids if request_id not in owned_queued_request_ids]
        self._remove_diffusion_kv_requests(remaining_request_ids)
        for request_id in remaining_request_ids:
            if self.scheduler.get_request_state(request_id) is not None:
                self.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    def step(self, scheduler_output: Any | None) -> None:
        """Prepare, submit, and advance queued work for one Engine scheduler round."""
        self._queued_round += 1
        if scheduler_output is not None and (self._queued_round == 1 or self._queued_round % 100 == 0):
            logger.info(
                "Queued pipeline scheduler round=%d step_id=%s new=%s cached=%s waiting=%s running=%s retained=%s",
                self._queued_round,
                scheduler_output.step_id,
                [request.request_id for request in scheduler_output.scheduled_new_reqs],
                list(scheduler_output.scheduled_cached_reqs.request_ids),
                scheduler_output.num_waiting_reqs,
                scheduler_output.num_running_reqs,
                sorted(self._queued_pipeline_batches),
            )
        if scheduler_output is not None and scheduler_output.is_empty:
            finished_ids = {
                request_id
                for request_id in scheduler_output.finished_req_ids
                if (state := self.scheduler.get_request_state(request_id)) is not None and state.is_finished()
            }
            self._emit_finished_outputs(finished_ids, None)
            scheduler_output = None
        handled_request_ids: set[str] = set()
        task_outputs = self._split_queued_scheduler_output(scheduler_output) if scheduler_output is not None else []
        admitted_outputs: list[Any] = []
        for output_index, task_output in enumerate(task_outputs):
            try:
                self._run_queued_pipeline_iteration(task_output, submit_only=True)
                admitted_outputs.append(task_output)
                handled_request_ids.update(task_output.scheduled_request_ids)
            except _QueuedAdmissionDeferredError:
                self._defer_queued_admission_tail(task_outputs[output_index:])
                break
            except Exception as exc:
                handled_request_ids.update(task_output.scheduled_request_ids)
                logger.error(
                    "Queued execution failed for diffusion requests %s",
                    task_output.scheduled_request_ids,
                    exc_info=True,
                )
                if not any(
                    (batch.request_id,) == tuple(task_output.scheduled_request_ids)
                    for batch in self._queued_pipeline_batches.values()
                ):
                    self._reject_queued_admission(task_output, exc)
                    if getattr(self.executor, "_is_failed", False) is True:
                        break
                else:
                    self._handle_queued_iteration_failure(task_output, exc)
        self._authorize_waiting_queued_batches()
        for task_output in admitted_outputs:
            batch = self._queued_pipeline_batch_for_scheduler_output(task_output)
            batch_id = None if batch is None else batch.batch_id
            try:
                self._run_queued_pipeline_iteration(task_output)
            except Exception as exc:
                logger.error(
                    "Queued execution failed for diffusion requests %s",
                    task_output.scheduled_request_ids,
                    exc_info=True,
                )
                if batch_id is None:
                    self._reject_queued_admission(task_output, exc)
                else:
                    self._handle_queued_iteration_failure(task_output, exc)
        self._advance_unhandled_queued_batches(handled_request_ids)
        self._advance_queued_decode_round()
        if self._should_wait_for_queued_pipeline_update():
            with self._cv:
                if (
                    not self.stop_event.is_set()
                    and self._rpc_queue.empty()
                    and self.abort_queue.empty()
                    and self._should_wait_for_queued_pipeline_update()
                ):
                    self._cv.wait()

    def _notify_queued_pipeline_update(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def _run_queued_pipeline_iteration(
        self,
        scheduler_output: Any,
        *,
        submit_only: bool = False,
    ) -> None:
        """Prepare a request, submit its non-blocking PP step, and commit on completion."""
        existing = self._queued_pipeline_batch_for_scheduler_output(scheduler_output)
        if existing is None:
            self._submit_queued_pipeline_batch(scheduler_output, authorize=not submit_only)
            batch = self._queued_pipeline_batch_for_scheduler_output(scheduler_output)
            if batch is None:
                raise RuntimeError("Queued pipeline submission did not retain its batch ownership.")
        else:
            batch = existing
        if not submit_only:
            self._authorize_waiting_queued_batches()
        if submit_only:
            return
        if batch.abort_requested:
            if not batch.cancelled:
                self._cancel_queued_pipeline_batch(batch)
            self._retire_queued_pipeline_batch(batch)
            if batch.batch_id in self._queued_pipeline_batches:
                return
            self._emit_finished_outputs({batch.request_id}, None)
            return
        if batch.failure is not None:
            if not batch.cancelled:
                self._cancel_queued_pipeline_batch(batch)
            self._retire_queued_pipeline_batch(batch)
            if batch.batch_id in self._queued_pipeline_batches:
                return
            self._finish_failed_queued_batch(batch)
            return
        if tuple(scheduler_output.scheduled_request_ids) != (batch.request_id,):
            raise RuntimeError("queued pipeline scheduler selected a request while another batch is retained")

        output = self._advance_queued_pipeline_batch(batch)
        if output is not None:
            self._emit_finished_outputs({batch.request_id}, output)

    def _queued_pipeline_batch_for_scheduler_output(self, scheduler_output: Any) -> _QueuedPipelineBatch | None:
        """Find the retained batch for one request-step descriptor."""
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        if len(request_ids) != 1:
            raise ValueError("Queued pipeline batch lookup requires exactly one scheduled request.")
        descriptor_matches = [
            batch
            for batch in self._queued_pipeline_batches.values()
            if batch.scheduler_output is scheduler_output and batch.request_id == request_ids[0]
        ]
        if len(descriptor_matches) > 1:
            raise RuntimeError(f"Queued pipeline descriptor {request_ids!r} has duplicate batch ownership.")
        if descriptor_matches:
            return descriptor_matches[0]
        request_state = self.scheduler.get_request_state(request_ids[0])
        if request_state is None:
            raise RuntimeError(f"Queued request {request_ids[0]!r} has no Scheduler state.")
        request_step = request_state.req.sampling_params.step_index
        if request_step is None:
            request_step = 0
        matches = [
            batch
            for batch in self._queued_pipeline_batches.values()
            if batch.request_id == request_ids[0] and batch.step_index == request_step
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"Queued pipeline has duplicate ownership for request-step {request_ids!r}, step={request_step}."
            )
        return matches[0] if matches else None

    def _finish_failed_queued_batch(self, batch: _QueuedPipelineBatch) -> None:
        """Complete a failed request only after its queued ownership is retired."""
        failure = batch.failure
        if failure is None or batch.batch_id in self._queued_pipeline_batches:
            raise RuntimeError("failed queued batch is not ready for scheduler completion")
        request_id = batch.request_id
        if self.scheduler.get_request_state(request_id) is not None:
            self.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ERROR)
        runner_output = BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    request_id=request_id,
                    step_index=None,
                    finished=True,
                    result=DiffusionOutput.from_exception(failure),
                )
            ]
        )
        self._emit_finished_outputs({request_id}, runner_output)

    def _handle_queued_iteration_failure(self, scheduler_output: Any, failure: BaseException) -> None:
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        batch = self._queued_pipeline_batch_for_scheduler_output(scheduler_output)
        if batch is None:
            raise RuntimeError(f"Queued failure has no owned batch for requests {request_ids!r}") from failure
        if batch.abort_requested:
            logger.error("Queued abort cleanup is pending; retaining ownership", exc_info=True)
            return
        if batch.failure is None:
            batch.failure = failure
        try:
            if not batch.cancelled:
                self._cancel_queued_pipeline_batch(batch)
            self._retire_queued_pipeline_batch(batch)
            if batch.batch_id in self._queued_pipeline_batches:
                return
            self._finish_failed_queued_batch(batch)
        except Exception:
            logger.error("Queued batch cleanup is pending; retaining ownership", exc_info=True)

    def _reject_queued_admission(self, scheduler_output: Any, failure: BaseException) -> None:
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        self.scheduler.finish_requests(list(request_ids), DiffusionRequestStatus.FINISHED_ERROR)
        runner_output = BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    request_id=request_id,
                    step_index=None,
                    finished=True,
                    result=DiffusionOutput.from_exception(failure),
                )
                for request_id in request_ids
            ]
        )
        self._emit_finished_outputs(set(request_ids), runner_output)

    def _defer_queued_admission(self, scheduler_output: Any) -> None:
        """Restore an unadmitted descriptor without changing its new/cached identity."""
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        if len(request_ids) != 1:
            raise ValueError("Queued admission deferral requires exactly one request.")
        request_id = request_ids[0]
        new_request_ids = {request.request_id for request in scheduler_output.scheduled_new_reqs}
        cached_request_ids = set(scheduler_output.scheduled_cached_reqs.request_ids)
        if (request_id in new_request_ids) == (request_id in cached_request_ids):
            raise RuntimeError("Queued admission descriptor has ambiguous new/cached request identity.")
        if request_id in new_request_ids:
            if not self.scheduler.defer_request(request_id):
                raise RuntimeError(f"Could not return unprepared request {request_id!r} to the waiting queue.")
            return
        if any(request_id == batch.request_id for batch in self._queued_pipeline_batches.values()):
            # A cached request that already owns a queued batch must remain RUNNING
            # while its retained transport work is advanced below.
            return
        if not self.scheduler.preempt_request(request_id):
            raise RuntimeError(f"Could not preempt cached request {request_id!r} for queued admission.")

    def _defer_queued_admission_tail(self, scheduler_outputs: list[Any]) -> None:
        """Return an unprocessed scheduler suffix in its original FIFO order."""
        for scheduler_output in reversed(scheduler_outputs):
            self._defer_queued_admission(scheduler_output)

    def _advance_unhandled_queued_batches(
        self,
        handled_request_ids: set[str],
    ) -> None:
        """Progress retained tasks that were not represented in this scheduler cycle."""
        for batch in list(self._queued_pipeline_batches.values()):
            if batch.request_id in handled_request_ids:
                continue
            try:
                if batch.abort_requested:
                    if not batch.cancelled:
                        self._cancel_queued_pipeline_batch(batch)
                    self._retire_queued_pipeline_batch(batch)
                    if batch.batch_id in self._queued_pipeline_batches:
                        continue
                    self._emit_finished_outputs({batch.request_id}, None)
                    continue
                if batch.failure is not None:
                    self._run_queued_pipeline_iteration(batch.scheduler_output)
                    continue
                output = self._advance_queued_pipeline_batch(batch)
                if output is not None:
                    self._emit_finished_outputs({batch.request_id}, output)
            except Exception as exc:
                if batch.abort_requested:
                    logger.error("Queued abort cleanup is pending; retaining ownership", exc_info=True)
                else:
                    self._handle_queued_iteration_failure(batch.scheduler_output, exc)

    def _has_queued_pipeline_work(self) -> bool:
        """Return whether retained queued ownership still needs an Engine round."""
        return getattr(self.od_config, "mode", "static") == "queued" and bool(self._queued_pipeline_batches)

    def _queued_pipeline_can_advance_without_schedule(self) -> bool:
        if not self._queued_pipeline_batches:
            return False
        self._maybe_start_queued_decode_round()
        if getattr(self, "_queued_decode_round", None) is not None:
            return True
        # A shared Future can become ready halfway through an Engine round.
        # Drain every committed member before the Scheduler sees the first
        # retired request as a candidate for the next wave.
        if any(
            batch.phase is _QueuedPipelineBatchPhase.STEP_COMMITTED for batch in self._queued_pipeline_batches.values()
        ):
            return True
        has_candidate = getattr(self.scheduler, "has_queued_admission_candidate", None)
        if not callable(has_candidate):
            return False
        owned_request_ids = {batch.request_id for batch in self._queued_pipeline_batches.values()}
        admission_capacity_available = len(self._queued_pipeline_batches) < self._queued_pipeline_admission_limit()
        return not has_candidate(owned_request_ids, admission_capacity_available=admission_capacity_available)

    def _queued_pipeline_admission_limit(self) -> int:
        """Bound prepared tasks by scheduler capacity, including finalizing rows."""
        scheduler_capacity = getattr(self.scheduler, "max_num_running_reqs", None)
        if type(scheduler_capacity) is not int or scheduler_capacity <= 0:
            scheduler_capacity = _max_num_seqs(self.od_config) * int(self.od_config.max_inflight_batches)
        if self._uses_grouped_queued_decode():
            return scheduler_capacity
        finalizing_batches = sum(
            batch.phase is _QueuedPipelineBatchPhase.FINALIZING for batch in self._queued_pipeline_batches.values()
        )
        return scheduler_capacity + finalizing_batches

    def _queued_denoise_batch_count(self) -> int:
        """Count active execution Futures, each of which may carry a microbatch."""
        return len(
            {
                id(batch.step_future)
                for batch in self._queued_pipeline_batches.values()
                if batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED and batch.step_future is not None
            }
        )

    def _should_wait_for_queued_pipeline_update(self) -> bool:
        """Wait only when retained work has no scheduler candidate to advance."""
        if not self._has_queued_pipeline_work():
            return False
        if not self._queued_pipeline_can_advance_without_schedule():
            prepared_capacity_available = len(self._queued_pipeline_batches) < self._queued_pipeline_admission_limit()
            if prepared_capacity_available:
                # A waiting request can be prepared while denoise slots are
                # occupied; the prepared FIFO will be authorized later.
                return False
            max_inflight = int(getattr(self.od_config, "max_inflight_batches", 1))
            has_waiting = getattr(self.scheduler, "has_queued_waiting_request", None)
            if not callable(has_waiting) or self._queued_denoise_batch_count() < max_inflight or not has_waiting():
                return False
        decode_round = getattr(self, "_queued_decode_round", None)
        if decode_round is not None and any(future.done() for future in decode_round.ready_futures):
            if not decode_round.dispatched and all(future.done() for future in decode_round.ready_futures):
                return False
        return not any(
            ((batch.failure is not None or batch.abort_requested) and not self._queued_decode_round_busy())
            or batch.phase is _QueuedPipelineBatchPhase.STEP_COMMITTED
            or (
                batch.phase in {_QueuedPipelineBatchPhase.AUTHORIZED, _QueuedPipelineBatchPhase.STEP_COMPLETED}
                and batch.step_future is not None
                and batch.step_future.done()
            )
            or (batch.decode_future is not None and batch.decode_future.done() and not self._queued_decode_round_busy())
            for batch in self._queued_pipeline_batches.values()
        )

    def _uses_grouped_queued_decode(self) -> bool:
        parallel = getattr(self.od_config, "parallel_config", None)
        return (
            getattr(self.od_config, "mode", "static") == "queued"
            and int(getattr(parallel, "pipeline_parallel_size", 1)) == 2
            and int(getattr(parallel, "vae_patch_parallel_size", 1)) == 1
            and int(getattr(parallel, "data_parallel_size", 1)) == 1
            and int(getattr(self.od_config, "num_gpus", 2)) == 2
            and not getattr(self.od_config, "streaming_output", False)
        )

    def _maybe_start_queued_decode_round(self) -> None:
        """Seal a decode cohort at the request threshold or denoise exhaustion."""
        if not self._uses_grouped_queued_decode() or getattr(self, "_queued_decode_round", None) is not None:
            return
        if any(
            batch.phase is _QueuedPipelineBatchPhase.STEP_COMPLETED for batch in self._queued_pipeline_batches.values()
        ):
            return
        pending = [
            batch
            for batch in self._queued_pipeline_batches.values()
            if batch.phase is _QueuedPipelineBatchPhase.FINALIZING
            and not batch.abort_requested
            and batch.failure is None
            and batch.decode_future is None
            and batch.decoded_output is None
            and not (
                batch.native_output is not None
                and (output := batch.native_output.get_request_output(batch.request_id)) is not None
                and output.result is not None
            )
        ]
        if not pending:
            return
        has_denoise = getattr(self.scheduler, "has_queued_denoise_request", None)
        if callable(has_denoise):
            remaining = has_denoise()
        else:
            remaining = any(not batch.finalizing for batch in self._queued_pipeline_batches.values())
        if len(pending) < int(self.od_config.max_inflight_batches) and remaining:
            return
        self._queued_decode_round = _QueuedDecodeRound(tuple(batch.request_id for batch in pending))

    def _queued_decode_round_busy(self) -> bool:
        decode_round = getattr(self, "_queued_decode_round", None)
        if decode_round is None:
            return False
        return any(not future.done() for future in decode_round.ready_futures) or any(
            batch.decode_future is not None and not batch.decode_future.done()
            for batch in self._queued_pipeline_batches.values()
        )

    def _advance_queued_decode_round(self) -> None:
        self._maybe_start_queued_decode_round()
        decode_round = getattr(self, "_queued_decode_round", None)
        if decode_round is None:
            return
        members = [
            batch for batch in self._queued_pipeline_batches.values() if batch.request_id in decode_round.request_ids
        ]
        if not members:
            self._queued_decode_round = None
            return
        if decode_round.dispatched:
            return
        # Rank0's metadata Future may precede rank1's final DiT. First drain
        # commits, then enqueue an independent readiness command on each rank.
        if any(
            batch.phase
            in {
                _QueuedPipelineBatchPhase.AUTHORIZED,
                _QueuedPipelineBatchPhase.STEP_COMPLETED,
                _QueuedPipelineBatchPhase.STEP_COMMITTED,
            }
            for batch in self._queued_pipeline_batches.values()
        ):
            return
        pp_size = int(self.od_config.parallel_config.pipeline_parallel_size)
        try:
            if not decode_round.ready_futures:
                # Cancellation may have retired a sealed member while the
                # current denoise wave was draining.
                decode_round.request_ids = tuple(batch.request_id for batch in members)
                for rank in range(pp_size):
                    future = self.executor.prepare_queued_decode(decode_round.request_ids, output_rank=rank)
                    if not isinstance(future, concurrent.futures.Future):
                        raise RuntimeError("Queued decode readiness requires a non-blocking Future.")
                    future.add_done_callback(lambda _future: self._notify_queued_pipeline_update())
                    decode_round.ready_futures.append(future)
                return
            if not all(future.done() for future in decode_round.ready_futures):
                return
            for future in decode_round.ready_futures:
                if tuple(future.result()) != decode_round.request_ids:
                    raise RuntimeError("Queued decode readiness returned unexpected request IDs.")
            decode_round.dispatched = True
            # Submit the entire cohort before consuming any output or running
            # all-rank cleanup. Per-rank Worker FIFOs serialize local VAE calls.
            for index, batch in enumerate(members):
                if batch.abort_requested or batch.failure is not None:
                    continue
                batch.output_rank = index % pp_size
                batch.decode_future = self.executor.execute_decode(
                    batch.request_id,
                    output_rank=batch.output_rank,
                    non_block=True,
                )
                if not isinstance(batch.decode_future, concurrent.futures.Future):
                    raise RuntimeError("Queued decode requires a non-blocking Worker Future.")
                batch.decode_future.add_done_callback(lambda _future: self._notify_queued_pipeline_update())
        except BaseException as exc:
            # A partial dispatch still owns Worker state. A readiness/dispatch
            # failure cannot resume DiT on a potentially inconsistent group.
            decode_round.dispatched = True
            for batch in members:
                batch.failure = exc
            self.executor._fail_queued_control("decode cohort dispatch", exc)
            raise

    def _reserve_queued_pipeline_batch(self, scheduler_output: Any) -> _QueuedPipelineBatch:
        if len(self._queued_pipeline_batches) >= self._queued_pipeline_admission_limit():
            raise _QueuedAdmissionDeferredError("queued pipeline prepared-request capacity is exhausted")
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        if len(request_ids) != 1:
            raise ValueError("Queued pipeline task construction requires exactly one scheduled request.")

        request_id = request_ids[0]
        request_state = self.scheduler.get_request_state(request_id)
        if request_state is None:
            raise RuntimeError(f"Queued request {request_id!r} has no Scheduler state.")
        step_index = request_state.req.sampling_params.step_index
        if step_index is None:
            step_index = 0
        batch_id = f"queued-{self._queued_pipeline_epoch}-{scheduler_output.step_id}"
        self._queued_pipeline_epoch += 1
        batch = _QueuedPipelineBatch(
            batch_id=batch_id,
            request_id=request_id,
            step_index=step_index,
            scheduler_output=scheduler_output,
        )
        if batch.batch_id in self._queued_pipeline_batches:
            raise ValueError(f"Queued pipeline batch {batch.batch_id!r} already exists.")
        self._queued_pipeline_batches[batch.batch_id] = batch
        return batch

    @staticmethod
    def _split_queued_scheduler_output(scheduler_output: Any) -> list[Any]:
        """Split a scheduler cycle into one-request task descriptors."""
        outputs = []
        for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
            outputs.append(
                dataclasses.replace(
                    scheduler_output,
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=CachedRequestData(request_ids=[request_id]),
                )
            )
        for request in scheduler_output.scheduled_new_reqs:
            outputs.append(
                dataclasses.replace(
                    scheduler_output,
                    scheduled_new_reqs=[request],
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                )
            )
        return outputs

    def _submit_queued_pipeline_batch(
        self,
        scheduler_output: Any,
        *,
        authorize: bool = True,
    ) -> _QueuedPipelineBatch:
        batch = self._reserve_queued_pipeline_batch(scheduler_output)
        try:
            batch.request_prepared = True
            if scheduler_output.scheduled_new_reqs:
                self.executor.prepare_pipeline_requests(scheduler_output)
            batch.phase = _QueuedPipelineBatchPhase.PREPARED
            if authorize:
                self._authorize_waiting_queued_batches()
            return batch
        except BaseException:
            batch.phase = _QueuedPipelineBatchPhase.FAILED
            raise

    def _authorize_waiting_queued_batches(self) -> None:
        """Submit Worker-local queued work as execution slots open.

        A queued PP step is a sequence of request-local microbatches.  The
        worker must receive the whole compatible sequence in one RPC so its
        rank-local loop can advance request ``i + 1`` while the other PP rank
        is finishing request ``i``.  Splitting this sequence at
        ``max_num_seqs`` creates a Worker busy-loop boundary and removes that
        overlap.  ``max_num_seqs`` still bounds scheduler admission; this
        method only coalesces already-admitted, prepared state.
        """
        self._maybe_start_queued_decode_round()
        if getattr(self, "_queued_decode_round", None) is not None:
            return
        available_slots = int(self.od_config.max_inflight_batches) - self._queued_denoise_batch_count()
        if available_slots <= 0:
            return

        # Preserve FIFO while keeping requests with different step-batch
        # compatibility keys in separate Worker commands.  The key lives in
        # the Scheduler-owned state; queued descriptors themselves intentionally
        # remain one-request records for commit/retirement bookkeeping.
        groups: list[tuple[Any, list[_QueuedPipelineBatch]]] = []
        for batch in self._queued_pipeline_batches.values():
            if batch.phase is not _QueuedPipelineBatchPhase.PREPARED:
                continue
            state = self.scheduler.get_request_state(batch.request_id)
            compatibility_key = getattr(state, "sampling_params_key", None)
            for group_key, group in groups:
                if compatibility_key == group_key:
                    group.append(batch)
                    break
            else:
                groups.append((compatibility_key, [batch]))

        for _compatibility_key, group in groups:
            if available_slots <= 0:
                break
            new_reqs = [request for batch in group for request in batch.scheduler_output.scheduled_new_reqs]
            cached_ids = [
                request_id for batch in group for request_id in batch.scheduler_output.scheduled_cached_reqs.request_ids
            ]
            # ``finished_req_ids`` belongs to the scheduler cycle that
            # produced the first descriptor.  It must not retire a state from
            # another cycle that was coalesced into this Worker-local queue.
            scheduler_output = dataclasses.replace(
                group[0].scheduler_output,
                scheduled_new_reqs=new_reqs,
                scheduled_cached_reqs=CachedRequestData(request_ids=cached_ids),
                finished_req_ids=set(),
            )
            output_ranks = [self._next_queued_pipeline_output_rank() for _ in group]
            execute_kwargs: dict[str, Any] = {
                "non_block": True,
                "output_rank": output_ranks[0],
            }
            if output_ranks[0] != 0:
                execute_kwargs["reply_rank"] = 0
            future = self.executor.execute_step(
                scheduler_output,
                **execute_kwargs,
            )
            if not isinstance(future, concurrent.futures.Future):
                raise RuntimeError("Queued execution requires a non-blocking step Future.")
            future.add_done_callback(lambda _future: self._notify_queued_pipeline_update())
            for batch, output_rank in zip(group, output_ranks, strict=True):
                batch.step_future = future
                batch.output_rank = output_rank
                batch.phase = _QueuedPipelineBatchPhase.AUTHORIZED
            available_slots -= 1

    def _next_queued_pipeline_output_rank(self) -> int:
        parallel = getattr(self.od_config, "parallel_config", None)
        pp_size = max(int(getattr(parallel, "pipeline_parallel_size", 1)), 1)
        if int(getattr(parallel, "vae_patch_parallel_size", 1)) > 1:
            return 0
        previous = getattr(self, "_queued_pipeline_last_output_rank", pp_size - 1)
        rank = (previous + 1) % pp_size
        self._queued_pipeline_last_output_rank = rank
        return rank

    def _commit_queued_pipeline_step(self, batch: _QueuedPipelineBatch) -> frozenset[str]:
        if self._queued_pipeline_batches.get(batch.batch_id) is not batch:
            raise ValueError("Queued pipeline batch is not owned by this Engine.")
        if batch.phase is not _QueuedPipelineBatchPhase.STEP_COMPLETED:
            raise RuntimeError("Queued pipeline batch has not completed its Worker step.")
        commit_pipeline_step = getattr(self.scheduler, "commit_pipeline_step", None)
        if not callable(commit_pipeline_step):
            batch.phase = _QueuedPipelineBatchPhase.FAILED
            raise RuntimeError("Queued pipeline execution requires StepScheduler.commit_pipeline_step().")
        resulting_step = batch.step_index + 1
        try:
            finalizing = frozenset(
                commit_pipeline_step(
                    batch.scheduler_output,
                    {batch.request_id: resulting_step},
                )
            )
            if not finalizing.issubset({batch.request_id}):
                raise RuntimeError("Scheduler finalized a request outside the queued pipeline batch.")
            batch.finalizing = batch.request_id in finalizing
            if finalizing:
                batch.phase = _QueuedPipelineBatchPhase.FINALIZING
            else:
                batch.phase = _QueuedPipelineBatchPhase.STEP_COMMITTED
            # A Worker Future may represent a whole request wave. Do not
            # authorize the next wave after the first request commits: that
            # lets max_inflight_batches=4 advance request 1..N while request
            # 0 is still committing, reordering latent feedback and creating
            # extra PP bubbles. Release the admission gate only after every
            # descriptor sharing this Future has committed its step.
            pending_wave_members = any(
                other is not batch
                and other.step_future is batch.step_future
                and other.phase
                in {
                    _QueuedPipelineBatchPhase.AUTHORIZED,
                    _QueuedPipelineBatchPhase.STEP_COMPLETED,
                }
                for other in self._queued_pipeline_batches.values()
            )
            if not pending_wave_members:
                self._authorize_waiting_queued_batches()
            return finalizing
        except BaseException:
            batch.phase = _QueuedPipelineBatchPhase.FAILED
            raise

    def _finalize_queued_pipeline_batch(self, batch: _QueuedPipelineBatch) -> BatchRunnerOutput | None:
        if batch.phase is not _QueuedPipelineBatchPhase.FINALIZING:
            raise RuntimeError("Queued pipeline batch is not ready for final decode.")
        if batch.decoded_output is not None:
            return batch.decoded_output
        if batch.native_output is not None:
            native_request = batch.native_output.get_request_output(batch.request_id)
            if native_request is not None and native_request.result is not None:
                batch.decoded_output = batch.native_output
                return batch.native_output
        if batch.decode_future is None:
            if self._uses_grouped_queued_decode():
                self._advance_queued_decode_round()
            else:
                batch.decode_future = self.executor.execute_decode(
                    batch.request_id,
                    output_rank=batch.output_rank,
                    non_block=True,
                )
                if not isinstance(batch.decode_future, concurrent.futures.Future):
                    raise RuntimeError("Queued decode requires a non-blocking Worker Future.")
                batch.decode_future.add_done_callback(lambda _future: self._notify_queued_pipeline_update())
        if batch.decode_future is None or not batch.decode_future.done() or self._queued_decode_round_busy():
            return None
        output = batch.decode_future.result()
        if not isinstance(output, BaseRunnerOutput):
            raise RuntimeError(f"Queued decode returned {type(output)!r}, expected BaseRunnerOutput")
        finished = output.get_request_output(batch.request_id)
        if finished is None or finished.result is None or finished.result.error is not None:
            raise RuntimeError("Queued decode returned an unsuccessful final output.")
        batch.decoded_output = output
        return output

    def _retire_queued_pipeline_batch(self, batch: _QueuedPipelineBatch) -> None:
        if self._queued_decode_round_busy():
            return
        if (batch.abort_requested or batch.failure is not None) and not batch.cancelled:
            return
        if batch.phase not in {
            _QueuedPipelineBatchPhase.STEP_COMMITTED,
            _QueuedPipelineBatchPhase.FINALIZING,
            _QueuedPipelineBatchPhase.CANCELLING,
        }:
            raise RuntimeError("Queued pipeline batch is not ready for retirement.")
        request_id = batch.request_id
        if (
            batch.request_prepared
            and not batch.request_cleanup_completed
            and (batch.finalizing or batch.cancelled or batch.abort_requested or batch.failure is not None)
        ):
            cleanup = self.executor.cleanup_finalized_pipeline_request(request_id)
            if not isinstance(cleanup, list) or len(cleanup) != self.od_config.num_gpus or not all(cleanup):
                raise RuntimeError("Queued request cleanup did not acknowledge every Worker.")
            batch.request_cleanup_completed = True
        if batch.finalizing and not batch.abort_requested and not batch.scheduler_completed:
            self.scheduler.complete_pipeline_request(request_id)
            batch.scheduler_completed = True
        if (batch.abort_requested or (batch.cancelled and batch.failure is None)) and not batch.scheduler_completed:
            self.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)
            batch.scheduler_completed = True
        self._queued_pipeline_batches.pop(batch.batch_id, None)
        decode_round = getattr(self, "_queued_decode_round", None)
        if decode_round is not None and not any(
            member.request_id in decode_round.request_ids for member in self._queued_pipeline_batches.values()
        ):
            self._queued_decode_round = None
        self._authorize_waiting_queued_batches()

    def _cancel_queued_pipeline_batch(self, batch: _QueuedPipelineBatch) -> None:
        if self._queued_pipeline_batches.get(batch.batch_id) is not batch:
            raise ValueError("Queued pipeline batch is not owned by this Engine.")
        if batch.cancelled:
            return
        if self._queued_decode_round_busy():
            return
        for future in (batch.step_future, batch.decode_future):
            if future is not None and not future.done():
                try:
                    future.result()
                except BaseException:
                    if not (batch.abort_requested or batch.failure is not None):
                        raise
        if batch.request_prepared and not batch.request_cleanup_completed:
            cleanup = self.executor.cleanup_finalized_pipeline_request(batch.request_id)
            if not isinstance(cleanup, list) or len(cleanup) != self.od_config.num_gpus or not all(cleanup):
                raise RuntimeError("Queued request cleanup did not acknowledge every Worker.")
            batch.request_cleanup_completed = True
        batch.cancelled = True
        batch.phase = _QueuedPipelineBatchPhase.CANCELLING

    def _advance_queued_pipeline_batch(self, batch: _QueuedPipelineBatch) -> BatchRunnerOutput | None:
        """Commit a step Future, finalize a finished request, then release ownership."""
        if batch.phase is _QueuedPipelineBatchPhase.PREPARED:
            return None
        if batch.phase is _QueuedPipelineBatchPhase.AUTHORIZED:
            if batch.step_future is None or not batch.step_future.done():
                return None
            try:
                result = batch.step_future.result()
                if not isinstance(result, BaseRunnerOutput):
                    raise RuntimeError(f"Native queued step returned {type(result)!r}, expected BaseRunnerOutput")
                wave = [
                    member
                    for member in self._queued_pipeline_batches.values()
                    if member.step_future is batch.step_future and member.phase is _QueuedPipelineBatchPhase.AUTHORIZED
                ]
                # Validate the entire Worker result before publishing any
                # member as committed. Otherwise Future completion midway
                # through iteration can rotate the next wave to its tail.
                for member in wave:
                    request_output = result.get_request_output(member.request_id)
                    if request_output is None:
                        raise RuntimeError("Native queued step returned no request output")
                    if request_output.result is not None and request_output.result.error is not None:
                        raise RuntimeError(request_output.result.error)
                for member in wave:
                    member.native_output = result
                    member.phase = _QueuedPipelineBatchPhase.STEP_COMPLETED
                for member in wave:
                    self._commit_queued_pipeline_step(member)
            except BaseException:
                batch.phase = _QueuedPipelineBatchPhase.FAILED
                raise
        if batch.phase is _QueuedPipelineBatchPhase.STEP_COMPLETED:
            self._commit_queued_pipeline_step(batch)
        output = (
            self._finalize_queued_pipeline_batch(batch) if batch.phase is _QueuedPipelineBatchPhase.FINALIZING else None
        )
        if batch.phase is _QueuedPipelineBatchPhase.FINALIZING and output is None:
            return None
        if batch.phase in {
            _QueuedPipelineBatchPhase.STEP_COMMITTED,
            _QueuedPipelineBatchPhase.FINALIZING,
            _QueuedPipelineBatchPhase.CANCELLING,
        }:
            self._retire_queued_pipeline_batch(batch)
            if batch.batch_id in self._queued_pipeline_batches:
                return None
        return output
