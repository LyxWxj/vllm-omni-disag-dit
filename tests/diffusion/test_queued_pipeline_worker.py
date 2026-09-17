# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineTransferGrant,
    PipelineTransferOffer,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEventType,
    PipelineStageSpec,
    PipelineTask,
    PipelineTaskStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Runner:
    def __init__(self) -> None:
        self.pipeline_batch_contexts = {}
        self.state_cache = {"req-a": object()}
        self.preparation_error: Exception | None = None
        self.execution_error: Exception | None = None
        self.feedback_adoptions = 0

    def prepare_pipeline_batch(self, task, spec, states):
        if self.preparation_error is not None:
            raise self.preparation_error
        context = SimpleNamespace(
            task=task,
            stage_spec=spec,
            states=tuple(states),
            request_state_ids=task.request_ids,
            status=PipelineTaskStatus.PENDING,
        )
        self.pipeline_batch_contexts[(spec.pp_stage_id, task.batch_id)] = context
        return context

    def execute_pipeline_stage(self, context, spec, intermediate_tensors):
        del spec, intermediate_tensors
        if self.execution_error is not None:
            context.status = PipelineTaskStatus.FAILED
            raise self.execution_error
        context.status = PipelineTaskStatus.ACTIVE
        return torch.tensor([3.0])

    def complete_pipeline_step(self, context, spec):
        del spec
        context.status = PipelineTaskStatus.COMPLETED
        return torch.tensor([7.0])

    def adopt_pipeline_feedback(self, context, spec, latents):
        del spec
        self.feedback_adoptions += 1
        context.feedback = latents
        context.status = PipelineTaskStatus.COMPLETED

    def cancel_pipeline_batch(self, pp_stage_id, batch_id):
        context = self.pipeline_batch_contexts[(pp_stage_id, batch_id)]
        context.status = PipelineTaskStatus.CANCELLED
        return context

    def release_pipeline_batch(self, pp_stage_id, batch_id):
        context = self.pipeline_batch_contexts[(pp_stage_id, batch_id)]
        if context.status not in {
            PipelineTaskStatus.COMPLETED,
            PipelineTaskStatus.CANCELLED,
            PipelineTaskStatus.FAILED,
        }:
            raise RuntimeError("Cannot release a non-terminal pipeline batch context.")
        return self.pipeline_batch_contexts.pop((pp_stage_id, batch_id))


def _worker() -> DiffusionWorker:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 4
    worker.model_runner = _Runner()
    worker._pipeline_stages = {}
    return worker


def _task(batch_id: str = "batch-a", *, epoch: int = 2) -> PipelineTask:
    return PipelineTask(batch_id=batch_id, request_ids=("req-a",), step_index=0, epoch=epoch)


def _spec(stage_id: int) -> PipelineStageSpec:
    return PipelineStageSpec(
        pp_stage_id=stage_id,
        world_size=2,
        is_first=stage_id == 0,
        is_last=stage_id == 1,
    )


def test_worker_requires_execute_authorization_before_progress() -> None:
    worker = _worker()
    task = _task()

    accepted = worker.enqueue_pipeline_batch(task, _spec(0))

    assert accepted.event_type is PipelineEventType.ACCEPTED
    assert worker.progress_pipeline(0) is None
    authorized = worker.authorize_pipeline_batch(0, task.batch_id)
    assert authorized.event_type is PipelineEventType.AUTHORIZED

    progress = worker.progress_pipeline(0)
    assert progress is not None
    assert progress.event.event_type is PipelineEventType.STAGE_COMPLETED
    torch.testing.assert_close(progress.output, torch.tensor([3.0]))


@pytest.mark.parametrize("rank", [0, 1])
def test_worker_selects_rank_local_pipeline_descriptor(mocker, rank: int) -> None:
    worker = _worker()
    specs = {0: _spec(0), 1: _spec(1)}
    mocker.patch(
        "vllm_omni.diffusion.distributed.parallel_state.get_pipeline_parallel_rank",
        return_value=rank,
    )

    event = worker.enqueue_pipeline_batch(_task(), specs)
    worker.authorize_pipeline_batch({0: 0, 1: 1}, "batch-a")

    assert event.pp_stage_id == rank
    assert worker.pipeline_stages[rank].spec == specs[rank]
    assert worker.model_runner.pipeline_batch_contexts[(rank, "batch-a")].states == (
        worker.model_runner.state_cache["req-a"],
    )


def test_worker_all_rank_event_poll_clears_every_rank(mocker) -> None:
    worker = _worker()
    local = worker._pipeline_event(PipelineEventType.ACCEPTED, _task(), 0)
    remote = worker._pipeline_event(PipelineEventType.ACCEPTED, _task("batch-b"), 1)
    worker._pipeline_events = [local]
    gather = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker._all_gather_rank_values",
        return_value=[[local], [remote]],
    )

    assert worker.poll_pipeline_events_all_ranks() == [local, remote]
    gather.assert_called_once_with([local])
    assert worker.poll_pipeline_events() == []


def test_first_stage_emits_step_completion_only_after_feedback() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)
    worker.progress_pipeline(0)

    event = worker.complete_pipeline_feedback(0, task.batch_id, torch.tensor([11.0]))

    assert event.event_type is PipelineEventType.STEP_COMPLETED
    assert worker.pipeline_stages[0].active_task is None
    released = worker.release_pipeline_batch(0, task.batch_id)
    assert released.event_type is PipelineEventType.RELEASED
    assert worker.model_runner.pipeline_batch_contexts == {}


def test_last_stage_completes_numerical_step_but_not_global_step() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(1))
    worker.authorize_pipeline_batch(1, task.batch_id)

    progress = worker.progress_pipeline(1, intermediate_tensors=object())

    assert progress is not None
    assert progress.event.event_type is PipelineEventType.STAGE_COMPLETED
    torch.testing.assert_close(progress.output, torch.tensor([7.0]))
    assert worker.pipeline_stages[1].active_task is None
    assert worker.release_pipeline_batch(1, task.batch_id).event_type is PipelineEventType.RELEASED


def test_worker_preserves_fifo_when_only_later_batch_is_authorized() -> None:
    worker = _worker()
    first, second = _task(), _task("batch-b")
    worker.enqueue_pipeline_batch(first, _spec(0))
    worker.enqueue_pipeline_batch(second, _spec(0))
    worker.authorize_pipeline_batch(0, second.batch_id)

    assert worker.progress_pipeline(0) is None
    worker.authorize_pipeline_batch(0, first.batch_id)
    progress = worker.progress_pipeline(0)
    assert progress is not None
    assert progress.event.task is first


def test_worker_cancellation_is_terminal_until_explicit_release() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))

    cancelled = worker.cancel_pipeline_batch(0, task.batch_id)

    assert cancelled.event_type is PipelineEventType.CANCELLED
    assert (0, task.batch_id) in worker.model_runner.pipeline_batch_contexts
    assert worker.release_pipeline_batch(0, task.batch_id).event_type is PipelineEventType.RELEASED


def test_generation_scoped_cancellation_does_not_cancel_reused_request_id() -> None:
    worker = _worker()
    old_task = _task("batch-old", epoch=2)
    new_task = _task("batch-new", epoch=3)
    worker.enqueue_pipeline_batch(old_task, _spec(0))
    worker.enqueue_pipeline_batch(new_task, _spec(0))

    events = worker.cancel_pipeline_requests([("req-a", 2)])

    assert [event.task.batch_id for event in events] == ["batch-old"]
    assert worker.pipeline_stages[0].terminal_statuses["batch-old"] is PipelineTaskStatus.CANCELLED
    assert all(task.batch_id != "batch-old" for task in worker.pipeline_stages[0].pending_tasks)
    assert any(task.batch_id == "batch-new" for task in worker.pipeline_stages[0].pending_tasks)
    assert worker.model_runner.pipeline_batch_contexts[(0, "batch-new")].status is PipelineTaskStatus.PENDING


def test_worker_rejects_release_before_stage_is_terminal_without_losing_context() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))

    with pytest.raises(RuntimeError, match="not terminal on stage"):
        worker.release_pipeline_batch(0, task.batch_id)

    assert (0, task.batch_id) in worker.model_runner.pipeline_batch_contexts


def test_worker_rolls_back_enqueue_before_acceptance_on_prepare_failure() -> None:
    worker = _worker()
    worker.model_runner.preparation_error = RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        worker.enqueue_pipeline_batch(_task(), _spec(0))

    assert 0 not in worker.pipeline_stages


def test_worker_can_install_valid_stage_after_initial_spec_validation_failure() -> None:
    worker = _worker()
    worker.model_runner.preparation_error = ValueError("invalid topology")
    invalid_spec = PipelineStageSpec(pp_stage_id=0, world_size=3, is_first=True, is_last=False)

    with pytest.raises(ValueError, match="invalid topology"):
        worker.enqueue_pipeline_batch(_task(), invalid_spec)

    worker.model_runner.preparation_error = None
    event = worker.enqueue_pipeline_batch(_task(), _spec(0))
    assert event.event_type is PipelineEventType.ACCEPTED
    assert worker.pipeline_stages[0].spec == _spec(0)


def test_prepare_failure_preserves_existing_stage_tombstones() -> None:
    worker = _worker()
    first = _task()
    worker.enqueue_pipeline_batch(first, _spec(0))
    worker.cancel_pipeline_batch(0, first.batch_id)
    worker.release_pipeline_batch(0, first.batch_id)
    stage = worker.pipeline_stages[0]
    worker.model_runner.preparation_error = RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        worker.enqueue_pipeline_batch(_task("batch-b"), _spec(0))

    assert worker.pipeline_stages[0] is stage
    assert first.batch_id in stage.retired_batches
    assert not stage.pending_tasks


def test_worker_execution_failure_becomes_releasable_terminal_state() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)
    worker.model_runner.execution_error = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        worker.progress_pipeline(0)

    assert worker.pipeline_stages[0].terminal_statuses[task.batch_id] is PipelineTaskStatus.FAILED
    assert worker.release_pipeline_batch(0, task.batch_id).event_type is PipelineEventType.RELEASED


def test_two_stage_cancellation_drains_feedback_without_step_completion() -> None:
    first_worker = _worker()
    last_worker = _worker()
    task = _task()
    first_worker.enqueue_pipeline_batch(task, _spec(0))
    last_worker.enqueue_pipeline_batch(task, _spec(1))
    first_worker.authorize_pipeline_batch(0, task.batch_id)
    last_worker.authorize_pipeline_batch(1, task.batch_id)

    activation = first_worker.progress_pipeline(0)
    assert activation is not None
    feedback = last_worker.progress_pipeline(1, activation.output)
    assert feedback is not None

    first_cancelled = first_worker.cancel_pipeline_batch(0, task.batch_id)
    last_cancelled = last_worker.cancel_pipeline_batch(1, task.batch_id)
    drained = first_worker.complete_pipeline_feedback(0, task.batch_id, feedback.output)

    assert first_cancelled.event_type is PipelineEventType.CANCELLED
    assert last_cancelled.event_type is PipelineEventType.CANCELLED
    assert drained.event_type is PipelineEventType.CANCELLED
    assert first_worker.model_runner.feedback_adoptions == 0
    assert first_worker.pipeline_stages[0].terminal_statuses[task.batch_id] is PipelineTaskStatus.CANCELLED
    assert last_worker.pipeline_stages[1].terminal_statuses[task.batch_id] is PipelineTaskStatus.CANCELLED
    assert first_worker.release_pipeline_batch(0, task.batch_id).event_type is PipelineEventType.RELEASED
    assert last_worker.release_pipeline_batch(1, task.batch_id).event_type is PipelineEventType.RELEASED


class _PPGroup:
    world_size = 2
    ranks = [0, 1]

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.rank_in_group = rank
        self.send_calls = []
        self.recv_calls = []

    def isend_tensor_dict(self, payload, dst):
        self.send_calls.append((payload, dst))
        return []

    def irecv_tensor_dict(self, src):
        self.recv_calls.append(src)
        return {"hidden_states": torch.tensor([1.0])}, [], []


def test_worker_starts_only_matching_granted_p2p_endpoint(mocker) -> None:
    sender = _worker()
    receiver = _worker()
    sender.rank = 0
    receiver.rank = 1
    sender_group = _PPGroup(0)
    receiver_group = _PPGroup(1)
    get_group = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        side_effect=[sender_group, receiver_group],
    )
    sender.initialize_pipeline_transports()
    receiver.initialize_pipeline_transports()
    assert get_group.call_count == 2
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    payload = {"hidden_states": torch.tensor([2.0])}
    sender.reserve_pipeline_send(offer, payload)
    assert sender_group.send_calls == []
    assert receiver.accept_pipeline_transfer_offer(offer)
    grant = PipelineTransferGrant(offer)

    assert sender.start_pipeline_transfer(grant)
    assert receiver.start_pipeline_transfer(grant)
    assert sender_group.send_calls == [(payload, 1)]
    assert receiver_group.recv_calls == [0]


def test_worker_readiness_rejects_missing_sender_reservation_before_receive(mocker) -> None:
    sender = _worker()
    receiver = _worker()
    sender.rank = 0
    receiver.rank = 1
    sender_group = _PPGroup(0)
    receiver_group = _PPGroup(1)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        side_effect=[sender_group, receiver_group],
    )
    sender.initialize_pipeline_transports()
    receiver.initialize_pipeline_transports()
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )

    with pytest.raises(KeyError, match="no reserved sender ticket"):
        sender.accept_pipeline_transfer_offer(offer)

    assert receiver_group.recv_calls == []


def test_worker_readiness_reserves_receive_credit_until_release(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    receiver_group = _PPGroup(1)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        return_value=receiver_group,
    )
    receiver.initialize_pipeline_transports(max_slots=1)
    first = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    second = PipelineTransferOffer(
        batch_id="batch-b",
        step_index=0,
        epoch=1,
        branch="conditional",
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )

    assert receiver.accept_pipeline_transfer_offer(first)
    with pytest.raises(RuntimeError, match="no receive credit"):
        receiver.accept_pipeline_transfer_offer(second)
    with pytest.raises(RuntimeError, match="reserved receive credit"):
        receiver.drain_pipeline()

    receiver.start_pipeline_transfer(PipelineTransferGrant(first))
    messages = receiver.poll_pipeline_received(PipelineEdgeKind.ACTIVATION)
    assert len(messages) == 1
    receiver.release_pipeline_received(PipelineEdgeKind.ACTIVATION, messages[0])
    assert receiver.accept_pipeline_transfer_offer(second)


def test_worker_two_receive_slots_do_not_double_count_leased_message(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    receiver_group = _PPGroup(1)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        return_value=receiver_group,
    )
    receiver.initialize_pipeline_transports(max_slots=2)

    def offer(batch_id: str) -> PipelineTransferOffer:
        return PipelineTransferOffer(
            batch_id=batch_id,
            step_index=0,
            epoch=1,
            branch="conditional",
            edge_kind=PipelineEdgeKind.ACTIVATION,
            src_rank=0,
            dst_rank=1,
        )

    first, second, third = offer("batch-a"), offer("batch-b"), offer("batch-c")
    assert receiver.accept_pipeline_transfer_offer(first)
    receiver.start_pipeline_transfer(PipelineTransferGrant(first))
    leased = receiver.poll_pipeline_received(PipelineEdgeKind.ACTIVATION)
    assert len(leased) == 1

    assert receiver.accept_pipeline_transfer_offer(second)
    with pytest.raises(RuntimeError, match="no receive credit"):
        receiver.accept_pipeline_transfer_offer(third)

    receiver.release_pipeline_received(PipelineEdgeKind.ACTIVATION, leased[0])
    assert receiver.accept_pipeline_transfer_offer(third)
