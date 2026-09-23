# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineTransferGrant,
    PipelineTransferOffer,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEvent,
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
        del intermediate_tensors
        if self.execution_error is not None:
            context.status = PipelineTaskStatus.FAILED
            raise self.execution_error
        context.status = PipelineTaskStatus.ACTIVE
        if spec.is_last:
            return torch.tensor([3.0])
        return SimpleNamespace(tensors={"hidden_states": torch.tensor([3.0])})

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


def _initialize_worker_transports(worker, rank: int, mocker):
    worker.rank = rank
    group = _PPGroup(rank)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    worker.initialize_pipeline_transports()
    return group


def test_worker_requires_execute_authorization_before_progress(mocker) -> None:
    worker = _worker()
    _initialize_worker_transports(worker, 0, mocker)
    task = _task()

    accepted = worker.enqueue_pipeline_batch(task, _spec(0))

    assert accepted.event_type is PipelineEventType.ACCEPTED
    assert worker.progress_pipeline(0) is None
    authorized = worker.authorize_pipeline_batch(0, task.batch_id)
    assert authorized.event_type is PipelineEventType.AUTHORIZED

    progress = worker.progress_pipeline(0)
    assert progress is not None
    assert progress.event.event_type is PipelineEventType.STAGE_COMPLETED
    assert isinstance(progress.output, PipelineTransferOffer)


def test_worker_rejects_progress_before_transport_initialization_without_advancing() -> None:
    worker = _worker()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)
    stage = worker.pipeline_stages[0]
    context = worker.model_runner.pipeline_batch_contexts[(0, task.batch_id)]

    with pytest.raises(RuntimeError, match="transports must be initialized"):
        worker.progress_pipeline(0)

    assert stage.active_task is None
    assert list(stage.pending_tasks) == [task]
    assert task.batch_id in stage.authorized_batches
    assert context.status is PipelineTaskStatus.PENDING


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


def test_pipeline_memory_budget_gathers_the_pp_group(mocker) -> None:
    worker = _worker()
    worker.device = torch.device("cpu")
    pp_group = SimpleNamespace(world_size=2, cpu_group=object())
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=pp_group)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.get_free_memory",
        return_value=100,
    )

    def gather(output, value, *, group):
        assert group is pp_group.cpu_group
        output[:] = [{"rank": 4, "free_bytes": 100}, {"rank": 5, "free_bytes": 80}]

    gather_mock = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.dist.all_gather_object",
        side_effect=gather,
    )

    assert worker.pipeline_stage_memory_budget_bytes() == [
        {"rank": 4, "free_bytes": 100},
        {"rank": 5, "free_bytes": 80},
    ]
    gather_mock.assert_called_once()


def test_first_stage_emits_step_completion_only_after_feedback(mocker) -> None:
    worker = _worker()
    _initialize_worker_transports(worker, 0, mocker)
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)
    progress = worker.progress_pipeline(0)
    assert progress is not None
    worker.start_pipeline_transfer(PipelineTransferGrant(progress.output))
    worker.retire_pipeline_send(progress.output.identity)

    event = worker.complete_pipeline_feedback(0, task.batch_id, torch.tensor([11.0]))

    assert event.event_type is PipelineEventType.STEP_COMPLETED
    assert worker.pipeline_stages[0].active_task is None
    released = worker.release_pipeline_batch(0, task.batch_id)
    assert released.event_type is PipelineEventType.RELEASED
    assert worker.model_runner.pipeline_batch_contexts == {}


def test_last_stage_completes_numerical_step_but_not_global_step(mocker) -> None:
    worker = _worker()
    _initialize_worker_transports(worker, 1, mocker)
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(1))
    worker.authorize_pipeline_batch(1, task.batch_id)

    progress = worker.progress_pipeline(1, intermediate_tensors=object())

    assert progress is not None
    assert progress.event.event_type is PipelineEventType.STAGE_COMPLETED
    assert isinstance(progress.output, PipelineTransferOffer)
    assert worker.pipeline_stages[1].active_task is None
    worker.start_pipeline_transfer(PipelineTransferGrant(progress.output))
    worker.retire_pipeline_send(progress.output.identity)
    assert worker.release_pipeline_batch(1, task.batch_id).event_type is PipelineEventType.RELEASED


def test_worker_preserves_fifo_when_only_later_batch_is_authorized(mocker) -> None:
    worker = _worker()
    _initialize_worker_transports(worker, 0, mocker)
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


def test_cancellation_rpc_consumes_acknowledgement_without_dropping_unrelated_event() -> None:
    worker = _worker()
    task = _task("batch-cancel")
    unrelated = PipelineTask(
        batch_id="batch-other",
        request_ids=("req-other",),
        step_index=0,
        epoch=task.epoch,
    )
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.cancel_pipeline_batch(0, task.batch_id)
    worker.pipeline_events.clear()
    unrelated_event = PipelineEvent(PipelineEventType.ACCEPTED, unrelated, 0, worker.rank)
    worker.pipeline_events.append(unrelated_event)

    acknowledgements = worker.cancel_pipeline_requests_all_ranks([("req-a", task.epoch)])

    assert [event.task.batch_id for event in acknowledgements] == [task.batch_id]
    assert worker.poll_pipeline_events() == [unrelated_event]


def test_cancellation_rpc_consumes_local_acknowledgement_when_peer_fails(mocker) -> None:
    worker = _worker()
    task = _task("batch-cancel")
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.pipeline_events.clear()

    def fail_after_local_cancel(_description, callback):
        callback()
        raise RuntimeError("peer cancellation failed")

    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker._run_and_gather_rank_values",
        side_effect=fail_after_local_cancel,
    )

    with pytest.raises(RuntimeError, match="peer cancellation failed"):
        worker.cancel_pipeline_requests_all_ranks([("req-a", task.epoch)])

    assert worker.poll_pipeline_events() == []


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


def test_worker_execution_failure_becomes_releasable_terminal_state(mocker) -> None:
    worker = _worker()
    _initialize_worker_transports(worker, 0, mocker)
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

    first_context = first_worker.model_runner.pipeline_batch_contexts[(0, task.batch_id)]
    last_context = last_worker.model_runner.pipeline_batch_contexts[(1, task.batch_id)]
    assert first_worker.pipeline_stages[0].start_next() is task
    first_context.status = PipelineTaskStatus.ACTIVE
    first_worker.pipeline_stages[0].await_feedback()
    assert last_worker.pipeline_stages[1].start_next() is task
    last_context.status = PipelineTaskStatus.COMPLETED
    last_worker.pipeline_stages[1].complete_active()

    first_cancelled = first_worker.cancel_pipeline_batch(0, task.batch_id)
    last_cancelled = last_worker.cancel_pipeline_batch(1, task.batch_id)
    drained = first_worker.complete_pipeline_feedback(0, task.batch_id, torch.tensor([7.0]))

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
        self.receive_payload = {"hidden_states": torch.tensor([1.0])}

    def isend_tensor_dict(self, payload, dst):
        self.send_calls.append((payload, dst))
        return []

    def irecv_tensor_dict(self, src):
        self.recv_calls.append(src)
        return self.receive_payload, [], []


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


def test_first_stage_progress_reserves_activation_without_returning_tensors(mocker) -> None:
    worker = _worker()
    worker.rank = 0
    group = _PPGroup(0)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    worker.initialize_pipeline_transports()
    worker.model_runner.execute_pipeline_stage = Mock(
        return_value=SimpleNamespace(tensors={"hidden_states": torch.tensor([3.0])})
    )
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)

    progress = worker.progress_pipeline(0)

    assert progress is not None
    assert isinstance(progress.output, PipelineTransferOffer)
    assert progress.output.edge_kind is PipelineEdgeKind.ACTIVATION
    ticket = worker.pipeline_send_tickets[progress.output.identity]
    assert ticket.message.payload.keys() == {"hidden_states"}
    assert not ticket.started
    assert group.send_calls == []


def test_last_stage_progress_reserves_feedback_without_returning_latents(mocker) -> None:
    worker = _worker()
    worker.rank = 1
    group = _PPGroup(1)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    worker.initialize_pipeline_transports()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(1))
    worker.authorize_pipeline_batch(1, task.batch_id)

    progress = worker.progress_pipeline(1, intermediate_tensors=object())

    assert progress is not None
    assert isinstance(progress.output, PipelineTransferOffer)
    assert progress.output.edge_kind is PipelineEdgeKind.FEEDBACK
    ticket = worker.pipeline_send_tickets[progress.output.identity]
    torch.testing.assert_close(ticket.message.payload["latents"], torch.tensor([7.0]))
    assert not ticket.started
    assert group.send_calls == []

    with pytest.raises(RuntimeError, match="retained transfer ownership"):
        worker.release_pipeline_batch(1, task.batch_id)
    assert (1, task.batch_id) in worker.model_runner.pipeline_batch_contexts

    worker.start_pipeline_transfer(PipelineTransferGrant(progress.output))
    assert worker.retire_pipeline_send(progress.output.identity)
    assert progress.output.identity not in worker.pipeline_send_tickets
    assert worker.release_pipeline_batch(1, task.batch_id).event_type is PipelineEventType.RELEASED


def test_first_stage_release_waits_for_feedback_receive_lease(mocker) -> None:
    worker = _worker()
    worker.rank = 0
    group = _PPGroup(0)
    feedback = torch.tensor([5.0])
    group.receive_payload = {"latents": feedback}
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    worker.initialize_pipeline_transports()
    task = _task()
    worker.enqueue_pipeline_batch(task, _spec(0))
    worker.authorize_pipeline_batch(0, task.batch_id)
    stage = worker.pipeline_stages[0]
    assert stage.start_next() is task
    stage.await_feedback()
    context = worker.model_runner.pipeline_batch_contexts[(0, task.batch_id)]
    context.status = PipelineTaskStatus.ACTIVE
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    assert worker.accept_pipeline_transfer_offer(offer)
    worker.start_pipeline_transfer(PipelineTransferGrant(offer))
    messages = worker.poll_pipeline_received(PipelineEdgeKind.FEEDBACK)
    assert len(messages) == 1
    worker.complete_pipeline_feedback(0, task.batch_id, messages[0].payload["latents"])

    with pytest.raises(RuntimeError, match="retained receive ownership"):
        worker.release_pipeline_batch(0, task.batch_id)
    assert (0, task.batch_id) in worker.model_runner.pipeline_batch_contexts

    worker.release_pipeline_received(PipelineEdgeKind.FEEDBACK, messages[0])
    assert worker.release_pipeline_batch(0, task.batch_id).event_type is PipelineEventType.RELEASED


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
    assert not receiver.accept_pipeline_transfer_offer(second)
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
    assert not receiver.accept_pipeline_transfer_offer(third)

    receiver.release_pipeline_received(PipelineEdgeKind.ACTIVATION, leased[0])
    assert receiver.accept_pipeline_transfer_offer(third)


def test_worker_progresses_one_step_through_activation_and_feedback(mocker) -> None:
    first = _worker()
    last = _worker()
    first_group = _PPGroup(0)
    last_group = _PPGroup(1)
    first.rank = 0
    last.rank = 1
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        side_effect=[first_group, last_group],
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=None,
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.is_available",
        return_value=False,
    )
    first.initialize_pipeline_transports()
    last.initialize_pipeline_transports()
    task = _task()
    for worker, stage_id in ((first, 0), (last, 1)):
        worker.enqueue_pipeline_batch(task, _spec(stage_id))
        worker.authorize_pipeline_batch(stage_id, task.batch_id)

    activation = first.progress_pipeline(0).output
    assert isinstance(activation, PipelineTransferOffer)
    assert first.accept_pipeline_transfer_offer(activation)
    assert last.accept_pipeline_transfer_offer(activation)
    activation_grant = PipelineTransferGrant(activation)
    first.start_pipeline_transfer(activation_grant)
    last_group.receive_payload = first.pipeline_send_tickets[activation.identity].message.payload
    last.start_pipeline_transfer(activation_grant)

    first_activation_progress = first.progress_pipeline_transfers()
    last_activation_progress = last.progress_pipeline_transfers()
    assert [completion.rank for completion in first_activation_progress.completions] == [0]
    assert [completion.rank for completion in last_activation_progress.completions] == [1]
    feedback = last_activation_progress.offers[0]
    assert feedback.edge_kind is PipelineEdgeKind.FEEDBACK

    assert first.accept_pipeline_transfer_offer(feedback)
    assert last.accept_pipeline_transfer_offer(feedback)
    feedback_grant = PipelineTransferGrant(feedback)
    last.start_pipeline_transfer(feedback_grant)
    first_group.receive_payload = last.pipeline_send_tickets[feedback.identity].message.payload
    first.start_pipeline_transfer(feedback_grant)

    last_feedback_progress = last.progress_pipeline_transfers()
    first_feedback_progress = first.progress_pipeline_transfers()
    assert [completion.rank for completion in last_feedback_progress.completions] == [1]
    assert [completion.rank for completion in first_feedback_progress.completions] == [0]
    assert first.pipeline_stages[0].terminal_statuses[task.batch_id] is PipelineTaskStatus.COMPLETED
    assert not first.pipeline_send_tickets
    assert not last.pipeline_send_tickets
    assert not first.pipeline_receive_reservations
    assert not last.pipeline_receive_reservations


def test_progress_poll_executes_next_first_stage_batch_while_prior_feedback_is_pending(mocker) -> None:
    worker = _worker()
    worker.rank = 0
    group = _PPGroup(0)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    worker.initialize_pipeline_transports(max_slots=1)

    first = _task("batch-a")
    second = PipelineTask(batch_id="batch-b", request_ids=("req-b",), step_index=0, epoch=3)
    worker.model_runner.state_cache["req-b"] = object()
    for task in (first, second):
        worker.enqueue_pipeline_batch(task, _spec(0))
        worker.authorize_pipeline_batch(0, task.batch_id)

    first_progress = worker.progress_pipeline_transfers()
    first_offer = first_progress.offers[0]
    assert first_offer.batch_id == first.batch_id
    assert worker.pipeline_stages[0].active_task is None
    assert worker.pipeline_stages[0].awaiting_feedback == {first.batch_id: first}

    grant = PipelineTransferGrant(first_offer)
    worker.accept_pipeline_transfer_offer(first_offer)
    worker.start_pipeline_transfer(grant)
    second_progress = worker.progress_pipeline_transfers()

    assert [completion.identity for completion in second_progress.completions] == [first_offer.identity]
    assert [offer.batch_id for offer in second_progress.offers] == [second.batch_id]
    assert worker.pipeline_stages[0].active_task is None
    assert worker.pipeline_stages[0].awaiting_feedback == {
        first.batch_id: first,
        second.batch_id: second,
    }

    event = worker.complete_pipeline_feedback(0, first.batch_id, torch.tensor([9.0]))

    assert event.event_type is PipelineEventType.STEP_COMPLETED
    assert worker.pipeline_stages[0].terminal_statuses[first.batch_id] is PipelineTaskStatus.COMPLETED
    assert worker.pipeline_stages[0].awaiting_feedback == {second.batch_id: second}


def test_two_batch_progress_runs_stage0_b_while_stage1_consumes_a(mocker) -> None:
    first = _worker()
    last = _worker()
    first.rank = 0
    last.rank = 1
    first_group = _PPGroup(0)
    last_group = _PPGroup(1)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.get_pp_group",
        side_effect=[first_group, last_group],
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=None,
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.is_available",
        return_value=False,
    )
    first.initialize_pipeline_transports(max_slots=2)
    last.initialize_pipeline_transports(max_slots=2)

    batch_a = _task("batch-a")
    batch_b = PipelineTask(batch_id="batch-b", request_ids=("req-b",), step_index=0, epoch=3)
    first.model_runner.state_cache["req-b"] = object()
    last.model_runner.state_cache["req-b"] = object()
    for worker in (first, last):
        for task in (batch_a, batch_b):
            worker.enqueue_pipeline_batch(task, _spec(worker.rank))
            worker.authorize_pipeline_batch(worker.rank, task.batch_id)

    def start_transfer(sender, receiver, receiver_group, offer) -> None:
        sender.accept_pipeline_transfer_offer(offer)
        receiver.accept_pipeline_transfer_offer(offer)
        grant = PipelineTransferGrant(offer)
        sender.start_pipeline_transfer(grant)
        receiver_group.receive_payload = sender.pipeline_send_tickets[offer.identity].message.payload
        receiver.start_pipeline_transfer(grant)

    # Round 1 creates and grants batch A's activation.
    first_round = first.progress_pipeline_transfers()
    last_round = last.progress_pipeline_transfers()
    activation_a = first_round.offers[0]
    assert activation_a.batch_id == batch_a.batch_id
    assert last_round.offers == []
    start_transfer(first, last, last_group, activation_a)

    # In one shared progress round, stage 0 forwards B while stage 1 executes A.
    first_round = first.progress_pipeline_transfers()
    last_round = last.progress_pipeline_transfers()
    activation_b = first_round.offers[0]
    feedback_a = last_round.offers[0]
    assert activation_b.batch_id == batch_b.batch_id
    assert feedback_a.batch_id == batch_a.batch_id
    assert first.model_runner.pipeline_batch_contexts[(0, batch_b.batch_id)].status is PipelineTaskStatus.ACTIVE
    assert last.pipeline_stages[1].terminal_statuses[batch_a.batch_id] is PipelineTaskStatus.COMPLETED
    start_transfer(first, last, last_group, activation_b)
    start_transfer(last, first, first_group, feedback_a)

    # Stage 1 consumes B while stage 0 adopts A's feedback; both remain identity-scoped.
    first_round = first.progress_pipeline_transfers()
    last_round = last.progress_pipeline_transfers()
    feedback_b = last_round.offers[0]
    assert feedback_b.batch_id == batch_b.batch_id
    assert any(event.task == batch_a for event in first.pipeline_events)
    start_transfer(last, first, first_group, feedback_b)

    first.progress_pipeline_transfers()
    last.progress_pipeline_transfers()
    assert first.pipeline_stages[0].terminal_statuses[batch_a.batch_id] is PipelineTaskStatus.COMPLETED
    assert first.pipeline_stages[0].terminal_statuses[batch_b.batch_id] is PipelineTaskStatus.COMPLETED
    assert last.pipeline_stages[1].terminal_statuses[batch_b.batch_id] is PipelineTaskStatus.COMPLETED


def test_release_rpc_consumes_acknowledgement_without_dropping_next_batch_event() -> None:
    worker = _worker()
    first = _task("batch-a")
    second = _task("batch-b")
    spec = _spec(0)
    worker.enqueue_pipeline_batch(first, spec)
    worker.authorize_pipeline_batch(0, first.batch_id)
    worker.pipeline_stages[0].start_next()
    worker.model_runner.pipeline_batch_contexts[(0, first.batch_id)].status = PipelineTaskStatus.COMPLETED
    worker.pipeline_stages[0].complete_active()
    worker.enqueue_pipeline_batch(second, spec)
    second_event = worker.pipeline_events[-1]

    acknowledgements = worker.release_pipeline_batch_all_ranks(0, first.batch_id)

    assert len(acknowledgements) == 1
    assert acknowledgements[0].event_type is PipelineEventType.RELEASED
    remaining = worker.poll_pipeline_events()
    assert second_event in remaining
    assert not any(
        event.event_type is PipelineEventType.RELEASED and event.task.batch_id == first.batch_id for event in remaining
    )


def test_worker_holds_receive_lease_until_consumer_event_completes(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    group = _PPGroup(1)
    event = Mock()
    event.query.side_effect = [False, True]
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=event,
    )
    receiver.initialize_pipeline_transports()
    task = _task()
    receiver.enqueue_pipeline_batch(task, _spec(1))
    receiver.authorize_pipeline_batch(1, task.batch_id)
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    receiver.accept_pipeline_transfer_offer(offer)
    receiver.start_pipeline_transfer(PipelineTransferGrant(offer))

    first_progress = receiver.progress_pipeline_transfers()
    assert first_progress.completions == []
    assert offer.identity in receiver.pipeline_receive_reservations
    second_progress = receiver.progress_pipeline_transfers()
    assert [completion.identity for completion in second_progress.completions] == [offer.identity]
    assert offer.identity not in receiver.pipeline_receive_reservations


def test_activation_waits_for_stage_authorization_before_consumption(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    group = _PPGroup(1)
    event = Mock()
    event.query.side_effect = [False, True]
    record_event = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=event,
    )
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    receiver.initialize_pipeline_transports()
    task = _task()
    receiver.enqueue_pipeline_batch(task, _spec(1))
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    receiver.accept_pipeline_transfer_offer(offer)
    receiver.start_pipeline_transfer(PipelineTransferGrant(offer))

    before_authorization = receiver.progress_pipeline_transfers()
    assert before_authorization.offers == []
    assert before_authorization.completions == []
    assert offer.identity in receiver.pipeline_receive_reservations
    assert len(receiver.pipeline_pending_received[PipelineEdgeKind.ACTIVATION]) == 1
    record_event.assert_not_called()

    receiver.authorize_pipeline_batch(1, task.batch_id)
    after_authorization = receiver.progress_pipeline_transfers()
    assert len(after_authorization.offers) == 1
    assert after_authorization.offers[0].edge_kind is PipelineEdgeKind.FEEDBACK
    assert after_authorization.completions == []
    assert offer.identity in receiver.pipeline_receive_reservations
    assert len(receiver.pipeline_pending_received[PipelineEdgeKind.ACTIVATION]) == 0

    after_device_completion = receiver.progress_pipeline_transfers()
    assert [completion.identity for completion in after_device_completion.completions] == [offer.identity]
    assert offer.identity not in receiver.pipeline_receive_reservations


def test_worker_rejects_stale_activation_identity_before_execution(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    group = _PPGroup(1)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    receiver.initialize_pipeline_transports()
    task = _task(epoch=2)
    receiver.enqueue_pipeline_batch(task, _spec(1))
    receiver.authorize_pipeline_batch(1, task.batch_id)
    stale = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=1,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    receiver.accept_pipeline_transfer_offer(stale)
    receiver.start_pipeline_transfer(PipelineTransferGrant(stale))

    with pytest.raises(RuntimeError, match="Stale pipeline message identity"):
        receiver.progress_pipeline_transfers()

    context = receiver.model_runner.pipeline_batch_contexts[(1, task.batch_id)]
    assert context.status is PipelineTaskStatus.PENDING
    assert receiver.model_runner.pipeline_batch_contexts[(1, task.batch_id)].task is task


def test_worker_rejects_stale_feedback_identity_before_adoption(mocker) -> None:
    receiver = _worker()
    receiver.rank = 0
    group = _PPGroup(0)
    group.receive_payload = {"latents": torch.tensor([5.0])}
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    receiver.initialize_pipeline_transports()
    task = _task(epoch=2)
    receiver.enqueue_pipeline_batch(task, _spec(0))
    receiver.authorize_pipeline_batch(0, task.batch_id)
    assert receiver.pipeline_stages[0].start_next() is task
    receiver.pipeline_stages[0].await_feedback()
    context = receiver.model_runner.pipeline_batch_contexts[(0, task.batch_id)]
    context.status = PipelineTaskStatus.ACTIVE
    stale = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index + 1,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    receiver.accept_pipeline_transfer_offer(stale)
    receiver.start_pipeline_transfer(PipelineTransferGrant(stale))

    with pytest.raises(RuntimeError, match="Stale pipeline message identity"):
        receiver.progress_pipeline_transfers()

    assert receiver.model_runner.feedback_adoptions == 0
    assert context.status is PipelineTaskStatus.ACTIVE


def test_cancelled_stage_one_drains_activation_without_execution(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    group = _PPGroup(1)
    record_event = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event")
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    receiver.initialize_pipeline_transports()
    task = _task()
    receiver.enqueue_pipeline_batch(task, _spec(1))
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    receiver.accept_pipeline_transfer_offer(offer)
    receiver.start_pipeline_transfer(PipelineTransferGrant(offer))
    receiver.cancel_pipeline_batch(1, task.batch_id)

    progress = receiver.progress_pipeline_transfers()

    assert progress.offers == []
    assert [completion.identity for completion in progress.completions] == [offer.identity]
    assert receiver.model_runner.pipeline_batch_contexts[(1, task.batch_id)].status is PipelineTaskStatus.CANCELLED
    assert offer.identity not in receiver.pipeline_receive_reservations
    record_event.assert_not_called()


def test_cancelled_stage_zero_drains_feedback_without_adoption(mocker) -> None:
    receiver = _worker()
    receiver.rank = 0
    group = _PPGroup(0)
    group.receive_payload = {"latents": torch.tensor([5.0])}
    record_event = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event")
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    receiver.initialize_pipeline_transports()
    task = _task()
    receiver.enqueue_pipeline_batch(task, _spec(0))
    receiver.authorize_pipeline_batch(0, task.batch_id)
    assert receiver.pipeline_stages[0].start_next() is task
    receiver.model_runner.pipeline_batch_contexts[(0, task.batch_id)].status = PipelineTaskStatus.ACTIVE
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    receiver.accept_pipeline_transfer_offer(offer)
    receiver.start_pipeline_transfer(PipelineTransferGrant(offer))
    receiver.cancel_pipeline_batch(0, task.batch_id)

    progress = receiver.progress_pipeline_transfers()

    assert [completion.identity for completion in progress.completions] == [offer.identity]
    assert receiver.model_runner.feedback_adoptions == 0
    assert offer.identity not in receiver.pipeline_receive_reservations
    record_event.assert_not_called()


def test_accelerator_consumer_event_failure_retains_receive_ownership(mocker) -> None:
    receiver = _worker()
    receiver.rank = 1
    group = _PPGroup(1)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_pp_group", return_value=group)
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=None,
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.is_available",
        return_value=True,
    )
    receiver.initialize_pipeline_transports()
    task = _task()
    receiver.enqueue_pipeline_batch(task, _spec(1))
    receiver.authorize_pipeline_batch(1, task.batch_id)
    offer = PipelineTransferOffer(
        batch_id=task.batch_id,
        step_index=task.step_index,
        epoch=task.epoch,
        branch=task.branch,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=0,
        dst_rank=1,
    )
    receiver.accept_pipeline_transfer_offer(offer)
    receiver.start_pipeline_transfer(PipelineTransferGrant(offer))

    with pytest.raises(RuntimeError, match="failed to record.*consumer completion"):
        receiver.progress_pipeline_transfers()

    assert offer.identity in receiver.pipeline_receive_reservations
    assert offer.identity in receiver.pipeline_receive_consumers
    assert receiver.pipeline_stages[1].terminal_statuses[task.batch_id] is PipelineTaskStatus.COMPLETED
    later = receiver.progress_pipeline_transfers()
    assert later.completions == []
    assert offer.identity in receiver.pipeline_receive_reservations
