# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineMessage,
    PipelineStageConnector,
    PipelineTransferGrant,
    PipelineTransferOffer,
)
from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineStageSpec,
    PipelineStageState,
    PipelineTask,
    PipelineTaskStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _task(batch_id: str = "batch-a") -> PipelineTask:
    return PipelineTask(batch_id=batch_id, request_ids=("req-a",), step_index=0, epoch=1)


def _grant() -> PipelineTransferGrant:
    return PipelineTransferGrant(
        PipelineTransferOffer(
            batch_id="batch-a",
            step_index=0,
            epoch=1,
            branch="conditional",
            edge_kind=PipelineEdgeKind.ACTIVATION,
            src_rank=0,
            dst_rank=1,
        )
    )


def test_stage_state_preserves_fifo_and_one_active_task() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    first, second = _task(), _task("batch-b")

    state.enqueue(first)
    state.enqueue(second)
    state.authorize(first.batch_id)
    state.authorize(second.batch_id)
    assert state.start_next() == first
    assert state.start_next() is None
    assert state.complete_active() == first
    assert state.start_next() == second


def test_stage_state_releases_compute_slot_while_first_batch_awaits_feedback() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    first, second = _task(), _task("batch-b")
    state.enqueue(first)
    state.enqueue(second)
    state.authorize(first.batch_id)
    state.authorize(second.batch_id)

    assert state.start_next() == first
    assert state.await_feedback() == first
    assert state.active_task is None
    assert state.awaiting_feedback == {first.batch_id: first}
    assert state.start_next() == second
    assert state.await_feedback() == second

    assert state.complete_feedback(first.batch_id) == first
    assert state.terminal_statuses[first.batch_id] is PipelineTaskStatus.COMPLETED
    assert state.awaiting_feedback == {second.batch_id: second}


def test_stage_state_rejects_reenqueue_after_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)
    state.start_next()
    state.complete_active()

    with pytest.raises(ValueError, match="already terminal"):
        state.enqueue(task)


def test_stage_state_rejects_duplicate_and_unknown_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True))
    task = _task()
    state.enqueue(task)
    with pytest.raises(ValueError, match="already pending"):
        state.enqueue(task)
    with pytest.raises(RuntimeError, match="active task"):
        PipelineStageState(state.spec).complete_active()


def test_stage_state_requires_authorization_without_bypassing_fifo_head() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    first, second = _task(), _task("batch-b")
    state.enqueue(first)
    state.enqueue(second)
    state.authorize(second.batch_id)

    assert state.start_next() is None
    state.authorize(first.batch_id)
    assert state.start_next() == first


def test_stage_state_cancel_and_retire_preserve_terminal_tombstone() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=0, world_size=2, is_first=True, is_last=False))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)

    assert state.cancel(task.batch_id)
    state.retire(task.batch_id)
    with pytest.raises(ValueError, match="already terminal"):
        state.enqueue(task)


def test_stage_state_cancellation_overrides_local_completion() -> None:
    state = PipelineStageState(PipelineStageSpec(pp_stage_id=1, world_size=2, is_first=False, is_last=True))
    task = _task()
    state.enqueue(task)
    state.authorize(task.batch_id)
    state.start_next()
    state.complete_active()

    assert state.cancel(task.batch_id)
    assert state.terminal_statuses[task.batch_id] is PipelineTaskStatus.CANCELLED
    assert task.batch_id not in state.completed_batches


def test_connector_enforces_credit_and_completion_before_release() -> None:
    connector = PipelineStageConnector(edge="0->1", max_slots=1)
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    with pytest.raises(RuntimeError, match="no send credit"):
        connector.enqueue_send(message)
    with pytest.raises(RuntimeError, match="before transport completion"):
        connector.release_send(ticket)
    connector.mark_send_complete(ticket)
    connector.release_send(ticket)
    assert connector.send_in_use == 0


def test_connector_granted_start_failure_preserves_started_send_ownership() -> None:
    class FailingTransport:
        def start_granted_transfer(self, grant, message):
            del grant, message
            raise RuntimeError("send failed")

        def poll(self):
            return []

    connector = PipelineStageConnector(edge="0->1", transport=FailingTransport())
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")

    ticket = connector.enqueue_send(message)
    assert connector.send_in_use == 1
    assert not ticket.started
    with pytest.raises(RuntimeError, match="send failed"):
        connector.start_granted_send(ticket, _grant())
    assert connector.send_in_use == 1
    assert ticket.started


def test_connector_bounds_receive_polling_and_retirement() -> None:
    class Transport:
        def __init__(self):
            self.messages = []

        def send(self, message):
            del message

        def poll(self, limit=None):
            if limit is not None:
                assert limit > 0
                messages, self.messages = self.messages[:limit], self.messages[limit:]
                return messages
            messages, self.messages = self.messages, []
            return messages

    transport = Transport()
    connector = PipelineStageConnector(edge="0->1", max_slots=2, transport=transport)
    messages = [
        PipelineMessage(batch_id="batch-a", step_index=i, epoch=1, branch="conditional", payload=i) for i in range(2)
    ]
    transport.messages.extend(messages)

    first = connector.poll_received(limit=1)
    assert first == [messages[0]]
    assert connector.receive_depth == 1
    second = connector.poll_received(limit=1)
    assert second == [messages[1]]
    assert connector.receive_depth == 2
    connector.release_received(first[0])
    connector.release_received(second[0])
    assert connector.receive_depth == 0
    connector.retire_batch("batch-a")


def test_connector_rejects_unbounded_legacy_transport_without_consuming() -> None:
    class LegacyTransport:
        def __init__(self, messages):
            self.messages = list(messages)
            self.poll_calls = 0

        def send(self, message):
            del message

        def poll(self):
            self.poll_calls += 1
            messages, self.messages = self.messages, []
            return messages

    messages = [
        PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload=i)
        for i in range(10_000)
    ]
    connector = PipelineStageConnector(edge="0->1", max_slots=2, transport=LegacyTransport(messages))

    with pytest.raises(RuntimeError, match="must support bounded polling"):
        connector.poll_received(limit=1)
    assert connector.transport.messages == messages
    assert connector.transport.poll_calls == 0
    assert connector.transport_pending == 0


def test_connector_passes_caller_limit_to_bounded_transport() -> None:
    class LimitedTransport:
        def __init__(self):
            self.requested_limits = []

        def send(self, message):
            del message

        def poll(self, limit=None):
            self.requested_limits.append(limit)
            return []

    transport = LimitedTransport()
    connector = PipelineStageConnector(edge="0->1", max_slots=8, transport=transport)

    assert connector.poll_received(limit=1) == []
    assert transport.requested_limits == [1]


def test_connector_receive_validation_is_failure_atomic() -> None:
    valid = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    invalid = PipelineMessage(batch_id="batch-b", step_index=0, epoch=1, branch="negative", payload="y")

    class Transport:
        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            return [valid, invalid]

    connector = PipelineStageConnector(edge="0->1", max_slots=2, transport=Transport())

    with pytest.raises(ValueError, match="conditional"):
        connector.poll_received(limit=2)
    assert connector.receive_depth == 0
    assert connector.transport_pending == 0


def test_connector_rejects_distinct_messages_with_duplicate_envelope_identity() -> None:
    first = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="first")
    duplicate = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="duplicate")

    class Transport:
        def __init__(self):
            self.polled = False

        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            if self.polled:
                return [duplicate]
            self.polled = True
            return [first]

    transport = Transport()
    connector = PipelineStageConnector(edge="0->1", max_slots=2, transport=transport)
    connector.poll_received()

    with pytest.raises(RuntimeError, match="duplicate leased message"):
        connector.poll_received()


def test_connector_holds_receive_credit_until_consumer_release() -> None:
    class Transport:
        def __init__(self, messages):
            self.messages = list(messages)

        def send(self, message):
            del message

        def poll(self, limit=None):
            messages, self.messages = self.messages[:limit], self.messages[limit:]
            return messages

    messages = [
        PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload=i) for i in range(2)
    ]
    connector = PipelineStageConnector(edge="0->1", max_slots=1, transport=Transport(messages))

    first = connector.poll_received()[0]
    assert connector.receive_depth == 1
    assert connector.poll_received() == []
    assert connector.transport.messages == [messages[1]]
    connector.release_received(first)
    second = connector.poll_received()[0]
    assert second == messages[1]
    connector.release_received(second)


def test_connector_rejects_retirement_while_receive_consumer_is_active() -> None:
    class Transport:
        def __init__(self, message):
            self.message = message

        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            if self.message is None:
                return []
            message, self.message = self.message, None
            return [message]

    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    connector = PipelineStageConnector(edge="0->1", transport=Transport(message))
    received = connector.poll_received()[0]

    with pytest.raises(RuntimeError, match="receive consumers complete"):
        connector.retire_batch("batch-a", discard_results=True)
    assert connector.receive_depth == 1
    connector.release_received(received)
    connector.retire_batch("batch-a")


def test_connector_discard_rejects_incomplete_send_without_transport_abort() -> None:
    connector = PipelineStageConnector(edge="0->1")
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    ticket.started = True

    with pytest.raises(RuntimeError, match="does not support abort"):
        connector.retire_batch("batch-a", discard_results=True)
    assert connector.send_in_use == 1
    assert not ticket.completed
    assert not ticket.released


def test_connector_discard_waits_for_transport_abort_before_release() -> None:
    class Transport:
        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            return []

        def abort(self, ticket):
            self.aborted = ticket
            return True

    transport = Transport()
    connector = PipelineStageConnector(edge="0->1", transport=transport)
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    ticket.started = True

    connector.retire_batch("batch-a", discard_results=True)
    assert transport.aborted is ticket
    assert ticket.completed
    assert ticket.released
    assert connector.send_in_use == 0


def test_connector_failed_close_wait_preserves_outstanding_ticket() -> None:
    class Transport:
        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            return []

        def wait(self, ticket):
            del ticket
            return False

        def close(self):
            raise AssertionError("close must not run after failed wait")

    connector = PipelineStageConnector(edge="0->1", transport=Transport())
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    ticket.started = True

    with pytest.raises(RuntimeError, match="wait did not complete"):
        connector.close(drain=True)
    assert not connector.closed
    assert connector.send_in_use == 1
    assert not ticket.completed
    assert not ticket.released


def test_connector_close_requires_drain_for_outstanding_send() -> None:
    class Transport:
        def __init__(self):
            self.close_calls = 0

        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            return []

        def wait(self, ticket):
            del ticket
            return True

        def close(self):
            self.close_calls += 1

    transport = Transport()
    connector = PipelineStageConnector(edge="0->1", transport=transport)
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    connector.enqueue_send(message)

    with pytest.raises(RuntimeError, match="outstanding transfers"):
        connector.close()
    connector.close(drain=True)
    assert connector.closed
    assert transport.close_calls == 1
    connector.close(drain=True)
    assert transport.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        connector.poll_received()


def test_connector_backend_close_failure_preserves_local_ownership() -> None:
    class Transport:
        def send(self, message):
            del message

        def poll(self, limit=None):
            del limit
            return []

        def wait(self, ticket):
            del ticket
            return True

        def close(self):
            raise RuntimeError("backend close failed")

    connector = PipelineStageConnector(edge="0->1", transport=Transport())
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="conditional", payload="x")
    ticket = connector.enqueue_send(message)
    ticket.started = True

    with pytest.raises(RuntimeError, match="backend close failed"):
        connector.close(drain=True)
    assert not connector.closed
    assert ticket.completed
    assert not ticket.released
    assert connector.send_in_use == 1


def test_connector_rejects_batch_retirement_after_close() -> None:
    connector = PipelineStageConnector(edge="0->1")
    connector.close()

    with pytest.raises(RuntimeError, match="closed"):
        connector.retire_batch("batch-a")


def test_connector_rejects_non_conditional_messages() -> None:
    connector = PipelineStageConnector(edge="0->1")
    message = PipelineMessage(batch_id="batch-a", step_index=0, epoch=1, branch="negative", payload=None)
    with pytest.raises(ValueError, match="conditional"):
        connector.enqueue_send(message)
