# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.distributed.group_coordinator import TensorMetadata
from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    DistributedP2PTransport,
    PipelineEdgeKind,
    PipelineMessage,
    PipelineStageConnector,
    PipelineTransferCoordinator,
    PipelineTransferGrant,
    PipelineTransferOffer,
    TransferTicket,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _coordinator(activation_edge: tuple[int, int] = (0, 1)) -> PipelineTransferCoordinator:
    return PipelineTransferCoordinator(
        activation_edges={activation_edge},
        feedback_edges={(activation_edge[1], activation_edge[0])},
    )


def _offer(
    batch_id: str,
    *,
    edge_kind: PipelineEdgeKind = PipelineEdgeKind.ACTIVATION,
    src_rank: int = 0,
    dst_rank: int = 1,
) -> PipelineTransferOffer:
    return PipelineTransferOffer(
        batch_id=batch_id,
        step_index=0,
        epoch=1,
        edge_kind=edge_kind,
        src_rank=src_rank,
        dst_rank=dst_rank,
    )


def test_transfer_requires_offer_and_receive_readiness() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-a")
    coordinator.offer(offer)

    assert coordinator.grant_ready() == []
    coordinator.mark_receive_ready(offer.identity)
    grant = coordinator.grant_ready()[0]
    assert grant.offer is offer


def test_receive_readiness_may_precede_offer_and_is_bound_to_destination() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-pre-ready")

    coordinator.mark_receive_ready(offer.identity, rank=1)
    assert coordinator.grant_ready() == []

    coordinator.offer(offer)
    assert coordinator.grant_ready()[0].offer is offer


def test_receive_readiness_rejects_wrong_reporting_rank() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-wrong-ready-rank")

    with pytest.raises(ValueError, match="destination rank"):
        coordinator.mark_receive_ready(offer.identity, rank=0)


def test_cancel_batch_discards_pre_ready_identity_until_retirement() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-cancelled")

    coordinator.mark_receive_ready(offer.identity, rank=1)
    coordinator.cancel_batch(offer.batch_id, offer.epoch)
    assert coordinator.offer(offer) is False
    assert coordinator.grant_ready() == []

    coordinator.retire_batch(offer.batch_id, offer.epoch)
    assert coordinator.offer(offer) is True


def test_transfer_preserves_fifo_within_each_edge() -> None:
    coordinator = _coordinator()
    first = _offer("batch-a")
    second = _offer("batch-b")
    coordinator.offer(first)
    coordinator.offer(second)
    coordinator.mark_receive_ready(second.identity)

    assert coordinator.grant_ready() == []
    coordinator.mark_receive_ready(first.identity)
    assert coordinator.grant_ready()[0].offer is first


def test_second_offer_on_same_directed_edge_cannot_overtake() -> None:
    coordinator = _coordinator()
    first = _offer("batch-a")
    second = _offer("batch-b")
    coordinator.offer(first)
    coordinator.offer(second)
    coordinator.mark_receive_ready(second.identity)

    assert coordinator.grant_ready() == []


def test_feedback_can_progress_when_activation_edge_is_blocked() -> None:
    coordinator = _coordinator()
    activation = _offer("batch-a")
    feedback = _offer(
        "batch-b",
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    coordinator.offer(activation)
    coordinator.offer(feedback)
    coordinator.mark_receive_ready(feedback.identity)

    assert coordinator.grant_ready()[0].offer is feedback


def test_opposite_directions_are_serialized_by_endpoint_ownership() -> None:
    coordinator = _coordinator()
    first = _offer("batch-a", src_rank=0, dst_rank=1)
    second = _offer(
        "batch-b",
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=1,
        dst_rank=0,
    )
    for offer in (first, second):
        coordinator.offer(offer)
        coordinator.mark_receive_ready(offer.identity)

    grants = coordinator.grant_ready(limit=2)

    assert [grant.offer.identity for grant in grants] == [second.identity]
    assert not coordinator.grant_ready()

    coordinator.complete(second.identity, second.src_rank)
    coordinator.complete(second.identity, second.dst_rank)
    assert coordinator.grant_ready()[0].offer is first


def test_transfer_rejects_reversed_activation_direction() -> None:
    coordinator = _coordinator()
    reversed_activation = _offer("batch-a", src_rank=1, dst_rank=0)

    with pytest.raises(ValueError, match="configured edge topology"):
        coordinator.offer(reversed_activation)


def test_coordinator_rejects_invalid_activation_edge() -> None:
    with pytest.raises(ValueError, match="feedback edges must exactly reverse"):
        PipelineTransferCoordinator(activation_edges={(0, 1)}, feedback_edges={(0, 1)})


def test_transfer_identity_rejects_replay_and_duplicate_completion() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-a")
    coordinator.offer(offer)
    with pytest.raises(ValueError, match="duplicate.*offer"):
        coordinator.offer(_offer("batch-a"))
    coordinator.mark_receive_ready(offer.identity)
    coordinator.grant_ready()
    assert not coordinator.complete(offer.identity, 0)
    with pytest.raises(ValueError, match="duplicate.*completion"):
        coordinator.complete(offer.identity, 0)
    assert coordinator.complete(offer.identity, 1)
    with pytest.raises(ValueError, match="duplicate.*offer"):
        coordinator.offer(_offer("batch-a"))


def test_batch_retirement_waits_for_both_transfer_endpoints() -> None:
    coordinator = _coordinator()
    offer = _offer("batch-retire")
    coordinator.offer(offer)
    coordinator.mark_receive_ready(offer.identity)
    coordinator.grant_ready()

    assert not coordinator.batch_retirement_ready(offer.batch_id, offer.epoch)
    assert not coordinator.complete(offer.identity, offer.src_rank)
    assert not coordinator.batch_retirement_ready(offer.batch_id, offer.epoch)
    assert coordinator.complete(offer.identity, offer.dst_rank)
    assert coordinator.batch_retirement_ready(offer.batch_id, offer.epoch)

    coordinator.retire_batch(offer.batch_id, offer.epoch)


class _Work:
    def __init__(self, *, completed: bool = False, wait_result=None, wait_error: BaseException | None = None) -> None:
        self.completed = completed
        self.wait_result = wait_result
        self.wait_error = wait_error
        self.wait_calls = 0

    def is_completed(self) -> bool:
        return self.completed

    def wait(self):
        self.wait_calls += 1
        if self.wait_error is not None:
            raise self.wait_error
        self.completed = True
        return self.wait_result


class _Group:
    def __init__(self, ranks=(2, 3), *, rank=2) -> None:
        self.ranks = list(ranks)
        self.rank = rank
        self.rank_in_group = self.ranks.index(rank)
        self.send_work = _Work()
        self.recv_work = _Work()
        self.send_calls = []
        self.recv_calls = []
        self.postprocess_calls = 0

    def isend_tensor_dict(self, payload, dst):
        self.send_calls.append((payload, dst))
        return [self.send_work]

    def irecv_tensor_dict(self, src):
        self.recv_calls.append(src)

        def postprocess():
            self.postprocess_calls += 1

        return {"hidden_states": "received"}, [self.recv_work], [postprocess]


def _grant() -> PipelineTransferGrant:
    return PipelineTransferGrant(
        _offer("batch-a", src_rank=2, dst_rank=3),
    )


def _message() -> PipelineMessage:
    return PipelineMessage(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        payload={"hidden_states": "sent"},
    )


def test_distributed_p2p_sender_uses_group_local_rank_and_waits() -> None:
    group = _Group()
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    message = _message()
    transport.start_granted_transfer(_grant(), message)
    ticket = TransferTicket(message)

    assert group.send_calls == [(message.payload, 1)]
    assert transport.wait(ticket)
    assert group.send_work.wait_calls == 1
    transport.close()


def test_distributed_p2p_sender_rejects_metadata_mismatch() -> None:
    group = _Group()
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
        payload_metadata=(("hidden_states", TensorMetadata("cuda", torch.float32, (2,))),),
    )
    message = PipelineMessage(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        payload={"hidden_states": torch.ones(1)},
    )

    with pytest.raises(ValueError, match="metadata does not match"):
        transport.start_granted_transfer(PipelineTransferGrant(offer), message)


def test_distributed_p2p_sender_rejects_completed_identity_replay() -> None:
    group = _Group()
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    message = _message()
    grant = _grant()
    transport.start_granted_transfer(grant, message)
    assert transport.wait(TransferTicket(message, started=True))

    with pytest.raises(ValueError, match="duplicate.*send identity"):
        transport.start_granted_transfer(grant, _message())
    assert group.send_calls == [(message.payload, 1)]
    transport.close()


def test_distributed_p2p_send_launch_failure_blocks_replay_and_close() -> None:
    group = _Group()

    def fail_send(payload, dst):
        group.send_calls.append((payload, dst))
        raise RuntimeError("ambiguous send failure")

    group.isend_tensor_dict = fail_send
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    message = _message()
    grant = _grant()

    with pytest.raises(RuntimeError, match="ambiguous send failure"):
        transport.start_granted_transfer(grant, message)
    with pytest.raises(ValueError, match="duplicate.*send identity"):
        transport.start_granted_transfer(grant, message)
    assert group.send_calls == [(message.payload, 1)]
    with pytest.raises(RuntimeError, match="launch outcome is ambiguous"):
        transport.wait(TransferTicket(message, started=True))
    with pytest.raises(RuntimeError, match="outstanding operations"):
        transport.close()


def test_distributed_p2p_sender_rejects_false_wait_completion() -> None:
    group = _Group()
    group.send_work.wait_result = False
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    message = _message()
    transport.start_granted_transfer(_grant(), message)

    with pytest.raises(RuntimeError, match="unsuccessful completion"):
        transport.wait(TransferTicket(message, started=True))
    with pytest.raises(ValueError, match="duplicate.*send identity"):
        transport.start_granted_transfer(_grant(), message)


def test_connector_enqueue_does_not_launch_p2p_before_grant() -> None:
    group = _Group()
    transport = DistributedP2PTransport(
        group=group,
        local_rank=2,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    connector = PipelineStageConnector(edge="2->3", transport=transport)
    ticket = connector.enqueue_send(_message())

    assert group.send_calls == []
    assert not ticket.started
    connector.start_granted_send(ticket, _grant())
    assert group.send_calls == [(_message().payload, 1)]
    assert ticket.started
    with pytest.raises(ValueError, match="already started"):
        connector.start_granted_send(ticket, _grant())


def test_distributed_p2p_receiver_publishes_only_after_completion() -> None:
    group = _Group(rank=3)
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    transport.start_granted_transfer(_grant())

    assert group.recv_calls == [0]
    assert transport.poll(limit=1) == []
    assert group.postprocess_calls == 0
    group.recv_work.completed = True
    received = transport.poll(limit=1)
    assert received[0].payload == {"hidden_states": "received"}
    assert group.postprocess_calls == 1
    transport.close()


def test_distributed_p2p_receiver_wait_failure_is_not_published() -> None:
    group = _Group(rank=3)
    group.recv_work.completed = True
    group.recv_work.wait_error = RuntimeError("receive failed")
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    transport.start_granted_transfer(_grant())

    with pytest.raises(RuntimeError, match="receive failed"):
        transport.poll(limit=1)
    assert group.postprocess_calls == 0
    assert group.recv_work.wait_calls == 1
    with pytest.raises(RuntimeError, match="receive failed"):
        transport.poll(limit=1)
    assert group.recv_work.wait_calls == 1


def test_distributed_p2p_postprocess_failure_preserves_prior_ready_message() -> None:
    class Group(_Group):
        def __init__(self):
            super().__init__(rank=3)
            self.works = [_Work(completed=True), _Work(completed=True)]
            self.callback_calls = [0, 0]

        def irecv_tensor_dict(self, src):
            index = len(self.recv_calls)
            self.recv_calls.append(src)

            def postprocess():
                self.callback_calls[index] += 1
                if index == 1:
                    raise RuntimeError("postprocess failed")

            return {"hidden_states": index}, [self.works[index]], [postprocess]

    group = Group()
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    first_grant = _grant()
    second_grant = PipelineTransferGrant(_offer("batch-b", src_rank=2, dst_rank=3))
    transport.start_granted_transfer(first_grant)
    transport.start_granted_transfer(second_grant)

    with pytest.raises(RuntimeError, match="postprocess failed"):
        transport.poll(limit=2)
    ready = transport.poll(limit=1)
    assert ready[0].batch_id == "batch-a"
    with pytest.raises(RuntimeError, match="postprocess failed"):
        transport.poll(limit=1)
    assert group.callback_calls == [1, 1]


def test_distributed_p2p_rejects_local_rank_mismatched_with_group() -> None:
    group = _Group(rank=2)

    with pytest.raises(ValueError, match="local rank does not match"):
        DistributedP2PTransport(
            group=group,
            local_rank=3,
            edge_kind=PipelineEdgeKind.ACTIVATION,
            src_rank=2,
            dst_rank=3,
        )


def test_distributed_p2p_receiver_rejects_duplicate_active_grant_before_irecv() -> None:
    group = _Group(rank=3)
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    grant = _grant()
    transport.start_granted_transfer(grant)

    with pytest.raises(ValueError, match="duplicate.*receive grant"):
        transport.start_granted_transfer(grant)
    assert group.recv_calls == [0]


def test_distributed_p2p_receiver_rejects_completed_grant_replay() -> None:
    group = _Group(rank=3)
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    grant = _grant()
    transport.start_granted_transfer(grant)
    group.recv_work.completed = True
    assert transport.poll(limit=1)

    with pytest.raises(ValueError, match="duplicate.*receive grant"):
        transport.start_granted_transfer(grant)
    assert group.recv_calls == [0]
    transport.close()


def test_distributed_p2p_receive_launch_failure_blocks_replay_and_close() -> None:
    group = _Group(rank=3)

    def fail_receive(src):
        group.recv_calls.append(src)
        raise RuntimeError("ambiguous receive failure")

    group.irecv_tensor_dict = fail_receive
    transport = DistributedP2PTransport(
        group=group,
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    grant = _grant()

    with pytest.raises(RuntimeError, match="ambiguous receive failure"):
        transport.start_granted_transfer(grant)
    with pytest.raises(ValueError, match="duplicate.*receive grant"):
        transport.start_granted_transfer(grant)
    assert group.recv_calls == [0]
    with pytest.raises(RuntimeError, match="outstanding operations"):
        transport.close()


def test_distributed_p2p_rejects_cross_group_endpoints() -> None:
    with pytest.raises(ValueError, match="belong to the supplied PP group"):
        DistributedP2PTransport(
            group=_Group(ranks=(0, 1), rank=0),
            local_rank=0,
            edge_kind=PipelineEdgeKind.ACTIVATION,
            src_rank=0,
            dst_rank=3,
        )


def test_distributed_p2p_rejects_close_with_pending_receive() -> None:
    transport = DistributedP2PTransport(
        group=_Group(rank=3),
        local_rank=3,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )
    transport.start_granted_transfer(_grant())

    with pytest.raises(RuntimeError, match="outstanding operations"):
        transport.close()
