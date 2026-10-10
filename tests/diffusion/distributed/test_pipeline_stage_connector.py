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
    PipelineTransferGrant,
    PipelineTransferOffer,
    TransferTicket,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


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

    def isend_tensor_dict(self, payload, dst, metadata_list=None):
        del metadata_list
        self.send_calls.append((payload, dst))
        return [self.send_work]

    def irecv_tensor_dict(self, src, metadata_list=None):
        del metadata_list
        self.recv_calls.append(src)

        def postprocess():
            self.postprocess_calls += 1

        return {"hidden_states": "received"}, [self.recv_work], [postprocess]


def _offer(batch_id: str = "batch-a") -> PipelineTransferOffer:
    return PipelineTransferOffer(
        batch_id=batch_id,
        step_index=0,
        epoch=1,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )


def _message() -> PipelineMessage:
    return PipelineMessage(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        payload={"hidden_states": "sent"},
    )


def _transport(group: _Group, local_rank: int) -> DistributedP2PTransport:
    return DistributedP2PTransport(
        group=group,
        local_rank=local_rank,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
    )


def test_sender_uses_group_local_rank_and_waits() -> None:
    group = _Group()
    transport = _transport(group, 2)
    message = _message()
    transport.start_granted_transfer(PipelineTransferGrant(_offer()), message)

    assert group.send_calls == [(message.payload, 1)]
    assert transport.wait(TransferTicket(message))
    assert group.send_work.wait_calls == 1
    transport.close()


def test_sender_rejects_metadata_mismatch() -> None:
    group = _Group()
    transport = _transport(group, 2)
    offer = PipelineTransferOffer(
        batch_id="batch-a",
        step_index=0,
        epoch=1,
        edge_kind=PipelineEdgeKind.ACTIVATION,
        src_rank=2,
        dst_rank=3,
        payload_metadata=(("hidden_states", TensorMetadata("cuda", torch.float32, (2,))),),
    )
    message = PipelineMessage("batch-a", 0, 1, {"hidden_states": torch.ones(1)})

    with pytest.raises(ValueError, match="metadata does not match"):
        transport.start_granted_transfer(PipelineTransferGrant(offer), message)


def test_connector_can_start_worker_local_send_without_grant_coordinator() -> None:
    group = _Group()
    transport = _transport(group, 2)
    connector = PipelineStageConnector(edge="2->3:activation", transport=transport)
    message = _message()
    ticket = connector.enqueue_send(message)

    connector.start_send(ticket, _offer())

    assert ticket.started
    assert group.send_calls == [(message.payload, 1)]


def test_receiver_publishes_only_after_completion() -> None:
    group = _Group(rank=3)
    transport = _transport(group, 3)
    transport.start_granted_transfer(PipelineTransferGrant(_offer()))

    assert group.recv_calls == [0]
    assert transport.poll(limit=1) == []
    group.recv_work.completed = True
    received = transport.poll(limit=1)
    assert received[0].payload == {"hidden_states": "received"}
    assert group.postprocess_calls == 1
    transport.close()


def test_postprocess_failure_preserves_prior_ready_message() -> None:
    class Group(_Group):
        def __init__(self):
            super().__init__(rank=3)
            self.works = [_Work(completed=True), _Work(completed=True)]
            self.callback_calls = [0, 0]

        def irecv_tensor_dict(self, src, metadata_list=None):
            del metadata_list
            index = len(self.recv_calls)
            self.recv_calls.append(src)

            def postprocess():
                self.callback_calls[index] += 1
                if index == 1:
                    raise RuntimeError("postprocess failed")

            return {"hidden_states": index}, [self.works[index]], [postprocess]

    group = Group()
    transport = _transport(group, 3)
    transport.start_granted_transfer(PipelineTransferGrant(_offer("batch-a")))
    transport.start_granted_transfer(PipelineTransferGrant(_offer("batch-b")))

    with pytest.raises(RuntimeError, match="postprocess failed"):
        transport.poll(limit=2)
    assert transport.poll(limit=1)[0].batch_id == "batch-a"
    with pytest.raises(RuntimeError, match="postprocess failed"):
        transport.poll(limit=1)


def test_transport_rejects_group_endpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="local rank does not match"):
        _transport(_Group(rank=2), 3)
