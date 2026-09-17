# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineTransferCoordinator,
    PipelineTransferOffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _coordinator(*edges: tuple[int, int]) -> PipelineTransferCoordinator:
    activation_edges = set(edges or ((0, 1),))
    return PipelineTransferCoordinator(
        activation_edges=activation_edges,
        feedback_edges={(dst, src) for src, dst in activation_edges},
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
        branch="conditional",
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
    assert coordinator.snapshot() == {
        "offers": 0,
        "ready": 0,
        "grants": 1,
        "completed": 0,
        "busy_ranks": (0, 1),
    }


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


def test_unready_activation_head_does_not_block_disjoint_replica() -> None:
    coordinator = _coordinator((0, 1), (2, 3))
    unready = _offer("batch-a", src_rank=0, dst_rank=1)
    ready = _offer("batch-b", src_rank=2, dst_rank=3)
    coordinator.offer(unready)
    coordinator.offer(ready)
    coordinator.mark_receive_ready(ready.identity)

    assert coordinator.grant_ready()[0].offer is ready


def test_busy_replica_does_not_block_same_kind_on_disjoint_replica() -> None:
    coordinator = _coordinator((0, 1), (2, 3))
    busy = _offer("batch-a", src_rank=0, dst_rank=1)
    blocked_same_replica = _offer("batch-b", src_rank=0, dst_rank=1)
    disjoint = _offer("batch-c", src_rank=2, dst_rank=3)
    for offer in (busy, blocked_same_replica, disjoint):
        coordinator.offer(offer)
        coordinator.mark_receive_ready(offer.identity)

    assert coordinator.grant_ready()[0].offer is busy
    assert coordinator.grant_ready()[0].offer is disjoint


def test_second_offer_on_same_directed_edge_cannot_overtake() -> None:
    coordinator = _coordinator()
    first = _offer("batch-a")
    second = _offer("batch-b")
    coordinator.offer(first)
    coordinator.offer(second)
    coordinator.mark_receive_ready(second.identity)

    assert coordinator.grant_ready() == []


def test_feedback_can_progress_when_activation_edge_is_blocked() -> None:
    coordinator = _coordinator((0, 1), (2, 3))
    activation = _offer("batch-a")
    feedback = _offer(
        "batch-b",
        edge_kind=PipelineEdgeKind.FEEDBACK,
        src_rank=3,
        dst_rank=2,
    )
    coordinator.offer(activation)
    coordinator.offer(feedback)
    coordinator.mark_receive_ready(feedback.identity)

    assert coordinator.grant_ready()[0].offer is feedback


def test_disjoint_endpoints_can_receive_multiple_grants() -> None:
    coordinator = _coordinator((0, 1), (2, 3))
    first = _offer("batch-a", src_rank=0, dst_rank=1)
    second = _offer("batch-b", src_rank=2, dst_rank=3)
    for offer in (first, second):
        coordinator.offer(offer)
        coordinator.mark_receive_ready(offer.identity)

    assert [grant.offer for grant in coordinator.grant_ready(limit=2)] == [first, second]


def test_shared_endpoint_remains_busy_until_both_sides_complete() -> None:
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

    granted = coordinator.grant_ready()[0].offer
    blocked = second if granted is first else first
    assert coordinator.grant_ready() == []
    assert not coordinator.complete(granted.identity, granted.src_rank)
    assert coordinator.grant_ready() == []
    assert coordinator.complete(granted.identity, granted.dst_rank)
    assert coordinator.grant_ready()[0].offer is blocked


def test_transfer_rejects_reversed_activation_direction() -> None:
    coordinator = _coordinator()
    reversed_activation = _offer("batch-a", src_rank=1, dst_rank=0)

    with pytest.raises(ValueError, match="configured edge topology"):
        coordinator.offer(reversed_activation)


def test_transfer_rejects_cross_replica_edge() -> None:
    coordinator = _coordinator((0, 1), (2, 3))
    cross_replica = _offer("batch-a", src_rank=0, dst_rank=3)

    with pytest.raises(ValueError, match="configured edge topology"):
        coordinator.offer(cross_replica)


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
