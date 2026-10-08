# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared factories for queued pipeline contract tests."""

from collections.abc import Iterable

from vllm_omni.diffusion.distributed.pipeline_stage_connector import (
    PipelineEdgeKind,
    PipelineTransferCoordinator,
    PipelineTransferOffer,
)
from vllm_omni.diffusion.worker.pipeline_state import PipelineStageSpec, PipelineTask


def pipeline_coordinator(
    activation_edges: Iterable[tuple[int, int]] | None = None,
) -> PipelineTransferCoordinator:
    """Build the single-replica activation/feedback topology used by tests."""
    activation_edges = {(0, 1)} if activation_edges is None else set(activation_edges)
    feedback_edges = {(dst, src) for src, dst in activation_edges}
    return PipelineTransferCoordinator(
        activation_edges=activation_edges,
        feedback_edges=feedback_edges,
    )


def pipeline_offer(
    batch_id: str,
    *,
    step_index: int = 0,
    epoch: int = 1,
    branch: str = "conditional",
    edge_kind: PipelineEdgeKind = PipelineEdgeKind.ACTIVATION,
    src_rank: int = 0,
    dst_rank: int = 1,
    payload_metadata: tuple[tuple[str, object], ...] = (),
) -> PipelineTransferOffer:
    """Build a metadata-only transfer offer with explicit identity defaults."""
    return PipelineTransferOffer(
        batch_id=batch_id,
        step_index=step_index,
        epoch=epoch,
        branch=branch,
        edge_kind=edge_kind,
        src_rank=src_rank,
        dst_rank=dst_rank,
        payload_metadata=payload_metadata,
    )


def pipeline_task(
    batch_id: str = "batch-a",
    *,
    request_id: str = "req-a",
    request_ids: tuple[str, ...] | None = None,
    step_index: int = 0,
    epoch: int = 2,
    branch: str = "conditional",
) -> PipelineTask:
    """Build a queued task while keeping request and transfer identity aligned."""
    return PipelineTask(
        batch_id=batch_id,
        request_ids=(request_id,) if request_ids is None else request_ids,
        step_index=step_index,
        epoch=epoch,
        branch=branch,
    )


def pipeline_stage_spec(stage_id: int, *, world_size: int = 2) -> PipelineStageSpec:
    """Build one logical stage in the M2 two-stage topology."""
    return PipelineStageSpec(
        pp_stage_id=stage_id,
        world_size=world_size,
        is_first=stage_id == 0,
        is_last=stage_id == world_size - 1,
    )
