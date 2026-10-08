# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-owned state and finalization policy for queued pipeline execution.

The Engine still owns request scheduling and executor calls.  This module only
owns the queued batch record and the policy that decides when a finalization is
safe and which physical rank receives a local VAE result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm_omni.diffusion.worker.pipeline_state import (
    PipelineEventType,
    PipelineStageSpec,
    PipelineTask,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput


class QueuedPipelineBatchPhase(str, Enum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    ADMISSION_PENDING = "admission_pending"
    SUBMITTED = "submitted"
    AUTHORIZED = "authorized"
    STEP_COMPLETED = "step_completed"
    STEP_COMMITTED = "step_committed"
    FINALIZATION_PENDING = "finalization_pending"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    FAILED = "failed"


@dataclass
class QueuedPipelineBatch:
    """Engine ownership record for one queued pipeline task."""

    task: PipelineTask
    scheduler_output: Any
    stage_specs: dict[int, PipelineStageSpec]
    stage_physical_ranks: dict[int, int]
    finalizing_request_ids: frozenset[str] = frozenset()
    decoded_output: BatchRunnerOutput | None = None
    finalization_handle: str | None = None
    finalization_output_rank: int | None = None
    release_acknowledged: bool = False
    transfer_retired: bool = False
    cleanup_completed_request_ids: set[str] = field(default_factory=set)
    scheduler_completed_request_ids: set[str] = field(default_factory=set)
    cancelled: bool = False
    failure: BaseException | None = None
    abort_requested: bool = False
    request_prepared: bool = False
    stage_enqueued: bool = False
    request_cleanup_completed: bool = False
    reserved_bytes: int = 0
    admission_acknowledgements: set[tuple[PipelineEventType, int, int]] = field(default_factory=set)
    phase: QueuedPipelineBatchPhase = QueuedPipelineBatchPhase.RESERVED


def distributed_vae_finalization_is_quiescent(
    batch: QueuedPipelineBatch,
    batches: Iterable[QueuedPipelineBatch],
    *,
    enabled: bool,
) -> bool:
    """Return whether ``batch`` may issue a distributed VAE collective."""
    if not enabled:
        return True
    retained = tuple(batches)
    if any(candidate is not batch and candidate.phase is QueuedPipelineBatchPhase.FINALIZING for candidate in retained):
        return False
    return all(
        candidate is batch
        or candidate.phase
        in {
            QueuedPipelineBatchPhase.RESERVED,
            QueuedPipelineBatchPhase.PREPARED,
            QueuedPipelineBatchPhase.FINALIZATION_PENDING,
        }
        for candidate in retained
    )


def select_output_owner_rank(
    batch: QueuedPipelineBatch,
    batches: Iterable[QueuedPipelineBatch],
    *,
    distributed_vae: bool,
    last_rank: int | None,
) -> int:
    """Select a physical rank for final output while balancing local VAE work."""
    if distributed_vae:
        return batch.stage_physical_ranks[0]

    ranks = [batch.stage_physical_ranks[stage_id] for stage_id in (0, 1)]
    outstanding = dict.fromkeys(ranks, 0)
    for candidate in batches:
        if (
            candidate is batch
            or candidate.phase is not QueuedPipelineBatchPhase.FINALIZING
            or candidate.finalization_handle is None
            or candidate.decoded_output is not None
            or candidate.finalization_output_rank not in outstanding
        ):
            continue
        outstanding[candidate.finalization_output_rank] += 1

    minimum = min(outstanding.values())
    if last_rank in ranks:
        offset = (ranks.index(last_rank) + 1) % len(ranks)
        ranks = ranks[offset:] + ranks[:offset]
    return next(rank for rank in ranks if outstanding[rank] == minimum)
