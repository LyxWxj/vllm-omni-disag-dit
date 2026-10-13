# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import concurrent.futures
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, BatchRunnerOutput


class _QueuedPipelineBatchPhase(str, Enum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    AUTHORIZED = "authorized"
    STEP_COMPLETED = "step_completed"
    STEP_COMMITTED = "step_committed"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    FAILED = "failed"


class _QueuedAdmissionDeferredError(RuntimeError):
    """The scheduler selected a request before queued capacity was available."""


@dataclass
class _QueuedPipelineBatch:
    batch_id: str
    request_id: str
    step_index: int
    scheduler_output: Any
    finalizing: bool = False
    decoded_output: BatchRunnerOutput | None = None
    scheduler_completed: bool = False
    cancelled: bool = False
    failure: BaseException | None = None
    abort_requested: bool = False
    request_prepared: bool = False
    request_cleanup_completed: bool = False
    step_future: concurrent.futures.Future[BaseRunnerOutput] | None = None
    decode_future: concurrent.futures.Future[BaseRunnerOutput] | None = None
    output_rank: int = 0
    native_output: BaseRunnerOutput | None = None
    phase: _QueuedPipelineBatchPhase = _QueuedPipelineBatchPhase.RESERVED


@dataclass
class _QueuedDecodeRound:
    request_ids: tuple[str, ...]
    ready_futures: list[concurrent.futures.Future] = dataclasses.field(default_factory=list)
    dispatched: bool = False
