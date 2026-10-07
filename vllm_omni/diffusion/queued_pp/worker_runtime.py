# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-owned lifecycle state for queued pipeline finalization."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.distributed.pipeline_stage_connector import PipelineEdgeKind, PipelineMessage


@dataclass
class PipelineTransportState:
    """Worker-owned transport reservations and consumer leases."""

    send_tickets: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    receive_reservations: dict[tuple[Any, ...], PipelineEdgeKind] = field(default_factory=dict)
    started_receive_ids: set[tuple[Any, ...]] = field(default_factory=set)
    receive_consumers: dict[tuple[Any, ...], tuple[PipelineEdgeKind, PipelineMessage, Any | None]] = field(
        default_factory=dict
    )
    pending_received: dict[PipelineEdgeKind, deque[PipelineMessage]] = field(
        default_factory=lambda: {
            PipelineEdgeKind.ACTIVATION: deque(),
            PipelineEdgeKind.FEEDBACK: deque(),
        }
    )


@dataclass
class PipelineFinalizationState:
    """Own finalization futures and device-completion metadata for one Worker."""

    futures: dict[str, Future[Any]] = field(default_factory=dict)
    executor: ThreadPoolExecutor | None = None
    published: set[str] = field(default_factory=set)
    device_events: dict[str, Any] = field(default_factory=dict)
    stream: Any | None = None

    def ensure_executor(self, rank: int) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"WanFinalDecode-rank{rank}",
            )
        return self.executor

    def clear_batch(self, batch_id: str) -> None:
        self.futures.pop(batch_id, None)
        self.published.discard(batch_id)
        self.device_events.pop(batch_id, None)

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None
