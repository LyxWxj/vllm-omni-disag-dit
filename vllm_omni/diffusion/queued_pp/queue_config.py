# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueuedPipelineConfig:
    """Resolved settings used only by the queued diffusion runtime."""

    enabled: bool
    max_inflight_batches: int

    @classmethod
    def from_config(cls, config: Any) -> QueuedPipelineConfig:
        mode = getattr(config, "mode", "static")
        if mode == "static":
            return cls(enabled=False, max_inflight_batches=1)
        if mode != "queued":
            raise ValueError(f"mode must be 'static' or 'queued', got {mode!r}")
        if not getattr(config, "step_execution", False):
            raise ValueError("mode='queued' requires step_execution=True")
        max_inflight_batches = getattr(config, "max_inflight_batches", 1)
        if type(max_inflight_batches) is not int or max_inflight_batches <= 0:
            raise ValueError(f"max_inflight_batches must be a positive integer, got {max_inflight_batches!r}")
        parallel_config = getattr(config, "parallel_config", None)
        pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1))
        if pp_size <= 1:
            raise ValueError("mode='queued' requires pipeline_parallel_size greater than 1")
        return cls(enabled=True, max_inflight_batches=max_inflight_batches)
