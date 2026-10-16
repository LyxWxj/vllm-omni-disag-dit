# SPDX-License-Identifier: Apache-2.0
"""Optional worker-local trace records for request-mode PP."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def trace_request_pipeline_event(kind: str, **fields: Any) -> None:
    """Append one rank-local event when request-pipeline tracing is enabled."""
    # The queued-PP-compatible renderer consumes only span records emitted by
    # the shared pipeline hooks.  Keep the request-mode lifecycle events
    # available for the native request-pipeline trace while allowing a strict
    # compatibility run to share the same worker files without extra records.
    if os.environ.get("VLLM_OMNI_PP_TRACE_FORMAT") == "queued_pp":
        return
    trace_dir = os.environ.get("VLLM_OMNI_PP_TRACE_DIR")
    if not trace_dir:
        return
    try:
        import torch

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    payload = {
        "timestamp_ns": time.monotonic_ns(),
        "physical_rank": rank,
        "kind": kind,
        **fields,
    }
    try:
        path = Path(trace_dir) / f"worker_rank{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
    except Exception:
        # Diagnostics must never change request correctness or collective order.
        return
