# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead JSONL tracing for retained-state diffusion PP clocks.

Tracing is disabled unless ``VLLM_OMNI_PP_TRACE_DIR`` is set.  The default
``SYNC=0`` path records host timestamps only; enabling ``SYNC=1`` is intended
for device-side diagnostics and deliberately synchronizes around each span.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_TRACE_DIR = os.environ.get("VLLM_OMNI_PP_TRACE_DIR")
_TRACE_SYNC = os.environ.get("VLLM_OMNI_PP_TRACE_SYNC", "0") == "1"
_LOCK = threading.Lock()
_EVENT_COUNTER = 0
_HANDLE = None


def enabled() -> bool:
    """Return whether PP trace output is configured for this process."""
    return bool(_TRACE_DIR)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _synchronize(device: Any) -> None:
    if not _TRACE_SYNC or device is None:
        return
    try:
        import torch

        device_type = getattr(device, "type", None)
        if device_type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize(device)
        elif device_type == "cuda":
            accelerator = getattr(torch, "accelerator", None)
            if accelerator is not None:
                accelerator.synchronize()
    except Exception:
        # Tracing must never make a request fail merely because a diagnostic
        # synchronization API is unavailable during worker teardown.
        return


def _write(event: str, name: str, *, pp_rank: int, pp_size: int, **fields: Any) -> None:
    global _EVENT_COUNTER, _HANDLE
    if not _TRACE_DIR:
        return
    trace_path = Path(_TRACE_DIR)
    trace_path.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if _HANDLE is None:
            _HANDLE = (trace_path / f"pp_rank_{pp_rank}.jsonl").open("a", encoding="utf-8")
        _EVENT_COUNTER += 1
        record = {
            "event_id": _EVENT_COUNTER,
            "pid": os.getpid(),
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "ts_ns": time.perf_counter_ns(),
            "event": event,
            "name": name,
        }
        record.update({key: _json_value(value) for key, value in fields.items()})
        _HANDLE.write(json.dumps(record, separators=(",", ":")) + "\n")
        _HANDLE.flush()


@contextmanager
def span(
    name: str,
    *,
    pp_rank: int,
    pp_size: int,
    device: Any = None,
    **fields: Any,
) -> Iterator[None]:
    """Record a begin/end interval with request and clock metadata."""
    if not _TRACE_DIR:
        yield
        return
    _synchronize(device)
    _write("begin", name, pp_rank=pp_rank, pp_size=pp_size, **fields)
    try:
        yield
    finally:
        _synchronize(device)
        _write("end", name, pp_rank=pp_rank, pp_size=pp_size, **fields)


def event(name: str, *, pp_rank: int, pp_size: int, device: Any = None, **fields: Any) -> None:
    """Record a point event without changing device scheduling by default."""
    if not _TRACE_DIR:
        return
    _synchronize(device)
    _write("instant", name, pp_rank=pp_rank, pp_size=pp_size, **fields)


def close() -> None:
    """Flush and close the process-local trace file."""
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.flush()
            _HANDLE.close()
            _HANDLE = None
