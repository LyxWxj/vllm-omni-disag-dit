"""Low-overhead PP stage timeline tracing.

Tracing is disabled unless ``VLLM_OMNI_PP_TRACE_DIR`` is set.  The trace is
written as JSONL by each distributed process so it works with both CUDA and
Ascend NPU workers.  ``VLLM_OMNI_PP_TRACE_SYNC=1`` synchronizes the device
around stage spans; this makes the host interval closer to device activity at
the cost of perturbing the schedule and must not be used for throughput
benchmarks.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator


_TRACE_DIR = os.environ.get("VLLM_OMNI_PP_TRACE_DIR")
_SYNC = os.environ.get("VLLM_OMNI_PP_TRACE_SYNC", "0").lower() in {"1", "true", "yes"}
_LOCK = threading.Lock()
_HANDLE = None
_PROCESS_RANK: int | None = None
_PP_RANK: int | None = None
_EVENT_COUNTER = 0


def _rank_info() -> tuple[int, int, int]:
    global _PROCESS_RANK, _PP_RANK
    if _PROCESS_RANK is None:
        try:
            import torch.distributed as dist

            _PROCESS_RANK = dist.get_rank() if dist.is_initialized() else os.getpid()
        except Exception:
            _PROCESS_RANK = os.getpid()
    if _PP_RANK is None:
        try:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_pipeline_parallel_rank,
                get_pipeline_parallel_world_size,
            )

            _PP_RANK = int(get_pipeline_parallel_rank())
            pp_size = int(get_pipeline_parallel_world_size())
        except Exception:
            _PP_RANK = 0
            pp_size = 1
    else:
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size

            pp_size = int(get_pipeline_parallel_world_size())
        except Exception:
            pp_size = 1
    return int(_PROCESS_RANK), int(_PP_RANK), pp_size


def enabled() -> bool:
    """Return whether PP timeline tracing is enabled."""
    return bool(_TRACE_DIR)


def _device_sync() -> None:
    if not _SYNC:
        return
    try:
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
    except Exception:
        # Tracing must never break inference if a platform's optional sync API
        # is unavailable during worker startup or teardown.
        return


def _write(event: dict[str, Any]) -> None:
    global _HANDLE, _EVENT_COUNTER
    if not enabled():
        return
    os.makedirs(_TRACE_DIR, exist_ok=True)
    process_rank, pp_rank, pp_size = _rank_info()
    event.update(
        {
            "pid": process_rank,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "ts_ns": time.perf_counter_ns(),
            "event_id": _EVENT_COUNTER,
        }
    )
    _EVENT_COUNTER += 1
    with _LOCK:
        if _HANDLE is None:
            path = os.path.join(_TRACE_DIR, f"pp_rank_{process_rank}.jsonl")
            _HANDLE = open(path, "a", encoding="utf-8")
        _HANDLE.write(json.dumps(event, separators=(",", ":")) + "\n")
        _HANDLE.flush()


@contextmanager
def span(name: str, **args: Any) -> Iterator[None]:
    """Record a begin/end interval around a PP operation."""
    if not enabled():
        yield
        return
    _device_sync()
    _write({"ph": "B", "name": name, "args": args})
    try:
        yield
    finally:
        _device_sync()
        _write({"ph": "E", "name": name, "args": args})


def close() -> None:
    """Flush and close the rank-local trace file."""
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.close()
            _HANDLE = None
