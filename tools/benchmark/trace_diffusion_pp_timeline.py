#!/usr/bin/env python3
"""Render PP stage activity traces produced by ``pp_trace``.

Start the server/worker with::

    VLLM_OMNI_PP_TRACE_DIR=/tmp/pp-trace \\
    VLLM_OMNI_PP_TRACE_SYNC=1 \\
    <normal vLLM-Omni server command>

Send one or more requests, stop the server cleanly, then run::

    python tools/benchmark/trace_diffusion_pp_timeline.py \\
        --trace-dir /tmp/pp-trace --output /tmp/pp-timeline

The script writes ``timeline.json`` (Chrome Trace format) and
``timeline.txt`` (human-readable bins and overlap statistics).  Sync tracing
is useful for proving stage ordering, but it inserts device synchronizations
and must not be used for throughput numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("pp_rank_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                if event.get("ph") in {"B", "E"} and "ts_ns" in event:
                    events.append(event)
    if not events:
        raise ValueError(f"no PP events found under {trace_dir}")
    return events


def _intervals(events: list[dict[str, Any]], name: str = "pp_stage_forward") -> list[dict[str, Any]]:
    open_events: dict[tuple[int, str, int], dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: int(item["ts_ns"])):
        if event.get("name") != name:
            continue
        key = (int(event["pid"]), str(event.get("name")), int(event.get("event_id", -1)))
        if event["ph"] == "B":
            # The end event gets a different event id. Match by rank/name and
            # preserve nesting order because the PP path is sequential per rank.
            open_events[(int(event["pid"]), str(event.get("name")), 0)] = event
        else:
            pending = open_events.pop((int(event["pid"]), str(event.get("name")), 0), None)
            if pending is None:
                continue
            if int(event["ts_ns"]) <= int(pending["ts_ns"]):
                continue
            result.append(
                {
                    "start_ns": int(pending["ts_ns"]),
                    "end_ns": int(event["ts_ns"]),
                    "pid": int(pending["pid"]),
                    "pp_rank": int(pending.get("pp_rank", 0)),
                    "args": pending.get("args", {}),
                }
            )
    return result


def _activity_stats(intervals: list[dict[str, Any]]) -> tuple[float, float, float]:
    points: list[tuple[int, int, int]] = []
    for interval in intervals:
        points.append((interval["start_ns"], 1, interval["pp_rank"]))
        points.append((interval["end_ns"], -1, interval["pp_rank"]))
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    points.sort(key=lambda item: (item[0], item[1]))
    active: set[int] = set()
    one_stage = 0
    any_stage = 0
    overlap = 0
    previous = points[0][0]
    for timestamp, delta, rank in points:
        duration = max(0, timestamp - previous)
        if active:
            any_stage += duration
            if len(active) == 1:
                one_stage += duration
            elif len(active) > 1:
                overlap += duration
        if delta > 0:
            active.add(rank)
        else:
            active.discard(rank)
        previous = timestamp
    return one_stage / 1e6, overlap / 1e6, any_stage / 1e6


def _render_text(intervals: list[dict[str, Any]], bin_us: int) -> str:
    origin = min(item["start_ns"] for item in intervals)
    end = max(item["end_ns"] for item in intervals)
    bin_ns = max(1, bin_us) * 1000
    lines = ["# PP stage activity timeline", "# time_ms active_pp_stages active_count"]
    for start in range(origin, end, bin_ns):
        stop = min(start + bin_ns, end)
        active = sorted(
            {
                item["pp_rank"]
                for item in intervals
                if item["start_ns"] < stop and item["end_ns"] > start
            }
        )
        if active:
            lines.append(f"{(start - origin) / 1e6:12.3f} {','.join(map(str, active)):>16} {len(active):12d}")
    one_stage_ms, overlap_ms, any_stage_ms = _activity_stats(intervals)
    ratio = one_stage_ms / any_stage_ms if any_stage_ms else math.nan
    lines.extend(
        [
            "",
            f"single_stage_only_ms={one_stage_ms:.3f}",
            f"overlap_ms={overlap_ms:.3f}",
            f"any_stage_active_ms={any_stage_ms:.3f}",
            f"single_stage_only_ratio={ratio:.6f}",
        ]
    )
    return "\n".join(lines) + "\n"


def render(trace_dir: Path, output_prefix: Path, bin_us: int) -> None:
    events = _load_events(trace_dir)
    intervals = _intervals(events)
    if not intervals:
        raise ValueError("no complete pp_stage_forward intervals found")
    origin = min(item["start_ns"] for item in intervals)
    chrome: list[dict[str, Any]] = []
    for item in events:
        chrome.append(
            {
                "name": item.get("name", "pp"),
                "cat": "vllm_omni.pp",
                "ph": item["ph"],
                "ts": (int(item["ts_ns"]) - origin) / 1000.0,
                "pid": item.get("pid", 0),
                "tid": item.get("pp_rank", 0),
                "args": item.get("args", {}),
            }
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps({"traceEvents": chrome}, indent=2), encoding="utf-8")
    output_prefix.with_suffix(".txt").write_text(_render_text(intervals, bin_us), encoding="utf-8")
    one_stage_ms, overlap_ms, any_stage_ms = _activity_stats(intervals)
    ratio = one_stage_ms / any_stage_ms if any_stage_ms else math.nan
    print(f"events={len(events)} intervals={len(intervals)}")
    print(f"single_stage_only_ms={one_stage_ms:.3f}")
    print(f"overlap_ms={overlap_ms:.3f}")
    print(f"any_stage_active_ms={any_stage_ms:.3f}")
    print(f"single_stage_only_ratio={ratio:.6f}")
    print(f"chrome_trace={output_prefix.with_suffix('.json')}")
    print(f"text_timeline={output_prefix.with_suffix('.txt')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix, without .json/.txt")
    parser.add_argument("--bin-us", type=int, default=100, help="Text timeline bin width in microseconds")
    args = parser.parse_args()
    render(args.trace_dir, args.output, args.bin_us)


if __name__ == "__main__":
    main()
