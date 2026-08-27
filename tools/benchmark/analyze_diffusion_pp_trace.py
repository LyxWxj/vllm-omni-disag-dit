#!/usr/bin/env python3
"""Summarize retained-state diffusion PP JSONL traces.

The input directory contains one ``pp_rank_*.jsonl`` file per worker.  The
summary reports stage activity and cross-rank overlap using host timestamps;
it does not claim device-kernel overlap unless a separate NPU profiler trace
confirms it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _load_intervals(trace_dir: Path) -> list[dict[str, Any]]:
    pending: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
    intervals: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("pp_rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") not in {"begin", "end"}:
                continue
            key = (
                record.get("name"),
                record.get("token_id"),
                record.get("clock"),
                record.get("microbatch_id"),
                record.get("cfg_branch"),
            )
            if record["event"] == "begin":
                pending[key].append(record)
                continue
            if not pending[key]:
                continue
            begin = pending[key].popleft()
            intervals.append(
                {
                    **{
                        field: begin.get(field)
                        for field in (
                            "name",
                            "pp_rank",
                            "pp_size",
                            "clock",
                            "token_id",
                            "request_ids",
                            "microbatch_id",
                            "step_idx",
                            "cfg_branch",
                            "model_phase",
                            "slot_id",
                        )
                    },
                    "start_ns": begin["ts_ns"],
                    "end_ns": record["ts_ns"],
                }
            )
    return intervals


def _merged_intervals(intervals: list[dict[str, Any]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for interval in sorted(intervals, key=lambda item: (item["start_ns"], item["end_ns"])):
        start, stop = interval["start_ns"], interval["end_ns"]
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        elif stop > merged[-1][1]:
            merged[-1] = (merged[-1][0], stop)
    return merged


def _union_ns(intervals: list[dict[str, Any]]) -> int:
    return sum(stop - start for start, stop in _merged_intervals(intervals))


def _overlap_ns(intervals: list[dict[str, Any]]) -> int:
    by_rank: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        by_rank[interval["pp_rank"]].append(interval)

    boundaries: list[tuple[int, int]] = []
    for rank_intervals in by_rank.values():
        for start, stop in _merged_intervals(rank_intervals):
            boundaries.append((start, 1))
            boundaries.append((stop, -1))
    active = 0
    overlap = 0
    previous = None
    for timestamp, delta in sorted(boundaries):
        if previous is not None and active >= 2:
            overlap += timestamp - previous
        active += delta
        previous = timestamp
    return overlap


def summarize(trace_dir: Path) -> dict[str, Any]:
    intervals = _load_intervals(trace_dir)
    stage_intervals = [interval for interval in intervals if interval["name"] == "stage_forward"]
    stage_by_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_by_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        all_by_rank[str(interval["pp_rank"])].append(interval)
    for interval in stage_intervals:
        stage_by_rank[str(interval["pp_rank"])].append(interval)
    active_ns = _union_ns(stage_intervals)
    overlap_ns = _overlap_ns(stage_intervals)
    all_span_active_ns = _union_ns(intervals)
    all_span_overlap_ns = _overlap_ns(intervals)
    return {
        "trace_dir": str(trace_dir),
        "interval_count": len(intervals),
        "ranks": sorted(all_by_rank),
        "stage_forward_intervals": len(stage_intervals),
        "active_stage_ms": active_ns / 1e6,
        "multi_stage_overlap_ms": overlap_ns / 1e6,
        "overlap_ratio": overlap_ns / active_ns if active_ns else 0.0,
        "per_rank": {
            rank: {
                "interval_count": len(items),
                "active_ms": _union_ns(items) / 1e6,
            }
            for rank, items in sorted(stage_by_rank.items())
        },
        "all_span_active_ms": all_span_active_ns / 1e6,
        "all_span_overlap_ms": all_span_overlap_ns / 1e6,
        "all_span_overlap_ratio": all_span_overlap_ns / all_span_active_ns if all_span_active_ns else 0.0,
        "all_span_per_rank": {
            rank: {
                "interval_count": len(items),
                "active_ms": _union_ns(items) / 1e6,
            }
            for rank, items in sorted(all_by_rank.items())
        },
        "intervals": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(args.trace_dir)
    output_dir = args.output_dir or args.trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    chrome_events = []
    for interval in summary["intervals"]:
        args_fields = {key: value for key, value in interval.items() if key not in {"name", "start_ns", "end_ns"}}
        chrome_events.extend(
            [
                {
                    "name": interval["name"],
                    "cat": "diffusion_pp",
                    "ph": "B",
                    "ts": interval["start_ns"] / 1000,
                    "pid": interval["pp_rank"],
                    "tid": interval["pp_rank"],
                    "args": args_fields,
                },
                {
                    "name": interval["name"],
                    "cat": "diffusion_pp",
                    "ph": "E",
                    "ts": interval["end_ns"] / 1000,
                    "pid": interval["pp_rank"],
                    "tid": interval["pp_rank"],
                    "args": args_fields,
                },
            ]
        )
    (output_dir / "timeline.json").write_text(json.dumps(chrome_events, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "intervals"}, indent=2))


if __name__ == "__main__":
    main()
