#!/usr/bin/env python3
"""Measure cross-rank NPU kernel overlap from CANN profiler CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

_COMPUTE_CORE_TYPES = frozenset({"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"})


def _rank_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"diffusion_rank(\d+)", part)
        if match:
            return match.group(1)
    raise ValueError(f"cannot determine diffusion rank from {path}")


def _load_compute_kernel_intervals(
    trace_root: Path,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, str]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sources: dict[str, str] = {}
    paths = sorted(trace_root.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if not paths:
        raise ValueError(f"no CANN kernel_details.csv files found below {trace_root}")

    for path in paths:
        rank = _rank_from_path(path)
        if rank in sources:
            raise ValueError(f"multiple kernel_details.csv files found for diffusion rank {rank}")
        sources[rank] = str(path)
        with path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row["Accelerator Core"].strip() not in _COMPUTE_CORE_TYPES:
                    continue
                start = float(row["Start Time(us)"].strip())
                duration = float(row["Duration(us)"].strip())
                if duration > 0:
                    intervals[rank].append((start, start + duration))
    if not intervals:
        raise ValueError("no positive-duration compute-core kernels found")
    return dict(intervals), sources


def _clip_intervals(
    intervals: list[tuple[float, float]],
    start_us: float,
    end_us: float,
) -> list[tuple[float, float]]:
    return [(max(start, start_us), min(end, end_us)) for start, end in intervals if end > start_us and start < end_us]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def _active_rank_durations(merged_by_rank: dict[str, list[tuple[float, float]]]) -> tuple[dict[int, float], int]:
    changes: dict[float, int] = defaultdict(int)
    for intervals in merged_by_rank.values():
        for start, end in intervals:
            changes[start] += 1
            changes[end] -= 1

    durations: dict[int, float] = defaultdict(float)
    active_ranks = 0
    previous: float | None = None
    maximum = 0
    for timestamp in sorted(changes):
        if previous is not None and active_ranks:
            durations[active_ranks] += timestamp - previous
        active_ranks += changes[timestamp]
        maximum = max(maximum, active_ranks)
        previous = timestamp
    return dict(durations), maximum


def summarize(
    trace_root: Path,
    *,
    window_start_us: float | None = None,
    window_end_us: float | None = None,
    git_revision: str | None = None,
) -> dict[str, object]:
    intervals_by_rank, sources = _load_compute_kernel_intervals(trace_root)
    if (window_start_us is None) != (window_end_us is None):
        raise ValueError("--window-start-us and --window-end-us must be specified together")
    window_mode = "explicit"
    if window_start_us is None:
        window_mode = "profiler_capture"
        window_start_us = min(start for intervals in intervals_by_rank.values() for start, _ in intervals)
        window_end_us = max(end for intervals in intervals_by_rank.values() for _, end in intervals)
    assert window_end_us is not None
    if window_start_us >= window_end_us:
        raise ValueError("kernel overlap window must have positive duration")

    clipped_by_rank = {
        rank: _clip_intervals(intervals, window_start_us, window_end_us)
        for rank, intervals in intervals_by_rank.items()
    }
    merged_by_rank = {rank: _merge_intervals(intervals) for rank, intervals in clipped_by_rank.items()}
    durations, maximum = _active_rank_durations(merged_by_rank)
    any_active_us = sum(durations.values())
    at_least_two_us = sum(duration for count, duration in durations.items() if count >= 2)

    return {
        "schema_version": 1,
        "git_revision": git_revision,
        "source": "CANN ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "selection": {
            "accelerator_core_types": sorted(_COMPUTE_CORE_TYPES),
            "interval_processing": "clip to window, then merge per rank before cross-rank sweep",
            "window": {
                "mode": window_mode,
                "start_us": window_start_us,
                "end_us": window_end_us,
            },
        },
        "source_files_per_rank": sources,
        "kernel_task_count_per_rank": {rank: len(intervals) for rank, intervals in sorted(clipped_by_rank.items())},
        "merged_kernel_active_us_per_rank": {
            rank: sum(end - start for start, end in intervals) for rank, intervals in sorted(merged_by_rank.items())
        },
        "active_rank_duration_us": {str(count): duration for count, duration in sorted(durations.items())},
        "any_rank_kernel_active_us": any_active_us,
        "at_least_2_ranks_active_us": at_least_two_us,
        "cross_rank_kernel_overlap_ratio": at_least_two_us / any_active_us if any_active_us else 0.0,
        "maximum_simultaneously_active_ranks": maximum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path, help="Directory containing diffusion_rank*/ CANN profiler output")
    parser.add_argument("--output", type=Path, help="Write the JSON summary here instead of stdout")
    parser.add_argument("--window-start-us", type=float, help="Inclusive CANN timestamp lower bound")
    parser.add_argument("--window-end-us", type=float, help="Exclusive CANN timestamp upper bound")
    parser.add_argument("--git-revision", help="Record the profiled source revision in the summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(
        args.trace_root,
        window_start_us=args.window_start_us,
        window_end_us=args.window_end_us,
        git_revision=args.git_revision,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
