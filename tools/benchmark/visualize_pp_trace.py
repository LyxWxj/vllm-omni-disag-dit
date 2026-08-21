#!/usr/bin/env python3
"""Visualize component-level diffusion PP traces.

The current trace format records request IDs, batch size, denoising step,
CFG branch, component name, and PP rank. This tool uses those fields directly
and never treats one DIT span as one request. Traces produced before that
metadata was added are still accepted, but are reported as ``legacy_trace``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interval:
    name: str
    start_ns: int
    end_ns: int
    pid: int
    pp_rank: int
    args: dict[str, Any]

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def component(self) -> str:
        return str(self.args.get("component") or self.name)

    @property
    def request_ids(self) -> tuple[str, ...]:
        value = self.args.get("request_ids", ())
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ()

    @property
    def step_idx(self) -> int | None:
        value = self.args.get("step_idx")
        return int(value) if isinstance(value, int) else None


_PARENT_NAMES = {
    "pipeline_forward",
    "pipeline_forward_batch",
    "pipeline_memory_profile",
}
_COMPONENT_COLORS = {
    "text_encoder": "#4c78a8",
    "vae_encode": "#59a14f",
    "dit": "#f28e2b",
    "scheduler_step": "#e15759",
    "pp_send_wait": "#b279a2",
    "cache_cleanup": "#9d9d9d",
    "vae_decode_prepare": "#76b7b2",
    "vae_decode": "#edc948",
    "vae_decode_broadcast": "#af7aa1",
    "vae_decode_skip": "#bab0ab",
}


def _load_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("pp_rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    if not events:
        raise ValueError(f"no trace events under {trace_dir}")
    return sorted(events, key=lambda event: int(event["ts_ns"]))


def _pair_intervals(events: list[dict[str, Any]]) -> list[Interval]:
    """Pair B/E events by process and name, supporting nested spans."""
    stacks: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    intervals: list[Interval] = []
    for event in events:
        name = str(event.get("name", "unknown"))
        key = (int(event.get("pid", 0)), name)
        if event.get("ph") == "B":
            stacks[key].append(event)
            continue
        if event.get("ph") != "E" or not stacks[key]:
            continue
        begin = stacks[key].pop()
        start_ns = int(begin["ts_ns"])
        end_ns = int(event["ts_ns"])
        if end_ns <= start_ns:
            continue
        intervals.append(
            Interval(
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                pid=int(begin.get("pid", 0)),
                pp_rank=int(begin.get("pp_rank", 0)),
                args=dict(begin.get("args") or {}),
            )
        )
    if not intervals:
        raise ValueError(f"no complete B/E spans under {len(events)} events")
    return sorted(intervals, key=lambda item: item.start_ns)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _activity_stats(intervals: list[Interval]) -> dict[str, float]:
    points: list[tuple[int, int, int]] = []
    for interval in intervals:
        points.extend(((interval.start_ns, 1, interval.pp_rank), (interval.end_ns, -1, interval.pp_rank)))
    if len(points) < 2:
        return {"single_stage_only_ms": 0.0, "overlap_ms": 0.0, "any_stage_active_ms": 0.0}
    points.sort(key=lambda item: (item[0], item[1]))
    active: set[int] = set()
    one_stage = overlap = any_stage = 0
    previous = points[0][0]
    for timestamp, delta, rank in points:
        duration = max(0, timestamp - previous)
        if active:
            any_stage += duration
            if len(active) == 1:
                one_stage += duration
            else:
                overlap += duration
        if delta > 0:
            active.add(rank)
        else:
            active.discard(rank)
        previous = timestamp
    return {
        "single_stage_only_ms": one_stage / 1e6,
        "overlap_ms": overlap / 1e6,
        "any_stage_active_ms": any_stage / 1e6,
    }


def _gaps(intervals: list[Interval], origin_ns: int) -> list[dict[str, Any]]:
    """Find rank-local gaps between non-parent component spans."""
    by_rank: dict[int, list[Interval]] = defaultdict(list)
    for interval in intervals:
        if interval.name not in _PARENT_NAMES:
            by_rank[interval.pp_rank].append(interval)
    result: list[dict[str, Any]] = []
    for rank, rank_intervals in by_rank.items():
        rank_intervals.sort(key=lambda item: item.start_ns)
        for previous, current in zip(rank_intervals, rank_intervals[1:]):
            gap_ns = current.start_ns - previous.end_ns
            if gap_ns <= 1e6:
                continue
            result.append(
                {
                    "pp_rank": rank,
                    "start_ms": (current.start_ns - origin_ns) / 1e6,
                    "gap_ms": gap_ns / 1e6,
                    "previous": previous.name,
                    "next": current.name,
                }
            )
    return sorted(result, key=lambda item: item["start_ms"])


def _batch_summary(intervals: list[Interval], origin_ns: int) -> list[dict[str, Any]]:
    """Collapse one pipeline forward call recorded independently by each rank."""
    groups: list[dict[str, Any]] = []
    for item in sorted((item for item in intervals if item.name in _PARENT_NAMES), key=lambda x: x.start_ns):
        key = (item.name, item.request_ids, item.args.get("batch_size"))
        group = next(
            (
                candidate
                for candidate in reversed(groups)
                if candidate["key"] == key and item.start_ns <= candidate["end_ns"]
            ),
            None,
        )
        if group is None:
            groups.append(
                {
                    "key": key,
                    "name": item.name,
                    "request_ids": list(item.request_ids),
                    "batch_size": item.args.get("batch_size"),
                    "start_ns": item.start_ns,
                    "end_ns": item.end_ns,
                    "ranks": {item.pp_rank},
                }
            )
        else:
            group["start_ns"] = min(group["start_ns"], item.start_ns)
            group["end_ns"] = max(group["end_ns"], item.end_ns)
            group["ranks"].add(item.pp_rank)

    return [
        {
            "batch_index": index,
            "name": group["name"],
            "request_ids": group["request_ids"],
            "batch_size": group["batch_size"],
            "start_ms": (group["start_ns"] - origin_ns) / 1e6,
            "duration_ms": (group["end_ns"] - group["start_ns"]) / 1e6,
            "pp_ranks": sorted(group["ranks"]),
        }
        for index, group in enumerate(groups)
    ]


def _span_summary(intervals: list[Interval]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Interval]] = defaultdict(list)
    for item in intervals:
        grouped[(item.component, item.pp_rank)].append(item)
    return [
        {
            "component": component,
            "pp_rank": rank,
            "count": len(items),
            "total_ms": sum(item.duration_ns for item in items) / 1e6,
            "max_ms": max(item.duration_ns for item in items) / 1e6,
        }
        for (component, rank), items in sorted(grouped.items())
    ]


def _summary(trace_dir: Path, intervals: list[Interval]) -> dict[str, Any]:
    origin_ns = min(item.start_ns for item in intervals)
    has_request_metadata = any(item.request_ids or "batch_size" in item.args for item in intervals)
    dit = [item for item in intervals if item.name == "pp_stage_forward"]
    activity = _activity_stats(dit)
    active_ms = activity["any_stage_active_ms"]
    activity["single_stage_only_ratio"] = (
        activity["single_stage_only_ms"] / active_ms if active_ms else None
    )
    return {
        "schema_version": 2,
        "trace_dir": str(trace_dir),
        "legacy_trace": not has_request_metadata,
        "origin_ns": origin_ns,
        "batches": _batch_summary(intervals, origin_ns),
        "span_summary": _span_summary(intervals),
        "gaps_gt_1ms": _gaps(intervals, origin_ns),
        "activity_pp_stage_forward": activity,
        "intervals": [
            {
                "name": item.name,
                "component": item.component,
                "pp_rank": item.pp_rank,
                "pid": item.pid,
                "start_ms": (item.start_ns - origin_ns) / 1e6,
                "duration_ms": item.duration_ns / 1e6,
                "request_ids": list(item.request_ids),
                "batch_size": item.args.get("batch_size"),
                "step_idx": item.step_idx,
                "cfg_branch": item.args.get("cfg_branch"),
                "args": _json_safe(item.args),
            }
            for item in intervals
        ],
    }


def _draw(intervals: list[Interval], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ranks = sorted({item.pp_rank for item in intervals})
    origin_ns = min(item.start_ns for item in intervals)
    rank_y = {rank: index for index, rank in enumerate(ranks)}
    fig, ax = plt.subplots(figsize=(18, max(4.5, 1.1 * len(ranks) + 2.0)))

    parents = [item for item in intervals if item.name in _PARENT_NAMES]
    for item in parents:
        ax.barh(
            rank_y[item.pp_rank],
            item.duration_ns / 1e9,
            left=(item.start_ns - origin_ns) / 1e9,
            height=0.86,
            color="#d9d9d9",
            edgecolor="#777777",
            linewidth=0.5,
            alpha=0.28,
        )

    seen_components: set[str] = set()
    for item in intervals:
        if item.name in _PARENT_NAMES:
            continue
        component = item.component
        seen_components.add(component)
        ax.barh(
            rank_y[item.pp_rank],
            item.duration_ns / 1e9,
            left=(item.start_ns - origin_ns) / 1e9,
            height=0.58,
            color=_COMPONENT_COLORS.get(component, "#7f7f7f"),
            edgecolor="#333333",
            linewidth=0.25,
            alpha=0.92,
        )

    for item in parents:
        ax.axvline((item.start_ns - origin_ns) / 1e9, color="#555555", alpha=0.18, linewidth=0.6)

    handles = [Patch(facecolor="#d9d9d9", edgecolor="#777777", alpha=0.5, label="pipeline forward batch")]
    handles.extend(
        Patch(facecolor=_COMPONENT_COLORS.get(name, "#7f7f7f"), label=name)
        for name in sorted(seen_components)
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=5, frameon=False, fontsize=8)
    ax.set_yticks(list(rank_y.values()), [f"PP rank {rank}" for rank in ranks])
    ax.set_xlabel("time (s)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _draw_html(intervals: list[Interval], output_path: Path, title: str) -> None:
    """Write a dependency-free SVG timeline for headless benchmark hosts."""
    ranks = sorted({item.pp_rank for item in intervals})
    origin_ns = min(item.start_ns for item in intervals)
    end_ns = max(item.end_ns for item in intervals)
    span_ns = max(1, end_ns - origin_ns)
    left, top, width, row_height = 150, 42, 1500, 34
    height = top + row_height * max(1, len(ranks)) + 70
    rank_y = {rank: top + index * row_height for index, rank in enumerate(ranks)}
    elements = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px;color:#202124}"
        ".axis{stroke:#ddd;stroke-width:1}.bar{stroke:#333;stroke-width:.3}"
        ".label{font-size:13px}.legend{font-size:12px} </style>",
        f"<h2>{html.escape(title)}</h2>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{left + width + 20}' "
        f"height='{height}' viewBox='0 0 {left + width + 20} {height}'>",
    ]
    for rank, y in rank_y.items():
        elements.append(f"<text class='label' x='8' y='{y + 5}'>PP rank {rank}</text>")
        elements.append(
            f"<line class='axis' x1='{left}' y1='{y + row_height / 2}' "
            f"x2='{left + width}' y2='{y + row_height / 2}'/>"
        )

    for item in intervals:
        x = left + (item.start_ns - origin_ns) / span_ns * width
        w = max(1.0, item.duration_ns / span_ns * width)
        y = rank_y[item.pp_rank] + (2 if item.name in _PARENT_NAMES else 8)
        h = row_height - (4 if item.name in _PARENT_NAMES else 16)
        component = item.component
        color = "#d9d9d9" if item.name in _PARENT_NAMES else _COMPONENT_COLORS.get(component, "#7f7f7f")
        details = json.dumps(item.args, ensure_ascii=True, separators=(",", ":"))
        tooltip = html.escape(f"{item.name} {item.duration_ns / 1e6:.3f} ms {details}")
        elements.append(
            f"<rect class='bar' x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='{color}' opacity='0.9'>"
            f"<title>{tooltip}</title></rect>"
        )
    for index in range(5):
        x = left + index * width / 4
        ms = (span_ns / 1e6) * index / 4
        elements.append(f"<line class='axis' x1='{x:.2f}' y1='{top - 18}' x2='{x:.2f}' y2='{height - 45}'/>")
        elements.append(f"<text class='label' x='{x:.2f}' y='{top - 24}'>{ms:.1f} ms</text>")
    components = sorted({item.component for item in intervals if item.name not in _PARENT_NAMES})
    legend_x = left
    legend_y = height - 20
    elements.append(f"<text class='legend' x='8' y='{legend_y}'>Components:</text>")
    for component in components:
        color = _COMPONENT_COLORS.get(component, "#7f7f7f")
        elements.append(f"<rect x='{legend_x}' y='{legend_y - 11}' width='10' height='10' fill='{color}'/>")
        elements.append(f"<text class='legend' x='{legend_x + 14}' y='{legend_y - 1}'>{html.escape(component)}</text>")
        legend_x += 100 + len(component) * 3
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "trace"


def render(trace_dirs: list[Path], labels: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for trace_dir, label in zip(trace_dirs, labels):
        intervals = _pair_intervals(_load_events(trace_dir))
        summary = _summary(trace_dir, intervals)
        summary["label"] = label
        summaries.append(summary)
        suffix = _safe_label(label)
        html_path = output_dir / f"components_{suffix}.html"
        _draw_html(intervals, html_path, f"Diffusion component timeline: {label}")
        print(f"components_timeline_html={html_path}")
        figure_path = output_dir / f"components_{suffix}.png"
        try:
            _draw(intervals, figure_path, f"Diffusion component timeline: {label}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("components_timeline=unavailable (install matplotlib for PNG output)")
        else:
            print(f"components_timeline={figure_path}")
        summary_path = output_dir / f"summary_{suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(f"summary={summary_path}")
        print(f"legacy_trace={summary['legacy_trace']}")
        print(f"batch_spans={len(summary['batches'])}")
        print(f"component_spans={len(summary['intervals'])}")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", type=Path, help="Directories containing pp_rank_*.jsonl")
    parser.add_argument("--labels", nargs="+", help="Labels corresponding to trace directories")
    parser.add_argument("--output-dir", type=Path, default=Path("pp-trace-visualization"))
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.trace_dirs):
        parser.error("--labels must have one label per trace directory")
    labels = args.labels or [path.name for path in args.trace_dirs]
    render(args.trace_dirs, labels, args.output_dir)


if __name__ == "__main__":
    main()
