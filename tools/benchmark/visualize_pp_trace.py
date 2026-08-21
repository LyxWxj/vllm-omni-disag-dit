#!/usr/bin/env python3
"""Visualize PP stage traces produced by the diffusion timeline tracer.

The trace format currently records stage and CFG branch, but not a request ID.
For the benchmark produced by ``run_diffusion_pp_timeline.sh`` this script
infers request IDs from the execution order: span 0 is warmup, followed by
eight spans per request (four denoise steps and two CFG branches per step).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Span:
    rank: int
    start_ns: int
    end_ns: int
    branch: int
    request: int | None
    step: int | None

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


def load_spans(trace_dir: Path) -> list[Span]:
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(trace_dir.glob("pp_rank_*.jsonl")):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        events = [event for event in events if event.get("name") == "pp_stage_forward"]
        if len(events) % 2:
            raise ValueError(f"unpaired events in {path}: {len(events)} records")
        rank = int(path.stem.rsplit("_", 1)[-1])
        by_rank[rank] = events

    if not by_rank:
        raise ValueError(f"no pp_rank_*.jsonl files under {trace_dir}")
    counts = {rank: len(events) // 2 for rank, events in by_rank.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"ranks have different span counts: {counts}")

    spans: list[Span] = []
    for rank, events in sorted(by_rank.items()):
        for index in range(0, len(events), 2):
            begin, end = events[index], events[index + 1]
            if begin.get("ph") != "B" or end.get("ph") != "E":
                raise ValueError(f"expected B/E pairs in {trace_dir}/pp_rank_{rank}.jsonl at {index}")
            start_ns, end_ns = int(begin["ts_ns"]), int(end["ts_ns"])
            if end_ns <= start_ns:
                raise ValueError(f"non-positive span in rank {rank} at pair {index // 2}")
            branch = int(begin.get("args", {}).get("cfg_branch", 0))
            if index == 0:
                request, step = None, None
            else:
                formal_index = index // 2 - 1
                request = formal_index // 8
                step = (formal_index % 8) // 2 + 1
            spans.append(Span(rank, start_ns, end_ns, branch, request, step))
    return spans


def summarize(spans: list[Span]) -> dict[str, Any]:
    origin = min(span.start_ns for span in spans)
    formal = [span for span in spans if span.request is not None]
    formal_origin = min(span.start_ns for span in formal)
    requests: dict[str, dict[str, float]] = {}
    for request in sorted({span.request for span in formal}):
        request_spans = [span for span in formal if span.request == request]
        start, end = min(s.start_ns for s in request_spans), max(s.end_ns for s in request_spans)
        active = sum(s.end_ns - s.start_ns for s in request_spans)
        requests[str(request)] = {
            "start_ms_from_trace_origin": (start - origin) / 1e6,
            "duration_ms": (end - start) / 1e6,
            "stage_active_ms": active / 1e6,
            "single_stage_ratio": active / (end - start),
        }
    all_start, all_end = min(s.start_ns for s in spans), max(s.end_ns for s in spans)
    active = sum(s.end_ns - s.start_ns for s in spans)
    return {
        "trace_dir": str(spans[0].__class__),
        "span_count": len(spans),
        "rank_count": len({span.rank for span in spans}),
        "trace_makespan_ms": (all_end - all_start) / 1e6,
        "stage_active_ms": active / 1e6,
        "stage_active_ratio": active / (all_end - all_start),
        "formal_request_count": len(requests),
        "formal_makespan_ms": (max(s.end_ns for s in formal) - formal_origin) / 1e6,
        "requests": requests,
        "spans": [
            {
                **asdict(span),
                "start_ms_from_trace_origin": (span.start_ns - origin) / 1e6,
                "duration_ms": span.duration_ms,
            }
            for span in spans
        ],
    }


def draw_dataset(ax: Any, spans: list[Span], label: str, formal_only: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    selected = [span for span in spans if not formal_only or span.request is not None]
    origin = min(span.start_ns for span in selected)
    requests = sorted({span.request for span in selected if span.request is not None})
    palette = plt.get_cmap("tab10")
    for span in selected:
        request_color = "#8c8c8c" if span.request is None else palette(span.request % 10)
        start_s = (span.start_ns - origin) / 1e9
        duration_s = (span.end_ns - span.start_ns) / 1e9
        ax.barh(
            span.rank,
            duration_s,
            left=start_s,
            height=0.68,
            color=request_color,
            edgecolor="#202124",
            linewidth=0.25,
            alpha=0.9,
            hatch="///" if span.branch else None,
        )

    ax.set_title(label + (" - formal requests" if formal_only else " - full trace"), loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("PP rank")
    ax.set_yticks(sorted({span.rank for span in selected}))
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor="#8c8c8c", label="warmup")]
    handles.extend(Patch(facecolor=palette(request % 10), label=f"request {request}") for request in requests)
    handles.extend(
        [
            Patch(facecolor="white", edgecolor="#202124", label="CFG branch 0"),
            Patch(facecolor="white", edgecolor="#202124", hatch="///", label="CFG branch 1"),
        ]
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=4, frameon=False, fontsize=8)


def render(trace_dirs: list[Path], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = [(label, load_spans(path)) for label, path in zip(labels, trace_dirs)]
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (label, spans), trace_dir in zip(datasets, trace_dirs):
        summary = summarize(spans)
        summary["label"] = label
        summary["trace_dir"] = str(trace_dir)
        summaries.append(summary)

    for formal_only, filename in [(False, "pp_trace_full.png"), (True, "pp_trace_formal.png")]:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 4.8 * len(datasets)), squeeze=False)
        for ax, (label, spans) in zip(axes[:, 0], datasets):
            draw_dataset(ax, spans, label, formal_only)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"full_timeline={output_dir / 'pp_trace_full.png'}")
    print(f"formal_timeline={output_dir / 'pp_trace_formal.png'}")
    print(f"summary={output_dir / 'summary.json'}")
    for summary in summaries:
        print(
            f"{summary['label']}: spans={summary['span_count']} requests={summary['formal_request_count']} "
            f"formal_makespan_ms={summary['formal_makespan_ms']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", type=Path, help="Directories containing trace/pp_rank_*.jsonl")
    parser.add_argument("--labels", nargs="+", help="Labels corresponding to trace directories")
    parser.add_argument("--output-dir", type=Path, default=Path("pp-trace-visualization"))
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.trace_dirs):
        parser.error("--labels must have one label per trace directory")
    labels = args.labels or [path.name for path in args.trace_dirs]
    render(args.trace_dirs, labels, args.output_dir)


if __name__ == "__main__":
    main()
