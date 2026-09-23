#!/usr/bin/env python3
"""Render Wan queued-PP timelines and latency comparisons from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

COLORS = {
    "stage0": "#1f77b4",
    "stage1": "#e67e22",
    "static": "#6c757d",
    "queued": "#167c80",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _trace_events(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["traceEvents"]


def _markers(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for trace_path in sorted((run_dir / "torch_profiler").rglob("trace_rank*.json")):
        rank = int(trace_path.stem.removeprefix("trace_rank"))
        for event in _trace_events(trace_path):
            if event.get("ph") != "X" or event.get("cat") != "user_annotation":
                continue
            name = event.get("name", "")
            if mode == "queued" and not name.startswith("queued_pp::stage"):
                continue
            if mode == "static" and name != "diffusion_step":
                continue
            markers.append(
                {
                    "rank": rank,
                    "name": name,
                    "start": float(event["ts"]),
                    "end": float(event["ts"] + event.get("dur", 0.0)),
                }
            )
    return markers


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_text(x: float, y: float, text: object, size: int = 13, **attrs: object) -> str:
    extra = " ".join(f'{key.replace("_", "-")}="{_esc(value)}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {extra}>{_esc(text)}</text>'


def render_timeline(queued_dir: Path, static_dir: Path, output: Path) -> dict[str, Any]:
    queued = _markers(queued_dir, "queued")
    static = _markers(static_dir, "static")
    all_markers = queued + static
    if not all_markers:
        raise RuntimeError("No profiler markers found")
    origin = min(item["start"] for item in all_markers)
    end = max(item["end"] for item in all_markers)
    max_ms = max((end - origin) / 1000.0, 1.0)
    width, left, right = 1900, 220, 1850
    row_height, row_gap = 58, 28
    height = 430
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        "<style>text{font-family:Arial,sans-serif}</style>",
        _svg_text(28, 34, "Wan2.2 PP2: queued inter-request pipeline vs main static PP", 24, fill="#172238"),
        _svg_text(
            28,
            60,
            "512x512, 16 frames, 8 denoise steps, 4 requests, max_num_seqs=2; profiler time is diagnostic only",
            14,
            fill="#526178",
        ),
    ]
    rows = [
        ("queued rank 0 / stage 0", queued, "stage0"),
        ("queued rank 1 / stage 1", queued, "stage1"),
        ("main rank 0", static, "static"),
        ("main rank 1", static, "static"),
    ]
    for row, (label, markers, color_key) in enumerate(rows):
        y = 88 + row * (row_height + row_gap)
        svg.append(_svg_text(28, y + 34, label, 14, fill="#172238", font_weight="600"))
        svg.append(
            f'<rect x="{left}" y="{y}" width="{right - left}" height="{row_height}" fill="#ffffff" stroke="#c7d0dd"/>'
        )
        for tick in range(0, math.ceil(max_ms / 500) * 500 + 1, 500):
            x = left + (right - left) * tick / max_ms
            svg.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + row_height}" stroke="#e1e6ee"/>')
            svg.append(_svg_text(x, y + row_height + 18, tick, 10, text_anchor="middle", fill="#526178"))
        selected = [item for item in markers if item["rank"] == (row % 2)]
        if color_key == "stage1":
            selected = [item for item in markers if item["rank"] == 1]
        elif color_key == "stage0":
            selected = [item for item in markers if item["rank"] == 0 and "stage0" in item["name"]]
        elif color_key == "static":
            selected = [item for item in markers if item["rank"] == (row % 2)]
        for marker in selected:
            start_ms = (marker["start"] - origin) / 1000.0
            duration_ms = (marker["end"] - marker["start"]) / 1000.0
            x = left + (right - left) * start_ms / max_ms
            bar_width = max((right - left) * duration_ms / max_ms, 2.0)
            svg.append(
                f'<rect x="{x:.1f}" y="{y + 10}" width="{bar_width:.1f}" height="38" '
                f'rx="3" fill="{COLORS[color_key]}" fill-opacity=".88"/>'
            )
            if bar_width > 105:
                short = marker["name"].split("::")[-1] if color_key != "static" else "diffusion_step"
                svg.append(_svg_text(x + 5, y + 34, short, 10, fill="#ffffff"))
    legend_y = height - 25
    for index, key in enumerate(("stage0", "stage1", "static")):
        x = 28 + index * 205
        svg.append(f'<rect x="{x}" y="{legend_y - 12}" width="15" height="15" fill="{COLORS[key]}"/>')
        labels = {"stage0": "queued stage 0", "stage1": "queued stage 1", "static": "main diffusion_step"}
        svg.append(_svg_text(x + 22, legend_y, labels[key], 12, fill="#526178"))
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")
    return {"queued_markers": len(queued), "main_markers": len(static), "timeline_ms": max_ms}


def _latency_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        manifest = _load_manifest(path)
        rows.extend(
            {
                "latency_ms": float(request["latency_ms"]),
                "throughput": float(manifest["throughput_requests_per_s"]),
                "run": str(path),
            }
            for request in manifest.get("requests", [])
        )
    return rows


def render_latency(queued_paths: list[Path], static_paths: list[Path], output: Path) -> dict[str, Any]:
    groups = {"queued": _latency_rows(queued_paths), "main": _latency_rows(static_paths)}
    summary: dict[str, Any] = {"groups": {}}
    for name, rows in groups.items():
        values = [row["latency_ms"] for row in rows]
        throughputs = [row["throughput"] for row in rows]
        summary["groups"][name] = {
            "request_count": len(values),
            "mean_latency_ms": mean(values),
            "stdev_latency_ms": stdev(values) if len(values) > 1 else 0.0,
            "mean_wave_throughput_requests_per_s": mean(throughputs),
            "runs": sorted({row["run"] for row in rows}),
        }
    width, height, left, right = 1100, 500, 180, 1030
    max_latency = max(summary["groups"][name]["mean_latency_ms"] for name in summary["groups"]) * 1.35
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        "<style>text{font-family:Arial,sans-serif}</style>",
        _svg_text(28, 36, "Wan2.2 long-workload end-to-end latency", 24, fill="#172238"),
        _svg_text(
            28,
            62,
            "512x512, 16 frames, 8 steps, 4 requests per wave; three non-profiler waves per mode",
            14,
            fill="#526178",
        ),
    ]
    metrics = [("mean latency (ms)", "mean_latency_ms"), ("latency stdev (ms)", "stdev_latency_ms")]
    for index, (label, key) in enumerate(metrics):
        y = 120 + index * 150
        svg.append(_svg_text(28, y + 30, label, 15, fill="#172238", font_weight="600"))
        scale = (
            max_latency
            if key == "mean_latency_ms"
            else max(summary["groups"][name][key] for name in summary["groups"]) * 1.6 + 1
        )
        for group_index, name in enumerate(("queued", "main")):
            value = summary["groups"][name][key]
            bar_y = y + 55 + group_index * 42
            bar_width = (right - left) * value / scale
            svg.append(_svg_text(28, bar_y + 20, name, 13, fill="#172238"))
            svg.append(
                f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="28" rx="3" fill="{COLORS[name]}"/>'
            )
            svg.append(_svg_text(left + bar_width + 8, bar_y + 20, f"{value:.1f}", 13, fill="#172238"))
    svg.append(
        _svg_text(
            28,
            455,
            "Non-profiler manifest values are authoritative; profiler traces are not included in this chart.",
            12,
            fill="#526178",
        )
    )
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queued-profile", type=Path, required=True)
    parser.add_argument("--main-profile", type=Path, required=True)
    parser.add_argument("--queued-run", type=Path, action="append", required=True)
    parser.add_argument("--main-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline = render_timeline(args.queued_profile, args.main_profile, args.output_dir / "wan22_pp_timeline.svg")
    latency = render_latency(args.queued_run, args.main_run, args.output_dir / "wan22_latency_comparison.svg")
    summary = {"timeline": timeline, "latency": latency}
    (args.output_dir / "wan22_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
