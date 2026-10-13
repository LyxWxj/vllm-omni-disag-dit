#!/usr/bin/env python3
"""Trace hook and HTML renderer for queued pipeline parallel benchmarks.

The benchmark runner imports :func:`install` when trace mode is enabled.  The
same file can render any completed run with ``render TRACE_DIR [OUTPUT]``;
trace output does not need a per-artifact copy of the visualizer.
"""

from __future__ import annotations

import argparse
import functools
import html
import json
import os
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_ACTIVE_STEP_STATE: ContextVar[Any | None] = ContextVar("queued_pp_active_step_state", default=None)
_COLORS = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#0891b2", "#be123c"]


def _trace_dir() -> Path | None:
    value = os.environ.get("VLLM_OMNI_PP_TRACE_DIR")
    return Path(value) if value else None


def _write(path_name: str, record: dict[str, Any]) -> None:
    trace_dir = _trace_dir()
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = {**record, "timestamp_ns": time.monotonic_ns()}
    with (trace_dir / path_name).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def _sync(worker: Any) -> None:
    if os.environ.get("VLLM_OMNI_PP_TRACE_SYNC") != "1":
        return
    import torch

    device = getattr(worker, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        event.synchronize()


def _request_state(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if method == "denoise_step":
        states = kwargs.get("states")
        return states[0] if states else None
    return args[0] if args else None


def _install_pipeline_hooks(worker: Any) -> None:
    pipeline = worker.model_runner.pipeline
    if getattr(pipeline, "_queued_pp_trace_installed", False):
        return

    for component in ("transformer", "transformer_2"):
        model = getattr(pipeline, component, None)
        original_forward = getattr(model, "forward", None)
        if not callable(original_forward) or getattr(model, "_queued_pp_trace_forward_installed", False):
            continue

        @functools.wraps(original_forward)
        def traced_forward(
            *args: Any,
            __component: str = component,
            __original: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            state = _ACTIVE_STEP_STATE.get()
            _sync(worker)
            started_ns = time.monotonic_ns()
            error = None
            try:
                return __original(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _sync(worker)
                _write(
                    f"worker_rank{worker.rank}.jsonl",
                    {
                        "kind": "dit_forward",
                        "phase": "span",
                        "rank": worker.rank,
                        "component": __component,
                        "request_id": getattr(state, "request_id", None),
                        "step_index": getattr(state, "step_index", None),
                        "start_ns": started_ns,
                        "end_ns": time.monotonic_ns(),
                        "trace_sync": os.environ.get("VLLM_OMNI_PP_TRACE_SYNC") == "1",
                        "sync_scope": "compute_stream" if os.environ.get("VLLM_OMNI_PP_TRACE_SYNC") == "1" else "host",
                        "error": error,
                    },
                )

        model.forward = traced_forward
        model._queued_pp_trace_forward_installed = True

    for method in ("denoise_step", "step_scheduler", "post_decode"):
        original = getattr(pipeline, method, None)
        if not callable(original):
            continue

        @functools.wraps(original)
        def traced(*args: Any, __method: str = method, __original: Any = original, **kwargs: Any) -> Any:
            state = _request_state(__method, args, kwargs)
            if __method == "denoise_step":
                token = _ACTIVE_STEP_STATE.set(state)
                try:
                    return __original(*args, **kwargs)
                finally:
                    _ACTIVE_STEP_STATE.reset(token)
            started_ns = time.monotonic_ns()
            error = None
            try:
                return __original(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _write(
                    f"worker_rank{worker.rank}.jsonl",
                    {
                        "kind": __method,
                        "phase": "span",
                        "rank": worker.rank,
                        "request_id": getattr(state, "request_id", None),
                        "step_index": getattr(state, "step_index", None),
                        "start_ns": started_ns,
                        "end_ns": time.monotonic_ns(),
                        "trace_sync": False,
                        "sync_scope": "host",
                        "error": error,
                    },
                )

        setattr(pipeline, method, traced)
    pipeline._queued_pp_trace_installed = True


def install() -> None:
    """Install process-local hooks before workers and Engine are created."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    if getattr(DiffusionWorker.execute_prepared_stepwise, "_queued_pp_trace_wrapper", False):
        return

    original_step = DiffusionWorker.execute_prepared_stepwise

    @functools.wraps(original_step)
    def trace_prepared_step(worker: Any, scheduler_output: Any, output_rank: int = 0) -> Any:
        _install_pipeline_hooks(worker)
        return original_step(worker, scheduler_output, output_rank=output_rank)

    trace_prepared_step._queued_pp_trace_wrapper = True
    DiffusionWorker.execute_prepared_stepwise = trace_prepared_step

    original_decode_ready = getattr(DiffusionWorker, "prepare_queued_decode", None)
    if callable(original_decode_ready):

        @functools.wraps(original_decode_ready)
        def trace_decode_ready(worker: Any, request_ids: tuple[str, ...]) -> Any:
            started_ns = time.monotonic_ns()
            error = None
            try:
                return original_decode_ready(worker, request_ids)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _write(
                    f"worker_rank{worker.rank}.jsonl",
                    {
                        "kind": "decode_ready",
                        "phase": "span",
                        "rank": worker.rank,
                        "request_ids": list(request_ids),
                        "start_ns": started_ns,
                        "end_ns": time.monotonic_ns(),
                        "trace_sync": False,
                        "sync_scope": "host",
                        "error": error,
                    },
                )

        DiffusionWorker.prepare_queued_decode = trace_decode_ready

    original_decode = DiffusionWorker.execute_prepared_decode

    @functools.wraps(original_decode)
    def trace_decode(worker: Any, request_id: str) -> Any:
        started_ns = time.monotonic_ns()
        error = None
        try:
            return original_decode(worker, request_id)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            _write(
                f"worker_rank{worker.rank}.jsonl",
                {
                    "kind": "decode_rpc",
                    "phase": "span",
                    "rank": worker.rank,
                    "request_id": request_id,
                    "start_ns": started_ns,
                    "end_ns": time.monotonic_ns(),
                    "trace_sync": False,
                    "sync_scope": "host",
                    "error": error,
                },
            )

    DiffusionWorker.execute_prepared_decode = trace_decode

    def wrap_engine(name: str, kind: str) -> None:
        original = getattr(DiffusionEngine, name, None)
        if not callable(original) or getattr(original, "_queued_pp_trace_wrapper", False):
            return

        @functools.wraps(original)
        def wrapped(engine: Any, *args: Any, __original: Any = original, **kwargs: Any) -> Any:
            started_ns = time.monotonic_ns()
            before = getattr(engine, "_queued_decode_round", None)
            was_dispatched = before is not None and before.dispatched
            result = __original(engine, *args, **kwargs)
            after = getattr(engine, "_queued_decode_round", None)
            if kind == "decode_round" and after is not None and after is not before:
                _write(
                    "engine_control.jsonl",
                    {
                        "kind": "decode_round",
                        "phase": "sealed",
                        "request_ids": list(after.request_ids),
                        "threshold": int(engine.od_config.max_inflight_batches),
                        "start_ns": started_ns,
                        "end_ns": time.monotonic_ns(),
                    },
                )
            elif (
                kind == "decode_dispatch"
                and before is not None
                and not was_dispatched
                and after is not None
                and after.dispatched
            ):
                _write(
                    "engine_control.jsonl",
                    {
                        "kind": "decode_dispatch",
                        "phase": "submitted",
                        "request_ids": list(after.request_ids),
                        "start_ns": started_ns,
                        "end_ns": time.monotonic_ns(),
                    },
                )
            return result

        wrapped._queued_pp_trace_wrapper = True
        setattr(DiffusionEngine, name, wrapped)

    wrap_engine("_maybe_start_queued_decode_round", "decode_round")
    wrap_engine("_advance_queued_decode_round", "decode_dispatch")

    original_authorize = DiffusionEngine._authorize_waiting_queued_batches

    @functools.wraps(original_authorize)
    def trace_authorize(engine: Any, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.monotonic_ns()
        before = {
            id(batch.step_future) for batch in engine._queued_pipeline_batches.values() if batch.step_future is not None
        }
        result = original_authorize(engine, *args, **kwargs)
        after = {
            id(batch.step_future)
            for batch in engine._queued_pipeline_batches.values()
            if batch.step_future is not None and id(batch.step_future) not in before
        }
        if after:
            _write(
                "engine_control.jsonl",
                {
                    "kind": "step_future",
                    "phase": "submitted",
                    "future_count": len(after),
                    "start_ns": started_ns,
                    "end_ns": time.monotonic_ns(),
                },
            )
        return result

    DiffusionEngine._authorize_waiting_queued_batches = trace_authorize

    original_commit = DiffusionEngine._commit_queued_pipeline_step

    @functools.wraps(original_commit)
    def trace_commit(engine: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.monotonic_ns()
        result = original_commit(engine, batch, *args, **kwargs)
        _write(
            "engine_control.jsonl",
            {
                "kind": "scheduler_commit",
                "request_id": batch.request_id,
                "step_index": batch.step_index,
                "finalizing": batch.finalizing,
                "start_ns": started_ns,
                "end_ns": time.monotonic_ns(),
            },
        )
        return result

    DiffusionEngine._commit_queued_pipeline_step = trace_commit


def _load_trace(trace_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    for path in sorted((trace_dir / "worker_events").glob("*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return manifest, events


def _request_parameters(manifest: dict[str, Any], requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = manifest.get("config", {})
    dimensions = config.get("dimensions", [None, None, None])
    default_frames, default_height, default_width = (dimensions + [None, None, None])[:3]
    configured = config.get("request_parameters", [])
    request_steps = config.get("request_steps", [])
    parameters: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        profile = configured[index] if index < len(configured) and isinstance(configured[index], dict) else {}
        parameters.append(
            {
                "height": request.get("height", profile.get("height", default_height)),
                "width": request.get("width", profile.get("width", default_width)),
                "num_frames": request.get("num_frames", profile.get("num_frames", default_frames)),
                "num_inference_steps": request.get(
                    "num_inference_steps",
                    profile.get(
                        "num_inference_steps",
                        request_steps[index] if index < len(request_steps) else config.get("num_inference_steps"),
                    ),
                ),
                "guidance_scale": request.get(
                    "guidance_scale", profile.get("guidance_scale", config.get("guidance_scale"))
                ),
            }
        )
    return parameters


def _render(trace_dir: Path, output: Path) -> None:
    manifest, events = _load_trace(trace_dir)
    requests = manifest.get("requests", [])
    parameters = _request_parameters(manifest, requests)
    if not any(event.get("kind") == "decode_dispatch" for event in events):
        # Older runs were captured before the mutable-round dispatch hook was
        # fixed. Derive the host dispatch interval from the readiness and first
        # Decode RPC timestamps without modifying the raw event files.
        for round_event in [event for event in events if event.get("kind") == "decode_round"]:
            request_ids = set(round_event.get("request_ids", []))
            ready = [
                event
                for event in events
                if event.get("kind") == "decode_ready" and request_ids.intersection(event.get("request_ids", []))
            ]
            decode = [
                event
                for event in events
                if event.get("kind") == "decode_rpc" and event.get("request_id") in request_ids
            ]
            if not ready or not decode:
                continue
            start_ns = max(event["end_ns"] for event in ready)
            end_ns = min(event["start_ns"] for event in decode)
            events.append(
                {
                    "kind": "decode_dispatch",
                    "phase": "derived",
                    "derived": True,
                    "request_ids": sorted(request_ids),
                    "start_ns": start_ns,
                    "end_ns": max(start_ns, end_ns),
                }
            )
    starts = [int(item["started_ns"]) for item in requests if item.get("started_ns")]
    starts.extend(int(item["start_ns"]) for item in events if item.get("start_ns"))
    if not starts:
        raise RuntimeError(f"No timestamped requests or events in {trace_dir}")
    base_ns = min(starts)
    duration = float(manifest.get("elapsed_s") or 0.0)
    spans: list[dict[str, Any]] = []
    for event in events:
        if "start_ns" not in event or "end_ns" not in event:
            continue
        start = max(0.0, (int(event["start_ns"]) - base_ns) / 1e9)
        end = max(start, (int(event["end_ns"]) - base_ns) / 1e9)
        duration = max(duration, end)
        kind = event.get("kind", "event")
        rank = event.get("rank")
        lane = f"rank{rank}.{kind}" if rank is not None else f"engine.{kind}"
        spans.append({**event, "lane": lane, "start": start, "end": end})
    if not spans:
        raise RuntimeError(f"No complete trace spans in {trace_dir}")

    request_ids = [item.get("request_id", f"request{index}") for index, item in enumerate(requests)]
    signatures = {
        (
            item.get("height"),
            item.get("width"),
            item.get("num_frames"),
            item.get("num_inference_steps"),
            item.get("guidance_scale"),
        )
        for item in parameters
    }
    workload_type = "homogeneous" if len(signatures) <= 1 else "heterogeneous"
    resolutions = sorted(
        {f"{item['width']}×{item['height']}" for item in parameters if item["width"] and item["height"]}
    )
    frame_counts = sorted({str(item["num_frames"]) for item in parameters if item["num_frames"] is not None})
    step_counts = sorted(
        {str(item["num_inference_steps"]) for item in parameters if item["num_inference_steps"] is not None}
    )
    colors = {request_id: _COLORS[index % len(_COLORS)] for index, request_id in enumerate(sorted(request_ids))}
    row_h, left, right, top = 34, 220, 1400, 42
    lanes = sorted({item["lane"] for item in spans})
    height = top + len(lanes) * row_h + 38
    scale = (right - left) / max(duration, 0.001)
    grid: list[str] = []
    tick_step = max(1, round(max(duration, 1.0) / 12))
    for second in range(0, int(duration) + tick_step, tick_step):
        x = left + min(second, duration) * scale
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 30}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 8}" class="axis" text-anchor="middle">{second}s</text>'
        )
    lane_y = {lane: top + index * row_h for index, lane in enumerate(lanes)}
    lane_svg = [
        f'<text x="{left - 10}" y="{y + 20}" class="label" text-anchor="end">{html.escape(lane)}</text>'
        f'<rect x="{left}" y="{y + 5}" width="{right - left}" height="21" class="track"/>'
        for lane, y in lane_y.items()
    ]
    bars: list[str] = []
    for span in spans:
        x = left + span["start"] * scale
        width = max(1.5, (span["end"] - span["start"]) * scale)
        request_id = span.get("request_id")
        request_label = request_id or ",".join(str(item) for item in span.get("request_ids", []))
        color_id = (
            request_id
            if request_id in colors
            else next((item for item in colors if request_id and request_id.startswith(item + "-")), None)
        )
        color = colors.get(color_id, "#64748b")
        duration_ms = (int(span["end_ns"]) - int(span["start_ns"])) / 1e6
        label = (
            f"{request_label or span.get('kind', 'engine')} · step {span.get('step_index', '')} · {duration_ms:.2f} ms"
        )
        bars.append(
            f'<rect x="{x:.1f}" y="{lane_y[span["lane"]] + 6}" width="{width:.1f}" '
            f'height="19" rx="2" fill="{color}"><title>{html.escape(label)}</title></rect>'
        )
    meta = manifest.get("config", {})
    legend = "".join(
        f'<span><i style="background:{colors[item]}"></i>{html.escape(item)}</span>' for item in sorted(colors)
    )
    summary = "".join(
        f"<tr><td>{html.escape(str(item.get('request_id', '')))}</td>"
        f"<td>{html.escape(str(param['width']))}×{html.escape(str(param['height']))}</td>"
        f"<td>{html.escape(str(param['num_frames']))}</td>"
        f"<td>{html.escape(str(param['num_inference_steps']))}</td>"
        f"<td>{html.escape(str(param['guidance_scale']))}</td>"
        f"<td>{item.get('seed', '')}</td>"
        f"<td>{float(item.get('latency_ms', 0.0)):.2f} ms</td>"
        f"<td>{html.escape(str(item.get('decoded_sha256', '')))}</td></tr>"
        for item, param in zip(requests, parameters, strict=True)
    )
    css = """body{margin:0;padding:20px;color:#1f2937;background:#fff;font:13px/1.45 system-ui,sans-serif}
    h1{font-size:20px;margin:0 0 4px}.meta{color:#64748b;margin-bottom:14px}
    .legend{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0;color:#64748b}
    .legend i{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}
    svg{width:100%;height:auto;display:block}.grid{stroke:#dbe3ec;stroke-width:1}
    .label,.axis{fill:#64748b;font-size:11px}.track{fill:#f1f5f9}
    table{border-collapse:collapse;width:100%;margin-top:16px;font-size:11px}
    th,td{padding:5px 7px;border-bottom:1px solid #e2e8f0;text-align:left;overflow-wrap:anywhere}
    th{font-weight:600}"""
    meta_line = (
        f"workload={workload_type} · {meta.get('request_count', len(requests))} requests · "
        f"PP={meta.get('pipeline_parallel_size', '?')} · "
        f"resolution={', '.join(resolutions) or '?'} · "
        f"frames={', '.join(frame_counts) or '?'} · "
        f"steps={', '.join(step_counts) or '?'} · "
        f"max_num_seqs={meta.get('max_num_seqs', '?')} · "
        f"max_inflight_batches={meta.get('max_inflight_batches', '?')} · "
        f"{duration:.3f}s · {manifest.get('throughput_requests_per_s', 0.0):.5f} req/s · "
        f"commit {html.escape(str(manifest.get('git_revision', 'unknown')))}"
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Queued PP trace</title><style>{css}</style><h1>Queued pipeline parallel trace</h1>
<div class="meta">{meta_line}</div><div class="legend">{legend}</div>
<svg viewBox="0 0 {right + 5} {height}" role="img" aria-label="Queued pipeline parallel trace">
<g>{"".join(grid)}{"".join(lane_svg)}{"".join(bars)}</g></svg>
<table><thead><tr><th>request</th><th>resolution</th><th>frames</th><th>steps</th>
<th>guidance</th><th>seed</th><th>latency</th><th>decoded SHA-256</th></tr></thead>
<tbody>{summary}</tbody></table></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"wrote {output} with {len(spans)} spans")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render one completed trace directory")
    render.add_argument("trace_dir", type=Path)
    render.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "render":
        _render(args.trace_dir, args.output or args.trace_dir / "queued_pp_trace.html")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        install()
    else:
        main()
