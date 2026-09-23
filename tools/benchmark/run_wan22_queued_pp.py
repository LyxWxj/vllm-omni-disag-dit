#!/usr/bin/env python3
"""Compare Wan2.2 PP=2 static-step and queued execution on identical requests."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any


def _write_worker_event(worker: Any, record: dict[str, Any]) -> None:
    trace_dir = os.environ.get("VLLM_OMNI_QUEUED_PP_TRACE_DIR")
    if not trace_dir:
        return
    rank = getattr(worker, "rank", "unknown")
    path = Path(trace_dir) / f"worker_rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")


def _stage_snapshot(worker: Any, stage_id: int) -> dict[str, Any]:
    stage = getattr(worker, "pipeline_stages", {}).get(stage_id)
    if stage is None:
        return {"pending": 0, "active": None, "awaiting_feedback": 0}
    return {
        "pending": len(stage.pending_tasks),
        "active": None if stage.active_task is None else stage.active_task.batch_id,
        "awaiting_feedback": len(stage.awaiting_feedback),
    }


def _task_fields(task: Any, stage_id: int, rank: int) -> dict[str, Any]:
    return {
        "batch_id": task.batch_id,
        "request_ids": list(task.request_ids),
        "step_index": task.step_index,
        "epoch": task.epoch,
        "branch": task.branch,
        "pp_stage_id": stage_id,
        "physical_rank": rank,
    }


def _tensor_bytes(value: Any) -> int:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
    except ImportError:
        return 0
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _install_worker_trace_hooks() -> None:
    """Record task lifecycle metadata and profiler ranges inside each Worker."""
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    def trace_worker_method(method_name: str) -> None:
        original = getattr(DiffusionWorker, method_name)
        if getattr(original, "_queued_pp_trace_hook", False):
            return

        @functools.wraps(original)
        def traced(worker: Any, offer: Any, *args: Any, **kwargs: Any) -> Any:
            started_ns = time.monotonic_ns()
            common = {
                "kind": "transfer_control",
                "method": method_name,
                "physical_rank": worker.rank,
                "identity": list(offer.identity),
                "edge_kind": offer.edge_kind.value,
                "src_rank": offer.src_rank,
                "dst_rank": offer.dst_rank,
            }
            _write_worker_event(worker, {**common, "phase": "enter", "timestamp_ns": started_ns})
            try:
                result = original(worker, offer, *args, **kwargs)
            except BaseException as exc:
                _write_worker_event(
                    worker,
                    {
                        **common,
                        "phase": "error",
                        "timestamp_ns": time.monotonic_ns(),
                        "duration_ns": time.monotonic_ns() - started_ns,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            _write_worker_event(
                worker,
                {
                    **common,
                    "phase": "return",
                    "timestamp_ns": time.monotonic_ns(),
                    "duration_ns": time.monotonic_ns() - started_ns,
                    "result": result if isinstance(result, (bool, type(None))) else type(result).__name__,
                },
            )
            return result

        traced._queued_pp_trace_hook = True  # type: ignore[attr-defined]
        setattr(DiffusionWorker, method_name, traced)

    for method_name in ("accept_pipeline_transfer_offer_all_ranks", "start_pipeline_transfer"):
        trace_worker_method(method_name)

    original_record = DiffusionWorker._record_pipeline_event
    if getattr(original_record, "_queued_pp_trace_hook", False):
        return

    @functools.wraps(original_record)
    def record_event(worker: Any, event: Any) -> Any:
        stage_id = event.pp_stage_id
        record = {
            "timestamp_ns": time.monotonic_ns(),
            "kind": "pipeline_event",
            "event_type": event.event_type.value,
            **_task_fields(event.task, stage_id, event.physical_rank),
            "stage_queue": _stage_snapshot(worker, stage_id),
        }
        _write_worker_event(worker, record)
        return original_record(worker, event)

    record_event._queued_pp_trace_hook = True  # type: ignore[attr-defined]
    DiffusionWorker._record_pipeline_event = record_event

    original_progress = DiffusionWorker.progress_pipeline

    @functools.wraps(original_progress)
    def progress_stage(worker: Any, pp_stage_id: int, *args: Any, **kwargs: Any) -> Any:
        import torch

        stage = getattr(worker, "pipeline_stages", {}).get(pp_stage_id)
        task = (
            None if stage is None else (stage.active_task or (stage.pending_tasks[0] if stage.pending_tasks else None))
        )
        label = "idle" if task is None else f"{task.batch_id}:step{task.step_index}:epoch{task.epoch}"
        started_ns = time.monotonic_ns()
        with torch.profiler.record_function(f"queued_pp::stage{pp_stage_id}::rank{worker.rank}::{label}"):
            result = original_progress(worker, pp_stage_id, *args, **kwargs)
        if result is not None:
            offer = result.output
            payload_bytes = 0
            if offer is not None:
                ticket = worker.pipeline_send_tickets.get(offer.identity)
                if ticket is not None:
                    payload_bytes = _tensor_bytes(ticket.message.payload)
            _write_worker_event(
                worker,
                {
                    "timestamp_ns": started_ns,
                    "kind": "stage_progress",
                    **_task_fields(result.event.task, result.event.pp_stage_id, result.event.physical_rank),
                    "duration_ns": time.monotonic_ns() - started_ns,
                    "transferred_bytes": payload_bytes,
                    "stage_queue": _stage_snapshot(worker, pp_stage_id),
                },
            )
        return result

    progress_stage._queued_pp_trace_hook = True  # type: ignore[attr-defined]
    DiffusionWorker.progress_pipeline = progress_stage

    original_transfers = DiffusionWorker.progress_pipeline_transfers

    @functools.wraps(original_transfers)
    def progress_transfers(worker: Any, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.monotonic_ns()
        progress = original_transfers(worker, *args, **kwargs)
        _write_worker_event(
            worker,
            {
                "timestamp_ns": started_ns,
                "kind": "transport_progress",
                "physical_rank": progress.rank,
                "offers": [
                    {
                        "batch_id": offer.batch_id,
                        "step_index": offer.step_index,
                        "epoch": offer.epoch,
                        "branch": offer.branch,
                        "edge_kind": offer.edge_kind.value,
                        "src_rank": offer.src_rank,
                        "dst_rank": offer.dst_rank,
                    }
                    for offer in progress.offers
                ],
                "completed_transfers": [list(completion.identity) for completion in progress.completions],
                "stage_queues": {str(stage_id): _stage_snapshot(worker, stage_id) for stage_id in (0, 1)},
                "retained_send_tickets": len(worker.pipeline_send_tickets),
                "receive_reservations": len(worker.pipeline_receive_reservations),
            },
        )
        return progress

    progress_transfers._queued_pp_trace_hook = True  # type: ignore[attr-defined]
    DiffusionWorker.progress_pipeline_transfers = progress_transfers


_install_worker_trace_hooks()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--mode", choices=("static", "queued"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--edge-buffer-slots", type=int, default=1)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="write per-rank PyTorch profiler traces")
    return parser.parse_args()


def _digest_output(output: Any) -> tuple[str, list[int], str]:
    import numpy as np
    import torch

    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if isinstance(output, np.ndarray):
        tensor = torch.from_numpy(output)
    elif isinstance(output, torch.Tensor):
        tensor = output
    else:
        raise TypeError(f"Expected tensor or ndarray output, got {type(output).__name__}")
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    return digest, list(tensor.shape), str(tensor.dtype)


async def _collect_request(omni: Any, args: argparse.Namespace, index: int) -> dict[str, Any]:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    request_id = f"wan22-queued-pp-{index:04d}"
    seed = args.seed + index
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0,
        output_type="pt",
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(seed),
        seed=seed,
    )
    prompt = {
        "prompt": "A small red boat crossing a still lake at sunrise.",
        "negative_prompt": "",
        "modalities": ["video"],
    }
    if args.arrival_interval_ms:
        await asyncio.sleep(index * args.arrival_interval_ms / 1000.0)
    started_ns = time.monotonic_ns()
    terminal = None
    async for output in omni.generate(prompt, sampling, request_id=request_id):
        if output.finished:
            terminal = output
    if terminal is None:
        raise RuntimeError(f"{request_id} returned no terminal output")
    digest, shape, dtype = _digest_output(terminal.images)
    return {
        "request_id": request_id,
        "seed": seed,
        "started_ns": started_ns,
        "finished_ns": time.monotonic_ns(),
        "latency_ms": (time.monotonic_ns() - started_ns) / 1e6,
        "decoded_sha256": digest,
        "decoded_shape": shape,
        "decoded_dtype": dtype,
        "stage_durations": {str(key): float(value) for key, value in (terminal.stage_durations or {}).items()},
        "peak_memory_mb": float(terminal.peak_memory_mb or 0.0),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(percentile * len(ordered)) - 1, 0)]


async def _run(args: argparse.Namespace) -> int:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    model_path = args.model.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    events_dir = output_dir / "worker_events"
    events_dir.mkdir()
    os.environ["VLLM_OMNI_QUEUED_PP_TRACE_DIR"] = str(events_dir)
    profiler_dir = output_dir / "torch_profiler"
    if args.profile:
        profiler_dir.mkdir()

    config: dict[str, Any] = {
        "model": str(model_path),
        "pipeline_parallel_size": 2,
        "max_num_seqs": args.max_num_seqs,
        "request_batch_max_wait_ms": 0.0,
        "step_execution": True,
        "mode": args.mode,
        "max_inflight_batches": args.max_inflight_batches,
        "edge_buffer_slots": args.edge_buffer_slots,
        "enforce_eager": True,
    }
    if args.profile:
        config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profiler_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
        }

    manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": str(model_path),
        "config": {
            "pipeline_parallel_size": 2,
            "step_execution": True,
            "max_num_seqs": args.max_num_seqs,
            "max_inflight_batches": args.max_inflight_batches,
            "edge_buffer_slots": args.edge_buffer_slots,
            "request_count": args.request_count,
            "arrival_interval_ms": args.arrival_interval_ms,
            "dimensions": [args.num_frames, args.height, args.width],
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": 1.0,
            "seed_start": args.seed,
            "profiler": args.profile,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    omni = None
    profiling = False
    try:
        omni = AsyncOmni(**config)
        if args.profile:
            await omni.start_profile(f"wan22_{args.mode}_pp2")
            profiling = True
        wave_started = time.perf_counter()
        requests = await asyncio.gather(*(_collect_request(omni, args, i) for i in range(args.request_count)))
        elapsed_s = time.perf_counter() - wave_started
        latencies = [item["latency_ms"] for item in requests]
        manifest.update(
            status="completed",
            elapsed_s=elapsed_s,
            throughput_requests_per_s=len(requests) / elapsed_s,
            latency_ms={
                "mean": sum(latencies) / len(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            requests=requests,
        )
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if omni is not None:
            try:
                if profiling:
                    await omni.stop_profile()
            finally:
                omni.shutdown(timeout=60)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    traces = sorted(profiler_dir.rglob("*.json")) if args.profile else []
    manifest["profiler_traces"] = [str(path.relative_to(output_dir)) for path in traces]
    manifest["worker_event_files"] = [str(path.relative_to(output_dir)) for path in sorted(events_dir.glob("*.jsonl"))]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
