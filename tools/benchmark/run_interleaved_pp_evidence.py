#!/usr/bin/env python3
"""Collect reproducible diffusion PP correctness and host-trace evidence.

The runner deliberately uses ``AsyncOmni`` from a file-backed Python process:
the worker launcher uses ``spawn`` for pipeline parallelism, which cannot
re-import a program submitted through standard input.  One invocation records
one controlled wave.  It writes the final CPU latent for every request along
with a manifest that is suitable for exact-hash and numeric comparisons.

The optional PP JSONL trace is host timing only.  It can demonstrate that
different PP stages were concurrently scheduled, but it is not a device-kernel
profiler and must not be used as a device-overlap claim.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Wan checkpoint directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for manifest, latents, and traces.")
    parser.add_argument("--pipeline-parallel-size", type=int, required=True, help="Pipeline-parallel world size.")
    parser.add_argument(
        "--diffusion-pp-schedule",
        choices=("serial", "interleaved"),
        default="interleaved",
        help="PP schedule to pass to the diffusion engine.",
    )
    parser.add_argument(
        "--step-execution",
        action="store_true",
        help="Use Wan's retained-state step runtime with serial PP (for PP1 parity).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Active request capacity; defaults to request count.",
    )
    parser.add_argument("--num-requests", type=int, default=1, help="Concurrent requests in the measured wave.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for request zero; later requests use seed + index.")
    parser.add_argument("--prompt", default="A lighthouse on a cliff in a storm, cinematic lighting.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--mode", choices=("t2v", "ti2v"), default="t2v")
    parser.add_argument("--image", type=Path, help="Input image for --mode ti2v.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=None)
    parser.add_argument("--boundary-ratio", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--request-id-prefix", default="evidence")
    parser.add_argument(
        "--abort-request-index",
        type=int,
        default=None,
        help="Abort this request after --abort-delay-ms; use for PP drain evidence.",
    )
    parser.add_argument(
        "--abort-delay-ms",
        type=float,
        default=100.0,
        help="Delay before aborting --abort-request-index; after --abort-after-stage-forward when enabled.",
    )
    parser.add_argument(
        "--abort-after-stage-forward",
        action="store_true",
        help="Wait for a rank-0 stage-forward trace event for the target request before aborting.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Enable retained-state PP JSONL tracing in this directory (SYNC=0).",
    )
    parser.add_argument(
        "--profile-prefix",
        default=None,
        help="Optional worker profiler prefix. Profiling is diagnostic, not a latency benchmark.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=Path,
        default=None,
        help="Enable the torch profiler and write one device trace per worker to this directory.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Disable graph capture for easier diagnostics.")
    args = parser.parse_args()

    if args.pipeline_parallel_size <= 0:
        parser.error("--pipeline-parallel-size must be positive")
    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        parser.error("--height and --width must be positive multiples of 16")
    if args.num_frames <= 0 or args.num_inference_steps <= 0:
        parser.error("--num-frames and --num-inference-steps must be positive")
    if args.abort_request_index is not None and not 0 <= args.abort_request_index < args.num_requests:
        parser.error("--abort-request-index must identify a submitted request")
    if args.abort_delay_ms < 0:
        parser.error("--abort-delay-ms must be non-negative")
    if args.abort_after_stage_forward and args.abort_request_index is None:
        parser.error("--abort-after-stage-forward requires --abort-request-index")
    if args.abort_after_stage_forward and args.trace_dir is None:
        parser.error("--abort-after-stage-forward requires --trace-dir")
    if args.mode == "ti2v":
        if args.image is None:
            parser.error("--image is required for --mode ti2v")
        if not args.image.is_file():
            parser.error(f"--image does not exist: {args.image}")
    elif args.image is not None:
        parser.error("--image is valid only with --mode ti2v")
    if args.diffusion_pp_schedule == "interleaved" and args.pipeline_parallel_size == 1:
        parser.error("interleaved PP evidence requires --pipeline-parallel-size greater than one")
    return args


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tensor_digest(tensor: Any) -> str:
    """Hash a contiguous CPU tensor without relying on framework serialization."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(tensor).__name__}")
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class RequestEvidence:
    request_id: str
    seed: int
    latency_ms: float
    status: str
    aborted: bool
    abort_message: str | None
    latent_file: str | None
    latent_sha256: str | None
    latent_shape: list[int] | None
    latent_dtype: str | None
    peak_memory_mb: float
    stage_durations: dict[str, float]


def _sampling_params(args: argparse.Namespace, *, seed: int) -> Any:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    extra_args: dict[str, Any] = {}
    if args.flow_shift is not None:
        extra_args["flow_shift"] = args.flow_shift
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        boundary_ratio=args.boundary_ratio,
        output_type="latent",
        generator=generator,
        seed=seed,
        extra_args=extra_args,
    )


def _prompt(args: argparse.Namespace, image: Any | None) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "modalities": ["video"],
    }
    if image is not None:
        # This matches the public I2V request envelope and exercises the
        # multi_modal_data path used by Wan's step protocol.
        prompt["multi_modal_data"] = {"image": image}
    return prompt


def _output_latent(output: Any) -> Any:
    """Extract the latent from the current diffusion output formatter contract."""
    if output.latents is not None:
        return output.latents
    # ``format_diffusion_outputs`` keeps a non-image tensor in ``images`` for
    # historical API compatibility, including Wan's ``output_type='latent'``.
    images = output.images
    if len(images) == 1:
        return images[0]
    return None


async def _collect_one(
    omni: Any,
    args: argparse.Namespace,
    *,
    request_id: str,
    seed: int,
    image: Any | None,
    output_dir: Path,
) -> RequestEvidence:
    import torch

    started = time.perf_counter()
    final_output = None
    async for output in omni.generate(
        _prompt(args, image),
        _sampling_params(args, seed=seed),
        request_id=request_id,
    ):
        if output.finished:
            final_output = output
    elapsed_ms = (time.perf_counter() - started) * 1000
    if final_output is None:
        raise RuntimeError(f"{request_id} completed without a terminal output")
    if final_output.aborted:
        return RequestEvidence(
            request_id=request_id,
            seed=seed,
            latency_ms=elapsed_ms,
            status="aborted",
            aborted=True,
            abort_message=final_output.abort_message,
            latent_file=None,
            latent_sha256=None,
            latent_shape=None,
            latent_dtype=None,
            peak_memory_mb=float(final_output.peak_memory_mb or 0.0),
            stage_durations={str(key): float(value) for key, value in (final_output.stage_durations or {}).items()},
        )
    latent = _output_latent(final_output)
    if latent is None:
        raise RuntimeError(
            f"{request_id} returned no latent. The evidence runner requires output_type='latent'; "
            f"received final_output_type={final_output.final_output_type!r}, "
            f"images={len(final_output.images)}."
        )

    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"{request_id} returned {type(latent).__name__}, not a latent tensor")
    latent = latent.detach().cpu().contiguous()
    latent_path = output_dir / "latents" / f"{request_id}.pt"
    torch.save(latent, latent_path)
    return RequestEvidence(
        request_id=request_id,
        seed=seed,
        latency_ms=elapsed_ms,
        status="completed",
        aborted=False,
        abort_message=None,
        latent_file=str(latent_path.relative_to(output_dir)),
        latent_sha256=_tensor_digest(latent),
        latent_shape=list(latent.shape),
        latent_dtype=str(latent.dtype),
        peak_memory_mb=float(final_output.peak_memory_mb or 0.0),
        stage_durations={str(key): float(value) for key, value in (final_output.stage_durations or {}).items()},
    )


def _filter_startup_trace(trace_dir: Path, output_dir: Path) -> Path:
    """Keep only measured requests while retaining the original raw trace."""
    filtered_dir = output_dir / "host_trace" / "measured_raw"
    filtered_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted(trace_dir.glob("pp_rank_*.jsonl")):
        retained = []
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "dummy_req_id" in (record.get("request_ids") or []):
                continue
            retained.append(json.dumps(record, separators=(",", ":")))
        (filtered_dir / source.name).write_text("\n".join(retained) + "\n", encoding="utf-8")
    return filtered_dir


async def _wait_for_stage_forward_begin(trace_dir: Path, request_id: str, timeout_s: float = 60.0) -> None:
    """Wait until the target request has a rank-0 forward span in the PP trace."""
    trace_path = trace_dir / "pp_rank_0.jsonl"
    request_prefix = f"{request_id}-"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if trace_path.is_file():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                request_ids = record.get("request_ids") or ()
                if (
                    record.get("event") == "begin"
                    and record.get("name") == "stage_forward"
                    and any(str(candidate).startswith(request_prefix) for candidate in request_ids)
                ):
                    return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for stage-0 forward of request {request_id!r}")


def _trace_summary(trace_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    analyzer = Path(__file__).with_name("analyze_diffusion_pp_trace.py")
    if not list(trace_dir.glob("pp_rank_*.jsonl")):
        return None
    measured_trace_dir = _filter_startup_trace(trace_dir, output_dir)
    subprocess.run(
        [sys.executable, str(analyzer), str(measured_trace_dir), "--output-dir", str(output_dir / "host_trace")],
        check=True,
    )
    summary_path = output_dir / "host_trace" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


async def _run(args: argparse.Namespace) -> int:
    # Keep these imports after argument parsing and trace setup: spawned workers
    # inherit the environment before importing pp_trace.py.
    import platform

    import torch
    from PIL import Image

    from vllm_omni.entrypoints.async_omni import AsyncOmni

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "latents").mkdir()
    trace_dir = args.trace_dir.resolve() if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
        os.environ["VLLM_OMNI_PP_TRACE_DIR"] = str(trace_dir)
        os.environ["VLLM_OMNI_PP_TRACE_SYNC"] = "0"

    image = Image.open(args.image).convert("RGB") if args.image is not None else None
    max_num_seqs = args.max_num_seqs or args.num_requests
    config = {
        "model": str(Path(args.model).resolve()),
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_num_seqs": max_num_seqs,
        "diffusion_pp_schedule": args.diffusion_pp_schedule,
        "step_execution": args.step_execution or args.diffusion_pp_schedule == "interleaved",
        "request_batch_max_wait_ms": 0.0,
        "enforce_eager": args.enforce_eager,
    }
    if args.torch_profiler_dir is not None:
        args.torch_profiler_dir.mkdir(parents=True, exist_ok=False)
        config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.torch_profiler_dir),
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
        }
    manifest: dict[str, Any] = {
        "status": "running",
        "git_revision": _git_revision(),
        "timestamp_unix_s": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "run_config": {
            **config,
            "mode": args.mode,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "boundary_ratio": args.boundary_ratio,
            "flow_shift": args.flow_shift,
            "num_requests": args.num_requests,
            "seed": args.seed,
            "trace_sync": 0 if trace_dir is not None else None,
            "abort_after_stage_forward": args.abort_after_stage_forward,
        },
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
        "requests": [],
    }
    _write_json(output_dir / "manifest.json", manifest)

    omni = None
    profile_started = False
    try:
        omni = AsyncOmni(**config)
        if args.profile_prefix is not None:
            await omni.start_profile(args.profile_prefix)
            profile_started = True

        wave_started = time.perf_counter()
        request_ids = [f"{args.request_id_prefix}-{index}" for index in range(args.num_requests)]
        tasks = [
            asyncio.create_task(
                _collect_one(
                    omni,
                    args,
                    request_id=request_id,
                    seed=args.seed + index,
                    image=image,
                    output_dir=output_dir,
                )
            )
            for index, request_id in enumerate(request_ids)
        ]
        if args.abort_request_index is not None:
            abort_request_id = request_ids[args.abort_request_index]
            if args.abort_after_stage_forward:
                assert trace_dir is not None
                await _wait_for_stage_forward_begin(trace_dir, abort_request_id)
            await asyncio.sleep(args.abort_delay_ms / 1000)
            await omni.abort(abort_request_id)
        results = await asyncio.gather(*tasks)
        if args.abort_request_index is not None:
            aborted = results[args.abort_request_index]
            if not aborted.aborted:
                raise RuntimeError(f"{aborted.request_id} did not receive an aborted terminal output")
            if any(result.aborted for index, result in enumerate(results) if index != args.abort_request_index):
                raise RuntimeError("A non-target request was aborted while draining the PP runtime")
        manifest["wave_latency_ms"] = (time.perf_counter() - wave_started) * 1000
        manifest["requests"] = [asdict(result) for result in results]
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        raise
    finally:
        try:
            if omni is not None and profile_started:
                await omni.stop_profile()
        finally:
            if omni is not None:
                shutdown_started = time.perf_counter()
                omni.shutdown(timeout=30)
                manifest["shutdown_latency_ms"] = (time.perf_counter() - shutdown_started) * 1000
            _write_json(output_dir / "manifest.json", manifest)

    if trace_dir is not None:
        manifest["host_trace_summary"] = _trace_summary(trace_dir, output_dir)
        _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
