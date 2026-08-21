#!/usr/bin/env python3
"""Run an offline diffusion batch experiment and export PP/timing traces.

This runner intentionally uses ``AsyncOmni(diffusion_batch_size=N)`` instead
of the online ``vllm serve`` entrypoint.  The current CLI does not expose
``diffusion_batch_size`` and therefore cannot test request-level batching.

The output directory contains:

* ``pp_trace/pp_rank_*.jsonl``: low-overhead PP spans;
* ``timeline.json`` and ``timeline.txt``: rendered PP activity timeline;
* ``requests.json``: request success, output metadata, and wall times;
* ``torch_profile/``: optional worker profiler traces.

Example:

    python tools/benchmark/run_offline_diffusion_batch_trace.py \
        --model ../models/Wan2.2-TI2V-5B-Diffusers \
        --deploy-config vllm_omni/deploy/wan2_2_ti2v.yaml \
        --pipeline-parallel-size 4 --diffusion-batch-size 4 \
        --request-batch-max-wait-ms 100 --request-count 4 \
        --image /path/to/input.jpg --num-inference-steps 4 \
        --output-dir results/wan-ti2v-batch4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=4)
    parser.add_argument("--diffusion-batch-size", type=int, default=4)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--request-batch-max-wait-ms", type=float, default=100.0)
    parser.add_argument("--prompt", default="A cinematic mountain landscape with slowly moving clouds.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync-trace", action="store_true")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--torch-profile-stack", action="store_true")
    parser.add_argument("--bin-us", type=int, default=100)
    args = parser.parse_args()
    if args.request_count < 1 or args.diffusion_batch_size < 1:
        parser.error("--request-count and --diffusion-batch-size must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    return args


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


async def _collect_one(omni: Any, prompt: dict[str, Any], params: Any, request_id: str) -> dict[str, Any]:
    start = time.perf_counter()
    last_output = None
    try:
        async for output in omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=[params],
        ):
            last_output = output
        if last_output is None:
            raise RuntimeError("request completed without an output")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "request_id": request_id,
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "final_output_type": getattr(last_output, "final_output_type", None),
            "is_pipeline_output": bool(getattr(last_output, "is_pipeline_output", False)),
            "has_images": bool(getattr(last_output, "images", None)),
            "metrics": _json_safe(getattr(last_output, "metrics", None)),
        }
    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_prompt(args: argparse.Namespace, model_class_name: str | None) -> dict[str, Any]:
    from vllm_omni.model_extras import build_image_to_video_prompt

    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "modalities": ["video"],
        "multi_modal_data": {},
    }
    if args.image is not None:
        from PIL import Image

        prompt["multi_modal_data"] = {"image": Image.open(args.image).convert("RGB")}
    return build_image_to_video_prompt(
        model_class_name=model_class_name,
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )


def _render_timeline(trace_dir: Path, output_dir: Path, bin_us: int) -> None:
    renderer = Path(__file__).with_name("trace_diffusion_pp_timeline.py")
    command = [
        sys.executable,
        str(renderer),
        "--trace-dir",
        str(trace_dir),
        "--output",
        str(output_dir / "timeline"),
        "--bin-us",
        str(bin_us),
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"timeline rendering failed with exit code {completed.returncode}")


async def _run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    trace_dir = output_dir / "pp_trace"
    profile_dir = output_dir / "torch_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    for path in trace_dir.glob("pp_rank_*.jsonl"):
        path.unlink()

    # pp_trace reads these variables at module import time, so set them before
    # importing AsyncOmni or any vLLM-Omni module.
    os.environ["VLLM_OMNI_PP_TRACE_DIR"] = str(trace_dir)
    os.environ["VLLM_OMNI_PP_TRACE_SYNC"] = "1" if args.sync_trace else "0"

    import torch
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_model_class_name
    from vllm_omni.platforms import current_omni_platform

    omni_kwargs: dict[str, Any] = {
        "model": args.model,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "diffusion_batch_size": args.diffusion_batch_size,
        # Keep the scheduler-visible value explicit as well. Stage startup
        # will use diffusion_batch_size as the final authoritative value.
        "max_num_seqs": args.diffusion_batch_size,
        "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
    }
    if args.deploy_config is not None:
        omni_kwargs["deploy_config"] = args.deploy_config
    if args.torch_profile:
        profile_dir.mkdir(parents=True, exist_ok=True)
        omni_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": args.torch_profile_stack,
            "torch_profiler_record_shapes": not args.torch_profile_stack,
        }

    print(f"[offline-batch] starting AsyncOmni: {omni_kwargs}", flush=True)
    omni = AsyncOmni(**omni_kwargs)
    profile_started = False
    try:
        model_class_name = get_model_class_name(omni)
        prompt = _build_prompt(args, model_class_name)

        def make_params(seed: int, steps: int) -> Any:
            generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
            return OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.fps,
                num_inference_steps=steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

        warmup_results = []
        for index in range(args.warmup_requests):
            warmup_results.append(
                await _collect_one(
                    omni,
                    prompt,
                    make_params(args.seed + index, args.num_inference_steps),
                    f"warmup-{index}",
                )
            )

        if args.torch_profile:
            print(f"[offline-batch] starting torch profiler in {profile_dir}", flush=True)
            await omni.start_profile(profile_prefix=str(profile_dir / "profile"))
            profile_started = True

        start = time.perf_counter()
        request_tasks = [
            _collect_one(
                omni,
                prompt,
                make_params(args.seed + 1000 + index, args.num_inference_steps),
                f"request-{index}",
            )
            for index in range(args.request_count)
        ]
        results = await asyncio.gather(*request_tasks)
        total_elapsed_ms = (time.perf_counter() - start) * 1000.0

        profile_results = []
        if profile_started:
            print("[offline-batch] stopping torch profiler", flush=True)
            profile_results = await omni.stop_profile()
            profile_started = False

        summary = {
            "model": args.model,
            "model_class_name": model_class_name,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "diffusion_batch_size": args.diffusion_batch_size,
            "max_num_seqs": args.diffusion_batch_size,
            "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
            "request_count": args.request_count,
            "warmup_requests": args.warmup_requests,
            "total_elapsed_ms": total_elapsed_ms,
            "warmup_results": warmup_results,
            "results": results,
            "torch_profile_results": _json_safe(profile_results),
            "trace_dir": str(trace_dir),
        }
        (output_dir / "requests.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not all(item["ok"] for item in results):
            raise RuntimeError("at least one measured request failed; see requests.json")

        _render_timeline(trace_dir, output_dir, args.bin_us)
        print(f"[offline-batch] complete: {output_dir}", flush=True)
        return 0
    finally:
        if profile_started:
            try:
                await omni.stop_profile()
            except Exception:
                pass
        omni.shutdown()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
