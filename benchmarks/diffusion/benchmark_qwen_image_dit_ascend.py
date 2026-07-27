#!/usr/bin/env python3
"""Benchmark one vLLM-Omni Qwen-Image DiT forward on Ascend NPU.

The script loads only the vLLM-Omni transformer implementation. Text encoder,
VAE, scheduler, and host-side pipeline work are excluded from both memory and
latency measurements. Synthetic inputs match the tensor shapes of a Qwen-Image
denoising step.

Example:
    python benchmarks/diffusion/benchmark_qwen_image_dit_ascend.py \
        --model /path/to/Qwen-Image \
        --local-files-only \
        --batch-sizes 1 2 4 8 \
        --height 1024 --width 1024 \
        --attention-backend auto \
        --warmup 3 --iterations 10
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import torch_npu
except ImportError as exc:
    raise SystemExit(
        "torch_npu is required. Run this script in an Ascend PyTorch environment."
    ) from exc

try:
    from vllm.config.load import LoadConfig
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm_omni.diffusion.data import (
        DiffusionParallelConfig,
        OmniDiffusionConfig,
    )
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )
    from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
    from vllm_omni.diffusion.worker.diffusion_worker import (
        _create_diffusion_worker_vllm_config,
        _make_diffusion_vllm_model_config,
    )
    from vllm_omni.platforms import current_omni_platform
except ImportError as exc:
    raise SystemExit(
        "A working vllm, vllm-ascend, and vllm-omni installation is required."
    ) from exc


GIB = 1024**3


class QwenImageDiTOnly(nn.Module):
    """Minimal loadable container for the native vLLM-Omni Qwen-Image DiT."""

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config,
            QwenImageTransformer2DModel,
        )
        self.transformer = QwenImageTransformer2DModel(
            od_config=od_config,
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure native vLLM-Omni Qwen-Image DiT latency on Ascend NPU."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Hugging Face model ID or local Qwen-Image model root.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face model revision.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require --model to be an existing local directory.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Ascend NPU device index (default: 0).",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Local HCCL initialization port (default: 29500).",
    )
    parser.add_argument(
        "--attention-backend",
        type=str.upper,
        choices=("AUTO", "FLASH_ATTN", "TORCH_SDPA"),
        default="AUTO",
        help="AUTO uses MindIE-SD FlashAttention when installed, otherwise SDPA.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to benchmark (default: 1 2 4 8).",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=256,
        help="Synthetic prompt embedding length used by joint attention.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Model and input dtype (default: bfloat16).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Normalized diffusion timestep passed to the DiT (default: 0.5).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Used only if the loaded transformer has guidance embeddings.",
    )
    parser.add_argument(
        "--vae-scale-factor",
        type=int,
        default=8,
        help="Qwen-Image VAE spatial compression factor (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("qwen_image_dit_ascend_results.json"),
        help="Result JSON path (default: qwen_image_dit_ascend_results.json).",
    )
    args = parser.parse_args()

    if args.local_files_only and not Path(args.model).is_dir():
        parser.error("--local-files-only requires --model to be a local directory")
    if args.device < 0:
        parser.error("--device must be non-negative")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1, 65535]")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("all --batch-sizes values must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.text_seq_len <= 0:
        parser.error("--text-seq-len must be positive")
    if args.vae_scale_factor <= 0:
        parser.error("--vae-scale-factor must be positive")
    alignment = args.vae_scale_factor * 2
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.height % alignment or args.width % alignment:
        parser.error(
            f"--height and --width must be divisible by {alignment} "
            "(VAE compression followed by 2x2 latent packing)"
        )
    return args


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def npu_synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def npu_memory_allocated(device: torch.device) -> int:
    return int(torch.npu.memory_allocated(device))


def npu_reset_peak_memory(device: torch.device) -> None:
    torch.npu.reset_peak_memory_stats(device)


def npu_peak_memory_allocated(device: torch.device) -> int:
    return int(torch.npu.max_memory_allocated(device))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def make_omni_config(args: argparse.Namespace, dtype: torch.dtype) -> OmniDiffusionConfig:
    attention_backend = None if args.attention_backend == "AUTO" else args.attention_backend
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel_size=1,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        model_class_name="QwenImagePipeline",
        dtype=dtype,
        revision=args.revision,
        num_gpus=1,
        master_port=args.master_port,
        enforce_eager=True,
        parallel_config=parallel_config,
        diffusion_attention_backend=attention_backend,
    )
    od_config.enrich_config()
    return od_config


def initialize_vllm_runtime(
    od_config: OmniDiffusionConfig,
    device: torch.device,
    device_index: int,
) -> Any:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(od_config.master_port)
    os.environ["LOCAL_RANK"] = str(device_index)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    current_omni_platform.set_device(device)
    vllm_config = _create_diffusion_worker_vllm_config(device, od_config)
    vllm_config.model_config = _make_diffusion_vllm_model_config(od_config)
    vllm_config.quant_config = od_config.quantization_config
    vllm_config.kernel_config.ir_op_priority = current_omni_platform.get_default_ir_op_priority(vllm_config)

    with set_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=od_config,
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=device_index,
        )
        # The generic single-rank initializer derives a temporary device from
        # global rank (rank 0 -> npu:0). Restore the explicitly requested NPU.
        current_omni_platform.set_device(device)
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        init_workspace_manager(device)
    return vllm_config


def load_vllm_dit(
    od_config: OmniDiffusionConfig,
    vllm_config: Any,
    device: torch.device,
) -> QwenImageDiTOnly:
    loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
    with set_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=od_config,
    ):
        container = loader.load_model(
            load_device=str(device),
            load_format="custom_pipeline",
            custom_pipeline_name=QwenImageDiTOnly,
            device=device,
        )
    return container


def make_inputs(
    model: QwenImageTransformer2DModel,
    od_config: OmniDiffusionConfig,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    latent_height = args.height // args.vae_scale_factor
    latent_width = args.width // args.vae_scale_factor
    packed_height = latent_height // 2
    packed_width = latent_width // 2
    image_seq_len = packed_height * packed_width
    joint_attention_dim = int(od_config.tf_model_config.get("joint_attention_dim", 3584))

    hidden_states = torch.randn(
        batch_size,
        image_seq_len,
        model.in_channels,
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        args.text_seq_len,
        joint_attention_dim,
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states_mask = torch.ones(
        batch_size,
        args.text_seq_len,
        device=device,
        dtype=torch.bool,
    )
    timestep = torch.full(
        (batch_size,),
        args.timestep,
        device=device,
        dtype=dtype,
    )
    guidance = None
    if model.guidance_embeds:
        guidance = torch.full(
            (batch_size,),
            args.guidance_scale,
            device=device,
            dtype=torch.float32,
        )

    img_shapes = [[(1, packed_height, packed_width)] for _ in range(batch_size)]
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "txt_seq_lens": [args.text_seq_len] * batch_size,
        "guidance": guidance,
        "return_dict": False,
    }


def run_forward(
    model: QwenImageTransformer2DModel,
    inputs: dict[str, Any],
) -> torch.Tensor:
    output = model(**inputs)
    return output.sample if hasattr(output, "sample") else output[0]


def benchmark_batch_size(
    model: QwenImageTransformer2DModel,
    od_config: OmniDiffusionConfig,
    vllm_config: Any,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    runtime_base_memory_bytes: int,
) -> dict[str, Any]:
    inputs = make_inputs(model, od_config, batch_size, args, device, dtype)
    npu_synchronize(device)
    input_memory_bytes = npu_memory_allocated(device) - runtime_base_memory_bytes

    with set_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=od_config,
    ):
        for _ in range(args.warmup):
            output = run_forward(model, inputs)
            npu_synchronize(device)
            del output

        npu_reset_peak_memory(device)
        latencies_ms: list[float] = []
        for _ in range(args.iterations):
            npu_synchronize(device)
            start = time.perf_counter_ns()
            output = run_forward(model, inputs)
            npu_synchronize(device)
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)
            del output

    mean_ms = statistics.fmean(latencies_ms)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "latency_ms": {
            "mean": mean_ms,
            "stddev": statistics.pstdev(latencies_ms),
            "min": min(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "p99": percentile(latencies_ms, 0.99),
            "max": max(latencies_ms),
            "samples": latencies_ms,
        },
        "throughput_images_per_second": batch_size * 1000.0 / mean_ms,
        "runtime_base_memory_gib": runtime_base_memory_bytes / GIB,
        "input_memory_gib": max(input_memory_bytes, 0) / GIB,
        "peak_memory_gib": npu_peak_memory_allocated(device) / GIB,
    }


def is_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "memory allocation",
            "memory_allocation",
            "acl_error_rt_memory",
        )
    )


def get_attention_backend(model: QwenImageTransformer2DModel) -> str:
    try:
        attention = model.transformer_blocks[0].attn.attn
        return str(attention.attn_backend.get_name())
    except (AttributeError, IndexError):
        return "unknown"


def device_name(device_index: int) -> str:
    try:
        return str(torch.npu.get_device_name(device_index))
    except Exception:
        return "unknown"


def print_result(result: dict[str, Any]) -> None:
    if result["status"] != "ok":
        print(
            f"{result['batch_size']:>5}  {'OOM':>10}  {'-':>10}  "
            f"{'-':>10}  {'-':>10}  {'-':>10}"
        )
        return
    latency = result["latency_ms"]
    print(
        f"{result['batch_size']:>5}  "
        f"{latency['mean']:>10.3f}  "
        f"{latency['p50']:>10.3f}  "
        f"{latency['p90']:>10.3f}  "
        f"{result['throughput_images_per_second']:>10.3f}  "
        f"{result['peak_memory_gib']:>10.3f}"
    )


def main() -> int:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    if not torch.npu.is_available():
        raise SystemExit("No available Ascend NPU was detected.")
    if args.device >= torch.npu.device_count():
        raise SystemExit(
            f"NPU device {args.device} does not exist; detected {torch.npu.device_count()} device(s)."
        )
    if not current_omni_platform.is_npu():
        raise SystemExit(
            f"vLLM-Omni selected {type(current_omni_platform).__name__}, expected NPUOmniPlatform."
        )

    device = current_omni_platform.get_torch_device(args.device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    od_config = make_omni_config(args, dtype)
    runtime_initialized = False
    container: QwenImageDiTOnly | None = None
    model: QwenImageTransformer2DModel | None = None

    try:
        vllm_config = initialize_vllm_runtime(od_config, device, args.device)
        runtime_initialized = True

        print(f"Loading native vLLM-Omni DiT from {args.model!r} on {device} ({args.dtype}) ...")
        container = load_vllm_dit(od_config, vllm_config, device)
        model = container.transformer
        model.eval()
        model.requires_grad_(False)
        npu_synchronize(device)
        runtime_base_memory_bytes = npu_memory_allocated(device)

        packed_height = args.height // args.vae_scale_factor // 2
        packed_width = args.width // args.vae_scale_factor // 2
        actual_attention_backend = get_attention_backend(model)
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": "vllm_omni.QwenImageTransformer2DModel",
            "model": args.model,
            "revision": args.revision,
            "device": str(device),
            "device_name": device_name(args.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "vllm": package_version("vllm"),
            "vllm_ascend": package_version("vllm-ascend"),
            "vllm_omni": package_version("vllm-omni"),
            "dtype": args.dtype,
            "attention_backend_requested": args.attention_backend,
            "attention_backend_actual": actual_attention_backend,
            "height": args.height,
            "width": args.width,
            "image_sequence_length": packed_height * packed_width,
            "text_sequence_length": args.text_seq_len,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "timestep": args.timestep,
            "seed": args.seed,
            "model_config": {
                "in_channels": model.in_channels,
                "joint_attention_dim": od_config.tf_model_config.get("joint_attention_dim", 3584),
                "num_layers": len(model.transformer_blocks),
                "num_attention_heads": od_config.tf_model_config.get("num_attention_heads", 24),
                "attention_head_dim": od_config.tf_model_config.get("attention_head_dim", 128),
                "guidance_embeds": model.guidance_embeds,
            },
        }

        print(
            f"Input: {args.height}x{args.width}, image tokens="
            f"{metadata['image_sequence_length']}, text tokens={args.text_seq_len}"
        )
        print(
            f"Attention backend: {actual_attention_backend} "
            f"(requested: {args.attention_backend})"
        )
        print(f"Warmup={args.warmup}, measured iterations={args.iterations}")
        print()
        print("batch     mean_ms      p50_ms      p90_ms     images/s    peak_GiB")
        print("-----  ----------  ----------  ----------  ----------  ----------")

        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch_size in args.batch_sizes:
                oom_result = None
                try:
                    result = benchmark_batch_size(
                        model,
                        od_config,
                        vllm_config,
                        batch_size,
                        args,
                        device,
                        dtype,
                        runtime_base_memory_bytes,
                    )
                except RuntimeError as exc:
                    if not is_out_of_memory(exc):
                        raise
                    oom_result = {
                        "batch_size": batch_size,
                        "status": "oom",
                        "error": str(exc),
                        "runtime_base_memory_gib": runtime_base_memory_bytes / GIB,
                    }
                if oom_result is not None:
                    result = oom_result
                    gc.collect()
                    torch.npu.empty_cache()
                    npu_synchronize(device)
                results.append(result)
                print_result(result)

        report = {"metadata": metadata, "results": results}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")
    finally:
        model = None
        container = None
        gc.collect()
        if runtime_initialized:
            destroy_distributed_env()
        torch.npu.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
