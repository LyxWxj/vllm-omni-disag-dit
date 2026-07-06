# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight runner for diffusion submodule stages."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class DiffusionSubmoduleRunner:
    """Single-forward runner for diffusion submodule stages."""

    def __init__(
        self,
        vllm_config: Any,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None

    def load_model(
        self,
        memory_pool_context_fn: Any | None = None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        t0 = time.perf_counter()
        with get_memory_context():
            with DeviceMemoryProfiler() as mem:
                self.pipeline = model_loader.load_model(
                    od_config=self.od_config,
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )

        logger.info(
            "DiffusionSubmoduleRunner[stage=%s]: loaded in %.3fs, %.3f GiB GPU",
            self.od_config.model_stage,
            time.perf_counter() - t0,
            mem.consumed_memory / GiB_bytes,
        )

        self._warmup()

    def _warmup(self) -> None:
        """Warmup CUDA kernels to avoid first-call latency.

        The first forward pass on GPU triggers CUDA context initialization,
        CuDNN benchmarking, and memory allocation.  Without warmup the first
        real request pays ~20s of one-time overhead.  A dummy forward with a
        tiny tensor pays this cost once at startup instead.
        """
        stage = getattr(self.od_config, "model_stage", None)
        if stage != "decode" or self.pipeline is None:
            return
        if not hasattr(self.pipeline, "vae") or self.pipeline.vae is None:
            return

        logger.info("DiffusionSubmoduleRunner[decode]: warming up VAE decode...")
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                dummy = torch.randn(
                    1, self.pipeline.vae.config.z_dim, 1, 16, 16,
                    device=self.device,
                    dtype=self.pipeline.vae.dtype,
                )
                self.pipeline.vae.decode(dummy, return_dict=False)
            logger.info(
                "DiffusionSubmoduleRunner[decode]: warmup done in %.3fs",
                time.perf_counter() - t0,
            )
        except Exception:
            logger.warning(
                "DiffusionSubmoduleRunner[decode]: warmup failed (%.3fs), "
                "first real request may be slow",
                time.perf_counter() - t0,
                exc_info=True,
            )

    # Stage → pipeline method mapping
    _STAGE_HANDLERS: dict[str, str] = {
        "encode": "execute_encode",
        "encode_text": "execute_encode_text",
        "encode_image": "execute_encode_image",
        "decode": "execute_decode",
    }

    def _execute_stage(self, stage: str, req: OmniDiffusionRequest) -> list:
        """Dispatch a request to the appropriate pipeline method by stage name.

        Supported stages:
          - encode:       generic encode (text + image combined)
          - encode_text:  text encoder only
          - encode_image: image/VAE encoder only
          - decode:       VAE decode
        """
        handler_name = self._STAGE_HANDLERS.get(stage)
        if handler_name is None:
            raise ValueError(
                f"DiffusionSubmoduleRunner got unknown model_stage {stage!r}. "
                f"Valid stages: {list(self._STAGE_HANDLERS)}"
            )
        handler = getattr(self.pipeline, handler_name, None)
        if handler is None:
            raise AttributeError(
                f"Pipeline {type(self.pipeline).__name__} does not implement {handler_name!r} "
                f"(required by model_stage={stage!r})."
            )
        return handler([req])

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not req.prompts:
            raise ValueError("Cannot execute diffusion submodule runner on an empty request.")

        stage = getattr(self.od_config, "model_stage", None)

        sampling = req.sampling_params
        if sampling.generator is None and sampling.seed is not None:
            if sampling.generator_device is not None:
                gen_device = sampling.generator_device
            elif self.device.type == "cpu":
                gen_device = "cpu"
            else:
                gen_device = self.device
            sampling.generator = torch.Generator(device=gen_device).manual_seed(sampling.seed)

        with torch.inference_mode():
            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
            ):
                results = self._execute_stage(stage, req)

        if not results:
            return DiffusionOutput(error=f"{stage} stage returned no outputs")

        stage_out = results[0]
        stage_items = stage_out.items() if isinstance(stage_out, dict) else vars(stage_out).items()
        payload = {k: v for k, v in stage_items if k != "req_id" and v is not None and not (k == "metadata" and not v)}

        if stage == "decode":
            # Video pipelines use "video" key; image pipelines use "image" key.
            decoded = payload.get("video") or payload.get("image")
            return DiffusionOutput(output=decoded, multimodal_output=payload)

        # encode, encode_text, encode_image all return multimodal_output
        return DiffusionOutput(output=None, multimodal_output=payload)
