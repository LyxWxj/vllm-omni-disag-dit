# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    build_wan_scheduler,
    create_transformer_from_config,
    load_transformer_config,
    resolve_wan_flow_shift,
    resolve_wan_sample_solver,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt, OmniTokensPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)
DEBUG_PERF = False


def get_wan22_i2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video
        custom_output = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            custom_output["video_fps_multiplier"] = multiplier
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "custom_output": custom_output,
        }

    return post_process_func


def get_wan22_i2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for I2V: load and resize input image."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            # Skip image processing if additional_information already has
            # pre-computed data (e.g., denoise stage dummy run).
            additional_info = None
            if isinstance(prompt, dict):
                additional_info = prompt.get("additional_information")
            elif hasattr(prompt, "additional_information"):
                additional_info = getattr(prompt, "additional_information", None)
            if additional_info and ("latent_condition" in additional_info or "prompt_embeds" in additional_info):
                return request

            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                raise ValueError(
                    """No image is provided. This model requires an image to run.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"""Unsupported image format {raw_image.__class__}.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            # Calculate dimensions based on aspect ratio if not provided
            if request.sampling_params.height is None or request.sampling_params.width is None:
                # Default max area for 480P
                max_area = 480 * 832
                aspect_ratio = image.height / image.width

                # Calculate dimensions maintaining aspect ratio
                mod_value = 16  # Must be divisible by 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize image to target dimensions
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22I2VPipeline(
    nn.Module,
    SupportImageInput,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """
    Wan2.2 Image-to-Video Pipeline.

    Supports both Wan2.1-style I2V (with CLIP image embeddings) and
    Wan2.2-style I2V (with expand_timesteps for TI2V-5B).
    """

    # Fine-grained component registry for DAG stage separation.
    # text_encoder and image_encoder can run in parallel as separate stages.
    _component_registry: ClassVar[dict[str, set[str]]] = {
        "text_encoder":  {"tokenizer", "text_encoder"},
        "image_encoder": {"image_processor", "image_encoder", "vae"},
        "transformer":   {"transformer", "transformer_2"},
        "scheduler":     {"scheduler"},
        "vae_decoder":   {"vae"},
    }
    _default_stage_layout: ClassVar[dict[str, list[str]]] = {
        "encode_text":  ["text_encoder", "scheduler"],
        "encode_image": ["image_encoder"],
        "denoise":      ["transformer", "scheduler"],
        "decode":       ["vae_decoder"],
    }

    @classmethod
    def build_dummy_run_request(
        cls,
        od_config: OmniDiffusionConfig,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> OmniDiffusionRequest | None:
        """Build a stage-aware warmup request.

        For denoise stage, creates a request with pre-computed tensors in
        ``additional_information`` so the warmup doesn't need to call
        tokenizer/text_encoder/VAE.
        """
        stage = getattr(od_config, "model_stage", None) or "diffusion"
        if stage == "diffusion":
            return OmniDiffusionRequest(
                prompts=[{"prompt": "dummy run"}],
                request_id=DUMMY_DIFFUSION_REQUEST_ID,
                sampling_params=OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=max(2, num_inference_steps),
                    guidance_scale=0.0,
                    num_outputs_per_prompt=1,
                ),
            )
        if stage != "denoise":
            return None

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16

        # Build dummy tensors matching the denoise stage's expected inputs.
        tf_config = getattr(od_config, "tf_model_config", {})
        get_tf = getattr(tf_config, "get", None)
        out_channels = int(get_tf("out_channels", 16) if callable(get_tf) else 16)
        hidden_dim = int(get_tf("hidden_size", 4096) if callable(get_tf) else 4096)

        vae_scale_factor_spatial = 8
        vae_scale_factor_temporal = 4
        num_latent_frames = max(1, (num_inference_steps - 1) // vae_scale_factor_temporal + 1)
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial

        info: dict[str, Any] = {
            "prompt_embeds": torch.zeros((1, 1, hidden_dim), dtype=dtype),
            "negative_prompt_embeds": None,
            "latent_condition": torch.zeros(
                (1, out_channels, num_latent_frames, latent_height, latent_width), dtype=dtype
            ),
            "first_frame_mask": torch.zeros(
                (1, 1, num_latent_frames, latent_height, latent_width), dtype=dtype
            ),
            "guidance_low": 5.0,
            "guidance_high": 5.0,
        }
        return OmniDiffusionRequest(
            prompts=[
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    additional_information=info,
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            ],
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                num_frames=num_latent_frames * vae_scale_factor_temporal + 1,
            ),
        )

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.stage = getattr(od_config, "model_stage", None) or "diffusion"

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        dtype = self.dtype

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Get owned components for this stage
        owned_components = self.get_stage_components(self.stage)
        logger.info(
            "Wan22I2VPipeline.__init__: stage=%s, owned_components=%s",
            self.stage,
            owned_components,
        )

        # Set up weights sources for transformer(s) only if needed
        owns_transformer = "transformer" in owned_components
        self.weights_sources = (
            [
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                ),
            ]
            if owns_transformer
            else []
        )
        # When transformer is not needed (e.g., encode_image stage), tell the
        # weight loader that weights were loaded during model initialization
        # so it skips the strict weight check.
        self.weights_loaded_by_model_init = not owns_transformer

        # Load model_index.json to detect available components
        try:
            model_index = _load_json(model, "model_index.json", local_files_only)
        except Exception:
            model_index = {}

        # Read expand_timesteps from model_index.json (for TI2V-5B style)
        self.expand_timesteps = model_index.get("expand_timesteps", False)

        # Check if this is a two-stage model (MoE with transformer_2)
        # transformer_2 may exist in model_index.json but be [null, null]
        self.has_transformer_2 = (
            "transformer_2" in model_index
            and model_index["transformer_2"][0] is not None
        )

        if self.has_transformer_2 and owns_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # Text encoder (only if needed by this stage)
        if "text_encoder" in owned_components:
            self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
            ).to(self.device)
        else:
            self.tokenizer = None
            self.text_encoder = None

        # Image encoder (CLIP) - optional, for Wan2.1-style I2V
        self.has_image_encoder = "image_encoder" in model_index and model_index["image_encoder"][0] is not None

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        subfolders = ["tokenizer", "text_encoder", "vae"]
        if self.has_image_encoder:
            subfolders.extend(["image_processor", "image_encoder"])
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)

        if self.has_image_encoder and "image_encoder" in owned_components:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model, subfolder="image_processor", local_files_only=local_files_only
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model, subfolder="image_encoder", torch_dtype=dtype, local_files_only=local_files_only
            ).to(self.device)
        else:
            self.image_processor = None
            self.image_encoder = None

        # VAE - load full VAE for any stage that needs it
        if "vae" in owned_components:
            self.vae = DistributedAutoencoderKLWan.from_pretrained(
                model, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only
            ).to(self.device)
            self.vae_encoder = None
            self.vae_decoder = None
        else:
            self.vae = None
            self.vae_encoder = None
            self.vae_decoder = None

        # Transformers (weights loaded via load_weights)
        if owns_transformer:
            # Load config from model directory or HF Hub to get correct in_channels for I2V models
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = create_transformer_from_config(
                transformer_config,
                quant_config=od_config.quantization_config,
            )
            if self.has_transformer_2:
                transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
                t2_quant = transformer_2_config.get("quantization_config")
                if isinstance(t2_quant, dict) and "quant_method" in t2_quant:
                    from vllm_omni.quantization.factory import build_quant_config

                    method = t2_quant["quant_method"]
                    kwargs = {k: v for k, v in t2_quant.items() if k != "quant_method"}
                    t2_quant = build_quant_config(method, **kwargs)
                else:
                    t2_quant = None
                self.transformer_2 = create_transformer_from_config(
                    transformer_2_config,
                    quant_config=t2_quant,
                )
            else:
                self.transformer_2 = None
        else:
            self.transformer = None
            self.transformer_2 = None

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        if "scheduler" in owned_components:
            self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)
        else:
            self.scheduler = None

        # VAE scale factors
        if self.vae is not None:
            self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
            self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8
        else:
            self.vae_scale_factor_temporal = 4
            self.vae_scale_factor_spatial = 8

        # MoE boundary ratio for two-stage denoising
        self.boundary_ratio = od_config.boundary_ratio

        # expand_timesteps is already set from model_index.json above

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        image_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        condition: torch.Tensor,
        first_frame_mask: torch.Tensor,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t

                # Select model and guidance scale based on timestep
                current_model = self.transformer
                current_guidance_scale = guidance_low
                if boundary_timestep is not None and t < boundary_timestep and self.transformer_2 is not None:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_high

                set_forward_context_denoise_step_idx(step_idx)

                # Prepare latent input
                if self.expand_timesteps:
                    # TI2V-5B style: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps for each patch
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # Wan2.1 style: concatenate condition with latents
                    latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                # Prepare kwargs for positive and negative predictions
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": image_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_image": image_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                # Predict noise with automatic CFG parallel handling
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        return latents

    def encode_image(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode image using CLIP image encoder."""
        device = device or self.device
        if self.image_encoder is None:
            raise ValueError("Image encoder not available for this model.")

        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    @classmethod
    def build_dummy_run_request(
        cls,
        od_config: OmniDiffusionConfig,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> OmniDiffusionRequest | None:
        """Build a stage-aware warmup request.

        Returns None for non-denoise stages (encode_text, encode_image, decode)
        because they use the submodule path which doesn't need DiffusionEngine
        dummy run.  For the denoise stage, returns a pre-built request with
        virtual tensors to avoid requiring text_encoder/vae.
        """
        stage = getattr(od_config, "model_stage", None) or "diffusion"
        if stage != "diffusion" and stage != "denoise":
            # encode_text, encode_image, decode use submodule path — no dummy run needed
            return None

        if stage == "diffusion":
            # Single-stage mode: use default dummy request
            return OmniDiffusionRequest(
                prompts=[{"prompt": "dummy run"}],
                request_id=DUMMY_DIFFUSION_REQUEST_ID,
                sampling_params=OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=max(2, num_inference_steps),
                    guidance_scale=0.0,
                    num_outputs_per_prompt=1,
                ),
            )

        # Denoise stage: build a pre-built request with virtual tensors
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16

        # Calculate latent dimensions
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8
        num_frames = 81  # Default for Wan2.2
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial

        # Get transformer config
        tf_config = getattr(od_config, "tf_model_config", None)
        get_tf_config = getattr(tf_config, "get", None)
        out_channels = int(get_tf_config("out_channels", 16) if callable(get_tf_config) else 16)

        # Build virtual tensors for denoise stage
        prompt_embeds = torch.zeros((1, 1, 4096), dtype=dtype)  # [batch, seq_len, dim]
        latents = torch.randn(
            (1, out_channels, num_latent_frames, latent_height, latent_width),
            dtype=dtype,
        )
        timesteps = torch.linspace(1000, 0, num_inference_steps, dtype=torch.float32)

        # For expand_timesteps mode, we need latent_condition and first_frame_mask
        latent_condition = torch.randn(
            (1, out_channels, num_latent_frames, latent_height, latent_width),
            dtype=dtype,
        )
        first_frame_mask = torch.zeros(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype)
        first_frame_mask[:, :, 1:, :, :] = 1.0  # Mark non-first frames for denoising

        info: dict[str, Any] = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
            "latents": latents,
            "timesteps": timesteps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "expand_timesteps": True,
            "guidance_low": 5.0,
            "guidance_high": 5.0,
            "latent_condition": latent_condition,
            "first_frame_mask": first_frame_mask,
        }

        return OmniDiffusionRequest(
            prompts=[
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    additional_information=info,
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            ],
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
            ),
        )

    # ---- Disaggregated stage execution (submodule path) ----

    def execute_encode_text(self, requests: list[OmniDiffusionRequest]) -> list[dict[str, Any]]:
        """Run text encoding for the encode_text submodule stage.

        Called by :class:`DiffusionSubmoduleRunner` when ``model_stage == "encode_text"``.
        """
        outputs: list[dict[str, Any]] = []
        for req in requests:
            sampling = req.sampling_params
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")

            if prompt is None:
                raise ValueError("Prompt is required for text encoding.")

            guidance_low = sampling.guidance_scale if sampling.guidance_scale_provided else 5.0
            do_cfg = guidance_low > 1.0

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=sampling.num_outputs_per_prompt or 1,
                max_sequence_length=sampling.max_sequence_length or 512,
            )
            outputs.append({
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_low": guidance_low,
                "guidance_high": sampling.guidance_scale_2 if sampling.guidance_scale_2 is not None else guidance_low,
            })
        return outputs

    def execute_encode_image(self, requests: list[OmniDiffusionRequest]) -> list[dict[str, Any]]:
        """Run image encoding for the encode_image submodule stage.

        Called by :class:`DiffusionSubmoduleRunner` when ``model_stage == "encode_image"``.

        For TI2V-5B style (expand_timesteps), uses VAE encoder to encode
        the image into latent condition.  For Wan2.1-style I2V, uses CLIP
        image encoder.
        """
        outputs: list[dict[str, Any]] = []
        for req in requests:
            sampling = req.sampling_params
            # Get image from request
            multi_modal_data = (
                req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
            )
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if raw_image is None:
                raise ValueError("Image is required for image encoding.")
            if isinstance(raw_image, list):
                raw_image = raw_image[0]
            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image).convert("RGB")
            else:
                image = raw_image

            height = sampling.height or 480
            width = sampling.width or 832
            num_frames = sampling.num_frames or 81

            if self.expand_timesteps:
                # TI2V-5B style: use VAE encoder
                if self.vae is None:
                    raise ValueError("VAE not available for image encoding.")

                from diffusers.video_processor import VideoProcessor

                video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

                if isinstance(image, PIL.Image.Image):
                    image_tensor = TF.to_tensor(image).to(self.device)
                    image_tensor = video_processor.preprocess(image_tensor, height=height, width=width)
                else:
                    image_tensor = image
                image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

                # Prepare condition: first frame is image, rest is zeros
                image_tensor = image_tensor.unsqueeze(2)  # [batch, channels, 1, height, width]

                # Adjust num_frames for VAE temporal scaling
                if num_frames % self.vae_scale_factor_temporal != 1:
                    num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
                num_frames = max(num_frames, 1)

                # Create video condition (first frame = image, rest = zeros)
                video_condition = torch.cat(
                    [image_tensor, image_tensor.new_zeros(image_tensor.shape[0], image_tensor.shape[1], num_frames - 1, height, width)],
                    dim=2,
                )
                video_condition = video_condition.to(device=self.device, dtype=self.dtype)

                # Encode through VAE
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")

                # Create first_frame_mask (1 for frames to denoise, 0 for condition)
                num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
                latent_height = height // self.vae_scale_factor_spatial
                latent_width = width // self.vae_scale_factor_spatial
                first_frame_mask = torch.zeros(1, 1, num_latent_frames, latent_height, latent_width, device=self.device)
                first_frame_mask[:, :, 1:, :, :] = 1.0  # Mark non-first frames for denoising

                outputs.append({
                    "latent_condition": latent_condition,
                    "first_frame_mask": first_frame_mask,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "expand_timesteps": True,
                })
            else:
                # Wan2.1-style I2V: use CLIP image encoder
                if self.has_image_encoder and self.transformer is not None and self.transformer.config.image_dim is not None:
                    image_embeds = self.encode_image(image, self.device)
                    outputs.append({"image_embeds": image_embeds, "expand_timesteps": False})
                else:
                    outputs.append({"image_embeds": None, "expand_timesteps": False})
        return outputs

    def execute_decode(self, requests: list[OmniDiffusionRequest]) -> list[dict[str, Any]]:
        """Run VAE decode for the decode submodule stage.

        Called by :class:`DiffusionSubmoduleRunner` when ``model_stage == "decode"``.
        Receives latents from the denoise stage and decodes them to video frames.
        """
        if self.vae is None:
            raise ValueError("VAE not available for decoding.")

        outputs: list[dict[str, Any]] = []
        for req in requests:
            # Extract latents and metadata from additional_information
            additional_info = {}
            if req.prompts:
                first_prompt = req.prompts[0]
                if isinstance(first_prompt, dict):
                    additional_info = first_prompt.get("additional_information", {}) or {}
                elif hasattr(first_prompt, "additional_information"):
                    additional_info = getattr(first_prompt, "additional_information", None) or {}

            latents = additional_info.get("latents")
            if latents is None:
                raise ValueError("Latents are required for decode stage.")

            height = additional_info.get("height", 480)
            width = additional_info.get("width", 832)
            num_frames = additional_info.get("num_frames", 81)

            # Denormalize latents
            latents = latents.to(device=self.device, dtype=self.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # Decode
            video = self.vae.decode(latents, return_dict=False)[0]
            video = video.to(torch.float32)

            outputs.append({
                "video": video,
                "height": height,
                "width": width,
                "num_frames": num_frames,
            })
        return outputs

    @staticmethod
    def _stage_payload_from_prompts(prompts: list[Any] | None) -> dict[str, Any]:
        """Extract ``additional_information`` from the first prompt dict."""
        if not prompts:
            raise ValueError("Wan22I2V stage request is missing prompts.")
        first = prompts[0]
        if not isinstance(first, dict) or not first.get("additional_information"):
            raise ValueError("Wan22I2V stage request is missing additional_information.")
        return first["additional_information"]

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        image: PIL.Image.Image | torch.Tensor | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float | tuple[float, float] = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: PIL.Image.Image | torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract pre-computed data from additional_information early
        # (disaggregated mode: denoise stage receives data from encode stages).
        additional_info = {}
        if req.prompts:
            first_prompt = req.prompts[0]
            if isinstance(first_prompt, dict):
                additional_info = first_prompt.get("additional_information", {}) or {}
            elif hasattr(first_prompt, "additional_information"):
                additional_info = getattr(first_prompt, "additional_information", None) or {}

        if prompt_embeds is None and "prompt_embeds" in additional_info:
            prompt_embeds = additional_info["prompt_embeds"]
            negative_prompt_embeds = additional_info.get("negative_prompt_embeds")
            guidance_low = additional_info.get("guidance_low", guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0])
            guidance_high = additional_info.get("guidance_high", guidance_low)

        if image_embeds is None and "image_embeds" in additional_info:
            image_embeds = additional_info["image_embeds"]

        latent_condition = additional_info.get("latent_condition")
        first_frame_mask = additional_info.get("first_frame_mask")

        # Get parameters from request or arguments
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        # In disaggregated mode (denoise stage), prompt_embeds must come from
        # the encode_text stage via additional_information. If not available,
        # raise a clear error instead of trying to encode (which would fail
        # because tokenizer/text_encoder are not loaded).
        if prompt is None and prompt_embeds is None:
            if self.stage == "denoise":
                raise ValueError(
                    "Denoise stage requires prompt_embeds from encode_text stage "
                    "via additional_information, but none was provided. "
                    "Ensure the encode_text stage runs first."
                )
            raise ValueError("Prompt or prompt_embeds is required for Wan2.2 generation.")

        # Get image from request (skip if we have pre-computed latent_condition)
        if image is None and latent_condition is None:
            multi_modal_data = (
                req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
            )
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if raw_image is None:
                raise ValueError("Image is required for I2V generation.")
            if isinstance(raw_image, list):
                if len(raw_image) > 1:
                    logger.warning(
                        """Received a list of image. Only a single image is supported by this model."""
                        """Taking only the first image for now."""
                    )
                raw_image = raw_image[0]
            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image)
            else:
                image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames or frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # Respect per-request guidance_scale when explicitly provided.
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        # Handle guidance scales
        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for I2V generation. using default value 0.875")

        # Validate inputs (skip if we have pre-computed latent_condition)
        if latent_condition is None:
            self.check_inputs(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                height=height,
                width=width,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        # Adjust num_frames to be compatible with VAE temporal scaling
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        # Generator setup
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Move pre-computed tensors to device
        if latent_condition is not None:
            latent_condition = latent_condition.to(device=device, dtype=dtype)
        if first_frame_mask is not None:
            first_frame_mask = first_frame_mask.to(device=device, dtype=dtype)

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start

        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        batch_size = prompt_embeds.shape[0]

        if DEBUG_PERF:
            _t_img_enc_start = time.perf_counter()
        # Skip image encoding if we have pre-computed latent_condition
        if latent_condition is not None:
            image_embeds = None  # Not needed when using latent_condition
        elif self.has_image_encoder and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(dtype)
        else:
            if image_embeds is not None:
                image_embeds = image_embeds.repeat(batch_size, 1, 1)
                image_embeds = image_embeds.to(dtype)
            else:
                image_embeds = None

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_img_enc_ms = (time.perf_counter() - _t_img_enc_start) * 1000

        sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        # Prepare latents (use out_channels=16 for VAE latent, not in_channels=36)
        num_channels_latents = self.transformer.config.out_channels

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()

        # In disaggregated mode, use pre-computed latent_condition and first_frame_mask
        # from encode_image stage if available.
        if latent_condition is not None and first_frame_mask is not None:
            # Use pre-computed values from encode_image stage.
            # latent_condition and first_frame_mask already have the correct
            # temporal dimension from VAE encoding — use it directly.
            num_latent_frames = latent_condition.shape[2]
            latent_height = latent_condition.shape[3]
            latent_width = latent_condition.shape[4]
            shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
            # Expand batch dimension if needed
            condition = latent_condition.expand(batch_size, -1, -1, -1, -1).contiguous()
            first_frame_mask = first_frame_mask.expand(batch_size, -1, -1, -1, -1).contiguous()
        else:
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            if isinstance(image, PIL.Image.Image):
                image = TF.to_tensor(image).to(device)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image
            image_tensor = image_tensor.to(device=device, dtype=torch.float32)

            # Handle last_image if provided
            if last_image is not None:
                if isinstance(last_image, PIL.Image.Image):
                    image = TF.to_tensor(last_image).to(device)
                    last_image_tensor = video_processor.preprocess(last_image, height=height, width=width)
                else:
                    last_image_tensor = last_image
                last_image_tensor = last_image_tensor.to(device=device, dtype=torch.float32)
            else:
                last_image_tensor = None

            latents, condition, first_frame_mask = self.prepare_latents(
                image=image_tensor,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
                last_image=last_image_tensor,
            )

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            condition=condition,
            first_frame_mask=first_frame_mask,
        )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For expand_timesteps mode, blend final latents with condition
        if self.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()

        if output_type == "latent" or self.vae is None:
            output = latents
        else:
            latents = latents.to(self.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_img_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, ImageEncoding=%.2f ms, "
                    "LatentPreparation=%.2f ms, Denoising=%.2f ms (%d steps), "
                    "Decoding=%.2f ms, StagesSum=%.2f ms, PipelineWall=%.2f ms, "
                    "Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_img_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        # For disaggregated denoise stage, pass latents in multimodal_output
        # so the decode stage can receive them via denoise_to_decode processor.
        # When vae is None (denoise stage), output is always latents.
        mm_output: dict[str, Any] = {}
        if self.stage == "denoise" or (self.vae is None and isinstance(output, torch.Tensor)):
            mm_output = {
                "latents": output,
                "height": height,
                "width": width,
                "num_frames": frame_num,
                "output_type": "np",
            }

        return DiffusionOutput(
            output=output if self.stage != "denoise" else None,
            multimodal_output=mm_output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode text prompts using T5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare latents for I2V generation.

        Returns:
            latents: Initial noise latents
            condition: Encoded image condition (concatenated with mask for non-expand mode)
            first_frame_mask: Mask for the first frame (1 for frames to denoise, 0 for condition)
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Prepare image condition
        image = image.unsqueeze(2)  # [batch, channels, 1, height, width]

        if self.expand_timesteps:
            # TI2V-5B style: only use first frame as condition
            video_condition = image
        elif last_image is None:
            # Pad with zeros for remaining frames
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            # First and last frame conditioning
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )

        video_condition = video_condition.to(device=device, dtype=self.dtype)

        # Encode through VAE
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        # Normalize latents
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = (latent_condition - latents_mean) * latents_std
        latent_condition = latent_condition.to(dtype)

        if self.expand_timesteps:
            # TI2V-5B style: create mask where first frame is 0 (condition), rest is 1 (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        # Wan2.1 style: create mask and concatenate with condition
        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=latent_condition.device
        )

        if last_image is None:
            mask_lat_size[:, :, 1:] = 0
        else:
            mask_lat_size[:, :, 1 : num_frames - 1] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        # Concatenate mask with condition for channel dimension
        condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        # For non-expand mode, first_frame_mask is not used in the same way
        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)

        return latents, condition, first_frame_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if image is None and image_embeds is None:
            raise ValueError("Provide either `image` or `image_embeds`. Cannot leave both undefined.")

        if image is not None and image_embeds is not None:
            raise ValueError("Cannot forward both `image` and `image_embeds`. Please provide only one.")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please provide only one."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# DMD2-distilled variant
# ---------------------------------------------------------------------------


class WanI2VDMD2Pipeline(DMD2PipelineMixin, Wan22I2VPipeline):
    """Wan 2.x I2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
