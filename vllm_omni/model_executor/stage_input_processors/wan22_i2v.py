# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Wan2.2 I2V disaggregated pipeline."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _read_mm(output: Any, stage_label: str, req_idx: int) -> dict[str, Any]:
    mm = getattr(output, "multimodal_output", None)
    if not mm or not isinstance(mm, dict):
        raise RuntimeError(
            f"[wan22_i2v.{stage_label}] upstream req#{req_idx} is missing multimodal_output "
            f"(got {type(mm).__name__})."
        )
    return mm


def _require(mm: dict[str, Any], key: str, stage_label: str, req_idx: int) -> Any:
    if key not in mm or mm[key] is None:
        raise RuntimeError(
            f"[wan22_i2v.{stage_label}] upstream req#{req_idx} missing required key {key!r}; "
            f"have {sorted(mm.keys())}."
        )
    return mm[key]


def encode_to_denoise(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Merge text_encoder and image_encoder outputs into denoise inputs.

    source_outputs[0] = text_encoder output (prompt_embeds, negative_prompt_embeds, guidance)
    source_outputs[1] = image_encoder output (image_embeds or latent_condition)

    Both outputs are combined into a single OmniTokensPrompt for the denoise stage.
    Supports both CLIP image encoder (Wan2.1-style) and VAE encoder (TI2V-5B style).
    """
    denoise_inputs: list[OmniTokensPrompt] = []

    # source_outputs may come from multiple input_sources (fan-in).
    # We need to identify which output is from text_encoder and which is from image_encoder.
    # The outputs are ordered by input_sources: [stage_0_output, stage_1_output]
    text_output = source_outputs[0] if len(source_outputs) > 0 else None
    image_output = source_outputs[1] if len(source_outputs) > 1 else None

    # Extract text encoder outputs
    text_mm = _read_mm(text_output, "encode_to_denoise_text", 0) if text_output else {}
    prompt_embeds = text_mm.get("prompt_embeds")
    negative_prompt_embeds = text_mm.get("negative_prompt_embeds")
    guidance_low = text_mm.get("guidance_low", 5.0)
    guidance_high = text_mm.get("guidance_high", 5.0)

    # Extract image encoder outputs
    image_mm = _read_mm(image_output, "encode_to_denoise_image", 1) if image_output else {}

    # Check which mode is being used
    expand_timesteps = image_mm.get("expand_timesteps", False)

    info: dict[str, Any] = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "guidance_low": guidance_low,
        "guidance_high": guidance_high,
        "expand_timesteps": expand_timesteps,
    }

    if expand_timesteps:
        # TI2V-5B style: VAE encoder outputs
        info["latent_condition"] = image_mm.get("latent_condition")
        info["first_frame_mask"] = image_mm.get("first_frame_mask")
        info["height"] = image_mm.get("height")
        info["width"] = image_mm.get("width")
        info["num_frames"] = image_mm.get("num_frames")
    else:
        # Wan2.1-style I2V: CLIP image encoder outputs
        info["image_embeds"] = image_mm.get("image_embeds")

    denoise_inputs.append(
        OmniTokensPrompt(
            prompt_token_ids=[],
            additional_information=info,
            multi_modal_data=None,
            mm_processor_kwargs=None,
        )
    )
    return denoise_inputs


def denoise_to_decode(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Extract denoise latents for the decode stage."""
    decode_inputs: list[OmniTokensPrompt] = []
    for i, den_out in enumerate(source_outputs):
        mm = _read_mm(den_out, "denoise_to_decode", i)
        info: dict[str, Any] = {
            "latents": _require(mm, "latents", "denoise_to_decode", i),
            "height": mm.get("height"),
            "width": mm.get("width"),
            "num_frames": mm.get("num_frames"),
            "output_type": mm.get("output_type", "np"),
        }
        decode_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[],
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return decode_inputs
