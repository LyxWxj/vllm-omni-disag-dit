# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image 2.1 diffusion stage topologies."""

from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_QWEN_IMAGE_21_MODEL_ARCH = "QwenImage21Pipeline"
_DIFFUSION_HANDOFF = "vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff"

_ENCODE_PAYLOAD_KEYS = (
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "image_pad_mask",
    "negative_image_pad_mask",
    "image_latents",
    "input_image_sizes",
)
_ENCODE_OPTIONAL_PAYLOAD_KEYS = (
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "negative_image_pad_mask",
    "image_latents",
    "input_image_sizes",
)


QWEN_IMAGE_21_PIPELINE = PipelineConfig(
    model_type="qwen_image_21",
    model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
    diffusers_class_name="QwenImage21Pipeline",
    default_deploy_config_name="qwen_image_21.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.FULL,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            engine_output_type="image",
        ),
    ),
)


QWEN_IMAGE_21_EG_PIPELINE = PipelineConfig(
    model_type="qwen_image_21_eg",
    model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
    diffusers_class_name="QwenImage21Pipeline",
    default_deploy_config_name="qwen_image_21_eg.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_output_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise_decode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE_DECODE,
            stage_input_payload_keys=_ENCODE_PAYLOAD_KEYS,
            stage_input_optional_payload_keys=_ENCODE_OPTIONAL_PAYLOAD_KEYS,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)


QWEN_IMAGE_21_EGD_PIPELINE = PipelineConfig(
    model_type="qwen_image_21_egd",
    model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
    diffusers_class_name="QwenImage21Pipeline",
    default_deploy_config_name="qwen_image_21_egd.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_output_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE,
            stage_input_payload_keys=_ENCODE_PAYLOAD_KEYS,
            stage_input_optional_payload_keys=_ENCODE_OPTIONAL_PAYLOAD_KEYS,
            stage_output_payload_keys=("latents",),
            input_sources=(0,),
            final_output=False,
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            engine_output_type="custom",
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DECODE,
            stage_input_payload_keys=("latents",),
            input_sources=(1,),
            final_output=True,
            final_output_type="image",
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)
