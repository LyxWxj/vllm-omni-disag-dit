# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for TeaCache backend.

This test verifies that TeaCache acceleration works correctly with diffusion models.
It uses minimal settings to keep test time short for CI.
"""

import asyncio
import uuid

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# Use random weights model for testing
models = ["riverclouds/qwen_image_random"]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_teacache(model_name: str):
    """Test TeaCache backend with diffusion model."""
    # Configure TeaCache with default settings for fast testing
    cache_config = {
        "rel_l1_thresh": 0.2,  # Default threshold
    }
    with OmniRunner(
        model_name,
        cache_backend="tea_cache",
        cache_config=cache_config,
    ) as runner:
        # Use minimal settings for fast testing
        height = 256
        width = 256
        num_inference_steps = 4  # Minimal steps for fast test

        outputs = runner.omni.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,  # Single output for speed
            ),
        )
        # Extract images from request_output['images']
        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")

        images = req_out.images

        # Verify generation succeeded
        assert images is not None
        assert len(images) == 1
        # Check image size
        assert images[0].width == width
        assert images[0].height == height


def _extract_images(output: OmniRequestOutput) -> list:
    if output.images:
        return output.images
    request_output = getattr(output, "request_output", None)
    if request_output is not None and request_output.images:
        return request_output.images
    return []


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("model_name", models)
def test_teacache_step_execution_two_gpu_interleaving(model_name: str):
    """Run two concurrent TeaCache requests through one two-rank SP cohort."""

    async def _run() -> None:
        omni = AsyncOmni(
            model=model_name,
            cache_backend="tea_cache",
            cache_config={"rel_l1_thresh": 0.2},
            step_execution=True,
            max_num_seqs=2,
            parallel_config=DiffusionParallelConfig(ulysses_degree=2),
        )
        release = asyncio.Event()
        arrival_lock = asyncio.Lock()
        num_arrived = 0

        async def _generate(prompt: str, request_id: str, seed: int) -> OmniRequestOutput:
            nonlocal num_arrived
            async with arrival_lock:
                num_arrived += 1
                if num_arrived == 2:
                    release.set()
            await release.wait()

            last_output = None
            sampling_params = OmniDiffusionSamplingParams(
                height=256,
                width=256,
                num_inference_steps=4,
                guidance_scale=0.0,
                seed=seed,
                num_outputs_per_prompt=1,
            )
            async for output in omni.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=[sampling_params],
            ):
                last_output = output
            if last_output is None:
                raise RuntimeError(f"No output received for request {request_id}")
            return last_output

        request_ids = [f"teacache-step-{i}-{uuid.uuid4().hex[:8]}" for i in range(2)]
        try:
            outputs = await asyncio.gather(
                _generate("a red cube on a white table", request_ids[0], 42),
                _generate("a blue sphere on a black table", request_ids[1], 43),
            )
        finally:
            omni.shutdown()

        assert num_arrived == 2
        assert {output.request_id for output in outputs} == set(request_ids)
        for output in outputs:
            images = _extract_images(output)
            assert len(images) == 1
            assert images[0].size == (256, 256)

    asyncio.run(_run())
