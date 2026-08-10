# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU system coverage for request-swappable Cache-DiT step execution."""

import asyncio
import uuid

import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


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
def test_cache_dit_step_execution_two_gpu_concurrent_requests():
    """Run two Cache-DiT trajectories through one two-rank SP cohort."""

    async def _run() -> None:
        omni = AsyncOmni(
            model="riverclouds/qwen_image_random",
            cache_backend="cache_dit",
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

        request_ids = [f"cache-dit-step-{index}-{uuid.uuid4().hex[:8]}" for index in range(2)]
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
