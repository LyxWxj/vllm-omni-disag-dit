#!/usr/bin/env python3
"""Test TeaCache with step-wise execution mode.

Usage:
    python test_teacache_stepwise.py --model ~/models/Qwen-Image
"""

import argparse
import time

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    parser = argparse.ArgumentParser(description="Test TeaCache with step-wise execution")
    parser.add_argument("--model", type=str, default="~/models/Qwen-Image", help="Model path")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cache-backend", type=str, default="tea_cache", help="Cache backend")
    parser.add_argument("--rel-l1-thresh", type=float, default=0.2, help="TeaCache threshold")
    args = parser.parse_args()

    print(f"=== Testing TeaCache with step-wise execution ===")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}")
    print(f"Cache backend: {args.cache_backend}")
    print(f"rel_l1_thresh: {args.rel_l1_thresh}")
    print()

    # Initialize with TeaCache enabled
    omni = Omni(
        model=args.model,
        cache_backend=args.cache_backend,
        cache_config={
            "rel_l1_thresh": args.rel_l1_thresh,
        },
        step_execution=True,  # Enable step-wise execution
    )

    prompt = "A beautiful sunset over the ocean, with golden clouds and calm waves"
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.steps,
        height=512,
        width=512,
    )

    print(f"Generating with prompt: {prompt}")
    print()

    start_time = time.time()
    outputs = omni.generate(prompt, sampling_params)
    elapsed = time.time() - start_time

    print()
    print(f"=== Generation complete ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Output type: {type(outputs)}")
    if outputs:
        print(f"Output count: {len(outputs)}")


if __name__ == "__main__":
    main()
