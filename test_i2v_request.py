#!/usr/bin/env python3
"""Send an I2V request to the disaggregated Wan2.2 service and save the output video."""

import argparse
import asyncio
import os
import time

import aiohttp


async def send_i2v_request(
    base_url: str,
    image_path: str,
    prompt: str,
    output_path: str,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 40,
    seed: int = 42,
):
    api_url = f"{base_url}/v1/videos"

    # Build multipart form
    form = aiohttp.FormData()
    form.add_field("prompt", prompt)
    form.add_field("height", str(height))
    form.add_field("width", str(width))
    form.add_field("num_frames", str(num_frames))
    form.add_field("num_inference_steps", str(num_inference_steps))
    form.add_field("seed", str(seed))

    # Attach input image
    image_file = open(image_path, "rb")
    form.add_field(
        "input_reference",
        image_file,
        filename=os.path.basename(image_path),
        content_type="application/octet-stream",
    )

    async with aiohttp.ClientSession() as session:
        # Step 1: Submit job
        print(f"Submitting I2V request to {api_url} ...")
        t0 = time.perf_counter()
        async with session.post(api_url, data=form) as resp:
            if resp.status != 200:
                print(f"ERROR: HTTP {resp.status}: {await resp.text()}")
                return
            resp_json = await resp.json()
            job_id = resp_json.get("id")
            job_status = resp_json.get("status")
            print(f"Job submitted: id={job_id}, status={job_status}")

        if not job_id:
            print("ERROR: No job id returned")
            return

        # Step 2: Poll until completed
        job_url = f"{api_url}/{job_id}"
        poll_interval = 2.0
        timeout = 600.0
        deadline = time.perf_counter() + timeout

        while job_status not in {"completed", "failed"}:
            await asyncio.sleep(poll_interval)
            async with session.get(job_url) as poll_resp:
                if poll_resp.status != 200:
                    print(f"Poll error: HTTP {poll_resp.status}")
                    return
                poll_json = await poll_resp.json()
                job_status = poll_json.get("status")

            elapsed = time.perf_counter() - t0
            print(f"  [{elapsed:.1f}s] status={job_status}")

            if time.perf_counter() >= deadline:
                print(f"ERROR: Timed out after {timeout}s")
                return

        if job_status == "failed":
            print(f"ERROR: Job failed: {poll_json}")
            return

        # Step 3: Download video
        content_url = f"{job_url}/content"
        async with session.get(content_url) as content_resp:
            if content_resp.status != 200:
                print(f"Download error: HTTP {content_resp.status}")
                return
            video_bytes = await content_resp.read()

        elapsed = time.perf_counter() - t0
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved to {output_path} ({len(video_bytes)} bytes, {elapsed:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I2V request to Wan2.2 disaggregated service")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Service base URL")
    parser.add_argument("--image", default="models/Wan2.2-TI2V-5B-Diffusers/examples/i2v_input.JPG", help="Input image path")
    parser.add_argument("--prompt", default="A car is moving on the road", help="Text prompt")
    parser.add_argument("--output", default="output_i2v.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(
        send_i2v_request(
            base_url=args.base_url,
            image_path=args.image,
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
            seed=args.seed,
        )
    )
