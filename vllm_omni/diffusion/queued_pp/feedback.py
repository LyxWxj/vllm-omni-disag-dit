# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents


def queued_feedback_device_group(pp_group: Any) -> Any:
    """Use the dedicated reverse PP2 communicator for latent feedback."""
    direction_groups = getattr(pp_group, "device_groups", None)
    if pp_group.world_size == 2 and direction_groups:
        return direction_groups[1]
    return pp_group.device_group


def prune_queued_latent_sends(pipeline: Any) -> None:
    pending_work = pipeline._pp_send_work
    pending_tensors = getattr(pipeline, "_pp_send_tensors", [])
    active_work = []
    active_tensors = []
    for index, work in enumerate(pending_work):
        if work.is_completed():
            continue
        active_work.append(work)
        if index < len(pending_tensors):
            active_tensors.append(pending_tensors[index])
    pipeline._pp_send_work = active_work
    pipeline._pp_send_tensors = active_tensors


def sync_queued_latent_sends(pipeline: Any) -> None:
    for handle in pipeline._pp_send_work:
        handle.wait()
    pipeline._pp_send_work = []
    pipeline._pp_send_tensors = []


def send_queued_latents(pipeline: Any, pp_group: Any, latents: torch.Tensor) -> None:
    handle = torch.distributed.isend(
        latents,
        dst=pp_group.ranks[0],
        group=queued_feedback_device_group(pp_group),
    )
    pipeline._pp_send_work.append(handle)
    pipeline._pp_send_tensors = [*getattr(pipeline, "_pp_send_tensors", []), latents]


def receive_queued_latents(pipeline: Any, pp_group: Any, latents: torch.Tensor | AsyncLatents) -> AsyncLatents:
    template = latents._tensor_dict["latents"] if isinstance(latents, AsyncLatents) else latents
    received = torch.empty_like(template)
    feedback_group = queued_feedback_device_group(pp_group)
    if received.is_cuda and pp_group.world_size == 2:
        stream = getattr(pipeline, "_queued_feedback_recv_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=received.device)
            pipeline._queued_feedback_recv_stream = stream
        # Serialize reuse of a cached receive buffer without blocking compute.
        stream.wait_stream(torch.cuda.current_stream(received.device))
        with torch.cuda.stream(stream):
            handle = torch.distributed.irecv(
                received,
                src=pp_group.ranks[-1],
                group=feedback_group,
            )
            received.record_stream(stream)

        def retain_for_consumer() -> None:
            received.record_stream(torch.cuda.current_stream(received.device))

        postprocess = [retain_for_consumer]
    else:
        handle = torch.distributed.irecv(
            received,
            src=pp_group.ranks[-1],
            group=feedback_group,
        )
        postprocess = []
    return AsyncLatents({"latents": received}, [handle], postprocess)
