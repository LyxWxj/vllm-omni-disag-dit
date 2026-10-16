# SPDX-License-Identifier: Apache-2.0
"""Closed request-batch pipeline parallel execution primitives."""

from vllm_omni.diffusion.request_pipeline.task import (
    RequestPipelineBatchState,
    RequestPipelineTask,
    build_request_pipeline_tasks,
)
from vllm_omni.diffusion.request_pipeline.trace import trace_request_pipeline_event
from vllm_omni.diffusion.request_pipeline.transport import RequestPipelineTransport

__all__ = [
    "RequestPipelineBatchState",
    "RequestPipelineTask",
    "RequestPipelineTransport",
    "build_request_pipeline_tasks",
    "trace_request_pipeline_event",
]
