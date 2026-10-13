# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Private runtime for opt-in queued diffusion pipeline parallelism."""

from vllm_omni.diffusion.queued_pp.queue_config import QueuedPipelineConfig

__all__ = ["QueuedPipelineConfig"]
