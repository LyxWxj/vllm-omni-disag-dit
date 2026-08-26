# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.distributed.pipeline_runtime import PipelineTensorSpec
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@dataclass(frozen=True)
class ReferenceVideoDecodeSpec:
    max_frames: int | None = None
    keep: Literal["first", "last"] = "first"


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(
        self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


@runtime_checkable
class SupportsPipelineTickExecution(Protocol):
    """Explicit stage-local protocol required by interleaved PP clocks."""

    supports_interleaved_pipeline_execution: ClassVar[bool] = True

    def build_microbatches(self, states: Sequence[StepRequestState]) -> list[tuple[StepRequestState, ...]]:
        """Build ordered homogeneous stage-0 microbatches."""
        ...

    def pipeline_transport_spec(self, states: Sequence[StepRequestState]) -> PipelineTensorSpec:
        """Describe fixed intermediate and feedback tensor buffers."""
        ...

    def pipeline_model_phase(self, states: Sequence[StepRequestState]) -> str:
        """Return explicit model phase metadata for a pipeline token."""
        ...

    def pipeline_forward_local_stage(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState],
        cfg_branch: str,
        intermediate_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one local PP transformer partition."""
        ...

    def pipeline_finish_microbatch(
        self,
        states: Sequence[StepRequestState],
        noise_pred: torch.Tensor,
        *,
        positive_noise_pred: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run last-stage CFG and scheduler work."""
        ...


@runtime_checkable
class SupportsPipelineTickConfiguration(Protocol):
    """Optional model-specific validation for interleaved PP startup."""

    def validate_interleaved_pipeline_configuration(self) -> None:
        """Reject model/configuration combinations unsupported by pipeline ticks."""
        ...


def validate_pipeline_tick_configuration(pipeline: object) -> None:
    """Run an optional model-specific interleaved PP configuration check."""

    if isinstance(pipeline, SupportsPipelineTickConfiguration):
        pipeline.validate_interleaved_pipeline_configuration()


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, and other operations that need to know
    the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> bool:
    """Return whether ``pipeline`` explicitly enables step execution."""

    # Structural protocol checks alone cannot distinguish a model that happens
    # to expose the four methods from one that has validated request-local
    # state for this runtime.  Subclasses may explicitly disable inherited
    # implementations (for example Wan VACE) with ``False``.
    return getattr(pipeline, "supports_step_execution", False) is True and isinstance(pipeline, SupportsStepExecution)


def supports_pipeline_tick_execution(pipeline: object) -> bool:
    """Return whether a step pipeline explicitly supports interleaved PP."""

    return (
        supports_step_execution(pipeline)
        and getattr(pipeline, "supports_interleaved_pipeline_execution", False) is True
        and isinstance(pipeline, SupportsPipelineTickExecution)
    )


@runtime_checkable
class SupportsPromptUpdate(Protocol):
    """Optional protocol for pipelines that support midway prompt updates.

    Pipelines typically implement this via
    :class:`~vllm_omni.diffusion.prompt_update.PromptUpdateMixin`.
    """

    supports_prompt_update: ClassVar[bool] = True

    def prepare_prompt_update(
        self,
        state: StepRequestState,
        prompt: str,
        event_id: str,
        transition_chunks: int | None = None,
    ) -> None:
        """Encode and queue a prompt update on request-local state."""
        ...


def supports_prompt_update(pipeline: object) -> bool:
    """Return whether ``pipeline`` implements :class:`SupportsPromptUpdate`."""

    return isinstance(pipeline, SupportsPromptUpdate)


@runtime_checkable
class SupportsRequestScopedCacheDiT(Protocol):
    """Optional protocol for pipelines that own Cache-DiT hook transitions."""

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Assume ownership of an enabled Cache-DiT backend."""
        ...

    def is_cache_dit_enabled(self) -> bool:
        """Return whether this pipeline currently has Cache-DiT installed."""
        ...


def adopt_request_scoped_cache_dit(pipeline: object, backend: CacheDiTBackend) -> bool:
    """Transfer an enabled Cache-DiT backend to an opted-in pipeline."""

    if not isinstance(pipeline, SupportsRequestScopedCacheDiT):
        return False
    pipeline.adopt_cache_dit_backend(backend)
    return True


def is_request_scoped_cache_dit_enabled(pipeline: object) -> bool:
    """Read Cache-DiT state from a pipeline that owns its lifecycle."""

    return isinstance(pipeline, SupportsRequestScopedCacheDiT) and pipeline.is_cache_dit_enabled()
