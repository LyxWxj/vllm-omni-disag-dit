# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


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

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""

    def denoise_step(self, state: DiffusionRequestState, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise step."""

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop."""


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, disaggregated stage separation, and
    other operations that need to know the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    **Coarse-grained declaration** (backward compatible)::

        _encoder_modules = ["text_encoder"]
        _dit_modules = ["transformer"]
        _vae_modules = ["vae"]
        _scheduler_modules = ["scheduler"]
        _tokenizer_modules = ["tokenizer"]

    **Fine-grained declaration** (optional, for multi-encoder/decoder)::

        _component_registry = {
            "text_encoder":  {"tokenizer", "text_encoder"},
            "vae_encoder":   {"vae_encoder"},
            "audio_encoder": {"audio_encoder"},
            "transformer":   {"transformer"},
            "scheduler":     {"scheduler"},
            "vae_decoder":   {"vae"},
            "audio_decoder": {"audio_decoder"},
        }
        _default_stage_layout = {
            "encode":  ["text_encoder", "scheduler"],
            "denoise": ["transformer", "scheduler"],
            "decode":  ["vae_decoder"],
        }

    When ``_component_registry`` and ``_default_stage_layout`` are both
    defined, :meth:`get_stage_components` uses them for fine-grained
    control.  Otherwise it falls back to the coarse-grained fields.

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
        _scheduler_modules: Scheduler modules shared across stages.
            Optional, defaults to ``[]``.
        _tokenizer_modules: Tokenizer modules used by encoder stages.
            Optional, defaults to ``[]``.
        _component_registry: Fine-grained component mapping.  Each key
            is a logical component name (e.g. ``"text_encoder"``), each
            value is the set of module attribute names it owns.  Optional,
            defaults to ``None`` (use coarse-grained fields instead).
        _default_stage_layout: Default stage→component-group mapping.
            Each key is a stage name (e.g. ``"encode"``), each value is
            a list of component-group names from ``_component_registry``.
            Optional, defaults to ``None`` (use coarse-grained auto-derive).
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []
    _scheduler_modules: ClassVar[list[str]] = []
    _tokenizer_modules: ClassVar[list[str]] = []
    _component_registry: ClassVar[dict[str, set[str]] | None] = None
    _default_stage_layout: ClassVar[dict[str, list[str]] | None] = None

    @classmethod
    def get_stage_components(cls, stage: str) -> set[str]:
        """Return the set of module attribute names required for *stage*.

        Resolution order:

        1. If ``_component_registry`` and ``_default_stage_layout`` are both
           defined, look up the component-group names for *stage* in the
           layout and union their module sets from the registry.
        2. Otherwise, use the coarse-grained auto-derivation:

           - ``"encode"``  → encoder + tokenizer + scheduler modules
           - ``"denoise"`` → dit + scheduler modules
           - ``"decode"``  → vae modules
           - ``"diffusion"`` → all modules

        Subclasses may override for fully custom mappings.

        Args:
            stage: Stage name (e.g. ``"encode"``, ``"denoise"``,
                ``"decode"``, ``"diffusion"``).

        Returns:
            Set of component attribute names (e.g. ``{"text_encoder",
            "tokenizer", "scheduler"}``).
        """
        # --- Path 1: fine-grained registry ---
        if cls._component_registry is not None and cls._default_stage_layout is not None:
            if stage == "diffusion":
                # All components.
                return set().union(*cls._component_registry.values())
            group_names = cls._default_stage_layout.get(stage)
            if group_names is None:
                raise ValueError(
                    f"Unknown stage {stage!r}. Defined stages: "
                    f"{sorted(cls._default_stage_layout)}. "
                    f"Override get_stage_components() for custom names."
                )
            result: set[str] = set()
            for name in group_names:
                modules = cls._component_registry.get(name)
                if modules is None:
                    raise ValueError(
                        f"Component group {name!r} not in _component_registry. "
                        f"Available: {sorted(cls._component_registry)}."
                    )
                result.update(modules)
            return result

        # --- Path 2: coarse-grained auto-derive (backward compatible) ---
        if stage == "diffusion":
            return set(
                cls._encoder_modules
                + cls._dit_modules
                + cls._vae_modules
                + cls._scheduler_modules
                + cls._tokenizer_modules
            )
        mapping: dict[str, list[str]] = {
            "encode": cls._encoder_modules + cls._tokenizer_modules + cls._scheduler_modules,
            "denoise": cls._dit_modules + cls._scheduler_modules,
            "decode": cls._vae_modules,
        }
        components = mapping.get(stage)
        if components is None:
            raise ValueError(
                f"Unknown stage {stage!r}. Supported: 'encode', 'denoise', "
                f"'decode', 'diffusion'. Override get_stage_components() for "
                f"custom stage names."
            )
        return set(components)


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)
