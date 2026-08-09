# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-swappable lifecycle adapter for hook-based TeaCache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm_omni.diffusion.cache.request_scope import (
    CacheCapabilities,
    CacheCloseReason,
    CacheDecisionScope,
    CacheHandle,
    CacheRequestMetadata,
    CacheStateScope,
)
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheRequestState
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout


def find_teacache_hook(pipeline: Any) -> TeaCacheHook | None:
    """Find the single hook-backed TeaCache target owned by a pipeline."""
    seen: set[int] = set()
    for attribute in ("transformer", "denoising_transformer", "bagel"):
        module = getattr(pipeline, attribute, None)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        registry = getattr(module, "_hook_registry", None)
        if registry is None:
            continue
        hook = registry.get_hook(TeaCacheHook._HOOK_NAME)
        if isinstance(hook, TeaCacheHook):
            return hook
    return None


class TeaCacheRequestAdapter:
    """Swap complete TeaCache trajectories around single-request forwards."""

    capabilities = CacheCapabilities(
        state_scope=CacheStateScope.REQUEST_SWAPPABLE,
        decision_scope=CacheDecisionScope.REQUEST,
        supports_packed_subset=False,
    )

    def __init__(self, hook: TeaCacheHook) -> None:
        self._hook = hook
        self._active_handle: CacheHandle | None = None

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> TeaCacheRequestAdapter:
        hook = find_teacache_hook(pipeline)
        if hook is None:
            raise ValueError(f"Pipeline {type(pipeline).__name__} has no hook-backed TeaCache state to swap.")
        return cls(hook)

    def open_request(self, metadata: CacheRequestMetadata) -> TeaCacheRequestState:
        del metadata
        return TeaCacheRequestState()

    def activate(self, handles: Sequence[CacheHandle], row_layout: RequestRowLayout) -> None:
        handle = self._single_handle(handles)
        if self._active_handle is not None:
            raise RuntimeError("TeaCache request state is already active.")
        if row_layout.request_ids != (handle.request_id,):
            raise ValueError("TeaCache row layout does not match the active request handle.")
        request_state = handle.opaque_state
        if not isinstance(request_state, TeaCacheRequestState):
            raise TypeError("TeaCache handle contains incompatible opaque state.")
        self._hook.bind_request_state(request_state)
        self._active_handle = handle

    def capture(self, handles: Sequence[CacheHandle]) -> Sequence[TeaCacheRequestState]:
        self._require_active(handles)
        return (self._hook.capture_request_state(),)

    def invalidate(self, handles: Sequence[CacheHandle]) -> None:
        self._single_handle(handles)

    def deactivate(self, handles: Sequence[CacheHandle]) -> None:
        self._require_active(handles)
        try:
            self._hook.unbind_request_state()
        finally:
            self._active_handle = None

    def close_request(self, handle: CacheHandle, reason: CacheCloseReason) -> None:
        del handle, reason

    @staticmethod
    def _single_handle(handles: Sequence[CacheHandle]) -> CacheHandle:
        if len(handles) != 1:
            raise ValueError("TeaCache request adapter requires exactly one handle.")
        return handles[0]

    def _require_active(self, handles: Sequence[CacheHandle]) -> CacheHandle:
        handle = self._single_handle(handles)
        if self._active_handle is not handle:
            raise RuntimeError("TeaCache handle is not the active request state.")
        return handle


__all__ = ["TeaCacheRequestAdapter", "find_teacache_hook"]
