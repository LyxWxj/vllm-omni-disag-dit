# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-swappable lifecycle adapter for Cache-DiT contexts."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.cache.request_scope import (
    CacheCapabilities,
    CacheCloseReason,
    CacheDecisionScope,
    CacheHandle,
    CacheRequestMetadata,
    CacheStateScope,
)
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout


@dataclass(frozen=True)
class _ContextTemplate:
    name: str
    init_args: tuple[Any, ...]
    init_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _ManagerTemplate:
    manager: Any
    contexts: tuple[_ContextTemplate, ...]


@dataclass(frozen=True)
class _ManagerState:
    manager: Any
    contexts: dict[str, Any]
    current_context: Any
    current_step_refreshed: bool


@dataclass(frozen=True)
class CacheDiTRequestState:
    """All mutable Cache-DiT contexts belonging to one request trajectory."""

    manager_states: tuple[_ManagerState, ...]


class CacheDiTRequestAdapter:
    """Swap Cache-DiT context maps without copying their cached tensors."""

    capabilities = CacheCapabilities(
        state_scope=CacheStateScope.REQUEST_SWAPPABLE,
        decision_scope=CacheDecisionScope.REQUEST,
        supports_packed_subset=False,
    )

    def __init__(self, backend: Any, pipeline: Any) -> None:
        self._backend = backend
        self._pipeline = pipeline
        self._active_handle: CacheHandle | None = None

        get_managers = getattr(backend, "get_request_context_managers", None)
        if not callable(get_managers):
            raise TypeError("Cache-DiT backend does not expose request context managers.")
        managers = tuple(get_managers())
        if not managers:
            raise ValueError("Cache-DiT backend has no request-swappable context managers.")

        self._templates = tuple(self._build_manager_template(manager) for manager in managers)
        self._idle_state = self._snapshot_state()

    def open_request(self, metadata: CacheRequestMetadata) -> CacheDiTRequestState:
        if self._active_handle is not None:
            raise RuntimeError("Cannot open Cache-DiT state while a request is active.")
        if not self._backend.is_enabled():
            raise RuntimeError("Cannot open a request on a disabled Cache-DiT backend.")

        try:
            self._backend.refresh(
                self._pipeline,
                num_inference_steps=metadata.num_inference_steps,
            )
            request_state = self._snapshot_state()
        finally:
            self._idle_state = self._create_idle_state()
        return request_state

    def activate(self, handles: Sequence[CacheHandle], row_layout: RequestRowLayout) -> None:
        handle = self._single_handle(handles)
        if self._active_handle is not None:
            raise RuntimeError("Cache-DiT request state is already active.")
        if row_layout.request_ids != (handle.request_id,):
            raise ValueError("Cache-DiT row layout does not match the active request handle.")
        request_state = handle.opaque_state
        if not isinstance(request_state, CacheDiTRequestState):
            raise TypeError("Cache-DiT handle contains incompatible opaque state.")

        try:
            self._install_state(request_state)
        except Exception:
            self._install_state(self._idle_state)
            raise
        self._active_handle = handle

    def capture(self, handles: Sequence[CacheHandle]) -> Sequence[CacheDiTRequestState]:
        self._require_active(handles)
        return (self._snapshot_state(),)

    def invalidate(self, handles: Sequence[CacheHandle]) -> None:
        handle = self._single_handle(handles)
        request_state = handle.opaque_state
        if isinstance(request_state, CacheDiTRequestState):
            self._clear_state(request_state)

    def deactivate(self, handles: Sequence[CacheHandle]) -> None:
        self._require_active(handles)
        try:
            self._install_state(self._idle_state)
        finally:
            self._active_handle = None

    def close_request(self, handle: CacheHandle, reason: CacheCloseReason) -> None:
        del reason
        request_state = handle.opaque_state
        if isinstance(request_state, CacheDiTRequestState):
            self._clear_state(request_state)

    def _build_manager_template(self, manager: Any) -> _ManagerTemplate:
        contexts = getattr(manager, "_cached_context_manager", None)
        if not isinstance(contexts, dict) or not contexts:
            raise ValueError("Cache-DiT context manager has no initialized context map.")
        if not callable(getattr(manager, "new_context", None)):
            raise TypeError("Cache-DiT context manager cannot create request contexts.")

        templates = []
        for name, context in contexts.items():
            if not hasattr(context, "_init_args") or not hasattr(context, "_init_kwargs"):
                raise ValueError(f"Cache-DiT context {name!r} does not preserve its initialization contract.")
            templates.append(
                _ContextTemplate(
                    name=name,
                    init_args=tuple(context._init_args),
                    init_kwargs=dict(context._init_kwargs),
                )
            )
        return _ManagerTemplate(manager=manager, contexts=tuple(templates))

    def _snapshot_state(self) -> CacheDiTRequestState:
        manager_states = []
        for template in self._templates:
            manager = template.manager
            contexts = getattr(manager, "_cached_context_manager", None)
            if not isinstance(contexts, dict):
                raise TypeError("Cache-DiT context manager replaced its context map with an incompatible value.")
            manager_states.append(
                _ManagerState(
                    manager=manager,
                    contexts=contexts,
                    current_context=getattr(manager, "_current_context", None),
                    current_step_refreshed=bool(getattr(manager, "_current_step_refreshed", False)),
                )
            )
        return CacheDiTRequestState(tuple(manager_states))

    def _create_idle_state(self) -> CacheDiTRequestState:
        for template in self._templates:
            manager = template.manager
            manager._cached_context_manager = {}
            manager._current_context = None
            manager._current_step_refreshed = False
            for context_template in template.contexts:
                args = copy.deepcopy(context_template.init_args)
                kwargs = copy.deepcopy(context_template.init_kwargs)
                kwargs["name"] = context_template.name
                manager.new_context(*args, **kwargs)
            if set(manager._cached_context_manager) != {context.name for context in template.contexts}:
                raise RuntimeError("Cache-DiT context manager recreated an incompatible idle context map.")
        return self._snapshot_state()

    def _install_state(self, state: CacheDiTRequestState) -> None:
        if len(state.manager_states) != len(self._templates):
            raise RuntimeError("Cache-DiT request state has an incompatible manager count.")
        for template, manager_state in zip(self._templates, state.manager_states, strict=True):
            if manager_state.manager is not template.manager:
                raise RuntimeError("Cache-DiT request state belongs to another context manager.")
        for manager_state in state.manager_states:
            manager = manager_state.manager
            manager._cached_context_manager = manager_state.contexts
            manager._current_context = manager_state.current_context
            manager._current_step_refreshed = manager_state.current_step_refreshed

    @staticmethod
    def _clear_state(state: CacheDiTRequestState) -> None:
        for manager_state in state.manager_states:
            for context in manager_state.contexts.values():
                clear_buffers = getattr(context, "clear_buffers", None)
                if callable(clear_buffers):
                    clear_buffers()

    @staticmethod
    def _single_handle(handles: Sequence[CacheHandle]) -> CacheHandle:
        if len(handles) != 1:
            raise ValueError("Cache-DiT request adapter requires exactly one handle.")
        return handles[0]

    def _require_active(self, handles: Sequence[CacheHandle]) -> CacheHandle:
        handle = self._single_handle(handles)
        if self._active_handle is not handle:
            raise RuntimeError("Cache-DiT handle is not the active request state.")
        return handle


__all__ = ["CacheDiTRequestAdapter", "CacheDiTRequestState"]
