# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped lifecycle for diffusion cache backends."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout


class CacheStateScope(str, Enum):
    """How many independent cache trajectories a backend can safely host."""

    EXCLUSIVE_TRAJECTORY = "exclusive_trajectory"
    REQUEST_SWAPPABLE = "request_swappable"
    BATCH_NATIVE = "batch_native"


class CacheDecisionScope(str, Enum):
    """The finest decision granularity implemented inside the backend."""

    BATCH = "batch"
    REQUEST = "request"
    BLOCK = "block"


class CacheCloseReason(str, Enum):
    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class CacheCapabilities:
    state_scope: CacheStateScope
    decision_scope: CacheDecisionScope
    supports_packed_subset: bool = False

    def __post_init__(self) -> None:
        if self.supports_packed_subset and self.state_scope != CacheStateScope.BATCH_NATIVE:
            raise ValueError("Packed cache subsets require batch-native request state.")


@dataclass(frozen=True)
class CacheRequestMetadata:
    request_id: str
    num_inference_steps: int
    execution_signature: Hashable

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Cache request_id must not be empty.")
        if self.num_inference_steps <= 0:
            raise ValueError("Cache num_inference_steps must be positive.")
        try:
            hash(self.execution_signature)
        except TypeError as exc:
            raise TypeError("Cache execution_signature must be hashable.") from exc


@dataclass(eq=False)
class CacheHandle:
    """Opaque request cache state owned by :class:`RequestScopedCacheRuntime`."""

    metadata: CacheRequestMetadata
    generation: int
    _state: Any = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _invalidated: bool = field(default=False, init=False, repr=False)

    @property
    def request_id(self) -> str:
        return self.metadata.request_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    @property
    def opaque_state(self) -> Any:
        return self._state


class RequestCacheAdapter(Protocol):
    """Backend-specific opaque state operations consumed by the runtime."""

    capabilities: CacheCapabilities

    def open_request(self, metadata: CacheRequestMetadata) -> Any: ...

    def activate(self, handles: Sequence[CacheHandle], row_layout: RequestRowLayout) -> None: ...

    def capture(self, handles: Sequence[CacheHandle]) -> Sequence[Any]: ...

    def invalidate(self, handles: Sequence[CacheHandle]) -> None: ...

    def deactivate(self, handles: Sequence[CacheHandle]) -> None: ...

    def close_request(self, handle: CacheHandle, reason: CacheCloseReason) -> Any: ...


class ExclusiveCacheAdapter:
    """Safe fallback for existing backends without request state swapping.

    The wrapped backend is refreshed once when its sole request is opened.
    It cannot preserve a trajectory while another request executes, so the
    runtime permits only one open handle.
    """

    capabilities = CacheCapabilities(
        state_scope=CacheStateScope.EXCLUSIVE_TRAJECTORY,
        decision_scope=CacheDecisionScope.BATCH,
        supports_packed_subset=False,
    )

    def __init__(self, backend: CacheBackend, pipeline: Any) -> None:
        self._backend = backend
        self._pipeline = pipeline

    def open_request(self, metadata: CacheRequestMetadata) -> None:
        if not self._backend.is_enabled():
            raise RuntimeError("Cannot open a request on a disabled cache backend.")
        self._backend.refresh(
            self._pipeline,
            num_inference_steps=metadata.num_inference_steps,
        )

    def activate(self, handles: Sequence[CacheHandle], row_layout: RequestRowLayout) -> None:
        del handles, row_layout

    def capture(self, handles: Sequence[CacheHandle]) -> Sequence[Any]:
        return [None] * len(handles)

    def invalidate(self, handles: Sequence[CacheHandle]) -> None:
        del handles

    def deactivate(self, handles: Sequence[CacheHandle]) -> None:
        del handles

    def close_request(self, handle: CacheHandle, reason: CacheCloseReason) -> None:
        del handle, reason


class CacheTransaction:
    """One real cache-enabled step over a validated request cohort."""

    def __init__(
        self,
        runtime: RequestScopedCacheRuntime,
        handles: tuple[CacheHandle, ...],
        row_layout: RequestRowLayout,
    ) -> None:
        self._runtime = runtime
        self._handles = handles
        self._row_layout = row_layout
        self._entered = False
        self._committed = False
        self._finished = False

    def __enter__(self) -> CacheTransaction:
        if self._entered:
            raise RuntimeError("CacheTransaction cannot be entered more than once.")
        self._runtime._begin_transaction(self._handles, self._row_layout)
        self._entered = True
        return self

    def commit(self) -> None:
        if not self._entered or self._finished:
            raise RuntimeError("CacheTransaction can only commit while active.")
        self._committed = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_value, traceback
        if not self._entered or self._finished:
            return
        try:
            self._runtime._finish_transaction(
                self._handles,
                commit=exc_type is None and self._committed,
            )
        finally:
            self._finished = True


class RequestScopedCacheRuntime:
    """Worker-local owner of request cache handles and step transactions."""

    def __init__(self, adapter: RequestCacheAdapter) -> None:
        self._adapter = adapter
        self._handles: dict[str, CacheHandle] = {}
        self._active_handles: tuple[CacheHandle, ...] = ()
        self._next_generation = 0

    @property
    def capabilities(self) -> CacheCapabilities:
        return self._adapter.capabilities

    def open_request(self, metadata: CacheRequestMetadata) -> CacheHandle:
        if metadata.request_id in self._handles:
            raise ValueError(f"Cache request {metadata.request_id!r} is already open.")
        if self.capabilities.state_scope == CacheStateScope.EXCLUSIVE_TRAJECTORY and self._handles:
            raise RuntimeError("Exclusive cache backend already owns an open request trajectory.")

        state = self._adapter.open_request(metadata)
        handle = CacheHandle(
            metadata=metadata,
            generation=self._next_generation,
            _state=state,
        )
        self._next_generation += 1
        self._handles[metadata.request_id] = handle
        return handle

    def transaction(
        self,
        handles: Sequence[CacheHandle],
        row_layout: RequestRowLayout,
    ) -> CacheTransaction:
        normalized = tuple(handles)
        self._validate_transaction(normalized, row_layout)
        return CacheTransaction(self, normalized, row_layout)

    def invalidate_request(self, handle: CacheHandle) -> None:
        self._validate_handle(handle)
        if handle in self._active_handles:
            raise RuntimeError("Cannot invalidate an active cache handle.")
        self._adapter.invalidate((handle,))
        handle._invalidated = True

    def close_request(self, handle: CacheHandle, reason: CacheCloseReason) -> Any:
        self._validate_handle(handle, allow_invalidated=True)
        if handle in self._active_handles:
            raise RuntimeError("Cannot close an active cache handle.")
        try:
            return self._adapter.close_request(handle, reason)
        finally:
            self._handles.pop(handle.request_id, None)
            handle._closed = True
            handle._state = None

    def _validate_transaction(
        self,
        handles: tuple[CacheHandle, ...],
        row_layout: RequestRowLayout,
    ) -> None:
        if not handles:
            raise ValueError("Cache transaction requires at least one handle.")
        if len({id(handle) for handle in handles}) != len(handles):
            raise ValueError("Cache transaction handles must be unique.")
        for handle in handles:
            self._validate_handle(handle)

        state_scope = self.capabilities.state_scope
        if state_scope != CacheStateScope.BATCH_NATIVE and len(handles) != 1:
            raise RuntimeError(f"Cache backend with state_scope={state_scope.value!r} cannot activate a cohort.")

        request_ids = tuple(handle.request_id for handle in handles)
        if row_layout.request_ids != request_ids:
            raise ValueError(
                "Cache transaction handles must follow row_layout.request_ids: "
                f"{request_ids} != {row_layout.request_ids}."
            )

    def _validate_handle(self, handle: CacheHandle, *, allow_invalidated: bool = False) -> None:
        current = self._handles.get(handle.request_id)
        if current is not handle or handle.closed:
            raise ValueError(f"Cache handle for {handle.request_id!r} is closed or stale.")
        if handle.invalidated and not allow_invalidated:
            raise RuntimeError(f"Cache handle for {handle.request_id!r} is invalidated.")

    def _begin_transaction(
        self,
        handles: tuple[CacheHandle, ...],
        row_layout: RequestRowLayout,
    ) -> None:
        self._validate_transaction(handles, row_layout)
        if self._active_handles:
            raise RuntimeError("Nested cache transactions are not supported.")
        self._active_handles = handles
        try:
            self._adapter.activate(handles, row_layout)
        except Exception:
            try:
                self._invalidate_handles(handles)
            finally:
                self._active_handles = ()
            raise

    def _finish_transaction(
        self,
        handles: tuple[CacheHandle, ...],
        *,
        commit: bool,
    ) -> None:
        if self._active_handles != handles:
            raise RuntimeError("Cache transaction is not the active transaction.")
        try:
            if commit:
                try:
                    states = tuple(self._adapter.capture(handles))
                    if len(states) != len(handles):
                        raise RuntimeError(f"Cache adapter captured {len(states)} states for {len(handles)} handles.")
                    for handle, state in zip(handles, states, strict=True):
                        handle._state = state
                except Exception:
                    self._invalidate_handles(handles)
                    raise
            else:
                self._invalidate_handles(handles)
        finally:
            try:
                self._adapter.deactivate(handles)
            finally:
                self._active_handles = ()

    def _invalidate_handles(self, handles: Sequence[CacheHandle]) -> None:
        self._adapter.invalidate(handles)
        for handle in handles:
            handle._invalidated = True


__all__ = [
    "CacheCapabilities",
    "CacheCloseReason",
    "CacheDecisionScope",
    "CacheHandle",
    "CacheRequestMetadata",
    "CacheStateScope",
    "CacheTransaction",
    "ExclusiveCacheAdapter",
    "RequestCacheAdapter",
    "RequestScopedCacheRuntime",
]
