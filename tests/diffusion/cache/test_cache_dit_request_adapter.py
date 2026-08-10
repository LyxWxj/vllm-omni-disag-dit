# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Cache-DiT request context swapping."""

from __future__ import annotations

from typing import Any

import pytest

from vllm_omni.diffusion.cache.cachedit.request_adapter import CacheDiTRequestAdapter
from vllm_omni.diffusion.cache.request_scope import (
    CacheCloseReason,
    CacheRequestMetadata,
    CacheStateScope,
    RequestScopedCacheRuntime,
)
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeContext:
    def __init__(self, *, name: str, num_inference_steps: int | None = None) -> None:
        self.name = name
        self.num_inference_steps = num_inference_steps
        self.buffers: dict[str, Any] = {}
        self.current_step = 0

    def clear_buffers(self) -> None:
        self.buffers.clear()


class _FakeContextManager:
    def __init__(self, name: str) -> None:
        self.name = name
        self._cached_context_manager: dict[str, _FakeContext] = {}
        self._current_context: _FakeContext | None = None
        self._current_step_refreshed = False
        self.new_context(name="blocks")

    def new_context(self, *args, **kwargs) -> _FakeContext:
        context = _FakeContext(*args, **kwargs)
        context._init_args = args
        context._init_kwargs = kwargs
        self._cached_context_manager[context.name] = context
        return context

    def get_context(self, name: str) -> _FakeContext:
        return self._cached_context_manager[name]

    def set_context(self, name: str) -> _FakeContext:
        self._current_context = self.get_context(name)
        return self._current_context


class _FakeBackend:
    def __init__(self, *managers: _FakeContextManager) -> None:
        self.enabled = True
        self.managers = managers
        self.refresh_calls: list[int] = []

    def is_enabled(self) -> bool:
        return self.enabled

    def get_request_context_managers(self):
        return self.managers

    def refresh(self, pipeline, num_inference_steps: int, verbose: bool = True) -> None:
        del pipeline, verbose
        self.refresh_calls.append(num_inference_steps)
        for manager in self.managers:
            names = tuple(manager._cached_context_manager)
            manager._cached_context_manager = {}
            manager._current_context = None
            manager._current_step_refreshed = False
            for name in names:
                manager.new_context(name=name, num_inference_steps=num_inference_steps)


def _metadata(request_id: str, steps: int) -> CacheRequestMetadata:
    return CacheRequestMetadata(
        request_id=request_id,
        num_inference_steps=steps,
        execution_signature=("qwen", "cache_dit", steps),
    )


def _layout(request_id: str) -> RequestRowLayout:
    return RequestRowLayout.from_request_row_counts([request_id], [1])


def test_interleaved_requests_restore_all_cache_dit_contexts() -> None:
    manager_a = _FakeContextManager("transformer-a")
    manager_b = _FakeContextManager("transformer-b")
    backend = _FakeBackend(manager_a, manager_b)
    runtime = RequestScopedCacheRuntime(CacheDiTRequestAdapter(backend, object()))
    assert runtime.capabilities.state_scope == CacheStateScope.REQUEST_SWAPPABLE

    handle_a = runtime.open_request(_metadata("req-a", 20))
    handle_b = runtime.open_request(_metadata("req-b", 30))
    assert backend.refresh_calls == [20, 30]

    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        context_a0 = manager_a.set_context("blocks")
        context_a1 = manager_b.set_context("blocks")
        assert context_a0.num_inference_steps == 20
        context_a0.current_step = 7
        context_a0.buffers["residual"] = object()
        context_a1.current_step = 8
        transaction.commit()

    with runtime.transaction([handle_b], _layout("req-b")) as transaction:
        context_b0 = manager_a.set_context("blocks")
        assert context_b0 is not context_a0
        assert context_b0.num_inference_steps == 30
        assert context_b0.current_step == 0
        assert context_b0.buffers == {}
        context_b0.current_step = 11
        transaction.commit()

    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        assert manager_a.get_context("blocks") is context_a0
        assert manager_a.get_context("blocks").current_step == 7
        assert "residual" in manager_a.get_context("blocks").buffers
        assert manager_b.get_context("blocks") is context_a1
        assert manager_b.get_context("blocks").current_step == 8
        transaction.commit()

    with runtime.transaction([handle_b], _layout("req-b")) as transaction:
        assert manager_a.get_context("blocks") is context_b0
        assert manager_a.get_context("blocks").current_step == 11
        transaction.commit()

    runtime.close_request(handle_a, CacheCloseReason.FINISHED)
    assert context_a0.buffers == {}
    runtime.close_request(handle_b, CacheCloseReason.FINISHED)


def test_failed_transaction_invalidates_and_detaches_cache_dit_contexts() -> None:
    manager = _FakeContextManager("transformer")
    backend = _FakeBackend(manager)
    runtime = RequestScopedCacheRuntime(CacheDiTRequestAdapter(backend, object()))
    handle = runtime.open_request(_metadata("req", 20))

    with pytest.raises(RuntimeError, match="step failed"), runtime.transaction([handle], _layout("req")):
        failed_context = manager.set_context("blocks")
        failed_context.buffers["residual"] = object()
        raise RuntimeError("step failed")

    assert handle.invalidated is True
    assert failed_context.buffers == {}
    assert manager.get_context("blocks") is not failed_context
    runtime.close_request(handle, CacheCloseReason.ERROR)


def test_adapter_rejects_backend_without_swappable_contexts() -> None:
    backend = _FakeBackend()

    with pytest.raises(ValueError, match="no request-swappable"):
        CacheDiTRequestAdapter(backend, object())
