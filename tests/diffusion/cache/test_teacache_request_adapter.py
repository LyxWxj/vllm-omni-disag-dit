# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for TeaCache request state swapping."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.request_scope import (
    CacheCloseReason,
    CacheRequestMetadata,
    CacheStateScope,
    RequestScopedCacheRuntime,
)
from vllm_omni.diffusion.cache.teacache import TeaCacheConfig, TeaCacheHook
from vllm_omni.diffusion.cache.teacache.request_adapter import (
    TeaCacheRequestAdapter,
    find_teacache_hook,
)
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _metadata(request_id: str) -> CacheRequestMetadata:
    return CacheRequestMetadata(
        request_id=request_id,
        num_inference_steps=20,
        execution_signature=("qwen", 1024, "cfg", True, "tea_cache"),
    )


def _layout(request_id: str) -> RequestRowLayout:
    return RequestRowLayout.from_request_row_counts([request_id], [1])


def _hook() -> TeaCacheHook:
    return TeaCacheHook(
        TeaCacheConfig(
            transformer_type="QwenImageTransformer2DModel",
            rel_l1_thresh=0.2,
        )
    )


def _set_branch_state(hook: TeaCacheHook, branch: str, value: float) -> None:
    hook.state_manager.set_context(f"teacache_{branch}")
    state = hook.state_manager.get_state()
    state.cnt = int(value)
    state.accumulated_rel_l1_distance = value
    state.previous_modulated_input = torch.tensor([value])
    state.previous_residual = torch.tensor([value + 1])
    state.previous_residual_encoder = torch.tensor([value + 2])


def test_pipeline_hook_discovery_uses_supported_teacache_targets() -> None:
    hook = _hook()
    registry = SimpleNamespace(get_hook=lambda name: hook if name == TeaCacheHook._HOOK_NAME else None)
    pipeline = SimpleNamespace(denoising_transformer=SimpleNamespace(_hook_registry=registry))

    assert find_teacache_hook(pipeline) is hook
    assert TeaCacheRequestAdapter.from_pipeline(pipeline)._hook is hook


def test_pipeline_hook_discovery_rejects_non_hook_teacache() -> None:
    pipeline = SimpleNamespace(transformer=SimpleNamespace())

    assert find_teacache_hook(pipeline) is None
    with pytest.raises(ValueError, match="no hook-backed TeaCache"):
        TeaCacheRequestAdapter.from_pipeline(pipeline)


def test_interleaved_requests_restore_all_cfg_branches_and_forward_counter() -> None:
    hook = _hook()
    runtime = RequestScopedCacheRuntime(TeaCacheRequestAdapter(hook))
    assert runtime.capabilities.state_scope == CacheStateScope.REQUEST_SWAPPABLE
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        _set_branch_state(hook, "positive", 3.0)
        _set_branch_state(hook, "negative", 4.0)
        hook._forward_cnt = 7
        transaction.commit()

    assert hook.state_manager._states == {}
    assert hook._forward_cnt == 0

    with runtime.transaction([handle_b], _layout("req-b")) as transaction:
        assert hook.state_manager._states == {}
        _set_branch_state(hook, "positive", 9.0)
        hook._forward_cnt = 2
        transaction.commit()

    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        assert hook._forward_cnt == 7
        assert set(hook.state_manager._states) == {
            "teacache_positive",
            "teacache_negative",
        }
        positive = hook.state_manager._states["teacache_positive"]
        negative = hook.state_manager._states["teacache_negative"]
        assert positive.cnt == 3
        assert positive.accumulated_rel_l1_distance == 3.0
        assert torch.equal(positive.previous_modulated_input, torch.tensor([3.0]))
        assert torch.equal(positive.previous_residual, torch.tensor([4.0]))
        assert torch.equal(positive.previous_residual_encoder, torch.tensor([5.0]))
        assert negative.cnt == 4
        transaction.commit()

    with runtime.transaction([handle_b], _layout("req-b")) as transaction:
        assert hook._forward_cnt == 2
        assert set(hook.state_manager._states) == {"teacache_positive"}
        assert hook.state_manager._states["teacache_positive"].cnt == 9
        transaction.commit()


def test_failed_transaction_invalidates_handle_and_detaches_hook_state() -> None:
    hook = _hook()
    runtime = RequestScopedCacheRuntime(TeaCacheRequestAdapter(hook))
    handle = runtime.open_request(_metadata("req"))

    with pytest.raises(RuntimeError, match="step failed"):
        with runtime.transaction([handle], _layout("req")):
            _set_branch_state(hook, "positive", 1.0)
            raise RuntimeError("step failed")

    assert handle.invalidated is True
    assert hook.state_manager._states == {}
    assert hook._forward_cnt == 0
    runtime.close_request(handle, CacheCloseReason.ERROR)
