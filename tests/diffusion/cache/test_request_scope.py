# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for request-scoped cache lifecycle and transactions."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.cache.request_scope import (
    CacheCapabilities,
    CacheCloseReason,
    CacheDecisionScope,
    CacheRequestMetadata,
    CacheStateScope,
    ExclusiveCacheAdapter,
    RequestScopedCacheRuntime,
)
from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _metadata(request_id: str) -> CacheRequestMetadata:
    return CacheRequestMetadata(
        request_id=request_id,
        num_inference_steps=20,
        execution_signature=("shape", 1024, "cfg", False),
    )


def _layout(*request_ids: str) -> RequestRowLayout:
    return RequestRowLayout.from_request_row_counts(request_ids, [1] * len(request_ids))


def test_metadata_and_capabilities_reject_invalid_contracts():
    with pytest.raises(ValueError, match="must not be empty"):
        CacheRequestMetadata("", 20, ("shape", 1024))
    with pytest.raises(ValueError, match="must be positive"):
        CacheRequestMetadata("req-a", 0, ("shape", 1024))
    with pytest.raises(TypeError, match="must be hashable"):
        CacheRequestMetadata("req-a", 20, ["not", "hashable"])
    with pytest.raises(ValueError, match="require batch-native"):
        CacheCapabilities(
            state_scope=CacheStateScope.REQUEST_SWAPPABLE,
            decision_scope=CacheDecisionScope.REQUEST,
            supports_packed_subset=True,
        )


class _RecordingAdapter:
    def __init__(self, state_scope: CacheStateScope, fail_operation: str | None = None) -> None:
        self.capabilities = CacheCapabilities(
            state_scope=state_scope,
            decision_scope=CacheDecisionScope.REQUEST,
            supports_packed_subset=state_scope == CacheStateScope.BATCH_NATIVE,
        )
        self.events = []
        self.fail_operation = fail_operation

    def _fail_if_requested(self, operation: str) -> None:
        if self.fail_operation == operation:
            raise RuntimeError(f"{operation} failed")

    def open_request(self, metadata):
        state = {"request_id": metadata.request_id, "version": 0}
        self.events.append(("open", metadata.request_id))
        return state

    def activate(self, handles, row_layout):
        self.events.append(
            (
                "activate",
                tuple(handle.request_id for handle in handles),
                row_layout.request_ids,
            )
        )
        self._fail_if_requested("activate")

    def capture(self, handles):
        states = [
            {
                "request_id": handle.request_id,
                "version": handle.opaque_state["version"] + 1,
            }
            for handle in handles
        ]
        self.events.append(("capture", tuple(handle.request_id for handle in handles)))
        self._fail_if_requested("capture")
        return states

    def invalidate(self, handles):
        self.events.append(("invalidate", tuple(handle.request_id for handle in handles)))
        self._fail_if_requested("invalidate")

    def deactivate(self, handles):
        self.events.append(("deactivate", tuple(handle.request_id for handle in handles)))
        self._fail_if_requested("deactivate")

    def close_request(self, handle, reason):
        self.events.append(("close", handle.request_id, reason))
        return {"closed": handle.request_id, "reason": reason.value}


def test_committed_transaction_captures_opaque_request_state():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle = runtime.open_request(_metadata("req-a"))

    with runtime.transaction([handle], _layout("req-a")) as transaction:
        transaction.commit()

    assert handle.invalidated is False
    assert handle.opaque_state == {"request_id": "req-a", "version": 1}
    assert adapter.events == [
        ("open", "req-a"),
        ("activate", ("req-a",), ("req-a",)),
        ("capture", ("req-a",)),
        ("deactivate", ("req-a",)),
    ]


@pytest.mark.parametrize("raise_inside", [False, True])
def test_uncommitted_or_failed_transaction_invalidates_handle(raise_inside):
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle = runtime.open_request(_metadata("req-a"))

    if raise_inside:
        with pytest.raises(RuntimeError, match="step failed"):
            with runtime.transaction([handle], _layout("req-a")):
                raise RuntimeError("step failed")
    else:
        with runtime.transaction([handle], _layout("req-a")):
            pass

    assert handle.invalidated is True
    with pytest.raises(RuntimeError, match="invalidated"):
        runtime.transaction([handle], _layout("req-a"))


def test_request_swappable_runtime_keeps_independent_handles():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with runtime.transaction([handle_b], _layout("req-b")) as transaction:
        transaction.commit()
    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        transaction.commit()

    assert handle_a.opaque_state["version"] == 1
    assert handle_b.opaque_state["version"] == 1


def test_capture_count_mismatch_invalidates_and_deactivates_handle():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    adapter.capture = lambda handles: []
    runtime = RequestScopedCacheRuntime(adapter)
    handle = runtime.open_request(_metadata("req-a"))

    with pytest.raises(RuntimeError, match="captured 0 states"):
        with runtime.transaction([handle], _layout("req-a")) as transaction:
            transaction.commit()

    assert handle.invalidated is True
    assert adapter.events[-2:] == [
        ("invalidate", ("req-a",)),
        ("deactivate", ("req-a",)),
    ]


@pytest.mark.parametrize("operation", ["activate", "capture", "invalidate", "deactivate"])
def test_adapter_failure_makes_handle_unusable(operation):
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE, fail_operation=operation)
    runtime = RequestScopedCacheRuntime(adapter)
    handle = runtime.open_request(_metadata("req-a"))

    with pytest.raises(RuntimeError, match=rf"{operation} failed"):
        if operation == "invalidate":
            runtime.invalidate_request(handle)
        else:
            with runtime.transaction([handle], _layout("req-a")) as transaction:
                transaction.commit()

    assert handle.invalidated is True
    with pytest.raises(RuntimeError, match="invalidated"):
        runtime.transaction([handle], _layout("req-a"))


def test_runtime_rejects_nested_transactions():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with runtime.transaction([handle_a], _layout("req-a")) as transaction:
        with pytest.raises(RuntimeError, match="Nested"):
            with runtime.transaction([handle_b], _layout("req-b")):
                pass
        transaction.commit()

    assert handle_a.invalidated is False
    assert handle_b.invalidated is False


def test_batch_native_runtime_commits_a_multi_request_cohort():
    adapter = _RecordingAdapter(CacheStateScope.BATCH_NATIVE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with runtime.transaction([handle_a, handle_b], _layout("req-a", "req-b")) as transaction:
        transaction.commit()

    assert handle_a.opaque_state["version"] == 1
    assert handle_b.opaque_state["version"] == 1
    assert runtime.capabilities.supports_packed_subset is True


def test_non_batch_native_runtime_rejects_a_multi_request_cohort():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with pytest.raises(RuntimeError, match="cannot activate a cohort"):
        runtime.transaction([handle_a, handle_b], _layout("req-a", "req-b"))


def test_transaction_rejects_handle_order_that_disagrees_with_layout():
    adapter = _RecordingAdapter(CacheStateScope.BATCH_NATIVE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle_a = runtime.open_request(_metadata("req-a"))
    handle_b = runtime.open_request(_metadata("req-b"))

    with pytest.raises(ValueError, match="must follow row_layout"):
        runtime.transaction([handle_b, handle_a], _layout("req-a", "req-b"))


def test_close_marks_handle_stale_and_preserves_reason():
    adapter = _RecordingAdapter(CacheStateScope.REQUEST_SWAPPABLE)
    runtime = RequestScopedCacheRuntime(adapter)
    handle = runtime.open_request(_metadata("req-a"))

    stats = runtime.close_request(handle, CacheCloseReason.ABORTED)

    assert stats == {"closed": "req-a", "reason": "aborted"}
    assert handle.closed is True
    with pytest.raises(ValueError, match="closed or stale"):
        runtime.transaction([handle], _layout("req-a"))


class _FakeCacheBackend:
    def __init__(self) -> None:
        self.enabled = True
        self.refresh_calls = []

    def is_enabled(self):
        return self.enabled

    def refresh(self, pipeline, num_inference_steps, verbose=True):
        self.refresh_calls.append((pipeline, num_inference_steps, verbose))


def test_exclusive_adapter_refreshes_once_and_rejects_overlapping_trajectory():
    backend = _FakeCacheBackend()
    pipeline = SimpleNamespace()
    runtime = RequestScopedCacheRuntime(ExclusiveCacheAdapter(backend, pipeline))
    handle = runtime.open_request(_metadata("req-a"))

    assert backend.refresh_calls == [(pipeline, 20, True)]
    with pytest.raises(RuntimeError, match="already owns"):
        runtime.open_request(_metadata("req-b"))

    with runtime.transaction([handle], _layout("req-a")) as transaction:
        transaction.commit()
    runtime.close_request(handle, CacheCloseReason.FINISHED)

    second = runtime.open_request(_metadata("req-b"))
    assert second.request_id == "req-b"
    assert backend.refresh_calls == [(pipeline, 20, True), (pipeline, 20, True)]


def test_exclusive_adapter_rejects_disabled_backend():
    backend = _FakeCacheBackend()
    backend.enabled = False
    runtime = RequestScopedCacheRuntime(ExclusiveCacheAdapter(backend, SimpleNamespace()))

    with pytest.raises(RuntimeError, match="disabled"):
        runtime.open_request(_metadata("req-a"))
