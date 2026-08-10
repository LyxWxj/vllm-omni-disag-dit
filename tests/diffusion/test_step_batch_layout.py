# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for explicit step request-row mappings."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.worker.batch_layout import RequestRowLayout
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _state(request_id: str, row_count: int) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
    )
    state.latents = torch.zeros((row_count, 1))
    state.timesteps = torch.tensor([1.0])
    return state


def test_contiguous_layout_splits_multiple_rows_per_request():
    layout = RequestRowLayout.from_request_row_counts(
        ["req-a", "req-b"],
        [2, 1],
    )

    split = layout.split_tensor(torch.tensor([[10.0], [11.0], [20.0]]))

    torch.testing.assert_close(split["req-a"], torch.tensor([[10.0], [11.0]]))
    torch.testing.assert_close(split["req-b"], torch.tensor([[20.0]]))


def test_contiguous_layout_split_returns_views_without_copying():
    layout = RequestRowLayout.from_request_row_counts(["req-a", "req-b"], [2, 1])
    value = torch.tensor([[10.0], [11.0], [20.0]])

    split = layout.split_tensor(value)

    split["req-a"][0, 0] = 99.0
    split["req-b"][0, 0] = 77.0
    assert value[0, 0].item() == 99.0
    assert value[2, 0].item() == 77.0


def test_layout_restores_request_local_order_from_interleaved_rows():
    layout = RequestRowLayout(
        request_ids=("req-a", "req-b"),
        row_to_request=(1, 0, 0),
        row_to_request_row=(0, 1, 0),
    )

    value = torch.tensor([[20.0], [11.0], [10.0]])
    split = layout.split_tensor(value)

    torch.testing.assert_close(split["req-a"], torch.tensor([[10.0], [11.0]]))
    torch.testing.assert_close(split["req-b"], torch.tensor([[20.0]]))
    split["req-a"][0, 0] = 99.0
    torch.testing.assert_close(value, torch.tensor([[20.0], [11.0], [10.0]]))


@pytest.mark.parametrize(
    ("row_to_request", "row_to_request_row", "message"),
    [
        ((0, 2), (0, 0), "out of range"),
        ((0, 0), (0, 0), "must be a permutation"),
        ((0, 0), (0, 2), "must be a permutation"),
    ],
)
def test_layout_rejects_invalid_row_ownership(row_to_request, row_to_request_row, message):
    with pytest.raises(ValueError, match=message):
        RequestRowLayout(
            request_ids=("req-a", "req-b"),
            row_to_request=row_to_request,
            row_to_request_row=row_to_request_row,
        )


def test_layout_compatibility_ignores_physical_order_but_checks_local_row_counts():
    identity = RequestRowLayout.from_request_row_counts(["req-a", "req-b"], [2, 1])
    reordered = RequestRowLayout(
        request_ids=("req-b", "req-a"),
        row_to_request=(0, 1, 1),
        row_to_request_row=(0, 1, 0),
    )

    reordered.validate_compatible(identity)

    incompatible = RequestRowLayout.from_request_row_counts(["req-a", "req-b"], [1, 2])
    with pytest.raises(ValueError, match="row count"):
        incompatible.validate_compatible(identity)


def test_layout_rejects_duplicate_request_ids_and_wrong_tensor_rows():
    with pytest.raises(ValueError, match="must be unique"):
        RequestRowLayout.from_request_row_counts(["req-a", "req-a"], [1, 1])

    layout = RequestRowLayout.from_request_row_counts(["req-a"], [2])
    with pytest.raises(ValueError, match="tensor has 1"):
        layout.split_tensor(torch.ones(1, 3))


def test_input_batch_layout_follows_selected_request_order_and_row_counts():
    states = [_state("req-a", 2), _state("req-b", 1)]

    batch = InputBatch.make_batch(
        states,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert batch.request_ids == ["req-b", "req-a"]
    assert batch.row_layout == RequestRowLayout.from_request_row_counts(
        ["req-b", "req-a"],
        [1, 2],
    )


def test_input_batch_rejects_duplicate_idx_mapping_entries():
    states = [_state("req-a", 1), _state("req-b", 1)]

    with pytest.raises(ValueError, match="must not contain duplicate"):
        InputBatch.make_batch(
            states,
            idx_mapping=torch.tensor([0, 0], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "idx_mapping",
    [
        torch.tensor([0.9], dtype=torch.float32),
        torch.tensor([False], dtype=torch.bool),
    ],
)
def test_input_batch_rejects_non_integer_idx_mapping(idx_mapping):
    with pytest.raises(TypeError, match="must use an integer dtype"):
        InputBatch.make_batch([_state("req-a", 1)], idx_mapping=idx_mapping)


def test_input_batch_checks_idx_mapping_range_before_int32_cast():
    idx_mapping = torch.tensor([2**32], dtype=torch.int64)

    with pytest.raises(ValueError, match="4294967296 is out of range"):
        InputBatch.make_batch([_state("req-a", 1)], idx_mapping=idx_mapping)
