# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _avoid_socket_setup(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def _queued_config(**overrides) -> OmniDiffusionConfig:
    values = {
        "model": "test",
        "mode": "queued",
        "step_execution": True,
        "max_num_seqs": 1,
        "parallel_config": DiffusionParallelConfig(pipeline_parallel_size=2),
    }
    values.update(overrides)
    return OmniDiffusionConfig(**values)


def test_static_mode_keeps_queued_capacity_fields_inert() -> None:
    config = OmniDiffusionConfig(model="test", max_inflight_batches=0, edge_buffer_slots=-1, stage_buffer_bytes=0)

    assert config.mode == "static"
    assert config.max_inflight_batches == 0
    assert config.edge_buffer_slots == -1
    assert config.stage_buffer_bytes == 0


def test_queued_mode_requires_step_execution() -> None:
    with pytest.raises(ValueError, match="requires step_execution=True"):
        _queued_config(step_execution=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parallel_config": DiffusionParallelConfig(pipeline_parallel_size=1)}, "pipeline_parallel_size=2"),
        ({"max_num_seqs": 2}, "max_num_seqs=1"),
        ({"max_inflight_batches": 2}, "max_inflight_batches=1"),
    ],
)
def test_queued_mode_rejects_out_of_scope_capacity_or_topology(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _queued_config(**overrides)


@pytest.mark.parametrize("field", ["max_inflight_batches", "edge_buffer_slots"])
def test_capacity_counts_are_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _queued_config(**{field: 0})


def test_stage_buffer_bytes_must_be_positive_when_explicit() -> None:
    with pytest.raises(ValueError, match="stage_buffer_bytes"):
        _queued_config(stage_buffer_bytes=0)


def test_valid_queued_contract_is_accepted_before_engine_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        OmniDiffusionConfig,
        "_resolve_master_port",
        lambda _self: 29500,
    )
    config = _queued_config()

    assert config.mode == "queued"
