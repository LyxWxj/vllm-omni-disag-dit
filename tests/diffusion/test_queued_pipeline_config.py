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
    config = OmniDiffusionConfig(model="test", max_inflight_batches=0)

    assert config.mode == "static"
    assert config.max_inflight_batches == 0


def test_queued_mode_requires_step_execution() -> None:
    with pytest.raises(ValueError, match="requires step_execution=True"):
        _queued_config(step_execution=False)


def test_max_inflight_batches_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_inflight_batches"):
        _queued_config(max_inflight_batches=0)


def test_queued_mode_accepts_multiple_scheduler_requests() -> None:
    config = _queued_config(max_num_seqs=2)

    assert config.max_num_seqs == 2


def test_queued_mode_accepts_multiple_inflight_batches() -> None:
    config = _queued_config(max_inflight_batches=2)

    assert config.max_inflight_batches == 2


def test_vae_patch_parallel_size_cannot_exceed_pipeline_size() -> None:
    with pytest.raises(ValueError, match="vae_patch_parallel_size cannot exceed pipeline_parallel_size"):
        DiffusionParallelConfig(pipeline_parallel_size=2, vae_patch_parallel_size=3)


def test_valid_queued_contract_is_accepted_before_engine_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        OmniDiffusionConfig,
        "_resolve_master_port",
        lambda _self: 29500,
    )
    config = _queued_config()

    assert config.mode == "queued"
