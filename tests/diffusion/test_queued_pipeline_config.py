# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.queued_pp.queue_config import QueuedPipelineConfig

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


def test_static_mode_leaves_queued_capacity_inert() -> None:
    config = OmniDiffusionConfig(model="test", max_inflight_batches=0)

    assert config.mode == "static"
    assert not QueuedPipelineConfig.from_config(config).enabled


def test_queued_mode_requires_step_execution() -> None:
    with pytest.raises(ValueError, match="requires step_execution=True"):
        _queued_config(step_execution=False)


def test_queued_mode_requires_pipeline_parallelism() -> None:
    with pytest.raises(ValueError, match="pipeline_parallel_size greater than 1"):
        _queued_config(parallel_config=DiffusionParallelConfig())


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_queued_capacity_must_be_positive_integer(capacity) -> None:
    with pytest.raises(ValueError, match="max_inflight_batches"):
        _queued_config(max_inflight_batches=capacity)


def test_queued_config_accepts_multiple_requests_and_inflight_batches() -> None:
    config = _queued_config(max_num_seqs=2, max_inflight_batches=4)

    assert config.max_num_seqs == 2
    assert QueuedPipelineConfig.from_config(config) == QueuedPipelineConfig(enabled=True, max_inflight_batches=4)


def test_unknown_execution_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be 'static' or 'queued'"):
        OmniDiffusionConfig(model="test", mode="compact")
