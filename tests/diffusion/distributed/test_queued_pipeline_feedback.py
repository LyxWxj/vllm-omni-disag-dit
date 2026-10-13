# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import group_coordinator as group_coordinator_module
from vllm_omni.diffusion.distributed.group_coordinator import PipelineGroupCoordinator
from vllm_omni.diffusion.queued_pp.feedback import (
    prune_queued_latent_sends,
    queued_feedback_device_group,
    receive_queued_latents,
    send_queued_latents,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Work:
    def __init__(self, completed: bool = False):
        self.completed = completed
        self.waited = False

    def is_completed(self):
        return self.completed

    def wait(self):
        self.waited = True


def _pp_group():
    activation, feedback = object(), object()
    return SimpleNamespace(
        world_size=2,
        ranks=[7, 9],
        device_group=activation,
        device_groups=[activation, feedback],
    )


def test_queued_feedback_uses_reverse_pp2_group() -> None:
    pp_group = _pp_group()
    assert queued_feedback_device_group(pp_group) is pp_group.device_groups[1]


def test_pipeline_coordinator_keeps_base_broadcast_contract(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "new_group", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        group_coordinator_module,
        "current_omni_platform",
        SimpleNamespace(get_torch_device=lambda _rank: torch.device("cpu")),
    )

    coordinator = PipelineGroupCoordinator([[0, 1]], local_rank=0, torch_distributed_backend="gloo")

    assert coordinator.shm_broadcaster is None


def test_queued_send_retains_latent_until_reverse_send_completes(monkeypatch) -> None:
    pp_group = _pp_group()
    pipeline = SimpleNamespace(_pp_send_work=[], _pp_send_tensors=[])
    latent = torch.ones(1, 2)
    work = _Work()
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "isend",
        lambda tensor, dst, group: calls.append((tensor, dst, group)) or work,
    )

    send_queued_latents(pipeline, pp_group, latent)

    assert calls == [(latent, 7, pp_group.device_groups[1])]
    assert pipeline._pp_send_work == [work]
    assert pipeline._pp_send_tensors == [latent]
    prune_queued_latent_sends(pipeline)
    assert pipeline._pp_send_tensors == [latent]
    work.completed = True
    prune_queued_latent_sends(pipeline)
    assert pipeline._pp_send_work == []
    assert pipeline._pp_send_tensors == []


def test_queued_receive_posts_on_reverse_group_and_resolves_lazily(monkeypatch) -> None:
    pp_group = _pp_group()
    pipeline = SimpleNamespace()
    template = torch.zeros(1, 2)
    work = _Work()
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "irecv",
        lambda tensor, src, group: calls.append((tensor, src, group)) or work,
    )

    pending = receive_queued_latents(pipeline, pp_group, template)

    assert calls[0][1:] == (9, pp_group.device_groups[1])
    assert pending._tensor is None
    resolved = pending.resolve()
    assert resolved is calls[0][0]
    assert work.waited
