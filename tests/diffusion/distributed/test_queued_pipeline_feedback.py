# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import group_coordinator as group_coordinator_module
from vllm_omni.diffusion.distributed.group_coordinator import PipelineGroupCoordinator
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin
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


def _rank_scheduler_pipeline() -> SimpleNamespace:
    pipeline = SimpleNamespace(_queued_pipeline_async_sends=True)
    pipeline._receive_scheduler_latents = MethodType(PipelineParallelMixin._receive_scheduler_latents, pipeline)
    pipeline.receive_scheduler_feedback = MethodType(PipelineParallelMixin.receive_scheduler_feedback, pipeline)
    pipeline.advance_step_for_pipeline_rank = MethodType(
        PipelineParallelMixin.advance_step_for_pipeline_rank,
        pipeline,
    )
    return pipeline


def test_rank_scheduler_owner_runs_solver_only_on_last_rank(monkeypatch) -> None:
    pipeline_parallel_module = __import__(
        "vllm_omni.diffusion.distributed.pipeline_parallel",
        fromlist=["get_pp_group"],
    )
    group = SimpleNamespace(world_size=2, is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(pipeline_parallel_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_parallel_module, "get_pp_group", lambda: group)
    scheduler_calls = []
    pipeline = _rank_scheduler_pipeline()

    def step_scheduler(state, noise_pred):
        scheduler_calls.append((state, noise_pred))
        state.step_index += 1

    pipeline.step_scheduler = step_scheduler
    state = SimpleNamespace(request_id="req-a", latents=torch.zeros(1, 2), step_index=0)
    noise_pred = torch.ones_like(state.latents)

    pipeline.advance_step_for_pipeline_rank(state, noise_pred)

    assert scheduler_calls == [(state, noise_pred)]
    assert state.step_index == 1


def test_rank_scheduler_helper_keeps_single_rank_pipeline_compatible(monkeypatch) -> None:
    pipeline_parallel_module = __import__(
        "vllm_omni.diffusion.distributed.pipeline_parallel",
        fromlist=["get_pipeline_parallel_world_size"],
    )
    def missing_pp_group():
        raise AssertionError("PP group is not initialized")

    monkeypatch.setattr(pipeline_parallel_module, "get_pipeline_parallel_world_size", missing_pp_group)
    pipeline = _rank_scheduler_pipeline()
    scheduler_calls = []

    def step_scheduler(state, noise_pred):
        scheduler_calls.append((state, noise_pred))
        state.step_index += 1

    pipeline.step_scheduler = step_scheduler
    state = SimpleNamespace(request_id="req-a", latents=torch.zeros(1, 2), step_index=0)
    noise_pred = torch.ones_like(state.latents)

    pipeline.advance_step_for_pipeline_rank(state, noise_pred)

    assert scheduler_calls == [(state, noise_pred)]
    assert state.step_index == 1


def test_first_rank_receives_feedback_and_advances_without_solver(monkeypatch) -> None:
    pipeline_parallel_module = __import__(
        "vllm_omni.diffusion.distributed.pipeline_parallel",
        fromlist=["get_pp_group"],
    )
    feedback_module = __import__(
        "vllm_omni.diffusion.queued_pp.feedback",
        fromlist=["receive_queued_latents"],
    )
    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr(pipeline_parallel_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_parallel_module, "get_pp_group", lambda: group)
    pipeline = _rank_scheduler_pipeline()
    scheduler_calls = []
    receive_calls = []
    received = torch.full((1, 2), 3.0)
    monkeypatch.setattr(
        feedback_module,
        "receive_queued_latents",
        lambda _pipeline, _group, _latents: receive_calls.append(True) or received,
    )
    pipeline.step_scheduler = lambda *args: scheduler_calls.append(args)
    state = SimpleNamespace(request_id="req-a", latents=torch.zeros(1, 2), step_index=0)

    pipeline.advance_step_for_pipeline_rank(state, torch.ones_like(state.latents))

    assert scheduler_calls == []
    assert receive_calls == [True]
    assert state.latents is received
    assert state.step_index == 1
