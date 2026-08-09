# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    RequestStepCost,
    StepCostCandidate,
    StepCostObservation,
    StepScheduler,
    order_step_cost_candidates,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _observed_cost(value: float, signature: object = ("shape-a", "cache-a")) -> RequestStepCost:
    cost = RequestStepCost()
    cost.observe(StepCostObservation(value, signature))
    return cost


def _request(request_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="prompt",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=3),
        request_id=request_id,
    )


class TestRequestStepCost:
    def test_starts_unknown_and_uses_arithmetic_mean(self) -> None:
        cost = RequestStepCost()

        assert cost.observed_steps == 0
        assert math.isinf(cost.mean_service_time_ms)
        assert cost.is_observed is False

        cost.observe(StepCostObservation(12.0, ("shape-a", "teacache")))
        cost.observe(StepCostObservation(18.0, ("shape-a", "teacache")))

        assert cost.observed_steps == 2
        assert cost.mean_service_time_ms == 15.0
        assert cost.last_service_time_ms == 18.0

    def test_signature_change_discards_incomparable_history(self) -> None:
        cost = _observed_cost(10.0, ("shape-a", "teacache", "tp1"))
        cost.observe(StepCostObservation(30.0, ("shape-b", "cache-dit", "tp2")))

        assert cost.observed_steps == 1
        assert cost.mean_service_time_ms == 30.0
        assert cost.execution_signature == ("shape-b", "cache-dit", "tp2")

    @pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
    def test_rejects_invalid_service_time(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            StepCostObservation(value, "signature")

    def test_requires_hashable_non_null_signature(self) -> None:
        with pytest.raises(ValueError, match="must not be None"):
            StepCostObservation(1.0, None)
        with pytest.raises(TypeError, match="must be hashable"):
            StepCostObservation(1.0, ["shape"])


class TestObservedCostOrdering:
    def test_unknown_requests_run_before_known_costs_stably(self) -> None:
        ordered = order_step_cost_candidates(
            [
                StepCostCandidate("known", _observed_cost(1.0)),
                StepCostCandidate("unknown-a", RequestStepCost()),
                StepCostCandidate("unknown-b", RequestStepCost()),
            ]
        )

        assert ordered == ["unknown-a", "unknown-b", "known"]

    def test_older_unknown_request_runs_first(self) -> None:
        ordered = order_step_cost_candidates(
            [
                StepCostCandidate("newer", RequestStepCost(), age_ticks=1),
                StepCostCandidate("older", RequestStepCost(), age_ticks=5),
            ]
        )

        assert ordered == ["older", "newer"]

    def test_known_requests_are_ordered_by_mean_cost(self) -> None:
        ordered = order_step_cost_candidates(
            [
                StepCostCandidate("slow", _observed_cost(20.0)),
                StepCostCandidate("fast", _observed_cost(5.0)),
            ]
        )

        assert ordered == ["fast", "slow"]

    def test_aging_eventually_promotes_an_expensive_request(self) -> None:
        ordered = order_step_cost_candidates(
            [
                StepCostCandidate("slow-old", _observed_cost(20.0), age_ticks=10),
                StepCostCandidate("fast-new", _observed_cost(5.0)),
            ],
            aging_credit_ms_per_tick=2.0,
        )

        assert ordered == ["slow-old", "fast-new"]


class TestStepSchedulerCostCommit:
    def test_commits_observation_only_after_valid_step_output(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=1))
        request_id = scheduler.add_request(_request("req"))
        scheduled = scheduler.schedule()

        scheduler.update_from_output(
            scheduled,
            RunnerOutput(
                request_id=request_id,
                step_index=1,
                step_cost_observation=StepCostObservation(8.0, ("shape", "teacache", "tp2")),
            ),
        )

        cost = scheduler.get_step_cost(request_id)
        assert cost is not None
        assert cost.observed_steps == 1
        assert cost.mean_service_time_ms == 8.0

    def test_commits_multiple_deferred_observations_for_an_active_request(self) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=2, cache_backend="tea_cache"))
        request_a = scheduler.add_request(_request("a"))
        request_b = scheduler.add_request(_request("b"))
        first = scheduler.schedule()
        scheduler.update_from_output(first, RunnerOutput(request_id=request_a, step_index=1))
        second = scheduler.schedule()

        scheduler.update_from_output(
            second,
            BatchRunnerOutput.from_list(
                [RunnerOutput(request_id=request_b, step_index=1)],
                step_cost_observations=[
                    (request_a, StepCostObservation(10.0, "signature")),
                    (request_a, StepCostObservation(20.0, "signature")),
                ],
            ),
        )

        cost = scheduler.get_step_cost(request_a)
        assert cost is not None
        assert cost.observed_steps == 2
        assert cost.mean_service_time_ms == 15.0

    @pytest.mark.parametrize(
        "output",
        [
            RunnerOutput(
                request_id="req",
                step_index=None,
                step_cost_observation=StepCostObservation(8.0, "signature"),
            ),
            RunnerOutput(
                request_id="req",
                step_index=1,
                result=DiffusionOutput(output=None, error="failed"),
                step_cost_observation=StepCostObservation(8.0, "signature"),
            ),
        ],
    )
    def test_rejected_step_does_not_commit_observation(self, output: RunnerOutput) -> None:
        scheduler = StepScheduler()
        scheduler.initialize(SimpleNamespace(max_num_seqs=1))
        request_id = scheduler.add_request(_request("req"))
        scheduled = scheduler.schedule()

        scheduler.update_from_output(scheduled, output)

        cost = scheduler.get_step_cost(request_id)
        assert cost is not None
        assert cost.observed_steps == 0
        assert math.isinf(cost.mean_service_time_ms)
