# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass


def _validate_signature(signature: Hashable) -> None:
    if signature is None:
        raise ValueError("step execution signature must not be None")
    try:
        hash(signature)
    except TypeError as exc:
        raise TypeError("step execution signature must be hashable") from exc


def _validate_service_time(service_time_ms: float) -> float:
    value = float(service_time_ms)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"step service time must be finite and non-negative, got {service_time_ms!r}")
    return value


@dataclass(frozen=True)
class StepCostObservation:
    """Measured service time for one successfully committed logical step.

    The producer owns the execution signature. It must change whenever an
    input shape, CFG mode, LoRA, cache backend, or parallel layout change
    makes older measurements incomparable.
    """

    service_time_ms: float
    execution_signature: Hashable

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_time_ms", _validate_service_time(self.service_time_ms))
        _validate_signature(self.execution_signature)


@dataclass
class RequestStepCost:
    """Online arithmetic mean of observed step service time for one request."""

    observed_steps: int = 0
    mean_service_time_ms: float = math.inf
    last_service_time_ms: float | None = None
    execution_signature: Hashable | None = None

    @property
    def is_observed(self) -> bool:
        return self.observed_steps > 0

    def reset(self, execution_signature: Hashable) -> None:
        _validate_signature(execution_signature)
        self.observed_steps = 0
        self.mean_service_time_ms = math.inf
        self.last_service_time_ms = None
        self.execution_signature = execution_signature

    def observe(self, observation: StepCostObservation) -> None:
        if observation.execution_signature != self.execution_signature:
            self.reset(observation.execution_signature)

        self.observed_steps += 1
        self.last_service_time_ms = observation.service_time_ms
        if self.observed_steps == 1:
            self.mean_service_time_ms = observation.service_time_ms
            return

        self.mean_service_time_ms += (observation.service_time_ms - self.mean_service_time_ms) / self.observed_steps


@dataclass(frozen=True)
class StepCostCandidate:
    """A ready request and its age in scheduler ticks."""

    request_id: str
    cost: RequestStepCost
    age_ticks: int = 0

    def __post_init__(self) -> None:
        if self.age_ticks < 0:
            raise ValueError(f"age_ticks must be non-negative, got {self.age_ticks}")


def order_step_cost_candidates(
    candidates: Iterable[StepCostCandidate],
    *,
    aging_credit_ms_per_tick: float = 0.0,
) -> list[str]:
    """Return stable unknown-first, shortest-observed-cost order.

    Aging subtracts a bounded-at-zero credit from known costs so an expensive
    request eventually outranks newer cheap requests. Input order breaks ties.
    This policy consumes observations only; it never probes a cache backend.
    """

    aging_credit = _validate_service_time(aging_credit_ms_per_tick)

    def priority(indexed: tuple[int, StepCostCandidate]) -> tuple[int, float, int]:
        index, candidate = indexed
        if not candidate.cost.is_observed:
            return (0, 0.0, index)
        effective_cost = max(
            0.0,
            candidate.cost.mean_service_time_ms - candidate.age_ticks * aging_credit,
        )
        return (1, effective_cost, index)

    ordered = sorted(enumerate(candidates), key=priority)
    return [candidate.request_id for _, candidate in ordered]
