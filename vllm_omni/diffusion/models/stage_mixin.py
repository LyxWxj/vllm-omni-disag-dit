# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm_omni.config.stage_config import (
    DiffusionStageOperator,
    DiffusionStageRole,
    resolve_diffusion_stage_role,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState


class DiffusionStageMixin:
    """Facade for role-authorized diffusion stage operations.

    The mixin owns only dispatch and capability validation. A pipeline supplies
    the primitive methods named by :class:`DiffusionStageOperator`, while the
    runner remains responsible for step scheduling and request state changes.
    """

    supports_stage_operator_dispatch = True

    def __init__(self, *args: Any, od_config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if od_config is not None:
            self.od_config = od_config
            self.stage_role = resolve_diffusion_stage_role(
                getattr(od_config, "stage_role", None),
                getattr(od_config, "model_stage", None),
            )
            self.initialize_stage_execution()

    def initialize_stage_execution(self, *, step_execution: bool | None = None) -> None:
        """Validate the primitive contract during pipeline initialization."""
        if step_execution is None:
            od_config = getattr(self, "od_config", None)
            step_execution = bool(
                getattr(od_config, "step_execution", False) or getattr(od_config, "streaming_output", False)
            )
        stage_role = self._diffusion_stage_role()
        operators = stage_role.step_operators if step_execution else frozenset({stage_role.default_request_operator})
        missing = sorted(
            {
                operator.primitive_name
                for operator in operators
                if not callable(getattr(self, operator.primitive_name, None))
            }
        )
        if missing:
            mode = "step-execution" if step_execution else "request-batching"
            raise TypeError(
                f"{type(self).__name__} does not implement the required {mode} primitives for "
                f"role {stage_role.value!r}: {', '.join(missing)}."
            )
        self._stage_contract_operators = operators
        self._stage_execution_initialized = True

    def run_stage(
        self,
        payload: DiffusionRequestBatch | InputBatch | StepRequestState,
        *,
        operator: DiffusionStageOperator | str | None = None,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one role-authorized request-level or step-level operation.

        Request-batching callers may omit ``operator`` and use the role's
        compatibility operator. Step-batching callers must provide the
        operator because one role can span multiple step operations.
        """
        stage_role = self._diffusion_stage_role()
        is_request_payload = isinstance(payload, DiffusionRequestBatch)
        operator = self._resolve_operator(stage_role, operator, is_request_payload)

        if operator.is_request_operator != is_request_payload:
            mode = "request" if is_request_payload else "step"
            expected = "request" if operator.is_request_operator else "step"
            raise TypeError(f"{operator.value} is a {expected} operator, but received a {mode} payload.")

        contract_operators = getattr(self, "_stage_contract_operators", None)
        if contract_operators is not None and operator not in contract_operators:
            raise ValueError(
                f"Operator {operator.value!r} is not enabled for role {stage_role.value!r} "
                f"in the configured execution mode."
            )

        missing_stages = operator.required_stages - stage_role.stages
        if missing_stages:
            missing = ", ".join(sorted(stage.value for stage in missing_stages))
            raise ValueError(
                f"Stage role {stage_role.value!r} does not support operator {operator.value!r}; "
                f"missing stages: {missing}."
            )

        if is_request_payload and payload.is_dummy_run():
            prepare_dummy = getattr(self, "_prepare_dummy_stage_payload", None)
            if callable(prepare_dummy):
                prepare_dummy(payload, operator=operator)

        if operator is DiffusionStageOperator.DENOISE_STEP and states is not None:
            kwargs["states"] = states

        return getattr(self, operator.primitive_name)(payload, **kwargs)

    def _resolve_operator(
        self,
        stage_role: DiffusionStageRole,
        operator: DiffusionStageOperator | str | None,
        is_request_payload: bool,
    ) -> DiffusionStageOperator:
        if operator is None:
            if not is_request_payload:
                raise ValueError("operator is required for step-level stage execution.")
            return stage_role.default_request_operator
        if isinstance(operator, DiffusionStageOperator):
            return operator
        try:
            return DiffusionStageOperator(operator)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown diffusion stage operator: {operator!r}.") from exc

    def _diffusion_stage_role(self) -> DiffusionStageRole:
        """Resolve the role from the pipeline or its stage configuration."""
        stage_role = getattr(self, "stage_role", None)
        od_config = getattr(self, "od_config", None)
        model_stage = getattr(self, "model_stage", None) or getattr(od_config, "model_stage", None)
        return resolve_diffusion_stage_role(stage_role, model_stage)
