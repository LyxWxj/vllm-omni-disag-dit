# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from vllm_omni.config.stage_config import (
    DiffusionStageOperator,
    DiffusionStageRole,
    resolve_diffusion_stage_role,
)
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState


class DiffusionStageMixin:
    """Dispatch stage operations to a pipeline's existing methods.

    Request-level operators use the conventional pipeline entry points
    (``forward``, ``encode_batch``, and ``decode_batch``). Step-level operators
    use the methods from the existing step-execution protocol. A pipeline only
    needs ``_stage_operator_methods`` when it uses a non-standard method name.
    """

    supports_stage_operator_dispatch = True
    _stage_operator_methods: ClassVar[Mapping[DiffusionStageOperator, str]] = {}

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
        """Validate and cache the methods required by the configured role."""
        if step_execution is None:
            od_config = getattr(self, "od_config", None)
            step_execution = bool(
                getattr(od_config, "step_execution", False) or getattr(od_config, "streaming_output", False)
            )

        stage_role = self._diffusion_stage_role()
        operators = stage_role.required_operators(step_execution=step_execution)
        resolved_methods = {}
        missing = []
        for operator in operators:
            method_name = self._resolve_stage_method_name(operator)
            method = getattr(self, method_name, None)
            if not callable(method):
                missing.append(f"{operator.value} ({method_name})")
            else:
                resolved_methods[operator] = method

        if missing:
            mode = "step-execution" if step_execution else "request-batching"
            raise TypeError(
                f"{type(self).__name__} does not implement the required {mode} methods for "
                f"role {stage_role.value!r}: {', '.join(sorted(missing))}."
            )

        self._stage_contract_operators = operators
        self._stage_operator_callables = resolved_methods
        self._stage_execution_initialized = True

    def run_stage(
        self,
        payload: DiffusionRequestBatch | InputBatch | StepRequestState,
        *,
        operator: DiffusionStageOperator | str | None = None,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one role-authorized request-level or step-level operation."""
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

        if is_request_payload and payload.is_dummy_run():
            prepare_dummy = getattr(self, "_prepare_dummy_stage_payload", None)
            if callable(prepare_dummy):
                prepare_dummy(payload, operator=operator)

        if operator is DiffusionStageOperator.DENOISE_STEP and states is not None:
            kwargs["states"] = states

        method = getattr(self, "_stage_operator_callables", {}).get(operator)
        if method is None:
            method = getattr(self, self._resolve_stage_method_name(operator))
        return method(payload, **kwargs)

    def step_output(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Return denoise latents for a downstream decode stage."""
        del kwargs
        if state.latents is None:
            raise ValueError("Denoise stage has no latents to transfer.")
        return DiffusionOutput(
            output=None,
            custom_output={"latents": state.latents},
            to_cpu=True,
        )

    def _resolve_stage_method_name(self, operator: DiffusionStageOperator) -> str:
        override = self._stage_operator_methods.get(operator)
        if override is not None:
            return override

        if operator is DiffusionStageOperator.FORWARD:
            return "forward"
        if operator is DiffusionStageOperator.ENCODE:
            return "encode_batch" if callable(getattr(self, "encode_batch", None)) else "encode"
        if operator is DiffusionStageOperator.DIFFUSE:
            return "diffuse_batch" if callable(getattr(self, "diffuse_batch", None)) else "forward"
        if operator is DiffusionStageOperator.DECODE:
            return "decode_batch" if callable(getattr(self, "decode_batch", None)) else "decode"
        return operator.value

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
