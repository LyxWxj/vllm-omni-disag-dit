# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the interleaved PP clock reference model."""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.distributed.pipeline_runtime import (
    PipelineClockSimulator,
    PipelineSlotState,
    PipelineToken,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _token(token_id: str, microbatch_id: int, *, compatibility_key: tuple[str, int] = ("wan22", 2)) -> PipelineToken:
    return PipelineToken(
        request_ids=(f"request-{token_id}",),
        row_map=(0,),
        step_idx=0,
        cfg_branch="positive",
        microbatch_id=microbatch_id,
        token_id=token_id,
        slot_id=None,
        compatibility_key=compatibility_key,
        model_phase="denoise",
    )


class TestPipelineClockSimulator:
    def test_pp4_fills_all_stages_and_completes_every_token_once(self) -> None:
        runtime = PipelineClockSimulator(num_stages=4, slots_per_edge=2)
        for index, token_id in enumerate(("A", "B", "C", "D")):
            runtime.submit(_token(token_id, index))

        records = runtime.run_until_idle(max_clocks=16)

        assert [record.active_stages for record in records[:4]] == [
            (0,),
            (0, 1),
            (0, 1, 2),
            (0, 1, 2, 3),
        ]
        assert [token.token_id for token in runtime.completed_tokens] == ["A", "B", "C", "D"]
        assert runtime.is_idle

    def test_dynamic_admission_enters_while_the_first_token_is_in_flight(self) -> None:
        runtime = PipelineClockSimulator(num_stages=3, slots_per_edge=2)
        runtime.submit(_token("A", 0))

        first = runtime.progress_one_clock()
        runtime.submit(_token("B", 1))
        records = (first, *runtime.run_until_idle(max_clocks=12))

        assert records[0].active_stages == (0,)
        assert records[1].active_stages == (0, 1)
        assert [token.token_id for token in runtime.completed_tokens] == ["A", "B"]

    def test_slot_capacity_and_state_transitions_prevent_overwrite(self) -> None:
        runtime = PipelineClockSimulator(num_stages=4, slots_per_edge=1)
        for index in range(8):
            runtime.submit(_token(str(index), index))

        runtime.run_until_idle(max_clocks=32)

        assert runtime.edge_max_occupancy == (1, 1, 1)
        allowed_transitions = {
            PipelineSlotState.FREE: {PipelineSlotState.RECV_POSTED},
            PipelineSlotState.RECV_POSTED: {PipelineSlotState.SEND_PENDING},
            PipelineSlotState.SEND_PENDING: {PipelineSlotState.READY},
            PipelineSlotState.READY: {PipelineSlotState.COMPUTING},
            PipelineSlotState.COMPUTING: {PipelineSlotState.FREE},
        }
        for edge_histories in runtime.slot_state_histories:
            for history in edge_histories:
                for before, after in zip(history, history[1:]):
                    assert after in allowed_transitions[before]

        assert [token.token_id for token in runtime.completed_tokens] == [str(index) for index in range(8)]

    def test_only_one_compatibility_cohort_can_be_in_flight(self) -> None:
        runtime = PipelineClockSimulator(num_stages=2)
        runtime.submit(_token("A", 0, compatibility_key=("wan22", 2)))

        with pytest.raises(ValueError, match="homogeneous compatibility cohort"):
            runtime.submit(_token("B", 1, compatibility_key=("wan22", 4)))

        runtime.run_until_idle(max_clocks=8)
        runtime.submit(_token("C", 2, compatibility_key=("wan22", 4)))
        runtime.run_until_idle(max_clocks=8)
        assert [token.token_id for token in runtime.completed_tokens] == ["A", "C"]

    def test_duplicate_token_is_rejected_before_it_can_be_lost(self) -> None:
        runtime = PipelineClockSimulator(num_stages=2)
        runtime.submit(_token("A", 0))

        with pytest.raises(ValueError, match="already admitted"):
            runtime.submit(_token("A", 1))
