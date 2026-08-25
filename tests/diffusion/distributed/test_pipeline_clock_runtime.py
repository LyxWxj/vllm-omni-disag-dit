# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for interleaved PP clock and P2P transport runtimes."""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.pipeline_runtime import (
    PipelineClockSimulator,
    PipelineP2PChannel,
    PipelineSlotState,
    PipelineToken,
    PipelineTransportHeader,
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

    def test_stalled_downstream_fills_credits_then_resumes_in_fifo_order(self) -> None:
        runtime = PipelineClockSimulator(num_stages=2, slots_per_edge=2)
        runtime.set_stage_ready(1, False)
        for index, token_id in enumerate(("A", "B", "C")):
            runtime.submit(_token(token_id, index))

        blocked = tuple(runtime.progress_one_clock() for _ in range(3))

        assert [record.active_stages for record in blocked] == [(0,), (0,), ()]
        assert [record.edge_credits for record in blocked] == [(1,), (0,), (0,)]
        assert runtime.edge_max_occupancy == (2,)

        runtime.set_stage_ready(1, True)
        resumed = runtime.progress_one_clock()
        runtime.run_until_idle(max_clocks=8)

        assert resumed.active_stages == (0, 1)
        assert resumed.completed_token_ids == ("A",)
        assert [token.token_id for token in runtime.completed_tokens] == ["A", "B", "C"]
        assert [token.slot_id for token in runtime.completed_tokens] == [0, 1, 0]

    def test_slot_state_transitions_prevent_overwrite(self) -> None:
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


def _run_p2p_channel_worker(
    rank: int,
    world_size: int,
    master_port: int,
    token_count: int,
    slots_per_edge: int,
    stage_one_stall_until: int,
    result_queue,
) -> None:
    """Run one rank of a tensor-only PP lane without loading a model."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        edge_groups = [
            dist.new_group([edge_rank, edge_rank + 1], backend="gloo") for edge_rank in range(world_size - 1)
        ]
        incoming = None
        if rank > 0:
            incoming = PipelineP2PChannel(
                source_rank=rank - 1,
                destination_rank=rank,
                tensor_shape=(2,),
                tensor_dtype=torch.float32,
                device="cpu",
                slots_per_edge=slots_per_edge,
                tag_base=(rank - 1) * 100,
                group=edge_groups[rank - 1],
            )

        outgoing = None
        if rank < world_size - 1:
            outgoing = PipelineP2PChannel(
                source_rank=rank,
                destination_rank=rank + 1,
                tensor_shape=(2,),
                tensor_dtype=torch.float32,
                device="cpu",
                slots_per_edge=slots_per_edge,
                tag_base=rank * 100,
                group=edge_groups[rank],
            )

        next_token_id = 0
        sent_token_ids: list[int] = []
        sent_clocks: list[int] = []
        completed: list[tuple[int, float]] = []
        saw_credit_exhaustion = False
        max_clocks = max(128, token_count * world_size * 12)

        for clock in range(max_clocks):
            if incoming is not None:
                incoming.poll()
            if outgoing is not None:
                outgoing.poll()

            if rank == 0:
                if next_token_id < token_count and outgoing is not None:
                    if outgoing.can_send:
                        header = PipelineTransportHeader(
                            token_id=next_token_id,
                            step_idx=0,
                            cfg_branch=0,
                        )
                        payload = torch.full((2,), float(next_token_id), dtype=torch.float32)
                        outgoing.send(header, payload)
                        sent_token_ids.append(next_token_id)
                        sent_clocks.append(clock)
                        next_token_id += 1
                    elif next_token_id >= slots_per_edge:
                        saw_credit_exhaustion = True
            elif rank != 1 or clock >= stage_one_stall_until:
                if incoming is not None and incoming.has_ready_message:
                    if rank == world_size - 1:
                        message = incoming.begin_compute()
                        if not message.header.flags & PipelineTransportHeader.SHUTDOWN_FLAG:
                            completed.append((message.header.token_id, float(message.payload[0])))
                        incoming.release_after_compute(message)
                    elif outgoing is not None and outgoing.can_send:
                        message = incoming.begin_compute()
                        if not message.header.flags & PipelineTransportHeader.SHUTDOWN_FLAG:
                            # Each intermediate stage mutates a fresh local
                            # output before releasing its receiver-owned input.
                            output = message.payload + rank
                            outgoing.send(message.header, output)
                        incoming.release_after_compute(message)

            # The production tick RPC is a global clock. The barrier makes the
            # CPU test deterministic and gives every rank a chance to poll
            # initial credits before rank 0 exhausts its local loop.
            dist.barrier()

        # Data is drained before shutdown starts. Keeping tombstones in a
        # separate phase prevents one stage from waiting for a downstream
        # credit while that downstream rank is still at the clock barrier.
        dist.barrier()
        shutdown_sent = False
        for _ in range(world_size + 2):
            if rank == 0:
                if not shutdown_sent:
                    outgoing.send_shutdown()
                    shutdown_sent = True
            else:
                incoming.poll()
                while incoming.has_ready_message:
                    if rank != world_size - 1 and not outgoing.can_send:
                        break
                    message = incoming.begin_compute()
                    if not message.header.flags & PipelineTransportHeader.SHUTDOWN_FLAG:
                        if rank == world_size - 1:
                            completed.append((message.header.token_id, float(message.payload[0])))
                        elif outgoing.can_send:
                            outgoing.send(message.header, message.payload + rank)
                    incoming.release_after_compute(message)
                if outgoing is not None and incoming.is_closed and not shutdown_sent:
                    outgoing.send_shutdown()
                    shutdown_sent = True
            dist.barrier()

        if incoming is not None:
            incoming.wait_for_sends()
        if outgoing is not None:
            outgoing.wait_for_sends()

        result_queue.put(
            (
                rank,
                {
                    "completed": completed,
                    "sent_token_ids": sent_token_ids,
                    "sent_clocks": sent_clocks,
                    "saw_credit_exhaustion": saw_credit_exhaustion,
                    "incoming_max_occupied": incoming.max_occupied if incoming is not None else 0,
                    "receive_histories": (
                        [[state.name for state in history] for history in incoming.receive_slot_state_histories]
                        if incoming is not None
                        else []
                    ),
                    "send_histories": (
                        [[state.name for state in history] for history in outgoing.send_slot_state_histories]
                        if outgoing is not None
                        else []
                    ),
                },
            )
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_p2p_channel_lane(
    *,
    world_size: int,
    token_count: int,
    slots_per_edge: int,
    stage_one_stall_until: int = 0,
) -> dict[int, dict[str, object]]:
    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    result_queue = manager.Queue()
    try:
        torch.multiprocessing.spawn(
            _run_p2p_channel_worker,
            args=(
                world_size,
                get_open_port(),
                token_count,
                slots_per_edge,
                stage_one_stall_until,
                result_queue,
            ),
            nprocs=world_size,
        )
        return {rank: result for rank, result in (result_queue.get() for _ in range(world_size))}
    finally:
        manager.shutdown()


class TestPipelineP2PChannel:
    def test_fixed_header_round_trips_without_object_metadata(self) -> None:
        encoded = PipelineTransportHeader(
            token_id=7,
            step_idx=3,
            cfg_branch=1,
            flags=9,
        ).for_slot(slot_id=2, send_sequence=11)
        buffer = torch.empty(PipelineTransportHeader.FIELD_COUNT, dtype=torch.int64)

        encoded.encode_into(buffer)

        assert PipelineTransportHeader.decode(buffer) == encoded
        assert buffer.numel() == PipelineTransportHeader.FIELD_COUNT

    def test_pp2_credit_backpressure_recovers_after_downstream_release(self) -> None:
        slots_per_edge = 2
        results = _run_p2p_channel_lane(
            world_size=2,
            token_count=5,
            slots_per_edge=slots_per_edge,
            stage_one_stall_until=10,
        )

        source = results[0]
        destination = results[1]
        assert source["sent_token_ids"] == [0, 1, 2, 3, 4]
        assert source["saw_credit_exhaustion"] is True
        assert source["sent_clocks"][slots_per_edge] >= 10
        assert destination["incoming_max_occupied"] == slots_per_edge
        assert [token_id for token_id, _ in destination["completed"]] == [0, 1, 2, 3, 4]

    def test_pp4_preserves_fifo_and_buffer_lifetimes(self) -> None:
        results = _run_p2p_channel_lane(
            world_size=4,
            token_count=8,
            slots_per_edge=2,
        )

        completed = results[3]["completed"]
        assert [token_id for token_id, _ in completed] == list(range(8))
        assert [value for _, value in completed] == [float(token_id + 3) for token_id in range(8)]

        receive_transitions = {
            "FREE": {"RECV_POSTED"},
            "RECV_POSTED": {"READY"},
            "READY": {"COMPUTING"},
            "COMPUTING": {"FREE"},
        }
        for rank in range(1, 4):
            for history in results[rank]["receive_histories"]:
                assert "READY" in history
                assert all(after in receive_transitions[before] for before, after in zip(history, history[1:]))

        send_transitions = {
            "FREE": {"SEND_PENDING"},
            "SEND_PENDING": {"FREE"},
        }
        for rank in range(3):
            for history in results[rank]["send_histories"]:
                assert all(after in send_transitions[before] for before, after in zip(history, history[1:]))
            assert any("SEND_PENDING" in history for history in results[rank]["send_histories"])
