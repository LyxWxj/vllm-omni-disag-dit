# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for interleaved PP clock and P2P transport runtimes."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.runtime import get_distributed_init_method, get_open_port
from vllm_omni.diffusion.distributed import pipeline_runtime as pipeline_runtime_module
from vllm_omni.diffusion.distributed.pipeline_runtime import (
    PipelineClockSimulator,
    PipelineP2PChannel,
    PipelineSlotState,
    PipelineTensorSpec,
    PipelineTickRuntime,
    PipelineToken,
    PipelineTransportHeader,
    pipeline_edge_pairs,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DeferredReleaseEvent:
    """Device-event stand-in that keeps one receive slot unavailable."""

    def __init__(self, pending_queries: int) -> None:
        self.pending_queries = pending_queries
        self.query_count = 0

    def query(self) -> bool:
        self.query_count += 1
        if self.pending_queries > 0:
            self.pending_queries -= 1
            return False
        return True


def _transport_buffer_inference_flags(channel: PipelineP2PChannel | None) -> list[bool]:
    if channel is None:
        return []
    if channel.is_source:
        buffers = [
            *[slot.header_buffer for slot in channel._send_slots],
            *[slot.payload_buffer for slot in channel._send_slots],
            *channel._credit_buffers,
        ]
    else:
        buffers = [
            *[slot.header_buffer for slot in channel._receive_slots],
            *[slot.payload_buffer for slot in channel._receive_slots],
            *channel._credit_buffers,
        ]
    return [buffer.is_inference() for buffer in buffers]


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
    def test_pp_edges_include_the_feedback_lane(self) -> None:
        assert pipeline_edge_pairs((4, 7, 9, 11)) == ((4, 7), (7, 9), (9, 11), (11, 4))

        with pytest.raises(ValueError, match="at least two ranks"):
            pipeline_edge_pairs((4,))

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
    rebuild_after_close: bool,
    inject_prepare_failure: bool,
    deferred_release_queries: int,
    construct_inside_inference_mode: bool,
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
            tuple(dist.new_group([edge_rank, edge_rank + 1], backend="gloo") for _ in range(slots_per_edge))
            for edge_rank in range(world_size - 1)
        ]
        credit_edge_groups = [
            dist.new_group([edge_rank, edge_rank + 1], backend="gloo") for edge_rank in range(world_size - 1)
        ]
        channel_context = torch.inference_mode() if construct_inside_inference_mode else nullcontext()
        with channel_context:
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
                    group=edge_groups[rank - 1][0],
                    slot_groups=edge_groups[rank - 1],
                    credit_group=credit_edge_groups[rank - 1],
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
                    group=edge_groups[rank][0],
                    slot_groups=edge_groups[rank],
                    credit_group=credit_edge_groups[rank],
                )

        release_events: list[_DeferredReleaseEvent] = []
        credit_return_event_queries: list[int] = []
        if incoming is not None and rank == 1 and deferred_release_queries:

            def record_compute_event() -> _DeferredReleaseEvent:
                event = _DeferredReleaseEvent(deferred_release_queries if not release_events else 0)
                release_events.append(event)
                return event

            incoming._record_compute_event = record_compute_event
            original_send_credit = incoming._send_credit

            def send_credit(slot_id: int) -> None:
                credit_return_event_queries.append(release_events[-1].query_count)
                original_send_credit(slot_id)

            incoming._send_credit = send_credit

        prepare_failure_preserved = not inject_prepare_failure
        if inject_prepare_failure:
            dist.barrier()
            if rank == 0:
                while outgoing.available_credits < slots_per_edge:
                    outgoing.poll()

                class InvalidCopyPayload:
                    shape = (2,)
                    dtype = torch.float32
                    device = torch.device("cpu")

                before_credits = outgoing.available_credits
                before_pending = outgoing.pending_work_count
                try:
                    outgoing.send(
                        PipelineTransportHeader(token_id=1, step_idx=0, cfg_branch=0),
                        InvalidCopyPayload(),
                    )
                except (TypeError, RuntimeError):
                    pass
                outgoing.poll()
                prepare_failure_preserved = (
                    outgoing.available_credits == before_credits and outgoing.pending_work_count == before_pending
                )
            dist.barrier()

        next_token_id = 0
        sent_token_ids: list[int] = []
        sent_sequences: list[int] = []
        sent_slot_ids: list[int] = []
        sent_clocks: list[int] = []
        completed: list[tuple[int, float]] = []
        received_slot_ids: list[int] = []
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
                        sent_header = outgoing.send(header, payload)
                        sent_token_ids.append(next_token_id)
                        sent_sequences.append(sent_header.send_sequence)
                        sent_slot_ids.append(sent_header.slot_id)
                        sent_clocks.append(clock)
                        next_token_id += 1
                    elif next_token_id >= slots_per_edge:
                        saw_credit_exhaustion = True
            elif rank != 1 or clock >= stage_one_stall_until:
                if incoming is not None and incoming.has_ready_message:
                    if rank == world_size - 1:
                        message = incoming.begin_compute()
                        received_slot_ids.append(message.header.slot_id)
                        if not message.header.flags & PipelineTransportHeader.SHUTDOWN_FLAG:
                            completed.append((message.header.token_id, float(message.payload[0])))
                        incoming.release_after_compute(message)
                    elif outgoing is not None and outgoing.can_send:
                        message = incoming.begin_compute()
                        received_slot_ids.append(message.header.slot_id)
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
        shutdown_started = False
        for _ in range((world_size + 2) * slots_per_edge + 4):
            if rank == 0:
                if not shutdown_started:
                    outgoing.begin_shutdown()
                    shutdown_started = True
                outgoing.progress_shutdown()
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
                if outgoing is not None:
                    if incoming.is_closed and not shutdown_started:
                        outgoing.begin_shutdown()
                        shutdown_started = True
                    if shutdown_started:
                        outgoing.progress_shutdown()
            dist.barrier()

        rebuild_succeeded = rebuild_after_close
        if rebuild_after_close:
            second_incoming = None
            second_credit_edge_groups = [
                dist.new_group([edge_rank, edge_rank + 1], backend="gloo") for edge_rank in range(world_size - 1)
            ]
            if rank > 0:
                second_incoming = PipelineP2PChannel(
                    source_rank=rank - 1,
                    destination_rank=rank,
                    tensor_shape=(2,),
                    tensor_dtype=torch.float32,
                    device="cpu",
                    slots_per_edge=1,
                    tag_base=(rank - 1) * 100,
                    group=edge_groups[rank - 1][0],
                    slot_groups=(edge_groups[rank - 1][0],),
                    credit_group=second_credit_edge_groups[rank - 1],
                )
            second_outgoing = None
            if rank < world_size - 1:
                second_outgoing = PipelineP2PChannel(
                    source_rank=rank,
                    destination_rank=rank + 1,
                    tensor_shape=(2,),
                    tensor_dtype=torch.float32,
                    device="cpu",
                    slots_per_edge=1,
                    tag_base=rank * 100,
                    group=edge_groups[rank][0],
                    slot_groups=(edge_groups[rank][0],),
                    credit_group=second_credit_edge_groups[rank],
                )

            second_message_sent = False
            second_message_received = False
            for _ in range(8):
                if rank == 0:
                    second_outgoing.poll()
                    if not second_message_sent and second_outgoing.can_send:
                        second_outgoing.send(
                            PipelineTransportHeader(token_id=99, step_idx=0, cfg_branch=0),
                            torch.full((2,), 99.0, dtype=torch.float32),
                        )
                        second_message_sent = True
                if rank == world_size - 1:
                    second_incoming.poll()
                    if not second_message_received and second_incoming.has_ready_message:
                        message = second_incoming.begin_compute()
                        rebuild_succeeded = message.header.token_id == 99
                        second_incoming.release_after_compute(message)
                        second_message_received = True
                dist.barrier()
            if rank == world_size - 1:
                rebuild_succeeded = rebuild_succeeded and second_message_received
            second_shutdown_started = False
            for _ in range((world_size + 2) * 2 + 4):
                if rank == 0:
                    if not second_shutdown_started:
                        second_outgoing.begin_shutdown()
                        second_shutdown_started = True
                    second_outgoing.progress_shutdown()
                if rank == world_size - 1:
                    second_incoming.poll()
                    while second_incoming.has_ready_message:
                        message = second_incoming.begin_compute()
                        second_incoming.release_after_compute(message)
                dist.barrier()
            if second_incoming is not None:
                second_incoming.wait_for_sends()
                rebuild_succeeded = rebuild_succeeded and second_incoming.pending_work_count == 0
            if second_outgoing is not None:
                second_outgoing.wait_for_sends()
                rebuild_succeeded = rebuild_succeeded and second_outgoing.pending_work_count == 0

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
                    "sent_sequences": sent_sequences,
                    "sent_slot_ids": sent_slot_ids,
                    "sent_clocks": sent_clocks,
                    "received_slot_ids": received_slot_ids,
                    "saw_credit_exhaustion": saw_credit_exhaustion,
                    "incoming_max_occupied": incoming.max_occupied if incoming is not None else 0,
                    "incoming_pending_work_count": incoming.pending_work_count if incoming is not None else 0,
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
                    "outgoing_pending_work_count": outgoing.pending_work_count if outgoing is not None else 0,
                    "rebuild_succeeded": rebuild_succeeded,
                    "prepare_failure_preserved": prepare_failure_preserved,
                    "deferred_release_event_queries": (release_events[0].query_count if release_events else 0),
                    "credit_return_event_queries": credit_return_event_queries,
                    "transport_buffer_inference_flags": (
                        _transport_buffer_inference_flags(incoming) + _transport_buffer_inference_flags(outgoing)
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
    rebuild_after_close: bool = False,
    inject_prepare_failure: bool = False,
    deferred_release_queries: int = 0,
    construct_inside_inference_mode: bool = False,
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
                rebuild_after_close,
                inject_prepare_failure,
                deferred_release_queries,
                construct_inside_inference_mode,
                result_queue,
            ),
            nprocs=world_size,
        )
        return {rank: result for rank, result in (result_queue.get() for _ in range(world_size))}
    finally:
        manager.shutdown()


class TestPipelineP2PChannel:
    def test_device_send_slot_retains_producer_until_downstream_credit(self) -> None:
        producer = torch.ones(2)
        slot = pipeline_runtime_module._P2PSendSlot(  # noqa: SLF001 - verify the channel's slot lifetime contract.
            0,
            torch.empty(PipelineTransportHeader.FIELD_COUNT, dtype=torch.int64),
            torch.empty(2),
        )

        slot.payload_owner = producer
        slot.begin_send(awaits_credit=True)
        slot.finish_send()

        assert slot.payload_owner is producer
        assert slot.awaits_credit is True

        slot.release_payload()
        assert slot.payload_owner is None
        assert slot.awaits_credit is False

    def test_header_rejects_values_that_do_not_fit_int64(self) -> None:
        with pytest.raises(ValueError, match="signed int64"):
            PipelineTransportHeader(token_id=2**63, step_idx=0, cfg_branch=0)

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
            rebuild_after_close=True,
            inject_prepare_failure=True,
        )

        source = results[0]
        destination = results[1]
        assert source["sent_token_ids"] == [0, 1, 2, 3, 4]
        assert source["sent_sequences"] == [0, 1, 2, 3, 4]
        assert source["sent_slot_ids"] == [0, 1, 0, 1, 0]
        assert destination["received_slot_ids"][:5] == [0, 1, 0, 1, 0]
        assert source["saw_credit_exhaustion"] is True
        assert source["sent_clocks"][slots_per_edge] >= 10
        assert destination["incoming_max_occupied"] == slots_per_edge
        assert [token_id for token_id, _ in destination["completed"]] == [0, 1, 2, 3, 4]
        assert source["outgoing_pending_work_count"] == 0
        assert destination["incoming_pending_work_count"] == 0
        assert source["rebuild_succeeded"] is True
        assert destination["rebuild_succeeded"] is True
        assert source["prepare_failure_preserved"] is True

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
            "COMPUTING": {"PENDING_RELEASE"},
            "PENDING_RELEASE": {"FREE"},
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
        assert all(results[rank]["incoming_pending_work_count"] == 0 for rank in range(1, 4))
        assert all(results[rank]["outgoing_pending_work_count"] == 0 for rank in range(3))

    def test_pp2_defers_credit_until_the_compute_event_completes(self) -> None:
        deferred_queries = 8
        results = _run_p2p_channel_lane(
            world_size=2,
            token_count=2,
            slots_per_edge=1,
            deferred_release_queries=deferred_queries,
        )

        source = results[0]
        destination = results[1]
        assert source["sent_token_ids"] == [0, 1]
        assert source["sent_clocks"][1] > source["sent_clocks"][0]
        assert destination["deferred_release_event_queries"] > deferred_queries
        assert destination["credit_return_event_queries"] == [deferred_queries + 1, 1]
        assert "PENDING_RELEASE" in destination["receive_histories"][0]
        assert destination["incoming_pending_work_count"] == 0
        assert source["outgoing_pending_work_count"] == 0

    def test_pp2_transport_buffers_are_mutable_after_inference_mode_construction(self) -> None:
        results = _run_p2p_channel_lane(
            world_size=2,
            token_count=2,
            slots_per_edge=1,
            construct_inside_inference_mode=True,
        )

        assert results[1]["completed"] == [(0, 0.0), (1, 1.0)]
        assert all(
            not is_inference
            for result in results.values()
            for is_inference in result["transport_buffer_inference_flags"]
        )
        assert results[1]["incoming_pending_work_count"] == 0
        assert results[0]["outgoing_pending_work_count"] == 0


class _TickPipeline:
    """Small stage-local pipeline used to exercise the production clock."""

    supports_step_execution = True
    supports_interleaved_pipeline_execution = True

    def __init__(self, stage: int, active_stages: list[tuple[int, int]]) -> None:
        self.stage = stage
        self.clock = 0
        self.active_stages = active_stages

    @staticmethod
    def build_microbatches(states):
        return [tuple(states)]

    @staticmethod
    def pipeline_transport_spec(states):
        rows = sum(int(state.latents.shape[0]) for state in states)
        width = int(states[0].latents.shape[1])
        return PipelineTensorSpec(
            intermediate_shape=(max(2, rows), width),
            intermediate_dtype=torch.float32,
            feedback_shape=(max(2, rows), width),
            feedback_dtype=torch.float32,
        )

    @staticmethod
    def pipeline_model_phase(states):
        del states
        return "main"

    def pipeline_forward_local_stage(self, input_batch, *, states, cfg_branch, intermediate_hidden_states):
        del states, cfg_branch
        self.active_stages.append((self.clock, self.stage))
        source = input_batch.latents if intermediate_hidden_states is None else intermediate_hidden_states
        return source + float(self.stage + 1)

    @staticmethod
    def pipeline_finish_microbatch(states, noise_pred, *, positive_noise_pred):
        if positive_noise_pred is not None:
            noise_pred = positive_noise_pred + noise_pred
        offset = 0
        outputs = []
        for state in states:
            rows = int(state.latents.shape[0])
            state.latents = noise_pred[offset : offset + rows].clone()
            state.step_index += 1
            outputs.append(state.latents)
            offset += rows
        return torch.cat(outputs, dim=0)


def _make_tick_state(request_id: str, value: float, *, cfg: bool, width: int = 1) -> StepRequestState:
    state = StepRequestState(request_id=request_id, sampling=SimpleNamespace(generator=None), prompt=None)
    state.latents = torch.full((1, width), value, dtype=torch.float32)
    state.timesteps = torch.tensor([1.0, 0.0], dtype=torch.float32)
    state.prompt_embeds = torch.zeros((1, 1, 1), dtype=torch.float32)
    state.negative_prompt_embeds = torch.zeros((1, 1, 1), dtype=torch.float32) if cfg else None
    state.do_true_cfg = cfg
    return state


def _run_pipeline_tick_runtime_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        state_cache = {
            "A": _make_tick_state("A", 0.0, cfg=True),
            "B": _make_tick_state("B", 10.0, cfg=True),
        }
        edge_pairs = [*zip(range(world_size), range(1, world_size)), (world_size - 1, 0)]
        edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        credit_edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        active_stages: list[tuple[int, int]] = []
        pipeline = _TickPipeline(rank, active_stages)
        runtime = PipelineTickRuntime(
            pipeline=pipeline,
            state_cache=state_cache,
            pp_ranks=tuple(range(world_size)),
            global_rank=rank,
            device="cpu",
            edge_groups={edge_pair: edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair},
            edge_credit_groups={
                edge_pair: credit_edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair
            },
            bootstrap_group=dist.group.WORLD,
        )

        pending_admissions = ["A"]
        completed: list[tuple[str, int, float]] = []
        for clock in range(48):
            if clock == 1:
                pending_admissions.append("B")
            if pending_admissions:
                runtime.admit([state_cache[request_id] for request_id in pending_admissions])
                pending_admissions = []

            pipeline.clock = clock
            local_completions = runtime.progress_one_clock()
            serialized = [
                (request_id, completion.step_idx)
                for completion in local_completions
                for request_id in completion.request_ids
            ]
            gathered: list[list[tuple[str, int]]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered, serialized)
            rank_zero_completions = gathered[0]
            if rank == 0:
                completed.extend(
                    (request_id, step_idx, float(state_cache[request_id].latents.item()))
                    for request_id, step_idx in rank_zero_completions
                )
            for request_id, step_idx in rank_zero_completions:
                if step_idx + 1 < state_cache[request_id].total_steps:
                    pending_admissions.append(request_id)
            dist.barrier()

            done = [len(completed) == 4 if rank == 0 else False]
            dist.broadcast_object_list(done, src=0)
            if done[0]:
                break
        dist.barrier()
        runtime.close()
        result_queue.put((rank, completed, active_stages))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pipeline_tick_abort_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    """Cancel after injection and prove the tombstone drains every PP edge."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        state_cache = {"A": _make_tick_state("A", 0.0, cfg=True)}
        edge_pairs = [*zip(range(world_size), range(1, world_size)), (world_size - 1, 0)]
        edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        credit_edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        active_stages: list[tuple[int, int]] = []
        pipeline = _TickPipeline(rank, active_stages)
        runtime = PipelineTickRuntime(
            pipeline=pipeline,
            state_cache=state_cache,
            pp_ranks=tuple(range(world_size)),
            global_rank=rank,
            device="cpu",
            edge_groups={edge_pair: edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair},
            edge_credit_groups={
                edge_pair: credit_edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair
            },
            bootstrap_group=dist.group.WORLD,
        )

        runtime.admit([state_cache["A"]])
        for clock in range(32):
            if clock == 1:
                runtime.cancel(("A",))
            pipeline.clock = clock
            assert runtime.progress_one_clock() == ()
            done = [not runtime.has_in_flight_work if rank == 0 else False]
            dist.broadcast_object_list(done, src=0)
            if done[0]:
                break
        else:
            raise RuntimeError("cancelled pipeline token did not drain")

        dist.barrier()
        runtime.close()
        result_queue.put((rank, active_stages, tuple(state_cache)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pipeline_tick_reconfigure_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    """Rebuild an idle fixed-shape lane before admitting a new tensor layout."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        state_cache = {
            "A": _make_tick_state("A", 0.0, cfg=False, width=1),
            "C": _make_tick_state("C", 10.0, cfg=False, width=2),
        }
        edge_pairs = [*zip(range(world_size), range(1, world_size)), (world_size - 1, 0)]
        edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        credit_edge_groups = {edge_pair: dist.new_group(list(edge_pair), backend="gloo") for edge_pair in edge_pairs}
        active_stages: list[tuple[int, int]] = []
        pipeline = _TickPipeline(rank, active_stages)
        runtime = PipelineTickRuntime(
            pipeline=pipeline,
            state_cache=state_cache,
            pp_ranks=tuple(range(world_size)),
            global_rank=rank,
            device="cpu",
            edge_groups={edge_pair: edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair},
            edge_credit_groups={
                edge_pair: credit_edge_groups[edge_pair] for edge_pair in edge_pairs if rank in edge_pair
            },
            bootstrap_group=dist.group.WORLD,
        )

        completed: list[str] = []
        for request_id in ("A", "C"):
            runtime.admit([state_cache[request_id]])
            for clock in range(16):
                pipeline.clock = clock
                completions = runtime.progress_one_clock()
                if rank == 0:
                    completed.extend(
                        completed_request_id
                        for completion in completions
                        for completed_request_id in completion.request_ids
                    )
                done = [request_id in completed if rank == 0 else False]
                dist.broadcast_object_list(done, src=0)
                dist.barrier()
                if done[0]:
                    break
            else:
                raise RuntimeError(f"{request_id} did not complete")
            assert not runtime.has_in_flight_work

        final_specs = tuple(runtime._spec_ids)  # noqa: SLF001 - verifies lane replacement.
        runtime.close()
        result_queue.put((rank, completed, final_specs, active_stages))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pp4_device_bootstrap_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    """Exercise the device bootstrap path with endpoint-only group maps."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    original_get_backend = dist.get_backend
    try:
        edge_pairs = pipeline_edge_pairs(tuple(range(world_size)))
        endpoint_slot_groups: dict[tuple[int, int], tuple[object, ...]] = {}
        for edge_pair in edge_pairs:
            edge_group = dist.new_group(list(edge_pair), backend="gloo")
            slot_groups = (edge_group, edge_group)
            if rank in edge_pair:
                endpoint_slot_groups[edge_pair] = slot_groups

        device_group_ids = {id(group) for groups in endpoint_slot_groups.values() for group in groups}

        def get_backend(group):
            if id(group) in device_group_ids:
                return "nccl"
            return original_get_backend(group)

        dist.get_backend = get_backend
        runtime = object.__new__(PipelineTickRuntime)
        runtime.edge_groups = {edge_pair: groups[0] for edge_pair, groups in endpoint_slot_groups.items()}
        runtime.edge_slot_groups = endpoint_slot_groups
        runtime.bootstrap_group = dist.group.WORLD
        runtime.pp_ranks = tuple(range(world_size))
        runtime.global_rank = rank
        runtime.slots_per_edge = 2
        runtime.device = "cpu"
        runtime._bootstrap_transport_edges(tag_base=40_000, channel_tag_span=16)  # noqa: SLF001
        result_queue.put((rank, True))
        dist.barrier()
    finally:
        dist.get_backend = original_get_backend
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_device_clock_error_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    """Fail one rank after a device send is prepared, then rebuild the lane."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    original_get_backend = dist.get_backend
    try:
        edge_pairs = pipeline_edge_pairs(tuple(range(world_size)))
        endpoint_groups: dict[tuple[int, int], object] = {}
        endpoint_slot_groups: dict[tuple[int, int], tuple[object, ...]] = {}
        endpoint_credit_groups: dict[tuple[int, int], object] = {}
        for edge_pair in edge_pairs:
            slot_groups = (dist.new_group(list(edge_pair), backend="gloo"),)
            credit_group = dist.new_group(list(edge_pair), backend="gloo")
            if rank in edge_pair:
                endpoint_groups[edge_pair] = slot_groups[0]
                endpoint_slot_groups[edge_pair] = slot_groups
                endpoint_credit_groups[edge_pair] = credit_group

        device_group_ids = {id(group) for group in endpoint_groups.values()}

        def get_backend(group):
            if id(group) in device_group_ids:
                return "nccl"
            return original_get_backend(group)

        dist.get_backend = get_backend

        def make_runtime(state_cache, active_stages):
            return PipelineTickRuntime(
                pipeline=_TickPipeline(rank, active_stages),
                state_cache=state_cache,
                pp_ranks=tuple(range(world_size)),
                global_rank=rank,
                device="cpu",
                edge_groups=endpoint_groups,
                edge_slot_groups=endpoint_slot_groups,
                edge_credit_groups=endpoint_credit_groups,
                bootstrap_group=dist.group.WORLD,
                slots_per_edge=1,
            )

        state_cache = {"A": _make_tick_state("A", 0.0, cfg=False)}
        runtime = make_runtime(state_cache, [])
        runtime.admit([state_cache["A"]])
        for channel in runtime._all_channels():  # noqa: SLF001 - device control on top of Gloo transport.
            channel._slot_requires_wait_thread = (True,)  # noqa: SLF001
        for _ in range(16):
            runtime._poll_channels()  # noqa: SLF001 - await initial source credit.
            dist.barrier()
        if rank == 0:
            spec = next(iter(runtime._spec_ids))  # noqa: SLF001 - inspect the admitted edge.
            assert runtime._forward_channel(0, spec).can_send  # noqa: SLF001

        original_poll_channels = runtime._poll_channels  # noqa: SLF001

        def poll_channels() -> None:
            original_poll_channels()
            if rank == 1:
                raise ValueError("injected stage-local failure")

        runtime._poll_channels = poll_channels  # noqa: SLF001

        try:
            runtime.progress_one_clock()
        except RuntimeError as exc:
            error_message = str(exc)
        else:
            error_message = ""
        assert not runtime.has_in_flight_work
        assert not state_cache
        runtime.close()
        failed_runtime_pending_work = [channel.pending_work_count for channel in runtime._all_channels()]  # noqa: SLF001

        recovered_state_cache = {"B": _make_tick_state("B", 10.0, cfg=False)}
        recovered_active_stages: list[tuple[int, int]] = []
        recovered = make_runtime(recovered_state_cache, recovered_active_stages)
        recovered.admit([recovered_state_cache["B"]])
        for channel in recovered._all_channels():  # noqa: SLF001 - device control on top of Gloo transport.
            channel._slot_requires_wait_thread = (True,)  # noqa: SLF001
        completions: list[str] = []
        for _ in range(8):
            for completion in recovered.progress_one_clock():
                completions.extend(completion.request_ids)
            dist.barrier()
        assert not recovered.has_in_flight_work
        recovered.close()
        recovered_pending_work = [channel.pending_work_count for channel in recovered._all_channels()]  # noqa: SLF001
        result_queue.put(
            (
                rank,
                error_message,
                runtime.clock,
                failed_runtime_pending_work,
                recovered_pending_work,
                completions,
            )
        )
        dist.barrier()
    finally:
        dist.get_backend = original_get_backend
        if dist.is_initialized():
            dist.destroy_process_group()


class TestPipelineTickRuntime:
    def test_pp4_overlaps_dynamic_admission_and_returns_feedback(self) -> None:
        world_size = 4
        mp_context = torch.multiprocessing.get_context("spawn")
        manager = mp_context.Manager()
        result_queue = manager.Queue()
        try:
            torch.multiprocessing.spawn(
                _run_pipeline_tick_runtime_worker,
                args=(world_size, get_distributed_init_method(), result_queue),
                nprocs=world_size,
            )
            results = {
                rank: (completed, active_stages)
                for rank, completed, active_stages in (result_queue.get() for _ in range(world_size))
            }
        finally:
            manager.shutdown()

        assert results[0][0] == [
            ("A", 0, 20.0),
            ("B", 0, 40.0),
            ("A", 1, 60.0),
            ("B", 1, 100.0),
        ]
        assert all(results[rank][0] == [] for rank in range(1, world_size))
        stages_by_clock: dict[int, set[int]] = {}
        for _, active_stages in results.values():
            for clock, stage in active_stages:
                stages_by_clock.setdefault(clock, set()).add(stage)
        assert any(len(stages) >= 2 for stages in stages_by_clock.values())

    def test_pp4_abort_drains_tombstones_without_more_forwards(self) -> None:
        world_size = 4
        mp_context = torch.multiprocessing.get_context("spawn")
        manager = mp_context.Manager()
        result_queue = manager.Queue()
        try:
            torch.multiprocessing.spawn(
                _run_pipeline_tick_abort_worker,
                args=(world_size, get_distributed_init_method(), result_queue),
                nprocs=world_size,
            )
            results = {
                rank: (active_stages, state_cache_keys)
                for rank, active_stages, state_cache_keys in (result_queue.get() for _ in range(world_size))
            }
        finally:
            manager.shutdown()

        assert results[0][0] == [(0, 0)]
        assert all(results[rank][0] == [] for rank in range(1, world_size))
        assert all(results[rank][1] == () for rank in range(world_size)), results

    def test_pp2_rebuilds_idle_lane_before_new_tensor_layout(self) -> None:
        world_size = 2
        mp_context = torch.multiprocessing.get_context("spawn")
        manager = mp_context.Manager()
        result_queue = manager.Queue()
        try:
            torch.multiprocessing.spawn(
                _run_pipeline_tick_reconfigure_worker,
                args=(world_size, get_distributed_init_method(), result_queue),
                nprocs=world_size,
            )
            results = {
                rank: (completed, specs, active_stages)
                for rank, completed, specs, active_stages in (result_queue.get() for _ in range(world_size))
            }
        finally:
            manager.shutdown()

        assert results[0][0] == ["A", "C"]
        assert all(results[rank][0] == [] for rank in range(1, world_size))
        assert all(specs[0].intermediate_shape == (2, 2) for _, specs, _ in results.values())
        assert all(active_stages for _, _, active_stages in results.values())

    def test_pp4_device_bootstrap_only_resolves_endpoint_groups(self) -> None:
        world_size = 4
        mp_context = torch.multiprocessing.get_context("spawn")
        manager = mp_context.Manager()
        result_queue = manager.Queue()
        try:
            torch.multiprocessing.spawn(
                _run_pp4_device_bootstrap_worker,
                args=(world_size, get_distributed_init_method(), result_queue),
                nprocs=world_size,
            )
            results = dict(result_queue.get() for _ in range(world_size))
        finally:
            manager.shutdown()

        assert results == {rank: True for rank in range(world_size)}

    def test_device_clock_error_converges_before_p2p_submission(self) -> None:
        world_size = 2
        mp_context = torch.multiprocessing.get_context("spawn")
        manager = mp_context.Manager()
        result_queue = manager.Queue()
        try:
            torch.multiprocessing.spawn(
                _run_device_clock_error_worker,
                args=(world_size, get_distributed_init_method(), result_queue),
                nprocs=world_size,
            )
            results = {
                rank: (message, clock, failed_pending_work, recovered_pending_work, completions)
                for rank, message, clock, failed_pending_work, recovered_pending_work, completions in (
                    result_queue.get() for _ in range(world_size)
                )
            }
        finally:
            manager.shutdown()

        assert all(message == "interleaved PP clock failed on pipeline rank(s) 1" for message, *_ in results.values())
        assert all(clock == 0 for _, clock, *_ in results.values())
        assert all(not any(failed_pending_work) for _, _, failed_pending_work, _, _ in results.values())
        assert all(not any(recovered_pending_work) for _, _, _, recovered_pending_work, _ in results.values())
        assert results[0][-1] == ["B"]
        assert results[1][-1] == []
