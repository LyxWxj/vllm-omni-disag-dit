# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion import io_support
from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import (
    DiffusionEngine,
    DiffusionExecutionMode,
    _RankPipelineResult,
    _RankPipelineTask,
)
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_startup_rank_result_does_not_block_result_pump(mocker) -> None:
    engine = object.__new__(DiffusionEngine)
    engine._loop_started = False
    engine._cv = threading.Condition(threading.RLock())
    engine._pipeline_step_results = queue.Queue()
    task = _RankPipelineTask("step-0", ("startup",), mocker.Mock())
    future = Future()
    output = mocker.sentinel.output
    future.set_result(output)
    callback_finished = threading.Event()

    def notify_result():
        engine._notify_rank_pipeline_result(task, 0, future)
        callback_finished.set()

    thread = threading.Thread(target=notify_result, daemon=True)
    try:
        with engine._cv:
            thread.start()
            assert callback_finished.wait(timeout=1), "Startup lock blocked the result pump callback"
            assert engine._pipeline_step_results.get_nowait().output is output
    finally:
        thread.join(timeout=2)


def test_rank_pipeline_completion_keeps_task_on_its_own_rank_queue() -> None:
    engine = object.__new__(DiffusionEngine)
    task = _RankPipelineTask("step-0", ("request-a", "request-b"), Mock())
    next_rank_task = _RankPipelineTask(task.task_id, task.request_ids, task.scheduler_output)
    engine._rank_pipeline_active = [task, None]
    engine._rank_pipeline_queues = [queue.Queue(), queue.Queue()]
    engine._rank_pipeline_queues[1].put_nowait(next_rank_task)
    engine._dispatch_pipeline_rank_tasks = Mock()

    engine._handle_rank_pipeline_result(_RankPipelineResult(task.task_id, rank=0, output=Mock()))

    assert engine._rank_pipeline_active == [None, None]
    assert engine._rank_pipeline_queues[1].get_nowait() is next_rank_task
    engine._dispatch_pipeline_rank_tasks.assert_called_once_with()


def test_rank_pipeline_decode_releases_other_ranks_without_waiting() -> None:
    engine = object.__new__(DiffusionEngine)
    task = _RankPipelineTask("decode:step-0:request-a", ("request-a",), None)
    engine._rank_pipeline_active = [None, task]
    engine._rank_pipeline_queues = [queue.Queue(), queue.Queue()]
    engine.executor = Mock()
    release_future = Future()
    release_future.set_result(AsyncDiffusionOutput(kind=AsyncOutputKind.RPC_RESULT))
    engine.executor.release_rank_step_requests.return_value = release_future
    engine.scheduler = StepScheduler()
    engine.scheduler.finish_pipeline_decode = Mock(return_value={"request-a"})
    engine._dispatch_pipeline_rank_tasks = Mock()
    output = Mock()

    completion = engine._handle_rank_pipeline_result(_RankPipelineResult(task.task_id, rank=1, output=output))

    engine.executor.release_rank_step_requests.assert_called_once_with(["request-a"], 0)
    engine.executor.collective_rpc.assert_not_called()
    engine.scheduler.finish_pipeline_decode.assert_called_once_with(output, ["request-a"])
    assert completion == (["request-a"], output, {"request-a"}, True)


def test_enqueue_pipeline_step_keeps_scheduler_batch_intact() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = StepScheduler()
    engine._prepare_kv_for_forward = Mock()
    engine._pipeline_scheduler_outputs = {}
    engine._rank_pipeline_queues = [queue.Queue(), queue.Queue()]
    scheduler_output = SimpleNamespace(step_id=4, scheduled_request_ids=["request-a", "request-b"])

    engine._enqueue_pipeline_step(scheduler_output)

    tasks = [rank_queue.get_nowait() for rank_queue in engine._rank_pipeline_queues]
    assert tasks[0].task_id == tasks[1].task_id == "4"
    assert tasks[0].request_ids == tasks[1].request_ids == ("request-a", "request-b")
    assert tasks[0].scheduler_output is scheduler_output
    assert tasks[1].scheduler_output is scheduler_output
    assert engine.scheduler._pipeline_inflight == {"request-a", "request-b"}


def test_rank_pipeline_queues_decode_for_each_completed_batch_request() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = StepScheduler()
    engine.scheduler.pipeline_decode_pending = Mock(return_value=["request-a", "request-b"])
    engine._rank_pipeline_active = [None, _RankPipelineTask("step-4", ("request-a", "request-b"), Mock())]
    engine._rank_pipeline_queues = [queue.Queue(), queue.Queue()]
    engine._commit_pipeline_step_result = Mock(
        return_value=(
            SimpleNamespace(scheduled_request_ids=["request-a", "request-b"]),
            Mock(),
            set(),
        )
    )
    engine._dispatch_pipeline_rank_tasks = Mock()

    completion = engine._handle_rank_pipeline_result(_RankPipelineResult("step-4", rank=1, output=Mock()))

    assert [engine._rank_pipeline_queues[1].get_nowait().request_ids for _ in range(2)] == [
        ("request-a",),
        ("request-b",),
    ]
    assert completion[0] == ["request-a", "request-b"]
    engine._dispatch_pipeline_rank_tasks.assert_called_once_with()


@pytest.mark.parametrize("pp_size", [2, 3])
@pytest.mark.parametrize("last_rank_first", [False, True])
def test_pipeline_startup_uses_rank_queues_and_decodes_final_output(pp_size, last_rank_first) -> None:
    scheduler = StepScheduler()
    scheduler.initialize(
        SimpleNamespace(
            max_num_seqs=1,
            step_execution=True,
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        )
    )
    request = OmniDiffusionRequest(
        prompt="warmup",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id="warmup",
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    def completed_future(value):
        future = Future()
        future.set_result(value)
        return future

    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine._pipeline_scheduler_outputs = {}
    engine._pipeline_step_results = queue.Queue()
    engine._rank_pipeline_queues = [queue.Queue() for _ in range(pp_size)]
    engine._rank_pipeline_active = [None] * pp_size
    engine._loop_started = False
    engine._cv = threading.Condition(threading.RLock())
    engine._closed = False
    engine._prepare_kv_for_forward = Mock()
    engine._process_aborts_queue = Mock()
    engine._release_pipeline_requests = Mock()
    engine._emit_outputs = Mock()
    engine._fail_engine = Mock()

    step_output = BatchRunnerOutput.from_list([RunnerOutput(request_id="warmup", step_index=1, finished=False)])
    stage0_future = Future() if last_rank_first else completed_future(step_output)

    def execute_rank_step(_scheduler_output, rank):
        return stage0_future if rank == 0 else completed_future(step_output)

    decode_output = BatchRunnerOutput.from_list(
        [RunnerOutput(request_id="warmup", step_index=1, finished=True, result=DiffusionOutput(output="video"))]
    )

    def execute_rank_decode(request_id):
        if last_rank_first:
            stage0_future.set_result(step_output)
        return completed_future(decode_output)

    engine.executor = Mock(
        execute_rank_step=Mock(side_effect=execute_rank_step),
        execute_rank_decode=Mock(side_effect=execute_rank_decode),
        release_rank_step_requests=Mock(return_value=completed_future(None)),
    )

    output, finished_ids = engine._execute_pipeline_step_synchronously(scheduler_output)

    assert output.get_request_output("warmup").result.output == "video"
    assert finished_ids == {"warmup"}
    assert scheduler.get_request_state("warmup").is_finished()
    assert engine.executor.execute_rank_step.call_count == pp_size
    engine.executor.execute_rank_decode.assert_called_once_with("warmup")
    engine._emit_outputs.assert_not_called()
    engine._fail_engine.assert_not_called()
    assert engine._rank_pipeline_active == [None] * pp_size
    assert all(tasks.empty() for tasks in engine._rank_pipeline_queues)


@pytest.mark.parametrize("model_class_name", ["DreamZeroPipeline", "LingBotWorldCausalDMDPipeline"])
def test_observation_conditioned_startup_skips_generic_warmup(model_class_name: str) -> None:
    """Observation-conditioned pipelines must not receive generic text requests."""
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    engine.od_config.model_class_name = model_class_name
    engine.od_config.diffusion_load_format = "default"
    engine.add_req_and_wait_for_response = Mock(side_effect=AssertionError("generic text warmup submitted"))
    engine.close = Mock()

    engine.run_startup_warmup()

    engine.add_req_and_wait_for_response.assert_not_called()
    engine.close.assert_not_called()


@pytest.mark.parametrize("step_execution", [False, True])
def test_dummy_run_uses_enough_steps_for_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
    step_execution: bool,
) -> None:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = type(
        "Config",
        (),
        {"model_class_name": "mock_model", "diffusion_load_format": "default"},
    )()
    engine.step_execution = step_execution
    engine.pre_process_func = None
    captured_requests = []

    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.supports_multimodal_input",
        lambda od_config: (False, False),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.get_dummy_run_num_frames",
        lambda model_class_name, supports_audio_input: 1,
    )

    def run_request(request):
        captured_requests.append(request)
        return type("Output", (), {"error": None})()

    engine.add_req_and_wait_for_response = run_request

    engine._dummy_run()

    assert len(captured_requests) == 1
    assert captured_requests[0].sampling_params.num_inference_steps == 2


def test_allgather_startup_runs_broadcast_dummy_request() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        },
        parallel_config=SimpleNamespace(data_parallel_size=2, sequence_parallel_size=1),
    )
    engine._dummy_run = Mock()

    engine.run_startup_warmup()

    engine._dummy_run.assert_called_once_with()


def test_dummy_run_num_frames_uses_explicit_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    class JointAudioVideoModel:
        dummy_run_num_frames = 2

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: JointAudioVideoModel,
    )

    assert io_support.get_dummy_run_num_frames("joint_audio_video", supports_audio_input=False) == 2


def test_dummy_run_num_frames_keeps_audio_output_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioOutputModel:
        support_audio_output = True

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: AudioOutputModel,
    )

    assert io_support.get_dummy_run_num_frames("audio_output", supports_audio_input=False) == 2


def test_dummy_run_num_frames_defaults_to_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoOnlyModel:
        pass

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: VideoOnlyModel,
    )

    assert io_support.get_dummy_run_num_frames("video_only", supports_audio_input=False) == 1


def test_dummy_run_num_frames_uses_audio_input_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: None,
    )

    assert io_support.get_dummy_run_num_frames("unknown", supports_audio_input=True) == 2


def test_dummy_run_image_count_resolves_hunyuan_architecture_alias() -> None:
    assert io_support.get_dummy_run_num_image_inputs("HunyuanImage3ForCausalMM") == 3
    assert io_support.get_dummy_run_num_image_inputs("unknown") == 1


def test_dense_mode_does_not_build_kv_profile_request() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    engine._make_dummy_request = Mock(side_effect=AssertionError("dense mode must not profile"))

    assert engine._prepare_diffusion_kv_profile_requests() is None
    engine._make_dummy_request.assert_not_called()


def test_explicit_profile_frames_preserve_warmup_skip(monkeypatch):
    engine = object.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    engine.od_config.model_class_name = "HunyuanImage3ForCausalMM"
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.supports_multimodal_input", lambda _: (True, False))
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.image_color_format", lambda _: "RGB")
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.get_dummy_run_num_frames", lambda *_: 0)
    kwargs = dict(height=1024, width=1024, guidance_scale=5.0, num_image_inputs=3)

    assert engine._make_dummy_request(**kwargs) is None
    request = engine._make_dummy_request(**kwargs, num_frames=1)

    assert request is not None
    assert request.sampling_params.num_frames == 1
    assert request.sampling_params.num_inference_steps == 1
    assert len(request.prompt["multi_modal_data"]["image"]) == 3


@pytest.mark.parametrize(
    ("execution_mode", "uses_dlo_dp", "max_num_seqs", "expected_profile_requests"),
    [
        (DiffusionExecutionMode.STEP_BATCH, False, 2, 2),
        (DiffusionExecutionMode.REQUEST_BATCH, True, 4, 1),
        (DiffusionExecutionMode.STEP_BATCH, True, 3, 3),
    ],
)
def test_paged_kv_profile_requests_match_per_rank_batch(
    execution_mode: DiffusionExecutionMode,
    uses_dlo_dp: bool,
    max_num_seqs: int,
    expected_profile_requests: int,
) -> None:
    engine = object.__new__(DiffusionEngine)
    engine.execution_mode = execution_mode
    engine.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        model_class_name="HunyuanImage3ForCausalMM",
        max_num_seqs=max_num_seqs,
        parallel_config=SimpleNamespace(data_parallel_size=max_num_seqs if uses_dlo_dp else 1),
        enable_distributed_layerwise_offload=uses_dlo_dp,
        dlo_use_allgather=True,
    )
    request = OmniDiffusionRequest(
        prompt="profile",
        request_id="profile-request",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    engine._make_dummy_request = Mock(return_value=request)
    prepared_layout = object()
    scheduler_kv_state = (
        DiffusionKVRequest(
            "profile-request/diffusion-kv/0",
            sequence_id=0,
            prefix_len=3,
            target_len=8,
            seq_len=12,
        ),
        DiffusionKVRequest(
            "profile-request/diffusion-kv/1",
            sequence_id=1,
            prefix_len=4,
            target_len=8,
            seq_len=13,
        ),
    )

    def preprocess(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        req.prepared_layout = prepared_layout
        req.diffusion_kv_requests = scheduler_kv_state
        return req

    engine._prepare_request_for_admission = Mock(side_effect=preprocess)

    result = engine._prepare_diffusion_kv_profile_requests()

    assert result is not None
    assert len(result) == expected_profile_requests
    assert len({profile_request.request_id for profile_request in result}) == expected_profile_requests
    assert all(profile_request.is_dummy_run() for profile_request in result)
    assert all(profile_request.prepared_layout is prepared_layout for profile_request in result)
    assert all(profile_request.diffusion_kv_requests is None for profile_request in result)
    assert len({id(profile_request.sampling_params) for profile_request in result}) == expected_profile_requests
    assert engine._diffusion_kv_profile_limits == (2, 13, 8)
    assert len(result) * engine._diffusion_kv_profile_limits[0] == expected_profile_requests * 2
    engine._make_dummy_request.assert_called_once_with(
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_image_inputs=3,
        num_frames=1,
    )
    engine._prepare_request_for_admission.assert_called_once_with(request)


@pytest.mark.parametrize(
    ("num_sequences", "seq_len", "target_len"),
    [
        (3, 16, 8),
        (1, 17, 8),
        (1, 16, 9),
    ],
)
def test_paged_kv_admission_rejects_shape_beyond_profile_envelope(
    num_sequences: int,
    seq_len: int,
    target_len: int,
) -> None:
    engine = object.__new__(DiffusionEngine)
    engine._diffusion_kv_profile_limits = (2, 16, 8)
    engine.pre_process_func = lambda request: request
    request = OmniDiffusionRequest(
        prompt="too large",
        request_id="too-large",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=tuple(
            DiffusionKVRequest(
                f"too-large/diffusion-kv/{sequence_id}",
                sequence_id=sequence_id,
                prefix_len=0,
                target_len=target_len,
                seq_len=seq_len,
            )
            for sequence_id in range(num_sequences)
        ),
    )

    with pytest.raises(ValueError, match="exceeds the startup memory-profile envelope"):
        engine._prepare_request_for_admission(request)


def test_paged_kv_admission_accepts_shape_at_profile_envelope() -> None:
    engine = object.__new__(DiffusionEngine)
    engine._diffusion_kv_profile_limits = (2, 16, 8)
    engine.pre_process_func = lambda request: request
    request = OmniDiffusionRequest(
        prompt="fits",
        request_id="fits",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=tuple(
            DiffusionKVRequest(
                f"fits/diffusion-kv/{sequence_id}",
                sequence_id=sequence_id,
                prefix_len=8,
                target_len=8,
                seq_len=16,
            )
            for sequence_id in range(2)
        ),
    )

    assert engine._prepare_request_for_admission(request) is request
