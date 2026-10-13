import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _trace_module():
    path = Path(__file__).parents[2] / "tools" / "benchmark" / "queued_pp_trace.py"
    spec = importlib.util.spec_from_file_location("queued_pp_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_queued_pp_trace_renderer_includes_control_and_rank_lanes(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    events_dir = trace_dir / "worker_events"
    events_dir.mkdir(parents=True)
    manifest = {
        "git_revision": "test-revision",
        "elapsed_s": 0.4,
        "throughput_requests_per_s": 5.0,
        "config": {
            "pipeline_parallel_size": 2,
            "request_count": 2,
            "max_num_seqs": 2,
            "max_inflight_batches": 2,
            "dimensions": [8, 64, 64],
        },
        "requests": [
            {
                "request_id": "req-0",
                "seed": 42,
                "started_ns": 1_000_000_000,
                "latency_ms": 300.0,
                "decoded_sha256": "abc",
                "height": 64,
                "width": 64,
                "num_frames": 8,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
            },
            {
                "request_id": "req-1",
                "seed": 43,
                "started_ns": 1_000_000_000,
                "latency_ms": 400.0,
                "decoded_sha256": "def",
                "height": 64,
                "width": 64,
                "num_frames": 8,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
            },
        ],
    }
    (trace_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (events_dir / "worker_rank0.jsonl").write_text(
        json.dumps(
            {
                "kind": "post_decode",
                "rank": 0,
                "request_id": "req-0",
                "start_ns": 1_100_000_000,
                "end_ns": 1_200_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (events_dir / "worker_rank1.jsonl").write_text(
        json.dumps(
            {
                "kind": "decode_ready",
                "rank": 1,
                "request_ids": ["req-0", "req-1"],
                "start_ns": 1_100_000_000,
                "end_ns": 1_150_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (events_dir / "worker_rank0.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "decode_rpc",
                    "rank": 0,
                    "request_id": "req-0",
                    "start_ns": 1_200_000_000,
                    "end_ns": 1_300_000_000,
                }
            )
            + "\n"
        )
        stream.write(
            json.dumps(
                {
                    "kind": "receive_scheduler_feedback",
                    "rank": 0,
                    "request_id": "req-0",
                    "step_index": 1,
                    "start_ns": 1_300_000_000,
                    "end_ns": 1_310_000_000,
                }
            )
            + "\n"
        )
    (events_dir / "engine_control.jsonl").write_text(
        json.dumps(
            {
                "kind": "decode_round",
                "phase": "sealed",
                "request_ids": ["req-0", "req-1"],
                "start_ns": 1_050_000_000,
                "end_ns": 1_060_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = trace_dir / "queued_pp_trace.html"
    _trace_module()._render(trace_dir, output)

    html = output.read_text(encoding="utf-8")
    assert "rank0.post_decode" in html
    assert "rank0.receive_scheduler_feedback" in html
    assert "rank1.decode_ready" in html
    assert "engine.decode_round" in html
    assert "engine.decode_dispatch" in html
    assert "req-0,req-1" in html
    assert "workload=heterogeneous" in html
    assert "64×64" in html
    assert "steps=2, 4" in html
    assert "abc" in html and "def" in html


def test_trace_installer_wraps_private_engine_queue_runtime() -> None:
    from vllm_omni.diffusion.queued_pp.engine_queue import QueuedPipelineEngineQueue
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    trace = _trace_module()
    engine_methods = (
        "_maybe_start_queued_decode_round",
        "_advance_queued_decode_round",
        "_authorize_waiting_queued_batches",
        "_commit_queued_pipeline_step",
    )
    original_engine_methods = {name: getattr(QueuedPipelineEngineQueue, name) for name in engine_methods}
    worker_methods = ("execute_prepared_stepwise", "prepare_queued_decode", "execute_prepared_decode")
    original_worker_methods = {name: getattr(DiffusionWorker, name) for name in worker_methods}
    try:
        trace.install()
        assert getattr(DiffusionWorker.execute_prepared_stepwise, "_queued_pp_trace_wrapper", False)
        assert all(
            getattr(getattr(QueuedPipelineEngineQueue, name), "_queued_pp_trace_wrapper", False)
            for name in engine_methods
        )
    finally:
        for name, method in original_worker_methods.items():
            setattr(DiffusionWorker, name, method)
        for name, method in original_engine_methods.items():
            setattr(QueuedPipelineEngineQueue, name, method)


def test_pipeline_trace_records_feedback_receive_as_its_own_span(tmp_path: Path, monkeypatch) -> None:
    trace_dir = tmp_path / "worker_events"
    monkeypatch.setenv("VLLM_OMNI_PP_TRACE_DIR", str(trace_dir))
    pipeline = SimpleNamespace(receive_scheduler_feedback=lambda _state: None)
    worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=pipeline), rank=0, device="cpu")
    state = SimpleNamespace(request_id="req-a", step_index=3)

    _trace_module()._install_pipeline_hooks(worker)
    pipeline.receive_scheduler_feedback(state)

    records = [json.loads(line) for line in (trace_dir / "worker_rank0.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["kind"] == "receive_scheduler_feedback"
    assert records[0]["request_id"] == "req-a"
    assert records[0]["step_index"] == 3
    assert records[0]["error"] is None
