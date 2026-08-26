from __future__ import annotations

import json
from pathlib import Path

from tools.benchmark.analyze_diffusion_pp_trace import summarize
from vllm_omni.diffusion.distributed import pp_trace


def test_trace_writes_request_and_clock_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pp_trace, "_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(pp_trace, "_TRACE_SYNC", False)
    monkeypatch.setattr(pp_trace, "_HANDLE", None)
    monkeypatch.setattr(pp_trace, "_EVENT_COUNTER", 0)

    with pp_trace.span(
        "stage_forward",
        pp_rank=1,
        pp_size=4,
        clock=3,
        token_id=7,
        request_ids=("request-a", "request-b"),
        microbatch_id=2,
        step_idx=1,
        cfg_branch="positive",
        model_phase="high_noise",
        slot_id=0,
    ):
        pass
    pp_trace.event("feedback_ready", pp_rank=1, pp_size=4, clock=3, token_id=7)
    pp_trace.close()

    records = [json.loads(line) for line in (tmp_path / "pp_rank_1.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["begin", "end", "instant"]
    assert records[0]["request_ids"] == ["request-a", "request-b"]
    assert records[0]["clock"] == 3
    assert records[0]["pp_rank"] == 1


def test_trace_summary_reports_multi_stage_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pp_trace, "_TRACE_DIR", None)
    records = {
        "pp_rank_0.jsonl": [
            {
                "event": "begin",
                "name": "stage_forward",
                "pp_rank": 0,
                "pp_size": 2,
                "ts_ns": 0,
                "token_id": 1,
                "clock": 0,
            },
            {
                "event": "end",
                "name": "stage_forward",
                "pp_rank": 0,
                "pp_size": 2,
                "ts_ns": 10,
                "token_id": 1,
                "clock": 0,
            },
        ],
        "pp_rank_1.jsonl": [
            {
                "event": "begin",
                "name": "stage_forward",
                "pp_rank": 1,
                "pp_size": 2,
                "ts_ns": 5,
                "token_id": 1,
                "clock": 1,
            },
            {
                "event": "end",
                "name": "stage_forward",
                "pp_rank": 1,
                "pp_size": 2,
                "ts_ns": 15,
                "token_id": 1,
                "clock": 1,
            },
        ],
    }
    for filename, events in records.items():
        (tmp_path / filename).write_text("\n".join(json.dumps(event) for event in events) + "\n")

    summary = summarize(tmp_path)
    assert summary["stage_forward_intervals"] == 2
    assert summary["multi_stage_overlap_ms"] == 0.000005
    assert summary["overlap_ratio"] == 1 / 3
