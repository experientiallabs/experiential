"""Tests for the real-tau2 eval harness: provenance resolution + results flattening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tau_real_eval import collect_records, resolve_tasks


def _span(trace_id: str, domain: str, task_id: str) -> dict:
    return {
        "traceId": trace_id,
        "spanId": f"{trace_id[:8]}a",
        "name": "chat tau2",
        "attributes": [
            {
                "key": "wmh.trace.metadata",
                "value": {
                    "stringValue": json.dumps({"domain": domain, "task_id": task_id, "reward": 1.0})
                },
            }
        ],
    }


def _write(tmp_path: Path, name: str, lines: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_resolve_tasks_maps_provenance_to_real_task_ids(tmp_path: Path) -> None:
    scenarios = _write(
        tmp_path,
        "scenarios.jsonl",
        [
            {"domain": "airline", "task": "t", "provenance": ["aaa"]},
            {"domain": "retail", "task": "t", "provenance": ["bbb"]},
        ],
    )
    corpus = _write(
        tmp_path,
        "corpus.jsonl",
        [
            _span("aaa", "airline", "12"),
            _span("aaa", "airline", "12"),  # later spans of the same trace are ignored
            _span("bbb", "retail", "7"),
            _span("ccc", "telecom", "1"),  # not pinned -> not returned
        ],
    )
    tasks = resolve_tasks(scenarios, corpus)
    assert [(t["domain"], t["task_id"], t["trace_id"]) for t in tasks] == [
        ("airline", "12", "aaa"),
        ("retail", "7", "bbb"),
    ]


def test_resolve_tasks_fails_loudly_on_missing_provenance(tmp_path: Path) -> None:
    scenarios = _write(
        tmp_path, "scenarios.jsonl", [{"domain": "airline", "task": "t", "provenance": ["zzz"]}]
    )
    corpus = _write(tmp_path, "corpus.jsonl", [_span("aaa", "airline", "12")])
    with pytest.raises(SystemExit, match="zzz"):
        resolve_tasks(scenarios, corpus)


def test_collect_records_pairs_by_trace_id_and_counts_trials(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "tasks": [{"id": "12"}],
                "simulations": [
                    {"id": "s1", "task_id": "12", "reward_info": {"reward": 1.0}},
                    {"id": "s2", "task_id": "12", "reward_info": {"reward": 0.0}},
                    {"id": "s3", "task_id": "99", "reward_info": {"reward": 1.0}},  # not ours
                    {"id": "s4", "task_id": "12", "reward_info": None},  # graderless
                ],
            }
        )
    )
    records = collect_records(results, {"12": "aaa"})
    assert [(r["scenario_id"], r["rollout_index"], r["reward"], r["success"]) for r in records] == [
        ("aaa", 0, 1.0, True),
        ("aaa", 1, 0.0, False),
        ("aaa", 2, 0.0, False),
    ]
    assert records[2]["errors"]  # missing reward_info is an error record, not a silent 0
