"""The runner on a synthetic matrix: offline, deterministic, and honest about divergence."""

import json
from pathlib import Path

import numpy as np
import pytest

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.reproduce.embedding import CachedTaskEmbedder
from wmo.reproduce.manifest import Manifest
from wmo.reproduce.runner import run_reproduction

_MODELS = [
    {
        "name": "cheap",
        "kind": "anthropic",
        "model": "m-cheap",
        "input_per_mtok": 1.0,
        "output_per_mtok": 2.0,
    },
    {
        "name": "pricey",
        "kind": "anthropic",
        "model": "m-pricey",
        "input_per_mtok": 10.0,
        "output_per_mtok": 20.0,
    },
]


def _matrix_dict(n_scenarios: int = 8) -> dict:
    outcomes = []
    for index in range(n_scenarios):
        for model, reward, cost in (("cheap", 0.5, 0.001), ("pricey", 1.0, 0.01)):
            outcomes.append(
                {
                    "scenario_id": f"s{index}",
                    "task": f"task text {index}",
                    "model": model,
                    "episode": 0,
                    "reward": reward,
                    "success": reward > 0.6,
                    "critique": "",
                    "steps": 1,
                    "stop_reason": "done",
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                    "cost_usd": cost,
                    "call_seconds": [0.5],
                    "replies": [],
                    "error": None,
                }
            )
    return {"pool": _MODELS, "outcomes": outcomes}


def _snapshot(tmp_path: Path, dim: int = 16) -> Path:
    snapshot = tmp_path / "data"
    snapshot.mkdir()
    matrix = _matrix_dict()
    (snapshot / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    rng = np.random.default_rng(7)
    n = len({o["scenario_id"] for o in matrix["outcomes"]})
    np.save(snapshot / "vectors.npy", rng.normal(size=(n, dim)))
    return snapshot


def _manifest(published_rows: list[dict]) -> Manifest:
    return Manifest.model_validate(
        {
            "name": "fixture",
            "title": "fixture benchmark",
            "cookbook": "docs/cookbook/routerbench.md",
            "exactness": "bit-exact",
            "kind": "matrix",
            "data": {"hf_repo": "org/unused", "files": ["matrix.json", "vectors.npy"]},
            "matrix": {
                "matrix_file": "matrix.json",
                "embedding_cache_file": "vectors.npy",
                "embedder_kind": "hashing",
                "embedder_dim": 16,
                "fallback": "pricey",
                "baselines": ["pricey"],
            },
            "published": published_rows,
        }
    )


def _measure(tmp_path: Path) -> dict:
    """First pass: measure what the fixture actually produces (the 'published' numbers)."""
    manifest = _manifest(
        [{"label": "probe", "baseline": "pricey", "accuracy": 0.0, "cost_per_run_usd": 0.0}]
    )
    run_reproduction(manifest, out_dir=tmp_path / "probe", data_dir=_snapshot(tmp_path))
    headline = json.loads(
        (tmp_path / "probe" / "report_vs_pricey.json").read_text(encoding="utf-8")
    )["headline"]
    return headline


def test_matrix_reproduction_is_deterministic_and_bit_exact(tmp_path: Path) -> None:
    headline = _measure(tmp_path)
    manifest = _manifest(
        [
            {
                "label": "routed vs pricey",
                "baseline": "pricey",
                "accuracy": headline["accuracy"],
                "cost_per_run_usd": headline["cost_per_run_usd"],
                "latency_p50_ms": headline["latency_p50_ms"],
            }
        ]
    )
    result = run_reproduction(manifest, out_dir=tmp_path / "run", data_dir=tmp_path / "data")
    assert result.reproduced, result.rows
    verdict = json.loads((tmp_path / "run" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["benchmark"] == "fixture"


def test_rerun_into_the_same_out_dir_survives_a_changed_fit(tmp_path: Path) -> None:
    """A rerun must not abort on the previous run's dial snapshot (stale by construction)."""
    headline = _measure(tmp_path)
    manifest = _manifest(
        [
            {
                "label": "routed vs pricey",
                "baseline": "pricey",
                "accuracy": headline["accuracy"],
                "cost_per_run_usd": headline["cost_per_run_usd"],
            }
        ]
    )
    out = tmp_path / "run"
    assert run_reproduction(manifest, out_dir=out, data_dir=tmp_path / "data").reproduced
    # Change the matrix bytes (a new episode on one scenario), so the refit is a DIFFERENT
    # fit and the first run's snapshot no longer describes it.
    snapshot = tmp_path / "data"
    matrix = json.loads((snapshot / "matrix.json").read_text(encoding="utf-8"))
    extra = dict(matrix["outcomes"][0])
    extra["episode"] = 1
    matrix["outcomes"].append(extra)
    (snapshot / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    result = run_reproduction(manifest, out_dir=out, data_dir=snapshot)
    assert result.rows  # completed and compared; no snapshot abort


def test_divergence_is_reported_not_absorbed(tmp_path: Path) -> None:
    headline = _measure(tmp_path)
    manifest = _manifest(
        [
            {
                "label": "wrong on purpose",
                "baseline": "pricey",
                "accuracy": headline["accuracy"] + 0.05,
                "cost_per_run_usd": headline["cost_per_run_usd"],
            }
        ]
    )
    result = run_reproduction(manifest, out_dir=tmp_path / "run", data_dir=tmp_path / "data")
    assert not result.reproduced
    row = result.rows[0]
    assert not row.fields["accuracy"][2]
    assert row.fields["cost_per_run_usd"][2]


def test_commands_manifest_refuses_without_spend_approval(tmp_path: Path) -> None:
    manifest = Manifest.model_validate(
        {
            "name": "live",
            "title": "live fixture",
            "cookbook": "docs/cookbook/tau-bench.md",
            "exactness": "protocol-exact",
            "kind": "commands",
            "data": {"hf_repo": "org/unused", "files": ["traces.jsonl"]},
            "commands": {
                "steps": [["build", "--file", "{data}/traces.jsonl"]],
                "report_file": "report.json",
                "estimated_spend_usd": 42.0,
            },
            "published": [
                {
                    "label": "row",
                    "accuracy": 0.5,
                    "cost_per_run_usd": 0.1,
                    "tolerance_accuracy": 0.2,
                    "tolerance_cost": 0.3,
                }
            ],
        }
    )
    snapshot = tmp_path / "data"
    snapshot.mkdir()
    with pytest.raises(PermissionError, match=r"\$42"):
        run_reproduction(manifest, out_dir=tmp_path / "run", data_dir=snapshot)


def test_cached_embedder_refuses_unseen_text_and_wrong_shape(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    matrix = OutcomeMatrix.load(snapshot / "matrix.json")
    embedder = CachedTaskEmbedder(matrix, snapshot / "vectors.npy")
    assert len(embedder.embed(["task text 0"])[0]) == 16
    with pytest.raises(ValueError, match="not in the embedding cache"):
        embedder.embed(["text the fit never saw"])
    short = np.load(snapshot / "vectors.npy")[:3]
    np.save(snapshot / "short.npy", short)
    with pytest.raises(ValueError, match="different matrix"):
        CachedTaskEmbedder(matrix, snapshot / "short.npy")


def test_cached_embedder_refuses_duplicate_task_texts(tmp_path: Path) -> None:
    """Two scenarios with identical task text cannot share a text-keyed vector row silently."""
    snapshot = tmp_path / "data"
    snapshot.mkdir()
    matrix = _matrix_dict(n_scenarios=4)
    for outcome in matrix["outcomes"]:
        if outcome["scenario_id"] == "s1":
            outcome["task"] = "task text 0"  # collides with s0's text
    (snapshot / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    np.save(snapshot / "vectors.npy", np.zeros((4, 8)))
    with pytest.raises(ValueError, match="share task text"):
        CachedTaskEmbedder(OutcomeMatrix.load(snapshot / "matrix.json"), snapshot / "vectors.npy")
