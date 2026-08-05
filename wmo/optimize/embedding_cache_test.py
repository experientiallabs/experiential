"""The recorded-vector embedder's contract: serve the fit's own vectors, refuse everything else."""

import json
from pathlib import Path

import numpy as np
import pytest

from wmo.optimize.embedding_cache import CachedTaskEmbedder
from wmo.optimize.outcomes import OutcomeMatrix

_MODELS = [
    {
        "name": "cheap",
        "kind": "anthropic",
        "model": "m-cheap",
        "input_per_mtok": 1.0,
        "output_per_mtok": 2.0,
    },
]


def _matrix_dict(n_scenarios: int = 4) -> dict:
    """A minimal valid matrix: one arm, one episode per scenario, distinct task texts."""
    outcomes = []
    for index in range(n_scenarios):
        outcomes.append(
            {
                "scenario_id": f"s{index}",
                "task": f"task text {index}",
                "model": "cheap",
                "episode": 0,
                "reward": 0.5,
                "success": False,
                "critique": "",
                "steps": 1,
                "stop_reason": "done",
                "usage": {"input_tokens": 10, "output_tokens": 10},
                "cost_usd": 0.001,
                "call_seconds": [0.5],
                "replies": [],
                "error": None,
            }
        )
    return {"pool": _MODELS, "outcomes": outcomes}


def _snapshot(tmp_path: Path, dim: int = 16) -> Path:
    """Write a matrix plus a row-aligned vector cache and return the directory."""
    snapshot = tmp_path / "data"
    snapshot.mkdir()
    matrix = _matrix_dict()
    (snapshot / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    rng = np.random.default_rng(7)
    n = len({o["scenario_id"] for o in matrix["outcomes"]})
    np.save(snapshot / "vectors.npy", rng.normal(size=(n, dim)))
    return snapshot


def test_serves_the_recorded_vector_for_seen_text_exactly(tmp_path: Path) -> None:
    """Row i of the cache must come back verbatim for scenario i's task text."""
    snapshot = _snapshot(tmp_path)
    matrix = OutcomeMatrix.load(snapshot / "matrix.json")
    embedder = CachedTaskEmbedder(matrix, snapshot / "vectors.npy")
    recorded = np.load(snapshot / "vectors.npy")
    assert embedder.dim == 16
    served = embedder.embed(["task text 0", "task text 2"])
    assert np.array_equal(np.asarray(served[0]), recorded[0])
    assert np.array_equal(np.asarray(served[1]), recorded[2])


def test_refuses_unseen_text_naming_the_count(tmp_path: Path) -> None:
    """A cache cannot embed text it has not seen; serve-time traffic needs the real embedder."""
    snapshot = _snapshot(tmp_path)
    matrix = OutcomeMatrix.load(snapshot / "matrix.json")
    embedder = CachedTaskEmbedder(matrix, snapshot / "vectors.npy")
    with pytest.raises(ValueError, match="1 of 2 texts are not in the embedding cache"):
        embedder.embed(["task text 1", "text the fit never saw"])


def test_refuses_a_cache_built_for_a_different_matrix(tmp_path: Path) -> None:
    """Row-count mismatch is the cheapest alignment check and must fail loudly at load."""
    snapshot = _snapshot(tmp_path)
    matrix = OutcomeMatrix.load(snapshot / "matrix.json")
    short = np.load(snapshot / "vectors.npy")[:2]
    np.save(snapshot / "short.npy", short)
    with pytest.raises(ValueError, match="different matrix"):
        CachedTaskEmbedder(matrix, snapshot / "short.npy")


def test_refuses_duplicate_task_texts(tmp_path: Path) -> None:
    """Two scenarios with identical task text cannot share a text-keyed vector row silently."""
    snapshot = tmp_path / "data"
    snapshot.mkdir()
    matrix = _matrix_dict()
    for outcome in matrix["outcomes"]:
        if outcome["scenario_id"] == "s1":
            outcome["task"] = "task text 0"  # collides with s0's text
    (snapshot / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    np.save(snapshot / "vectors.npy", np.zeros((4, 8)))
    with pytest.raises(ValueError, match="share task text"):
        CachedTaskEmbedder(OutcomeMatrix.load(snapshot / "matrix.json"), snapshot / "vectors.npy")
