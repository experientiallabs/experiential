"""Tests for the query-embedding sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wmo.serving.query_embeddings import QUERY_EMBEDDING_FILENAME, QueryEmbeddingStore


def _unit(dim: int, seed: int = 0) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim)
    return vector / np.linalg.norm(vector)


def _resolved(store: QueryEmbeddingStore, ref: str) -> np.ndarray:
    """The vector behind `ref`, narrowed: these cases all expect one to be there."""
    vector = store.get(ref)
    assert vector is not None, ref
    return vector


def test_a_vector_round_trips_through_its_ref(tmp_path: Path) -> None:
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    vector = _unit(512)
    ref = store.append("chatcmpl-abc", vector)
    assert ref == f"{QUERY_EMBEDDING_FILENAME}#chatcmpl-abc"
    resolved = store.get(ref)
    assert resolved is not None
    # float16 holds ~3 decimal digits, which is the documented precision of this store.
    assert np.allclose(resolved, vector, atol=1e-3)


def test_a_disabled_store_records_nothing_and_refs_nothing() -> None:
    store = QueryEmbeddingStore(None)
    assert store.append("chatcmpl-abc", _unit(8)) is None
    assert store.get("anything#chatcmpl-abc") is None


def test_rows_append_rather_than_replace(tmp_path: Path) -> None:
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path)
    first, second = _unit(64, seed=1), _unit(64, seed=2)
    store.append("one", first)
    store.append("two", second)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert np.allclose(_resolved(store, "x#one"), first, atol=1e-3)
    assert np.allclose(_resolved(store, "x#two"), second, atol=1e-3)


def test_an_unknown_id_resolves_to_none(tmp_path: Path) -> None:
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    store.append("one", _unit(16))
    assert store.get(f"{QUERY_EMBEDDING_FILENAME}#missing") is None


def test_one_unreadable_row_does_not_hide_the_rest(tmp_path: Path) -> None:
    # The same resilience `RequestLog.replay` has: a line truncated by a hard kill costs that
    # row, not every row after it.
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path)
    store.append("one", _unit(16))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "two", "dim": 16, "f1\n')
    wanted = _unit(16, seed=5)
    store.append("three", wanted)
    assert np.allclose(_resolved(store, "x#three"), wanted, atol=1e-3)


def test_a_row_whose_length_contradicts_its_dim_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path)
    store.append("one", _unit(16))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["dim"] = 32  # the payload still holds 16
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert store.get("x#one") is None


def test_an_id_carrying_the_ref_separator_is_refused(tmp_path: Path) -> None:
    # It would make the ref ambiguous to split, and no completion id contains one.
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    with pytest.raises(ValueError, match="separates"):
        store.append("chatcmpl#abc", _unit(8))


def test_only_one_dimensional_vectors_are_accepted(tmp_path: Path) -> None:
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    with pytest.raises(ValueError, match="one query vector"):
        store.append("one", np.zeros((2, 8)))


@pytest.mark.parametrize(("dim", "max_kb_per_1k"), [(512, 1600), (3072, 8600)])
def test_the_documented_size_per_1k_requests_holds(
    tmp_path: Path, dim: int, max_kb_per_1k: int
) -> None:
    """Pins the module docstring's size table, which is what an operator provisions disk from."""
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path)
    for index in range(20):
        store.append(f"chatcmpl-{index:032x}", _unit(dim, seed=index))
    per_1k_kb = path.stat().st_size / 20 * 1000 / 1024
    assert per_1k_kb <= max_kb_per_1k, f"dim {dim}: {per_1k_kb:.0f} KB per 1k requests"


def test_the_store_rotates_once_past_its_cap_and_keeps_one_generation(tmp_path: Path) -> None:
    # A default-on store on a busy endpoint is otherwise a slow disk leak; the cap bounds it at
    # twice the limit forever.
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path, max_bytes=2000)
    for index in range(12):
        store.append(f"row{index}", _unit(256, seed=index))
    previous = tmp_path / "query_embeddings.1.jsonl"
    assert previous.is_file(), sorted(p.name for p in tmp_path.iterdir())
    assert path.stat().st_size < 2000 + 2000  # the current file restarted after the roll
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "query_embeddings.1.jsonl",
        QUERY_EMBEDDING_FILENAME,
    ]


def test_a_ref_still_resolves_after_one_rotation(tmp_path: Path) -> None:
    # Written to survive row-size drift: append until the roll first happens, then stop, so this
    # tests exactly one rotation rather than however many the byte math happens to produce.
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    previous = tmp_path / "query_embeddings.1.jsonl"
    store = QueryEmbeddingStore(path, max_bytes=1500)
    early = _unit(256, seed=99)
    store.append("early", early)
    for index in range(50):
        store.append(f"filler{index}", _unit(256, seed=index))
        if previous.is_file():
            break
    assert previous.is_file()
    assert np.allclose(_resolved(store, f"{QUERY_EMBEDDING_FILENAME}#early"), early, atol=1e-3)


def test_a_ref_older_than_the_retention_window_resolves_to_none(tmp_path: Path) -> None:
    # The bounded-retention trade, made explicit: past one rotation the vector is gone and the
    # caller gets None rather than a stale or wrong answer.
    path = tmp_path / QUERY_EMBEDDING_FILENAME
    store = QueryEmbeddingStore(path, max_bytes=900)
    store.append("ancient", _unit(256))
    for index in range(24):
        store.append(f"filler{index}", _unit(256, seed=index))
    assert store.get(f"{QUERY_EMBEDDING_FILENAME}#ancient") is None


def test_rotation_never_grows_past_two_generations(tmp_path: Path) -> None:
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME, max_bytes=800)
    for index in range(60):
        store.append(f"row{index}", _unit(256, seed=index))
    assert len(list(tmp_path.iterdir())) == 2
