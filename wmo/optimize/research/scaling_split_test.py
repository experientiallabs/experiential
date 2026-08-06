"""Tests for the fixed-test / growing-train scaling split."""

from __future__ import annotations

import pytest

from wmo.common.core.types import Trace
from wmo.optimize.research.scaling_split import partition_corpus, subsample_train


def _corpus(n: int) -> list[Trace]:
    # Distinct ids spread across the hash space so bands are populated.
    return [Trace(trace_id=f"trace-{i:04d}") for i in range(n)]


def test_partition_is_a_pure_function_of_trace_id() -> None:
    corpus = _corpus(200)
    a = partition_corpus(corpus)
    # Reordering the input must not move any trace between bands.
    b = partition_corpus(list(reversed(corpus)))
    assert {t.trace_id for t in a.test} == {t.trace_id for t in b.test}
    assert {t.trace_id for t in a.valid} == {t.trace_id for t in b.valid}
    assert {t.trace_id for t in a.train_pool} == {t.trace_id for t in b.train_pool}


def test_bands_are_disjoint_and_cover_everything() -> None:
    corpus = _corpus(300)
    s = partition_corpus(corpus)
    ids = lambda xs: {t.trace_id for t in xs}  # noqa: E731 - terse local
    assert ids(s.test) | ids(s.valid) | ids(s.train_pool) == ids(corpus)
    assert not (ids(s.test) & ids(s.valid))
    assert not (ids(s.test) & ids(s.train_pool))
    assert not (ids(s.valid) & ids(s.train_pool))


def test_test_and_valid_are_fixed_as_corpus_grows() -> None:
    small = partition_corpus(_corpus(100))
    large = partition_corpus(_corpus(400))  # superset of the first 100 ids
    small_ids = {t.trace_id for t in _corpus(100)}
    # Every trace shared by both corpora keeps its band; the larger corpus only grows the pool.
    for band in ("test", "valid"):
        small_band = {t.trace_id for t in getattr(small, band)}
        large_band = {t.trace_id for t in getattr(large, band) if t.trace_id in small_ids}
        assert small_band == large_band


def test_fractions_roughly_match_request() -> None:
    s = partition_corpus(_corpus(1000), test_frac=0.2, valid_frac=0.15)
    assert 0.15 < len(s.test) / 1000 < 0.25
    assert 0.10 < len(s.valid) / 1000 < 0.20


def test_subsample_is_nested_for_a_fixed_seed() -> None:
    pool = _corpus(100)
    small = subsample_train(pool, 10, seed=0)
    large = subsample_train(pool, 25, seed=0)
    # n=10 sample is a prefix of the n=25 sample: scaling up adds traces, never resamples.
    assert [t.trace_id for t in small] == [t.trace_id for t in large[:10]]


def test_subsample_caps_at_pool_size() -> None:
    pool = _corpus(8)
    got = subsample_train(pool, 100, seed=1)
    assert len(got) == 8


def test_subsample_order_is_input_independent() -> None:
    pool = _corpus(50)
    a = subsample_train(pool, 20, seed=3)
    b = subsample_train(list(reversed(pool)), 20, seed=3)
    assert [t.trace_id for t in a] == [t.trace_id for t in b]


def test_partition_rejects_degenerate_fractions() -> None:
    with pytest.raises(ValueError, match="< 1"):
        partition_corpus(_corpus(10), test_frac=0.6, valid_frac=0.5)
    with pytest.raises(ValueError, match="> 0"):
        partition_corpus(_corpus(10), test_frac=0.0, valid_frac=0.2)
