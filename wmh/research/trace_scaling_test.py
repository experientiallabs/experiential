"""Tests for the trace-scaling-law ablation (fakes only — no network, no GEPA engine)."""

from __future__ import annotations

import wmh.research.trace_scaling as ts
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.research.ablation import run_ablation
from wmh.research.trace_scaling import BASE, GEPA, TraceScalingAblation


def _trace(i: int) -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get", arguments={"i": i}),
        observation=Observation(content=f"obs {i}"),
    )
    return Trace(trace_id=f"trace-{i:04d}", steps=[step])


def _corpus(n: int) -> list[Trace]:
    return [_trace(i) for i in range(n)]


class _FakeProvider:
    pass


class _FakeJudge:
    pass


def _fake_backends():  # noqa: ANN202 - test factory
    return (_FakeProvider(), _FakeJudge(), None)


def test_conditions_are_mode_cross_count() -> None:
    ab = TraceScalingAblation(
        _corpus(200), "BASE", make_backends=_fake_backends, counts=[10, 20], budget=4
    )
    labels = [c.label for c in ab.conditions()]
    # Default modes are (base, gepa); grid is mode × count in declared order.
    assert labels == ["base@10", "base@20", "gepa@10", "gepa@20"]


def test_counts_capped_at_train_pool_and_deduped() -> None:
    # 20-trace corpus -> small pool; 1000/2000 collapse to the pool size and dedupe to one count.
    ab = TraceScalingAblation(
        _corpus(20),
        "BASE",
        make_backends=_fake_backends,
        counts=[10, 1000, 2000],
        modes=[BASE],
        budget=4,
    )
    pool = len(ab.split.train_pool)
    assert ab.counts == [10, pool]  # 1000 and 2000 both cap to pool, deduped to one entry


def test_base_mode_scores_base_prompt_without_gepa(monkeypatch) -> None:  # noqa: ANN001
    scored_prompt = ""
    test_ids: list[str] = []
    train_n = -1

    def fake_score(prompt, held_out, *, provider, judge, embedder, train, top_k, **_):  # noqa: ANN001, ANN003, ANN202
        nonlocal scored_prompt, test_ids, train_n
        scored_prompt = prompt
        test_ids = [t.trace_id for t in held_out]
        train_n = len(train)
        return 0.5

    def fake_optimize(*a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        raise AssertionError("base mode must not call GEPA")

    monkeypatch.setattr(ts, "score_prompt", fake_score)
    monkeypatch.setattr(ts, "optimize_prompt", fake_optimize)

    ab = TraceScalingAblation(
        _corpus(200),
        "BASE_PROMPT",
        make_backends=_fake_backends,
        counts=[5],
        modes=[BASE],
        budget=4,
    )
    score = ab.run(ab.conditions()[0], seed=0)
    assert score == 0.5
    assert scored_prompt == "BASE_PROMPT"  # scored the base prompt verbatim
    assert train_n == 5
    # Test set is the fixed band, not the train sample.
    assert set(test_ids) == {t.trace_id for t in ab.split.test}


def test_gepa_mode_optimizes_then_scores_winner(monkeypatch) -> None:  # noqa: ANN001
    class _Result:
        prompt = "EVOLVED"

    train_n = -1
    valid_ids: list[str] = []
    budget_seen = -1
    scored_prompt = ""

    def fake_optimize(train, valid, base, *, provider, judge, embedder, budget, seed):  # noqa: ANN001, ANN202
        nonlocal train_n, valid_ids, budget_seen
        train_n = len(train)
        valid_ids = [t.trace_id for t in valid]
        budget_seen = budget
        return _Result()

    def fake_score(prompt, held_out, *, provider, judge, embedder, train, top_k, **_):  # noqa: ANN001, ANN003, ANN202
        nonlocal scored_prompt
        scored_prompt = prompt
        return 0.8

    monkeypatch.setattr(ts, "optimize_prompt", fake_optimize)
    monkeypatch.setattr(ts, "score_prompt", fake_score)

    ab = TraceScalingAblation(
        _corpus(200), "BASE", make_backends=_fake_backends, counts=[7], modes=[GEPA], budget=9
    )
    score = ab.run(ab.conditions()[0], seed=1)
    assert score == 0.8
    assert scored_prompt == "EVOLVED"  # the winning prompt, not the base
    assert train_n == 7
    assert budget_seen == 9
    # GEPA selects on the fixed valid band.
    assert set(valid_ids) == {t.trace_id for t in ab.split.valid}


def test_run_ablation_end_to_end_with_fakes(monkeypatch) -> None:  # noqa: ANN001
    # Fidelity rises with n_train so the report shape (mean/std per condition) is exercised.
    monkeypatch.setattr(ts, "optimize_prompt", lambda *a, **k: type("R", (), {"prompt": "E"})())
    monkeypatch.setattr(
        ts,
        "score_prompt",
        lambda prompt, held_out, **k: 0.3 + 0.001 * len(k["train"]),
    )
    ab = TraceScalingAblation(
        _corpus(200),
        "BASE",
        make_backends=_fake_backends,
        counts=[10, 30],
        modes=[BASE],
        budget=4,
    )
    report = run_ablation(ab, seeds=[0, 1])
    assert report.name == "trace-scaling-law"
    assert [c.condition.label for c in report.conditions] == ["base@10", "base@30"]
    # More train traces -> higher fidelity in this fake.
    by_label = {c.condition.label: c.mean for c in report.conditions}
    assert by_label["base@30"] > by_label["base@10"]
