"""Tests for the GEPA-scaling-law ablation (fakes only - no network, no GEPA engine)."""

from __future__ import annotations

from collections.abc import Callable

import wmh.research.gepa_scaling as gs
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.replay import ReplayReport, StepResult
from wmh.research.ablation import run_ablation
from wmh.research.gepa_scaling import GepaScalingAblation


def _trace(i: int, n_steps: int = 1) -> Trace:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get", arguments={"i": i, "s": s}),
            observation=Observation(content=f"obs {i}.{s}"),
        )
        for s in range(n_steps)
    ]
    return Trace(trace_id=f"trace-{i:04d}", steps=steps)


def _corpus(n: int, n_steps: int = 1) -> list[Trace]:
    return [_trace(i, n_steps) for i in range(n)]


class _FakeProvider:
    pass


class _FakeJudge:
    pass


def _fake_backends():  # noqa: ANN202 - test factory
    return (_FakeProvider(), _FakeJudge(), None)


def test_conditions_are_the_grid_with_tn_bb_labels() -> None:
    ab = GepaScalingAblation(
        _corpus(200),
        "BASE",
        make_backends=_fake_backends,
        grid=[(64, 0), (64, 1), (64, 8), (16, 8)],
    )
    labels = [c.label for c in ab.conditions()]
    assert labels == ["t64_b0", "t64_b1", "t64_b8", "t16_b8"]
    params = [c.params for c in ab.conditions()]
    assert params[0] == {"n_train": 64, "budget": 0}
    assert params[3] == {"n_train": 16, "budget": 8}


def test_grid_counts_capped_at_pool_and_deduped() -> None:
    # 20-trace corpus -> small pool; 1000 and 2000 both cap to the pool and collapse to one point.
    ab = GepaScalingAblation(
        _corpus(20),
        "BASE",
        make_backends=_fake_backends,
        grid=[(10, 8), (1000, 8), (2000, 8)],
    )
    pool = len(ab.split.train_pool)
    assert ab.grid == [(10, 8), (pool, 8)]


def test_budget_zero_scores_base_prompt_without_gepa(monkeypatch) -> None:  # noqa: ANN001
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
        raise AssertionError("budget=0 must not call GEPA")

    monkeypatch.setattr(gs, "score_prompt", fake_score)
    monkeypatch.setattr(gs, "optimize_prompt", fake_optimize)

    ab = GepaScalingAblation(
        _corpus(200), "BASE_PROMPT", make_backends=_fake_backends, grid=[(5, 0)]
    )
    score = ab.run(ab.conditions()[0], seed=0)
    assert score == 0.5
    assert scored_prompt == "BASE_PROMPT"  # scored the base prompt verbatim
    assert train_n == 5
    # Test set is the fixed band, not the train sample.
    assert set(test_ids) == {t.trace_id for t in ab.split.test}


def test_budget_positive_optimizes_then_scores_winner(monkeypatch) -> None:  # noqa: ANN001
    class _Result:
        prompt = "EVOLVED"

    train_n = -1
    budget_seen = -1
    valid_ids: list[str] = []
    scored_prompt = ""

    def fake_optimize(train, valid, base, *, budget, seed, **_):  # noqa: ANN001, ANN003, ANN202
        nonlocal train_n, budget_seen, valid_ids
        train_n = len(train)
        budget_seen = budget
        valid_ids = [t.trace_id for t in valid]
        return _Result()

    def fake_score(prompt, held_out, **_):  # noqa: ANN001, ANN003, ANN202
        nonlocal scored_prompt
        scored_prompt = prompt
        return 0.8

    monkeypatch.setattr(gs, "optimize_prompt", fake_optimize)
    monkeypatch.setattr(gs, "score_prompt", fake_score)

    ab = GepaScalingAblation(_corpus(200), "BASE", make_backends=_fake_backends, grid=[(7, 9)])
    score = ab.run(ab.conditions()[0], seed=1)
    assert score == 0.8
    assert scored_prompt == "EVOLVED"  # the winning prompt, not the base
    assert train_n == 7
    assert budget_seen == 9
    # GEPA selects on the capped valid subset, drawn from the fixed valid band.
    assert set(valid_ids) <= {t.trace_id for t in ab.split.valid}
    assert set(valid_ids) == {t.trace_id for t in ab.gepa_valid}


def test_gepa_valset_capped_by_steps() -> None:
    # 3-step traces, cap 7 steps -> at most 2 traces (6 steps) fit under the cap; never empty.
    ab = GepaScalingAblation(
        _corpus(200, n_steps=3),
        "BASE",
        make_backends=_fake_backends,
        grid=[(5, 1)],
        gepa_val_steps=7,
    )
    n_steps = sum(len(t.steps) for t in ab.gepa_valid)
    assert 0 < n_steps <= 7
    assert len(ab.gepa_valid) == 2

    # A cap smaller than one trace still keeps one trace (GEPA needs a non-empty valset).
    tiny = GepaScalingAblation(
        _corpus(200, n_steps=3),
        "BASE",
        make_backends=_fake_backends,
        grid=[(5, 1)],
        gepa_val_steps=1,
    )
    assert len(tiny.gepa_valid) == 1


def test_gepa_valset_cap_skips_oversized_traces_and_keeps_filling() -> None:
    """Greedy fill, not a prefix: an over-cap trace mid-list must not starve the valset (a real
    swe-bench failure mode - one 30-step trace early in the shuffle left GEPA a 2-step valset)."""
    from wmh.research.gepa_scaling import _cap_by_steps

    mixed = [_trace(0, 1), _trace(1, 30), _trace(2, 1), _trace(3, 2), _trace(4, 4)]
    picked = _cap_by_steps(mixed, 8)
    assert sum(len(t.steps) for t in picked) == 8  # 1 + 1 + 2 + 4: the 30-step trace is skipped
    assert [t.trace_id for t in picked] == ["trace-0000", "trace-0002", "trace-0003", "trace-0004"]


def test_hard_threshold_prescore_builds_step_filter(monkeypatch) -> None:  # noqa: ANN001
    """With hard_threshold set, the run pre-scores probe+valset steps with the base prompt and
    passes GEPA a filter accepting exactly the below-threshold steps (select_on_hard=True)."""
    filters: list[Callable[[Step], bool] | None] = []
    selects: list[bool] = []
    trains: list[list[Trace]] = []

    def fake_replay(prompt, held_out, provider, judge, **_):  # noqa: ANN001, ANN003, ANN202
        # Steps whose observation content ends in ".0" score low, the rest high.
        results = []
        for trace in held_out:
            for step in trace.steps:
                low = step.observation.content.endswith(".0")
                results.append(
                    StepResult(
                        trace_id=trace.trace_id,
                        action="a",
                        actual=step.observation.content,
                        predicted="p",
                        score=0.2 if low else 1.0,
                    )
                )
        return ReplayReport(results=results, n_steps=len(results))

    def fake_optimize(train, valid, base, *, hard_step_filter, select_on_hard, **_):  # noqa: ANN001, ANN003, ANN202
        filters.append(hard_step_filter)
        selects.append(select_on_hard)
        trains.append(train)
        return type("R", (), {"prompt": "EVOLVED"})()

    monkeypatch.setattr(gs, "replay", fake_replay)
    monkeypatch.setattr(gs, "optimize_prompt", fake_optimize)
    monkeypatch.setattr(gs, "score_prompt", lambda *a, **k: 0.9)

    ab = GepaScalingAblation(
        _corpus(200, n_steps=2),
        "BASE",
        make_backends=_fake_backends,
        grid=[(6, 4)],
        hard_threshold=0.5,
    )
    assert ab.run(ab.conditions()[0], seed=0) == 0.9
    hard_filter = filters[0]
    assert hard_filter is not None
    assert selects == [True]
    # The filter accepts exactly the pre-scored-low steps (identity-keyed on the train sample).
    for trace in trains[0]:
        assert hard_filter(trace.steps[0]) is True  # "obs i.0" scored 0.2
        assert hard_filter(trace.steps[1]) is False  # "obs i.1" scored 1.0


def test_empty_train_pool_yields_empty_grid_not_t0_points() -> None:
    # 3 traces -> pool can be 0 when every trace_id hashes into the test/valid bands (force it
    # with fractions that leave a sliver): use a big valid_frac so the pool is tiny/empty. The
    # invariant under test: the zero filter applies AFTER the pool cap, so no t0_bN point survives.
    ab = GepaScalingAblation(
        _corpus(3),
        "BASE",
        make_backends=_fake_backends,
        grid=[(64, 8)],
        test_frac=0.5,
        valid_frac=0.49,
    )
    assert all(n > 0 for n, _ in ab.grid)
    if not ab.split.train_pool:
        assert ab.grid == []


def test_val_fill_is_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="val_fill"):
        GepaScalingAblation(
            _corpus(50), "BASE", make_backends=_fake_backends, grid=[(5, 1)], val_fill="gredy"
        )


def test_inclusive_val_fill_keeps_long_traces() -> None:
    """`val_fill="inclusive"` stops once the step budget is REACHED without skipping long traces -
    removing the greedy fill's short-trace bias (the valset-representativeness failure)."""
    from wmh.research.gepa_scaling import _fill_by_steps

    mixed = [_trace(0, 1), _trace(1, 30), _trace(2, 1), _trace(3, 2)]
    picked = _fill_by_steps(mixed, 8)
    # Takes traces in order until >= 8 steps: 1 + 30 -> stops at 31 (long trace INCLUDED).
    assert [t.trace_id for t in picked] == ["trace-0000", "trace-0001"]
    # Never empty, even for a non-positive budget: an empty valset would make GEPA silently
    # validate on the train traces.
    assert len(_fill_by_steps(mixed, 0)) == 1

    ab = GepaScalingAblation(
        _corpus(200, n_steps=3),
        "BASE",
        make_backends=_fake_backends,
        grid=[(5, 1)],
        gepa_val_steps=7,
        val_fill="inclusive",
    )
    assert sum(len(t.steps) for t in ab.gepa_valid) >= 7  # reaches the budget, never starves


def test_minibatch_size_forwarded(monkeypatch) -> None:  # noqa: ANN001
    sizes: list[int | None] = []

    def fake_optimize(train, valid, base, *, minibatch_size, **_):  # noqa: ANN001, ANN003, ANN202
        sizes.append(minibatch_size)
        return type("R", (), {"prompt": "E"})()

    monkeypatch.setattr(gs, "optimize_prompt", fake_optimize)
    monkeypatch.setattr(gs, "score_prompt", lambda *a, **k: 0.5)
    ab = GepaScalingAblation(
        _corpus(50), "BASE", make_backends=_fake_backends, grid=[(5, 2)], minibatch_size=8
    )
    ab.run(ab.conditions()[0], seed=0)
    assert sizes == [8]


def test_recheck_steps_builds_disjoint_valset_slice(monkeypatch) -> None:  # noqa: ANN001
    """With recheck_steps set, GEPA gets a step-capped recheck set DISJOINT from its selection
    valset (both drawn from the fixed valid band), and it is forwarded to optimize_prompt."""
    rechecks: list[list[Trace] | None] = []

    def fake_optimize(train, valid, base, *, recheck, **_):  # noqa: ANN001, ANN003, ANN202
        rechecks.append(recheck)
        return type("R", (), {"prompt": "EVOLVED"})()

    monkeypatch.setattr(gs, "optimize_prompt", fake_optimize)
    monkeypatch.setattr(gs, "score_prompt", lambda *a, **k: 0.9)

    ab = GepaScalingAblation(
        _corpus(200, n_steps=2),
        "BASE",
        make_backends=_fake_backends,
        grid=[(6, 4)],
        gepa_val_steps=6,
        recheck_steps=6,
    )
    ab.run(ab.conditions()[0], seed=0)
    recheck = rechecks[0]
    assert recheck is not None
    assert 0 < sum(len(t.steps) for t in recheck) <= 6
    val_ids = {t.trace_id for t in ab.gepa_valid}
    assert val_ids.isdisjoint({t.trace_id for t in recheck})

    # Default (recheck_steps=0) forwards None - the re-check falls back to the valset.
    ab_off = GepaScalingAblation(
        _corpus(200, n_steps=2), "BASE", make_backends=_fake_backends, grid=[(6, 4)]
    )
    ab_off.run(ab_off.conditions()[0], seed=0)
    assert rechecks[1] is None


def test_run_ablation_end_to_end_with_fakes(monkeypatch) -> None:  # noqa: ANN001
    # Fidelity rises with budget so the report shape (mean/std per condition) is exercised.
    monkeypatch.setattr(gs, "optimize_prompt", lambda *a, **k: type("R", (), {"prompt": "E"})())
    monkeypatch.setattr(
        gs,
        "score_prompt",
        lambda prompt, held_out, **k: 0.5 if prompt == "BASE" else 0.7,
    )
    ab = GepaScalingAblation(
        _corpus(200),
        "BASE",
        make_backends=_fake_backends,
        grid=[(16, 0), (16, 4)],
    )
    report = run_ablation(ab, seeds=[0, 1])
    assert report.name == "gepa-scaling-law"
    assert [c.condition.label for c in report.conditions] == ["t16_b0", "t16_b4"]
    by_label = {c.condition.label: c.mean for c in report.conditions}
    assert by_label["t16_b4"] > by_label["t16_b0"]
