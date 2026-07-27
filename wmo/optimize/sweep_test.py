"""Tests for the sweep library: the scenario cut, the cost projection, and the coverage contract.

The two CLI faces are covered end to end by `wmo/cli/route_app_test.py` and
`wmo/cli/optimize_model_app_test.py`; these exercise the shared core directly, where the edge
cases are cheaper to state than to stage through a command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.config import HarnessConfig, save_config
from wmo.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.sweep import (
    SweepError,
    SweepPlan,
    SweepRun,
    Unevenness,
    coverage,
    plan_sweep,
    preflight_pool,
    resolve_config,
    unevenness,
)
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.pool import PoolEntry, load_pool
from wmo.serving.traces_source import TRACES_FILENAME
from wmo.tracking import RunRecord, UsageTotals


def _traces(count: int = 30) -> list[Trace]:
    return [
        Trace(
            trace_id=f"tr-{index:03d}",
            steps=[
                Step(
                    action=Action(kind=ActionKind.TOOL_CALL, name="ls", arguments={"path": "."}),
                    observation=Observation(content="a.txt"),
                    task=f"task tr-{index:03d}",
                )
            ],
        )
        for index in reversed(range(count))
    ]


def _model_dir(tmp_path: Path, *, with_corpus: bool = True) -> Path:
    model_dir = tmp_path / ".wmo" / "models" / "support"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
            train_split=0.8,
        ),
        model_dir,
    )
    if with_corpus:
        write_traces_jsonl(_traces(), model_dir / TRACES_FILENAME)
    return model_dir


def _pool_file(tmp_path: Path) -> Path:
    path = tmp_path / "pool.toml"
    path.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n",
        encoding="utf-8",
    )
    return path


def _plan(tmp_path: Path, **overrides: int) -> SweepPlan:
    model_dir = _model_dir(tmp_path)
    settings = {"scenarios": 3, "episodes": 1, "max_steps": 4} | overrides
    return plan_sweep(
        model_dir=model_dir,
        config=resolve_config(model_dir),
        pool=load_pool(_pool_file(tmp_path)),
        out_path=tmp_path / "matrix.json",
        traces_file=None,
        assume_input_tokens=2000,
        assume_output_tokens=250,
        **settings,
    )


def test_the_plan_cuts_the_same_held_out_prefix_every_time(tmp_path: Path) -> None:
    # Deterministic by trace id, so `scenarios` always names the same tasks on the same corpus:
    # a matrix is only comparable to another one measured on the same scenario set.
    first = _plan(tmp_path)
    second = _plan(tmp_path)
    assert [scenario.task for scenario in first.scenarios] == [
        scenario.task for scenario in second.scenarios
    ]
    assert len(first.scenarios) == 3
    assert not first.tiny_corpus


def test_the_cost_projection_multiplies_real_cell_counts_by_the_assumed_tokens(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    # 3 scenarios x 1 episode x 4 calls = 12 calls at (2000 x $1 + 250 x $2)/1e6 = $0.0025 each.
    assert plan.cells == 3
    line = plan.cost_lines[0]
    assert line.calls == 12
    assert plan.total_usd == pytest.approx(0.03)


def test_a_bigger_step_budget_projects_proportionally_more(tmp_path: Path) -> None:
    # The negative control for the projection: the CALL count is real, so doubling the cap
    # doubles the number, which is what makes the estimate worth reading.
    assert _plan(tmp_path, max_steps=8).total_usd == pytest.approx(
        _plan(tmp_path, max_steps=4).total_usd * 2
    )


def test_a_corpus_the_build_did_not_keep_says_where_to_pass_it(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path, with_corpus=False)
    with pytest.raises(SweepError) as caught:
        plan_sweep(
            model_dir=model_dir,
            config=resolve_config(model_dir),
            pool=load_pool(_pool_file(tmp_path)),
            out_path=tmp_path / "matrix.json",
            traces_file=None,
            scenarios=3,
            episodes=1,
            max_steps=4,
            assume_input_tokens=2000,
            assume_output_tokens=250,
        )
    message = str(caught.value)
    assert "no trace corpus" in message and "--traces" in message


def test_an_unwritable_destination_is_refused_before_anything_is_spent(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file, not a directory", encoding="utf-8")
    with pytest.raises(SweepError, match="cannot write the outcome matrix"):
        plan_sweep(
            model_dir=model_dir,
            config=resolve_config(model_dir),
            pool=load_pool(_pool_file(tmp_path)),
            out_path=blocker / "matrix.json",
            traces_file=None,
            scenarios=3,
            episodes=1,
            max_steps=4,
            assume_input_tokens=2000,
            assume_output_tokens=250,
        )
    assert blocker.is_file()  # the check is pure: it creates nothing on the way out


def test_a_missing_pool_file_is_a_sweep_error_not_a_traceback(tmp_path: Path) -> None:
    with pytest.raises(SweepError):
        preflight_pool(tmp_path / "nope.toml")


def _matrix(cells: dict[tuple[str, str], list[float | None]]) -> OutcomeMatrix:
    """A matrix from {(model, scenario): [reward per episode]}; None is an unscored episode."""
    names = sorted({model for model, _ in cells})
    return OutcomeMatrix(
        pool=[
            PoolEntry(
                name=name,
                kind=ProviderKind.OPENAI,
                model=f"{name}-1",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
            for name in names
        ],
        outcomes=[
            ScenarioOutcome(
                scenario_id=scenario,
                task=f"task {scenario}",
                model=model,
                episode=episode,
                reward=reward,
                error=None if reward is not None else "throttled (429)",
            )
            for (model, scenario), rewards in cells.items()
            for episode, reward in enumerate(rewards)
        ],
    )


def test_identical_scored_evidence_is_even() -> None:
    rows = coverage(_matrix({("a", "s1"): [0.5], ("b", "s1"): [0.9]}))
    assert unevenness(rows) is Unevenness.EVEN
    assert [row.scored for row in rows] == [1, 1]


def test_the_same_losses_for_every_candidate_are_still_a_comparison() -> None:
    # Like-for-like on less data: the counts show the loss and a fit over it is not biased.
    rows = coverage(_matrix({("a", "s1"): [None], ("b", "s1"): [None], ("a", "s2"): [0.5]}))
    assert unevenness(rows) is Unevenness.SCENARIOS  # b has no s2 row at all
    losses = _matrix(
        {("a", "s1"): [0.5, None], ("b", "s1"): [0.9, None]},
    )
    assert unevenness(coverage(losses)) is Unevenness.EVEN


def test_different_scenario_sets_and_different_episode_counts_are_told_apart() -> None:
    # Two distinct biases, so two distinct verdicts: the CLI prints a different warning for each,
    # and naming which one happened is what makes the message actionable.
    different_sets = _matrix(
        {("a", "s1"): [0.5], ("a", "s2"): [0.5], ("b", "s1"): [0.9], ("b", "s2"): [None]}
    )
    assert unevenness(coverage(different_sets)) is Unevenness.SCENARIOS
    thinner_episodes = _matrix(
        {("a", "s1"): [0.5, 0.5, 0.5], ("b", "s1"): [0.9, None, None]},
    )
    assert unevenness(coverage(thinner_episodes)) is Unevenness.EPISODES


def test_coverage_carries_the_first_error_of_a_candidate_that_never_scored() -> None:
    rows = coverage(_matrix({("a", "s1"): [0.5], ("b", "s1"): [None]}))
    never_scored = next(row for row in rows if row.candidate == "b")
    assert never_scored.scored == 0
    assert never_scored.first_error is not None and "429" in never_scored.first_error
    assert never_scored.lost_scenarios == ("s1",)


def _run(*, metered: int, unmetered: int, cost: float = 0.0) -> SweepRun:
    """A `SweepRun` with the given metering coverage, for the reporting rules alone."""
    return SweepRun(
        matrix=_matrix({("a", "s1"): [0.5]}),
        candidate_usd=0.001,
        world_model_usage=RunRecord(
            run_id="sweep-1", kind="sweep", total=UsageTotals(cost_usd=cost)
        ),
        episodes_metered=metered,
        episodes_unmetered=unmetered,
        usage_path=None,
    )


def test_full_metering_coverage_reports_no_gap() -> None:
    assert _run(metered=6, unmetered=0, cost=0.12).metering_gap is None
    assert _run(metered=6, unmetered=0, cost=0.12).world_model_usd == pytest.approx(0.12)


def test_metering_nothing_says_unknown_rather_than_zero() -> None:
    # `evaluate_pool` accepts any Env, and one that exposes no usage record leaves the world
    # model's cost UNKNOWN. Reporting that as $0.00 is exactly the zero the numbers-honesty rule
    # forbids: it would read as "the simulator was free".
    gap = _run(metered=0, unmetered=6).metering_gap
    assert gap is not None
    assert "not measured" in gap and "unknown rather than zero" in gap


def test_partial_metering_says_how_partial() -> None:
    gap = _run(metered=4, unmetered=2, cost=0.08).metering_gap
    assert gap is not None
    assert "partial: 4 of 6 episode(s)" in gap
