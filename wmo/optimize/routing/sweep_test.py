"""Tests for the sweep library: the scenario cut, the cost projection, and the coverage contract.

The two CLI faces are covered end to end by `wmo/cli/route_app_test.py` and
`wmo/cli/optimize_model_app_test.py`; these exercise the shared core directly, where the edge
cases are cheaper to state than to stage through a command.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from wmo.common.config import HarnessConfig, save_config
from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmo.common.observability import Phase, RunRecord, UsageTotals
from wmo.common.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.common.providers.pool import PoolEntry, load_pool
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.routing.compression import CompressionConfig
from wmo.optimize.routing.evaluation import CellKey
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.sweep import (
    SweepError,
    SweepPlan,
    SweepRun,
    Unevenness,
    coverage,
    execute_sweep,
    plan_sweep,
    preflight_pool,
    resolve_config,
    resumable_cells,
    unevenness,
)
from wmo.optimize.routing.sweep_partial import partial_path
from wmo.simulation.ingest.otel_writer import write_traces_jsonl
from wmo.simulation.serving.traces_source import TRACES_FILENAME

if TYPE_CHECKING:
    from collections.abc import Callable

    from wmo.runtime.environment import Env
    from wmo.simulation.model.world_model import WorldModel


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


def _plan(
    tmp_path: Path,
    *,
    scenarios: int = 3,
    episodes: int = 1,
    max_steps: int = 4,
    compression: CompressionConfig | None = None,
    max_concurrency: int = 1,
    out_name: str = "matrix.json",
) -> SweepPlan:
    model_dir = _model_dir(tmp_path)
    return plan_sweep(
        model_dir=model_dir,
        config=resolve_config(model_dir),
        pool=load_pool(_pool_file(tmp_path)),
        out_path=tmp_path / out_name,
        traces_file=None,
        assume_input_tokens=2000,
        assume_output_tokens=250,
        scenarios=scenarios,
        episodes=episodes,
        max_steps=max_steps,
        compression=compression,
        max_concurrency=max_concurrency,
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


def test_preflight_skips_disabled_entries_and_refuses_an_all_disabled_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `enabled = false` is the roster's per-model toggle: a turned-off entry is neither
    # prepared (its backend may legitimately be unusable right now) nor swept.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pool_file = tmp_path / "pool.toml"
    pool_file.write_text(
        """
[[model]]
name = "on"
kind = "openai"
model = "gpt-5.4"

[[model]]
name = "off"
kind = "openai"
model = "gpt-5.4"
api_key_env = "TEST_VAR_NOBODY_SETS"
enabled = false
""",
        encoding="utf-8",
    )
    # The disabled entry names an unset api_key_env, which would fail preflight loudly if it
    # were still a candidate; skipping it is the point of the toggle. (Schema validation still
    # covers it: the toggle removes an entry from selection, not from load.)
    preflight = preflight_pool(pool_file)
    assert [entry.name for entry in preflight.pool.models] == ["on"]

    pool_file.write_text(
        pool_file.read_text(encoding="utf-8").replace(
            'model = "gpt-5.4"', 'model = "gpt-5.4"\nenabled = false'
        ),
        encoding="utf-8",
    )
    with pytest.raises(SweepError, match="disabled"):
        preflight_pool(pool_file)


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


class _StubEnv:
    """A scoring env that reports a fixed world-model usage record after it closes."""

    def __init__(self, usd: float, *, metered: bool = True, raises: bool = False) -> None:
        self._usd = usd
        self._metered = metered
        self._raises = raises
        self.last_score = EpisodeScore(reward=0.5, success=True, critique="fine")

    def reset(self, task: str | None = None, seed_state: object | None = None) -> EnvState:
        if self._raises:
            raise RuntimeError("env exploded mid-sweep")
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        return

    @property
    def usage(self) -> RunRecord | None:
        """None models an env that keeps no metering at all (any `Env` may be handed in)."""
        if not self._metered:
            return None
        totals = UsageTotals(calls=1, input_tokens=10, output_tokens=2, cost_usd=self._usd)
        return RunRecord(run_id="s", kind="serve", total=totals, by_phase={Phase.SERVE: totals})


class _FrozenWorldModel:
    """Just the `frozen()` surface `execute_sweep` needs."""

    def __init__(self) -> None:
        self.froze = False

    @contextmanager
    def frozen(self) -> Iterator[_FrozenWorldModel]:
        self.froze = True
        yield self


def _execute(tmp_path: Path, envs: list[_StubEnv]):  # noqa: ANN202 - returns the SweepRun
    """Run `execute_sweep` over a one-candidate, N-scenario plan handing out `envs` in order."""
    plan = _plan(tmp_path, scenarios=len(envs))
    handed = iter(envs)
    world_model = _FrozenWorldModel()
    return (
        execute_sweep(
            plan,
            world_model=cast("WorldModel", world_model),
            env_factory=lambda: cast("Env", next(handed)),
            runs_dir=tmp_path / "runs",
        ),
        world_model,
    )


def test_every_episodes_world_model_usage_is_harvested_exactly_once(tmp_path: Path) -> None:
    # The harvest takes the PREVIOUS env's record when the next episode starts, plus one at the
    # end. Three episodes must therefore contribute three records, not two and not four.
    run, world_model = _execute(tmp_path, [_StubEnv(0.10), _StubEnv(0.10), _StubEnv(0.10)])
    assert world_model.froze  # the sweep ran frozen, so no cell feeds another cell's retrieval
    assert run.episodes_metered == 3 and run.episodes_unmetered == 0
    assert run.world_model_usd == pytest.approx(0.30)
    assert run.metering_gap is None
    assert run.usage_path is not None and run.usage_path.is_file()
    assert run.world_model_usage.kind == "sweep"


def test_an_env_that_keeps_no_metering_is_counted_not_guessed_at(tmp_path: Path) -> None:
    # `evaluate_pool` accepts any Env. One that exposes no usage leaves the world-model cost
    # UNKNOWN for that episode, and the run says how much of the sweep it could not see.
    run, _wm = _execute(tmp_path, [_StubEnv(0.10), _StubEnv(0.0, metered=False)])
    assert run.episodes_metered == 1 and run.episodes_unmetered == 1
    gap = run.metering_gap
    assert gap is not None and "partial: 1 of 2 episode(s)" in gap


def test_nothing_metered_writes_no_run_record_and_says_it_is_unknown(tmp_path: Path) -> None:
    run, _wm = _execute(tmp_path, [_StubEnv(0.0, metered=False)])
    assert run.episodes_metered == 0
    assert run.usage_path is None  # a $0.00 record would read as "the simulator was free"
    assert not (tmp_path / "runs").exists()
    gap = run.metering_gap
    assert gap is not None and "unknown rather than zero" in gap


def test_a_sweep_that_dies_mid_flight_still_accounts_for_what_it_already_spent(
    tmp_path: Path,
) -> None:
    """The episodes that ran before the failure were paid for on the world model's side.

    `evaluate_pool` raises unguarded on a provider it cannot build or an env that does not score.
    Letting the harvest die with the exception would make that spend unaccountable.
    """
    envs = [_StubEnv(0.10), _StubEnv(0.03, raises=True)]
    with pytest.raises(RuntimeError, match="env exploded"):
        _execute(tmp_path, envs)
    written = list((tmp_path / "runs").glob("*.json"))
    assert len(written) == 1
    salvaged = RunRecord.model_validate_json(written[0].read_text(encoding="utf-8"))
    assert salvaged.kind == "sweep"
    # Both sessions: the episode that completed AND the one that died. The failing episode's
    # session was still opened against the world model and still metered, so leaving it out
    # would under-report what the aborted sweep actually cost.
    assert salvaged.total.cost_usd == pytest.approx(0.13)


# --------------------------------------------------------- concurrency, persistence, and resume
# The three properties a six-hour grid needs from this library: cells overlap, a cell that
# completed is on disk, and a run that died buys only what is missing.


class _DoneProvider:
    """Candidate provider that finishes the episode on its first call, deterministically."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text='{"done": true, "summary": "finished"}',
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


@pytest.fixture
def candidates(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub candidate construction; returns the list every constructed model id lands in.

    `pool_provider` resolves `get_provider` from its module at call time, which is the seam the
    CLI tests already use. Counting constructions is how a resume test proves that a skipped cell
    was not merely overwritten but never ran.
    """
    built: list[str] = []

    def _get_provider(config: ProviderConfig, api_key: str | None = None) -> _DoneProvider:
        built.append(config.model)
        return _DoneProvider(config)

    monkeypatch.setattr("wmo.common.providers.pool.get_provider", _get_provider)
    return built


class _SlowEnv(_StubEnv):
    """A stub env whose episode takes real wall time, so overlap is measurable."""

    def reset(self, task: str | None = None, seed_state: object | None = None) -> EnvState:
        time.sleep(0.05)
        return super().reset(task, seed_state)


class _ExplodingEnv(_StubEnv):
    """A stub env that refuses to start: the transport fault a long grid eventually hits."""

    def reset(self, task: str | None = None, seed_state: object | None = None) -> EnvState:
        raise RuntimeError("world model session refused")


def _factory(envs: list[_StubEnv]) -> Callable[[], Env]:
    """Hand out `envs` in order, safely from several threads."""
    handed = iter(envs)
    lock = threading.Lock()

    def build() -> Env:
        with lock:
            return cast("Env", next(handed))

    return build


def _run_plan(
    plan: SweepPlan,
    envs: list[_StubEnv],
    *,
    runs_dir: Path,
    remeasure: frozenset[CellKey] = frozenset(),
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
) -> SweepRun:
    return execute_sweep(
        plan,
        world_model=cast("WorldModel", _FrozenWorldModel()),
        env_factory=_factory(envs),
        runs_dir=runs_dir,
        remeasure=remeasure,
        on_outcome=on_outcome,
    )


def _sidecar_rows(plan: SweepPlan) -> list[ScenarioOutcome]:
    """Every row in the sidecar, header skipped, in the order it was written."""
    lines = partial_path(plan.out_path).read_text(encoding="utf-8").splitlines()
    return [ScenarioOutcome.model_validate_json(line) for line in lines[1:]]


def _comparable(matrix: OutcomeMatrix) -> str:
    """The matrix as JSON with per-call WALL TIMES dropped, which no two runs can share."""
    payload = matrix.model_dump(mode="json")
    for row in payload["outcomes"]:
        row["call_seconds"] = len(row["call_seconds"])
    return json.dumps(payload, sort_keys=True)


def test_a_cell_is_on_disk_before_the_next_one_starts(
    tmp_path: Path, candidates: list[str]
) -> None:
    # The whole crash-safety claim in one assertion: by the time a cell is reported, its row is
    # already durable, so a kill one microsecond later cannot lose it.
    plan = _plan(tmp_path, scenarios=3)
    seen_lines: list[int] = []

    def check(outcome: ScenarioOutcome) -> None:
        rows = _sidecar_rows(plan)
        assert rows[-1].scenario_id == outcome.scenario_id
        seen_lines.append(len(rows))

    run = _run_plan(
        plan, [_StubEnv(0.10) for _ in range(3)], runs_dir=tmp_path / "runs", on_outcome=check
    )
    assert seen_lines == [1, 2, 3]
    assert len(run.matrix.outcomes) == 3
    # ...and the sidecar is gone once the matrix it protected exists.
    assert not partial_path(plan.out_path).exists()
    assert plan.out_path.is_file()


def test_a_crash_keeps_its_paid_cells_and_the_resume_buys_only_the_rest(
    tmp_path: Path, candidates: list[str]
) -> None:
    plan = _plan(tmp_path, scenarios=3)
    envs = [_StubEnv(0.10), _StubEnv(0.10), _ExplodingEnv(0.10)]
    with pytest.raises(RuntimeError, match="session refused"):
        _run_plan(plan, envs, runs_dir=tmp_path / "runs")
    assert len(_sidecar_rows(plan)) == 2  # the two that completed, not the one that died
    assert not plan.out_path.exists()  # no matrix: the sweep never finished

    built_before = len(candidates)
    resumed = _run_plan(_plan(tmp_path, scenarios=3), [_StubEnv(0.10)], runs_dir=tmp_path / "runs")
    assert len(candidates) - built_before == 1  # ONE cell ran, not three
    assert resumed.resumed_cells == 2
    assert len(resumed.matrix.outcomes) == 3
    assert resumed.episodes_metered == 1  # this attempt's world-model spend, not the last one's
    assert not partial_path(plan.out_path).exists()


def test_a_resumed_run_says_its_world_model_figure_is_this_attempt_only(
    tmp_path: Path, candidates: list[str]
) -> None:
    # The earlier attempt persisted its own kind="sweep" record, so adding the two here would
    # double count; saying which cells this number covers is the honest alternative.
    plan = _plan(tmp_path, scenarios=2)
    with pytest.raises(RuntimeError):
        _run_plan(plan, [_StubEnv(0.10), _ExplodingEnv(0.0)], runs_dir=tmp_path / "runs")
    resumed = _run_plan(_plan(tmp_path, scenarios=2), [_StubEnv(0.10)], runs_dir=tmp_path / "runs")
    gap = resumed.metering_gap
    assert gap is not None and "this attempt only" in gap
    assert resumed.world_model_usd == pytest.approx(0.10)


def test_a_sidecar_from_a_different_plan_is_refused_rather_than_merged(
    tmp_path: Path, candidates: list[str]
) -> None:
    # Two arms in one matrix is a fabricated comparison, so the mismatch is fatal and names the
    # field that changed. Refused BEFORE the spend question too, via `resumable_cells`.
    plan = _plan(tmp_path, scenarios=3)
    with pytest.raises(RuntimeError):
        _run_plan(plan, [_StubEnv(0.10), _ExplodingEnv(0.0)], runs_dir=tmp_path / "runs")
    changed = _plan(tmp_path, scenarios=3, episodes=2)
    with pytest.raises(SweepError, match="episodes per cell changed"):
        resumable_cells(changed)
    with pytest.raises(SweepError, match="DIFFERENT plan"):
        _run_plan(changed, [_StubEnv(0.10) for _ in range(6)], runs_dir=tmp_path / "runs")


def test_a_remeasured_cell_replaces_its_row_and_the_row_says_so(
    tmp_path: Path, candidates: list[str]
) -> None:
    plan = _plan(tmp_path, scenarios=3)
    with pytest.raises(RuntimeError, match="session refused"):
        _run_plan(
            plan, [_StubEnv(0.10), _StubEnv(0.10), _ExplodingEnv(0.10)], runs_dir=tmp_path / "runs"
        )
    retry = CellKey.of(_sidecar_rows(plan)[0])
    resumed = _run_plan(
        _plan(tmp_path, scenarios=3),
        [_StubEnv(0.10), _StubEnv(0.10)],
        runs_dir=tmp_path / "runs",
        remeasure=frozenset({retry}),
    )
    assert resumed.remeasured_cells == 1
    assert resumed.resumed_cells == 1  # the other completed cell was still reused
    rows = {CellKey.of(row).model_dump_json(): row for row in resumed.matrix.outcomes}
    assert rows[retry.model_dump_json()].remeasured is True
    assert sum(row.remeasured for row in resumed.matrix.outcomes) == 1
    assert len(resumed.matrix.outcomes) == 3  # replaced, not appended


def test_the_matrix_is_the_same_evidence_at_one_cell_at_a_time_or_four(
    tmp_path: Path, candidates: list[str]
) -> None:
    """The control for the whole change: concurrency is a speed knob, not a measurement change.

    Compared with per-call wall times dropped, because those are the one field two runs of
    anything cannot share. Everything else, including row ORDER, must match exactly, or the
    matrix digests that `fit` and `tune` identify evidence by would drift with the thread count.
    """
    sequential = _run_plan(
        _plan(tmp_path, scenarios=3, out_name="one.json"),
        [_StubEnv(0.10) for _ in range(3)],
        runs_dir=tmp_path / "runs",
    )
    concurrent = _run_plan(
        _plan(tmp_path, scenarios=3, max_concurrency=4, out_name="four.json"),
        [_StubEnv(0.10) for _ in range(3)],
        runs_dir=tmp_path / "runs",
    )
    assert _comparable(concurrent.matrix) == _comparable(sequential.matrix)


def test_concurrent_cells_overlap_and_still_meter_every_session(
    tmp_path: Path, candidates: list[str]
) -> None:
    # Four 50ms episodes: sequentially 200ms of wall clock, four at a time one episode's worth.
    # The metering assertion rides along because the harvest is what the redesign put at risk:
    # four envs are alive at once, and each one's record must land exactly once.
    plan = _plan(tmp_path, scenarios=4, max_concurrency=4)
    started = time.monotonic()
    run = _run_plan(plan, [_SlowEnv(0.10) for _ in range(4)], runs_dir=tmp_path / "runs")
    elapsed = time.monotonic() - started
    assert elapsed < 0.15, f"4 overlapping cells took {elapsed:.3f}s"
    assert run.episodes_metered == 4 and run.episodes_unmetered == 0
    assert run.world_model_usd == pytest.approx(0.40)
    assert run.metering_gap is None
    assert len(_load_records(tmp_path / "runs")) == 1


def _load_records(runs_dir: Path) -> list[RunRecord]:
    return [
        RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(runs_dir.glob("*.json"))
    ]


def test_a_plan_identity_changes_with_what_it_measures_and_not_with_how_fast(
    tmp_path: Path,
) -> None:
    base = _plan(tmp_path, scenarios=3)
    assert _plan(tmp_path, scenarios=3, max_concurrency=8).identity == base.identity
    assert _plan(tmp_path, scenarios=3, episodes=2).identity != base.identity
    assert _plan(tmp_path, scenarios=2).identity != base.identity
    assert (
        _plan(
            tmp_path,
            scenarios=3,
            compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
        ).identity
        != base.identity
    )
    assert len(base.identity.digest) == 16


def test_a_concurrency_below_one_is_refused_at_plan_time(tmp_path: Path) -> None:
    with pytest.raises(SweepError, match="at least 1"):
        _plan(tmp_path, max_concurrency=0)


def test_pool_digest_ignores_the_enabled_field() -> None:
    """The digest predates `enabled`; hashing it would orphan every pre-upgrade sidecar.

    The digested pool is already filtered to enabled entries, so the field carries no
    information there, and an unchanged roster must keep its recorded digest across the
    upgrade that added the field.
    """
    from wmo.common.providers.pool import ModelPool, PoolEntry
    from wmo.optimize.routing.sweep import _pool_digest

    entry = PoolEntry(name="on", kind=ProviderKind.OPENAI, model="gpt-5.4")
    explicit = entry.model_copy(update={"enabled": True})
    assert _pool_digest(ModelPool(models=[entry])) == _pool_digest(ModelPool(models=[explicit]))
    assert '"enabled"' not in entry.model_dump_json(exclude={"enabled"})
