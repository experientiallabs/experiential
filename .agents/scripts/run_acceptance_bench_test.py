"""Tests for the acceptance benchmark's kill switch and all-in spend metering.

`.agents/` is outside the repo gate's `testpaths`, so this file is run explicitly:

    uv run pytest -q .agents/scripts/run_acceptance_bench_test.py

It exists because the cap is the only thing standing between a typo and a four-figure bill, and
because the mechanism depends on a real property of `evaluate_pool` (it swallows `Exception` from
`on_outcome`) that a future refactor could change under us.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from wmo.core.types import Action, EnvState, Observation
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import Scenario
from wmo.optimize.reward import EpisodeScore
from wmo.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import ModelPool, PoolEntry
from wmo.tracking import Phase, RunRecord, UsageTotals

_SPEC = importlib.util.spec_from_file_location(
    "run_acceptance_bench", Path(__file__).with_name("run_acceptance_bench.py")
)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
# Registered before execution because `@dataclass` resolves annotations through
# `sys.modules[cls.__module__]`, which does not exist yet for a module loaded from a path.
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


CALLS_PER_EPISODE = 2  # the scripted provider: one tool call, then done
USAGE_PER_CALL = TokenUsage(input_tokens=10, output_tokens=5)
EPISODE_USAGE = TokenUsage(
    input_tokens=USAGE_PER_CALL.input_tokens * CALLS_PER_EPISODE,
    output_tokens=USAGE_PER_CALL.output_tokens * CALLS_PER_EPISODE,
)
JUDGE_USD = 0.20
SERVE_USD = 0.30


class _ScriptedProvider:
    """Provider whose completions are a fixed script: one tool call, then done."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config = entry.provider_config()
        self._script = [
            '{"tool": "ls", "arguments": {}}',
            '{"done": true, "summary": "finished"}',
        ]
        self._index = 0
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        text = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        self.calls += 1
        return Completion(text=text, usage=USAGE_PER_CALL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


class _MeteredFakeEnv:
    """Env shaped like `WorldModelEnv`: scores on close and exposes the session's usage record."""

    last_score = EpisodeScore(reward=1.0, success=True, critique="fine")

    def __init__(self) -> None:
        self.usage = RunRecord(
            run_id="fake",
            kind="serve",
            total=UsageTotals(cost_usd=SERVE_USD + JUDGE_USD),
            by_phase={
                Phase.SERVE: UsageTotals(cost_usd=SERVE_USD),
                Phase.JUDGE: UsageTotals(cost_usd=JUDGE_USD),
            },
        )

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        return None


class _UnmeteredFakeEnv(_MeteredFakeEnv):
    """An env that keeps no usage record: the gap must be counted, not guessed at as $0."""

    def __init__(self) -> None:
        self.usage = None


def _entry(name: str = "candidate-a") -> PoolEntry:
    return PoolEntry(
        name=name,
        kind=ProviderKind.OPENAI,
        model="custom-a",
        input_per_mtok=100_000.0,
        output_per_mtok=100_000.0,
    )


def _cells(count: int, entry: PoolEntry) -> list[tuple[Scenario, PoolEntry]]:
    return [
        (Scenario(task=f"task {i}", provenance=[f"trace-{i}"]), entry) for i in range(count)
    ]


def _run(
    cells: list[tuple[Scenario, PoolEntry]],
    spend: bench.Spend,
    rows_path: Path,
    *,
    env_factory: type[_MeteredFakeEnv] = _MeteredFakeEnv,
    providers: list[_ScriptedProvider] | None = None,
) -> list[object]:
    """Drive `run_cells` over fake cells, persisting rows the way the live grid does."""
    made = providers if providers is not None else []

    def provider_factory(entry: PoolEntry) -> Provider:
        provider = _ScriptedProvider(entry)
        made.append(provider)
        return cast("Provider", provider)

    meter = bench.EnvUsageMeter(env_factory, spend)
    with rows_path.open("a", encoding="utf-8") as handle:

        def persist(outcome: object) -> None:
            handle.write(cast("bench.ScenarioOutcome", outcome).model_dump_json() + "\n")
            handle.flush()

        return bench.run_cells(
            cells,
            compression=None,
            meter=meter,
            tools_hint="",
            spend=spend,
            on_row=persist,
            label="test",
            provider_factory=provider_factory,
        )


def _rows(rows_path: Path) -> list[dict[str, object]]:
    if not rows_path.exists():
        return []
    return [
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line
    ]


def test_cap_exceeded_cannot_be_swallowed_by_the_callback_guard() -> None:
    """`CapExceeded` must sit outside `Exception`, which is what `evaluate_pool` catches."""
    assert issubclass(bench.CapExceeded, BaseException)
    assert not issubclass(bench.CapExceeded, Exception)


def test_evaluate_pool_swallows_an_ordinary_callback_exception() -> None:
    """The hazard `CapExceeded` exists for: a plain raise in `on_outcome` does NOT stop the sweep.

    Pinning it here means a future change to `evaluate_pool`'s guard shows up as a failure in the
    runner's own tests rather than as a grid that quietly blew through its cap.
    """
    seen: list[str] = []

    def raiser(outcome: bench.ScenarioOutcome) -> None:
        seen.append(outcome.scenario_id)
        raise ValueError("a broken progress pipe")

    matrix = evaluate_pool(
        _MeteredFakeEnv,
        ModelPool(models=[_entry()]),
        [cell[0] for cell in _cells(3, _entry())],
        provider_factory=cast("object", _ScriptedProvider),
        max_steps=4,
        on_outcome=raiser,
    )

    assert len(seen) == 3
    assert len(matrix.outcomes) == 3


def test_cap_breach_stops_further_episodes_and_persists_completed_rows(tmp_path: Path) -> None:
    """A breach mid-cell stops the run after the episode in flight, and that row is on disk."""
    entry = _entry()
    per_episode = entry.cost_usd(EPISODE_USAGE) + SERVE_USD + JUDGE_USD
    rows_path = tmp_path / "rows.jsonl"
    spend = bench.Spend(cap_usd=per_episode * 0.5)  # the first episode alone breaches it

    collected = _run(_cells(4, entry), spend, rows_path)

    assert len(collected) == 1, "the cap must stop the sweep after the episode already in flight"
    assert len(_rows(rows_path)) == 1, "the completed episode must survive the halt"
    assert spend.exceeded
    assert spend.episodes == 1


def test_budget_is_checked_before_scheduling_a_cell(tmp_path: Path) -> None:
    """The reliable half of the kill switch: an exhausted budget schedules nothing at all.

    No exception is involved here, which is the point. Even if `CapExceeded` were swallowed, or
    `evaluate_pool` grew concurrency that lost it, a spent budget still cannot buy another cell.
    """
    entry = _entry()
    rows_path = tmp_path / "rows.jsonl"
    spend = bench.Spend(cap_usd=1.0, candidate_usd=2.0)
    providers: list[_ScriptedProvider] = []

    collected = _run(_cells(4, entry), spend, rows_path, providers=providers)

    assert collected == []
    assert providers == [], "no provider may be built for a cell that was never scheduled"
    assert _rows(rows_path) == []


def test_env_and_judge_cost_alone_can_trip_the_cap(tmp_path: Path) -> None:
    """The cap is all-in: the world model's serve + judge spend counts toward it.

    The candidate side is priced at zero here, so the only thing that can breach the cap is the
    evaluation infrastructure, which the old runner never metered at all.
    """
    entry = PoolEntry(
        name="free-candidate",
        kind=ProviderKind.OPENAI,
        model="custom-free",
        input_per_mtok=0.0,
        output_per_mtok=0.0,
    )
    rows_path = tmp_path / "rows.jsonl"
    spend = bench.Spend(cap_usd=(SERVE_USD + JUDGE_USD) * 0.5)

    collected = _run(_cells(3, entry), spend, rows_path)

    assert len(collected) == 1
    assert spend.candidate_usd == 0.0
    assert spend.judge_usd == pytest.approx(JUDGE_USD)
    assert spend.serve_usd == pytest.approx(SERVE_USD)
    assert spend.env_episodes_metered == 1


def test_an_env_without_a_usage_record_is_counted_not_assumed_free(tmp_path: Path) -> None:
    entry = _entry()
    rows_path = tmp_path / "rows.jsonl"
    spend = bench.Spend(cap_usd=entry.cost_usd(EPISODE_USAGE) * 0.5)

    _run(_cells(2, entry), spend, rows_path, env_factory=_UnmeteredFakeEnv)

    assert spend.env_episodes_metered == 0
    assert spend.env_episodes_unmetered == 1
    assert spend.env_judge_usd == 0.0


def test_arms_cover_the_approved_grid() -> None:
    """The five approved arms, and the learned one is the endpoint (no local model)."""
    assert bench.ARM_NAMES == (
        "off",
        "llmlingua2-endpoint",
        "truncate-protect-task",
        "control-random",
        "control-truncate",
    )
    assert bench.ARM_COMPRESSORS["control-truncate"] == "truncate"
    assert bench.ARM_COMPRESSORS["control-random"] == "random-removal"


def test_compression_for_stamps_the_running_version() -> None:
    """Every config carries the version that will RUN, mirroring `wmo optimize route`."""
    bench.register_arms()
    config = bench.compression_for("truncate-protect-task", 0.4)
    assert config.compressor_version == bench.get_compressor("truncate-protect-task").version
    assert config.aggressiveness == pytest.approx(0.4)
