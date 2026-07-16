"""Tests for the model-comparison grid (no network: fake providers via provider_factory)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.evals.failover import SameModelFailover
from wmh.evals.grid import (
    CONDITIONS,
    GridCell,
    GridResult,
    ModelSpec,
    _make_judge,
    _make_target,
    merge_results,
    run_grid,
)
from wmh.optimize.judge import RubricJudge
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class _FakeProvider:
    """Canned world-model JSON for rollouts + a fixed judge score, tagged by model id."""

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
        if "grade a world model" in system:  # the judge prompt marker
            return Completion(text='{"score": 0.8, "critique": "close enough"}')
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


def _factory(config: ProviderConfig) -> _FakeProvider:
    return _FakeProvider(config)


def _tiny_trace_file(tmp_path: Path) -> str:
    """One OTel-GenAI chat+tool span pair -> one trace with one tool-call step."""
    llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    p = tmp_path / "traces.otel.jsonl"
    p.write_text(json.dumps(llm) + "\n" + json.dumps(tool) + "\n", encoding="utf-8")
    return str(p)


def test_run_grid_produces_a_cell_per_model_and_condition(tmp_path) -> None:  # noqa: ANN001 - fixture
    traces = _tiny_trace_file(tmp_path)
    gepa = tmp_path / "gepa_opus.txt"
    gepa.write_text("EVOLVED PROMPT", encoding="utf-8")

    result = run_grid(
        suite_name="tiny",
        files=[traces],
        models=[
            ModelSpec("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8"),
            ModelSpec("Qwen", "openai", "qwen-agentworld"),
        ],
        gepa_prompts={"Opus 4.8": str(gepa)},  # only Opus has a GEPA prompt
        base_prompt="BASE PROMPT",
        judge_provider="bedrock",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_region=None,
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )

    by_model: dict[str, list[str]] = {}
    for cell in result.cells:
        by_model.setdefault(cell.model_label, []).append(cell.condition)
    # Opus has a GEPA prompt -> all 4 conditions; Qwen has none -> only base + base_rag.
    assert set(by_model["Opus 4.8"]) == set(CONDITIONS)
    assert set(by_model["Qwen"]) == {"base", "base_rag"}
    # The judge is pinned regardless of target.
    assert result.judge_model == "us.anthropic.claude-opus-4-8"


def test_base_identical_gepa_prompt_skips_gepa_cells(tmp_path) -> None:  # noqa: ANN001 - fixture
    # A GEPA prompt byte-identical to base carries no lift; its gepa/gepa_rag cells must be skipped
    # so the grid never reports same-prompt-rerun noise as a GEPA delta.
    traces = _tiny_trace_file(tmp_path)
    noop = tmp_path / "noop.txt"
    # Trailing whitespace (a common editor/export artifact) must NOT defeat the guard: compared
    # stripped, this is still the base prompt.
    noop.write_text("BASE PROMPT\n", encoding="utf-8")
    result = run_grid(
        suite_name="tiny",
        files=[traces],
        models=[ModelSpec("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8")],
        gepa_prompts={"Opus 4.8": str(noop)},
        base_prompt="BASE PROMPT",
        judge_provider="bedrock",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_region=None,
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )
    conds = {c.condition for c in result.cells}
    assert conds == {"base", "base_rag"}  # no gepa/gepa_rag - the no-op prompt was treated as none


def test_grid_cost_is_none_for_unpriced_model(tmp_path) -> None:  # noqa: ANN001 - fixture
    traces = _tiny_trace_file(tmp_path)
    result = run_grid(
        suite_name="tiny",
        files=[traces],
        models=[
            ModelSpec("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8"),
            ModelSpec("Qwen", "openai", "qwen-mystery-model-no-price"),
        ],
        gepa_prompts=None,
        base_prompt="BASE PROMPT",
        judge_provider="bedrock",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_region=None,
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )
    opus = next(c for c in result.cells if c.model_label == "Opus 4.8")
    qwen = next(c for c in result.cells if c.model_label == "Qwen")
    # Priced model -> real (maybe 0.0) target cost; unpriced -> None (omit label), not a fake 0.
    assert opus.cost_usd is not None
    assert qwen.cost_usd is None
    # Every cell yields a fidelity in [0, 1] and scores the (fallback) held-out step.
    assert 0.0 <= opus.fidelity <= 1.0
    assert opus.n_steps == 1


def test_bedrock_judge_and_target_get_same_model_chains() -> None:
    built: list[str] = []

    def tracking_factory(config: ProviderConfig) -> _FakeProvider:
        built.append(f"{config.model}@{config.region}")
        return _FakeProvider(config)

    # Bedrock judge -> SameModelFailover of EXACTLY [primary opus-4.8, direct-Anthropic opus-4.8].
    # The judge NEVER switches to a different model (no resilience models) - only the same model on
    # the unlimited direct API, so cells stay comparable.
    judge = _make_judge("bedrock", "us.anthropic.claude-opus-4-8", "us-west-1", tracking_factory)
    assert isinstance(judge, RubricJudge)  # the only judge kind; narrows for the attr check below
    assert isinstance(judge._provider, SameModelFailover)  # noqa: SLF001 - inspect wrapped provider
    assert built == [
        "us.anthropic.claude-opus-4-8@us-west-1",  # pinned Bedrock primary
        "claude-opus-4-8@None",  # SAME model, direct Anthropic API (unlimited key)
    ]

    # Bedrock target -> region fallback (SAME model), then direct-Anthropic on the SAME model.
    built.clear()
    target = _make_target(
        ModelSpec("Opus", "bedrock", "us.anthropic.claude-opus-4-8", "us-west-1"), tracking_factory
    )
    assert isinstance(target, SameModelFailover)
    assert built == [
        "us.anthropic.claude-opus-4-8@us-west-1",
        "us.anthropic.claude-opus-4-8@us-east-1",
        "claude-opus-4-8@None",
    ]

    # Non-Bedrock target -> a single provider (no fallback).
    built.clear()
    single = _make_target(ModelSpec("GPT", "openai", "gpt-5.5"), tracking_factory)
    assert not isinstance(single, SameModelFailover)

    # Bedrock target with NO explicit region: skip the region-spread rung (it could just re-hit the
    # ambient region as a no-op duplicate); chain is [ambient primary, same-model direct-Anthropic].
    built.clear()
    none_region = _make_target(
        ModelSpec("Opus", "bedrock", "us.anthropic.claude-opus-4-8"), tracking_factory
    )
    assert isinstance(none_region, SameModelFailover)
    assert built == ["us.anthropic.claude-opus-4-8@None", "claude-opus-4-8@None"]


def _cell(model: str, condition: str, fidelity: float) -> GridCell:
    return GridCell(
        model_label=model,
        provider="openai",
        model=model,
        condition=condition,
        condition_label=condition,
        fidelity=fidelity,
        error_flag_acc=1.0,
        n_steps=100,
    )


def test_capped_provider_clamps_target_max_tokens() -> None:
    from wmh.evals.grid import CappedProvider

    seen: dict[str, int] = {}

    class _Recorder:
        def __init__(self) -> None:
            self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")

        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192) -> Completion:  # noqa: ANN001
            seen["max_tokens"] = max_tokens
            return Completion(text="{}")

        def embed(self, texts) -> list:  # noqa: ANN001
            return [[0.0] for _ in texts]

        def verify(self):  # noqa: ANN202 - unused on the eval path
            raise NotImplementedError

    capped = CappedProvider(_Recorder(), 4096)
    capped.complete("s", [Message(role="user", content="u")], max_tokens=8192)
    assert seen["max_tokens"] == 4096  # clamped down
    capped.complete("s", [Message(role="user", content="u")], max_tokens=512)
    assert seen["max_tokens"] == 512  # smaller request left alone
    assert capped.config.model == "gpt-5.5"  # config passthrough


def test_merge_results_concatenates_cells_and_keeps_first_metadata() -> None:
    api = GridResult(
        suite="terminal-tasks",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_provider="bedrock",
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        total_test_steps=100,
        total_test_traces=12,
        cells=[_cell("GPT-5.5", "base", 0.6)],
    )
    qwen = GridResult(
        suite="terminal-tasks",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_provider="bedrock",
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        total_test_steps=100,
        total_test_traces=12,
        cells=[_cell("Qwen-AgentWorld", "base", 0.4)],
    )
    merged = merge_results([api, qwen])
    assert [c.model_label for c in merged.cells] == ["GPT-5.5", "Qwen-AgentWorld"]
    assert merged.suite == "terminal-tasks"  # metadata from the first result
    assert merged.total_test_steps == 100
    assert merged.total_test_traces == 12


def test_merge_results_rejects_incomparable_results() -> None:
    def _r(  # noqa: PLR0913 - one keyword per comparability field, each with a default
        *,
        suite: str = "tau-bench",
        judge_version: str = "rubric-v2",
        train_split: float = 0.7,
        val_frac: float = 0.15,
        top_k: int = 5,
        seed: int = 0,
        embed_dim: int = 512,
        model: str = "GPT-5.5",
    ) -> GridResult:
        return GridResult(
            suite=suite,
            judge_model="us.anthropic.claude-opus-4-8",
            judge_provider="bedrock",
            judge_version=judge_version,
            train_split=train_split,
            val_frac=val_frac,
            top_k=top_k,
            seed=seed,
            sample_turns="all",
            embed_dim=embed_dim,
            cells=[_cell(model, "base", 0.6)],
        )

    # Different suites must not be merged into one chart.
    with pytest.raises(ValueError, match="one suite"):
        merge_results([_r(), _r(suite="swe-bench")])
    # Different judge versions score on different scales (the whole point of stamping it).
    with pytest.raises(ValueError, match="judge_version"):
        merge_results([_r(), _r(judge_version="rubric-v1")])
    # Drifted split/retrieval flags reserve a different held-out band -> not comparable. The
    # self-hosted grid runs in a separate process where such drift is realistic.
    with pytest.raises(ValueError, match="train_split"):
        merge_results([_r(), _r(train_split=0.8)])
    with pytest.raises(ValueError, match="val_frac"):
        merge_results([_r(), _r(val_frac=0.1)])
    with pytest.raises(ValueError, match="seed"):
        merge_results([_r(), _r(seed=1)])
    with pytest.raises(ValueError, match="top_k"):
        merge_results([_r(), _r(top_k=8)])
    # RAG phi dimensionality drift puts base_rag/gepa_rag cells on a different retrieval scale.
    with pytest.raises(ValueError, match="embed_dim"):
        merge_results([_r(), _r(embed_dim=256)])
    # Identical comparability fields -> merges fine.
    merged = merge_results([_r(), _r(model="Qwen")])
    assert [c.model_label for c in merged.cells] == ["GPT-5.5", "Qwen"]


def test_run_grid_falls_back_to_2way_when_no_room_for_val_band(tmp_path) -> None:  # noqa: ANN001
    # train_split=1.0 makes the default val_frac=(1-1.0)/2=0, which has no room for a 3-way split;
    # run_grid must fall back to the plain 2-way split rather than crashing in split_traces_3way.
    traces = _tiny_trace_file(tmp_path)
    result = run_grid(
        suite_name="tiny",
        files=[traces],
        models=[ModelSpec("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8")],
        gepa_prompts=None,
        base_prompt="BASE PROMPT",
        judge_provider="bedrock",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_region=None,
        train_split=1.0,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )
    assert result.val_frac == 0.0  # clamped: recorded honestly as "no reserved val band"
    assert result.total_test_traces == 1  # 2-way fallback still scores the tiny corpus
    assert {c.condition for c in result.cells} == {"base", "base_rag"}


def test_grid_bar_label_uses_lowercase_wmh(tmp_path) -> None:  # noqa: ANN001 - fixture
    traces = _tiny_trace_file(tmp_path)
    result = run_grid(
        suite_name="tiny",
        files=[traces],
        models=[ModelSpec("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8")],
        gepa_prompts=None,
        base_prompt="BASE PROMPT",
        judge_provider="bedrock",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_region=None,
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )
    rag = next(c for c in result.cells if c.condition == "base_rag")
    assert rag.bar_label == "Opus 4.8\nwmh/rag"
