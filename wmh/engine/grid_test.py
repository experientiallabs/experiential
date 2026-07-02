"""Tests for the model-comparison grid (no network: fake providers via provider_factory)."""

from __future__ import annotations

import json
from pathlib import Path

from wmh.engine.grid import (
    CONDITIONS,
    ModelSpec,
    _make_judge,
    _make_target,
    run_grid,
)
from wmh.providers.base import Completion, Message, ProviderConfig
from wmh.providers.fallback import FallbackProvider


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
        judge_kind="rubric",
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
        judge_kind="rubric",
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


def test_bedrock_judge_and_target_get_fallback_chains() -> None:
    built: list[str] = []

    def tracking_factory(config: ProviderConfig) -> _FakeProvider:
        built.append(f"{config.model}@{config.region}")
        return _FakeProvider(config)

    # Bedrock judge -> a FallbackProvider (primary opus-4.8 + resilience models).
    judge = _make_judge(
        "bedrock", "us.anthropic.claude-opus-4-8", "us-west-1", "rubric", tracking_factory
    )
    assert isinstance(judge._provider, FallbackProvider)  # noqa: SLF001 - inspect wrapped provider
    assert any("sonnet" in b for b in built)  # fell through to the resilience model config

    # Bedrock target -> region-fallback chain (SAME model across regions).
    built.clear()
    target = _make_target(
        ModelSpec("Opus", "bedrock", "us.anthropic.claude-opus-4-8", "us-west-1"), tracking_factory
    )
    assert isinstance(target, FallbackProvider)
    assert built == [
        "us.anthropic.claude-opus-4-8@us-west-1",
        "us.anthropic.claude-opus-4-8@us-east-1",
    ]

    # Non-Bedrock target -> a single provider (no fallback).
    built.clear()
    single = _make_target(ModelSpec("GPT", "openai", "gpt-5.5"), tracking_factory)
    assert not isinstance(single, FallbackProvider)


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
        judge_kind="rubric",
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        embed_dim=2,
        provider_factory=_factory,
    )
    rag = next(c for c in result.cells if c.condition == "base_rag")
    assert rag.bar_label == "Opus 4.8\nwmh/rag"
