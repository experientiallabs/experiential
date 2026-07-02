"""Model-comparison grid: run one eval suite across many (model x condition) cells.

A "grid" answers a single question for a benchmark: how do different serving models fare, each under
base / +RAG / +GEPA / +GEPA+RAG, on the SAME held-out split, scored by the SAME judge? It reuses the
open-loop eval (`wmh.engine.eval.evaluate_files`) once per cell and rolls the per-file report into a
`GridCell`. Two invariants make cells comparable:

- The **judge is pinned** (a single Bedrock Opus 4.8 rubric judge) across every cell, independent of
  the target model — a Qwen target must not be judged by Qwen.
- Target token **cost is metered separately** from the judge (a `MeteredProvider` wraps only the
  target), so a cell reports target-side cost, not judge cost. Cost is `None` when the model has no
  pricing row (see `wmh.tracking.pricing.price_for`) rather than a misleading 0.

`run_grid` is provider-agnostic: each `ModelSpec` names a provider/model the registry can build, so
a self-hosted OpenAI-compatible model (e.g. Qwen-AgentWorld on vLLM) is just `provider="openai"`
with `OPENAI_BASE_URL` in the environment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

from wmh.engine.eval import EvalReport, evaluate_files
from wmh.optimize.judge import Judge, LLMJudge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider
from wmh.providers.fallback import FallbackProvider
from wmh.retrieval import HashingEmbedder
from wmh.tracking import MeteredProvider, RunTracker
from wmh.tracking.pricing import price_for

# Judge fallback chain: the pinned judge is Opus 4.8, but on a capacity error it fails over to these
# so long grids don't stall on judge throttling. (Model waterfall, not accounts — endflow/stackwise
# lack Bedrock access; the `default` profile reaches all of these.)
_JUDGE_FALLBACK_MODELS = (
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-6-v1",
)

# Regions a Bedrock TARGET fails over across (same model, so what's measured is unchanged — this
# only spreads throttling load, it does not switch models).
_TARGET_FALLBACK_REGIONS = ("us-east-1",)

# The four prompt/retrieval conditions each model is evaluated under. `gepa`/`gepa_rag` require a
# per-(benchmark x model) evolved prompt; they are skipped for a model with no GEPA prompt.
CONDITIONS = ("base", "base_rag", "gepa", "gepa_rag")

# Display labels (lowercase "wmh" per the chart convention).
_CONDITION_LABELS = {
    "base": "base",
    "base_rag": "wmh/rag",
    "gepa": "wmh/gepa",
    "gepa_rag": "wmh/gepa/rag",
}


@dataclass(frozen=True)
class ModelSpec:
    """A serving model to benchmark: display label + how the provider registry builds it."""

    label: str  # e.g. "Opus 4.8"
    provider: str  # ProviderKind value: "bedrock" | "openai" | ...
    model: str  # provider model id, e.g. "us.anthropic.claude-opus-4-8"
    region: str | None = None


class GridCell(BaseModel):
    """One (model x condition) result on a benchmark's held-out split."""

    model_label: str
    provider: str
    model: str
    condition: str  # one of CONDITIONS
    condition_label: str  # human label incl. lowercase "wmh"
    fidelity: float  # step-weighted mean judge score across the split
    error_flag_acc: float  # step-weighted fraction where predicted is_error matched actual
    n_steps: int
    cost_usd: float | None = None  # target-side USD; None when the model has no pricing row

    @property
    def bar_label(self) -> str:
        """Two-line x-axis label: "Opus 4.8\\nwmh/rag" (lowercase wmh)."""
        return f"{self.model_label}\n{self.condition_label}"


class GridResult(BaseModel):
    """A full grid: every cell plus the split/judge metadata that makes the numbers reproducible."""

    suite: str
    judge_model: str
    judge_provider: str
    train_split: float
    top_k: int
    seed: int
    sample_turns: str
    total_test_steps: int = 0
    total_test_traces: int = 0
    cells: list[GridCell] = Field(default_factory=list)


def _fallback(factory, configs: list[ProviderConfig]) -> Provider:  # noqa: ANN001 - factory injectable
    """A FallbackProvider over `configs` (a single-element chain returns a plain provider)."""
    chain = [factory(c) for c in configs]
    return chain[0] if len(chain) == 1 else FallbackProvider(chain)


def _make_judge(
    judge_provider: str,
    judge_model: str,
    region: str | None,
    kind: str,
    factory,  # noqa: ANN001 - a Provider builder (ProviderConfig) -> Provider, injectable for tests
) -> Judge:
    """Build the pinned judge as a FALLBACK chain (primary `judge_model`, then resilience models).

    The judge is never metered as target cost. For a Bedrock judge it fails over across
    `_JUDGE_FALLBACK_MODELS` on a capacity error so a throttled Opus doesn't stall the whole grid.
    """
    kind_enum = ProviderKind(judge_provider)
    models = [judge_model]
    if kind_enum is ProviderKind.BEDROCK:
        models += [m for m in _JUDGE_FALLBACK_MODELS if m != judge_model]
    configs = [ProviderConfig(kind=kind_enum, model=m, region=region) for m in models]
    llm = _fallback(factory, configs)
    return RubricJudge(llm) if kind == "rubric" else LLMJudge(llm)


def _make_target(spec: ModelSpec, factory) -> Provider:  # noqa: ANN001 - factory injectable for tests
    """Build a target provider. A Bedrock target fails over across `_TARGET_FALLBACK_REGIONS` (SAME
    model — spreads throttle without changing what's measured); other providers are single."""
    kind = ProviderKind(spec.provider)
    if kind is ProviderKind.BEDROCK:
        regions = [spec.region] + [r for r in _TARGET_FALLBACK_REGIONS if r != spec.region]
        configs = [ProviderConfig(kind=kind, model=spec.model, region=r) for r in regions]
        return _fallback(factory, configs)
    return factory(ProviderConfig(kind=kind, model=spec.model, region=spec.region))


def _target_cost(model: str, tracker: RunTracker) -> float | None:
    """Target-side USD from the metered tracker, or None when the model has no pricing row."""
    if price_for(model) is None:
        return None
    return tracker.totals().cost_usd


def _aggregate(report: EvalReport) -> tuple[float, float, int]:
    """(fidelity, step-weighted error-flag accuracy, total steps) from an EvalReport."""
    total = report.total_steps
    if total == 0:
        return 0.0, 0.0, 0
    err = sum(r.error_flag_accuracy * r.n_steps for r in report.per_file.values()) / total
    return report.overall_fidelity, err, total


def run_grid(
    *,
    suite_name: str,
    files: list[str],
    models: list[ModelSpec],
    gepa_prompts: dict[str, str] | None,
    base_prompt: str,
    judge_provider: str,
    judge_model: str,
    judge_region: str | None,
    judge_kind: str,
    train_split: float,
    top_k: int,
    seed: int,
    sample_turns: str,
    embed_dim: int,
    max_holdout_traces: int | None = None,
    provider_factory=get_provider,  # noqa: ANN001 - injectable for tests (no network)
) -> GridResult:
    """Run every (model x condition) cell of the grid and return the rolled-up result.

    `gepa_prompts` maps a `ModelSpec.label` to an evolved-prompt file path; a model absent from it
    skips the `gepa`/`gepa_rag` conditions. The judge is built once and shared across all cells.
    """
    from pathlib import Path

    from wmh.engine.build import split_traces
    from wmh.ingest import get_adapter

    judge = _make_judge(judge_provider, judge_model, judge_region, judge_kind, provider_factory)
    result = GridResult(
        suite=suite_name,
        judge_model=judge_model,
        judge_provider=judge_provider,
        train_split=train_split,
        top_k=top_k,
        seed=seed,
        sample_turns=sample_turns,
    )
    paths = [Path(f) for f in files]

    # Held-out trace count (for reporting) — the same split each cell scores, after any cap.
    adapter = get_adapter("otel-genai")
    for path in paths:
        traces = adapter.from_file(str(path))
        _, holdout = split_traces(traces, train_split)
        holdout = holdout or traces
        if max_holdout_traces is not None:
            holdout = holdout[:max_holdout_traces]
        result.total_test_traces += len(holdout)
    for spec in models:
        gepa_prompt_file = (gepa_prompts or {}).get(spec.label)
        for condition in CONDITIONS:
            uses_gepa = condition in ("gepa", "gepa_rag")
            if uses_gepa and gepa_prompt_file is None:
                continue  # no evolved prompt for this model -> skip its GEPA cells
            if uses_gepa:
                assert gepa_prompt_file is not None  # noqa: S101 - narrowed by the guard above
                prompt = Path(gepa_prompt_file).read_text(encoding="utf-8")
            else:
                prompt = base_prompt
            use_rag = condition in ("base_rag", "gepa_rag")
            # Meter ONLY the target so cost is target-side, never judge cost. The target itself may
            # be a region-fallback chain (Bedrock); MeteredProvider records whichever entry served.
            tracker = RunTracker(run_id=uuid.uuid4().hex, kind="eval-grid")
            target: Provider = MeteredProvider(_make_target(spec, provider_factory), tracker)
            embedder = HashingEmbedder(dim=embed_dim) if use_rag else None
            with tracker.timed():
                report = evaluate_files(
                    paths,
                    prompt,
                    target,
                    judge,
                    embedder=embedder,
                    train_split=train_split,
                    top_k=top_k,
                    sample_turns=sample_turns,
                    seed=seed,
                    max_holdout_traces=max_holdout_traces,
                )
            fidelity, err, steps = _aggregate(report)
            result.total_test_steps = steps
            result.cells.append(
                GridCell(
                    model_label=spec.label,
                    provider=spec.provider,
                    model=spec.model,
                    condition=condition,
                    condition_label=_CONDITION_LABELS[condition],
                    fidelity=fidelity,
                    error_flag_acc=err,
                    n_steps=steps,
                    cost_usd=_target_cost(spec.model, tracker),
                )
            )
    return result
