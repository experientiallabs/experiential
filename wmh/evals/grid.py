"""Model-comparison grid: run one eval suite across many (model x condition) cells.

A "grid" answers a single question for a benchmark: how do different serving models fare, each under
base / +RAG / +GEPA / +GEPA+RAG, on the SAME held-out split, scored by the SAME judge? It reuses
the open-loop eval (`wmh.evals.open_loop.evaluate_files`) once per cell and rolls the per-file
report into a `GridCell`. Two invariants make cells comparable:

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

from wmh.evals.open_loop import EvalReport, evaluate_files
from wmh.optimize.judge import Judge, LLMJudge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Completion, Message, Provider
from wmh.providers.fallback import FallbackProvider, anthropic_direct_id
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

# Cap on the TARGET's generation per step. A world-model observation is short JSON; a reasoning
# target (GPT-5.5) otherwise spends the full 8192-token budget on reasoning, making each step
# ~80s and a whole grid many hours. 4096 leaves ample room for reasoning + the observation while
# roughly halving worst-case latency. The judge is never capped (it needs its full rubric budget).
DEFAULT_TARGET_MAX_TOKENS = 4096


class CappedProvider:
    """Wraps a target provider, clamping each completion's `max_tokens` to a ceiling.

    Only the eval TARGET is wrapped (not the judge): observation prediction needs a short output, so
    a lower ceiling bounds a reasoning model's per-step latency without affecting judge scoring.
    """

    def __init__(self, inner: Provider, cap: int) -> None:
        self._inner = inner
        self._cap = cap
        self.config = inner.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return self._inner.complete(
            system, messages, temperature=temperature, max_tokens=min(max_tokens, self._cap)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def verify(self):  # noqa: ANN201 - delegate; unused on the eval path
        return self._inner.verify()


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


def merge_results(results: list[GridResult]) -> GridResult:
    """Combine several `GridResult`s (e.g. a 4-API-model grid + a separate Qwen grid) into one.

    A self-hosted model runs in its own process (its OpenAI-compatible base URL is process-global
    via `OPENAI_BASE_URL`), so its cells arrive in a separate result JSON. All results must be the
    same suite/split — they score the same held-out set — so metadata is taken from the first and
    cells are concatenated. `total_test_steps`/`total_test_traces` take the max across results
    (equal in practice; max guards against a capped dry-run being merged with a full run).
    """
    if not results:
        raise ValueError("merge_results requires at least one GridResult")
    head = results[0]
    merged = GridResult(
        suite=head.suite,
        judge_model=head.judge_model,
        judge_provider=head.judge_provider,
        train_split=head.train_split,
        top_k=head.top_k,
        seed=head.seed,
        sample_turns=head.sample_turns,
        total_test_steps=max(r.total_test_steps for r in results),
        total_test_traces=max(r.total_test_traces for r in results),
    )
    for r in results:
        merged.cells.extend(r.cells)
    return merged


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
    configs = [ProviderConfig(kind=kind_enum, model=judge_model, region=region)]
    if kind_enum is ProviderKind.BEDROCK:
        # Fail over first to the SAME judge model on the direct Anthropic API (unlimited key), so a
        # throttled Bedrock Opus judge stays Opus rather than dropping to a different Bedrock model;
        # only then to the Bedrock resilience models.
        direct = anthropic_direct_id(judge_model)
        if direct is not None:
            configs.append(ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct))
        configs += [
            ProviderConfig(kind=kind_enum, model=m, region=region)
            for m in _JUDGE_FALLBACK_MODELS
            if m != judge_model
        ]
    llm = _fallback(factory, configs)
    return RubricJudge(llm) if kind == "rubric" else LLMJudge(llm)


def _make_target(spec: ModelSpec, factory) -> Provider:  # noqa: ANN001 - factory injectable for tests
    """Build a target provider. A Bedrock target fails over across `_TARGET_FALLBACK_REGIONS` (SAME
    model — spreads throttle without changing what's measured); other providers are single."""
    kind = ProviderKind(spec.provider)
    if kind is ProviderKind.BEDROCK:
        regions = [spec.region] + [r for r in _TARGET_FALLBACK_REGIONS if r != spec.region]
        configs = [ProviderConfig(kind=kind, model=spec.model, region=r) for r in regions]
        # Then fail over to the SAME model on the direct Anthropic API (unlimited key), so a target
        # throttled across all Bedrock regions still produces real predictions on the identical
        # model instead of scoring the step 0 — critical for Opus 4.8 under Bedrock load.
        direct = anthropic_direct_id(spec.model)
        if direct is not None:
            configs.append(ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct))
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
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
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
            capped = CappedProvider(_make_target(spec, provider_factory), target_max_tokens)
            target: Provider = MeteredProvider(capped, tracker)
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
