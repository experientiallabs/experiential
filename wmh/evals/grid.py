"""Model-comparison grid: run one eval suite across many (model x condition) cells.

A "grid" answers a single question for a benchmark: how do different serving models fare, each under
base / +RAG / +GEPA / +GEPA+RAG, on the SAME held-out split, scored by the SAME judge? It reuses
the open-loop eval (`wmh.evals.open_loop.evaluate_files`) once per cell and rolls the per-file
report into a `GridCell`. Two invariants make cells comparable:

- The **judge is pinned** (a single Bedrock Opus 4.8 `RubricJudge`) across every cell, independent
  of the target model - a Qwen target must not be judged by Qwen - and it never switches models
  (only to the SAME model on the direct Anthropic API under Bedrock throttling; see
  `wmh.evals.failover`). Its `JUDGE_VERSION` is stamped on the result so numbers from different
  judge generations are never silently compared.
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

from wmh.evals.failover import anthropic_direct_id, same_model_chain
from wmh.evals.open_loop import EvalReport, evaluate_files
from wmh.optimize.judge import JUDGE_VERSION, Judge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Completion, Message, Provider
from wmh.retrieval import HashingEmbedder
from wmh.tracking import MeteredProvider, RunTracker
from wmh.tracking.pricing import price_for

# Regions a Bedrock TARGET fails over across (same model, so what's measured is unchanged - this
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
    # The rubric-judge version every cell was scored under. Stamped so results from different judge
    # generations are never silently compared (rubric-v1 numbers run ~0.12 higher than rubric-v2).
    judge_version: str = JUDGE_VERSION
    train_split: float
    val_frac: float = 0.0  # validation fraction reserved for GEPA; test band = 1 - train - val
    top_k: int
    seed: int
    sample_turns: str
    # Retrieval phi dimensionality (`base_rag`/`gepa_rag` cells) and any dry-run holdout cap. Both
    # change what/how cells are scored, so they are comparability fields (see `merge_results`):
    # merging RAG cells built at different embed_dim, or a capped dry-run with a full run, would put
    # incomparable fidelities in one chart.
    embed_dim: int = 0
    max_holdout_traces: int | None = None
    total_test_steps: int = 0
    total_test_traces: int = 0
    cells: list[GridCell] = Field(default_factory=list)


def merge_results(results: list[GridResult]) -> GridResult:
    """Combine several `GridResult`s (e.g. a 4-API-model grid + a separate Qwen grid) into one.

    A self-hosted model runs in its own process (its OpenAI-compatible base URL is process-global
    via `OPENAI_BASE_URL`), so its cells arrive in a separate result JSON. All results must be the
    same suite/split - they score the same held-out set - so metadata is taken from the first and
    cells are concatenated. `total_test_steps`/`total_test_traces` take the max across results
    (equal in practice; max guards against a capped dry-run being merged with a full run).
    """
    if not results:
        raise ValueError("merge_results requires at least one GridResult")
    head = results[0]
    # Every result merged into one chart must be directly comparable: same suite, same judge
    # (model + rubric version), and the SAME held-out split + retrieval config. Drift in any of
    # these means the cells were scored on different bands or on different scales, so merging them
    # would put incomparable fidelities side by side (rubric-v1 runs ~0.12 above rubric-v2; a
    # different train_split/val_frac/seed reserves a different test band). The self-hosted grid
    # runs in a SEPARATE process, so a flag typo there is a realistic way to get silent drift.
    # Fail loudly instead of taking the first result's metadata and hiding the mismatch.
    comparability_fields = (
        "suite",
        "judge_model",
        "judge_version",
        "train_split",
        "val_frac",
        "top_k",
        "seed",
        "sample_turns",
        "embed_dim",
        "max_holdout_traces",
    )
    for field in comparability_fields:
        values = {getattr(r, field) for r in results}
        if len(values) > 1:
            raise ValueError(
                f"merge_results needs one {field} (cells not comparable across values); "
                f"got {sorted(str(v) for v in values)}"
            )
    merged = GridResult(
        suite=head.suite,
        judge_model=head.judge_model,
        judge_provider=head.judge_provider,
        judge_version=head.judge_version,
        train_split=head.train_split,
        val_frac=head.val_frac,
        top_k=head.top_k,
        seed=head.seed,
        sample_turns=head.sample_turns,
        embed_dim=head.embed_dim,
        max_holdout_traces=head.max_holdout_traces,
        total_test_steps=max(r.total_test_steps for r in results),
        total_test_traces=max(r.total_test_traces for r in results),
    )
    for r in results:
        merged.cells.extend(r.cells)
    return merged


def _make_judge(
    judge_provider: str,
    judge_model: str,
    region: str | None,
    factory,  # noqa: ANN001 - a Provider builder (ProviderConfig) -> Provider, injectable for tests
) -> Judge:
    """Build the pinned `RubricJudge`. The judge NEVER switches to a different model.

    A judge that silently swapped models mid-grid would score cells on different scales and make
    fidelity numbers incomparable (see `docs/reference/failover.md`). The ONLY failover allowed is
    to the SAME model on the direct Anthropic API (unlimited key) when the primary is a throttled
    Bedrock Anthropic model - identical model, different endpoint - so what's measured is unchanged.
    The judge is never metered as target cost.
    """
    kind_enum = ProviderKind(judge_provider)
    configs = [ProviderConfig(kind=kind_enum, model=judge_model, region=region)]
    if kind_enum is ProviderKind.BEDROCK:
        direct = anthropic_direct_id(judge_model)
        if direct is not None:
            configs.append(ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct))
    return RubricJudge(same_model_chain(configs, factory))


def _make_target(spec: ModelSpec, factory) -> Provider:  # noqa: ANN001 - factory injectable for tests
    """Build a target provider. A Bedrock target fails over across `_TARGET_FALLBACK_REGIONS` (SAME
    model - spreads throttle without changing what's measured); other providers are single."""
    kind = ProviderKind(spec.provider)
    if kind is ProviderKind.BEDROCK:
        # Spread across regions only when the primary region is EXPLICIT: with region=None the
        # primary already resolves via AWS_REGION/the boto3 chain, and appending a literal
        # us-east-1 rung could just re-hit the same endpoint (a no-op duplicate) if that is what
        # the ambient region resolves to. The direct-Anthropic rung below is the real resilience.
        regions: list[str | None] = [spec.region]
        if spec.region is not None:
            regions += [r for r in _TARGET_FALLBACK_REGIONS if r != spec.region]
        configs = [ProviderConfig(kind=kind, model=spec.model, region=r) for r in regions]
        # Then fail over to the SAME model on the direct Anthropic API (unlimited key), so a target
        # throttled across all Bedrock regions still produces real predictions on the identical
        # model instead of scoring the step 0 - critical for Opus 4.8 under Bedrock load.
        direct = anthropic_direct_id(spec.model)
        if direct is not None:
            configs.append(ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct))
        return same_model_chain(configs, factory)
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
    train_split: float,
    top_k: int,
    seed: int,
    sample_turns: str,
    embed_dim: int,
    val_frac: float | None = None,
    max_holdout_traces: int | None = None,
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
    provider_factory=get_provider,  # noqa: ANN001 - injectable for tests (no network)
) -> GridResult:
    """Run every (model x condition) cell of the grid and return the rolled-up result.

    `gepa_prompts` maps a `ModelSpec.label` to an evolved-prompt file path; a model absent from it
    skips the `gepa`/`gepa_rag` conditions. The judge is built once (a pinned `RubricJudge`) and
    shared across all cells.

    Cells are scored on the reserved **test** band of the SAME 3-way `train/val/test` split GEPA
    used to evolve its prompts (`val_frac` defaults to `(1 - train_split) / 2`, matching
    `wmh build`), so a `+GEPA` cell is never scored on the `val` traces its prompt was selected on.
    """
    from pathlib import Path

    from wmh.engine.build import split_traces, split_traces_3way
    from wmh.ingest import get_adapter

    if val_frac is None:
        val_frac = (1.0 - train_split) / 2
    # A 3-way split needs a strictly positive val band that still leaves a non-empty test band. When
    # `train_split` leaves no room (e.g. train_split >= 1.0, or a --val-frac that overflows the
    # [0,1) line) fall back to the plain 2-way split instead of crashing in `split_traces_3way`.
    use_3way = val_frac > 0 and train_split + val_frac < 1
    if not use_3way:
        val_frac = 0.0
    judge = _make_judge(judge_provider, judge_model, judge_region, provider_factory)
    result = GridResult(
        suite=suite_name,
        judge_model=judge_model,
        judge_provider=judge_provider,
        train_split=train_split,
        val_frac=val_frac,
        top_k=top_k,
        seed=seed,
        sample_turns=sample_turns,
        embed_dim=embed_dim,
        max_holdout_traces=max_holdout_traces,
    )
    paths = [Path(f) for f in files]

    # Held-out trace count (for reporting) - the reserved TEST band each cell scores, after any cap.
    # Uses the same 3-way split as the scorer so the count matches what is actually evaluated.
    adapter = get_adapter("otel-genai")
    for path in paths:
        traces = adapter.from_file(str(path))
        if use_3way:
            _, _, holdout = split_traces_3way(traces, train_split, val_frac)
        else:
            _, holdout = split_traces(traces, train_split)
        holdout = holdout or traces
        if max_holdout_traces is not None:
            holdout = holdout[:max_holdout_traces]
        result.total_test_traces += len(holdout)
    for spec in models:
        gepa_prompt_file = (gepa_prompts or {}).get(spec.label)
        gepa_prompt: str | None = None
        if gepa_prompt_file is not None:
            text = Path(gepa_prompt_file).read_text(encoding="utf-8")
            # A GEPA run that finds no improvement returns the base prompt verbatim. Scoring its
            # `gepa`/`gepa_rag` cells would just re-run `base`/`base_rag` and report the judge/
            # sampling noise between the two runs as a spurious "GEPA lift". Treat a base-identical
            # evolved prompt as NO evolved prompt so the grid never presents that noise as a delta.
            # Compare stripped so a prompt that differs only by trailing whitespace (a common
            # artifact of hand-edited/exported prompt files) is still caught as a no-op.
            gepa_prompt = text if text.strip() != base_prompt.strip() else None
        for condition in CONDITIONS:
            uses_gepa = condition in ("gepa", "gepa_rag")
            if uses_gepa and gepa_prompt is None:
                continue  # no (real) evolved prompt for this model -> skip its GEPA cells
            prompt = gepa_prompt if uses_gepa else base_prompt
            assert prompt is not None  # noqa: S101 - narrowed by the guard above
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
                    val_frac=val_frac,
                    top_k=top_k,
                    sample_turns=sample_turns,
                    seed=seed,
                    max_holdout_traces=max_holdout_traces,
                )
            fidelity, err, steps = _aggregate(report)
            # Every cell scores the same held-out band with the same sampling, so `steps` is
            # invariant across cells; take the max defensively so the reported count can never
            # under-report if a future per-model option makes one cell score fewer steps.
            result.total_test_steps = max(result.total_test_steps, steps)
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
