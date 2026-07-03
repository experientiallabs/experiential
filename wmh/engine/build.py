"""The build pipeline behind `wmh build`.

ingest -> normalize -> split(train/test) -> embed/index -> GEPA optimize -> write `.wmh/` artifact.
Ingestion is part of the build: there is no separate ingest step.
"""

from __future__ import annotations

import hashlib
import json

from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.core.types import Trace
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.reporting import BuildReporter, NullReporter
from wmh.ingest import VendorPull, get_adapter
from wmh.optimize import GEPAOptimizer, OptimizeResult, RubricJudge
from wmh.providers import get_provider
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _count_steps(traces: list[Trace]) -> int:
    return sum(len(trace.steps) for trace in traces)


def ingest(
    config: HarnessConfig, *, file: str | None = None, vendor: VendorPull | None = None
) -> list[Trace]:
    """Load + normalize traces from a file upload or a vendor SDK pull into `Trace` objects."""
    adapter = get_adapter(config.trace_adapter)
    if file is not None:
        return adapter.from_file(file)
    if vendor is not None:
        return adapter.from_vendor(vendor)
    raise ValueError("ingest needs either a file path or a vendor pull")


def _trace_fraction(trace: Trace) -> float:
    """Stable hash of `trace_id` mapped to [0, 1). Order-independent, reproducible across runs."""
    digest = hashlib.blake2b(trace.trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/held-out split for GEPA (held-out is never seen during evolution).

    Assignment is by a stable hash of `trace_id`, so the same corpus always splits the same way
    regardless of order and rebuilds are reproducible. `train_split` is the target train fraction.
    """
    train: list[Trace] = []
    test: list[Trace] = []
    for trace in traces:
        (train if _trace_fraction(trace) < train_split else test).append(trace)
    return train, test


def split_traces_3way(
    traces: list[Trace], train_frac: float, val_frac: float
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    """Deterministic train / validation / test split by the same stable `trace_id` hash.

    GEPA needs THREE disjoint sets, not two: `train` seeds reflection minibatches, `val` selects
    among candidate prompts, and `test` is the truly-held-out set we report on — never seen by GEPA.
    Using the val set as the reported held-out number (the old 2-way behaviour) lets GEPA select a
    candidate on the very examples we then grade it on, so any "lift" can be selection overfitting.

    Cut points are on the SAME [0,1) hash line as `split_traces`, so `train` is prefix-compatible:
    `[0, train_frac)` = train, `[train_frac, train_frac+val_frac)` = val, rest = test.
    Requires `train_frac + val_frac < 1` so test is non-empty.
    """
    valid = train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1
    if not valid:
        raise ValueError(
            f"need train_frac>0, val_frac>0, train_frac+val_frac<1; "
            f"got train_frac={train_frac}, val_frac={val_frac}"
        )
    train: list[Trace] = []
    val: list[Trace] = []
    test: list[Trace] = []
    val_cut = train_frac + val_frac
    for trace in traces:
        f = _trace_fraction(trace)
        if f < train_frac:
            train.append(trace)
        elif f < val_cut:
            val.append(trace)
        else:
            test.append(trace)
    return train, val, test


def build(
    config: HarnessConfig,
    *,
    file: str | None = None,
    vendor: VendorPull | None = None,
    root: str = ".wmh",
    serve_provider: Provider | None = None,
    embedder: Embedder | None = None,
    reporter: BuildReporter | None = None,
) -> OptimizeResult:
    """Ingest traces and run the full build, creating + persisting the artifact under `root`.

    `serve_provider` / `embedder` are injectable for testing; in production they are constructed
    from `config` (serve provider via the registry, embedder = offline HashingEmbedder sized to
    `config.embed_dim`). `reporter` receives progress events (defaults to a no-op). Returns the
    GEPA OptimizeResult (also persisted).
    """
    report = reporter or NullReporter()
    paths = ArtifactPaths(root)
    traces = ingest(config, file=file, vendor=vendor)
    if not traces:
        raise ValueError("no traces ingested; nothing to build")
    report.ingest_done(len(traces), _count_steps(traces))

    train, test = split_traces(traces, config.train_split)
    report.split_done(len(train), len(test))

    provider = serve_provider or get_provider(config.serve_provider_config())
    embed = embedder or HashingEmbedder(dim=config.embed_dim)

    # Serving index over the full corpus: at serve time we retrieve from everything we have seen.
    retriever = EmbeddingRetriever(embed)
    retriever.index(traces)
    report.index_done(_count_steps(traces))

    # GEPA evolves the env prompt under serving conditions: it retrieves demos the same way the
    # world model will, but from a SEPARATE retriever it re-indexes over train-only (so held-out
    # steps never retrieve themselves). The embedder is stateless, so it's safe to share.
    # The progress total is GEPA's REAL translated metric-call budget (reported via on_budget),
    # not the iteration count — sizing the bar with iterations made it hit 100% while thousands
    # of valset calls were still running.
    metric_total = {"calls": config.gepa_budget}

    def _on_budget(total: int) -> None:
        metric_total["calls"] = total
        report.optimize_start(total)

    def _on_rollout(done: int, score: float | None) -> None:
        report.rollout(done, metric_total["calls"], score)

    # Optimize against the SAME scorer we evaluate with (RubricJudge), so GEPA hill-climbs the
    # metric we actually report. The coarser LLMJudge here would let GEPA improve a proxy objective
    # that doesn't move the reported rubric fidelity.
    optimizer = GEPAOptimizer(
        provider,
        RubricJudge(provider),
        retriever=EmbeddingRetriever(embed),
        on_rollout=_on_rollout,
        on_budget=_on_budget,
    )
    # Every GEPA iteration re-scores the whole valset, so an uncapped held-out split multiplies
    # wall-clock and spend by its step count for no selection benefit (fidelity saturates fast —
    # see docs/trace_scaling_law.md). Candidate selection only needs a stable sample; the full
    # held-out split still backs `wmh eval`.
    gepa_val = _cap_gepa_valset(test or train)
    result = optimizer.optimize(train, gepa_val, BASE_ENV_PROMPT, config.gepa_budget)
    report.optimize_done(
        result.metrics.held_out_accuracy, len(result.frontier), result.metrics.rollouts_used
    )

    _persist(paths, config, retriever, result)
    return result


# Ceiling on GEPA's candidate-selection valset, in steps (~one full-val pass per iteration).
_GEPA_VAL_STEP_CAP = 64


def _cap_gepa_valset(traces: list[Trace], max_steps: int = _GEPA_VAL_STEP_CAP) -> list[Trace]:
    """A prefix of `traces` totalling at most `max_steps` steps (always at least one trace).

    `split_traces` already shuffles deterministically, so the prefix is an unbiased, stable
    sample of the held-out split.
    """
    capped: list[Trace] = []
    steps = 0
    for trace in traces:
        if capped and steps + len(trace.steps) > max_steps:
            break
        capped.append(trace)
        steps += len(trace.steps)
    return capped


def _persist(
    paths: ArtifactPaths,
    config: HarnessConfig,
    retriever: EmbeddingRetriever,
    result: OptimizeResult,
) -> None:
    """Write config, prompts, frontier, metrics, and the retrieval index under `.wmh/`."""
    save_config(config, paths.root)
    paths.base_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.base_prompt.write_text(BASE_ENV_PROMPT, encoding="utf-8")
    paths.optimized_prompt.write_text(result.prompt, encoding="utf-8")
    paths.frontier.write_text(json.dumps(result.frontier, indent=2), encoding="utf-8")
    paths.metrics.write_text(result.metrics.model_dump_json(indent=2), encoding="utf-8")
    retriever.save(paths.index)
