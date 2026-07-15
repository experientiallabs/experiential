"""The scenario-set build pipeline behind `wmh scenarios build`.

facets -> embed -> cluster -> name -> select -> synthesize -> coverage. One entry point,
`build_scenario_set`, that takes already-extracted facets so callers (research runs, tests) can
cache or substitute them; `wmh scenarios build` extracts them fresh.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from wmh.core.types import Trace
from wmh.providers.base import Embedder, Provider
from wmh.scenarios.mining.clustering import cluster_facets, name_clusters, normalize_rows
from wmh.scenarios.mining.facets import TraceFacet
from wmh.scenarios.mining.selection import (
    DEDUP_THRESHOLD,
    PROPORTIONAL_FRACTION,
    SelectedTrace,
    hybrid_select,
)
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet, ScenarioSynthesizer
from wmh.scenarios.verification import ChecklistJudge


class ScenarioBuildConfig(BaseModel):
    """Knobs for one scenario-set build."""

    budget: int = 20  # scenarios to construct
    k: int | None = None  # cluster count; default sqrt(n)
    seed: int = 0
    validate_checklists: bool = True  # back-agreement gate inside the build (drop on repeat fail)
    dedup_threshold: float = DEDUP_THRESHOLD
    proportional_fraction: float = PROPORTIONAL_FRACTION
    coverage_tau: float = 0.7  # facet counts as covered when cosine-within-tau of a selection
    # LLM calls in the build (facet extraction, synthesis + back-agreement) are independent
    # per trace/selection: run them on a small thread pool, order-preserving. 1 = sequential.
    concurrency: int = 8


def build_scenario_set(
    traces: list[Trace],
    facets: list[TraceFacet],
    provider: Provider,
    embedder: Embedder,
    config: ScenarioBuildConfig,
    *,
    judge_provider: Provider | None = None,
) -> ScenarioSet:
    """Construct a representative scenario set from a facet-annotated trace corpus.

    `provider` drives cluster naming and scenario synthesis; `embedder` embeds facet summaries.
    `judge_provider` backs the inline checklist validation (defaults to `provider`) — pass a
    different model to keep synthesis and validation families separate. Raises when traces/facets
    are empty or misaligned.
    """
    if not traces or not facets:
        raise ValueError("need a non-empty trace corpus and facets to build a scenario set")
    if len(traces) != len(facets):
        raise ValueError(f"{len(traces)} traces but {len(facets)} facets")

    embeddings = np.asarray(embedder.embed([facet.embed_text() for facet in facets]))
    labels, clusters = cluster_facets(facets, embeddings, k=config.k, seed=config.seed)
    name_clusters(provider, clusters, facets)

    selections = hybrid_select(
        facets,
        embeddings,
        labels,
        config.budget,
        proportional_fraction=config.proportional_fraction,
        dedup_threshold=config.dedup_threshold,
    )

    traces_by_id = {trace.trace_id: trace for trace in traces}
    facets_by_id = {facet.trace_id: facet for facet in facets}
    cluster_names = {cluster.cluster_id: cluster.name for cluster in clusters}
    synthesizer = ScenarioSynthesizer(provider)
    judge = ChecklistJudge(judge_provider or provider) if config.validate_checklists else None

    def _synthesize_one(selection: SelectedTrace) -> EvalScenario | None:
        source = traces_by_id[selection.trace_id]
        scenario = synthesizer.synthesize(source, facets_by_id[selection.trace_id])
        if judge is not None:
            # A generated checklist must correctly grade the very episode it was distilled
            # from; one that misgrades its own source can't be trusted on new trajectories.
            # One regeneration retry, then drop — an invalid scenario never leaves the build.
            if not _checklist_agrees(judge, scenario, source):
                scenario = synthesizer.synthesize(source, facets_by_id[selection.trace_id])
                if not _checklist_agrees(judge, scenario, source):
                    return None
        scenario.cluster_name = cluster_names.get(selection.cluster_id, "")
        scenario.weight = selection.weight
        if selection.pinned_failure is not None:
            scenario.failure_category = selection.pinned_failure
        return scenario

    # Selections are independent (synthesis + back-agreement are per-trace LLM round trips), so
    # they run on a small thread pool; `pool.map` preserves selection order, keeping the built
    # set (and its weight renormalization) deterministic. concurrency=1 is the sequential loop.
    if config.concurrency > 1 and len(selections) > 1:
        with ThreadPoolExecutor(max_workers=min(config.concurrency, len(selections))) as pool:
            maybe_scenarios = list(pool.map(_synthesize_one, selections))
    else:
        maybe_scenarios = [_synthesize_one(selection) for selection in selections]
    scenarios = [scenario for scenario in maybe_scenarios if scenario is not None]
    dropped = sum(1 for scenario in maybe_scenarios if scenario is None)

    selected_ids = {scenario.provenance[0] for scenario in scenarios}
    coverage = _corpus_coverage(facets, embeddings, selected_ids, tau=config.coverage_tau)
    total_weight = sum(scenario.weight for scenario in scenarios)
    if dropped and total_weight > 0:  # dropped scenarios must not leave weights summing < 1
        for scenario in scenarios:
            scenario.weight /= total_weight
    return ScenarioSet(
        scenarios=scenarios,
        clusters=clusters,
        corpus_traces=len(traces),
        corpus_coverage=coverage,
        coverage_tau=config.coverage_tau,
    )


def _checklist_agrees(judge: ChecklistJudge, scenario: EvalScenario, source: Trace) -> bool:
    """Back-agreement: the judge's verdict on the SOURCE trajectory must match its recorded
    outcome. Traces without a recorded outcome can't disagree, so they pass."""
    if not scenario.checklist:
        return False
    reward = source.metadata.get("reward")
    if not isinstance(reward, int | float):
        return True
    verdict = judge.score(scenario.task, scenario.checklist, source.steps)
    return verdict.success == (float(reward) >= 1.0)


def _corpus_coverage(
    facets: list[TraceFacet],
    embeddings: np.ndarray,
    selected_ids: set[str],
    *,
    tau: float,
) -> float:
    """Fraction of corpus facets within cosine `tau` of at least one selected facet."""
    selected_rows = [i for i, facet in enumerate(facets) if facet.trace_id in selected_ids]
    if not selected_rows:
        return 0.0
    matrix = normalize_rows(embeddings)
    similarities = matrix @ matrix[np.asarray(selected_rows)].T
    return float((similarities.max(axis=1) >= tau).mean())
