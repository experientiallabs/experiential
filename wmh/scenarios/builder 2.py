"""The scenario-set build pipeline behind `wmh scenarios build`.

facets -> embed -> cluster -> name -> select -> synthesize -> coverage. One entry point,
`build_scenario_set`, that takes already-extracted facets so callers (research runs, tests) can
cache or substitute them; `wmh scenarios build` extracts them fresh.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from wmh.core.types import Trace
from wmh.providers.base import Embedder, Provider
from wmh.scenarios.clustering import cluster_facets, name_clusters, normalize_rows
from wmh.scenarios.facets import TraceFacet
from wmh.scenarios.selection import (
    DEDUP_THRESHOLD,
    PROPORTIONAL_FRACTION,
    hybrid_select,
)
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet, ScenarioSynthesizer


class ScenarioBuildConfig(BaseModel):
    """Knobs for one scenario-set build."""

    budget: int = 20  # scenarios to construct
    k: int | None = None  # cluster count; default sqrt(n)
    seed: int = 0
    dedup_threshold: float = DEDUP_THRESHOLD
    proportional_fraction: float = PROPORTIONAL_FRACTION
    coverage_tau: float = 0.7  # facet counts as covered when cosine-within-tau of a selection


def build_scenario_set(
    traces: list[Trace],
    facets: list[TraceFacet],
    provider: Provider,
    embedder: Embedder,
    config: ScenarioBuildConfig,
) -> ScenarioSet:
    """Construct a representative scenario set from a facet-annotated trace corpus.

    `provider` drives cluster naming and scenario synthesis; `embedder` embeds facet summaries.
    Raises when traces/facets are empty or misaligned.
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
    scenarios: list[EvalScenario] = []
    for selection in selections:
        scenario = synthesizer.synthesize(
            traces_by_id[selection.trace_id], facets_by_id[selection.trace_id]
        )
        scenario.cluster_name = cluster_names.get(selection.cluster_id, "")
        scenario.weight = selection.weight
        if selection.pinned_failure is not None:
            scenario.failure_category = selection.pinned_failure
        scenarios.append(scenario)

    selected_ids = {selection.trace_id for selection in selections}
    coverage = _corpus_coverage(facets, embeddings, selected_ids, tau=config.coverage_tau)
    return ScenarioSet(
        scenarios=scenarios,
        clusters=clusters,
        corpus_traces=len(traces),
        corpus_coverage=coverage,
        coverage_tau=config.coverage_tau,
    )


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
