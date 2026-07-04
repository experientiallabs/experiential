"""Scenario-set construction: distill a trace corpus into a representative eval scenario set.

The pipeline (Clio-style facets -> embed -> cluster -> select -> synthesize -> verify):

    facets = FacetExtractor(provider).extract_all(traces)
    scenario_set = build_scenario_set(traces, facets, provider, embedder, config)
    verdicts = verify_scenarios(scenario_set, traces, world_model, agent, judge_provider)

Exposed via `wmh scenarios build` / `wmh scenarios verify` on the CLI.
"""

from wmh.scenarios.builder import ScenarioBuildConfig, build_scenario_set
from wmh.scenarios.clustering import TraceCluster, cluster_facets, name_clusters
from wmh.scenarios.facets import FacetExtractor, Outcome, TraceFacet, tool_signature, trace_digest
from wmh.scenarios.selection import SelectedTrace, hybrid_select, semdedup_keep
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet, ScenarioSynthesizer
from wmh.scenarios.verification import (
    ChecklistJudge,
    ChecklistResult,
    ScenarioVerdict,
    VerificationReport,
    verify_scenarios,
)

__all__ = [
    "ChecklistJudge",
    "ChecklistResult",
    "EvalScenario",
    "FacetExtractor",
    "Outcome",
    "ScenarioBuildConfig",
    "ScenarioSet",
    "ScenarioSynthesizer",
    "ScenarioVerdict",
    "SelectedTrace",
    "TraceCluster",
    "TraceFacet",
    "VerificationReport",
    "build_scenario_set",
    "cluster_facets",
    "hybrid_select",
    "name_clusters",
    "semdedup_keep",
    "tool_signature",
    "trace_digest",
    "verify_scenarios",
]
