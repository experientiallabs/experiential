"""Scenario-set construction: distill a trace corpus into a representative eval scenario set.

The pipeline (Clio-style facets -> embed -> cluster -> select -> synthesize -> verify), organized
as one subpackage per stage — `mining/`, `synthesis/`, `verification/` — with `builder` on top:

    facets = FacetExtractor(provider).extract_all(traces)
    scenario_set = build_scenario_set(traces, facets, provider, embedder, config)
    verdicts = verify_scenarios(scenario_set, traces, world_model, agent, judge_provider)

Exposed via `wmh scenarios build` / `wmh scenarios verify` on the CLI.
"""

from wmh.scenarios.builder import ScenarioBuildConfig, build_scenario_set
from wmh.scenarios.mining import (
    FacetExtractor,
    Outcome,
    SelectedTrace,
    TraceCluster,
    TraceFacet,
    cluster_facets,
    hybrid_select,
    name_clusters,
    semdedup_keep,
    tool_signature,
    trace_digest,
)
from wmh.scenarios.synthesis import (
    EvalScenario,
    ScenarioSet,
    ScenarioSynthesizer,
    infer_harness,
    scenario_from_task,
)
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
    "infer_harness",
    "name_clusters",
    "scenario_from_task",
    "semdedup_keep",
    "tool_signature",
    "trace_digest",
    "verify_scenarios",
]
