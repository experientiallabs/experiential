"""The whole decision-variance pipeline offline, on the grounded ticket world.

Every stage runs through its owner on the grounded ticket world of
:mod:`~exp.optimize.evaluation.variance.tests.grounded_world_test`; only the wire is scripted.
The variance build is replaced by a fixture executor that saves the exposure plan, a
world-learning trace dataset and a scoped, fit-only retrieval index over it (the grounded world
is not an ``exp build`` project). The module run also replaces the rules owner's extraction with
the fixture's saved ticket rule book; :func:`test_agentic_roles_authored_rubric_and_learned_rules`
instead learns the rule book from the traces through the real extraction owner and executes its
membership boundary in the bound world. Everything else (binding the rule book into the mining
runtime, TraceML labels, error mining with world-learning search, the coalition, the packet with
observed replay episodes, the agentic world and rubric authors, the freeze, the judges, the
families, admission, records, executable qualification, Harbor export and cross-talk) is real.

The scripted judge answers from an explicit synthetic policy: an episode passes exactly when its
last recorded ticket outcome is ``right``, except that the first replicate of every group fails,
so group advantages carry signal. This proves plumbing, not model quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    SourceIdentity,
    sha256_json,
)
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import Trace, TraceSource
from exp.common.traces.ingest import TraceNormalizationResult, persist_trace_dataset
from exp.common.traces.trace import TraceOutcome
from exp.optimize.evaluation.executable.controls import DeclaredControl
from exp.optimize.evaluation.executable.qualification import load_executable_qualification
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.optimize.evaluation.rules.extraction import EXTRACTION_TOOL
from exp.optimize.evaluation.rules.research_test import ROLE_MARKERS, AgentScript, default_reads
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RulesStageReport,
    RulesStageSpec,
    load_rulebook,
    run_rules_stage,
)
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.authoring_packet import (
    WHOLE_TASK,
    AuthoringPacket,
    SourceRef,
    build_authoring_packet,
    load_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.authoring_packet_test import save_book, tool_reply
from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL, SCENARIO_TOOL
from exp.optimize.evaluation.variance.behavior_labeling import LABEL_TOOL
from exp.optimize.evaluation.variance.build_records import EXPOSURE_TYPE, VarianceBuildSpec
from exp.optimize.evaluation.variance.contracts import OptionCall
from exp.optimize.evaluation.variance.decision_measure import CHOICE_TOOL
from exp.optimize.evaluation.variance.dense_prompt import PLAN_TOOL as DENSE_PLAN_TOOL
from exp.optimize.evaluation.variance.dense_prompt import REPORT_TOOL as DENSE_REPORT_TOOL
from exp.optimize.evaluation.variance.evidence import TraceSearch
from exp.optimize.evaluation.variance.evidence_tools import AGENTIC_PROFILE, EvidenceToolProfile
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import FamilyReport
from exp.optimize.evaluation.variance.judge_prompt import (
    ASSESS_TOOL,
    DEFAULT_AGENTIC_JUDGE,
    JUDGE_PROMPT_VERSION,
)
from exp.optimize.evaluation.variance.judge_test import _call
from exp.optimize.evaluation.variance.judged_test import state_judge
from exp.optimize.evaluation.variance.pipeline import (
    SpendConsentRequired,
    StageExecutor,
    load_stage_record,
    plan_decision_variance,
    run_decision_variance,
    stage_outputs,
)
from exp.optimize.evaluation.variance.pipeline_records import RECORDS_STAGE
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    BuildOutputs,
    BuildSettings,
    CostLine,
    DecisionVarianceSpec,
    FamiliesOutputs,
    MiningOutputs,
    MiningSettings,
    PipelineApproval,
    PipelineReport,
    RecordsOutputs,
    RecordsSettings,
    RulesOutputs,
    RulesSettings,
    StageContext,
    StageDependencies,
    StageName,
    StageRun,
    TraceMLOutputs,
    TraceMLSettings,
    VariantQualification,
)
from exp.optimize.evaluation.variance.pipeline_stages import group_job
from exp.optimize.evaluation.variance.point_options import OPTIONS_TOOL
from exp.optimize.evaluation.variance.records import AuditDisposition
from exp.optimize.evaluation.variance.role_prompts import (
    DRAFT_TOOL_NAME,
    PLAN_TOOL,
    SELF_CHECK_TOOL,
)
from exp.optimize.evaluation.variance.rubric_agent import (
    CRITIQUE_TOOL,
    RubricAuthoringProfile,
)
from exp.optimize.evaluation.variance.rubric_agent import (
    PLAN_TOOL as RUBRIC_PLAN_TOOL,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL, author_rubric, save_rubric
from exp.optimize.evaluation.variance.rubric_author_test import covering_answer, criterion
from exp.optimize.evaluation.variance.rubric_critic import CRITIC_TOOL
from exp.optimize.evaluation.variance.rubric_critic_test import review as design_review
from exp.optimize.evaluation.variance.rubric_revision import REVISION_TOOL, load_revision
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec, schedule_state
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    ACTIONS,
    INITIAL,
    GroundedFixture,
    TicketActor,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.optimize.evaluation.variance.world_agent import PLAN_TOOL as WORLD_PLAN_TOOL
from exp.optimize.evaluation.variance.world_author import WORLD_TOOL
from exp.optimize.evaluation.variance.world_author_test import world_answer
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.admission import EvaluationSpecification, persist_specification
from exp.simulation.evaluation.tasks import TaskVerifier
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.error_tags import ERROR_BANK_ID
from exp.simulation.mining.traceml import ACTION_LABELS, TRACEML_BANK_ID, load_traceml_labels
from exp.simulation.mining.traceml_moves import MOVE_TOOL
from exp.simulation.retrieval import CorpusScope, RAGLineageBinding, persist_trace_rag
from exp.simulation.rules.contracts_test import call, result
from exp.simulation.rules.model import RuleBook

REVISION = "test"
"""Producing source identity of every saved artifact."""

TIER_B = "actor-b"
"""The second actor tier's catalog alias."""

FIRST_REPLICATES = ("-cell-0", "-branch-0")
"""Episode id suffixes of each group's first replicate, which the policy judges a failure."""

Answer = Callable[[JsonObject], JsonObject]
"""Maps a request payload to the forced tool's arguments."""


class Interrupted(BaseException):
    """Simulates the process stopping in the middle of a judge request."""


def _user(payload: JsonObject) -> JsonObject:
    """The parsed JSON user content of a request."""
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[-1], dict)
    content = json.loads(str(messages[-1]["content"]))
    assert isinstance(content, dict)
    return content


def _find(payload: JsonObject, key: str) -> JsonValue:
    """The value under ``key`` in the first JSON message content that has it, else None."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.startswith("{"):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and key in parsed:
            return parsed[key]
    return None


def _packet(payload: JsonObject) -> AuthoringPacket:
    """The authoring packet an author request carries."""
    return AuthoringPacket.model_validate(_find(payload, "authoring_packet"))


def _relocate(value: JsonValue, supplied: frozenset[str], variants: tuple[str, ...]) -> JsonValue:
    """Point fixture rule locators and variant ids a packet does not supply at ones it does.

    A rule book learned from traces names its rules by content digest, and a whole-task packet
    has only its baseline variant, so scripted answers written for the saved ticket book are
    re-pointed at what the packet actually supplies.
    """
    rules = sorted(ref for ref in supplied if ref.startswith("rule:"))
    if isinstance(value, dict):
        fixed: JsonObject = {}
        for key, item in value.items():
            if key == "scenario_ids" and isinstance(item, list):
                kept = [row for row in item if row in variants]
                fixed[key] = list(kept or variants)
            elif key == "scenario_id" and isinstance(item, str) and item not in variants:
                fixed[key] = variants[0]
            else:
                fixed[key] = _relocate(item, supplied, variants)
        return fixed
    if isinstance(value, list):
        return [_relocate(item, supplied, variants) for item in value]
    if isinstance(value, str) and value.startswith("rule:") and value not in supplied:
        return rules[0] if rules else "tool:check_sla"
    return value


def _rebind(answer: JsonObject, packet: AuthoringPacket) -> None:
    """Bind the packet's own rules when the scripted world binds rules the packet lacks."""
    world = answer.get("world")
    if not isinstance(world, dict) or not isinstance(world.get("rule_bindings"), list):
        return
    known = {rule.rule_id for rule in packet.rules}
    bindings = world["rule_bindings"]
    assert isinstance(bindings, list)
    if all(isinstance(row, dict) and row.get("rule_id") in known for row in bindings):
        return
    world["rule_bindings"] = [
        {
            "rule_id": rule.rule_id,
            "use": "enforced" if rule.status == "supported" else "diagnostic",
            "note": "bound",
        }
        for rule in packet.rules
    ]


def _supplied_only(value: JsonValue, supplied: frozenset[str]) -> JsonValue:
    """Drop rule locators the decision author was not given."""
    if isinstance(value, dict):
        return {key: _supplied_only(item, supplied) for key, item in value.items()}
    if isinstance(value, list):
        return [
            _supplied_only(item, supplied)
            for item in value
            if not (isinstance(item, str) and item.startswith("rule:") and item not in supplied)
        ]
    return value


def semantic(request: JsonObject) -> str:
    """Pass a complete episode whose last ticket outcome is right; first replicates fail."""
    evidence = request["episode_evidence"]
    assert isinstance(evidence, dict)
    verdict = state_judge(("tickets", "outcome"), "right")(evidence)
    binding = request["judge_binding"]
    assert isinstance(binding, dict)
    if verdict == "pass" and str(binding["episode_id"]).endswith(FIRST_REPLICATES):
        return "fail"
    return verdict


def assessment(
    request: JsonObject, verdict: str, problems: tuple[str, ...] = ()
) -> JsonHttpResponse:
    """Answer ``assess_episode`` for one verdict with a finding on every rubric criterion."""
    evidence = request["episode_evidence"]
    criteria = request["rubric_criteria"]
    assert isinstance(evidence, dict) and isinstance(criteria, list)
    events = evidence["events"]
    assert isinstance(events, list)
    scored = [e for e in events if isinstance(e, dict) and e["scope"] == "scored"]
    decisive = (scored or events)[0]
    assert isinstance(decisive, dict)
    status = {"pass": "met", "fail": "not_met", "unassessable": "insufficient_evidence"}[verdict]
    findings: list[JsonValue] = []
    for row in criteria:
        assert isinstance(row, dict)
        found = status if row["role"] == "required_correctness" else "met"
        findings.append(
            {
                "criterion_id": row["criterion_id"],
                "status": found,
                "event_refs": [] if found == "insufficient_evidence" else [decisive["event_id"]],
                "reason": f"{row['criterion_id']} is {found}.",
            }
        )
    return _call(
        ASSESS_TOOL,
        {
            "verdict": verdict,
            "criterion_findings": findings,
            "reason": f"Overall {verdict}.",
            "decisive_refs": [] if verdict == "unassessable" else [decisive["event_id"]],
            "rubric_problems": list(problems),
        },
    )


def decision_answer(payload: JsonObject) -> JsonObject:
    """The decision-state author's answer, grounded in the before-state and the tier rule.

    Each option carries its executable first call, so a family without explicit actions forces
    exactly these calls. Rule locators the author was not given are dropped.
    """
    state = tier_decision(0)
    options = tuple(
        option.model_copy(update={"call": OptionCall(tool=option.option_id)})
        for option in state.options
    )
    answer: JsonObject = {
        "state": state.model_copy(update={"options": options}).model_dump(
            mode="json", exclude_defaults=True
        ),
        "grounding": [
            {
                "path": "/tickets/severity",
                "revealed_by": ["check_sla"],
                "refs": ["state:/tickets/severity", "rule:tickets.tier"],
            }
        ],
        "limitations": ["Simulated tickets only."],
    }
    citable = _find(payload, "citable_locators")
    assert isinstance(citable, list)
    fixed = _supplied_only(answer, frozenset(str(ref) for ref in citable))
    assert isinstance(fixed, dict)
    return fixed


def extraction_answer(payload: JsonObject) -> JsonObject:
    """Rule proposals from one trajectory: the check_sla lock boundary and handling effects."""
    view = _find(payload, "trajectory")
    assert isinstance(view, dict) and isinstance(view["steps"], list)
    rules: list[JsonValue] = []
    seen: set[str] = set()
    for step in view["steps"]:
        assert isinstance(step, dict) and isinstance(step["action"], dict)
        name = str(step["action"]["name"])
        evidence = [step["observation_locator"] or step["locator"]]
        if name in seen:
            continue
        seen.add(name)
        if name == "check_sla":
            rules.append(
                {
                    "tool": "check_sla",
                    "kind": "precondition",
                    "statement": "check_sla answers only for a ticket nobody holds.",
                    "predicate": {
                        "op": "in",
                        "left": {"scope": "before", "path": "/tickets/lock"},
                        "right": {"value": ["none"]},
                    },
                    "evidence": evidence,
                    "confidence": 0.8,
                }
            )
        elif name in ("escalate", "close"):
            rules.append(
                {
                    "tool": name,
                    "kind": "effect",
                    "statement": f"{name} records the ticket's handling status.",
                    "evidence": evidence,
                    "confidence": 0.8,
                }
            )
    return {"trace_id": view["trace_id"], "rules": rules, "limitations": ["One trajectory."]}


def rules_answer(role: str, opening: JsonObject) -> JsonObject:
    """Answer the rules agents: ticket research, supported weighings and a short memory."""
    if role == "research":
        return extraction_answer({"messages": [{"content": json.dumps(opening)}]})
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return {
            "rule_id": rule["rule_id"],
            "status": "supported",
            "rationale": "The saved ticket trajectories observe it.",
            "supporting": rule["evidence"][:1],
        }
    return {"summary": "check_sla answers only for an unheld ticket.", "entries": []}


def rules_reads(role: str, opening: JsonObject) -> tuple[str, JsonObject]:
    """One harmless read per rules agent role."""
    if role == "research":
        return "search_related_steps", {"tool_name": "check_sla", "top_k": 2}
    return default_reads(role, opening)


RULES_AGENT = AgentScript(rules_answer, reads=rules_reads)
"""The scripted research, weighing and condensation agents of the rules stage."""


def rules_agent_role(payload: JsonObject) -> str | None:
    """The rules agent role a request belongs to, from the start of its system prompt."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return None
    system = str(messages[0].get("content") or "")
    role = next((name for mark, name in ROLE_MARKERS.items() if system.startswith(mark)), None)
    return None if role == "extraction" else role


def scenario_answer() -> JsonObject:
    """The scenario author's answer: the urgent and routine tiers."""
    scenarios = (tier_scenario("urgent", "high"), tier_scenario("routine", "low"))
    return {
        "scenarios": [s.model_dump(mode="json", exclude_defaults=True) for s in scenarios],
        "limitations": ["Tier variants."],
    }


def selector_answer() -> JsonObject:
    """One point selector choosing the start, before the actor has read the tier."""
    return {
        "points": [
            {
                "step": 0,
                "decision": "Escalate or close the ticket",
                "alternatives": ["Check the service tier first"],
                "deciding_state": [],
                "rationale": "Attempts diverge on the ticket's handling.",
                "evidence_refs": ["point:0", "group"],
            }
        ],
        "audit": [
            {
                "category": "guess_required",
                "explanation": "Some attempts never read the tier before handling the ticket.",
                "evidence_refs": ["group"],
            }
        ],
        "limitations": ["Simulated tickets only."],
    }


def critic_answer() -> JsonObject:
    """One critic estimating the chosen point."""
    return {
        "estimates": [
            {"step": 0, "probability": 0.6, "rationale": "Base rate.", "evidence_refs": ["group"]}
        ],
        "limitations": ["Simulated tickets only."],
    }


WORLD_PLAN: JsonObject = {
    "state_machine": "tickets: open -> escalated | closed; close refuses unless lock is none",
    "programs": ["read_ticket", "check_sla", "escalate", "close"],
    "failure_modes": ["held ticket refused", "missing lock refused"],
    "fixtures": ["held close", "escalate high"],
    "supported_by": ["rule:tickets.lock"],
}
"""The world author's plan."""

RUBRIC_PLAN: JsonObject = {
    "success_condition": "The ticket is handled as its tier requires and reported truthfully.",
    "requirements": [
        {
            "statement": "A high tier ticket is escalated and a low tier ticket closed.",
            "supported_by": ["rule:tickets.tier"],
            "scenario_ids": ["urgent", "routine"],
            "accounts_for": [],
        }
    ],
    "alternatives": ["Reading the ticket before checking the tier."],
}
"""The rubric author's plan."""

RUBRIC_CRITIQUE: JsonObject = {
    "findings": [
        {"check": 16, "status": "satisfied", "criteria": ["handled"], "note": "Alternatives."}
    ],
    "revisions": [],
}
"""The rubric author's self-critique."""

AMBIGUOUS = "criterion handled does not say whether the tier must be read before handling"
"""The rubric problem the troubled judge reports."""

CHECKED = criterion("tier.checked", question="Was the ticket's tier read before it was handled?")
"""The criterion a revision adds, which resolves the problem."""


def moves_answer(window: JsonObject) -> JsonObject:
    """Judge a TraceML window: every step after a failed one is a pivot citing that failure."""
    context, shown, focal = window["context_steps"], window["focal_steps"], window["focal_locators"]
    assert isinstance(context, list) and isinstance(shown, list) and isinstance(focal, list)
    steps = [*context, *shown]
    moves: list[JsonValue] = []
    for before, step in zip(steps, steps[1:], strict=False):
        assert isinstance(before, dict) and isinstance(step, dict)
        if step["locator"] in focal and before["status"] == "failed":
            moves.append(
                {
                    "kind": "pivot",
                    "step": step["locator"],
                    "evidence": [before["locator"]],
                    "rationale": "The agent changed course after the failed step.",
                }
            )
    return {"judged_steps": focal, "moves": moves, "limitations": ["Scripted fixture judge."]}


def revision_answer(payload: JsonObject) -> JsonObject:
    """A revised rubric adding the tier-check criterion and resolving every cited problem."""
    cited = _user(payload)["problems"]
    assert isinstance(cited, list)
    return {
        **covering_answer(CHECKED),
        "resolutions": [
            {"problem_id": row["problem_id"], "change": "Required reading the tier first."}
            for row in cited
            if isinstance(row, dict)
        ],
    }


def unrevised(request: JsonObject) -> bool:
    """Whether a judge request applies a rubric without the revision's criterion."""
    criteria = request["rubric_criteria"]
    assert isinstance(criteria, list)
    return all(isinstance(row, dict) and row["criterion_id"] != "tier.checked" for row in criteria)


def _contents(payload: JsonObject) -> list[JsonValue]:
    """Every JSON message content of a request, parsed."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    parsed: list[JsonValue] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.startswith("{"):
            try:
                parsed.append(json.loads(content))
            except json.JSONDecodeError:
                continue
    return parsed


def _deep(value: JsonValue, key: str) -> JsonValue:
    """The first value under ``key`` anywhere in a JSON value, depth first."""
    if isinstance(value, dict):
        if key in value:
            return value[key]
        value = list(value.values())
    for item in value if isinstance(value, list) else ():
        found = _deep(item, key)
        if found is not None:
            return found
    return None


def options_answer(payload: JsonObject) -> JsonObject:
    """Derived point options citing the first and last continuation events the draw may read."""
    locators = _deep(_contents(payload), "locators")
    assert isinstance(locators, list)
    events = sorted(str(ref) for ref in locators if "/event:" in str(ref))
    assert events, "the point's continuations carry no scored event"
    return {
        "decision": "How to handle the ticket.",
        "options": [
            {
                "option_id": "escalate",
                "action": "Escalate the ticket",
                "evidence_refs": [events[0]],
            },
            {"option_id": "close", "action": "Close the ticket", "evidence_refs": [events[-1]]},
        ],
    }


def choice_answer(payload: JsonObject) -> JsonObject:
    """Classify an episode's first scored actor event as its first derived option."""
    episode = _deep(_contents(payload), "episode")
    assert isinstance(episode, dict) and isinstance(episode["events"], list)
    actor = [
        row["event_id"]
        for row in episode["events"]
        if isinstance(row, dict) and row["owner"] == "actor" and row["scope"] == "scored"
    ]
    if not actor:
        return {
            "option_id": "unclear",
            "action_event": "",
            "evidence_refs": [episode["events"][0]["event_id"]],
            "rationale": "No scored actor event.",
        }
    labels = _deep(payload["tools"], "enum")
    assert isinstance(labels, list)
    return {
        "option_id": labels[0],
        "action_event": actor[0],
        "evidence_refs": [actor[0]],
        "rationale": "The first scored actor call.",
    }


ANSWERS: dict[str, Answer] = {
    OPTIONS_TOOL: options_answer,
    CHOICE_TOOL: choice_answer,
    REVISION_TOOL: revision_answer,
    "choose_points": lambda _: selector_answer(),
    "estimate_points": lambda _: critic_answer(),
    DECISION_STATE_TOOL: decision_answer,
    SCENARIO_TOOL: lambda _: scenario_answer(),
    WORLD_PLAN_TOOL: lambda _: WORLD_PLAN,
    WORLD_TOOL: lambda payload: world_answer(_packet(payload)),
    RUBRIC_PLAN_TOOL: lambda _: RUBRIC_PLAN,
    RUBRIC_TOOL: lambda _: covering_answer(),
    CRITIQUE_TOOL: lambda _: RUBRIC_CRITIQUE,
    EXTRACTION_TOOL: extraction_answer,
    MOVE_TOOL: lambda payload: moves_answer(_user(payload)),
    LABEL_TOOL: lambda _: {"labels": [], "notes": "No behavior beyond the library."},
    CRITIC_TOOL: lambda _: design_review(),
}
"""Every non-judge role's answer, by the forced tool it must call."""

_ROLE_DRAWS = ("You derive the decision an agent", "You classify one finished episode")
"""Openings of the records stage's option proposer and choice classifier, which never search."""

PACKET_TOOLS = frozenset(
    {REVISION_TOOL, WORLD_PLAN_TOOL, WORLD_TOOL, RUBRIC_PLAN_TOOL, RUBRIC_TOOL, CRITIQUE_TOOL}
)
"""Author tools whose answers are re-pointed at what the request's packet supplies."""

FREE_CHOICE = "required"
"""The tool choice of an agentic role turn that may read evidence or close its phase."""

ROLE_PLAN: JsonObject = {
    "hypotheses": ["The tier decides the handling."],
    "planned_reads": ["group"],
}
"""An agentic selector or critic draw's plan."""


def _called(payload: JsonObject) -> set[str]:
    """Tools the conversation's earlier assistant turns called."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    names: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        for made in calls if isinstance(calls, list) else ():
            function = made.get("function") if isinstance(made, dict) else None
            if isinstance(function, dict):
                names.add(str(function.get("name")))
    return names


def _draft_answer(payload: JsonObject) -> JsonObject:
    """The draft an agentic draw submits: its role's answer, chosen by the draft schema."""
    tools = payload["tools"]
    assert isinstance(tools, list)
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == DRAFT_TOOL_NAME:
            schema = function.get("parameters")
            properties = schema.get("properties") if isinstance(schema, dict) else None
            if isinstance(properties, dict) and "points" in properties:
                return selector_answer()
            if isinstance(properties, dict) and "options" in properties:
                return options_answer(payload)
            if isinstance(properties, dict) and "option_id" in properties:
                return choice_answer(payload)
            return critic_answer()
    raise AssertionError("an agentic investigate turn offered no draft tool")


def _free_turn(payload: JsonObject) -> tuple[str, JsonObject]:
    """One agentic role turn after the plan: one evidence read, the draft, then the self check."""
    called = _called(payload)
    if "list_divergence_points" not in called:
        return "list_divergence_points", {}
    if DRAFT_TOOL_NAME not in called:
        return DRAFT_TOOL_NAME, _draft_answer(payload)
    return SELF_CHECK_TOOL.name, {"issues": [], "revise": False}


def _judge_phase(tool: str, request: JsonObject) -> JsonObject:
    """An agentic judge's plan, criterion checks or self check, citing the decisive event."""
    criteria = request["rubric_criteria"]
    evidence = request["episode_evidence"]
    assert isinstance(criteria, list) and isinstance(evidence, dict)
    events = evidence["events"]
    assert isinstance(events, list)
    scored = [e for e in events if isinstance(e, dict) and e["scope"] == "scored"]
    decisive = (scored or events)[0]
    assert isinstance(decisive, dict)
    ids = [str(row["criterion_id"]) for row in criteria if isinstance(row, dict)]
    if tool == "record_plan":
        return {"criteria": [{"criterion_id": c, "needs": str(decisive["event_id"])} for c in ids]}
    if tool == "record_criterion_checks":
        verdict = semantic(request)
        status = {"pass": "met", "fail": "not_met", "unassessable": "insufficient_evidence"}
        return {
            "findings": [
                {
                    "criterion_id": c,
                    "status": status[verdict],
                    "event_refs": [] if verdict == "unassessable" else [decisive["event_id"]],
                    "reason": f"{c} read at the decisive event.",
                }
                for c in ids
            ],
            "leaning": verdict,
        }
    return {"issues": [], "revise": False}


def dense_reply(payload: JsonObject) -> JsonHttpResponse | None:
    """Answer a dense judge draw: plan every candidate, then report each absent."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[1], dict):
        return None
    content = messages[1].get("content")
    if not isinstance(content, str) or '"candidates":' not in content:
        return None
    data = json.loads(content)
    ids = [row["candidate_id"] for row in data["candidates"]]
    tags = {row["candidate_id"]: row["tag_id"] for row in data["candidates"]}
    choice = payload.get("tool_choice")
    forced = choice.get("function") if isinstance(choice, dict) else None
    if isinstance(forced, dict) and forced.get("name") == DENSE_PLAN_TOOL:
        checks = [{"candidate_id": key, "evidence_needed": "actor events"} for key in ids]
        return tool_reply(DENSE_PLAN_TOOL, {"checks": checks})
    findings = [
        {
            "candidate_id": key,
            "tag_id": tags[key],
            "decision": "absent",
            "rationale": "The scripted check finds no such behavior.",
        }
        for key in ids
    ]
    return tool_reply(DENSE_REPORT_TOOL, {"findings": findings})


class Wire(JsonHttpTransport):
    """The scripted author, role and judge wire, answering by the request's forced tool.

    Attributes:
        requests: Every non-judge request, as ``(tool, payload)``.
        judged: Every judge request's ``<judge definition>/<episode id>``, in order.
        interrupt_at: Judge request number (1-based, counted from now) that raises
            :class:`Interrupted` once, if set.
        troubled: Whether the judge reports a rubric problem for a judge request, if set.
        usage: Whether replies report token usage, which agentic role budgets price.
        phases: Every agentic judge phase request before an assessment, by its tool.
        dense: Every dense judge request of the records stage's in-flow dense reward.
    """

    def __init__(self, *, usage: bool = False) -> None:
        """Start with no request served."""
        self.requests: list[tuple[str, JsonObject]] = []
        self.judged: list[str] = []
        self.interrupt_at: int | None = None
        self.troubled: Callable[[JsonObject], bool] | None = None
        self.usage = usage
        self.phases: list[str] = []
        self.dense: list[JsonObject] = []

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse metadata reads.

        Raises:
            AssertionError: Always.
        """
        del url, headers, timeout_seconds
        raise AssertionError("the wire received a metadata request")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Judge an episode from the policy, run a rules agent, or answer the forced role tool.

        Raises:
            Interrupted: At the configured judge request.
        """
        del url, headers, timeout_seconds
        dense = dense_reply(payload)
        if dense is not None:
            self.dense.append(payload)
            return self.priced(dense)
        rules_role = rules_agent_role(payload)
        if rules_role is not None:
            self.requests.append((f"rules:{rules_role}", payload))
            return self.priced(RULES_AGENT._reply(payload))
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        tool = str(function.get("name")) if isinstance(function, dict) else FREE_CHOICE
        judged = _find(payload, "judge_binding")
        if judged is None:
            self.requests.append((tool, payload))
            return self.reply(*self.role(tool, payload))
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        request = json.loads(str(messages[1]["content"]))
        if tool != ASSESS_TOOL:
            self.phases.append(tool)
            return self.reply(tool, _judge_phase(tool, request))
        binding = request["judge_binding"]
        self.judged.append(f"{binding['judge_definition']}/{binding['episode_id']}")
        if self.interrupt_at is not None:
            self.interrupt_at -= 1
            if self.interrupt_at == 0:
                self.interrupt_at = None
                raise Interrupted
        problems = (AMBIGUOUS,) if self.troubled is not None and self.troubled(request) else ()
        response = assessment(request, semantic(request), problems)
        return self.priced(response)

    def role(self, tool: str, payload: JsonObject) -> tuple[str, JsonObject]:
        """Answer one author or role request by its forced tool, or an agentic free turn."""
        if tool == FREE_CHOICE:
            return _free_turn(payload)
        if tool == PLAN_TOOL.name:
            return tool, ROLE_PLAN
        if tool == SELF_CHECK_TOOL.name:
            return tool, {"issues": [], "revise": False}
        answer = ANSWERS[tool](payload)
        if tool in PACKET_TOOLS and _find(payload, "authoring_packet") is not None:
            packet = _packet(payload)
            variants = tuple(variant.variant_id for variant in packet.variants)
            fixed = _relocate(answer, packet.locators(), variants)
            assert isinstance(fixed, dict)
            answer = fixed
            if tool == WORLD_TOOL:
                _rebind(answer, packet)
        return tool, answer

    def reply(self, tool: str, arguments: JsonObject) -> JsonHttpResponse:
        """One reply calling ``tool``, with token usage when the wire is priced."""
        return self.priced(tool_reply(tool, arguments))

    def priced(self, response: JsonHttpResponse) -> JsonHttpResponse:
        """Add token usage a priced catalog can price, when the wire is priced."""
        if not self.usage:
            return response
        assert isinstance(response.body, dict)
        usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
        return JsonHttpResponse(
            status_code=response.status_code, body={**response.body, "usage": usage}
        )

    def calls(self) -> tuple[int, int]:
        """Role and judge requests so far."""
        return len(self.requests), len(self.judged)


@dataclass
class TieredActor(JsonHttpTransport):
    """The fixture's scripted actor policy, answering as whichever actor model was requested.

    Attributes:
        policy: The fixture's actor transport, which records every request.
    """

    policy: TicketActor

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Delegate a metadata read, which the policy refuses."""
        return self.policy.get(url, headers=headers, timeout_seconds=timeout_seconds)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer from the policy under the requested model's identity."""
        reply = self.policy.post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        assert isinstance(reply.body, dict)
        return JsonHttpResponse(
            status_code=reply.status_code, body={**reply.body, "model": payload["model"]}
        )


TRACE_STEPS: dict[str, tuple[tuple[str, JsonObject], ...]] = {
    "trace-1": (
        ("check_sla", {"severity": "high", "lock": "none"}),
        ("escalate", {"status": "escalated"}),
    ),
    "trace-2": (
        ("read_ticket", {"ticket": "T-2", "status": "open"}),
        ("check_sla", {"severity": "low", "lock": "none"}),
        ("close", {"status": "closed"}),
    ),
}
"""Each world-learning trajectory's tool calls and replies, covering every ticket tool."""


def _ticket_trace(trace_id: str) -> Trace:
    """A world-learning trajectory: check the tier, then handle the ticket the tier requires."""
    spans = []
    for index, (name, reply) in enumerate(TRACE_STEPS[trace_id]):
        key = f"k{index}"
        spans.append(call(f"c{index}", 2 * index, name, {}, key))
        spans.append(result(f"r{index}", 2 * index + 1, name, reply, key))
    return Trace(
        trace_id=trace_id,
        task="Resolve the support ticket with the tools you have.",
        tools=(),
        spans=tuple(spans),
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="tickets.otlp", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def exposure_plan() -> ExposurePlan:
    """Mining probes, world learning and final evaluation, fixed before anything is tagged."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "mining_probes", "sources": "world_learning", "held": "final_evaluation"},
        traces={"trace-1": "sources", "trace-2": "sources", "trace-final": "held"},
    )


@dataclass(frozen=True)
class FixtureBuild:
    """The variance-build stand-in: the exposure plan and the world-learning trace dataset.

    Attributes:
        fixture: The grounded fixture whose world model the stage names.
    """

    fixture: GroundedFixture

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Key the stand-in by its fixed content."""
        del project, spec
        return {"fixture": "grounded-tickets-v1"}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[()], tuple[str, ...]]:
        """The stand-in makes no call."""
        del project, spec, code_revision
        return (), ("Fixture build: grounded ticket world; exposure fixed before tagging.",)

    def run(self, context: StageContext) -> StageRun:
        """Save the exposure plan and the trace dataset, or adopt the saved ones."""
        store = context.project.artifacts
        if not store.exists("fixture-exposure"):
            store.write_json(
                artifact_id="fixture-exposure",
                artifact_type=EXPOSURE_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=datetime.now(UTC), code_revision=REVISION
                ),
                files={"exposure.json": exposure_plan()},
            )
        exposure = artifact_input(store.read("fixture-exposure").manifest)
        if not store.exists("fixture-traces"):
            persist_trace_dataset(
                TraceNormalizationResult(
                    traces=(_ticket_trace("trace-1"), _ticket_trace("trace-2")), issues=()
                ),
                store,
                created_at=datetime.now(UTC),
                code_revision=REVISION,
                dataset_id="fixture-traces",
            )
        dataset = artifact_input(store.read("fixture-traces").manifest)
        scope = CorpusScope.declare(
            name="world-learning-retrieval",
            kind="trace",
            ids=("trace-1", "trace-2"),
            source_plan=exposure,
        )
        if not store.exists("fixture-index"):
            persist_trace_rag(
                store,
                (dataset,),
                tuple(
                    RAGLineageBinding(trace_id=trace_id, lineage_id="sources", partition="fit")
                    for trace_id in ("trace-1", "trace-2")
                ),
                created_at=datetime.now(UTC),
                code_revision=REVISION,
                rag_id="fixture-index",
                corpus_scope=scope,
            )
        index = artifact_input(store.read("fixture-index").manifest)
        outputs = BuildOutputs(
            report=exposure,
            exposure=exposure,
            world_model=self.fixture.definition.world_model,
            trace_dataset=dataset,
            rules_traces=("trace-1", "trace-2"),
            index=index,
            retrieval_scope=scope,
        )
        return StageRun(outputs, (exposure, dataset, index), {"fixture": True})


def fixture_rules_stage(
    project: ProjectStore,
    spec: RulesStageSpec,
    *,
    exposure: ExposurePlan,
    traces: tuple[Trace, ...],
    code_revision: str,
    search: TraceSearch | None = None,
    decisions: DecisionsTraceTagger | None = None,
) -> RulesStageReport:
    """The rules owner's stand-in: the saved ticket rule book, after the owner's exposure check."""
    del search, decisions
    exposure.require((trace.trace_id for trace in traces), "world_learning")
    rulebook, book = save_book(project)
    report = _report(spec, book, rulebook)
    if not project.artifacts.exists(report.identity + "-result"):
        project.artifacts.write_json(
            artifact_id=report.identity + "-result",
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(rulebook,),
            ),
            files={"report.json": report},
        )
    return report


def _report(spec: RulesStageSpec, book: RuleBook, rulebook: ArtifactInput) -> RulesStageReport:
    """A rules-stage report naming the saved ticket rule book."""
    return RulesStageReport(
        spec=spec,
        identity="pipeline-ticket-rules-stage",
        extraction_identity="pipeline-ticket-rules-extraction",
        contracts=dict(book.contracts),
        draws=(),
        failures=(),
        rulebook=rulebook,
        rules=len(book.rules),
        rule_counts={"supported": len(book.rules), "contradicted": 0, "unknown": 0},
        open_questions=len(book.questions),
        diagnostic=(),
        requires_witness=(),
        sessions=(),
        session_ids=(),
        properties=None,
        workloads=(),
        triage=(),
        families_must_stay_diagnostic=(),
    )


def write_bank(root: Path) -> Path:
    """Write the local ``traceml-action-v1`` bank with every action column."""
    banks = root / "banks"
    banks.mkdir(exist_ok=True)
    questions = "".join(
        f'\n[[questions]]\nquestion_id = "action.{label}"\nfamily = "action"\n'
        f'text = "Is the step {label}?"\n'
        for label in ACTION_LABELS
    )
    (banks / f"{TRACEML_BANK_ID}.toml").write_text(
        f'bank_id = "{TRACEML_BANK_ID}"\nversion = 1\n{questions}'
    )
    errors = "".join(
        f'\n[[questions]]\nquestion_id = "{question}"\nfamily = "error"\n'
        f'text = "Did the agent commit {question}?"\n'
        for question in (FIRED_ERROR, "error.destructive_action")
    )
    (banks / f"{ERROR_BANK_ID}.toml").write_text(
        f'bank_id = "{ERROR_BANK_ID}"\nversion = 1\n{errors}'
    )
    behaviors = "".join(
        f'\n[[questions]]\nquestion_id = "{question}"\nfamily = "behavior.{side}"\n'
        f'text = "Does the focal action show {question}?"\n'
        for question, side in BASE_BEHAVIORS
    )
    (banks / "behavior-base-v1.toml").write_text(
        f'bank_id = "behavior-base-v1"\nversion = 1\n{behaviors}'
    )
    return banks


BASE_BEHAVIORS = (
    ("changed_implementation", "positive"),
    ("verified_result", "positive"),
    ("unsupported_success_claim", "negative"),
    ("modified_evaluation", "negative"),
    ("skipped_required_step", "negative"),
    ("exploited_metric", "negative"),
    ("hid_failure", "negative"),
)
"""The in-flow dense vocabulary's base behavior bank, as the fixture publishes it."""


FIRED_ERROR = "error.unverified_success_claim"
"""The error proposition the scripted tagger finds in every attempt and step."""


def step_tagger(project: ProjectStore) -> DecisionsTraceTagger:
    """The real decisions tagger over a scripted boundary labelling every step a read."""
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0]

    def decide(request: DecisionRequest) -> JsonObject:
        """Answer every proposition: the step is a read."""
        questions = request.questions
        return {
            "answers": {
                question: {
                    "type": "noul",
                    "noul": 0.9 if question in ("action.read", FIRED_ERROR) else 0.05,
                }
                for question in questions
            },
            "usage": {"cost": 0.0},
        }

    return DecisionsTraceTagger(decide, model, maximum_concurrency=2)


def add_tier(project: ProjectStore, alias: str) -> None:
    """Add a second actor alias on the fixture's connection."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["actor"]
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    alias: ModelRecord(
                        connection=record.connection,
                        model=f"{alias}-v1",
                        billing_source=BillingSource.CUSTOMER_MANAGED,
                        capabilities=ModelCapabilities(
                            supports_tools=True,
                            supports_structured_output=True,
                            context_window_tokens=200_000,
                            maximum_output_tokens=8_192,
                        ),
                    ),
                }
            }
        ),
    )


def rebound(fixture: GroundedFixture, alias: str) -> GroundedFixture:
    """Add an actor tier and rebind every frozen setting to the changed catalog."""
    add_tier(fixture.project, alias)
    catalog = load_model_catalog(fixture.project.model_catalog_path)
    digest = sha256_json(catalog.model_dump(mode="json"))
    definition = fixture.definition.model_copy(update={"catalog_sha256": digest})
    job = fixture.job.model_copy(
        update={"catalog_sha256": digest, "configuration": definition.model_dump(mode="json")}
    )
    settings = fixture.settings.model_copy(update={"catalog_sha256": digest})
    return replace(fixture, job=job, definition=definition, settings=settings)


def mining_rubric(fixture: GroundedFixture) -> ArtifactInput:
    """Author and save the mining-probe task's rubric before the run, as an operator would."""
    project = fixture.project
    rulebook, book = save_book(project)
    start = load_continuation_checkpoint(project.artifacts, fixture.job.study.checkpoint)
    packet = build_authoring_packet(
        family_id="ticket-mining",
        task=fixture.definition.task,
        checkpoint=fixture.job.study.checkpoint,
        before_state=INITIAL,
        public_history=start.actor.messages,
        remaining_calls=5,
        sources=(
            SourceRef(
                source_id="trace-1",
                kind="trace",
                trace_id="trace-1",
                summary="An agent checked the tier and escalated the ticket.",
            ),
        ),
        exposure=exposure_plan(),
        rule_book=rulebook,
        book=book,
        decision=tier_decision(0),
        scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
    )
    reference = save_packet(project, packet, code_revision=REVISION)
    draft = author_rubric(
        project, fixture.settings, reference, identity="mining-rubric", code_revision=REVISION
    )
    assert draft.status == "ready", draft.failure
    return save_rubric(project, draft, code_revision=REVISION)


def make_spec(
    fixture: GroundedFixture,
    rubric: ArtifactInput | None,
    bank: Path,
    *,
    actors: tuple[str, ...] = ("actor", TIER_B),
    dispositions: tuple[AuditDisposition, ...] = (),
    crosstalk: bool = True,
    agentic: bool = False,
    schedule: TrainingScheduleSpec | None = None,
    qualification: VariantQualification | None = None,
) -> DecisionVarianceSpec:
    """A small run: G=3, J=1, C=1, K=2, M=3.

    By default selectors and critics answer once, the judge is single pass, actions are explicit
    and the rules stage only binds a saved book. ``agentic`` keeps the default agentic selector,
    critic and judge profiles, derives option actions from the decision-state author and gives
    the rules stage a base state, so the learned membership boundary is executed.
    """
    settings = fixture.settings
    forced = EvidenceToolProfile()
    rules = RulesStageSpec(author=settings)
    if agentic:
        rules = RulesStageSpec(
            author=settings,
            base_state=INITIAL,
            observed_values={"before:/tickets/lock": ("held",)},
        )
    roles: JsonObject = (
        {}
        if agentic
        else {
            "judge_prompt_set": JUDGE_PROMPT_VERSION,
            "judge_evidence_tool_budget": 0,
            "selector_profile": forced.model_dump(mode="json"),
            "critic_profile": forced.model_dump(mode="json"),
        }
    )
    mining = MiningSettings.model_validate(
        {
            "job": fixture.job.model_dump(mode="json"),
            "rubric": None if rubric is None else rubric.model_dump(mode="json"),
            "judge": settings.model_dump(mode="json"),
            "selector": settings.model_dump(mode="json"),
            "critic": settings.model_dump(mode="json"),
            "actors": list(actors),
            "attempts": 3,
            "selector_draws": 1,
            "critic_draws": 1,
            "points": 2,
            "continuations": 3,
            "group_physical_attempts": 40,
            "schedule": None if schedule is None else schedule.model_dump(mode="json"),
            **roles,
        }
    )
    return DecisionVarianceSpec(
        build=BuildSettings(spec=VarianceBuildSpec()),
        # The stand-in rules owner ignores the memory profile, so the default jev_memory profile
        # stays on with the fixture's decision alias; the agentic run learns rules by research.
        rules=RulesSettings(
            spec=rules, maximum_traces=4, jev_memory=not agentic, jev_alias="judge"
        ),
        traceml=TraceMLSettings(
            decision_alias="judge",
            bank_root=bank,
            steps_per_call=2,
            move_judge=settings,
            error_bank=ERROR_BANK_ID,
        ),
        mining=mining,
        authoring=AuthoringSettings(
            decision_author=settings,
            scenario_author=settings,
            world_author=settings,
            rubric_author=settings,
            actions={} if agentic else ACTIONS,
            world_prompt_sets=("failure-mode-first",),
            rubric_profile=RubricAuthoringProfile(variants=("failure-patterns-first",)),
            repeats=6,
            confirmation_repeats=6,
            dispositions=dispositions,
            # The decision-variance measurement's free continuations and agentic classifier and
            # reviewer have their own offline tests; this run keeps it off to stay small.
            decision_variance=False,
        ),
        records=RecordsSettings(
            intent="Escalate or close a support ticket as its service tier requires.",
            crosstalk=crosstalk,
            crosstalk_attempts=3,
            crosstalk_top_k=1,
            crosstalk_repeats=2,
            qualification=qualification,
        ),
    )


@dataclass
class Run:
    """One fixture project wired for the pipeline.

    Attributes:
        fixture: The grounded world and its transports.
        wire: The author, role and judge wire.
        rubric: The operator's mining rubric, or None when the run authors it.
        bank: The TraceML bank directory.
        executors: The build stand-in.
        dependencies: The injected TraceML tagger and Harbor root.
    """

    fixture: GroundedFixture
    wire: Wire
    rubric: ArtifactInput | None
    bank: Path
    executors: dict[StageName, StageExecutor]
    dependencies: StageDependencies

    @property
    def project(self) -> ProjectStore:
        """The project."""
        return self.fixture.project

    def calls(self) -> tuple[int, int, int, int]:
        """Actor, predictor, role and judge requests so far."""
        return (
            len(self.fixture.actor.requests),
            len(self.fixture.predictor.requests),
            *self.wire.calls(),
        )

    def run(self, spec: DecisionVarianceSpec, *, approve: bool = True) -> PipelineReport:
        """Plan, consent once and run or replay the pipeline."""
        plan = plan_decision_variance(
            self.project, spec, code_revision=REVISION, executors=self.executors
        )
        approval = PipelineApproval.for_plan(plan) if approve else None
        return run_decision_variance(
            self.project,
            spec,
            code_revision=REVISION,
            approval=approval,
            executors=self.executors,
            dependencies=self.dependencies,
        )


def wired(root: Path, monkeypatch: pytest.MonkeyPatch, *, agentic: bool = False) -> Run:
    """A grounded fixture with two actor tiers, the scripted wire and the stand-ins installed.

    Args:
        root: Project root.
        monkeypatch: Installs the scripted transports.
        agentic: Price the judge alias and report usage, which agentic role budgets need, and
            keep the real rules extraction and the authored mining rubric instead of the saved
            rule book and the operator's rubric.

    Returns:
        The wired run.
    """
    fixture = rebound(grounded_fixture(root, monkeypatch, priced=agentic), TIER_B)
    wire = Wire(usage=agentic)
    actor = TieredActor(fixture.actor)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_stages.run_rules_stage",
        run_rules_stage if agentic else fixture_rules_stage,
    )
    rubric = None if agentic else mining_rubric(fixture)
    return Run(
        fixture=fixture,
        wire=wire,
        rubric=rubric,
        bank=write_bank(root),
        executors={"variance-build": FixtureBuild(fixture)},
        dependencies=StageDependencies(
            traceml_tagger=step_tagger(fixture.project),
            rules_decisions=step_tagger(fixture.project),
            harbor_root=root / "harbor",
        ),
    )


def dispositions_for(project: ProjectStore, report: PipelineReport) -> tuple[AuditDisposition, ...]:
    """A reviewer disposition for every audit finding of every family of a run."""
    record = load_stage_record(project, report.stages[4].identity)
    assert record is not None
    found: dict[str, AuditDisposition] = {}
    for row in stage_outputs(record, FamiliesOutputs).families:
        assert row.report is not None
        family = FamilyReport.model_validate_json(
            project.artifacts.read_bytes(row.report.artifact_id, "report.json")
        )
        for finding in family.findings:
            found[finding.finding_id] = AuditDisposition(
                finding_id=finding.finding_id,
                resolution="accepted_with_reason",
                rationale="Reviewed against the saved grounded trajectories.",
                resolver="pipeline-reviewer",
            )
    return tuple(found[key] for key in sorted(found))


def reviewed(run: Run, dispositions: tuple[AuditDisposition, ...]) -> PipelineReport:
    """Run or replay the reviewed registration, so its families stage is saved.

    Tests that change only a records field compare against this run. Running it here, instead of
    relying on an earlier test having run it, keeps each test independent of order and of how a
    runner distributes the module.
    """
    report = run.run(make_spec(run.fixture, run.rubric, run.bank, dispositions=dispositions))
    assert report.complete
    return report


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Run, PipelineReport]]:
    """Run the whole pipeline once over a fresh two-tier grounded project."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Module scope outlives the per-test XDG isolation: keep the host's admission limits
        # (and their concurrency) out of the scripted, order-sensitive run.
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("pipeline-xdg")))
        run = wired(tmp_path_factory.mktemp("pipeline"), monkeypatch)
        spec = make_spec(run.fixture, run.rubric, run.bank)
        yield run, run.run(spec)


def _outputs[Model: ContractModel](
    run: Run, report: PipelineReport, index: int, model: type[Model]
) -> Model:
    """One stage's typed outputs."""
    record = load_stage_record(run.project, report.stages[index].identity)
    assert record is not None
    return stage_outputs(record, model)


def test_every_stage_runs_through_its_owner_on_two_tiers(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Stages run in order, each saved, with the rule book bound before any probe."""
    run, report = pipeline
    project = run.project
    assert report.complete and [row.replayed for row in report.stages] == [False] * 6
    assert [row.stage for row in report.stages] == [
        "variance-build",
        "rules",
        "traceml",
        "error-mining",
        "families",
        "records",
    ]
    rules = _outputs(run, report, 1, RulesOutputs)
    bound = rules.job.configuration
    assert bound["rule_book"] == rules.rulebook.model_dump(mode="json")
    assert rules.job.study.checkpoint != run.fixture.job.study.checkpoint
    labels = _outputs(run, report, 2, TraceMLOutputs).labels
    assert set(labels) == {"actor", TIER_B}
    for reference in labels.values():
        saved = load_traceml_labels(project.artifacts, reference)
        assert saved.moves_protocol is not None
        assert all(trace.coverage.moves == "judged" for trace in saved.traces)
    assert any(tool == MOVE_TOOL for tool, _ in run.wire.requests)
    traceml = _outputs(run, report, 2, TraceMLOutputs)
    assert set(traceml.errors) == set(traceml.error_steps) == {"actor", TIER_B}
    build = _outputs(run, report, 0, BuildOutputs)
    mining = _outputs(run, report, 3, MiningOutputs)
    assert mining.profile.prompt_version == JUDGE_PROMPT_VERSION
    assert mining.rubric == run.rubric and mining.rubric_packet is None
    assert [tier.tier for tier in mining.tiers] == ["actor", TIER_B]
    for tier in mining.tiers:
        saved = ActorCriticReport.model_validate_json(
            project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
        )
        assert saved.group_tier is not None and saved.group_tier.label == tier.tier
        assert saved.spec.initial.study.checkpoint == rules.job.study.checkpoint
        # The tier's labels and its error-ranked attempt points are both registered references.
        assert [ref.artifact for ref in saved.spec.references] == [
            labels[tier.tier],
            traceml.error_points[tier.tier],
        ]
        assert [ref.kind for ref in saved.spec.references] == ["traceml_labels", "error_points"]
        assert saved.spec.initial.actor_alias == tier.tier
        # Selectors and critics search the scoped world-learning index, never held-out traces.
        assert tier.retrieval is not None and saved.spec.retrieval == tier.retrieval
        assert tier.retrieval.index == build.index
        assert tier.retrieval.scope == build.retrieval_scope
        assert tier.retrieval.task_id == "ticket"
    families = _outputs(run, report, 4, FamiliesOutputs).families
    assert [(row.tier, row.point) for row in families] == [("actor", 0), (TIER_B, 0)]
    for row in families:
        assert row.failure is None, row.failure
        assert None not in (row.packet, row.world, row.rubric, row.freeze)
        assert row.packet is not None
        packet = load_packet(project, row.packet)
        patterns = {pattern.pattern_id for pattern in packet.coalition.patterns}
        assert patterns == {f"actor.{FIRED_ERROR}", f"{TIER_B}.{FIRED_ERROR}"}
        episodes = packet.observed_episodes
        assert [episode.episode_id for episode in episodes] == ["trace-1", "trace-2"]
        assert [step.tool for step in episodes[1].steps] == ["read_ticket", "check_sla", "close"]
        assert set(row.judging) == {"urgent", "routine"}
        for binding in row.judging.values():
            assert binding.world_freeze == row.freeze
            assert binding.judge.definition.rubric == row.rubric
            assert binding.judge.definition.profile == mining.profile
    usage = [row.usage.physical_attempts for row in report.stages]
    assert usage[0] == 0 and all(count > 0 for count in usage[2:5])


def test_an_unset_group_ceiling_leaves_room_for_every_attempt_to_reach_the_horizon(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Without group_physical_attempts a group of G attempts gets G x maximum_model_calls.

    The registered start's own ceiling (64 in the live spec) was sized for its capture; a group
    of 16 attempts at a 20-call horizon kept under it would exhaust the ledger after a few
    attempts and leave the rest outcome-unknown.
    """
    run, _ = pipeline
    spec = make_spec(run.fixture, run.rubric, run.bank)
    unset = spec.model_copy(
        update={"mining": spec.mining.model_copy(update={"group_physical_attempts": None})}
    )
    job = run.fixture.job.model_copy(update={"maximum_physical_attempts": 1})
    group = group_job(run.project, unset, job, "actor")
    assert group.study.repeats == unset.mining.attempts
    assert group.maximum_physical_attempts == unset.mining.attempts * job.maximum_model_calls
    roomy = job.model_copy(update={"maximum_physical_attempts": 10_000})
    assert group_job(run.project, unset, roomy, "actor").maximum_physical_attempts == 10_000
    assert group_job(run.project, spec, job, "actor").maximum_physical_attempts == 40


def test_a_continuation_ceiling_below_the_horizon_is_raised_to_room_for_every_continuation(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """continuation_physical_attempts is a floor: M continuations at the horizon always fit.

    The live spec's 64 against 8 continuations at a 20-call horizon (160) would exhaust each
    point's ledger and leave its last continuations outcome-unknown.
    """
    run, _ = pipeline
    mining = make_spec(run.fixture, run.rubric, run.bank).mining
    horizon = mining.registered.maximum_model_calls
    low = mining.model_copy(update={"continuation_physical_attempts": 1})
    assert low.continuation_ceiling() == mining.continuations * horizon
    assert low.continuation_ceiling(8) == 8 * horizon
    high = mining.model_copy(update={"continuation_physical_attempts": 10_000})
    assert high.continuation_ceiling() == 10_000


def test_dispositions_rekey_only_families_and_records_with_no_new_call(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """A changed authoring field replays every earlier stage and admits the reviewed families."""
    run, first = pipeline
    project = run.project
    dispositions = dispositions_for(project, first)
    spec = make_spec(run.fixture, run.rubric, run.bank, dispositions=dispositions)
    before = run.calls()
    reviewed = run.run(spec)
    assert reviewed.complete
    assert [row.replayed for row in reviewed.stages] == [True] * 4 + [False] * 2
    assert [row.identity for row in reviewed.stages[:4]] == [
        row.identity for row in first.stages[:4]
    ]
    assert reviewed.stages[4].identity != first.stages[4].identity
    families = _outputs(run, reviewed, 4, FamiliesOutputs).families
    assert all(row.admitted for row in families), [row.reasons for row in families]
    records = _outputs(run, reviewed, 5, RecordsOutputs)
    assert set(records.training) == {"actor/0", f"{TIER_B}/0"}
    assert set(records.selectors) == {"actor", TIER_B}
    assert records.harbor and all(Path(row.path).is_dir() for row in records.harbor)
    assert all(row.diagnostic for row in records.harbor)
    assert records.crosstalk is not None or records.crosstalk_skipped is not None
    after_review = run.calls()
    # Only the cross-talk probe may buy new episodes; every family draw and judgment replays.
    assert after_review[2] == before[2]
    replay = run.run(spec, approve=False)
    assert [row.replayed for row in replay.stages] == [True] * 6
    assert run.calls() == after_review


def test_a_changed_records_field_rekeys_only_the_records_stage(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Disabling cross-talk keeps every earlier stage and makes no call."""
    run, first = pipeline
    dispositions = dispositions_for(run.project, first)
    reviewed(run, dispositions)
    spec = make_spec(run.fixture, run.rubric, run.bank, dispositions=dispositions, crosstalk=False)
    before = run.calls()
    report = run.run(spec)
    assert [row.replayed for row in report.stages] == [True] * 5 + [False]
    assert _outputs(run, report, 5, RecordsOutputs).crosstalk is None
    assert run.calls() == before


def test_unsaved_stages_need_consent_to_the_exact_plan(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """No approval, or an approval of another plan, is refused before any call."""
    run, _ = pipeline
    spec = make_spec(run.fixture, run.rubric, run.bank, crosstalk=False).model_copy(
        update={"records": RecordsSettings(intent="Another intent.", crosstalk=False)}
    )
    before = run.calls()
    with pytest.raises(SpendConsentRequired, match="unsaved"):
        run.run(spec, approve=False)
    stale = PipelineApproval(plan_sha256="0" * 64, ceiling_usd=10.0, unpriced_allowed=True)
    with pytest.raises(SpendConsentRequired, match="another plan"):
        run_decision_variance(
            run.project,
            spec,
            code_revision=REVISION,
            approval=stale,
            executors=run.executors,
            dependencies=run.dependencies,
        )
    assert run.calls() == before


@dataclass(frozen=True)
class PricedRecords:
    """The records executor with one extra priced line, to exercise the ceiling."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The records executor's keys."""
        return RECORDS_STAGE.keys(project, spec)

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """The records plan plus a one-dollar line."""
        costs, lines = RECORDS_STAGE.plan(project, spec, code_revision)
        return (*costs, CostLine(name="priced-probe", attempts=1, usd=1.0)), lines

    def run(self, context: StageContext) -> StageRun:
        """The records executor's run."""
        return RECORDS_STAGE.run(context)


def test_the_plan_prices_every_unsaved_stage_and_a_low_ceiling_stops_before_it(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Saved stages carry no cost; a stage whose bound exceeds the ceiling never starts."""
    run, _ = pipeline
    spec = make_spec(run.fixture, run.rubric, run.bank).model_copy(
        update={
            "records": RecordsSettings(
                intent="A priced intent.",
                crosstalk=True,
                crosstalk_attempts=3,
                crosstalk_top_k=1,
                crosstalk_repeats=2,
            )
        }
    )
    executors: dict[StageName, StageExecutor] = {**run.executors, "records": PricedRecords()}
    plan = plan_decision_variance(run.project, spec, code_revision=REVISION, executors=executors)
    assert [row.saved for row in plan.stages] == [True] * 5 + [False]
    assert all(not row.costs for row in plan.stages[:5])
    records = plan.stages[5]
    assert records.lines and records.known_usd == 1.0
    assert {line.name for line in records.unpriced} >= {"actor", "judge"}
    approval = PipelineApproval.for_plan(plan, ceiling_usd=0.5)
    before = run.calls()
    report = run_decision_variance(
        run.project,
        spec,
        code_revision=REVISION,
        approval=approval,
        executors=executors,
        dependencies=run.dependencies,
    )
    assert report.stopped is not None and "records" in report.stopped
    assert len(report.stages) == 5 and not report.complete
    assert run.calls() == before


def test_rubric_problems_from_judging_drive_a_revision_and_a_new_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmation judging reports a rubric problem; the revised judges own the final family."""
    run = wired(tmp_path, monkeypatch)
    run.wire.troubled = unrevised
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    report = run.run(spec)
    assert report.complete
    (family,) = _outputs(run, report, 4, FamiliesOutputs).families
    assert family.failure is None, family.failure
    assert set(family.revisions) == {"urgent", "routine"}
    for variant, reference in family.revisions.items():
        revision = load_revision(run.project, reference)
        assert revision.status == "revised" and revision.problems
        assert revision.initial.rubric == family.rubric
        binding = family.judging[variant]
        assert binding.judge.definition == revision.final
        assert binding.judge.definition.rubric != family.rubric
    assert family.report is not None
    final = FamilyReport.model_validate_json(
        run.project.artifacts.read_bytes(family.report.artifact_id, "report.json")
    )
    assert final.coverage and all(not row.rubric_problems for row in final.coverage.values())
    before = run.calls()
    assert [row.replayed for row in run.run(spec, approve=False).stages] == [True] * 6
    assert run.calls() == before


def test_an_interrupted_stage_resumes_without_recalling_finished_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge interruption mid-mining resumes: earlier stages replay, nothing is asked twice."""
    run = wired(tmp_path, monkeypatch)
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    run.wire.interrupt_at = 4
    with pytest.raises(Interrupted):
        run.run(spec)
    finished = [
        row.stage
        for row in plan_decision_variance(
            run.project, spec, code_revision=REVISION, executors=run.executors
        ).stages
        if row.saved
    ]
    assert finished == ["variance-build", "rules", "traceml"]
    before = run.calls()
    report = run.run(spec)
    assert report.complete
    assert [row.replayed for row in report.stages] == [True] * 3 + [False] * 3
    # The interrupted request's intent was recorded before dispatch, so its outcome stays unknown
    # and is never sent again; every other saved judgment replays.
    judged = run.wire.judged
    assert len(judged) == len(set(judged)), "no judgment is asked twice"
    assert run.calls()[0] >= before[0]


def ticket_qualification(project: ProjectStore, verifier: TaskVerifier) -> VariantQualification:
    """Independent ticket acceptance and ten controls on each variant's first feasible branch.

    Per partition: two valid outcomes that differ only in what the agent said, a wrong outcome,
    a changed tier (a prohibited effect) and a missing step log (unavailable evidence).

    Args:
        project: Project receiving the specification.
        verifier: The task verifier, whose permitted tools and preserved paths the independent
            specification keeps.

    Returns:
        The qualification settings.
    """
    specification = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="A handled ticket's outcome is right and its tier and lock are kept.",
            predicate=TaskVerifier(
                description="Authored ticket acceptance with an independent outcome schema.",
                outcome_schema={
                    "properties": {
                        "final_state": {
                            "properties": {
                                "tickets": {
                                    "properties": {"outcome": {"const": "right"}},
                                    "required": ["outcome"],
                                }
                            },
                            "required": ["tickets"],
                        }
                    },
                    "required": ["final_state"],
                },
                allowed_tools=verifier.allowed_tools,
                preserved_paths=verifier.preserved_paths,
            ),
            subject="outcome",
        ),
        created_at=datetime.now(UTC),
        code_revision=REVISION,
    )
    edits: dict[str, tuple[JsonValue, ...]] = {
        "valid-one": (),
        "valid-two": (),
        "goal-miss": (
            {"op": "set", "path": ["final_state", "tickets", "outcome"], "value": "wrong"},
        ),
        "side-effect": (
            {"op": "set", "path": ["final_state", "tickets", "severity"], "value": "low"},
        ),
        "unknown": ({"op": "set", "path": ["steps"], "value": None},),
    }
    controls = []
    for partition in ("development", "confirmation"):
        for index, (kind, changes) in enumerate(edits.items()):
            said = {"op": "set", "path": ["final_answer"], "value": f"{partition} control {index}"}
            expected: Literal["pass", "fail", "unassessable"] = (
                "unassessable"
                if kind == "unknown"
                else "pass"
                if kind.startswith("valid")
                else "fail"
            )
            controls.append(
                DeclaredControl(
                    id=f"{partition}-{kind}",
                    lineage=f"ticket-pipeline-{partition}-{index}",
                    partition=partition,
                    expected=expected,
                    branch=0,
                    edits=(said, *changes),
                    rationale="Authored observation-level control over the feasible branch.",
                )
            )
    return VariantQualification(specification=specification, controls=tuple(controls))


def test_admitted_variants_are_qualified_before_their_harbor_export(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """A variant whose feasible run and controls qualify is exported qualified; others record why.

    The scripted actor always escalates, so the routine (low tier) variant has no successful
    feasibility run: it stays diagnostic with its failure recorded, never silently qualified.
    """
    run, first = pipeline
    project = run.project
    dispositions = dispositions_for(project, first)
    reviewed(run, dispositions)
    qualification = ticket_qualification(project, run.fixture.job.study.verifier)
    spec = make_spec(
        run.fixture,
        run.rubric,
        run.bank,
        dispositions=dispositions,
        crosstalk=False,
        qualification=qualification,
    )
    report = run.run(spec)
    assert report.complete
    assert [row.replayed for row in report.stages] == [True] * 5 + [False]
    families = _outputs(run, report, 4, FamiliesOutputs).families
    judging = {(row.tier, row.point): row.judging for row in families}
    records = _outputs(run, report, 5, RecordsOutputs)
    assert records.harbor
    qualified = [row for row in records.harbor if not row.diagnostic]
    assert {row.scenario_id for row in qualified} == {"urgent"}
    for row in records.harbor:
        assert row.task is not None
        task = load_executable_task(project, row.task)
        assert task.judge == judging[(row.tier, row.point)][row.scenario_id].judge
        binding = verify_harbor_export(Path(row.path))
        if row.diagnostic:
            assert row.failure is not None or row.refusals, row
            assert binding["qualification"] == "diagnostic_only"
            continue
        assert row.qualification is not None and row.failure is None and not row.refusals
        assert load_executable_qualification(project, row.qualification).qualified
        assert binding["qualification"] == "qualified_observable_contract"
        assert binding["qualification_evidence"] == row.qualification.model_dump(mode="json")
    routine = [row for row in records.harbor if row.scenario_id == "routine"]
    assert all(row.diagnostic and row.failure for row in routine)
    before = run.calls()
    assert [row.replayed for row in run.run(spec, approve=False).stages] == [True] * 6
    assert run.calls() == before


def test_a_training_schedule_drives_its_tier_and_leaves_the_others(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """The scheduled tier runs the schedule's warm-up rollout and records its completion."""
    run, first = pipeline
    project = run.project
    mining = _outputs(run, first, 3, MiningOutputs)
    base = ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(mining.tiers[0].report.artifact_id, "report.json")
    ).spec
    schedule = TrainingScheduleSpec(name="pipeline-schedule", base=base)
    spec = make_spec(run.fixture, run.rubric, run.bank, crosstalk=False, schedule=schedule)
    report = run.run(spec)
    assert report.complete
    assert [row.replayed for row in report.stages] == [True] * 3 + [False] * 3
    scheduled, other = _outputs(run, report, 3, MiningOutputs).tiers
    state = schedule_state(project, schedule)
    assert state.pending is None and len(state.completed) == 1
    (completed,) = state.completed
    assert completed.result == scheduled.report and completed.plan.kind == "warmup"
    assert completed.plan.spec is not None
    saved = ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(scheduled.report.artifact_id, "report.json")
    )
    assert saved.spec == completed.plan.spec and saved.spec != base
    assert scheduled.job == completed.plan.spec.initial
    assert scheduled.judge == completed.plan.spec.episode_judge
    # The other tier is not scheduled: its registration and report are unchanged.
    assert other.report == mining.tiers[1].report
    before = run.calls()
    assert [row.replayed for row in run.run(spec, approve=False).stages] == [True] * 6
    assert run.calls() == before
    assert len(schedule_state(project, schedule).completed) == 1


def test_agentic_roles_authored_rubric_and_learned_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default agentic selectors, critics and judge, an authored mining rubric, derived actions,
    and a rule book learned from the traces whose membership boundary is executed."""
    run = wired(tmp_path, monkeypatch, agentic=True)
    spec = make_spec(run.fixture, None, run.bank, actors=("actor",), crosstalk=False, agentic=True)
    assert spec.mining.selector_profile == spec.mining.critic_profile == AGENTIC_PROFILE
    assert spec.mining.judge_prompt_set == DEFAULT_AGENTIC_JUDGE
    assert spec.mining.rubric is None and not spec.authoring.actions
    report = run.run(spec)
    assert report.complete, report.stopped
    project = run.project
    # Rules: one real extraction draw per world-learning trace, then executed challenges.
    rules = _outputs(run, report, 1, RulesOutputs)
    stage = RulesStageReport.model_validate_json(
        project.artifacts.read_bytes(rules.report.artifact_id, "report.json")
    )
    # Rules are researched agentically per trace, weighed, then condensed into a memory.
    assert len(stage.research) == 2 and stage.memory is not None
    roles = {tool for tool, _ in run.wire.requests if tool.startswith("rules:")}
    assert roles >= {"rules:research", "rules:weighing", "rules:condensation"}
    assert not any(tool == EXTRACTION_TOOL for tool, _ in run.wire.requests)
    book = load_rulebook(project, rules.rulebook)
    assert set(book.contracts) == {"read_ticket", "check_sla", "escalate", "close"}
    (boundary,) = [rule for rule in book.rules if rule.predicate is not None]
    assert boundary.tool == "check_sla" and boundary.predicate is not None
    assert boundary.predicate.op == "in"
    assert stage.challenges is not None and rules.challenges == len(stage.challenges.runs) >= 2
    assert {run_.case.workload.expected for run_ in stage.challenges.runs} >= {
        "permitted",
        "refused",
    }
    assert rules.job.configuration["rule_book"] == rules.rulebook.model_dump(mode="json")
    # Error mining: the rubric was authored over a whole-task packet before any family.
    mining = _outputs(run, report, 3, MiningOutputs)
    assert mining.profile.prompt_version == DEFAULT_AGENTIC_JUDGE
    assert mining.rubric_packet is not None
    packet = load_packet(project, mining.rubric_packet)
    assert packet.brief.decision == WHOLE_TASK and not packet.brief.options
    assert [variant.variant_id for variant in packet.variants] == ["baseline"]
    assert [episode.episode_id for episode in packet.observed_episodes] == ["trace-1", "trace-2"]
    (tier,) = mining.tiers
    assert tier.judge.definition.rubric == mining.rubric
    assert tier.retrieval is not None
    saved = ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
    )
    assert saved.spec.selector_profile == saved.spec.critic_profile == AGENTIC_PROFILE
    # Agentic roles plan, read, draft and self check; their opening offers world-learning search.
    tools = [tool for tool, _ in run.wire.requests]
    assert {PLAN_TOOL.name, FREE_CHOICE, "choose_points", "estimate_points"} <= set(tools)
    plans = [
        payload
        for tool, payload in run.wire.requests
        if tool == PLAN_TOOL.name and not any(n in json.dumps(payload) for n in _ROLE_DRAWS)
    ]
    assert plans and all("world-learning-retrieval" in json.dumps(p) for p in plans)
    # The records stage derived options at every continued point and classified its choices.
    assert {OPTIONS_TOOL, CHOICE_TOOL} <= set(tools)
    # The agentic judge planned, checked and self checked before every assessment.
    assert run.wire.judged
    assert {
        "record_plan",
        "record_criterion_checks",
        "record_self_check",
    } <= set(run.wire.phases)
    assert run.wire.phases.count("record_self_check") == len(run.wire.judged)
    # Families: option actions come from the decision-state author's executable calls.
    (family,) = _outputs(run, report, 4, FamiliesOutputs).families
    assert family.failure is None, family.failure
    assert family.report is not None
    final = FamilyReport.model_validate_json(
        project.artifacts.read_bytes(family.report.artifact_id, "report.json")
    )
    assert final.actions is not None
    assert {key: call.name for key, call in final.actions.items()} == {
        "escalate": "escalate",
        "close": "close",
    }
    before = run.calls()
    assert [row.replayed for row in run.run(spec, approve=False).stages] == [True] * 6
    assert run.calls() == before
