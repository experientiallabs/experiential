"""Bind analyst tool-trust inputs to saved held-out fidelity evaluations."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, sha256_json
from exp.common.evaluations.reconstruction import DualReport, WorkflowEvidence, mechanical_checks
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import refuse_known_ineligible_sources
from exp.optimize.evaluation.fidelity.memory import refuse_ineligible_memory
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity


class _Step(ContractModel):
    """Exact mechanical row emitted by the maintained reconstruction evaluator."""

    step: int = Field(ge=0, strict=True)
    status: Literal["valid", "unknown", "unsupported", "failure", "blocked", "unattempted"]
    reference_context_matches: bool = Field(strict=True)
    reply_exact: bool | None = Field(strict=True)
    trajectory_reply_exact: bool | None = Field(strict=True)
    trajectory_state_exact: bool | None = Field(strict=True)
    state_exact: bool | None = Field(strict=True)


def load_tool_fidelity(
    project: ProjectStore,
    plans: Sequence[ArtifactInput],
    world_model: ArtifactInput,
    excluded_task_lineages: Sequence[str],
) -> JsonObject:
    """Resolve held-out tool measurements without pooling serving or memory conditions.

    Args:
        project: Immutable plans and reports owned by the existing evaluation project.
        plans: Manifest-bound completed ``exp eval --fidelity`` registrations.
        world_model: Exact simulator artifact used by the actor council.
        excluded_task_lineages: Actor task lineages that cannot calibrate their own analysis.

    Returns:
        One profile per saved plan, with planned denominators and reference limits.

    Raises:
        ValueError: A digest, simulator, held-out boundary, report or denominator differs.
    """
    if len({plan.artifact_id for plan in plans}) != len(plans):
        raise ValueError("duplicate fidelity plans would count the same measurement twice")
    refuse_known_ineligible_sources(project.artifacts, tuple(plans), maximum_artifacts=100_000)
    profiles: list[JsonObject] = []
    excluded = set(excluded_task_lineages)
    for reference in plans:
        stored = project.artifacts.read(reference.artifact_id)
        if (
            artifact_input(stored.manifest) != reference
            or stored.manifest.artifact_type != "fidelity-plan"
        ):
            raise ValueError("fidelity plan identity or manifest digest differs")
        prepared = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(reference.artifact_id, "plan.json")
        )
        spec = prepared.spec
        if spec.memory is not None:
            refuse_ineligible_memory(
                project, spec.memory.lineage, checkpoint=spec.memory.checkpoint
            )
        if prepared.id != reference.artifact_id or spec.world_model != world_model:
            raise ValueError("fidelity plan does not bind the selected simulator")
        if spec.task.partition != "held_out":
            raise ValueError("tool fidelity requires an explicitly held-out task")
        if spec.task.lineage_group_id in excluded:
            raise ValueError("fidelity task lineage overlaps an actor council task")
        if spec.online_research is not None or (
            spec.memory is not None and (spec.memory.mode != "frozen" or not spec.memory.reserved)
        ):
            raise ValueError("tool fidelity requires frozen held-out evaluation without learning")
        report_id = prepared.id + "-report"
        if not project.artifacts.exists(report_id):
            raise ValueError("fidelity plan has no completed saved report")
        report_stored = project.artifacts.read(report_id)
        if report_stored.manifest.artifact_type != "workflow-dual-evaluation":
            raise ValueError("saved fidelity report has a different artifact type")
        refuse_known_ineligible_sources(
            project.artifacts, (artifact_input(report_stored.manifest),), maximum_artifacts=100_000
        )
        result = DualReport.model_validate_json(
            project.artifacts.read_bytes(report_id, "report.json")
        )
        if (
            result.workflow_id != spec.workload.id
            or result.reference_sha256 != sha256_json(spec.references)
            or result.judge_config_sha256 != sha256_json(spec.judge)
            or result.planned_steps != len(spec.workload.actions)
            or result.mechanical.get("planned_steps") != len(spec.workload.actions)
        ):
            raise ValueError("saved fidelity report identity or planned denominator differs")
        intent_id = report_id + "-intent"
        if not project.artifacts.exists(intent_id):
            raise ValueError("saved fidelity report has no immutable evaluation intent")
        intent = project.artifacts.read(intent_id)
        if report_stored.manifest.inputs != (artifact_input(intent.manifest),):
            raise ValueError("saved fidelity report is not bound to its evaluation intent")
        evidence = WorkflowEvidence.model_validate_json(
            project.artifacts.read_bytes(intent_id, "evidence.json")
        )
        if (
            result.evidence_sha256 != sha256_json(evidence)
            or tuple(step.action for step in evidence.steps)
            != tuple(
                {"name": action.name, "arguments": action.arguments}
                for action in spec.workload.actions
            )
            or result.mechanical != mechanical_checks(evidence, spec.references)
        ):
            raise ValueError("saved fidelity report differs from its raw evaluation evidence")
        rows = result.mechanical.get("steps")
        if not isinstance(rows, list) or len(rows) != len(spec.workload.actions):
            raise ValueError("saved fidelity report does not cover every planned action")
        steps = tuple(_Step.model_validate(row) for row in rows)
        if tuple(step.step for step in steps) != tuple(range(len(rows))):
            raise ValueError("saved fidelity report step inventory differs from the workload")
        tools: dict[str, JsonObject] = {}
        for action, step in zip(spec.workload.actions, steps, strict=True):
            counts = tools.setdefault(action.name, _empty_counts(action.name))
            _increment(counts, "planned")
            _increment(counts, "valid", step.status == "valid")
            _increment(counts, "failed", step.status in ("failure", "unknown", "unsupported"))
            _increment(counts, "blocked", step.status == "blocked")
            _increment(counts, "unattempted", step.status == "unattempted")
            _increment(counts, "archived_reply_assessed", step.trajectory_reply_exact is not None)
            _increment(counts, "archived_reply_exact", step.trajectory_reply_exact is True)
            _increment(counts, "context_reply_assessed", step.reply_exact is not None)
            _increment(counts, "context_reply_exact", step.reply_exact is True)
            _increment(counts, "state_assessed", step.state_exact is not None)
            _increment(counts, "state_exact", step.state_exact is True)
        profiles.append(
            {
                "profile_id": prepared.id,
                "plan": reference.model_dump(mode="json"),
                "report": artifact_input(report_stored.manifest).model_dump(mode="json"),
                "comparison_scope": {
                    "reference_capability": spec.references.capability,
                    "task_lineage": spec.task.lineage_group_id,
                    "knowledge_and_serving_artifact": world_model.model_dump(mode="json"),
                    "memory": None if spec.memory is None else spec.memory.model_dump(mode="json"),
                    "retrieval": None
                    if spec.retrieval is None
                    else spec.retrieval.model_dump(mode="json"),
                    "roles": None if spec.roles is None else spec.roles.model_dump(mode="json"),
                    "output_protocol": spec.output_protocol,
                    "json_object_output": spec.json_object_output,
                    "catalog_sha256": prepared.catalog_sha256,
                    "pooling": "none; this profile contains one saved evaluation plan",
                },
                "tools": [tools[name] for name in sorted(tools)],
            }
        )
    return {
        "profiles": profiles,
        "limitations": [
            "Archived reply agreement does not establish native state or transition fidelity.",
            "Context-qualified reply and state measurements require matching recorded prestates.",
            "Counts retain failed, blocked and unattempted slots; small samples remain uncertain.",
            "Serving and memory profiles remain separate; no pooled reliability is inferred.",
            "No native environment was accessed while resolving these saved measurements.",
        ],
    }


def _empty_counts(tool: str) -> JsonObject:
    """Initialize every count, including unavailable planned outcomes."""
    return {
        "tool": tool,
        "planned": 0,
        "valid": 0,
        "failed": 0,
        "blocked": 0,
        "unattempted": 0,
        "archived_reply_assessed": 0,
        "archived_reply_exact": 0,
        "context_reply_assessed": 0,
        "context_reply_exact": 0,
        "state_assessed": 0,
        "state_exact": 0,
    }


def _increment(counts: JsonObject, key: str, include: bool = True) -> None:
    """Advance a typed count without treating unavailable values as successful outcomes."""
    if include:
        value = counts[key]
        if not isinstance(value, int):
            raise ValueError("tool fidelity counter is not an integer")
        counts[key] = value + 1
