"""Glue between tag admission and the per-rule Markdown memory set.

Under the default ``rule_tags`` admission of :mod:`~exp.optimize.evaluation.rules.memory_stage`,
the admission verdict decides every seed rule's status, and the weigher writes the rationale for
that verdict. This module turns those into memory edits and keeps the verdict authoritative:

- :func:`admission_proposals`: one host ``patch`` per seed rule setting its status from the
  verdict, with the verdict's summary (supporting, contradicting, uncertain trajectories and
  independent lineages) followed by the weigher's rationale as the status rationale;
- :func:`research_notes`: System Two's notes per topic, read by the consolidator;
- :func:`unadmitted_proposals`: after governance, a host ``patch`` returning to ``unknown`` any
  rule that is ``supported`` without an admission of its current truism (a rule an inspector
  added, raised or rewrote), so support always means admission;
- :func:`save_guard_table`: the family's action-guard and invariant table from admitted rules;
- :func:`final_lifecycles` and :func:`save_lifecycles`: every rule's declared lifecycle moved
  through consolidation and governance by the transition table, persisted with its statechart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from exp.optimize.evaluation.rules.memory_edits import (
    FilePatch,
    PatchField,
    PatchOp,
    Proposal,
    RuleFields,
)
from exp.optimize.evaluation.rules.rule_consolidation import ConsolidationDraw
from exp.optimize.evaluation.rules.rule_tags import RuleAdmissionReport
from exp.optimize.evaluation.rules.system_two import TopicResearch
from exp.optimize.evaluation.rules.weighing import WeighingDraw
from exp.simulation.rules.admission import GuardTable, guard_table
from exp.simulation.rules.memory_set import MemorySet, rule_sha256
from exp.simulation.rules.rule_lifecycle import RuleLifecycle, RuleState, move, statechart

ADMISSION_AUTHOR = "tag_admission"
"""Author name of host status edits decided by tag admission."""

ADMISSION_PERMISSION: tuple[frozenset[PatchOp], frozenset[PatchField]] = (
    frozenset(("patch",)),
    frozenset(("status", "status_rationale")),
)

GUARD_TABLE_TYPE = "rule-guard-table"
"""Artifact type of a saved family guard table."""

GUARDS_JSON = "guards.json"

LIFECYCLE_TYPE = "rule-lifecycle"
"""Artifact type of saved rule lifecycles and their statechart."""

LIFECYCLES_JSON = "lifecycles.json"
STATECHART_MMD = "statechart.mmd"


def admission_proposals(
    memory: MemorySet, report: RuleAdmissionReport, weighed: Sequence[WeighingDraw]
) -> list[tuple[str, str, Proposal]]:
    """One status edit per admitted-or-not seed rule, the verdict first and the why after it.

    Args:
        memory: The seed set.
        report: The admission run over it.
        weighed: The weigher's rationale draws.

    Returns:
        ``(author, variant, proposal)`` in set order.
    """
    rationales = {draw.rule_id: draw.weighing for draw in weighed if draw.weighing is not None}
    found: list[tuple[str, str, Proposal]] = []
    for record in report.rules:
        verdict = record.verdict
        rule = memory.rule(verdict.rule_id)
        if rule is None:
            continue
        weighing = rationales.get(rule.rule_id)
        text = verdict.summary()
        if weighing is not None:
            text = f"{text}\n\n{weighing.rationale}"
        citations = () if weighing is None else (*weighing.supporting, *weighing.contradicting)
        found.append(
            (
                ADMISSION_AUTHOR,
                "",
                Proposal(
                    title=f"Admit {rule.rule_id}: {verdict.status}",
                    rationale=verdict.reason,
                    citations=citations[:32] or (rule.rule_id,),
                    patches=(
                        FilePatch(
                            op="patch",
                            path=rule.path,
                            base_sha256=rule_sha256(rule),
                            fields=RuleFields(status=verdict.status, status_rationale=text[:4000]),
                        ),
                    ),
                ),
            )
        )
    return found


def research_notes(research: Sequence[TopicResearch]) -> dict[str, JsonValue]:
    """System Two's notes per topic: objectives asked, rules drafted, what stayed open."""
    notes: dict[str, list[JsonValue]] = {}
    for item in research:
        answer = item.answer
        notes.setdefault(item.topic, []).append(
            {
                "variant": item.variant,
                "round": item.round,
                "objectives": [o.question for o in item.objectives],
                "drafted": [] if answer is None else [r.truism for r in answer.rules],
                "open_questions": [] if answer is None else list(answer.open_questions),
                "limitations": [] if answer is None else list(answer.limitations),
                "failure": item.failure,
            }
        )
    return {topic: rows for topic, rows in notes.items()}


def admitted_statements(report: RuleAdmissionReport) -> dict[str, str]:
    """The truism each admitted rule was compiled from, by rule."""
    return {r.compiled.rule_id: r.compiled.truism for r in report.rules if r.verdict.admitted}


def unadmitted_proposals(
    memory: MemorySet, report: RuleAdmissionReport
) -> list[tuple[str, str, Proposal]]:
    """Return ``supported`` rules without an admission of their current truism to ``unknown``."""
    admitted = admitted_statements(report)
    found: list[tuple[str, str, Proposal]] = []
    for rule in memory.rules:
        if rule.status != "supported" or admitted.get(rule.rule_id) == rule.truism:
            continue
        why = (
            "Its truism changed after admission, so its compiled tags no longer describe it."
            if rule.rule_id in admitted
            else "It was never admitted by corpus-wide decision tags."
        )
        found.append(
            (
                ADMISSION_AUTHOR,
                "",
                Proposal(
                    title=f"Not admitted: {rule.rule_id}",
                    rationale=why,
                    citations=(rule.rule_id,),
                    patches=(
                        FilePatch(
                            op="patch",
                            path=rule.path,
                            base_sha256=rule_sha256(rule),
                            fields=RuleFields(
                                status="unknown",
                                status_rationale=(
                                    f"Tag admission: not admitted. {why}\n\n{rule.status_rationale}"
                                )[:4000],
                            ),
                        ),
                    ),
                ),
            )
        )
    return found


def family_guard_table(family: str, memory: MemorySet, report: RuleAdmissionReport) -> GuardTable:
    """The family's guard table from the final set and the admission run."""
    compiled = {r.compiled.rule_id: r.compiled.questions for r in report.rules}
    return guard_table(
        family,
        memory.tools,
        memory.rules,
        report.verdicts(),
        compiled,
        admitted_statements(report),
        report.policy,
    )


def save_guard_table(
    project: ProjectStore,
    table: GuardTable,
    *,
    identity: str,
    inputs: Sequence[ArtifactInput],
    code_revision: str,
) -> ArtifactInput:
    """Save a guard table as one artifact, or adopt the identical one already saved.

    Raises:
        ValueError: An artifact with this identity holds a different table.
    """
    store = project.artifacts
    artifact_id = f"{identity}-guards-{sha256_json(table.model_dump(mode='json'))[:16]}"
    if not store.exists(artifact_id):
        try:
            manifest = store.write_json(
                artifact_id=artifact_id,
                artifact_type=GUARD_TABLE_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
                ),
                files={GUARDS_JSON: table},
            )
        except ArtifactAlreadyExistsError:
            pass
        else:
            return artifact_input(manifest)
    stored = store.read(artifact_id)
    saved = GuardTable.model_validate_json(stored.payloads[GUARDS_JSON])
    if stored.manifest.artifact_type != GUARD_TABLE_TYPE or saved != table:
        raise ValueError(f"saved guard table {artifact_id} differs from this table")
    return artifact_input(stored.manifest)


def load_guard_table(project: ProjectStore, reference: ArtifactInput) -> GuardTable:
    """Read one pinned guard table.

    Raises:
        ValueError: The manifest digest changed or the artifact is not a guard table.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"guard table {reference.artifact_id} manifest digest changed")
    if stored.manifest.artifact_type != GUARD_TABLE_TYPE:
        raise ValueError(f"artifact {reference.artifact_id} is not a {GUARD_TABLE_TYPE}")
    return GuardTable.model_validate_json(stored.payloads[GUARDS_JSON])


def verdict_briefs(report: RuleAdmissionReport) -> Mapping[str, dict[str, JsonValue]]:
    """Each verdict as the rationale writer reads it, without the per-trajectory rows."""
    return {
        r.verdict.rule_id: r.verdict.model_dump(mode="json", exclude={"trajectories"})
        for r in report.rules
    }


def final_lifecycles(
    report: RuleAdmissionReport,
    consolidations: Sequence[ConsolidationDraw],
    table: GuardTable,
) -> tuple[RuleLifecycle, ...]:
    """Every rule's lifecycle after consolidation and governance, each move checked.

    Raises:
        IllegalTransitionError: A recorded outcome does not fit the transition table.
    """
    drawn = {draw.rule_id: draw for draw in consolidations}
    withdrawn = {w.rule_id: w.reason for w in table.withdrawn}
    found: list[RuleLifecycle] = []
    for record in report.rules:
        lifecycle = record.lifecycle
        draw = drawn.get(lifecycle.rule_id)
        if draw is not None:
            target = RuleState.CONSOLIDATED if draw.answer else RuleState.CONSOLIDATION_FAILED
            lifecycle = move(lifecycle, "consolidate", target, draw.failure or draw.variant)
        if lifecycle.rule_id in withdrawn:
            lifecycle = move(lifecycle, "govern", RuleState.WITHDRAWN, withdrawn[lifecycle.rule_id])
        found.append(lifecycle)
    return tuple(found)


def save_lifecycles(
    project: ProjectStore,
    lifecycles: Sequence[RuleLifecycle],
    *,
    identity: str,
    inputs: Sequence[ArtifactInput],
    code_revision: str,
) -> ArtifactInput:
    """Save the lifecycles with the generated statechart, or adopt the identical saved ones."""
    store = project.artifacts
    payload = [lifecycle.model_dump(mode="json") for lifecycle in lifecycles]
    artifact_id = f"{identity}-lifecycles-{sha256_json(payload)[:16]}"
    if store.exists(artifact_id):
        return artifact_input(store.read(artifact_id).manifest)
    files = {
        LIFECYCLES_JSON: json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n",
        STATECHART_MMD: statechart().encode(),
    }
    try:
        manifest = store.write(
            artifact_id=artifact_id,
            artifact_type=LIFECYCLE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
            ),
            files=files,
        )
    except ArtifactAlreadyExistsError:
        manifest = store.read(artifact_id).manifest
    return artifact_input(manifest)
