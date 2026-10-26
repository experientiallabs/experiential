"""Protected development revision of proposed reconstructions, without native promotion."""

import json
from collections.abc import Callable
from datetime import datetime
from importlib.resources import files
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    WorkflowEvidence,
    mechanical_checks,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.reconstruction.models import Claim, Interface, RawInputs, ReconstructionBundle


class ProtectedCase(ContractModel):
    """Frozen development regression actions, references and planned denominator.

    Protection means earlier correct behavior must not regress. These are development
    cases whose outcome summaries may guide repair, not an independent confirmation holdout.

    Attributes:
        case_id: Logical workload identity shared by both candidates.
        references_sha256: Exact independent development reference inventory.
        actions_sha256: Ordered public action inventory, without native answers.
        planned_steps: All scheduled transition slots, including blocked positions.
    """

    case_id: str
    references_sha256: Sha256
    actions_sha256: Sha256
    planned_steps: int = Field(gt=0, strict=True)


class RevisionPlan(ContractModel):
    """Parent, admitted evidence and protected-suite identity for one bounded revision.

    Attributes:
        run_id: Unique revision attempt; consumed IDs never regenerate automatically.
        parent_sha256: Frozen parent behavioral artifact.
        inputs_sha256: Exact original inputs plus admitted append-only evidence.
        cases: Protected development cases inaccessible to confirmation selection.
        minimum_gains: Required false-to-true outcomes; unavailable checks are not gains.
        require_complete: Additionally require every protected outcome to be exactly true.
        deadline: Latest development selection decision.
        research_history_sha256: Exact note treatment, absent when notes are disabled.
    """

    run_id: str
    parent_sha256: Sha256
    inputs_sha256: Sha256
    cases: tuple[ProtectedCase, ...] = Field(min_length=1)
    minimum_gains: int = Field(default=1, ge=1, strict=True)
    require_complete: bool = Field(default=False, strict=True)
    deadline: AwareDatetime
    research_history_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        """Require distinct protected case identities before dispatch."""
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate protected case")
        return self


class ClaimReplacement(ContractModel):
    """An explicit correction bound to the exact immutable parent proposition.

    Attributes:
        index: Zero-based position in the parent claim inventory.
        expected_sha256: Digest of that original claim, preventing a stale patch.
        replacement: Corrected proposition validated against admitted sources.
    """

    index: int = Field(ge=0, strict=True)
    expected_sha256: Sha256
    replacement: Claim


class RevisionPatch(ContractModel):
    """One precise textual change and its source-backed claim additions.

    Attributes:
        old_text: Exact unique span of the parent rules.
        new_text: Replacement semantic content.
        replace_claims: Exact parent claims superseded without retaining contradictions.
        append_claims: Additional propositions citing only admitted raw sources.
        python_source: Optional complete source replacement, never executed here.
        interface: Optional explicitly evaluated schema revision.
    """

    old_text: str = Field(min_length=1)
    new_text: str
    replace_claims: tuple[ClaimReplacement, ...] = ()
    append_claims: tuple[Claim, ...] = ()
    python_source: str | None = None
    interface: Interface | None = None


def _revised_claims(parent: tuple[Claim, ...], patch: RevisionPatch) -> tuple[Claim, ...]:
    """Apply identity-bound corrections and additions without changing parent history."""
    claims = list(parent)
    seen: set[int] = set()
    for change in patch.replace_claims:
        if (
            change.index in seen
            or change.index >= len(parent)
            or sha256_json(parent[change.index].model_dump(mode="json")) != change.expected_sha256
        ):
            raise ValueError("Replacement does not identify a unique exact parent claim")
        seen.add(change.index)
        claims[change.index] = change.replacement
    return (*claims, *patch.append_claims)


class ResearchNote(ContractModel):
    """Development interpretation, never an original native observation.

    Attributes:
        id: Unique investigation note identity.
        phase: Development only; confirmation cannot enter this channel.
        content: Hypotheses, rejected approaches or tested lessons.
        investigation_sha256: Locator digest for the underlying investigation evidence.
        source_trace_ids: Original episodes exposed to this investigation.
        disposition: Explicit status; a rejected hypothesis never becomes an accepted rule.
    """

    id: str = Field(min_length=1)
    phase: Literal["development"] = "development"
    content: str = Field(min_length=1)
    investigation_sha256: Sha256
    source_trace_ids: tuple[str, ...] = ()
    disposition: Literal["accepted", "rejected", "unresolved"] = "unresolved"


class RevisionEvaluation(ContractModel):
    """Complete paired evidence, references and dual result for one frozen candidate.

    Attributes:
        knowledge_sha256: Exact candidate evaluated.
        evidence: Its actual carried predictions and all planned positions.
        references: Separate independent development truth.
        report: Mechanical measurements and required judge status tied to these inputs.
    """

    knowledge_sha256: Sha256
    evidence: WorkflowEvidence
    references: EvaluationReferences
    report: DualReport


def _outcomes(
    receipt: RevisionEvaluation, bundle: ReconstructionBundle, case: ProtectedCase
) -> dict[str, bool | None]:
    """Recompute semantic selection outcomes without crediting recovered availability.

    Reports retain failures in their planned denominators. Only valid predictions can
    provide a false-to-true semantic gain; an outage or blocked suffix is unavailable
    for this separate development selection decision.
    """
    evidence, references, report = receipt.evidence, receipt.references, receipt.report
    mechanical = mechanical_checks(evidence, references)
    if (
        receipt.knowledge_sha256 != bundle.digest
        or evidence.knowledge_sha256 != bundle.digest
        or report.knowledge_sha256 != bundle.digest
        or evidence.workflow_id != case.case_id
        or report.workflow_id != case.case_id
        or len(evidence.steps) != case.planned_steps
        or sha256_json([s.action for s in evidence.steps]) != case.actions_sha256
        or sha256_json(references.model_dump(mode="json")) != case.references_sha256
        or report.evidence_sha256 != sha256_json(evidence.model_dump(mode="json"))
        or report.reference_sha256 != case.references_sha256
        or report.planned_steps != case.planned_steps
        or sha256_json(report.mechanical) != sha256_json(mechanical)
    ):
        raise ValueError("Protected evaluation identity, inventory or mechanical result differs")
    outcomes: dict[str, bool | None] = {}
    rows = mechanical["steps"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        for key in ("trajectory_reply_exact", "trajectory_state_exact"):
            value = row[key] if row["status"] == "valid" else None
            if value is not None and not isinstance(value, bool):
                raise ValueError("protected outcome is not boolean or unavailable")
            outcomes[f"{case.case_id}/{row['step']}/{key}"] = value
    whole = (
        mechanical["workflow_joint_exact"]
        if all(step.status == "valid" for step in evidence.steps)
        else None
    )
    if whole is not None and not isinstance(whole, bool):
        raise ValueError("invalid whole-workflow outcome")
    outcomes[case.case_id + "/whole"] = whole
    return outcomes


def _gate(
    before: dict[str, bool | None],
    after: dict[str, bool | None],
    plan: RevisionPlan,
    judge_ok: bool,
) -> tuple[bool, list[str], list[str]]:
    """Select gains without regressions or unavailable-to-success substitutions."""
    gained = [k for k in before if before[k] is False and after[k] is True]
    lost = [k for k in before if before[k] is True and after[k] is not True]
    available = [k for k in after if after[k] is not None]
    selected = bool(available) and judge_ok and not lost and len(gained) >= plan.minimum_gains
    if plan.require_complete:
        selected = (
            selected and len(available) == len(after) and all(after[k] is True for k in after)
        )
    return selected, gained, lost


def _revision_feedback(receipt: RevisionEvaluation) -> JsonObject:
    """Describe development failures without revealing protected instance or reference values."""
    return {
        "case_id": receipt.evidence.workflow_id,
        "planned_steps": receipt.report.planned_steps,
        "outcomes": receipt.report.mechanical["steps"],
        "workflow_joint_exact": receipt.report.mechanical["workflow_joint_exact"],
        "first_semantic_divergence": receipt.report.mechanical["first_semantic_divergence"],
        "first_hard_stop": receipt.report.mechanical["first_hard_stop"],
        "judge_status": receipt.report.judge_status,
        "judge_axes": [
            {"dimension_id": axis.dimension_id, "status": axis.status, "score": axis.score}
            for axis in receipt.report.axes
        ],
    }


def revise_reconstruction(
    parent: ReconstructionBundle,
    admitted_inputs: RawInputs,
    plan: RevisionPlan,
    *,
    artifacts: ArtifactStore,
    complete: Callable[[JsonObject, str], JsonObject],
    evaluate: Callable[[ReconstructionBundle, ProtectedCase], RevisionEvaluation],
    clock: Callable[[], datetime],
    code_revision: str,
    research_history: tuple[ResearchNote, ...] = (),
) -> str:
    """Run one revision and paired protected evaluations; preserve every attempted phase."""
    parent = ReconstructionBundle.model_validate_json(parent.model_dump_json())
    notes = tuple(ResearchNote.model_validate(n.model_dump(mode="json")) for n in research_history)
    note_data = [n.model_dump(mode="json") for n in notes]
    if len({n.id for n in notes}) != len(notes):
        raise ValueError("Duplicate research note identity")
    if (sha256_json(note_data) if notes else None) != plan.research_history_sha256:
        raise ValueError("Research history differs from the registered treatment")
    admitted_inputs = RawInputs.model_validate_json(admitted_inputs.model_dump_json())
    if (
        parent.digest != plan.parent_sha256
        or sha256_json(admitted_inputs.model_dump(mode="json")) != plan.inputs_sha256
    ):
        raise ValueError("Registration does not bind parent and admitted inputs")
    old, new = parent.sources, admitted_inputs
    if (
        new.episodes[: len(old.episodes)] != old.episodes
        or new.description != old.description
        or new.published_contracts != old.published_contracts
        or new.mode != old.mode
    ):
        raise ValueError("Admitted source expansion must only append immutable episodes")
    base = stable_id("reconstruction-learning", {"run_id": plan.run_id})
    parent_manifest = artifacts.read(parent.artifact_id).manifest
    if json.loads(artifacts.read_bytes(parent.artifact_id, "artifact.json")) != parent.model_dump(
        mode="json"
    ):
        raise ValueError("Parent artifact bytes differ")
    refs = (artifact_input(parent_manifest),)

    def save(label: str, value: JsonObject) -> str:
        """Keep every development phase as an immutable child of the parent artifact."""
        identity = base + "-" + label
        artifacts.write_json(
            artifact_id=identity,
            artifact_type="reconstruction-development",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=clock(), inputs=refs, code_revision=code_revision
            ),
            files={"record.json": value},
        )
        return identity

    def invoke(
        label: str, request: JsonObject, callback: Callable[[], ContractModel | JsonObject]
    ) -> ContractModel | JsonObject:
        """Retain failures and late responses without silently repeating a phase."""
        if clock() >= plan.deadline:
            raise TimeoutError("Development revision deadline reached")
        save(label + "-intent", request)
        try:
            result = callback()
            value = result.model_dump(mode="json") if isinstance(result, ContractModel) else result
            save(label + "-result", value)
        except Exception as error:  # noqa: BLE001 - preserve injected evaluator/provider failures.
            save(
                label + "-failure",
                {
                    "error_type": type(error).__name__,
                    "logical_scope_consumed": True,
                    "dispatch_status": "callback_owned",
                },
            )
            raise
        if clock() >= plan.deadline:
            raise TimeoutError("Late result preserved without selection")
        return result

    registration_id = save(
        "registration",
        {
            "phase": "development",
            "plan": plan.model_dump(mode="json"),
            "parent": parent.model_dump(mode="json"),
            "admitted_inputs": admitted_inputs.model_dump(mode="json"),
            "research_history": note_data,
        },
    )
    before: dict[str, bool | None] = {}
    evaluations = []
    for case in plan.cases:
        receipt = RevisionEvaluation.model_validate(
            invoke(
                "parent-" + case.case_id,
                {"knowledge_sha256": parent.digest, "case": case.model_dump(mode="json")},
                lambda case=case: evaluate(parent.model_copy(deep=True), case),
            )
        )
        before.update(_outcomes(receipt, parent, case))
        evaluations.append(receipt)
    request: JsonObject = {
        "messages": [
            {
                "role": "system",
                "content": files("exp.simulation.reconstruction")
                .joinpath("prompts/reconstructed-revision.txt")
                .read_text(encoding="utf-8"),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "parent": parent.model_dump(mode="json"),
                        "admitted_inputs": admitted_inputs.model_dump(mode="json"),
                        "claim_source_ids": list(admitted_inputs.source_map()),
                        "parent_claim_sha256": [
                            sha256_json(c.model_dump(mode="json")) for c in parent.claims
                        ],
                        "development_feedback": [_revision_feedback(r) for r in evaluations],
                        "research_history": note_data,
                        "output_schema": RevisionPatch.model_json_schema(),
                    },
                    allow_nan=False,
                ),
            },
        ]
    }
    patch = RevisionPatch.model_validate(
        invoke("revision", request, lambda: complete(request, base + "-revision"))
    )
    if parent.text_rules.count(patch.old_text) != 1:
        raise ValueError("Revision must match exactly one parent text span")
    data = parent.model_dump(mode="json")
    data.update(
        artifact_id=base + "-child",
        sources=admitted_inputs.model_dump(mode="json"),
        inputs_sha256=plan.inputs_sha256,
        text_rules=parent.text_rules.replace(patch.old_text, patch.new_text, 1),
        claims=[c.model_dump(mode="json") for c in _revised_claims(parent.claims, patch)],
    )
    if patch.python_source is not None:
        data["python_source"] = patch.python_source
    if patch.interface is not None:
        if parent.python_source is not None and patch.python_source is None:
            raise ValueError("Interface revision requires a corresponding Python replacement")
        data["interface"] = patch.interface.model_dump(mode="json")
    child = ReconstructionBundle.model_validate(data)
    artifacts.write_json(
        artifact_id=child.artifact_id,
        artifact_type="reconstruction",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=clock(),
            inputs=tuple(
                sorted(
                    (
                        *refs,
                        artifact_input(artifacts.read(registration_id).manifest),
                        artifact_input(artifacts.read(base + "-revision-result").manifest),
                    ),
                    key=lambda item: item.artifact_id,
                )
            ),
            code_revision=code_revision,
        ),
        files={"artifact.json": child.model_dump(mode="json")},
    )
    after: dict[str, bool | None] = {}
    child_evaluations: list[RevisionEvaluation] = []
    for case in plan.cases:
        receipt = RevisionEvaluation.model_validate(
            invoke(
                "child-" + case.case_id,
                {"knowledge_sha256": child.digest, "case": case.model_dump(mode="json")},
                lambda case=case: evaluate(child.model_copy(deep=True), case),
            )
        )
        after.update(_outcomes(receipt, child, case))
        child_evaluations.append(receipt)
    judge_ok = all(r.report.judge_status == "completed" for r in (*evaluations, *child_evaluations))
    selected, gained, lost = _gate(before, after, plan, judge_ok)
    return save(
        "decision",
        {
            "phase": "development",
            "authority": "proposed_reconstruction_only",
            "selected": selected,
            "gained": gained,
            "lost": lost,
            "plan": plan.model_dump(mode="json"),
            "parent_artifact_id": parent.artifact_id,
            "child_artifact_id": child.artifact_id,
            "child_sha256": child.digest,
            "parent_reports": [r.model_dump(mode="json") for r in evaluations],
            "child_reports": [r.model_dump(mode="json") for r in child_evaluations],
        },
    )


def load_selected_reconstruction(
    artifacts: ArtifactStore, decision_id: str
) -> ReconstructionBundle:
    """Recheck a persisted development selection; never grant native qualification."""
    decision = json.loads(artifacts.read_bytes(decision_id, "record.json"))
    plan = RevisionPlan.model_validate(decision["plan"])
    parent = ReconstructionBundle.model_validate_json(
        artifacts.read_bytes(decision["parent_artifact_id"], "artifact.json")
    )
    child = ReconstructionBundle.model_validate_json(
        artifacts.read_bytes(decision["child_artifact_id"], "artifact.json")
    )
    if (
        parent.digest != plan.parent_sha256
        or child.digest != decision["child_sha256"]
        or child.inputs_sha256 != plan.inputs_sha256
        or decision["phase"] != "development"
        or decision["authority"] != "proposed_reconstruction_only"
        or child.sources.episodes[: len(parent.sources.episodes)] != parent.sources.episodes
        or child.sources.description != parent.sources.description
        or child.sources.published_contracts != parent.sources.published_contracts
        or child.sources.mode != parent.sources.mode
    ):
        raise ValueError("Selection artifact binding differs")
    before, after, receipts = {}, {}, []
    for key, bundle, outcomes in (
        ("parent_reports", parent, before),
        ("child_reports", child, after),
    ):
        if len(decision[key]) != len(plan.cases):
            raise ValueError("Incomplete protected inventory")
        for case, raw in zip(plan.cases, decision[key], strict=True):
            receipt = RevisionEvaluation.model_validate(raw)
            outcomes.update(_outcomes(receipt, bundle, case))
            receipts.append(receipt)
    selected, gained, lost = _gate(
        before, after, plan, all(r.report.judge_status == "completed" for r in receipts)
    )
    if (
        not selected
        or decision["selected"] is not True
        or decision["gained"] != gained
        or decision["lost"] != lost
    ):
        raise ValueError("No valid protected development selection")
    return child
