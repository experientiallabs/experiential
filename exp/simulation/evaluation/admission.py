"""Verified label witnesses and named refusals that gate label admission and qualification.

A witnessed label is admitted only by re-deriving its saved witness from the artifact store,
never by a caller's approval. Two witness kinds exist:

- ``specification_check``: an independent saved specification graded on the labeled outcome;
- ``native_execution``: the same kind of specification graded on the observation of a saved
  native materialization receipt whose input bytes, adapter digest, edit plan and output bytes
  are re-checked, whose output bytes and projection a maintained adapter re-derives from the
  input bytes, and whose edit plan is exactly the one the labeled outcome proposed; or a
  record receipt re-executed under an explicitly supplied native runtime, with exact actions,
  observations and checkpoints and an explicit observed-goal predicate. Neither proves prose.

Every refusal carries one named reason, so qualification and paid work stop with a cause
the caller can act on:

- ``self_approval``: a generated label approves itself, directly or because its specification
  is, copies, or derives from generated or unclassified artifacts;
- ``unverifiable_witness``: a witness is missing, differs from its digest, or re-derives a
  different observation or verdict;
- ``contaminated_lineage``: an exposed, reserved or unresolved lineage, or a task whose own
  evidence reaches the simulator's knowledge;
- ``sealed_partition_breach``: calibration and evaluation partitions overlap;
- ``inadequate_evidence``: too few admitted labels for a partition, verdict or rubric axis.
"""

import json
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Literal, Self

from jsonschema import Draft202012Validator
from pydantic import Field, ValidationError, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
    strict_equal,
    validate_artifact_file_path,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ArtifactStore, ArtifactStoreError, artifact_input
from exp.runtime.environments.artifact_edits import (
    ArtifactEditAdapter,
    NativeArtifactBinding,
    NativeArtifactPlan,
)
from exp.simulation.evaluation.record_execution import (
    RECORD_RECEIPT_TYPE,
    RecordReplayRuntime,
    verify_record_execution,
)
from exp.simulation.evaluation.tasks import (
    WITNESSED_AUTHORITIES,
    EvaluationTask,
    OutcomeRule,
    TaskVerifier,
    Verdict,
    VerifierChallenge,
    grade,
    load_evaluation_task,
    normalized_outcome,
)

AdmissionReason = Literal[
    "self_approval",
    "unverifiable_witness",
    "contaminated_lineage",
    "sealed_partition_breach",
    "inadequate_evidence",
]
WitnessKind = Literal["specification_check", "native_execution"]

# Admitted labels required for each verdict in each partition (and each rubric axis): one
# case per side can agree by coincidence, two per side is the smallest repeated contrast. This
# is a structural admission floor, not a sample size that certifies judge reliability; error
# rates and their uncertainty are reported separately by calibration.
MINIMUM_ADMITTED_PER_VERDICT = 2

NATIVE_RECEIPT_TYPE = "native-artifact-materialization"
SPECIFICATION_TYPE = "evaluation-specification"
WITNESS_TYPE = "label-witness"
# Recorded and human-authored evidence types that may ground a specification. Every artifact in
# a specification's provenance closure must have one of these types; any other type, including
# every model-generated proposal or construction type and any type added later, makes each label
# the specification witnesses a generated label. Deterministic rendering adds no authority.
RECORDED_EVIDENCE_TYPES = frozenset(
    {
        "authored-intent",
        "authored-requirement",
        "authored-task",
        "evaluation-task",
        "human-label-set",
        SPECIFICATION_TYPE,
        "trace-dataset",
    }
)
_WITNESS_KIND: dict[str, WitnessKind] = {
    "synthetic_specification": "specification_check",
    "native_execution": "native_execution",
}


class RecordedWorkbookProvenance(ContractModel):
    """Version-one external workbook bytes, without authenticity or review authority.

    Archive identity records the operator's provenance declaration. Admission separately
    requires a file SourceIdentity and matching attached bytes, then checks every ancestor.
    This evidence kind establishes neither native outcomes nor independently reviewed labels.
    """

    archive_sha256: Sha256
    archive_member: str = Field(min_length=1, max_length=512)
    raw_sha256: Sha256
    role: Literal["initial", "reference"]
    origin: str = Field(min_length=1)
    exposure: str = Field(min_length=1)
    authority: str = Field(min_length=1)

    @model_validator(mode="after")
    def safe_member(self) -> Self:
        """Require a canonical relative XLSX member without path traversal."""
        member = validate_artifact_file_path(self.archive_member)
        if member.as_posix() != self.archive_member or member.suffix != ".xlsx":
            raise ValueError("recorded workbook member must be a safe relative .xlsx path")
        return self


def _recorded_workbook_source(store: ArtifactStore, reference: ArtifactInput) -> bool:
    """Validate the exact external-workbook-source-v1 format, not its type name alone."""
    stored = store.read(reference.artifact_id)
    manifest = stored.manifest
    if (
        manifest.schema_version != 1
        or manifest.source is None
        or manifest.source.kind != "file"
        or set(stored.payloads) != {"workbook.xlsx", "provenance.json"}
    ):
        return False
    try:
        provenance = RecordedWorkbookProvenance.model_validate_json(
            stored.payloads["provenance.json"]
        )
    except ValueError:
        return False
    source_parts = manifest.source.source_id.split("!")
    return (
        len(source_parts) == 2
        and bool(source_parts[0])
        and source_parts[1] == provenance.archive_member
        and manifest.source.sha256 == provenance.raw_sha256
        and sha256_bytes(stored.payloads["workbook.xlsx"]) == provenance.raw_sha256
    )


class EvaluationSpecification(ContractModel):
    """Independent acceptance predicate that a witness grades, never the predicate under test.

    Attributes:
        description: What the specification establishes and where it came from.
        predicate: Data-only predicate; it never runs generated code or contacts a provider.
        subject: ``outcome`` grades the labeled outcome itself; ``native_observation`` grades
            the observation of a native receipt.
    """

    description: str = Field(min_length=1)
    predicate: TaskVerifier
    subject: Literal["outcome", "native_observation"]


class LabelWitness(ContractModel):
    """Saved, re-derivable evidence that one outcome deserves one verdict.

    Attributes:
        kind: Specification check over the outcome, or over a native receipt's observation.
        specification: Exact saved ``evaluation-specification`` artifact.
        receipt: Exact native materialization receipt, required for a native witness only.
        outcome_sha256: Digest of the labeled outcome this witness is about.
        observation: The exact JSON graded by the specification.
        verdict: The specification's mechanical verdict on ``observation``.
        reason: The specification's mechanical reason.
    """

    kind: WitnessKind
    specification: ArtifactInput
    receipt: ArtifactInput | None = None
    outcome_sha256: Sha256
    observation: JsonObject
    verdict: Verdict
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def receipt_kind(self) -> Self:
        """Bind native witnesses, and only native witnesses, to a native receipt."""
        if (self.kind == "native_execution") != (self.receipt is not None):
            raise ValueError("a native witness, and only a native witness, names a receipt")
        return self


class AdmissionRefusal(ContractModel):
    """One named reason that blocks admission or qualification.

    Attributes:
        reason: Named refusal category.
        subject: Case ID, lineage, partition or evidence scope the refusal is about.
        detail: What went wrong and what to do about it.
    """

    reason: AdmissionReason
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class EvidenceAdmission(ContractModel):
    """Store-verified admission result consumed by pure calibration and qualification.

    Attributes:
        verified: Challenge ID to the digest of the exact challenge whose witness re-verified.
        refusals: Every named refusal; any one blocks qualification and paid refinement.
        ledger: Exposure-ledger snapshot identity consulted, when one was.
    """

    verified: dict[str, Sha256] = Field(default_factory=dict)
    refusals: tuple[AdmissionRefusal, ...] = ()
    ledger: str | None = None

    def verified_ids(self, challenges: Iterable[VerifierChallenge]) -> frozenset[str]:
        """IDs of these exact challenges whose witness re-verified, ignoring edited copies."""
        return frozenset(
            challenge.id
            for challenge in challenges
            if self.verified.get(challenge.id) == sha256_json(challenge)
        )


class EvidenceAdmissionError(ValueError):
    """Admission refused before any paid work, with every named reason."""

    def __init__(self, refusals: Sequence[AdmissionRefusal]) -> None:
        """Render every refusal as ``[reason] subject: detail``.

        Args:
            refusals: Nonempty named refusals.
        """
        self.refusals = tuple(refusals)
        super().__init__(
            "evidence admission refused: "
            + "; ".join(f"[{row.reason}] {row.subject}: {row.detail}" for row in self.refusals)
        )


def require_admitted(admission: EvidenceAdmission) -> None:
    """Raise before paid work when any admission refusal exists.

    Args:
        admission: Store-verified admission result.

    Raises:
        EvidenceAdmissionError: At least one named refusal exists.
    """
    if admission.refusals:
        raise EvidenceAdmissionError(admission.refusals)


def persist_specification(
    store: ArtifactStore,
    specification: EvaluationSpecification,
    *,
    sources: Sequence[ArtifactInput] = (),
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Save an independent specification with its digest-checked provenance.

    Args:
        store: Project artifact owner.
        specification: Predicate and subject.
        sources: Artifacts the specification was derived from; a source closure outside
            ``RECORDED_EVIDENCE_TYPES`` makes every label it witnesses a generated label.
        created_at: Publication time.
        code_revision: Producing source identity.

    Returns:
        Exact manifest reference, idempotent for identical content and provenance.
    """
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("specification source differs from its declared manifest digest")
    inputs = tuple(sorted(set(sources), key=lambda ref: ref.artifact_id))
    identity = stable_id(
        SPECIFICATION_TYPE,
        {"specification": sha256_json(specification), "sources": [sha256_json(r) for r in inputs]},
    )
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=SPECIFICATION_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=created_at, inputs=inputs, code_revision=code_revision
        ),
        files={"specification.json": specification.model_dump(mode="json")},
    )
    return artifact_input(manifest)


def record_label_witness(
    store: ArtifactStore,
    *,
    specification: ArtifactInput,
    outcome: WorkflowEvidence | JsonObject,
    receipt: ArtifactInput | None = None,
    created_at: datetime,
    code_revision: str,
    record_runtime: RecordReplayRuntime | None = None,
) -> ArtifactInput:
    """Grade one outcome, or its native receipt, against a specification and save the result.

    Args:
        store: Project artifact owner holding the specification and receipt.
        specification: Saved independent specification.
        outcome: Labeled outcome; a native witness requires it to propose the receipt's plan.
        receipt: Native materialization receipt for a native-execution witness.
        created_at: Publication time.
        code_revision: Producing source identity.
        record_runtime: Explicit trusted native runtime for record receipts; never loaded
            from receipt contents. Required to re-execute record-world evidence.

    Returns:
        Exact ``label-witness`` reference to place on the challenge as ``witness``.

    Raises:
        ValueError: An input differs from its digest or the receipt is not this outcome's.
    """
    spec = _specification(store, specification)
    kind: WitnessKind = "specification_check" if receipt is None else "native_execution"
    if receipt is None:
        if spec.subject != "outcome":
            raise ValueError("a specification check needs an outcome specification")
        observation = _outcome_value(outcome)
    else:
        if spec.subject != "native_observation":
            raise ValueError("a native witness needs a native-observation specification")
        _require_record_goal(store, receipt, spec.predicate)
        observation = _native_observation(store, receipt, outcome, record_runtime)
    result = grade(spec.predicate, observation)
    witness = LabelWitness(
        kind=kind,
        specification=specification,
        receipt=receipt,
        outcome_sha256=sha256_json(outcome),
        observation=observation,
        verdict=result.status,
        reason=result.reason,
    )
    identity = stable_id(WITNESS_TYPE, {"witness": sha256_json(witness)})
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    inputs = tuple(sorted({specification, *(() if receipt is None else (receipt,))}, key=_key))
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=WITNESS_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=created_at, inputs=inputs, code_revision=code_revision
        ),
        files={"witness.json": witness.model_dump(mode="json")},
    )
    return artifact_input(manifest)


def verify_label_witness(
    store: ArtifactStore,
    challenge: VerifierChallenge,
    *,
    label: Verdict,
    compared: Sequence[TaskVerifier],
    record_runtime: RecordReplayRuntime | None = None,
) -> AdmissionRefusal | None:
    """Re-derive one witnessed label from the store; refuse anything that does not reproduce.

    Args:
        store: Project artifact owner.
        challenge: Witnessed label.
        label: Verdict the witness must establish (the case's primary label).
        compared: Predicates under qualification and generated proposals; a witness whose
            specification is one of them would let a predicate approve itself.
        record_runtime: Explicit native replay capability for record-execution receipts.

    Returns:
        None when the witness re-verifies, otherwise one named refusal.
    """

    def refuse(reason: AdmissionReason, detail: str) -> AdmissionRefusal:
        """Name one refusal for this challenge."""
        return AdmissionRefusal(reason=reason, subject=challenge.id, detail=detail)

    if challenge.authority not in WITNESSED_AUTHORITIES or challenge.witness is None:
        return refuse("unverifiable_witness", "the label names no saved label witness")
    try:
        witness = _witness(store, challenge.witness)
        specification = _specification(store, witness.specification)
        generated = generated_ancestor(store, witness.specification)
    except (ArtifactStoreError, OSError, ValueError, ValidationError) as error:
        return refuse("unverifiable_witness", f"witness cannot be re-read: {error}")
    if generated is not None:
        return refuse(
            "self_approval",
            f"the witness specification derives from {generated}, which is not recorded "
            "evidence; author the specification from artifacts of the recorded evidence "
            f"types ({', '.join(sorted(RECORDED_EVIDENCE_TYPES))}) or validated "
            "external-workbook-source-v1 evidence",
        )
    predicate = _predicate_fingerprint(specification.predicate)
    if any(_predicate_fingerprint(row) == predicate for row in compared):
        return refuse(
            "self_approval",
            "the witness specification is a predicate under qualification or a generated "
            "proposal, so it would approve itself; use an independent specification",
        )
    if witness.kind != _WITNESS_KIND[challenge.authority]:
        return refuse("unverifiable_witness", f"{challenge.authority} needs a {witness.kind}")
    if witness.outcome_sha256 != sha256_json(challenge.outcome):
        return refuse("unverifiable_witness", "the witness is about a different outcome")
    try:
        if witness.receipt is not None:
            _require_record_goal(store, witness.receipt, specification.predicate)
        observation = (
            _outcome_value(challenge.outcome)
            if witness.receipt is None
            else _native_observation(store, witness.receipt, challenge.outcome, record_runtime)
        )
    except (ArtifactStoreError, OSError, ValueError, ValidationError) as error:
        return refuse("unverifiable_witness", f"native receipt does not re-verify: {error}")
    if specification.subject != (
        "outcome" if witness.receipt is None else "native_observation"
    ) or not strict_equal(observation, witness.observation):
        return refuse("unverifiable_witness", "the re-derived observation differs from the saved")
    verdict = grade(specification.predicate, observation).status
    if verdict != witness.verdict:
        return refuse("unverifiable_witness", "the specification re-grades to another verdict")
    if verdict != label:
        return refuse(
            "unverifiable_witness",
            f"the witness establishes {verdict}, but the label declares {label}",
        )
    return None


def has_outcome_constraints(verifier: TaskVerifier) -> bool:
    """Reject empty and root-object-only goals, irrespective of policy or annotations.

    This is a minimum structural condition, not semantic qualification. Nonempty checks
    can still be vacuous or wrong and require independent positive and negative controls.
    """
    if verifier.rules or verifier.relations or verifier.table_relations:
        return True
    schema = _schema_assertions(expand_local_references(verifier.outcome_schema))
    return schema not in ({}, {"type": "object"}, {"type": ["object"]})


def _predicate_fingerprint(verifier: TaskVerifier) -> Sha256:
    """Detect syntactic copies despite annotations or reordered conjunctions.

    This is a conservative copy check, not a proof that differently expressed predicates were
    authored independently. Provenance checks remain necessary; logically equivalent but
    structurally different programs cannot establish independent truth merely by differing here.
    """

    def conjunction(rules: tuple[OutcomeRule, ...]) -> list[JsonObject]:
        """Canonicalize conjunctions without reinterpreting their literal values."""
        values = {sha256_json(rule): rule.model_dump(mode="json") for rule in rules}
        return [values[key] for key in sorted(values)]

    value: JsonObject = verifier.model_dump(mode="json", exclude={"description"})
    value["outcome_schema"] = _schema_assertions(expand_local_references(verifier.outcome_schema))
    value["rules"] = conjunction(verifier.rules)
    value["required_paths"] = sorted(
        {
            *verifier.required_paths,
            *(path for row in verifier.table_relations for path in (row.source, row.target)),
        }
    )
    value["preserved_paths"] = (
        [] if verifier.preserve_state else sorted(set(verifier.preserved_paths))
    )
    value["allowed_tools"] = (
        None if verifier.allowed_tools is None else sorted(set(verifier.allowed_tools))
    )
    value["relations"] = [
        list(pair)
        for pair in sorted({tuple(sorted((row.left, row.right))) for row in verifier.relations})
    ]
    table_rules = {
        sha256_json(
            row.model_copy(update={"key_columns": tuple(sorted(row.key_columns))})
        ): row.model_copy(update={"key_columns": tuple(sorted(row.key_columns))}).model_dump(
            mode="json"
        )
        for row in verifier.table_relations
    }
    value["table_relations"] = [table_rules[key] for key in sorted(table_rules)]
    value["argument_rules"] = {
        name: conjunction(rules) for name, rules in verifier.argument_rules.items()
    }
    if verifier.argument_schemas:
        value["argument_schemas"] = {
            name: _schema_assertions(expand_local_references(schema))
            for name, schema in verifier.argument_schemas.items()
        }
    return sha256_json(value)


def _schema_assertions(schema: JsonValue) -> JsonValue:
    """Remove annotations only at schema positions, preserving identically named data fields."""
    if not isinstance(schema, dict):
        return schema
    result: JsonObject = {}
    for key, value in schema.items():
        # grade() uses this validator without a FormatChecker. Unknown vocabulary and
        # annotations do not constrain its verdict and cannot manufacture a second oracle.
        if key not in Draft202012Validator.VALIDATORS or key == "format":
            continue
        if key in {
            "additionalProperties",
            "contains",
            "contentSchema",
            "else",
            "if",
            "items",
            "not",
            "propertyNames",
            "then",
            "unevaluatedItems",
            "unevaluatedProperties",
        }:
            value = _schema_assertions(value)
        elif key in {"properties", "patternProperties", "dependentSchemas"} and isinstance(
            value, dict
        ):
            value = {name: _schema_assertions(child) for name, child in value.items()}
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(value, list):
            value = [_schema_assertions(child) for child in value]
            if key != "prefixItems":
                value = sorted(value, key=sha256_json)
        elif key in {"required", "enum"} and isinstance(value, list):
            value = sorted(value, key=sha256_json)
        elif key == "dependentRequired" and isinstance(value, dict):
            value = {
                name: sorted(required, key=sha256_json) if isinstance(required, list) else required
                for name, required in value.items()
            }
        result[key] = value
    return result


def adequacy_refusals(
    rows: Iterable[tuple[str, Verdict, bool, frozenset[str]]],
    *,
    partitions: Sequence[str],
    axes: Sequence[str] = (),
) -> tuple[AdmissionRefusal, ...]:
    """Refuse partitions, verdicts or rubric axes with too few admitted labels.

    Args:
        rows: ``(partition, label, admitted, covered axes)`` for every case.
        partitions: Partitions that must each hold admitted positive and negative labels.
        axes: Rubric criteria that need the same admitted contrast in every partition.

    Returns:
        One ``inadequate_evidence`` refusal per deficient partition/verdict or axis/verdict.
    """
    admitted = [row for row in rows if row[2]]
    refusals = []
    for partition in partitions:
        for verdict in ("pass", "fail"):
            scopes = [("", None), *((f" on axis {axis}", axis) for axis in axes)]
            for label, axis in scopes:
                count = sum(
                    row[0] == partition and row[1] == verdict and (axis is None or axis in row[3])
                    for row in admitted
                )
                if count < MINIMUM_ADMITTED_PER_VERDICT:
                    refusals.append(
                        AdmissionRefusal(
                            reason="inadequate_evidence",
                            subject=f"{partition}{label}",
                            detail=(
                                f"{count} admitted {verdict} labels; at least "
                                f"{MINIMUM_ADMITTED_PER_VERDICT} witnessed or attested labels "
                                "are required (a structural floor, not a statistical "
                                "certification)"
                            ),
                        )
                    )
    return tuple(refusals)


def task_lineage_challenges(
    store: ArtifactStore, task: EvaluationTask
) -> tuple[VerifierChallenge, ...]:
    """Every label shown to authoring of this task or any ancestor revision.

    Args:
        store: Project artifact owner.
        task: Verified evaluation task.

    Returns:
        Challenges on the task and its parent chain, which are development by construction.
    """
    found = list(task.challenges)
    parent = task.parent
    while parent is not None:
        ancestor = load_evaluation_task(store, parent)
        found.extend(ancestor.challenges)
        parent = ancestor.parent
    return tuple(found)


def _key(reference: ArtifactInput) -> str:
    """Sort key for deterministic manifest input order."""
    return reference.artifact_id


def _witness(store: ArtifactStore, reference: ArtifactInput) -> LabelWitness:
    """Read one digest-checked label witness."""
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError("label witness differs from its declared digest")
    if stored.manifest.artifact_type != WITNESS_TYPE:
        raise ValueError(f"{reference.artifact_id} is not a saved label witness")
    witness = LabelWitness.model_validate_json(
        store.read_bytes(reference.artifact_id, "witness.json")
    )
    expected = {witness.specification, *(() if witness.receipt is None else (witness.receipt,))}
    if set(stored.manifest.inputs) != expected:
        raise ValueError("label witness manifest inputs differ from its evidence")
    return witness


def _specification(store: ArtifactStore, reference: ArtifactInput) -> EvaluationSpecification:
    """Read one digest-checked specification."""
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError("specification differs from its declared digest")
    if stored.manifest.artifact_type != SPECIFICATION_TYPE:
        raise ValueError(f"{reference.artifact_id} is not a saved evaluation specification")
    return EvaluationSpecification.model_validate_json(
        store.read_bytes(reference.artifact_id, "specification.json")
    )


def generated_ancestor(store: ArtifactStore, reference: ArtifactInput) -> str | None:
    """The first artifact in a provenance closure that is not recorded evidence.

    Unclassified types are refused like generated ones, so a new generated type fails closed.

    Raises:
        ValueError: An artifact in the closure differs from its declared digest.
    """
    queue, seen = deque((reference,)), set[str]()
    while queue:
        current = queue.popleft()
        if current.artifact_id in seen:
            continue
        seen.add(current.artifact_id)
        manifest = store.read(current.artifact_id).manifest
        if artifact_input(manifest) != current:
            raise ValueError("specification provenance differs from its declared digest")
        recorded = manifest.artifact_type in RECORDED_EVIDENCE_TYPES or (
            manifest.artifact_type == "external-workbook-source"
            and _recorded_workbook_source(store, current)
        )
        if not recorded or (
            manifest.artifact_type == "evaluation-task"
            and EvaluationTask.model_validate_json(
                store.read_bytes(current.artifact_id, "task.json")
            ).parent
            is not None
        ):
            return f"{current.artifact_id} ({manifest.artifact_type})"
        queue.extend(manifest.inputs)
    return None


def _outcome_value(outcome: WorkflowEvidence | JsonObject) -> JsonObject:
    """The graded JSON of a labeled outcome."""
    return normalized_outcome(outcome) if isinstance(outcome, WorkflowEvidence) else outcome


def _require_record_goal(
    store: ArtifactStore, receipt: ArtifactInput, predicate: TaskVerifier
) -> None:
    """Require an explicit native goal observation, not process completion or empty checks.

    This is a minimum evidence condition, not a proof the authored goal matches user intent.
    Refusal tasks may assert actual reply fields; state tasks assert actual resulting state.
    Schema-only record witnesses must express their goal as a data-only outcome rule.
    """
    if store.read(receipt.artifact_id).manifest.artifact_type != RECORD_RECEIPT_TYPE:
        return
    paths = [rule.path for rule in predicate.rules]
    paths.extend(
        path for relation in predicate.relations for path in (relation.left, relation.right)
    )
    paths.extend(
        path
        for relation in predicate.table_relations
        for path in (relation.source, relation.target)
    )
    if not any(
        path == "/final_state"
        or path.startswith("/final_state/")
        or (path.startswith("/steps/") and "/reply/" in path)
        for path in paths
    ):
        raise ValueError(
            "record witness requires an explicit observable goal rule over native state "
            "or replies; "
            "no detected error or completed execution alone cannot establish task success"
        )


def _native_observation(
    store: ArtifactStore,
    reference: ArtifactInput,
    outcome: WorkflowEvidence | JsonObject,
    record_runtime: RecordReplayRuntime | None = None,
) -> JsonObject:
    """Re-check a native receipt end to end and return its native observation.

    The receipt's input bytes, adapter digest, edit plan and output bytes are re-read under
    their digests, and its plan must be exactly the plan the labeled outcome proposed.
    """
    stored = store.read(reference.artifact_id)
    manifest = stored.manifest
    if manifest.artifact_type == RECORD_RECEIPT_TYPE:
        if record_runtime is None:
            raise ValueError(
                "record witnesses require an explicitly supplied native replay runtime"
            )
        execution = verify_record_execution(store, reference, record_runtime)
        value = _outcome_value(outcome)
        for key in ("initial_state", "steps", "final_state", "final_answer", "stop_reason"):
            if key not in value or not strict_equal(value[key], execution.outcome[key]):
                raise ValueError(f"record receipt differs from the labeled outcome's {key}")
        return execution.outcome
    if artifact_input(manifest) != reference or manifest.artifact_type != NATIVE_RECEIPT_TYPE:
        raise ValueError("native witness does not name an exact native materialization receipt")
    receipt = _object(store.read_bytes(reference.artifact_id, "receipt.json"))
    binding = NativeArtifactBinding.model_validate(receipt.get("binding"))
    plan = NativeArtifactPlan.model_validate(receipt.get("plan"))
    if receipt.get("authority") != "native_artifact_transformation":
        raise ValueError("receipt is not a native artifact transformation")
    if artifact_input(store.read(binding.input_artifact.artifact_id).manifest) != (
        binding.input_artifact
    ):
        raise ValueError("native input bytes differ from the receipt's binding")
    if plan.input_sha256 != binding.input_artifact.sha256:
        raise ValueError("native plan names a different input artifact")
    adapter = receipt.get("adapter")
    if not isinstance(adapter, dict) or sha256_json(adapter) != binding.adapter_sha256:
        raise ValueError("native adapter digest differs from the receipt's binding")
    if binding.input_artifact not in manifest.inputs:
        raise ValueError("native receipt does not retain its input artifact")
    projection = receipt.get("native_projection")
    if not isinstance(projection, dict):
        raise ValueError("native receipt has no native observation")
    owned = {row.path for row in manifest.files}
    if {"outputs/" + path for path in binding.output_paths} - owned:
        raise ValueError("native receipt does not retain every declared output")
    proposed = _proposed_plan(_outcome_value(outcome), binding.plan_key)
    if not strict_equal(proposed, plan.model_dump(mode="json")):
        raise ValueError("native receipt executed a different plan than the labeled outcome")
    outputs = {
        path: store.read_bytes(reference.artifact_id, "outputs/" + path)
        for path in binding.output_paths
    }
    _require_rederived(store, binding, plan, outputs, projection)
    return {
        "native_projection": projection,
        "outputs": {path: sha256_bytes(payload) for path, payload in outputs.items()},
    }


def maintained_native_adapters() -> tuple[ArtifactEditAdapter, ...]:
    """The adapters trusted to re-derive native receipts, never chosen by receipt contents.

    Raises:
        ValueError: The optional ``artifact-edits`` dependencies are not installed.
    """
    try:
        from exp.runtime.environments.workbook_edits import WorkbookEditAdapter
    except ImportError as error:
        raise ValueError(
            "re-deriving a native workbook receipt needs the artifact-edits extra"
        ) from error
    return (WorkbookEditAdapter(),)


def _require_rederived(
    store: ArtifactStore,
    binding: NativeArtifactBinding,
    plan: NativeArtifactPlan,
    outputs: dict[str, bytes],
    projection: JsonObject,
) -> None:
    """Re-run the receipt's pure transformation and require its saved bytes and projection.

    A receipt is a store artifact anyone can write, so its projection is evidence only when
    the maintained adapter it names derives that projection and those output bytes again from
    the bound input bytes.
    """
    adapter = next(
        (
            row
            for row in maintained_native_adapters()
            if sha256_json(row.definition) == binding.adapter_sha256
        ),
        None,
    )
    if adapter is None:
        raise ValueError(
            "no maintained native adapter matches the receipt's adapter digest, so its "
            "observation cannot be re-derived; materialize under a maintained adapter"
        )
    manifest = store.read(binding.input_artifact.artifact_id).manifest
    files = {row.path: store.read_bytes(manifest.artifact_id, row.path) for row in manifest.files}
    native = adapter.apply(files, plan.operations, binding.output_paths)
    if dict(native.files) != outputs:
        raise ValueError("native output bytes differ from what the adapter derives")
    if not strict_equal(native.projection, projection):
        raise ValueError("native projection differs from what the adapter derives")


def _proposed_plan(value: JsonObject, plan_key: str) -> JsonValue:
    """The native edit plan in an outcome's final candidate state."""
    final_state = value.get("final_state")
    if not isinstance(final_state, dict) or plan_key not in final_state:
        raise ValueError("the labeled outcome proposes no native edit plan")
    return final_state[plan_key]


def _object(payload: bytes) -> JsonObject:
    """Parse one JSON object."""
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value
