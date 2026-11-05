"""Immutable preparation resumes retain original authority and explicit execution windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import AwareDatetime, Field, TypeAdapter

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.optimize.evaluation.factory.author_retries import (
    load_author_execution_window,
    load_author_retry,
    operation_identity,
)
from exp.optimize.evaluation.spending import operation_directory, operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_pilot import (
    BootstrapPilotRevision,
    apply_pilot_revision,
    validate_pilot_revision,
)
from exp.optimize.evaluation.variance.bootstrap_protocols import validate_protocol_successor
from exp.optimize.evaluation.variance.bootstrap_source import (
    BootstrapExecutionSource,
    current_bootstrap_execution_source,
)
from exp.runtime.models.spend_ceiling import OperationSpendCeiling, SpendAuthorization
from exp.simulation.mining.environment_review import review_protocol_sha256

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec

RESUME_TYPE = "bootstrap-resume-binding"
INPUTS_TYPE = "bootstrap-resume-inputs"
REGISTRATION_TYPE = "bootstrap-resume-registration"
ResumeChange = Literal[
    "cached_tagging_inventory",
    "explicit_terminal_503_retry",
    "execution_watchdog_renewal",
    "compatibility_review_revision",
    "pilot_selection_revision",
    "preparation_protocol_revision",
    "family_embedding_revision",
]


class BootstrapResumeInputs(ContractModel):
    """Exact decoded original files frozen before execution-window registration.

    Attributes:
        spec: Original specification JSON, without schema-default reinterpretation.
        plan: Original full displayed plan JSON, including its original producer.
        approval: Original approval JSON naming the exact original plan digest.
        source_build: Immutable completed source-build outputs used by that plan.
    """

    spec: JsonObject
    plan: JsonObject
    approval: JsonObject
    source_build: ProjectBuildArtifacts


class BootstrapReviewRevision(ContractModel):
    """Explicit review-only revision; model, evidence and logical study controls stay fixed."""

    output_tokens: int = Field(ge=1, le=8192, strict=True)
    maximum_physical_attempts: int = Field(ge=1, le=10, strict=True)
    protocol_sha256: Sha256


class BootstrapResumeBinding(ContractModel):
    """Original authority and exact execution changes a reviewed producer may continue.

    Attributes:
        original_inputs: Frozen original specification, plan and approval provenance.
        original_revision: Producer frozen into original preparation/build identities.
        producer_revision: Actual reviewed producer writing new artifacts.
        original_spec_sha256: Exact original decoded specification digest.
        original_plan_sha256: Exact original decoded plan digest, before schema defaults.
        original_approval_sha256: Exact original decoded approval digest.
        source_build: Unchanged immutable source outputs.
        original_build: Original semantic variance-build identity.
        original_operation: Original preparation/shared spend operation identity.
        ledger_sha256: Canonical original ledger identity, shared with author grants.
        grants: Exact append-only authorization history at registration.
        ceiling_usd: Original effective dollar ceiling.
        unpriced_attempts: Original effective whole-operation unpriced ceiling.
        preparation_usd: Original explicitly approved preparation dollar allowance.
        preparation_unpriced_attempts: Original preparation unpriced allowance.
        deadline: Original absolute operational deadline, preserved as evidence.
        execution_deadline: Explicit finite renewed watchdog, or None for the original stop.
        execution_spec_sha256: Exact execution specification including permitted review changes.
        execution_window: Native author-owner permission for the selected exact review settings.
        permitted_changes: Exact reviewed repair and optional watchdog-renewal permissions.
        author_retries: Separately registered exact terminal-response retries.
        previous_binding: Exact predecessor, or None for the immutable root registration.
        review_revision: Explicit review output, physical-attempt and protocol revision.
        execution_build: Separately pinned revised build identity, or None for original semantics.
        pilot_revision: Explicit selection and saved-build evidence, or None for review repairs.
        protocol_revision: Exact reviewed source contracts, without configuration changes.
        embedding_revision: Genuine completed exact-family embedding and rules revision.
    """

    execution_source: BootstrapExecutionSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    original_inputs: ArtifactInput
    original_revision: str = Field(min_length=1)
    producer_revision: str = Field(min_length=1)
    original_spec_sha256: Sha256
    original_plan_sha256: Sha256
    original_approval_sha256: Sha256
    source_build: ProjectBuildArtifacts
    original_build: str
    original_operation: str
    ledger_sha256: Sha256
    grants: tuple[SpendAuthorization, ...]
    ceiling_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(ge=0)
    preparation_usd: float = Field(ge=0)
    preparation_unpriced_attempts: int = Field(ge=0)
    deadline: AwareDatetime
    execution_deadline: AwareDatetime | None = None
    execution_spec_sha256: Sha256
    execution_window: ArtifactInput | None = None
    permitted_changes: tuple[ResumeChange, ...] = Field(min_length=1)
    author_retries: tuple[ArtifactInput, ...] = ()
    previous_binding: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    review_revision: BootstrapReviewRevision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    execution_build: str | None = Field(default=None, exclude_if=lambda value: value is None)

    pilot_revision: BootstrapPilotRevision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    protocol_revision: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    embedding_revision: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def build_revision(self) -> str:
        """Keep legacy identities, but bind revised review builds to their actual producer."""
        if self.pilot_revision is not None:
            return self.pilot_revision.build_producer_revision
        return self.original_revision if self.review_revision is None else self.producer_revision

    @property
    def build_identity(self) -> str:
        """Return the exact separately pinned execution build."""
        return self.execution_build or self.original_build

    @property
    def stop(self) -> datetime:
        """Return the effective finite operational stop without mutating original evidence."""
        return self.execution_deadline or self.deadline


def _original(
    inputs: BootstrapResumeInputs,
) -> tuple[BootstrapPreparationSpec, str, str, str, float, int]:
    """Validate exact old JSON against its owning schemas and operation identity.

    Args:
        inputs: Exact decoded original files with their source-build references.

    Returns:
        Typed original settings, producer, build, operation and approved allowances.

    Raises:
        ValueError: Original approval, settings, task state or operation identity differs.
    """
    # This owner imports the binding; original model parsing runs only after both modules load.
    from exp.optimize.evaluation.variance.bootstrap_plan import (  # noqa: PLC0415
        BootstrapApproval,
        BootstrapPlan,
        BootstrapPreparationSpec,
        _identity,
    )

    spec = BootstrapPreparationSpec.model_validate(inputs.spec)
    plan = BootstrapPlan.model_validate(inputs.plan)
    approval = BootstrapApproval.model_validate(inputs.approval)
    if plan.build_saved or plan.completed_tasks or plan.remaining_tasks or plan.resume_binding:
        raise ValueError(
            "bootstrap resume requires an unfinished variance build without task state"
        )
    if approval.plan_sha256 != sha256_json(inputs.plan):
        raise ValueError("original approval does not pin the exact original plan JSON")
    if approval.ceiling_usd < plan.known_usd or approval.unpriced_attempts < plan.unpriced_attempts:
        raise ValueError("original approval does not cover its original preparation inventory")
    if plan.spec_sha256 != sha256_json(inputs.spec) or sha256_json(spec) != sha256_json(
        inputs.spec
    ):
        raise ValueError("original preparation settings differ from their approved schema")
    if _identity(spec, plan.build_identity, plan.code_revision) != plan.identity:
        raise ValueError("original preparation identity differs from its exact approved inputs")
    return (
        spec,
        plan.code_revision,
        plan.build_identity,
        plan.identity,
        approval.ceiling_usd,
        approval.unpriced_attempts,
    )


def _ledger(project: ProjectStore, operation: str) -> OperationSpendCeiling:
    """Open only the preexisting original ledger, never manufacture fresh resume authority."""
    if not (operation_directory(project, operation) / "ledger.sqlite").is_file():
        raise ValueError("original preparation spend ledger is missing; restore its exact history")
    return operation_spend_ceiling(project, operation)


def _source_references(source: ProjectBuildArtifacts) -> tuple[ArtifactInput, ...]:
    """Return the source build's exact five immutable roots."""
    return (
        source.trace_dataset,
        source.task_set,
        source.serving_rag,
        source.fit_rag,
        source.world_model,
    )


def _preparation_authority(
    ledger: OperationSpendCeiling,
    dollars: float,
    attempts: int,
) -> None:
    """Require the original exact preparation grant already present in the shared ledger."""
    preparation = [row for row in ledger.summary().authorizations if row.stage == "preparation"]
    if not preparation or (
        preparation[-1].ceiling_usd != dollars or preparation[-1].unpriced_attempts != attempts
    ):
        raise ValueError("original approval differs from the preexisting preparation authority")


def freeze_bootstrap_resume_inputs(
    project: ProjectStore,
    *,
    original_spec_file: Path,
    original_plan_file: Path,
    original_approval_file: Path,
    code_revision: str,
) -> ArtifactInput:
    """Freeze explicitly supplied original consent before registering author-owner windows.

    Args:
        project: Owner of the unchanged source and preexisting operation spend ledger.
        original_spec_file: Explicit original external specification file.
        original_plan_file: Explicit original external full plan file.
        original_approval_file: Explicit original external approval file.
        code_revision: Actual producer freezing the unchanged original evidence.

    Returns:
        Immutable original-input provenance usable by window and retry registration.

    Raises:
        ValueError: Original consent, source identity or existing ledger authority differs.
    """
    source = project.load_project().build
    if source is None:
        raise ValueError("original source build is missing")
    decoder = TypeAdapter(JsonObject)
    inputs = BootstrapResumeInputs(
        spec=decoder.validate_json(original_spec_file.read_bytes()),
        plan=decoder.validate_json(original_plan_file.read_bytes()),
        approval=decoder.validate_json(original_approval_file.read_bytes()),
        source_build=source,
    )
    _, _, _, operation, dollars, attempts = _original(inputs)
    _preparation_authority(_ledger(project, operation), dollars, attempts)
    references = _source_references(source)
    for reference in references:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("bootstrap resume source reference changed")
    identity = stable_id(INPUTS_TYPE, inputs)
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=INPUTS_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(references, key=lambda row: row.artifact_id)),
            ),
            files={"inputs.json": inputs},
        )
    return artifact_input(project.artifacts.read(identity).manifest)


def _load_inputs(project: ProjectStore, reference: ArtifactInput) -> BootstrapResumeInputs:
    """Read original JSON and require the same selected source build and ancestry.

    Args:
        project: Owner of the immutable inputs and selected completed source build.
        reference: Exact frozen original-input artifact.

    Returns:
        Original JSON objects after reference and ancestry verification.

    Raises:
        ValueError: The original artifact, selected source or source ancestry differs.
    """
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if manifest.artifact_type != INPUTS_TYPE or artifact_input(manifest) != reference:
        raise ValueError("bootstrap resume original-input reference changed")
    inputs = BootstrapResumeInputs.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "inputs.json")
    )
    if project.load_project().build != inputs.source_build:
        raise ValueError("bootstrap resume source build differs from the original source build")
    if manifest.inputs != tuple(
        sorted(_source_references(inputs.source_build), key=lambda row: row.artifact_id)
    ):
        raise ValueError("bootstrap resume original-input source ancestry differs")
    return inputs


def _execution_spec(
    spec: BootstrapPreparationSpec,
    deadline: datetime | None,
    review_revision: BootstrapReviewRevision | None = None,
    pilot_revision: BootstrapPilotRevision | None = None,
) -> BootstrapPreparationSpec:
    """Apply registered review and pilot changes and renew only known operational deadlines.

    Args:
        spec: Original frozen preparation settings, including any business JSON.
        deadline: Explicit operational stop, or None to preserve original deadlines.
        review_revision: Review-only changes, or None to retain original build semantics.
        pilot_revision: Explicit three-field pilot revision, without changing build inputs.

    Returns:
        A typed specification differing only in registered review, pilot and deadline fields.
    """
    if review_revision is not None:
        build = spec.build
        reviewer, families = build.reviewer, build.spec.environment_families
        assert reviewer is not None and families is not None
        if (
            review_revision.output_tokens < reviewer.output_tokens
            or review_revision.maximum_physical_attempts > reviewer.maximum_physical_attempts
        ):
            raise ValueError("review revision cannot lower output or enlarge physical attempts")
        reviewer = reviewer.model_copy(
            update={
                "output_tokens": review_revision.output_tokens,
                "maximum_physical_attempts": review_revision.maximum_physical_attempts,
            }
        )
        families = families.model_copy(
            update={
                "review": families.review.model_copy(
                    update={
                        "maximum_output_tokens": review_revision.output_tokens,
                    }
                )
            }
        )
        build = build.model_copy(
            update={
                "reviewer": reviewer,
                "spec": build.spec.model_copy(update={"environment_families": families}),
            }
        )
        spec = type(spec).model_validate(
            spec.model_copy(update={"build": build}).model_dump(mode="json")
        )
    if pilot_revision is not None:
        spec = apply_pilot_revision(spec, pilot_revision)
    if deadline is None:
        return spec
    settings = spec.bootstrap
    updates = {
        "author": settings.author.model_copy(update={"deadline": deadline}),
        "runtime": settings.runtime.model_copy(update={"deadline": deadline}),
        "job": settings.job.model_copy(update={"deadline": deadline}),
    }
    if settings.reviewer is not None:
        updates["reviewer"] = settings.reviewer.model_copy(update={"deadline": deadline})
    if settings.researcher is not None:
        updates["researcher"] = settings.researcher.model_copy(update={"deadline": deadline})
    if settings.rules is not None:
        updates["rules"] = settings.rules.model_copy(
            update={"author": settings.rules.author.model_copy(update={"deadline": deadline})}
        )
    return type(spec).model_validate(
        spec.model_copy(update={"bootstrap": settings.model_copy(update=updates)}).model_dump(
            mode="json"
        )
    )


def _validate_window(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
    spec: BootstrapPreparationSpec,
) -> None:
    """Require native reviewer-window authority for an explicit operational renewal.

    Args:
        project: Owner of the immutable execution-window artifact.
        binding: Original authority and optional renewed execution permission.
        spec: Original unchanged preparation settings.

    Raises:
        ValueError: Renewal permission, window provenance or original authority differs.
    """
    renewed = binding.execution_deadline is not None
    if renewed != ("execution_watchdog_renewal" in binding.permitted_changes):
        raise ValueError("bootstrap watchdog renewal needs its exact explicit permission")
    if not renewed:
        if binding.execution_window is not None:
            raise ValueError("an unchanged watchdog cannot carry a renewed author window")
        return
    if binding.execution_window is None:
        raise ValueError("bootstrap watchdog renewal needs a native author execution window")
    window = load_author_execution_window(project, binding.execution_window)
    reviewer = _execution_spec(spec, None, binding.review_revision).build.reviewer
    assert reviewer is not None
    if (
        window.settings_sha256 != sha256_json(reviewer)
        or window.original_deadline != binding.deadline
        or window.execution_deadline != binding.execution_deadline
        or window.operation_ledger_sha256 != binding.ledger_sha256
        or window.operation_ceiling_usd != binding.ceiling_usd
        or window.operation_unpriced_attempts != binding.unpriced_attempts
        or binding.original_inputs not in window.sources
    ):
        raise ValueError("bootstrap author execution window differs from original authority")


def bootstrap_author_retries(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
) -> dict[str, ArtifactInput]:
    """Map exact original review identities to same-operation terminal retry permissions.

    Args:
        project: Owner of the immutable retry registrations.
        binding: Original operation authority and exact permitted retry references.

    Returns:
        One registered retry per original logical review identity.

    Raises:
        ValueError: Retry permission, identity, original authority or watchdog differs.
    """
    result: dict[str, ArtifactInput] = {}
    if binding.author_retries and "explicit_terminal_503_retry" not in binding.permitted_changes:
        raise ValueError("bootstrap terminal retries require their explicit permission")
    for reference in binding.author_retries:
        retry = load_author_retry(project, reference)
        identity = retry.parent_intent.artifact_id.removesuffix("-intent")
        if identity in result:
            raise ValueError("bootstrap resume repeats a terminal author retry identity")
        if (
            retry.operation_ledger_sha256 != binding.ledger_sha256
            or retry.operation_ceiling_usd != binding.ceiling_usd
            or retry.operation_unpriced_attempts != binding.unpriced_attempts
            or retry.execution_window != binding.execution_window
            or retry.effective_retry_deadline != binding.stop
        ):
            raise ValueError("bootstrap terminal retry differs from original authority or window")
        result[identity] = reference
    return result


def _registration_identity(binding: BootstrapResumeBinding) -> str:
    """Give each exact predecessor one append-only successor registration."""
    if binding.previous_binding is None:
        return binding.original_operation + "-resume"
    return stable_id(
        "bootstrap-resume-successor",
        {
            "operation": binding.original_operation,
            "previous": binding.previous_binding.model_dump(mode="json"),
        },
    )


def _validate_predecessor(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
    spec: BootstrapPreparationSpec,
    *,
    historical: bool = False,
) -> None:
    """Verify successor ancestry without changing the historical root selection."""
    if (binding.protocol_revision is not None) != (
        "preparation_protocol_revision" in binding.permitted_changes
    ):
        raise ValueError("source successor requires its exact explicit protocol permission")
    if (binding.embedding_revision is not None) != (
        "family_embedding_revision" in binding.permitted_changes
    ):
        raise ValueError("family embedding successor requires its exact explicit permission")
    if binding.previous_binding is None:
        if (
            binding.review_revision is not None
            or binding.execution_build is not None
            or binding.pilot_revision is not None
            or binding.protocol_revision is not None
            or binding.embedding_revision is not None
        ):
            raise ValueError("review revision requires its exact predecessor binding")
        return
    if binding.execution_build is None or (
        binding.review_revision is None
        and binding.pilot_revision is None
        and binding.protocol_revision is None
        and binding.embedding_revision is None
    ):
        raise ValueError("successor requires an explicitly pinned revision")
    if binding.author_retries:
        raise ValueError("review revision cannot reuse historical author retry registrations")
    if (
        binding.review_revision is not None
        and "compatibility_review_revision" not in binding.permitted_changes
    ):
        raise ValueError("successor lacks its explicit review revision permission")
    manifest = project.artifacts.read(binding.previous_binding.artifact_id).manifest
    parent = _load_bootstrap_resume(
        project,
        binding.previous_binding,
        spec=spec,
        code_revision=manifest.code_revision,
        historical=True,
    )
    if (
        parent.original_inputs != binding.original_inputs
        or parent.original_operation != binding.original_operation
        or parent.ledger_sha256 != binding.ledger_sha256
        or parent.grants != binding.grants
        or parent.source_build != binding.source_build
    ):
        raise ValueError("successor differs from its predecessor's original authority")
    if (binding.pilot_revision is not None) != (
        "pilot_selection_revision" in binding.permitted_changes
    ):
        raise ValueError("pilot revision requires its exact explicit permission")
    if binding.protocol_revision is not None:
        validate_protocol_successor(project, binding, parent, historical=historical)
        return
    if binding.embedding_revision is not None:
        from exp.optimize.evaluation.variance.family_embedding import (
            validate_family_embedding_successor,
        )

        validate_family_embedding_successor(project, binding, parent, historical=historical)
        return
    if binding.pilot_revision is not None:
        if (
            binding.review_revision != parent.review_revision
            or binding.build_identity != parent.build_identity
        ):
            raise ValueError(
                "pilot revision cannot change its predecessor's review or build identity"
            )
        baseline = bootstrap_execution_spec(project, parent, spec)
        validate_pilot_revision(
            project, binding.pilot_revision, baseline, expected_build=parent.build_identity
        )
        return
    if parent.pilot_revision is not None:
        raise ValueError("a saved pilot revision cannot revert to review-build semantics")
    assert binding.review_revision is not None
    previous_review = _execution_spec(spec, None, parent.review_revision).build.reviewer
    assert previous_review is not None
    if (
        binding.review_revision.output_tokens < previous_review.output_tokens
        or binding.review_revision.maximum_physical_attempts
        > previous_review.maximum_physical_attempts
    ):
        raise ValueError("successor cannot lower output or enlarge predecessor attempts")


def register_bootstrap_resume(
    project: ProjectStore,
    *,
    original_inputs: ArtifactInput,
    code_revision: str,
    permitted_changes: tuple[ResumeChange, ...],
    execution_deadline: datetime | None = None,
    execution_window: ArtifactInput | None = None,
    author_retries: tuple[ArtifactInput, ...] = (),
    previous_binding: ArtifactInput | None = None,
    review_revision: BootstrapReviewRevision | None = None,
    pilot_revision: BootstrapPilotRevision | None = None,
    protocol_revision: ArtifactInput | None = None,
    embedding_revision: ArtifactInput | None = None,
) -> ArtifactInput:
    """Bind a reviewed producer to original consent and an optional native finite watchdog.

    Args:
        project: Owner of the original inputs, unchanged source and original spend ledger.
        original_inputs: Frozen exact specification, full plan and original approval.
        code_revision: Actual producer of newly written artifacts.
        permitted_changes: Exact reviewed repairs and optional operational watchdog renewal.
        execution_deadline: Explicit renewed stop, at most six hours after registration.
        execution_window: Native permission for the unchanged original review settings.
        author_retries: Separate exact terminal-response retries under the same window.
        previous_binding: Existing root or successor continued through an append-only selection.
        review_revision: Explicit changed review bounds and current protocol, only for successors.
        pilot_revision: Concrete reviewed selection and truthful existing completed-build binding.
        protocol_revision: Reviewed source-only successor control, with unchanged pilot/settings.
        embedding_revision: Actual completed exact-family target representation and reviewed rules.

    Returns:
        Immutable resume permission and its exact fresh bootstrap execution specification.

    Raises:
        ValueError: Source, settings, original authority, renewal or retry permissions differ.
    """
    inputs = _load_inputs(project, original_inputs)
    spec, revision, build, operation, dollars, attempts = _original(inputs)
    if revision == code_revision:
        raise ValueError("a bootstrap resume binding requires a distinct actual producer")
    reviewer = spec.build.reviewer
    assert reviewer is not None
    if execution_deadline is not None and (
        execution_deadline.tzinfo is None
        or not datetime.now(UTC) < execution_deadline <= datetime.now(UTC) + timedelta(hours=6)
    ):
        raise ValueError("bootstrap renewed execution watchdog must be finite and within six hours")
    ledger = _ledger(project, operation)
    _preparation_authority(ledger, dollars, attempts)
    summary = ledger.summary()
    if (previous_binding is None) != (
        review_revision is None
        and pilot_revision is None
        and protocol_revision is None
        and embedding_revision is None
    ):
        raise ValueError("review revision requires its exact predecessor binding")
    if (review_revision is not None) != ("compatibility_review_revision" in permitted_changes):
        raise ValueError("review revision needs its exact explicit permission")
    if review_revision is not None and review_revision.protocol_sha256 != review_protocol_sha256():
        raise ValueError("review revision differs from the current review protocol")
    if (pilot_revision is not None) != ("pilot_selection_revision" in permitted_changes):
        raise ValueError("pilot revision needs its exact explicit permission")
    execution = _execution_spec(spec, execution_deadline, review_revision, pilot_revision)
    if embedding_revision is not None:
        from exp.optimize.evaluation.variance.family_embedding import apply_family_embedding_spec

        execution = apply_family_embedding_spec(
            project, embedding_revision, execution, historical=protocol_revision is not None
        )
    execution_build = None
    if pilot_revision is not None:
        execution_build = pilot_revision.saved_build.artifact_id.removesuffix("-result")
    elif review_revision is not None:
        from exp.optimize.evaluation.variance.build import plan_variance_build  # noqa: PLC0415

        actual_reviewer = execution.build.reviewer
        assert actual_reviewer is not None
        execution_build = plan_variance_build(
            project,
            execution.build.spec,
            code_revision=code_revision,
            reviewer_model=actual_reviewer.model,
            reviewer_configuration_sha256=sha256_json(actual_reviewer),
        ).identity
    binding = BootstrapResumeBinding(
        execution_source=current_bootstrap_execution_source(code_revision),
        original_inputs=original_inputs,
        original_revision=revision,
        producer_revision=code_revision,
        original_spec_sha256=sha256_json(inputs.spec),
        original_plan_sha256=sha256_json(inputs.plan),
        original_approval_sha256=sha256_json(inputs.approval),
        source_build=inputs.source_build,
        original_build=build,
        original_operation=operation,
        ledger_sha256=operation_identity(ledger),
        grants=summary.authorizations,
        ceiling_usd=summary.ceiling_usd,
        unpriced_attempts=summary.unpriced_attempt_ceiling,
        preparation_usd=dollars,
        preparation_unpriced_attempts=attempts,
        deadline=reviewer.deadline,
        execution_deadline=execution_deadline,
        execution_spec_sha256=sha256_json(execution),
        execution_window=execution_window,
        permitted_changes=permitted_changes,
        author_retries=author_retries,
        previous_binding=previous_binding,
        review_revision=review_revision,
        execution_build=execution_build,
        pilot_revision=pilot_revision,
        protocol_revision=protocol_revision,
        embedding_revision=embedding_revision,
    )
    _validate_predecessor(project, binding, spec)
    _validate_window(project, binding, spec)
    bootstrap_author_retries(project, binding)
    references = tuple(
        sorted(
            {
                original_inputs,
                *(() if previous_binding is None else (previous_binding,)),
                *author_retries,
                *(() if pilot_revision is None else pilot_revision.references),
                *(() if protocol_revision is None else (protocol_revision,)),
                *(() if embedding_revision is None else (embedding_revision,)),
                *(() if execution_window is None else (execution_window,)),
            },
            key=lambda row: row.artifact_id,
        )
    )
    for reference in references:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("bootstrap resume permission reference changed")
    identity = stable_id(RESUME_TYPE, binding)
    anchor = _registration_identity(binding)
    with CoordinationLock(
        project.artifacts.project_directory / (binding.original_operation + ".lock"), timeout=0
    ):
        if (
            previous_binding is not None
            and not project.artifacts.exists(anchor)
            and project.artifacts.exists(operation + "-result")
        ):
            raise ValueError("a completed preparation cannot register another review revision")
        if project.artifacts.exists(anchor):
            existing = ArtifactInput.model_validate_json(
                project.artifacts.read_bytes(anchor, "binding.json")
            )
            if existing.artifact_id != identity:
                raise ValueError("original operation already has a different resume binding")
        if not project.artifacts.exists(identity):
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type=RESUME_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=references,
                ),
                files={"binding.json": binding, "execution-spec.json": execution},
            )
        reference = artifact_input(project.artifacts.read(identity).manifest)
        if not project.artifacts.exists(anchor):
            project.artifacts.write_json(
                artifact_id=anchor,
                artifact_type=REGISTRATION_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(reference,),
                ),
                files={"binding.json": reference},
            )
        load_bootstrap_resume(project, reference, spec=spec, code_revision=code_revision)
        return reference


def bootstrap_execution_spec(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
    spec: BootstrapPreparationSpec,
) -> BootstrapPreparationSpec:
    """Verify only the registered review, pilot and operational deadline fields changed.

    Args:
        project: Owner of the saved binding and execution specification.
        binding: Verified original semantics and optional execution deadline.
        spec: Exact unchanged original preparation specification.

    Returns:
        Typed execution settings with all unrelated fields and study controls unchanged.

    Raises:
        ValueError: Saved execution settings exceed registered review or deadline changes.
    """
    reference = stable_id(RESUME_TYPE, binding)
    execution = type(spec).model_validate_json(
        project.artifacts.read_bytes(reference, "execution-spec.json")
    )
    expected = _execution_spec(
        spec, binding.execution_deadline, binding.review_revision, binding.pilot_revision
    )
    if binding.embedding_revision is not None:
        from exp.optimize.evaluation.variance.family_embedding import apply_family_embedding_spec

        expected = apply_family_embedding_spec(
            project, binding.embedding_revision, expected, historical=True
        )
    if sha256_json(execution) != binding.execution_spec_sha256 or execution != expected:
        raise ValueError(
            "bootstrap execution settings changed beyond permitted review or operational deadlines"
        )
    return execution


def load_bootstrap_resume(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    spec: BootstrapPreparationSpec,
    code_revision: str,
) -> BootstrapResumeBinding:
    """Verify original authority, ancestry and the actual current executing source."""
    return _load_bootstrap_resume(
        project,
        reference,
        spec=spec,
        code_revision=code_revision,
        historical=False,
    )


def _load_bootstrap_resume(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    spec: BootstrapPreparationSpec,
    code_revision: str,
    historical: bool,
) -> BootstrapResumeBinding:
    """Verify exact original semantics and spend history, plus explicit deadline-only permissions.

    Args:
        project: Owner of original evidence and the same preexisting operation ledger.
        reference: Exact immutable resume-binding artifact reference.
        spec: Unchanged original preparation settings.
        code_revision: Actual producer writing all new artifact envelopes.

    Returns:
        Verified semantic/source/authority binding, without renewing it on replay.

    Raises:
        ValueError: Inputs, provenance, producer, authority or operational permissions differ.
    """
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if manifest.artifact_type != RESUME_TYPE or artifact_input(manifest) != reference:
        raise ValueError("bootstrap resume binding reference differs from its immutable artifact")
    binding = BootstrapResumeBinding.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "binding.json")
    )
    if (
        binding.producer_revision != code_revision
        or manifest.code_revision != code_revision
        or reference.artifact_id != stable_id(RESUME_TYPE, binding)
    ):
        raise ValueError("bootstrap resume actual producer differs from its binding")
    if binding.execution_source is not None and (
        binding.execution_source.producer_revision != code_revision
        or (
            not historical
            and binding.execution_source != current_bootstrap_execution_source(code_revision)
        )
    ):
        raise ValueError("bootstrap execution source differs from its registered witness")
    anchor = project.artifacts.read(_registration_identity(binding)).manifest
    selected = ArtifactInput.model_validate_json(
        project.artifacts.read_bytes(anchor.artifact_id, "binding.json")
    )
    if (
        anchor.artifact_type != REGISTRATION_TYPE
        or anchor.inputs != (reference,)
        or selected != reference
    ):
        raise ValueError("original operation selected a different resume binding")
    if sha256_json(spec) != binding.original_spec_sha256:
        raise ValueError("bootstrap resume settings differ from the original specification")
    inputs = _load_inputs(project, binding.original_inputs)
    _, revision, build, operation, dollars, attempts = _original(inputs)
    if (
        (revision, build, operation, dollars, attempts)
        != (
            binding.original_revision,
            binding.original_build,
            binding.original_operation,
            binding.preparation_usd,
            binding.preparation_unpriced_attempts,
        )
        or inputs.source_build != binding.source_build
        or tuple(map(sha256_json, (inputs.spec, inputs.plan, inputs.approval)))
        != (
            binding.original_spec_sha256,
            binding.original_plan_sha256,
            binding.original_approval_sha256,
        )
        or spec.build.reviewer is None
        or binding.deadline != spec.build.reviewer.deadline
    ):
        raise ValueError("bootstrap resume original semantic identity differs")
    ledger = _ledger(project, binding.original_operation)
    summary = ledger.summary()
    if (
        operation_identity(ledger) != binding.ledger_sha256
        or summary.authorizations != binding.grants
        or summary.ceiling_usd != binding.ceiling_usd
        or summary.unpriced_attempt_ceiling != binding.unpriced_attempts
    ):
        raise ValueError("bootstrap resume original spend authority changed")
    if binding.execution_deadline is not None and not (
        manifest.created_at < binding.execution_deadline <= manifest.created_at + timedelta(hours=6)
    ):
        raise ValueError("bootstrap renewed execution watchdog lost its finite registered window")
    expected_inputs = tuple(
        sorted(
            {
                binding.original_inputs,
                *(() if binding.previous_binding is None else (binding.previous_binding,)),
                *binding.author_retries,
                *(() if binding.pilot_revision is None else binding.pilot_revision.references),
                *(() if binding.protocol_revision is None else (binding.protocol_revision,)),
                *(() if binding.embedding_revision is None else (binding.embedding_revision,)),
                *(() if binding.execution_window is None else (binding.execution_window,)),
            },
            key=lambda row: row.artifact_id,
        )
    )
    if manifest.inputs != expected_inputs:
        raise ValueError("bootstrap resume permission ancestry differs")
    _validate_predecessor(project, binding, spec, historical=historical)
    _validate_window(project, binding, spec)
    bootstrap_author_retries(project, binding)
    bootstrap_execution_spec(project, binding, spec)
    return binding


def read_historical_bootstrap_resume(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    spec: BootstrapPreparationSpec,
) -> BootstrapResumeBinding:
    """Read intrinsic saved provenance only; this grants no current execution or new lifecycle."""
    producer = project.artifacts.read(reference.artifact_id).manifest.code_revision
    return _load_bootstrap_resume(
        project, reference, spec=spec, code_revision=producer, historical=True
    )
