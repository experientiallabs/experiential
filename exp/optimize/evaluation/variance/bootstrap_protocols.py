"""Reviewed source-only successors preserve the original pilot and spend authority."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.rules.agent_loop import AGENT_LOOP_PROTOCOL
from exp.optimize.evaluation.rules.direct_research import direct_research_protocol_sha256
from exp.optimize.evaluation.rules.evidence_pages import EVIDENCE_DELIVERY_PROTOCOL
from exp.optimize.evaluation.rules.memory_stage import MEMORY_STAGE_PROTOCOL
from exp.optimize.evaluation.rules.rule_tags import rule_tagging_protocol_sha256
from exp.optimize.evaluation.rules.system_two import system_two_protocol_sha256
from exp.optimize.evaluation.variance.bootstrap_admission import BOOTSTRAP_ADMISSION_PROTOCOL
from exp.optimize.evaluation.variance.bootstrap_pilot import load_pilot_build
from exp.optimize.evaluation.variance.bootstrap_prompts import bootstrap_prompts_sha256
from exp.optimize.evaluation.variance.bootstrap_reads import BOOTSTRAP_READ_PROTOCOL

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
    from exp.optimize.evaluation.variance.bootstrap_resume import BootstrapResumeBinding

BOOTSTRAP_SOURCE_PROTOCOL = "bootstrap-source-revision-v3"
PROTOCOL_REVISION_TYPE = "bootstrap-protocol-revision"

_SOURCE_FILES = (
    "exp/optimize/evaluation/factory/authoring.py",
    "exp/optimize/evaluation/rules/agent_loop.py",
    "exp/optimize/evaluation/rules/evidence_pages.py",
    "exp/optimize/evaluation/rules/locator_inventory.py",
    "exp/optimize/evaluation/rules/system_one.py",
    "exp/optimize/evaluation/rules/system_two.py",
    "exp/optimize/evaluation/rules/memory_stage.py",
    "exp/optimize/evaluation/rules/governance.py",
    "exp/optimize/evaluation/rules/research.py",
    "exp/optimize/evaluation/rules/direct_research.py",
    "exp/optimize/evaluation/rules/rule_tags.py",
    "exp/optimize/evaluation/rules/stage.py",
    "exp/optimize/evaluation/rules/weighing.py",
    "exp/optimize/evaluation/rules/condense.py",
    "exp/optimize/evaluation/rules/inspectors.py",
    "exp/optimize/evaluation/variance/bootstrap_agents.py",
    "exp/optimize/evaluation/variance/bootstrap_prompts.py",
    "exp/optimize/evaluation/variance/bootstrap_reads.py",
    "exp/optimize/evaluation/variance/bootstrap_selection.py",
    "exp/optimize/evaluation/variance/bootstrap_protocols.py",
    "exp/optimize/evaluation/variance/bootstrap_resume.py",
    "exp/optimize/evaluation/variance/bootstrap_plan.py",
    "exp/optimize/evaluation/variance/pipeline_inputs.py",
    "exp/optimize/evaluation/variance/bootstrap_admission.py",
)


class BootstrapSourceSnapshot(ContractModel):
    """Exact implementation and semantic identities reviewed for preparation.

    Attributes:
        files: The closed source set and its actual installed bytes, without path aliases.
        protocols: Current prompt, reader, loop and evidence contract identities.
    """

    files: dict[str, Sha256]
    protocols: dict[str, str]


class BootstrapProtocolRevision(ContractModel):
    """A source-only continuation, admitting no configuration or evidence substitution.

    Attributes:
        protocol: Native successor contract version.
        previous_binding: Exact immutable predecessor.
        previous_producer_revision: That predecessor's real executor producer.
        producer_revision: Actual reviewed source producing the successor.
        execution_spec_sha256: Unchanged predecessor execution settings.
        saved_build: Actual historical completed build, not a copied report.
        build_producer_revision: Actual historical build producer.
        compatible_stages: Only the completed variance build may be reused.
        source: Exact current reviewed implementation and semantic fingerprints.
    """

    protocol: Literal[
        "bootstrap-source-revision-v1",
        "bootstrap-source-revision-v2",
        "bootstrap-source-revision-v3",
    ] = BOOTSTRAP_SOURCE_PROTOCOL
    previous_binding: ArtifactInput
    previous_producer_revision: str = Field(min_length=1)
    producer_revision: str = Field(min_length=1)
    execution_spec_sha256: Sha256
    saved_build: ArtifactInput
    build_producer_revision: str = Field(min_length=1)
    compatible_stages: tuple[Literal["variance_build"], ...] = Field(min_length=1, max_length=1)
    source: BootstrapSourceSnapshot


def current_bootstrap_source() -> BootstrapSourceSnapshot:
    """Fingerprint the closed source contract without credentials or dispatch."""
    root = Path(__file__).resolve().parents[4]
    return BootstrapSourceSnapshot(
        files={name: sha256_bytes((root / name).read_bytes()) for name in _SOURCE_FILES},
        protocols={
            "native_successor": BOOTSTRAP_SOURCE_PROTOCOL,
            "cost_admission": BOOTSTRAP_ADMISSION_PROTOCOL,
            "bootstrap": bootstrap_prompts_sha256(),
            "step_reader": BOOTSTRAP_READ_PROTOCOL,
            "agent_loop": AGENT_LOOP_PROTOCOL,
            "evidence_delivery": EVIDENCE_DELIVERY_PROTOCOL,
            "system_two": system_two_protocol_sha256(),
            "memory_stage": MEMORY_STAGE_PROTOCOL,
            "direct_research": direct_research_protocol_sha256(),
            "rule_questions": rule_tagging_protocol_sha256(),
        },
    )


def _predecessor(
    project: ProjectStore,
    reference: ArtifactInput,
) -> tuple[BootstrapResumeBinding, BootstrapPreparationSpec]:
    """Resolve original settings through the existing native ancestry owner."""
    # The resume owner imports this module; decoding runs after both owners finish loading.
    from exp.optimize.evaluation.variance.bootstrap_plan import (
        BootstrapPreparationSpec,  # noqa: PLC0415
    )
    from exp.optimize.evaluation.variance.bootstrap_resume import (  # noqa: PLC0415
        BootstrapResumeBinding,
        _load_bootstrap_resume,
        _load_inputs,
    )

    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference:
        raise ValueError("source successor predecessor reference changed")
    saved = BootstrapResumeBinding.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "binding.json")
    )
    inputs = _load_inputs(project, saved.original_inputs)
    spec = BootstrapPreparationSpec.model_validate(inputs.spec)
    return _load_bootstrap_resume(
        project,
        reference,
        spec=spec,
        code_revision=manifest.code_revision,
        historical=True,
    ), spec


def _validate_revision(
    project: ProjectStore,
    revision: BootstrapProtocolRevision,
    code_revision: str,
    *,
    verify_source: bool = True,
) -> BootstrapResumeBinding:
    """Reject unreviewed source, settings, producer or saved-stage substitutions."""
    parent, _ = _predecessor(project, revision.previous_binding)
    pilot = parent.pilot_revision
    if pilot is None:
        raise ValueError("source successor requires an existing completed pilot build binding")
    load_pilot_build(project, pilot)
    if (
        revision.producer_revision != code_revision
        or revision.producer_revision == parent.producer_revision
        or revision.previous_producer_revision != parent.producer_revision
        or revision.execution_spec_sha256 != parent.execution_spec_sha256
        or revision.saved_build != pilot.saved_build
        or revision.build_producer_revision != pilot.build_producer_revision
        or revision.compatible_stages != ("variance_build",)
        or (verify_source and revision.source != current_bootstrap_source())
    ):
        raise ValueError(
            "source successor differs from reviewed source, settings or build provenance"
        )
    return parent


def propose_bootstrap_protocol_revision(
    project: ProjectStore,
    *,
    previous_binding: ArtifactInput,
    code_revision: str,
) -> BootstrapProtocolRevision:
    """Describe concrete unchanged-stage compatibility for review, without writing authority."""
    parent, _ = _predecessor(project, previous_binding)
    pilot = parent.pilot_revision
    if pilot is None:
        raise ValueError("source successor requires an existing completed pilot build binding")
    return BootstrapProtocolRevision(
        previous_binding=previous_binding,
        previous_producer_revision=parent.producer_revision,
        producer_revision=code_revision,
        execution_spec_sha256=parent.execution_spec_sha256,
        saved_build=pilot.saved_build,
        build_producer_revision=pilot.build_producer_revision,
        compatible_stages=("variance_build",),
        source=current_bootstrap_source(),
    )


def register_bootstrap_protocol_revision(
    project: ProjectStore,
    revision: BootstrapProtocolRevision,
    *,
    code_revision: str,
) -> ArtifactInput:
    """Append exact reviewed source compatibility, never a grant or completed stage."""
    _validate_revision(project, revision, code_revision)
    identity = stable_id(PROTOCOL_REVISION_TYPE, revision)
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=PROTOCOL_REVISION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted(
                        (revision.previous_binding, revision.saved_build),
                        key=lambda r: r.artifact_id,
                    )
                ),
            ),
            files={"revision.json": revision},
        )
    reference = artifact_input(project.artifacts.read(identity).manifest)
    load_bootstrap_protocol_revision(project, reference, code_revision=code_revision)
    return reference


def load_bootstrap_protocol_revision(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    code_revision: str,
) -> BootstrapProtocolRevision:
    """Verify immutable source control, exact ancestry and the current executing implementation."""
    return _load_protocol_revision(
        project, reference, code_revision=code_revision, verify_source=True
    )


def _load_protocol_revision(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    code_revision: str,
    verify_source: bool,
) -> BootstrapProtocolRevision:
    """Verify historical controls intrinsically, comparing current source only for execution."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    revision = BootstrapProtocolRevision.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "revision.json")
    )
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != PROTOCOL_REVISION_TYPE
        or manifest.code_revision != code_revision
        or reference.artifact_id != stable_id(PROTOCOL_REVISION_TYPE, revision)
        or manifest.inputs
        != tuple(
            sorted((revision.previous_binding, revision.saved_build), key=lambda r: r.artifact_id)
        )
    ):
        raise ValueError("source successor control identity, producer or ancestry differs")
    _validate_revision(project, revision, code_revision, verify_source=verify_source)
    return revision


def validate_protocol_successor(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
    parent: BootstrapResumeBinding,
    *,
    historical: bool = False,
) -> None:
    """Allow source changes only, retaining the exact parent pilot, watchdog and authority."""
    assert binding.protocol_revision is not None
    revision = _load_protocol_revision(
        project,
        binding.protocol_revision,
        code_revision=binding.producer_revision,
        verify_source=not historical,
    )
    if (
        revision.previous_binding != binding.previous_binding
        or revision.previous_producer_revision != parent.producer_revision
        or binding.review_revision != parent.review_revision
        or binding.pilot_revision != parent.pilot_revision
        or binding.embedding_revision != parent.embedding_revision
        or binding.build_identity != parent.build_identity
        or binding.execution_spec_sha256 != parent.execution_spec_sha256
        or binding.execution_deadline != parent.execution_deadline
        or binding.execution_window != parent.execution_window
        or set(binding.permitted_changes)
        != set(parent.permitted_changes) | {"preparation_protocol_revision"}
    ):
        raise ValueError("source successor changes unreviewed settings, stage reuse or lifecycle")


def read_historical_bootstrap_protocol(
    project: ProjectStore, reference: ArtifactInput
) -> BootstrapProtocolRevision:
    """Validate original source evidence intrinsically without authorizing a current executor."""
    producer = project.artifacts.read(reference.artifact_id).manifest.code_revision
    return _load_protocol_revision(project, reference, code_revision=producer, verify_source=False)
