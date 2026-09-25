"""Explicit new judging passes over an evaluation's unchanged saved rollouts."""

from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    sorted_unique_inputs,
    stable_id,
)
from exp.common.evaluations import EvaluationProtocol
from exp.common.models import CompletionCostReservation, ModelCatalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.prepare import PreparedModelEvaluation, read_evaluation_judge
from exp.optimize.router.automatic.provisional import _judge_request_reservation
from exp.optimize.router.judging.contracts import JudgeSetupArtifact
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy


class EvaluationJudgingRevision(ArtifactEnvelope):
    """A reviewed fresh judging pass, independent of the frozen simulation recipe.

    Attributes:
        revision_id: Content-derived pass identity.
        prepared_sha256: Exact original evaluation preparation, including prices and models.
        setup: Exact judge setup for fresh probe identities.
        request: New input reservation using the selected judge's declared capacity.
        previous: Prior judging revision, when another explicit retry is prepared.
    """

    revision_id: str
    prepared_sha256: Sha256
    setup: ArtifactInput
    request: CompletionCostReservation
    previous: ArtifactInput | None


def prepare_judging_revision(
    project: ProjectStore,
    prepared: PreparedModelEvaluation,
    catalog: ModelCatalog,
    *,
    previous: ArtifactInput | None = None,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Freeze a fresh judging pass without changing rollouts or making provider calls.

    Args:
        project: Owner of the saved evaluation.
        prepared: Original frozen simulation and judge selection.
        catalog: Credential-free active catalog used to verify model and pricing pins.
        previous: Exact earlier pass, or None for the first explicit judging retry.
        created_at: Timestamp for the reviewed pass.
        code_revision: Producer revision retained for audit.

    Returns:
        Immutable pass pointer ready for display and launch consent.

    Raises:
        ValueError: Model or prices drift, or the previous pass belongs to another evaluation.
    """
    selected = read_evaluation_judge(project, prepared.judge_setup)
    model, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(selected.judge_alias)
    if model != selected.judge_model:
        raise ValueError("judge catalog changed; prepare a new evaluation")
    request = _judge_request_reservation(
        caps,
        judge_model=model,
        maximum_input_tokens=None,
        maximum_output_tokens=prepared.judge_request.maximum_output_tokens,
        maximum_attempts=RetryPolicy().maximum_attempts,
    )
    capacity_fields = {"maximum_input_tokens", "estimated_maximum_call_cost_usd"}
    if request.model_dump(exclude=capacity_fields) != prepared.judge_request.model_dump(
        exclude=capacity_fields
    ):
        raise ValueError("judge prices or output settings changed; prepare a new evaluation")
    if previous is not None:
        read_judging_revision(project, prepared, previous)
    digest = sha256_json(prepared)
    identity = {
        "producer": code_revision,
        "prepared": digest,
        "request": request.model_dump(mode="json"),
        "previous": previous.model_dump(mode="json") if previous else None,
    }
    setup = selected.model_copy(
        update={
            "setup_id": stable_id("evaluation-judge-setup", identity),
            "created_at": created_at,
            "code_revision": code_revision,
        }
    )
    kind = project.artifacts.read(prepared.judge_setup.artifact_id).manifest.artifact_type
    _, setup_manifest = project.artifacts.write_or_replay(
        artifact_id=setup.setup_id,
        artifact_type=kind,
        envelope=setup,
        envelope_path="setup.json",
        envelope_type=type(setup),
        files={"setup.json": canonical_json_bytes(setup)},
    )
    setup_input = artifact_input(setup_manifest)
    revision = EvaluationJudgingRevision(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=sorted_unique_inputs(
            prepared.judge_setup, setup_input, *((previous,) if previous is not None else ())
        ),
        revision_id=stable_id("evaluation-judging", identity),
        prepared_sha256=digest,
        setup=setup_input,
        request=request,
        previous=previous,
    )
    _, manifest = project.artifacts.write_or_replay(
        artifact_id=revision.revision_id,
        artifact_type="evaluation-judging-revision",
        envelope=revision,
        envelope_path="judging.json",
        envelope_type=EvaluationJudgingRevision,
        files={"judging.json": canonical_json_bytes(revision)},
    )
    return artifact_input(manifest)


def read_judging_revision(
    project: ProjectStore, prepared: PreparedModelEvaluation, pointer: ArtifactInput
) -> EvaluationJudgingRevision:
    """Verify an immutable judging pass against its exact original evaluation.

    Args:
        project: Owner of the original preparation and retained judge setup artifacts.
        prepared: Frozen original model, protocol, and pricing settings.
        pointer: Exact reviewed judging revision to load.

    Returns:
        Verified revision whose rubric, prompt, model, and prices remain unchanged.

    Raises:
        ValueError: A pointer, semantic pin, or evaluation identity differs.
    """
    stored = project.artifacts.read(pointer.artifact_id)
    if (
        stored.manifest.artifact_type != "evaluation-judging-revision"
        or artifact_input(stored.manifest) != pointer
    ):
        raise ValueError("judging revision pointer changed")
    revision = EvaluationJudgingRevision.model_validate_json(
        project.artifacts.read_bytes(pointer.artifact_id, "judging.json")
    )
    if revision.revision_id != pointer.artifact_id or revision.prepared_sha256 != sha256_json(
        prepared
    ):
        raise ValueError("judging revision belongs to another evaluation")
    if prepared.judge_setup not in revision.inputs or revision.setup not in revision.inputs:
        raise ValueError("judging revision is missing its original judge setup")
    original = read_evaluation_judge(project, prepared.judge_setup)
    revised = read_evaluation_judge(project, revision.setup)
    identity_fields = {"setup_id", "created_at", "code_revision"}
    if original.model_dump(exclude=identity_fields) != revised.model_dump(exclude=identity_fields):
        raise ValueError("judging revision changed the frozen rubric, model, or prompt")
    capacity_fields = {"maximum_input_tokens", "estimated_maximum_call_cost_usd"}
    if revision.request.model_dump(exclude=capacity_fields) != prepared.judge_request.model_dump(
        exclude=capacity_fields
    ):
        raise ValueError("judging revision changed the frozen judge prices or output settings")
    return revision


def revised_judge_setup(
    project: ProjectStore, prepared: PreparedModelEvaluation, pointer: ArtifactInput
) -> tuple[JudgeSetupArtifact, CompletionCostReservation, EvaluationProtocol]:
    """Bind fresh request and exclusion identities while retaining verified prior judgments.

    Args:
        project: Owner of the saved judging pass.
        prepared: Original immutable evaluation preparation.
        pointer: Reviewed judging revision independent of the simulation recipe.

    Returns:
        Exact setup, expanded input reservation, and fresh judgment protocol identity.
    """
    revision = read_judging_revision(project, prepared, pointer)
    selected = read_evaluation_judge(project, revision.setup)
    protocol = prepared.setup.simulation_protocol.model_copy(
        update={"protocol_id": revision.revision_id}
    )
    return selected, revision.request, protocol
