"""Run a registered judge challenge study through the dual evaluator, then record it.

The study (:mod:`exp.optimize.evaluation.tasks.challenges`) fixes the saved outputs, their
registered edits and every judge arm before any judgment is requested. This module binds each
arm to one pinned catalog model and supplies the study's evaluator callback: every blinded
output is judged by the maintained dual evaluator
(:func:`exp.common.evaluations.reconstruction.evaluate`), whose intent, response and report are
saved per judgment, so an interrupted run resumes without buying a saved judgment again and a
completed study replays with no call or credential. Outputs are judged as saved workflow
evidence; a normalized JSON outcome has no evaluator path and is refused before any call.
"""

from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

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
from exp.common.evaluations.reconstruction import (
    EvaluationReferences,
    JudgeConfig,
    WorkflowEvidence,
    evaluate,
)
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.tasks.challenges import (
    CHALLENGE_PROTOCOL,
    JudgeChallengeStudy,
    record_challenge_study,
    run_challenge_study,
    saved_challenge_judgments,
    study_items,
)
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.evaluation.calibration import CalibrationJudgment
from exp.simulation.evaluation.tasks import load_evaluation_task

REFERENCES = EvaluationReferences(
    capability="description_only",
    description=(
        "A fixed saved output under a registered judge challenge; no native reference is "
        "supplied, so judge the recorded workflow on its own evidence."
    ),
)


class ChallengeArmModel(ContractModel):
    """The pinned catalog model that serves one judge arm.

    Attributes:
        arm_id: Judge arm of the study.
        alias: Configured catalog alias.
        model: Exact model and capability snapshot; its model ID is the arm's judge model.
    """

    arm_id: str = Field(min_length=1)
    alias: str = Field(min_length=1)
    model: ModelSnapshot


class ChallengeRunSpec(ContractModel):
    """A registered challenge study with the models, allowances and deadline that run it.

    Attributes:
        study: The registered fixed-output study.
        models: One pinned model per judge arm.
        catalog_sha256: Frozen credential-free catalog definition.
        output_tokens: Output allowance of every judgment.
        attempts_per_judgment: Physical attempts each judgment may use, retries included.
        deadline: Latest dispatch time; saved judgments replay after it.
    """

    study: JudgeChallengeStudy
    models: tuple[ChallengeArmModel, ...] = Field(min_length=1)
    catalog_sha256: Sha256
    output_tokens: int = Field(default=4096, ge=1, le=65536)
    attempts_per_judgment: int = Field(default=2, ge=1, le=4)
    deadline: AwareDatetime

    @model_validator(mode="after")
    def one_model_per_arm(self) -> Self:
        """Require exactly one model per arm, serving the arm's judge model identity."""
        arms = {arm.arm_id: arm for arm in self.study.judges}
        if sorted(row.arm_id for row in self.models) != sorted(arms):
            raise ValueError("bind exactly one model to every judge arm of the study")
        for row in self.models:
            if row.model.model_id != arms[row.arm_id].config.model_identity:
                raise ValueError(
                    f"arm {row.arm_id} judges with {arms[row.arm_id].config.model_identity}, "
                    f"but {row.alias} serves {row.model.model_id}"
                )
        return self

    def planned(self, arm_id: str) -> int:
        """Judgments one arm makes: every study output, once per repeat."""
        arm = next(arm for arm in self.study.judges if arm.arm_id == arm_id)
        return arm.repeats * len(study_items(self.study))

    @property
    def maximum_physical_attempts(self) -> int:
        """Shared ceiling over every arm's planned judgments and their retries."""
        return self.attempts_per_judgment * sum(self.planned(row.arm_id) for row in self.models)

    @property
    def identity(self) -> str:
        """Run identity: the registered study with its models and allowances."""
        return stable_id(
            "judge-challenge-run", {"spec": sha256_json(self), "protocol": CHALLENGE_PROTOCOL}
        )


def preflight_challenge_run(project: ProjectStore, spec: ChallengeRunSpec) -> None:
    """Check the task, outputs and models without credentials or provider calls.

    Raises:
        ValueError: The pack's task is not this project's task with the same verifier, an
            output is not saved workflow evidence, the catalog changed, or a model is not the
            pinned tool-capable model with enough capacity behind an admitted transport.
    """
    pack = spec.study.pack
    if load_evaluation_task(project.artifacts, pack.task).verifier != pack.candidate:
        raise ValueError("the pack's candidate verifier differs from its saved evaluation task")
    if any(not isinstance(item.outcome, WorkflowEvidence) for item in study_items(spec.study)):
        raise ValueError(
            "challenge runs judge saved workflow evidence through the dual evaluator; register "
            "outputs as WorkflowEvidence, not normalized JSON outcomes"
        )
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("the model catalog changed; register the run on the current catalog")
    for row in spec.models:
        snapshot, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(row.alias)
        if snapshot != row.model or caps.supports_tools is not True:
            raise ValueError(f"judge {row.alias} is not the pinned tool-capable model")
        if caps.context_window_tokens is None or (
            caps.maximum_output_tokens is not None
            and spec.output_tokens > caps.maximum_output_tokens
        ):
            raise ValueError(f"judge {row.alias} lacks declared context or output capacity")
        require_admitted_transport(catalog, row.alias, consumer="challenge judge")


def saved_challenge_run(project: ProjectStore, spec: ChallengeRunSpec) -> ArtifactInput | None:
    """The exact registered run's result, verified without credentials or calls.

    Returns:
        The immutable result reference, or None when this complete run has no result.
        Study-only results never establish the requested model and allowance provenance.

    Raises:
        ValueError: A result lacks its matching immutable run registration.
    """
    store = project.artifacts
    if not store.exists(spec.identity):
        return None
    registration = _saved_run_registration(project, spec)
    result = store.read(spec.identity)
    if registration is None or registration not in result.manifest.inputs:
        raise ValueError("saved challenge run has no matching immutable registration")
    if saved_challenge_judgments(project, spec.study, artifact_id=spec.identity) is None:
        return None
    return artifact_input(result.manifest)


def _saved_run_registration(project: ProjectStore, spec: ChallengeRunSpec) -> ArtifactInput | None:
    """Read and verify the complete run configuration frozen before dispatch."""
    identity = spec.identity + "-registration"
    if not project.artifacts.exists(identity):
        return None
    saved = ChallengeRunSpec.model_validate_json(project.artifacts.read_bytes(identity, "run.json"))
    if saved != spec:
        raise ValueError("saved challenge run differs from its full registration")
    return artifact_input(project.artifacts.read(identity).manifest)


def run_challenge(
    project: ProjectStore, spec: ChallengeRunSpec, *, code_revision: str
) -> ArtifactInput:
    """Judge every study output with every arm, then record the study and its report.

    Args:
        project: Project owning the saved task, catalog and results.
        spec: Registered study, models and allowances.
        code_revision: Producing implementation identity.

    Returns:
        The recorded study result.
    """
    saved = saved_challenge_run(project, spec)
    if saved is not None:
        return saved
    preflight_challenge_run(project, spec)
    store, identity = project.artifacts, spec.identity
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        registration = _saved_run_registration(project, spec)
        if registration is None:
            registration = artifact_input(
                store.write_json(
                    artifact_id=identity + "-registration",
                    artifact_type="judge-challenge-run-registration",
                    envelope=ArtifactEnvelope(
                        schema_version=1,
                        created_at=datetime.now(UTC),
                        code_revision=code_revision,
                        inputs=(spec.study.pack.task,),
                    ),
                    files={"run.json": spec.model_dump(mode="json")},
                )
            )
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            ledger=DispatchLedger(
                store.project_directory / (identity + ".sqlite"),
                spec.maximum_physical_attempts,
                sha256_json(spec),
            ),
            artifacts=store,
            code_revision=code_revision,
            deadline=spec.deadline,
        )
        catalog = load_model_catalog(project.model_catalog_path)
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        budget = RequestBudget(project, identity=identity, maximum_cost_usd=None)
        completions: dict[str, StructuredCompletion] = {}
        for row in spec.models:
            model = registry.resolve(row.alias)
            completions[row.arm_id] = StructuredCompletion(
                client=model.client,
                artifacts=store,
                plan=CompletionPlan(
                    model=model.snapshot,
                    capabilities=model.capabilities,
                    deadline=spec.deadline,
                    output_tokens=spec.output_tokens,
                    maximum_cost_per_call=1,
                    code_revision=code_revision,
                    output_protocol="tool",
                ),
                budget=budget,
            )

        def judge(config: JudgeConfig, blinded: JsonObject, key: str) -> CalibrationJudgment:
            """Judge one blinded output through the dual evaluator, replaying a saved report.

            Args:
                config: The arm's exact judge configuration.
                blinded: Judge-visible case: its blinded ID and outcome.
                key: Stable request key ``study:arm:repeat:item``.

            Returns:
                The judgment bound to the exact output, configuration and saved report.
            """
            evidence = WorkflowEvidence.model_validate(blinded["outcome"])
            report_id = stable_id("judge-challenge-judgment", {"run": identity, "key": key})
            report = evaluate(
                evidence,
                REFERENCES,
                judge=config,
                complete=completions[key.split(":")[1]],
                artifacts=store,
                artifact_id=report_id,
                clock=lambda: datetime.now(UTC),
                code_revision=code_revision,
            )
            return CalibrationJudgment(
                case_id=str(blinded["case_id"]),
                outcome_sha256=sha256_json(evidence),
                judge_sha256=report.judge_config_sha256,
                judge_completion_sha256=report.judge_completion_sha256,
                report=artifact_input(store.read(report_id).manifest),
                axes=report.axes,
            )

        judgments = run_challenge_study(spec.study, judge)
        return record_challenge_study(
            project,
            spec.study,
            judgments,
            code_revision=code_revision,
            artifact_id=identity,
            registration=registration,
        )


def challenge_attempts(spec: ChallengeRunSpec) -> dict[str, int]:
    """Physical attempts each arm's model may use, for the credential-free estimate."""
    attempts: dict[str, int] = {}
    for row in spec.models:
        attempts[row.alias] = attempts.get(row.alias, 0) + (
            spec.attempts_per_judgment * spec.planned(row.arm_id)
        )
    return attempts
