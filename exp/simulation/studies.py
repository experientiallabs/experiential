"""Typed six-study registrations over one injected maintained workflow executor."""

from collections.abc import Callable
from datetime import datetime
from typing import Literal, Self

from filelock import FileLock
from pydantic import Field, TypeAdapter, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import DualReport
from exp.common.project import ArtifactStore

Study = Literal["traces", "rules", "retrieval", "research", "continuity", "inspection"]


class StudyJob(ContractModel):
    """One registered workload; scientific inputs and results remain external artifacts.

    Attributes:
        id: Unique prospective job identity.
        study: One of the six reconstruction questions.
        treatment: Explicit arm name; a name alone does not prove an intervention.
        implementation: Target implementation identity.
        lineage: Dependency/independence cluster, not a claim of independent draws.
        draw: Prospective random/model draw index.
        phase: Development may select; confirmation must not feed construction.
        config: Frozen workload, input and information-control configuration.
        dependencies: Earlier jobs that must have completed operationally.
        planned_steps: Complete planned transition denominator.
        planned_judges: Complete planned judge-result denominator.
    """

    id: ArtifactId
    study: Study
    treatment: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    draw: int = Field(ge=0, strict=True)
    phase: Literal["development", "confirmation"]
    config: JsonObject
    dependencies: tuple[ArtifactId, ...] = ()
    planned_steps: int = Field(ge=0, strict=True)
    planned_judges: int = Field(ge=0, strict=True)


class StudyPlan(ContractModel):
    """Frozen ordered studies without another provider, session or scoring implementation.

    Attributes:
        id: Run identity, never reused with changed jobs.
        code_revision: Exact integrated source identity.
        jobs: Ordered independent/dependent work units.
        information_controls: Declared intervention and confirmation isolation rules.
    """

    id: ArtifactId
    code_revision: str = Field(min_length=1)
    jobs: tuple[StudyJob, ...] = Field(min_length=1)
    information_controls: JsonObject

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Reject duplicate jobs or forward/unregistered dependencies."""
        seen: set[str] = set()
        confirmation: set[str] = set()
        for job in self.jobs:
            if job.id in seen or not set(job.dependencies) <= seen:
                raise ValueError("study jobs require unique IDs and earlier dependencies")
            if job.phase == "development" and set(job.dependencies) & confirmation:
                raise ValueError("development cannot depend on confirmation outcomes")
            seen.add(job.id)
            if job.phase == "confirmation":
                confirmation.add(job.id)
        return self


class StudyJobResult(ContractModel):
    """Actual dual reports or construction artifacts, not a scientific success label.

    Attributes:
        reports: Evaluations returned by the same maintained workflow path.
        artifacts: Additional construction/revision/selection artifact IDs.
        delivered_inputs_sha256: Actual input identities for intervention auditing.
    """

    reports: tuple[DualReport, ...] = ()
    artifacts: tuple[ArtifactId, ...] = ()
    delivered_inputs_sha256: tuple[Sha256, ...] = ()


def _run_studies_locked(
    plan: StudyPlan,
    *,
    execute: Callable[[StudyJob], StudyJobResult],
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
) -> JsonObject:
    """Resume completed work; consumed intents never silently dispatch again."""

    def save(identity: str, value: JsonObject) -> None:
        """Publish immutable orchestration records through the project artifact owner."""
        artifacts.write_json(
            artifact_id=identity,
            artifact_type="reconstruction-study",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=clock(), inputs=(), code_revision=plan.code_revision
            ),
            files={"record.json": value},
        )

    def read(identity: str) -> JsonObject | None:
        """Distinguish an absent artifact from corruption or an invalid saved contract."""
        if not artifacts.exists(identity):
            return None
        return _JSON_OBJECT.validate_json(artifacts.read_bytes(identity, "record.json"))

    registration = plan.model_dump(mode="json")
    previous = read(plan.id + "-registration")
    if previous is None:
        save(plan.id + "-registration", registration)
    elif previous != registration:
        raise ValueError("study registration changed; use its original plan or a new run")
    rows: list[JsonObject] = []
    states: dict[str, str] = {}
    for job in plan.jobs:
        base = stable_id("study-job", {"plan": plan.id, "job": job.id})
        row = read(base + "-terminal")
        job_hash = sha256_json(job.model_dump(mode="json"))
        if row is not None:
            if row["job_sha256"] != job_hash:
                raise ValueError("saved study result belongs to another job")
        else:
            row = {
                "job_id": job.id,
                "job_sha256": job_hash,
                "study": job.study,
                "treatment": job.treatment,
                "phase": job.phase,
                "planned_steps": job.planned_steps,
                "planned_judges": job.planned_judges,
            }
            if read(base + "-intent") is not None:
                row["status"] = "ambiguous"
            elif any(states[name] != "completed" for name in job.dependencies):
                row["status"] = "blocked"
            else:
                save(base + "-intent", job.model_dump(mode="json"))
                try:
                    result = execute(job)
                    if (
                        len(result.reports) != job.planned_judges
                        or sum(report.planned_steps for report in result.reports)
                        != job.planned_steps
                    ):
                        raise ValueError("actual evaluation denominators differ from registration")
                    row.update(status="completed", result=result.model_dump(mode="json"))
                except Exception as error:  # noqa: BLE001 - retain failed jobs and continue independent work.
                    row.update(status="failure", error_type=type(error).__name__)
            if row["status"] not in ("ambiguous", "blocked"):
                save(base + "-terminal", row)
        states[job.id] = str(row["status"])
        rows.append(row)
    return {
        "plan_sha256": sha256_json(registration),
        "jobs": rows,
        "planned_steps": sum(job.planned_steps for job in plan.jobs),
        "planned_judges": sum(job.planned_judges for job in plan.jobs),
        "interpretation": "operational completion is not native accuracy or release approval",
    }


_JSON_OBJECT = TypeAdapter(JsonObject)


def run_studies(
    plan: StudyPlan,
    *,
    execute: Callable[[StudyJob], StudyJobResult],
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
) -> JsonObject:
    """Hold one project-scoped run lease; a live worker cannot be labelled crashed.

    Consumed intents without a result remain ambiguous and never authorize redispatch.
    A blocked dependent is computed on each read rather than made permanently terminal.
    """
    lock = artifacts.project_directory.resolve() / ("study-" + plan.id + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock, timeout=0):
        return _run_studies_locked(plan, execute=execute, artifacts=artifacts, clock=clock)
