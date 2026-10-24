"""Study registrations preserve scope, failed denominators and consumed work."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, stable_id
from exp.common.project import ProjectConfig, ProjectStore
from exp.simulation.studies import StudyJob, StudyJobResult, StudyPlan, run_studies


def job(identity: str, **changes: str | int | tuple[str, ...]) -> StudyJob:
    """Return a zero-prediction construction job without pretending to evaluate behavior."""
    values: JsonObject = {
        "id": identity,
        "study": "traces",
        "treatment": "construction-only",
        "implementation": "toy",
        "lineage": "shared-toy",
        "draw": 0,
        "phase": "development",
        "config": {},
        "planned_steps": 0,
        "planned_judges": 0,
    }
    return StudyJob.model_validate({**values, **changes})


def plan(*jobs: StudyJob) -> StudyPlan:
    """Bind ordered work and information restrictions before execution."""
    return StudyPlan(
        id="smoke",
        code_revision="fixture",
        jobs=jobs,
        information_controls={"confirmation_to_development": False},
    )


def test_resume_does_not_repeat_completed_or_failed_jobs(tmp_path: Path) -> None:
    """A failed denominator is retained; independent work can still finish."""
    project = ProjectStore(tmp_path, "studies")
    project.initialize(ProjectConfig(project_id="studies"))
    calls: list[str] = []

    def execute(value: StudyJob) -> StudyJobResult:
        """Fail the first job and retain the planned registration."""
        calls.append(value.id)
        if value.id == "failed":
            raise ValueError("fixture failure")
        return StudyJobResult()

    registration = plan(job("failed"), job("blocked", dependencies=("failed",)), job("independent"))
    kwargs = {
        "execute": execute,
        "artifacts": project.artifacts,
        "clock": lambda: datetime.now(UTC),
    }
    first = run_studies(registration, **kwargs)
    assert run_studies(registration, **kwargs) == first
    assert calls == ["failed", "independent"]
    rows = first["jobs"]
    assert isinstance(rows, list)
    assert [row["status"] for row in rows if isinstance(row, dict)] == [
        "failure",
        "blocked",
        "completed",
    ]


def test_consumed_intent_is_ambiguous_without_dispatch(tmp_path: Path) -> None:
    """A crash after admission cannot become another silent purchase on resume."""
    project = ProjectStore(tmp_path, "studies")
    project.initialize(ProjectConfig(project_id="studies"))
    registration = plan(job("crashed"))
    base = stable_id("study-job", {"plan": registration.id, "job": "crashed"})
    project.artifacts.write_json(
        artifact_id=base + "-intent",
        artifact_type="reconstruction-study",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), inputs=(), code_revision="fixture"
        ),
        files={"record.json": registration.jobs[0].model_dump(mode="json")},
    )
    calls: list[str] = []
    result = run_studies(
        registration,
        execute=lambda value: calls.append(value.id) or StudyJobResult(),
        artifacts=project.artifacts,
        clock=lambda: datetime.now(UTC),
    )
    assert not calls
    rows = result["jobs"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["status"] == "ambiguous"


def test_confirmation_cannot_feed_development() -> None:
    """Confirmation outcomes are unavailable as a registered repair dependency."""
    with pytest.raises(ValueError, match="cannot depend on confirmation"):
        plan(
            job("confirmation", phase="confirmation"), job("repair", dependencies=("confirmation",))
        )


def test_live_study_lease_cannot_be_mistaken_for_a_crash(tmp_path: Path) -> None:
    """A second coordinator cannot publish an ambiguous terminal over an active worker."""
    from filelock import FileLock, Timeout

    project = ProjectStore(tmp_path, "studies")
    project.initialize(ProjectConfig(project_id="studies"))
    calls: list[str] = []
    lock = project.artifacts.project_directory.resolve() / "study-smoke.lock"
    with FileLock(lock, timeout=0), pytest.raises(Timeout):
        run_studies(
            plan(job("active")),
            execute=lambda value: calls.append(value.id) or StudyJobResult(),
            artifacts=project.artifacts,
            clock=lambda: datetime.now(UTC),
        )
    assert not calls and not project.artifacts.list_ids()
