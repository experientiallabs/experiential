"""Loopback-only review adapter over the canonical task and judging services.

The TypeScript review app never edits task, rubric, score-history, or calibration artifacts
directly. It calls this local adapter, which delegates every mutation to the W5 and W6 services.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import SplitResult, urlsplit
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from wmo.common.core.artifacts import ArtifactId
from wmo.common.judging import (
    CalibrationError,
    CalibrationReport,
    HumanScoreHistory,
    HumanScoreReview,
    JudgeCalibration,
    JudgeCalibrationService,
    JudgeScoreObservation,
    Judgment,
    RubricReview,
    RubricReviewDraft,
    RubricReviewError,
    ScoreAnchor,
)
from wmo.common.judging.rubric import RubricDimension
from wmo.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.rollouts import RolloutArtifact
from wmo.common.tasks import LoadedTaskSet, TaskCase, resolve_task_set
from wmo.simulation.mining.coverage import CoverageReport

ScoreValue = Literal[0, 1, 2, 3, 4, 5]
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class ReviewServerError(ValueError):
    """Raised when a local review request cannot be mapped to canonical evidence."""


class ReviewRolloutView(BaseModel):
    """One locally stored rollout paired with its visible judgment and task lineage."""

    rollout: RolloutArtifact
    lineage_id: ArtifactId | None = None
    judgment: Judgment | None = None


class ReviewSnapshot(BaseModel):
    """The complete local review state rendered by the TypeScript application."""

    project_id: str
    local_data_notice: str
    task_set: LoadedTaskSet
    coverage: CoverageReport | None = None
    rubric_review: RubricReviewDraft
    human_score_history: HumanScoreHistory
    rollouts: tuple[ReviewRolloutView, ...]
    calibration_reports: tuple[CalibrationReport, ...]
    calibrations: tuple[JudgeCalibration, ...]


class RubricMutation(BaseModel):
    """One UI request that the canonical rubric-review service validates and persists."""

    dimension_id: ArtifactId | None = None
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, min_length=1)
    anchors: tuple[ScoreAnchor, ...] | None = None
    dimension: RubricDimension | None = None
    dimensions: tuple[RubricDimension, ...] | None = None
    dimension_ids: tuple[ArtifactId, ...] | None = None
    confirmed: bool = False


class ScoreOverride(BaseModel):
    """A human score or correction for one stored rollout and rubric dimension."""

    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    score: ScoreValue
    submission_id: UUID


class CalibrationApproval(BaseModel):
    """Explicit confirmation fields for one persisted calibration-report approval."""

    confirmed: bool = False
    accept_insufficient_risk: bool = False


class ReviewMutationResponse(BaseModel):
    """Updated local state plus a precise result message for one review write."""

    snapshot: ReviewSnapshot
    notice: str


@dataclass(frozen=True)
class _RolloutRecord:
    """One parsed rollout with the immutable artifact directory that owns it."""

    artifact_id: ArtifactId
    rollout: RolloutArtifact


@dataclass(frozen=True)
class _JudgmentRecord:
    """One parsed judgment with the immutable artifact directory that owns it."""

    artifact_id: ArtifactId
    judgment: Judgment


class ReviewApplication:
    """Maps loopback UI actions to the existing W5 task and W6 judging services."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        code_revision: str,
        task_set_id: ArtifactId | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create one local adapter without reading credentials or starting a provider client.

        Args:
            store: One initialized project-local WMO store.
            code_revision: Exact current repository revision for newly produced artifacts.
            task_set_id: Optional explicit task-set selection when no saved draft exists.
            clock: Optional deterministic clock used by review and label persistence.
        """
        if not code_revision.strip():
            raise ReviewServerError("review producer revision must be nonempty")
        self._store = store
        self._code_revision = code_revision
        self._task_set_id = task_set_id
        self._clock = _utc_now if clock is None else clock

    def snapshot(self) -> ReviewSnapshot:
        """Return the current verified task, review, score, rollout, and calibration state.

        Returns:
            A presentation-ready view composed only from local immutable artifacts and review.json.

        Raises:
            ReviewServerError: The project has no unambiguous task set or saved review state is
                invalid.
        """
        loaded = self._loaded_task_set()
        rubric_review = self._rubric_review(loaded)
        score_review = HumanScoreReview.open(self._store)
        coverage = _load_coverage(self._store, loaded)
        rollout_records = _load_rollout_records(self._store)
        judgments = _load_judgment_records(self._store)
        task_by_id = {task.task_id: task for task in loaded.tasks}
        finalized = rubric_review.draft.finalized_rubric
        judgment_by_rollout = _judgment_by_rollout(
            judgments,
            rubric_id=None if finalized is None else finalized.rubric_id,
        )
        rollouts = tuple(
            ReviewRolloutView(
                rollout=record.rollout,
                lineage_id=_lineage_for_rollout(record.rollout, task_by_id),
                judgment=judgment_by_rollout.get(record.rollout.rollout_id),
            )
            for record in sorted(rollout_records.values(), key=lambda item: item.rollout.rollout_id)
        )
        reports = tuple(
            report
            for report in _load_calibration_reports(self._store)
            if finalized is not None and report.rubric_id == finalized.rubric_id
        )
        calibrations = tuple(
            calibration
            for calibration in _load_calibrations(self._store)
            if finalized is not None and calibration.rubric_id == finalized.rubric_id
        )
        return ReviewSnapshot(
            project_id=self._store.paths.project_id,
            local_data_notice=(
                "This review stays on this machine. WMO reads local artifacts and writes only "
                "the project review draft plus immutable approved artifacts. Browser restarts "
                "resume review.json; remove the local project directory yourself to clear all "
                "project data."
            ),
            task_set=loaded,
            coverage=coverage,
            rubric_review=rubric_review.draft,
            human_score_history=score_review.history,
            rollouts=rollouts,
            calibration_reports=reports,
            calibrations=calibrations,
        )

    def rubric_action(
        self,
        action: Literal["accept", "reject", "edit", "add", "replace_all", "order", "finalize"],
        mutation: RubricMutation,
    ) -> ReviewSnapshot:
        """Apply one validated rubric change through W6's persisted review service.

        Args:
            action: The exact service transition requested by the local interface.
            mutation: Typed UI fields consumed by that transition.

        Returns:
            The complete updated local review state.

        Raises:
            ReviewServerError: The request omits required fields or rejects finalization consent.
        """
        review = self._rubric_review(self._loaded_task_set())
        match action:
            case "accept":
                review.accept(_required_id(mutation.dimension_id, "dimension_id"))
            case "reject":
                review.reject(_required_id(mutation.dimension_id, "dimension_id"))
            case "edit":
                review.edit(
                    _required_id(mutation.dimension_id, "dimension_id"),
                    name=mutation.name,
                    description=mutation.description,
                    anchors=mutation.anchors,
                )
            case "add":
                review.add(_required_dimension(mutation.dimension))
            case "replace_all":
                review.replace_all(_required_dimensions(mutation.dimensions))
            case "order":
                review.order(_required_ids(mutation.dimension_ids, "dimension_ids"))
            case "finalize":
                if not mutation.confirmed:
                    raise ReviewServerError("finalizing a rubric requires explicit confirmation")
                review.finalize()
        return self.snapshot()

    def score_override(self, override: ScoreOverride) -> ReviewMutationResponse:
        """Append or correct one human score and refresh a ready calibration report.

        Args:
            override: Existing rollout, frozen lineage, rubric dimension, and zero-to-five score.

        Returns:
            The refreshed local review state and the calibration refresh outcome.

        Raises:
            ReviewServerError: The rubric, rollout, lineage, or dimension cannot be verified.
        """
        review = self._rubric_review(self._loaded_task_set())
        rubric = review.draft.finalized_rubric
        if rubric is None:
            raise ReviewServerError("finalize the rubric before adding calibration labels")
        if override.dimension_id not in {item.dimension_id for item in rubric.dimensions}:
            raise ReviewServerError("the score dimension is not part of the finalized rubric")
        record = _load_rollout_records(self._store).get(override.rollout_id)
        if record is None:
            raise ReviewServerError("the score target rollout is not stored in this project")
        task = _task_for_rollout(record.rollout, self._loaded_task_set().tasks)
        if task.lineage_group_id != override.lineage_id:
            raise ReviewServerError("the supplied lineage does not match the rollout's frozen task")
        judgment = _judgment_by_rollout(
            _load_judgment_records(self._store),
            rubric_id=rubric.rubric_id,
        ).get(
            override.rollout_id,
        )
        if judgment is None or judgment.rubric_id != rubric.rubric_id:
            raise ReviewServerError("the rollout needs a stored judgment for this finalized rubric")
        if override.dimension_id not in {item.dimension_id for item in judgment.dimensions}:
            raise ReviewServerError("the stored judgment does not score the requested dimension")

        score_review = HumanScoreReview.open(self._store)
        score_review.upsert(
            rubric_id=rubric.rubric_id,
            rollout_id=override.rollout_id,
            lineage_id=override.lineage_id,
            dimension_id=override.dimension_id,
            score=override.score,
            submission_id=str(override.submission_id),
            created_at=self._now(),
        )
        notice = self._refresh_calibration(rubric_id=rubric.rubric_id, score_review=score_review)
        return ReviewMutationResponse(snapshot=self.snapshot(), notice=notice)

    def approve_calibration(
        self,
        report_id: ArtifactId,
        approval: CalibrationApproval,
    ) -> ReviewMutationResponse:
        """Approve one exact visible W6 report only after explicit human confirmation.

        Args:
            report_id: Visible persisted calibration report selected for approval.
            approval: Explicit confirmation and low-sample risk-acceptance fields.

        Returns:
            The refreshed local review state and immutable approval notice.

        Raises:
            ReviewServerError: Confirmation, finalization, report, or risk requirements fail.
        """
        if not approval.confirmed:
            raise ReviewServerError("calibration approval requires explicit confirmation")
        review = self._rubric_review(self._loaded_task_set())
        rubric = review.draft.finalized_rubric
        if rubric is None:
            raise ReviewServerError("finalize the rubric before approving judge calibration")
        report = next(
            (
                item
                for item in _load_calibration_reports(self._store)
                if item.report_id == report_id and item.rubric_id == rubric.rubric_id
            ),
            None,
        )
        if report is None:
            raise ReviewServerError("the selected calibration report is missing for this rubric")
        if report.status == "insufficient" and not approval.accept_insufficient_risk:
            raise ReviewServerError(
                "fewer than ten eligible rollouts require explicit insufficient-risk acceptance"
            )
        if report.status != "insufficient" and approval.accept_insufficient_risk:
            raise ReviewServerError(
                "insufficient-risk acceptance applies only to an insufficient report"
            )
        service = JudgeCalibrationService()
        calibration = service.approve(
            self._store,
            report,
            approved_at=self._now(),
            accept_insufficient_labels=approval.accept_insufficient_risk,
        )
        stored = service.write_calibration(
            self._store,
            report=report,
            calibration=calibration,
        )
        return ReviewMutationResponse(
            snapshot=self.snapshot(),
            notice=f"Human-calibrated judge artifact {stored.calibration_id} approved locally.",
        )

    def _loaded_task_set(self) -> LoadedTaskSet:
        """Resolve the saved or explicitly selected task set for this local review session."""
        try:
            persisted = _persisted_task_set_id(self._store)
            if persisted is not None and self._task_set_id not in {None, persisted}:
                raise ReviewServerError(
                    "--task-set conflicts with the task set already persisted in review.json"
                )
            return resolve_task_set(self._store.artifacts, persisted or self._task_set_id)
        except ArtifactCorruptionError as exc:
            raise ReviewServerError(str(exc)) from exc

    def _rubric_review(self, task_set: LoadedTaskSet) -> RubricReview:
        """Open W6's resumable draft against the immutable W5 task-set identity."""
        try:
            return RubricReview.open(
                self._store,
                source_task_set_id=task_set.task_set.task_set_id,
                code_revision=self._code_revision,
                clock=self._clock,
            )
        except RubricReviewError as exc:
            raise ReviewServerError(str(exc)) from exc

    def _refresh_calibration(
        self,
        *,
        rubric_id: ArtifactId,
        score_review: HumanScoreReview,
    ) -> str:
        """Write a new calibration report when a provisional report already fixes its evidence.

        The adapter never invents a lineage split or judge evidence. It only reuses the latest
        persisted report's exact split and observations after W6 has durably frozen the labels.
        """
        reports = tuple(
            report
            for report in _load_calibration_reports(self._store)
            if report.rubric_id == rubric_id
        )
        if not reports:
            return (
                "Score saved locally. A calibration report will refresh after the judge and "
                "frozen lineage split are available."
            )
        source_report = max(reports, key=lambda item: (item.created_at, item.report_id))
        created_at = self._now()
        try:
            label_set = score_review.finalize(
                rubric_id=rubric_id,
                code_revision=self._code_revision,
                created_at=created_at,
            )
            observations = _observations_for_active_scores(
                self._store,
                history=score_review.history,
                rubric_id=rubric_id,
            )
            service = JudgeCalibrationService()
            report = service.build_report(
                self._store,
                rubric_id=rubric_id,
                label_set_id=label_set.label_set_id,
                router_lineage_split_id=source_report.router_lineage_split_id,
                observations=observations,
                created_at=created_at,
                code_revision=self._code_revision,
            )
            stored = service.write_report(self._store, report)
        except (CalibrationError, ProjectStoreError, ValueError) as exc:
            return f"Score saved locally. Calibration refresh needs more verified evidence: {exc}"
        return (
            "Score saved and calibration report "
            f"{stored.report_id} refreshed with status {stored.status}."
        )

    def _now(self) -> datetime:
        """Return a timezone-aware timestamp before a local mutable write occurs."""
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ReviewServerError("review clock must return a timezone-aware time")
        return value


def create_review_app(
    root: Path,
    project_id: str,
    *,
    code_revision: str,
    task_set_id: ArtifactId | None = None,
    port: int = 8017,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Create the loopback API over one project-local WMO review store.

    Args:
        root: Local `.wmo` root containing the project directory.
        project_id: One validated project identifier below the local root.
        code_revision: Exact current producer revision for newly approved artifacts.
        task_set_id: Optional task-set selection before a draft has persisted one.
        port: Exact loopback port accepted in the HTTP Host boundary.
        clock: Optional deterministic clock for tests and local review writes.

    Returns:
        A FastAPI application with no auth, tenant, provider, or deployment integration.

    Raises:
        ReviewServerError: The requested local loopback port is outside the valid range.
    """
    if not 1 <= port <= 65535:
        raise ReviewServerError("review port must be between 1 and 65535")
    application = ReviewApplication(
        ProjectStore(root, project_id),
        code_revision=code_revision,
        task_set_id=task_set_id,
        clock=clock,
    )
    app = FastAPI(
        title="WMO local review",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def enforce_loopback_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Reject DNS rebinding and cross-origin requests before any local evidence read."""
        try:
            _validate_loopback_request(request, port=port)
        except ReviewServerError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a local liveness response without reading customer evidence."""
        return {"status": "ok"}

    @app.get("/api/review", response_model=ReviewSnapshot)
    def get_review() -> ReviewSnapshot:
        """Return the current local review workspace."""
        try:
            return application.snapshot()
        except _LOCAL_REVIEW_ERRORS as exc:
            raise _http_error(exc) from exc

    @app.post("/api/review/rubric/{action}", response_model=ReviewSnapshot)
    def mutate_rubric(
        action: Literal["accept", "reject", "edit", "add", "replace_all", "order", "finalize"],
        mutation: RubricMutation,
    ) -> ReviewSnapshot:
        """Apply one canonical rubric-review transition to review.json."""
        try:
            return application.rubric_action(action, mutation)
        except _LOCAL_REVIEW_ERRORS as exc:
            raise _http_error(exc) from exc

    @app.post("/api/review/score", response_model=ReviewMutationResponse)
    def override_score(override: ScoreOverride) -> ReviewMutationResponse:
        """Persist an append-only human score and refresh reusable calibration evidence."""
        try:
            return application.score_override(override)
        except _LOCAL_REVIEW_ERRORS as exc:
            raise _http_error(exc) from exc

    @app.post(
        "/api/review/calibration/{report_id}/approve",
        response_model=ReviewMutationResponse,
    )
    def approve_calibration(
        report_id: ArtifactId,
        approval: CalibrationApproval,
    ) -> ReviewMutationResponse:
        """Persist an explicitly confirmed human-calibrated W6 judge artifact."""
        try:
            return application.approve_calibration(report_id, approval)
        except _LOCAL_REVIEW_ERRORS as exc:
            raise _http_error(exc) from exc

    return app


_LOCAL_REVIEW_ERRORS = (
    ArtifactCorruptionError,
    CalibrationError,
    ProjectStoreError,
    ReviewServerError,
    RubricReviewError,
    ValueError,
)


def _http_error(error: Exception) -> HTTPException:
    """Translate one expected local validation failure into a readable HTTP response."""
    return HTTPException(status_code=400, detail=str(error))


def _validate_loopback_request(request: Request, *, port: int) -> None:
    """Require one exact loopback Host and same-origin Origin or Referer when supplied."""
    host_values = _raw_header_values(request, b"host")
    if len(host_values) != 1:
        raise ReviewServerError("local review requests require exactly one Host header")
    allowed_hosts = {
        f"127.0.0.1:{port}",
        f"localhost:{port}",
        f"[::1]:{port}",
    }
    host = host_values[0].casefold()
    if host not in allowed_hosts:
        raise ReviewServerError("local review Host must be loopback with the expected port")
    expected = _origin_tuple(urlsplit(f"http://{host}"), label="Host")
    for header_name in (b"origin", b"referer"):
        values = _raw_header_values(request, header_name)
        if len(values) > 1:
            raise ReviewServerError(
                f"local review requests may include at most one {header_name.decode()} header"
            )
        if values:
            supplied = _origin_tuple(
                urlsplit(values[0]),
                label=header_name.decode().title(),
            )
            if supplied != expected:
                raise ReviewServerError(
                    f"local review {header_name.decode()} must match the loopback Host origin"
                )


def _raw_header_values(request: Request, name: bytes) -> tuple[str, ...]:
    """Return every decoded raw request-header value for duplicate rejection."""
    return tuple(
        value.decode("latin-1").strip()
        for key, value in request.scope.get("headers", ())
        if key.lower() == name
    )


def _origin_tuple(value: SplitResult, *, label: str) -> tuple[str, str, int]:
    """Parse one strict local HTTP origin for Host, Origin, and Referer comparison."""
    try:
        port = value.port
    except ValueError as exc:
        raise ReviewServerError(f"local review {label} has an invalid port") from exc
    if (
        value.scheme.casefold() != "http"
        or value.hostname is None
        or port is None
        or value.username is not None
        or value.password is not None
    ):
        raise ReviewServerError(f"local review {label} must be an explicit HTTP origin")
    return value.scheme.casefold(), value.hostname.casefold(), port


def _persisted_task_set_id(store: ProjectStore) -> ArtifactId | None:
    """Return the task-set identity already selected by a valid resumable review draft."""
    review = store.read_review()
    if review is None:
        return None
    if not isinstance(review, dict):
        raise ReviewServerError("review.json must be a JSON object")
    saved = review.get("rubric_review")
    if saved is None:
        return None
    try:
        return RubricReviewDraft.model_validate(saved).source_task_set_id
    except ValueError as exc:
        raise ReviewServerError("review.json contains an invalid rubric review draft") from exc


def _load_coverage(store: ProjectStore, task_set: LoadedTaskSet) -> CoverageReport | None:
    """Load coverage only from the selected immutable W5 task-set artifact."""
    envelope = task_set.task_set
    if envelope.coverage_path is None:
        return None
    try:
        return CoverageReport.model_validate_json(
            store.artifacts.read_bytes(envelope.task_set_id, envelope.coverage_path)
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise ReviewServerError("the selected task-set coverage report is corrupt") from exc


def _load_rollout_records(store: ProjectStore) -> dict[ArtifactId, _RolloutRecord]:
    """Discover persisted rollout records without rereading raw traces or provider data."""
    records: dict[ArtifactId, _RolloutRecord] = {}
    for artifact_id in store.artifacts.list_ids():
        stored = store.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "rollout":
            continue
        try:
            rollout = RolloutArtifact.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "rollout.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ReviewServerError(f"rollout artifact {artifact_id} is corrupt") from exc
        if rollout.artifact_id != artifact_id:
            raise ReviewServerError(
                f"rollout artifact {artifact_id} does not match its record identity"
            )
        if rollout.rollout_id in records:
            raise ReviewServerError(f"rollout ID {rollout.rollout_id} is stored more than once")
        records[rollout.rollout_id] = _RolloutRecord(
            artifact_id=artifact_id,
            rollout=rollout,
        )
    return records


def _load_judgment_records(store: ProjectStore) -> dict[ArtifactId, _JudgmentRecord]:
    """Discover immutable W6 judgments from their own artifact owner only."""
    records: dict[ArtifactId, _JudgmentRecord] = {}
    for artifact_id in store.artifacts.list_ids():
        stored = store.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judgment":
            continue
        try:
            judgment = Judgment.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "judgment.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ReviewServerError(f"judgment artifact {artifact_id} is corrupt") from exc
        if judgment.judgment_id != artifact_id:
            raise ReviewServerError(
                f"judgment artifact {artifact_id} does not match its record identity"
            )
        records[judgment.judgment_id] = _JudgmentRecord(
            artifact_id=artifact_id,
            judgment=judgment,
        )
    return records


def _load_calibration_reports(store: ProjectStore) -> tuple[CalibrationReport, ...]:
    """Read immutable W6 calibration reports in deterministic review order."""
    reports: list[CalibrationReport] = []
    for artifact_id in store.artifacts.list_ids():
        stored = store.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judge-calibration-report":
            continue
        try:
            report = CalibrationReport.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "report.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ReviewServerError(
                f"judge-calibration report artifact {artifact_id} is corrupt"
            ) from exc
        if report.report_id != artifact_id:
            raise ReviewServerError(
                f"judge-calibration report {artifact_id} does not match its record identity"
            )
        reports.append(report)
    return tuple(sorted(reports, key=lambda item: (item.created_at, item.report_id)))


def _load_calibrations(store: ProjectStore) -> tuple[JudgeCalibration, ...]:
    """Read immutable W6 judge calibrations and fail closed on semantic corruption."""
    calibrations: list[JudgeCalibration] = []
    for artifact_id in store.artifacts.list_ids():
        stored = store.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judge-calibration":
            continue
        try:
            calibration = JudgeCalibration.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "calibration.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ReviewServerError(f"judge-calibration artifact {artifact_id} is corrupt") from exc
        if calibration.calibration_id != artifact_id:
            raise ReviewServerError(
                f"judge-calibration {artifact_id} does not match its record identity"
            )
        calibrations.append(calibration)
    return tuple(sorted(calibrations, key=lambda item: (item.created_at, item.calibration_id)))


def _judgment_by_rollout(
    records: dict[ArtifactId, _JudgmentRecord],
    *,
    rubric_id: ArtifactId | None = None,
) -> dict[ArtifactId, Judgment]:
    """Choose the latest deterministic judgment for each rollout and optional rubric."""
    by_rollout: dict[ArtifactId, Judgment] = {}
    for record in sorted(
        records.values(), key=lambda item: (item.judgment.created_at, item.judgment.judgment_id)
    ):
        if rubric_id is not None and record.judgment.rubric_id != rubric_id:
            continue
        by_rollout[record.judgment.rollout_id] = record.judgment
    return by_rollout


def _observations_for_active_scores(
    store: ProjectStore,
    *,
    history: HumanScoreHistory,
    rubric_id: ArtifactId,
) -> tuple[JudgeScoreObservation, ...]:
    """Map active local labels to their exact persisted W6 judgment and rollout inputs.

    This adapter does not create scores or evidence. It merely supplies W6's calibration service
    the immutable input references required to rebuild a report after a human correction.
    """
    rollouts = _load_rollout_records(store)
    judgments = _load_judgment_records(store)
    by_rollout: dict[ArtifactId, _JudgmentRecord] = {}
    for record in sorted(
        judgments.values(), key=lambda item: (item.judgment.created_at, item.judgment.judgment_id)
    ):
        if record.judgment.rubric_id == rubric_id:
            by_rollout[record.judgment.rollout_id] = record
    observations: list[JudgeScoreObservation] = []
    active_scores = sorted(
        history.for_rubric(rubric_id).active_scores(),
        key=lambda item: (item.rollout_id, item.lineage_id, item.dimension_id),
    )
    for score in active_scores:
        rollout_record = rollouts.get(score.rollout_id)
        judgment_record = by_rollout.get(score.rollout_id)
        if rollout_record is None or judgment_record is None:
            raise ReviewServerError(
                "a human score needs matching stored rollout and judgment evidence"
            )
        dimension = next(
            (
                item
                for item in judgment_record.judgment.dimensions
                if item.dimension_id == score.dimension_id
            ),
            None,
        )
        if dimension is None:
            raise ReviewServerError(
                "a human score dimension needs matching stored judgment evidence"
            )
        try:
            judgment_input = artifact_input(
                store.artifacts.read(judgment_record.artifact_id).manifest
            )
            rollout_input = artifact_input(
                store.artifacts.read(rollout_record.artifact_id).manifest
            )
        except ArtifactCorruptionError as exc:
            raise ReviewServerError("a human score source artifact is no longer readable") from exc
        observations.append(
            JudgeScoreObservation(
                judgment=judgment_input,
                source_rollout=rollout_input,
                dimension_id=score.dimension_id,
                raw_score=dimension.raw_score,
                evidence_span_ids=dimension.evidence_span_ids,
            )
        )
    if not observations:
        raise ReviewServerError("calibration refresh needs at least one active human score")
    return tuple(observations)


def _lineage_for_rollout(
    rollout: RolloutArtifact,
    task_by_id: dict[ArtifactId, TaskCase],
) -> ArtifactId | None:
    """Return the frozen W5 lineage for one rollout when its selected task is still present."""
    task = task_by_id.get(rollout.task_id)
    return None if task is None else task.lineage_group_id


def _task_for_rollout(rollout: RolloutArtifact, tasks: tuple[TaskCase, ...]) -> TaskCase:
    """Resolve one rollout's selected task or reject an untraceable review write."""
    for task in tasks:
        if task.task_id == rollout.task_id:
            return task
    raise ReviewServerError("the score target rollout does not belong to the selected task set")


def _required_id(value: ArtifactId | None, field_name: str) -> ArtifactId:
    """Require one named artifact identity from a rubric mutation."""
    if value is None:
        raise ReviewServerError(f"{field_name} is required")
    return value


def _required_ids(value: tuple[ArtifactId, ...] | None, field_name: str) -> tuple[ArtifactId, ...]:
    """Require one non-null ordered collection from a rubric mutation."""
    if value is None:
        raise ReviewServerError(f"{field_name} is required")
    return value


def _required_dimension(value: RubricDimension | None) -> RubricDimension:
    """Require one complete human-authored rubric dimension from a mutation."""
    if value is None:
        raise ReviewServerError("dimension is required")
    return value


def _required_dimensions(value: tuple[RubricDimension, ...] | None) -> tuple[RubricDimension, ...]:
    """Require one complete replacement scale set from a mutation."""
    if value is None:
        raise ReviewServerError("dimensions is required")
    return value


def _utc_now() -> datetime:
    """Return the local adapter's timezone-aware clock value."""
    return datetime.now(UTC)


def _loopback_host(value: str) -> str:
    """Validate that the local review adapter cannot bind a non-loopback interface."""
    if value not in _LOOPBACK_HOSTS:
        raise argparse.ArgumentTypeError("local review may bind only to a loopback host")
    return value


def _current_revision() -> str:
    """Return the repository revision that will produce new local review artifacts."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "local-unversioned"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the local review adapter on loopback for the top-level TypeScript app.

    Args:
        argv: Optional explicit arguments for tests or an embedded local launcher.
    """
    parser = argparse.ArgumentParser(description="Serve the WMO local review adapter on loopback.")
    parser.add_argument("--root", type=Path, default=Path(".wmo"))
    parser.add_argument("--project", default="default")
    parser.add_argument("--task-set", default=None)
    parser.add_argument("--host", type=_loopback_host, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        ProjectStore(arguments.root, arguments.project).load_project()
    except ProjectStoreError as exc:
        parser.error(str(exc))
    uvicorn.run(
        create_review_app(
            arguments.root,
            arguments.project,
            code_revision=_current_revision(),
            task_set_id=arguments.task_set,
            port=arguments.port,
        ),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
