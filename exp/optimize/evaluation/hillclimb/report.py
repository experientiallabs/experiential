"""The climb's final line: baseline against the best prompt on fresh test repeats.

The best prompt is the last kept candidate after the reported rounds. A final panel compares it
with the baseline on the test lineages with ``final_repeats`` fresh continuations per arm, and the
same keep rule decides the verdict. Those lineages already confirmed every kept edit, so the best
prompt was selected on them: the verdict is a development reading, not an independent holdout
estimate, and the report says so in its scope. The final comparison is registered once per
baseline and best pair, with its deadline and code revision: a later report whose best prompt is
unchanged (every later round reverted) replays that comparison and its unknowns instead of
drawing a new one, so an unchanged prompt cannot be final-tested again until one run says merge.
"Within noise, do not merge" is a complete, valid result, as is "no edit kept" when every round
reverted. The report lists every round, kept and reverted.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.core.locks import CoordinationLock
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, climb_split
from exp.optimize.evaluation.hillclimb.history import (
    incumbent_version,
    load_history,
    round_identity,
)
from exp.optimize.evaluation.hillclimb.measurement import StageMeasurement
from exp.optimize.evaluation.hillclimb.prompts import (
    PromptVersion,
    baseline_version,
    load_prompt_version,
    persist_prompt_version,
)
from exp.optimize.evaluation.hillclimb.stages import (
    remeasure,
    run_stage,
    stage_attempts,
    stage_jobs,
    validate_stage,
)

REPORT_TYPE = "hill-climb-report"
FINAL_TYPE = "hill-climb-final-registration"
FINAL_SCOPE = (
    "fresh repeats on the test lineages that already confirmed every kept edit, so the best "
    "prompt was selected on these lineages and merge is optimistic; not an independent holdout "
    "estimate: audit the best prompt on lineages the climb never touched before claiming a gain"
)
Verdict = Literal["merge", "within noise, do not merge", "no edit kept"]


class ClimbReportSpec(ContractModel):
    """The ``climb-report`` payload.

    Attributes:
        climb: Climb design.
        rounds: Decided rounds the report covers, from the first; zero covers none.
        deadline: Deadline of every final-comparison continuation.
    """

    climb: ClimbSpec
    rounds: int = Field(ge=0, le=200, strict=True)
    deadline: AwareDatetime

    def identity(self) -> str:
        """Report identity: one final comparison per climb and round count."""
        return stable_id(REPORT_TYPE, {"climb": self.climb.identity(), "rounds": self.rounds})


class FinalRegistration(ContractModel):
    """The one final comparison of a baseline and best pair, fixed before its first dispatch.

    Attributes:
        climb: Climb identity.
        baseline: Baseline prompt version.
        best: Best prompt version compared against it.
        deadline: Deadline of every final continuation.
        code_revision: Execution revision the final panel runs under.
    """

    climb: str
    baseline: ArtifactInput
    best: ArtifactInput
    deadline: AwareDatetime
    code_revision: str = Field(min_length=1)


def final_identity(climb: str, baseline: PromptVersion, best: PromptVersion) -> str:
    """Identity of a pair's final comparison, independent of the reported round count."""
    return stable_id(
        FINAL_TYPE, {"climb": climb, "baseline": baseline.identity, "best": best.identity}
    )


def _final_registration(
    store: ArtifactStore,
    identity: str,
    registration: FinalRegistration,
    *,
    created_at: datetime,
) -> FinalRegistration:
    """Save a pair's final registration once and return the saved one.

    Raises:
        ValueError: The pair's final comparison was registered under another code revision.
    """
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=FINAL_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=registration.code_revision,
                inputs=tuple(
                    sorted(
                        (registration.baseline, registration.best),
                        key=lambda ref: ref.artifact_id,
                    )
                ),
            ),
            files={"registration.json": registration},
        )
    if store.read(identity).manifest.artifact_type != FINAL_TYPE:
        raise ValueError("saved final registration has the wrong artifact type")
    saved = FinalRegistration.model_validate_json(store.read_bytes(identity, "registration.json"))
    if saved.code_revision != registration.code_revision:
        raise ValueError(
            f"the final comparison of this best prompt was registered under code revision "
            f"{saved.code_revision}; report under that revision so it replays, never redraws"
        )
    return saved


class RoundSummary(ContractModel):
    """One reported round, for the person reading the report.

    Attributes:
        round: One-based round number.
        kind: Edit kind.
        change_note: Why the edit was proposed.
        proposed_by: ``operator`` or ``analyst``.
        kept: Whether it was kept.
        reason: Why it was kept or reverted.
        candidate: Saved prompt version, retained either way.
    """

    round: int
    kind: str
    change_note: str
    proposed_by: Literal["operator", "analyst"]
    kept: bool
    reason: str
    candidate: ArtifactInput


class ClimbReport(ContractModel):
    """The saved final line of a climb.

    Attributes:
        climb: Climb identity.
        rounds: Every reported round in order.
        baseline: Baseline prompt version.
        best: Best prompt version: the last kept candidate, or the baseline.
        best_system_prompt: Its complete prompt.
        final: Fresh test comparison of baseline and best, absent when no edit was kept.
        verdict: ``merge`` only when the final comparison passes the keep rule.
        scope: Interpretation limits of the final comparison.
    """

    climb: str
    rounds: tuple[RoundSummary, ...]
    baseline: ArtifactInput
    best: ArtifactInput
    best_system_prompt: str | None
    final: StageMeasurement | None
    verdict: Verdict
    scope: str = FINAL_SCOPE


def report_jobs(project: ProjectStore, spec: ClimbReportSpec) -> tuple[ContinuationJob, ...]:
    """Every continuation job the final comparison could dispatch, both arms, for cost review."""
    split = climb_split(project, spec.climb)
    return tuple(
        job for job in stage_jobs(spec.climb, split, "final", spec.deadline) for _ in range(2)
    )


def report_attempts(project: ProjectStore, spec: ClimbReportSpec) -> int:
    """Maximum actor HTTP attempts of the final comparison."""
    return stage_attempts(spec.climb, climb_split(project, spec.climb), "final")


def preflight_climb_report(project: ProjectStore, spec: ClimbReportSpec) -> bool:
    """Check the reported rounds and the final comparison without writes or calls.

    Args:
        project: Project holding the climb.
        spec: Report payload.

    Returns:
        Whether a final comparison needs to run (an edit was kept).
    """
    history = load_history(project, spec.climb, spec.rounds)
    best = incumbent_version(project, spec.climb, history)
    baseline = baseline_version(spec.climb.identity(), spec.climb.baseline_prompt)
    if best == baseline:
        return False
    validate_stage(project, spec.climb, climb_split(project, spec.climb), "final", (baseline, best))
    return True


def saved_climb_report(project: ProjectStore, spec: ClimbReportSpec) -> ClimbReport | None:
    """Replay a saved report, refusing one whose final comparison no longer reproduces."""
    identity = spec.identity()
    if not project.artifacts.exists(identity):
        return None
    if project.artifacts.read(identity).manifest.artifact_type != REPORT_TYPE:
        raise ValueError("saved climb report has the wrong artifact type")
    report = ClimbReport.model_validate_json(project.artifacts.read_bytes(identity, "report.json"))
    if report.final is not None:
        arms = (
            load_prompt_version(project.artifacts, report.baseline),
            load_prompt_version(project.artifacts, report.best),
        )
        remeasure(project, spec.climb, climb_split(project, spec.climb), report.final, arms)
    return report


def run_climb_report(
    project: ProjectStore, spec: ClimbReportSpec, *, code_revision: str
) -> ClimbReport:
    """Compare baseline and best on fresh test repeats and save the climb's final line.

    Args:
        project: Project holding the climb.
        spec: Report payload.
        code_revision: Installed execution revision of the final panel.

    Returns:
        The saved report; a saved one replays without calls, and a best prompt already
        final-tested replays its registered comparison.

    Raises:
        ValueError: The best prompt's final comparison was registered under another revision
            and this report is not yet saved; a saved report replays under any revision.
    """
    climb = spec.climb.identity()
    store = project.artifacts
    with CoordinationLock(store.project_directory / f"{climb}.lock", timeout=0):
        saved = saved_climb_report(project, spec)
        if saved is not None:
            return saved
        history = load_history(project, spec.climb, spec.rounds)
        now = datetime.now(UTC)
        baseline_v = baseline_version(climb, spec.climb.baseline_prompt)
        best_v = incumbent_version(project, spec.climb, history)
        baseline = persist_prompt_version(
            store, baseline_v, created_at=now, code_revision=code_revision
        )
        best = persist_prompt_version(store, best_v, created_at=now, code_revision=code_revision)
        final = None
        verdict: Verdict = "no edit kept"
        registered: set[ArtifactInput] = set()
        if best_v != baseline_v:
            pair = final_identity(climb, baseline_v, best_v)
            registration = _final_registration(
                store,
                pair,
                FinalRegistration(
                    climb=climb,
                    baseline=baseline,
                    best=best,
                    deadline=spec.deadline,
                    code_revision=code_revision,
                ),
                created_at=now,
            )
            registered.add(artifact_input(store.read(pair).manifest))
            final = run_stage(
                project,
                spec.climb,
                climb_split(project, spec.climb),
                "final",
                f"final {pair}",
                (baseline_v, best_v),
                registration.deadline,
                code_revision=code_revision,
            )
            verdict = "merge" if final.passed else "within noise, do not merge"
        summaries = []
        for record in history:
            candidate = load_prompt_version(store, record.candidate)
            assert candidate.edit is not None
            summaries.append(
                RoundSummary(
                    round=record.round,
                    kind=candidate.edit.kind,
                    change_note=candidate.edit.change_note,
                    proposed_by="operator" if candidate.proposal is None else "analyst",
                    kept=record.kept,
                    reason=record.reason,
                    candidate=record.candidate,
                )
            )
        report = ClimbReport(
            climb=climb,
            rounds=tuple(summaries),
            baseline=baseline,
            best=best,
            best_system_prompt=best_v.system_prompt,
            final=final,
            verdict=verdict,
        )
        inputs = {
            baseline,
            best,
            *(artifact_input(store.read(round_identity(climb, r.round)).manifest) for r in history),
            *(() if final is None else (final.panel,)),
            *registered,
        }
        store.write_json(
            artifact_id=spec.identity(),
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report
