"""Unknown-outcome retries of resumed fidelity runs: recorded before dispatch, shown in headlines.

A fidelity run interrupted mid-execution resumes in place with ``--retry-incomplete``. Saved
work replays; a request that left the process without a saved response has an unknown outcome
and is sent again. The lost response may have been seen (a gateway log, a terminal), so such a
retry is transparent but not provably uninfluenced. Each one is recorded here, durably and
before it is sent. A trace with one is a trace re-attempt: every report that includes it is
qualified in its headline with the held-out audit's label (``qualified: trace re-attempts``),
names the re-dispatches by stage and the affected traces, keeps every trace in the primary
score, and adds the score over unaffected traces only and over affected traces only, each with
its denominator. Nothing is dropped by default. Held-out audits record the same retries as
children of their receipt (:mod:`exp.optimize.evaluation.audit.receipts`).
"""

import json
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ArtifactEnvelope, ContractModel
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.simulation.world_model.resumption import RecoveryStage

QUALIFICATION = "qualified: trace re-attempts"
JUDGE_OPERATION = "judge"
STAGE_ORDER: tuple[RecoveryStage, ...] = ("prediction", "judge", "investigation", "retrieval")
_TYPE = "fidelity-unknown-outcome"


class UnknownOutcome(ContractModel):
    """One step of one stage re-dispatched after an unknown outcome.

    Attributes:
        stage: Prediction, judge, investigation, or retrieval (an interrupted preparation).
        step: Measured step index from 0, or None for a run-level judge or consolidation.
            Disclosure lines count
            steps from 1, like the per-turn report.
        attempts: Dispatches of this step at this stage: the lost one plus each re-dispatch.
    """

    stage: RecoveryStage
    step: int | None = Field(ge=0)
    attempts: int = Field(ge=2)

    def line(self) -> str:
        """One per-step disclosure line, naming the stage and attempt count."""
        where = (
            ("memory consolidation" if self.stage == "investigation" else self.stage)
            if self.step is None
            else f"step {self.step + 1}"
        )
        return (
            f"{where} re-dispatched after an unknown outcome ({self.stage}, "
            f"{self.attempts} attempts)"
        )


class Qualification(ContractModel):
    """Headline qualification of a report whose traces include unknown-outcome retries.

    Attributes:
        prediction: Re-dispatched prediction steps.
        judge: Re-dispatched judges.
        investigation: Re-dispatched investigations.
        retrieval: Steps whose interrupted preparation was prepared again.
        affected: Traces with at least one retry.
        traces: Traces in the report; the primary score keeps all of them.
        exact_unaffected: Exact replies over unaffected traces only.
        steps_unaffected: Measured steps over unaffected traces only.
        exact_affected: Exact replies over affected traces only.
        steps_affected: Measured steps over affected traces only.
    """

    prediction: int = Field(ge=0)
    judge: int = Field(ge=0)
    investigation: int = Field(ge=0)
    retrieval: int = Field(ge=0)
    affected: int = Field(ge=1)
    traces: int = Field(ge=1)
    exact_unaffected: int = Field(ge=0)
    steps_unaffected: int = Field(ge=0)
    exact_affected: int = Field(ge=0)
    steps_affected: int = Field(ge=0)

    def lines(self) -> tuple[str, ...]:
        """The re-dispatches by stage, then both sensitivity figures with their denominators."""
        unaffected = self.traces - self.affected
        return (
            stage_line(
                {
                    "prediction": self.prediction,
                    "judge": self.judge,
                    "investigation": self.investigation,
                    "retrieval": self.retrieval,
                },
                self.affected,
                self.traces,
            ),
            f"  Unaffected traces only  {self.exact_unaffected}/{self.steps_unaffected} replies "
            f"exact ({_share(self.exact_unaffected, self.steps_unaffected)}) · "
            f"{unaffected}/{unaffected} traces",
            f"  Re-attempted traces only  {self.exact_affected}/{self.steps_affected} replies "
            f"exact ({_share(self.exact_affected, self.steps_affected)}) · "
            f"{self.affected}/{self.affected} traces",
        )


def stage_counts(rows: Iterable[UnknownOutcome]) -> dict[RecoveryStage, int]:
    """Re-dispatched steps per stage, every stage present."""
    counts: dict[RecoveryStage, int] = {stage: 0 for stage in STAGE_ORDER}
    for row in rows:
        counts[row.stage] += 1
    return counts


def stage_line(counts: Mapping[RecoveryStage, int], affected: int, traces: int) -> str:
    """One line naming the re-dispatches by stage and the affected traces.

    Args:
        counts: Re-dispatched steps per stage.
        affected: Traces with at least one re-dispatch.
        traces: Traces in the report.

    Returns:
        The line; retrieval appears only when an interrupted preparation was repeated.
    """
    stages = " · ".join(
        f"{stage} {counts.get(stage, 0)}"
        for stage in STAGE_ORDER
        if stage != "retrieval" or counts.get(stage, 0)
    )
    return (
        f"  re-dispatched after an unknown outcome  {stages} · {affected} of {traces} traces "
        "affected (a lost response may have been seen)"
    )


def qualify(
    traces: list[tuple[tuple[UnknownOutcome, ...], int, int]],
) -> Qualification | None:
    """Qualify a report from each trace's retries, exact replies and measured steps.

    Args:
        traces: ``(retries, exact, steps)`` for every trace in the report.

    Returns:
        The qualification, or None when no trace has an unknown-outcome retry.
    """
    affected = [row for row in traces if row[0]]
    if not affected:
        return None
    clean = [row for row in traces if not row[0]]
    counts = stage_counts(retry for retries, _, _ in affected for retry in retries)
    return Qualification(
        prediction=counts["prediction"],
        judge=counts["judge"],
        investigation=counts["investigation"],
        retrieval=counts["retrieval"],
        affected=len(affected),
        traces=len(traces),
        exact_unaffected=sum(exact for _, exact, _ in clean),
        steps_unaffected=sum(steps for *_, steps in clean),
        exact_affected=sum(exact for _, exact, _ in affected),
        steps_affected=sum(steps for *_, steps in affected),
    )


class RecoveryLog:
    """Durable, idempotent record of one run's unknown-outcome re-dispatches.

    Each re-dispatch is one immutable artifact, numbered within the run and written before
    the request is sent. Recording the same re-dispatch again (a second resumption) is a
    no-op, so attempt counts stay exact across repeated interruptions.
    """

    def __init__(
        self,
        project: ProjectStore,
        run: str,
        *,
        code_revision: str,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind the run whose re-dispatches are recorded.

        Args:
            project: Project owning the run.
            run: The run's registration identity.
            code_revision: Producing implementation identity.
            clock: Aware clock for the records' provenance.
        """
        self._project, self._run = project, run
        self._revision, self._clock = code_revision, clock
        self._lock = threading.Lock()

    def record(self, stage: RecoveryStage, operation: str, logical: str) -> None:
        """Record one re-dispatch before it is sent.

        Args:
            stage: Where the outcome was unknown.
            operation: Workload call ID of the step, or the judge operation.
            logical: The re-dispatch's logical request identity.
        """
        artifacts = self._project.artifacts
        with self._lock:
            number = 1
            while artifacts.exists(identity := _record_id(self._run, number)):
                if _read(self._project, identity)["logical"] == logical:
                    return
                number += 1
            artifacts.write_json(
                artifact_id=identity,
                artifact_type=_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=self._clock(), code_revision=self._revision
                ),
                files={
                    "record.json": {
                        "run": self._run,
                        "stage": stage,
                        "operation": operation,
                        "logical": logical,
                    }
                },
            )


def recorded_outcomes(
    project: ProjectStore, run: str, actions: tuple[ToolCall, ...]
) -> tuple[UnknownOutcome, ...]:
    """Every step of a run re-dispatched after an unknown outcome, without provider calls.

    Args:
        project: Project owning the run.
        run: The run's registration identity.
        actions: The run's workload actions, which number its steps.

    Returns:
        One row per step and stage, ordered by step (the judge last), then stage.
    """
    counts: dict[tuple[str, str], int] = {}
    number = 1
    while project.artifacts.exists(identity := _record_id(run, number)):
        record = _read(project, identity)
        key = (record["stage"], record["operation"])
        counts[key] = counts.get(key, 0) + 1
        number += 1
    steps = {action.call_id: index for index, action in enumerate(actions)}
    rows = [
        UnknownOutcome.model_validate(
            {
                "stage": stage,
                "step": None
                if operation == JUDGE_OPERATION or operation.startswith("dream:")
                else steps[operation],
                "attempts": count + 1,
            }
        )
        for (stage, operation), count in counts.items()
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.step is None, row.step or 0, STAGE_ORDER.index(row.stage)),
        )
    )


def _record_id(run: str, number: int) -> str:
    """Identity of a run's ``number``-th recorded re-dispatch."""
    return f"{run}-unknown-outcome-{number}"


def _read(project: ProjectStore, identity: str) -> dict[str, str]:
    """One saved re-dispatch record."""
    record = json.loads(project.artifacts.read_bytes(identity, "record.json"))
    if not isinstance(record, dict):
        raise ValueError(f"unknown-outcome record {identity} is not an object")
    return {str(key): str(value) for key, value in record.items()}


def _share(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"
