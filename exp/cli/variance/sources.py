"""Read-only data sources behind the front-door live view.

Every source here is durable, so the same view works in the process running the front door and
in a later ``exp variance watch`` from another terminal:

- the run's transition journal (``runtime/front-door/front-door-<spec16>/transitions.jsonl``),
  written by the front door's checked state machines: each phase's state and when it moved;
- the operation's spend ledger (``runtime/spend/<operation>/ledger.sqlite``): one reservation per
  physical provider attempt, by role, with its state, joined to the gateway's
  ``provider-physical-attempt`` intent and response records in ``gateway/traffic.db`` for the
  HTTP status, lateness, purpose and latency of each finished attempt;
- the host-wide provider admission database: calls in flight per alias and open breakers;
- the project store: each start's ``decision-variance-run`` registration and its saved
  ``<identity>-stage`` records, read through :class:`~exp.common.project.ProjectStore`.

Nothing here writes: SQLite files open read-only (``mode=ro``, ``query_only``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exp.common.project import ProjectStore

PIPELINE_STAGES = ("variance-build", "rules", "traceml", "error-mining", "families", "records")
"""A start's pipeline stages, in order."""

OPERATION_PREFIX = "front-door-cluster-"
"""Spend operation of one consented front-door plan: the prefix and the digest's first 16."""

ALLOWANCE_STAGE = "operation-unpriced-allowance"
"""Ledger authorization row holding the real-attempt cap."""

_ACTIVE = ("reserved", "dispatched")


def parse_time(text: str | None) -> datetime | None:
    """Parse an ISO timestamp, or return None."""
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def read_only(path: Path) -> sqlite3.Connection:
    """Open one SQLite file strictly read-only."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    connection.execute("PRAGMA query_only=1")
    return connection


# ---------------------------------------------------------------- transition journal


@dataclass
class PhaseRow:
    """One phase as the journal last recorded it.

    Attributes:
        name: The phase (``slot tag-banks``, ``pipeline <task>``...).
        state: Its current state (``planned``, ``running``, ``done``, ``failed``...).
        started: When it moved to ``running`` in this run, if it did.
        ended: When it left ``running``, if it did.
        replayed: Whether it was already saved when the run started.
    """

    name: str
    state: str
    started: datetime | None = None
    ended: datetime | None = None
    replayed: bool = False


@dataclass
class RunSegment:
    """One front-door run inside a registration's journal (from its ``planned`` event on).

    Attributes:
        subject: The run's journal subject (``front-door-<spec16>``).
        state: The run's state (``planned``, ``running``, ``finished``, ``stopped``, ``failed``).
        planned: When the run's plan was recorded.
        ended: When it reached a final state, if it did.
        phases: Every phase in first-seen order.
    """

    subject: str
    state: str
    planned: datetime | None
    ended: datetime | None = None
    phases: dict[str, PhaseRow] = field(default_factory=dict)


_FINAL_RUN = {"finished", "stopped", "failed", "replaying"}


def journal_segments(path: Path) -> list[RunSegment]:
    """Split a registration's transition journal into its runs, oldest first."""
    if not path.exists():
        return []
    segments: list[RunSegment] = []
    current: RunSegment | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue  # a line being written
        machine, subject, to = row.get("machine"), row.get("subject", ""), row.get("to", "")
        moment = parse_time(row.get("at"))
        if machine == "front-door-run":
            if row.get("from") is None or (to == "planned" and row.get("event") == "planned"):
                current = RunSegment(subject=subject, state=to, planned=moment)
                segments.append(current)
                continue
            if current is None:
                continue
            current.state = to
            if to in _FINAL_RUN:
                current.ended = moment
        elif machine == "front-door-phase" and current is not None:
            phase = current.phases.get(subject)
            if phase is None:
                phase = PhaseRow(
                    name=subject, state=to, replayed=row.get("from") is None and to == "done"
                )
                current.phases[subject] = phase
            phase.state = to
            if to == "running":
                phase.started, phase.ended = moment, None
            elif row.get("from") == "running":
                phase.ended = moment
            elif to == "done" and phase.started is None:
                phase.replayed = True
    return segments


def segment_after(segments: list[RunSegment], moment: datetime | None) -> RunSegment | None:
    """The first run planned at or after ``moment``, or the last run when ``moment`` is None."""
    if not segments:
        return None
    if moment is None:
        return segments[-1]
    later = [row for row in segments if row.planned is not None and row.planned >= moment]
    return later[0] if later else None


# ---------------------------------------------------------------- spend ledger and traffic


@dataclass
class Call:
    """One physical attempt of the run: its reservation joined to its gateway record."""

    seq: int
    role: str
    state: str
    reserved: float
    attempt: str | None
    pid: int | None = None
    status: int | None = None
    late: bool = False
    done: float | None = None
    purpose: str | None = None


@dataclass
class RoleRow:
    """One role's calls, for the calls panel."""

    role: str
    total: int = 0
    done: int = 0
    in_flight: int = 0
    errors: int = 0
    rate_limited: int = 0
    timeouts: int = 0
    last_minute: int = 0
    p50: float | None = None
    purposes: Counter[str] = field(default_factory=Counter)


class Traffic:
    """Incremental reader of one operation's spend ledger and its attempts' gateway records."""

    def __init__(self, ledger: Path, traffic_db: Path | None, project_id: str) -> None:
        """Bind the ledger and gateway database paths; nothing is read yet."""
        self.ledger = ledger
        self.traffic_db = traffic_db
        self.project_id = project_id
        self.calls: dict[int, Call] = {}
        self.allowance: int | None = None
        self.authorized: datetime | None = None
        self.error = ""

    def refresh(self) -> None:
        """Re-read the ledger, then resolve attempts whose outcome is not known yet."""
        if not self.ledger.exists():
            self.error = f"no spend ledger yet ({self.ledger.parent.name})"
            return
        try:
            with read_only(self.ledger) as ledger:
                for stage, attempts, moment in ledger.execute(
                    "SELECT stage, unpriced_attempts, authorized_at FROM authorizations "
                    "ORDER BY seq"
                ):
                    if self.authorized is None:
                        self.authorized = parse_time(moment)
                    if stage == ALLOWANCE_STAGE:
                        self.allowance = int(attempts)
                for seq, role, state, reserved, attempt, pid in ledger.execute(
                    "SELECT seq, role, state, reserved_at, attempt_id, pid FROM reservations"
                ):
                    call = self.calls.get(seq)
                    if call is None:
                        moment = parse_time(reserved)
                        stamp = moment.timestamp() if moment else 0.0
                        self.calls[seq] = Call(seq, role, state, stamp, attempt, pid)
                    else:
                        call.state, call.attempt = state, attempt or call.attempt
            self.error = ""
        except sqlite3.Error as error:
            self.error = f"ledger: {error}"
            return
        self._resolve()

    def _resolve(self) -> None:
        """Read the HTTP status, lateness, finish time and purpose of finished attempts."""
        pending = [
            call
            for call in self.calls.values()
            if call.attempt and call.status is None and call.state in ("settled", "unknown")
        ]
        if not pending or self.traffic_db is None or not self.traffic_db.exists():
            return
        try:
            with read_only(self.traffic_db) as db:
                for start in range(0, len(pending), 400):
                    self._resolve_chunk(db, pending[start : start + 400])
        except sqlite3.Error as error:
            self.error = f"traffic.db: {error}"

    def _resolve_chunk(self, db: sqlite3.Connection, chunk: list[Call]) -> None:
        """Resolve one chunk of attempts from their response and intent records."""
        by_response = {f"provider-attempt-{call.attempt}-response": call for call in chunk}
        marks = ",".join("?" * len(by_response))
        for artifact_id, head, tail, made in db.execute(
            "SELECT f.artifact_id, substr(CAST(f.payload AS TEXT), 1, 40), "
            "substr(CAST(f.payload AS TEXT), -80), "
            "json_extract(CAST(a.manifest AS TEXT), '$.created_at') "
            "FROM project_artifact_files f JOIN project_artifacts a "
            "ON a.project_id = f.project_id AND a.artifact_id = f.artifact_id "
            f"WHERE f.project_id = ? AND f.artifact_id IN ({marks})",
            [self.project_id, *by_response],
        ):
            call = by_response[artifact_id]
            found = re.search(r'"http_status":(\d+)', head or "")
            call.status = int(found.group(1)) if found else -1
            call.late = '"returned_after_deadline":true' in (tail or "")
            moment = parse_time(made)
            call.done = moment.timestamp() if moment else None
        by_intent = {
            f"provider-attempt-{call.attempt}-intent": call
            for call in chunk
            if call.purpose is None
        }
        if not by_intent:
            return
        marks = ",".join("?" * len(by_intent))
        for artifact_id, head in db.execute(
            "SELECT artifact_id, substr(CAST(payload AS TEXT), 1, 200) FROM project_artifact_files "
            f"WHERE project_id = ? AND artifact_id IN ({marks})",
            [self.project_id, *by_intent],
        ):
            found = re.search(r'"canonical_request":\{"artifact_id":"([^"]+)"', head or "")
            purpose = re.sub(r"-[0-9a-f]{12,}.*", "", found.group(1)) if found else "direct"
            by_intent[artifact_id].purpose = purpose

    def used(self) -> int:
        """Attempts counted against the real-attempt cap (every reservation not released)."""
        return sum(1 for call in self.calls.values() if call.state != "released")

    def pids(self) -> set[int]:
        """Processes that reserved attempts in this operation."""
        return {call.pid for call in self.calls.values() if call.pid}

    def roles(self, now: float, since: float | None = None) -> list[RoleRow]:
        """Per-role rows, busiest first, optionally only calls reserved at or after ``since``."""
        rows: dict[str, RoleRow] = {}
        latencies: dict[str, list[float]] = {}
        for call in self.calls.values():
            if since is not None and call.reserved < since:
                continue
            row = rows.setdefault(call.role, RoleRow(call.role))
            row.total += 1
            row.last_minute += now - call.reserved <= 60
            if call.state in _ACTIVE:
                row.in_flight += 1
            elif call.state == "settled":
                row.done += 1
            if call.state == "unknown" or call.late:
                row.timeouts += 1
            if call.status is not None and call.status >= 400:
                row.errors += 1
            if call.status == 429:
                row.rate_limited += 1
            if call.done is not None and call.status == 200:
                latencies.setdefault(call.role, []).append(max(0.0, call.done - call.reserved))
            if call.purpose:
                row.purposes[call.purpose] += 1
        for role, values in latencies.items():
            values.sort()
            rows[role].p50 = values[len(values) // 2]
        return sorted(rows.values(), key=lambda row: -row.total)


# ---------------------------------------------------------------- host admission


@dataclass(frozen=True)
class Admission:
    """Host-wide provider admission: calls in flight by alias and open breakers."""

    in_flight: Mapping[str, int] = field(default_factory=dict)
    open_breakers: tuple[str, ...] = ()


def read_admission(path: Path | None) -> Admission:
    """Read the host-wide admission database, or nothing when it is absent or locked."""
    if path is None or not path.exists():
        return Admission()
    try:
        with read_only(path) as db:
            flight: Counter[str] = Counter()
            for alias, provider, count in db.execute(
                "SELECT alias, provider, count(*) FROM slots GROUP BY alias, provider"
            ):
                flight[alias or provider] += count
            breakers = tuple(
                f"{provider} ({state})"
                for provider, state in db.execute("SELECT provider, state FROM breakers")
                if state != "closed" and "fixture" not in str(provider)
            )
    except sqlite3.Error:
        return Admission()
    return Admission(dict(flight), breakers)


def admission_path() -> Path | None:
    """The host-wide provider admission database, or None when its owner cannot say."""
    try:
        from exp.common.auth.paths import provider_data_dir
        from exp.runtime.models.providers.backpressure import ADMISSION_DATABASE_NAME

        return provider_data_dir() / ADMISSION_DATABASE_NAME
    except Exception:  # noqa: BLE001 - the panel is optional
        return None


# ---------------------------------------------------------------- project store


@dataclass(frozen=True)
class StartStages:
    """One start's newest pipeline registration and which of its stages are saved.

    Attributes:
        task_id: The start's task.
        run: The ``decision-variance-run`` registration, if any.
        saved: Each stage's saved ``-stage`` record artifact, by stage.
        saved_at: When each saved stage record was written, by stage.
        identities: Each stage's identity, by stage.
    """

    task_id: str
    run: str | None = None
    saved: Mapping[str, str] = field(default_factory=dict)
    saved_at: Mapping[str, datetime] = field(default_factory=dict)
    identities: Mapping[str, str] = field(default_factory=dict)


def pipeline_stages(
    project: ProjectStore, tasks: Iterable[str], since: datetime | None = None
) -> dict[str, StartStages]:
    """Each start's newest pipeline registration (at or after ``since``) and its saved stages."""
    wanted = set(tasks)
    if not wanted:
        return {}
    store = project.artifacts
    newest: dict[str, tuple[datetime, str, dict[str, str]]] = {}
    for artifact_id in store.list_ids(artifact_type="decision-variance-run"):
        stored = store.read(artifact_id)
        moment = stored.manifest.created_at
        if since is not None and moment < since:
            continue
        spec = json.loads(stored.payloads["spec.json"])
        task = ((spec.get("mining") or {}).get("bootstrap") or {}).get("task_id")
        if task in wanted and (task not in newest or newest[task][0] < moment):
            newest[task] = (moment, artifact_id, json.loads(stored.payloads["stages.json"]))
    result = {}
    for task in wanted:
        if task not in newest:
            result[task] = StartStages(task)
            continue
        _, run, identities = newest[task]
        saved = {
            stage: identity + "-stage"
            for stage, identity in identities.items()
            if store.exists(identity + "-stage")
        }
        saved_at = {
            stage: store.read(record).manifest.created_at for stage, record in saved.items()
        }
        result[task] = StartStages(task, run, saved, saved_at, identities)
    return result


@dataclass
class Outcome:
    """What the finished pipelines produced, for the closing summary.

    Attributes:
        qualified_points: Point-bar states across every tier (``qualified``, ``rejected``...).
        error_points: Error points TraceML mined, by tier.
        judged: Tiers whose group the epoch rubric judged, by start.
        families: Family outcomes (admitted, refused) across starts.
        admitted: Admitted tasks across starts' records stages.
        harbor: Harbor package paths, qualified first.
        diagnostic: Harbor packages exported only as diagnostics.
    """

    qualified_points: Counter[str] = field(default_factory=Counter)
    error_points: Counter[str] = field(default_factory=Counter)
    judged: dict[str, list[str]] = field(default_factory=dict)
    families: Counter[str] = field(default_factory=Counter)
    admitted: int = 0
    harbor: list[str] = field(default_factory=list)
    diagnostic: list[str] = field(default_factory=list)


def _payload(project: ProjectStore, artifact_id: str, path: str | None = None) -> object:
    """Parse one JSON payload of an artifact (its only file when ``path`` is None)."""
    stored = project.artifacts.read(artifact_id)
    name = path or next(iter(stored.payloads))
    return json.loads(stored.payloads[name])


def _object(value: object) -> dict[str, Any]:
    """A JSON object, or an empty one for anything else."""
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def outcome(project: ProjectStore, records: Mapping[str, Mapping[str, str]]) -> Outcome:
    """Summarize saved stage records, ``{task: {stage: record artifact}}``, for the closing view."""
    result = Outcome()
    for task, stages in records.items():
        if "traceml" in stages:
            record = _object(_payload(project, stages["traceml"], "record.json"))
            summary, outputs = _object(record.get("summary")), _object(record.get("outputs"))
            for tier, attempts in _object(summary.get("error_points")).items():
                found = _object(attempts).values()
                result.error_points[tier] += sum(len(row) for row in found if isinstance(row, list))
            result.judged[task] = sorted(_object(summary.get("judged")))
            for ref in _object(outputs.get("point_bars")).values():
                try:
                    bars = _object(_payload(project, _object(ref)["artifact_id"]))
                except Exception:  # noqa: BLE001 - an unreadable bar only drops its count
                    continue
                for attempt in _object(bars.get("attempts")).values():
                    for point in _object(attempt).get("points", []):
                        result.qualified_points[str(_object(point).get("state", "?"))] += 1
        if "families" in stages:
            record = _object(_payload(project, stages["families"], "record.json"))
            for row in _object(record.get("summary")).get("families", []):
                result.families["admitted" if _object(row).get("admitted") else "refused"] += 1
        if "records" in stages:
            record = _object(_payload(project, stages["records"], "record.json"))
            summary = _object(record.get("summary"))
            result.admitted += int(summary.get("admitted") or 0)
            qualified = [str(path) for path in summary.get("qualified") or []]
            result.harbor.extend(qualified)
            result.diagnostic.extend(
                str(path) for path in summary.get("harbor") or [] if path not in qualified
            )
    return result
