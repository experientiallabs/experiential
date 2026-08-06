"""Live run emission: what a running command tells the platform, and what it never risks.

The one rule this module exists to hold: EMISSION MUST NEVER BREAK A RUN. A tau grid arm is
hundreds of dollars of measured cells and a `wmo optimize model` sweep is the entire spend of that
command, so telemetry is allowed to be missing and is never allowed to raise. Concretely:

- Emission is off unless a platform credential AND an organization resolve
  (`wmo.runtime.runs.client`),
  and which way it went is one INFO line at start rather than a surprise later.
- Every call into the emitter is wrapped: the first failure of a run logs one WARNING, the rest go
  to debug, and nothing propagates to the caller. That covers a permanent refusal, an unreachable
  platform, and an exhausted seq band alike.
- The queue is bounded, and every hook flushes at the seam it fires on. A push that fails puts its
  events back and the oldest are dropped once the bound is reached, so an hours-long outage costs
  telemetry (recoverable later with `wmo runs backfill`) and never memory.
- Nothing blocks on a deadline of its own. There is no watchdog and no background thread: every
  push happens inside a callback the run was making anyway, so turning emission on can never make
  a long-running sweep block or hang where it would otherwise have made progress.

Seq bands come from `wmo.runtime.runs.schema` and are the same ones `wmo runs backfill` uses, so
what a run reports live and what a later backfill replays converge on one identity instead of
double-counting.
Three kinds of seq, and which one a fact gets depends on whether the artifacts can name its
position:

- DERIVED, from the artifact itself: `run.meta` at band 0 seq 1, and each `ledger.line` at
  `ledger_walk_seq(position)` for its position in the arm's ledger. A line's index is fixed the
  moment it is appended, so this process, its siblings, and a backfill months later all compute the
  same seq and the platform keeps ONE copy. Being told a derived seq is already held is the design
  working, not a collision.
- ALLOCATED ascending, per chunk: a completed chunk k's cells in band k+1, in the order they were
  measured.
- ALLOCATED descending, from the ceiling of the process's first-owned chunk band: heartbeats and a
  live `run.status`, which have no artifact position and so can never be re-derived. Descending
  keeps them clear of the ascending walk in the same band; `SeqBands` raises if the two meet.

A run is also RESUMED before its first event: the emitter probes what the platform already holds and
continues past it. Without that, a second invocation (a resumed pipeline, a restarted arm) renumbers
from the floor, every event is discarded as a replay of a seq already held, and the run looks
healthy while its new telemetry silently vanishes.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Lock

from pydantic import BaseModel, ConfigDict

from wmo.core.types import JsonObject
from wmo.optimize.routing.outcomes import ScenarioOutcome
from wmo.optimize.routing.pipeline import Stage, StageRecord
from wmo.optimize.telemetry.backfill import cell_payload
from wmo.runtime.runs.client import ControlCommand, PushAck, PushRejected, RunsSink, runs_sink
from wmo.runtime.runs.reader import EventRow, RunsReader
from wmo.runtime.runs.schema import (
    CELL_BATCH_CAP,
    LEDGER_LINE,
    RUN_LEVEL_BAND,
    RUN_META_SEQ,
    RUN_SEQ_BAND,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    SeqBands,
    cell_band,
    grid_arm_external_id,
    ledger_walk_seq,
    pipeline_external_id,
)

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 15.0
"""Least time between heartbeats, and the age at which buffered cells are sent. Driven by callbacks
the run already makes, never by a timer thread: a grid buys a cell every few tens of seconds, which
is a finer cadence than any panel needs."""

QUEUE_LIMIT = 2000
"""Events held in memory when the platform is unreachable. Past this the oldest are dropped: a run
that cannot reach the platform for hours must not convert telemetry into memory pressure."""

FRONTIER_PAGE = 500
"""Events per page of the frontier read, matching the route's own default."""

FRONTIER_MAX_PAGES = 40
"""Pages the frontier scan will walk (20k type-filtered events) before giving up. A scan that has to
STOP EARLY reports no frontier at all rather than the lowest seq it happened to reach: a partial
scan's minimum is higher than the true frontier, so using it would re-issue seqs, which is the exact
failure this read exists to prevent."""

FrontierReader = Callable[[str, int], int | None]
"""Reports a run's DESCENDING frontier in one band, or None when it has none yet (or cannot be
read). Injected, because locating it needs a read the emitter otherwise never makes."""

SinkFactory = Callable[[], "RunsSink | None"]
"""Opens the transport, or returns None when this machine cannot push. Injected so a test can drive
the real sink over a fake transport, and so `--no-emit` needs no special case."""

CONTROL_STOP = "stop"
CONTROL_RETRY_UNSCORED = "retry_unscored"
CONTROL_FORCE_FROM_STAGE = "force_from_stage"

ACK_ACKED = "acked"
ACK_DONE = "done"
ACK_REJECTED = "rejected"

# Why the grid runner declines the two commands it cannot honor. Both name real capabilities; they
# are just already owned by the run's own resume semantics rather than by a remote nudge.
REJECT_RETRY_UNSCORED = (
    "the grid runner owns its own retry pass: every transiently-failed cell is retried once, the "
    "spent retries are recorded per chunk so a restart cannot buy a second one, and a chunk that "
    "failed systemically is deliberately left alone. Re-run the arm to resume it."
)
REJECT_FORCE_FROM_STAGE = (
    "a grid arm has chunks, not stages, and its resume unit is the chunk file on disk. Delete the "
    "chunk files that should be re-bought, then re-run the arm."
)


def _status_value(status: RunStatus | str) -> str | None:
    """A status as JSON carries it, or None when this build cannot name it.

    `RunStatus(status)` would be the tidy normalization, and it raises on a value outside the enum,
    which is exactly what these hooks may not do: a caller (or a future status this build has not
    heard of) must not be able to end a paid run through the telemetry path.

    Sending it as text is equally wrong, and less obviously so: the platform's status vocabulary is
    CLOSED, so an unrecognized value is refused as a permanent 4xx, and a permanent refusal drops
    the whole batch - including the ledger lines and cells riding with it. One unknown status would
    cost a chunk's telemetry rather than its own event. So the event is skipped instead: the run
    stays whatever the platform last heard, which is honest, and the warning names what was
    dropped.
    """
    if isinstance(status, RunStatus):
        return status.value
    try:
        return RunStatus(status).value
    except ValueError:
        log.warning(
            "run telemetry: %r is not a status this build knows; not reporting it "
            "(the platform refuses unknown statuses, and the refusal would drop the batch)",
            status,
        )
        return None


def frontier_from_reader(reader: RunsReader, external_id: str, band: int) -> int | None:
    """The LOWEST descending seq one band holds, which is that walk's frontier.

    Two properties make this cheap and exact. Only heartbeats and a live `run.status` ever descend,
    and the server filters by type in the database, so the scan sees a few thousand rows on a run
    with a hundred thousand cells. And within a band the descending seqs strictly DECREASE over
    time, so the most recently arrived one is the lowest: a newest-first window that contains any of
    this band's events already contains its frontier.

    That is why the tail read is a fast path rather than a heuristic, and why it is not enough on
    its own. Sibling chunk processes share one run's log under their own bands, so a busy sibling's
    heartbeats can fill the newest page and leave none of this band's in it. Finding nothing there
    means "look further", NOT "there is no frontier": treating it as the latter leaves the ceiling
    spent and re-issues a dead process's terminal status.

    Returns None only when the run genuinely holds no descending event in this band, or when the
    scan could not be completed (see `FRONTIER_MAX_PAGES`).
    """
    floor = band * RUN_SEQ_BAND
    ceiling = (band + 1) * RUN_SEQ_BAND

    def lowest_of(events: Iterable[EventRow]) -> int | None:
        mine = [event.seq for event in events if floor < event.seq <= ceiling]
        return min(mine) if mine else None

    lowest: int | None = None
    for event_type in (RunEventType.HEARTBEAT, RunEventType.RUN_STATUS):
        newest = reader.list_events(
            external_id, tail=True, limit=FRONTIER_PAGE, event_type=event_type
        )
        found = lowest_of(newest.events)
        if found is None:
            found = _scan_for_frontier(reader, external_id, event_type, lowest_of)
        if found is not None and (lowest is None or found < lowest):
            lowest = found
    return lowest


def _scan_for_frontier(
    reader: RunsReader,
    external_id: str,
    event_type: str,
    lowest_of: Callable[[Iterable[EventRow]], int | None],
) -> int | None:
    """Page one event type from the start of the log, returning the band's lowest seq.

    Only reached when the newest page held nothing for this band, which on a chunked arm means a
    sibling's traffic pushed it out. Gives up rather than guessing past `FRONTIER_MAX_PAGES`,
    because a partial answer is worse than none here: it would be too HIGH, and rebasing to a seq
    above the true frontier re-issues the ones between.
    """
    cursor = 0
    lowest: int | None = None
    for _page in range(FRONTIER_MAX_PAGES):
        page = reader.list_events(
            external_id, after_pos=cursor, limit=FRONTIER_PAGE, event_type=event_type
        )
        found = lowest_of(page.events)
        if found is not None and (lowest is None or found < lowest):
            lowest = found
        if not page.events or page.last_pos <= cursor:
            return lowest
        cursor = page.last_pos
    log.warning(
        "run telemetry for %s: gave up locating the %s frontier after %d pages, so this "
        "invocation's heartbeats may collide with the previous one's and be discarded. The run "
        "itself is unaffected.",
        external_id,
        event_type,
        FRONTIER_MAX_PAGES,
    )
    return None


def platform_frontier(external_id: str, band: int) -> int | None:
    """`frontier_from_reader` against the saved credential, or None when this machine cannot read.

    Never raises: a machine that cannot read simply does not rebase (`_rebase_descending` treats
    None as "leave the ceiling alone"), which costs a resumed run's heartbeats and not the run.
    """
    reader = RunsReader.open()
    if reader is None:
        return None
    with reader:
        return frontier_from_reader(reader, external_id, band)


def _undeclared(refused: PushRejected) -> bool:
    """Whether a refusal means "this run was never declared" rather than "this payload is wrong".

    Matched on the ingest's own wording because the status alone cannot tell them apart: both come
    back as 422, but one is fixed by restating `run.meta` and the other by fixing the payload.
    """
    message = str(refused).lower()
    return refused.status_code == 422 and ("run.meta" in message or "first batch" in message)


def now_iso() -> str:
    """This instant, as the wire's timestamp. Live emission is the ONLY place a clock is read.

    A backfill takes every timestamp from the artifacts (`wmo.optimize.telemetry.backfill`); a
    live hook has no artifact to read yet, which is the whole difference between the two paths.
    """
    return datetime.now(UTC).isoformat()


class GridSnapshot(BaseModel):
    """A whole-run progress and spend snapshot, summed across every process feeding the run.

    Whole-run rather than per-process on purpose: a heartbeat assembled from one of three
    processes' own counters would report a third of the grid, and the panel's numbers would jump
    with whichever writer spoke last. The tau runner builds this from the shared ledger.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    done: int
    scored: int
    total: int | None = None
    candidate_usd: float = 0.0
    compressor_usd: float = 0.0
    wm_usd: float = 0.0


class _Reporter:
    """Shared half of both emitters: the never-raise guard, one warning per run, and flushing.

    Not a public surface. Callers hold a `GridEmitter` or a `PipelineEmitter`, and every method on
    those is a no-op when emission is off, so no caller carries an `if emitter` branch.
    """

    def __init__(
        self,
        sink: RunsSink | None,
        external_id: str,
        *,
        queue_limit: int = QUEUE_LIMIT,
        top_band: int | None = None,
        frontier: FrontierReader | None = None,
    ) -> None:
        """Hold the transport for one run, or None when emission is off.

        `top_band` is the band whose DESCENDING walk this writer uses, when it has one, so a resume
        can rebase that walk as well as the ascending one.
        """
        self._sink = sink
        self._top_band = top_band
        self._frontier = frontier
        # The run's declaration. Kept out of the queue's eviction path and re-sendable: the platform
        # refuses every batch of an undeclared run, so losing this one event would end the run's
        # telemetry permanently rather than costing one event.
        self._meta: RunEvent | None = None
        self._resumed = False
        self._resume_pending = False
        self._external_id = external_id
        self._bands = SeqBands()
        self._queue: deque[tuple[RunEvent, bool]] = deque()
        self._resumed_from = 0
        self._pulled_at_resume: tuple[ControlCommand, ...] = ()
        self._queue_limit = queue_limit
        self._dropped = 0
        self._warned = False

    @property
    def enabled(self) -> bool:
        """Whether anything is actually being pushed."""
        return self._sink is not None

    @property
    def external_id(self) -> str:
        """The run this reporter feeds."""
        return self._external_id

    @contextmanager
    def _quiet(self, what: str) -> Iterator[None]:
        """Run a block of emission, absorbing anything it raises.

        Deliberately catching `Exception` rather than the client's own error types: the point is a
        guarantee about the RUN, and a guarantee that holds only for the failures anticipated today
        is not one. A paid grid must not die because telemetry found a new way to break.
        """
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - telemetry never propagates into a paid run
            self._warn(f"{what}: {exc}")

    def _resume(self) -> int:
        """Probe what the platform already holds and continue numbering past it.

        Called once, before a run's first event. Without it, a re-invocation of a run that already
        reported (a resumed pipeline, a restarted arm) renumbers from the band floor, every event
        is discarded as a replay of a seq the platform already has, and the run looks healthy while
        its new telemetry silently vanishes. A fresh run answers 0 and numbering starts at the
        floor as before.

        Returns:
            The run's resume mark, or 0 when the platform has never heard of it.
        """
        sink = self._sink
        if sink is None:
            return 0
        self._resume_pending = False
        ack: PushAck | None = None
        try:
            ack = sink.probe(self._external_id)
        except Exception as exc:  # noqa: BLE001 - telemetry never propagates into a paid run
            # An unreachable platform is NOT a fresh run. Numbering from the floor on this guess
            # would have every event of a resumed run discarded as a replay, so the probe is
            # retried at the next flush instead.
            # Retried at the next flush rather than standing as "fresh run" forever. Events this
            # emitter numbers before that retry succeeds may still be discarded as replays, because
            # seqs are assigned when an event is queued; what this prevents is a resumed run
            # numbering from the floor for its whole life.
            self._resume_pending = True
            self._warn(f"could not read the run's resume mark, will retry: {exc}")
            return 0
        # Only a mark INSIDE band 0 says anything about the run-level walk: a grid arm whose chunk
        # bands pushed `last_seq` to 100004 must leave band 0 starting where the derived ledger
        # walk expects it, or every ledger line would be renumbered off its artifact position.
        if 0 < ack.last_seq <= RUN_SEQ_BAND:
            self._bands.resume_at(RUN_LEVEL_BAND, ack.last_seq + 1)
        # And the descending walk, which nothing else rebases and which `last_seq` cannot locate
        # (see `_rebase_descending`).
        self._rebase_descending()
        self._resumed_from = ack.last_seq
        self._resumed = ack.last_seq > 0
        self._pulled_at_resume = ack.control
        return ack.last_seq

    def _rebase_descending(self) -> None:
        """Continue the descending walk below where the previous invocation left it.

        This needs a READ, and `last_seq` cannot substitute for it: a descending walk's frontier is
        the MINIMUM of the seqs it issued, `last_seq` is the maximum over the whole run, and a
        maximum cannot locate a minimum. Rebasing from `last_seq - 1` re-issues every descending seq
        but the first, which costs the terminal status and leaves a finished run reading `running`.

        Reading it is cheap because the newest descending event IS the frontier: each successive one
        is lower than the last. A run with no descending events yet has no frontier, and the ceiling
        is already correct, so None means "leave it alone".
        """
        band = self._top_band
        if band is None or self._frontier is None:
            return
        frontier: int | None = None
        with self._quiet("could not locate the run's heartbeat frontier"):
            frontier = self._frontier(self._external_id, band)
        if frontier is None:
            return
        with self._quiet("could not rebase the heartbeat walk"):
            self._bands.resume_from_top(band, frontier - 1)

    def _emit(self, band: int, event_type: str, ts: str, payload: JsonObject) -> None:
        """Number one event from its band's ascending walk and hold it for the next flush.

        Numbering happens here rather than at push time so a batch's seqs follow the order the
        facts happened in, which is the order a replayed backfill has to agree with.
        """
        self._queue_event(band, event_type, ts, payload, self._bands.take)

    def _emit_at(self, seq: int, event_type: str, ts: str, payload: JsonObject) -> None:
        """Queue one event at a seq DERIVED from the artifacts, not allocated from a counter.

        The two facts a live run and a later backfill both describe (`run.meta`, and each
        `ledger.line` at its position in the ledger file) get their seq from the artifact itself,
        so the two paths converge on one identity instead of writing the same fact twice. A
        derived seq the platform already holds is therefore expected, not a collision, which is
        why `_check_accepted` counts it separately.
        """
        if self._sink is None:
            return
        with self._quiet(f"could not queue a {event_type} event"):
            self._enqueue(
                RunEvent(
                    external_id=self._external_id,
                    seq=seq,
                    ts=ts,
                    type=event_type,
                    payload=payload,
                ),
                derived=True,
            )

    def _emit_from_top(self, band: int, event_type: str, ts: str, payload: JsonObject) -> None:
        """Queue one event descending from its band's ceiling.

        For facts with no artifact position (a heartbeat, a live `run.status`), which therefore
        cannot be re-derived. Descending keeps them clear of the ascending derived walk in the same
        band, so neither has to know the other's count; `SeqBands` raises if the two ever meet.
        """
        self._queue_event(band, event_type, ts, payload, self._bands.take_from_top)

    def _queue_event(
        self,
        band: int,
        event_type: str,
        ts: str,
        payload: JsonObject,
        allocate: Callable[[int], int],
    ) -> None:
        """Allocate a seq with `allocate` and queue the event. Silent when emission is off."""
        if self._sink is None:
            return
        with self._quiet(f"could not queue a {event_type} event"):
            self._enqueue(
                RunEvent(
                    external_id=self._external_id,
                    seq=allocate(band),
                    ts=ts,
                    type=event_type,
                    payload=payload,
                ),
                derived=False,
            )

    def _enqueue(self, event: RunEvent, *, derived: bool) -> None:
        """Hold one numbered event, remembering where its seq came from."""
        if event.type == RunEventType.RUN_META:
            self._meta = event
        self._queue.append((event, derived))
        self._drop_overflow()

    def _drop_overflow(self) -> None:
        """Keep the queue bounded, oldest first.

        A platform outage during a twelve-hour grid must cost telemetry, never memory. Oldest
        first because the newest events are the ones a panel is waiting on, and because the run's
        own artifacts remain the durable record either way: what is lost here is recoverable with
        `wmo runs backfill`.
        """
        while len(self._queue) > self._queue_limit:
            # Oldest first, EXCEPT the declaration: it is the oldest event by definition, and the
            # platform refuses every batch of a run that has not been declared, so evicting it would
            # cost the whole run's telemetry rather than one event.
            victim = 0
            if len(self._queue) > 1 and self._queue[0][0] is self._meta:
                victim = 1
            del self._queue[victim]
            self._dropped += 1
            if self._dropped == 1 or self._dropped % self._queue_limit == 0:
                log.warning(
                    "run telemetry for %s has dropped %d queued event(s): the queue is full at %d "
                    "and the oldest are discarded. The run is unaffected, and `wmo runs backfill` "
                    "can replay the run's artifacts afterwards.",
                    self._external_id,
                    self._dropped,
                    self._queue_limit,
                )

    def _flush(self) -> tuple[ControlCommand, ...]:
        """Push what is queued and return any control commands that rode back.

        A failed push puts its events back at the FRONT of the queue, so the next flush retries
        them in order; the bound is what keeps that from growing without limit.
        """
        sink = self._sink
        if sink is None or not self._queue:
            return self._take_resume_control()
        if self._resume_pending:
            # The resume probe never got an answer. Ask again before numbering anything else on the
            # assumption that this run is new.
            self._resume()
        batch = list(self._queue)
        self._queue.clear()
        events = [event for event, _derived in batch]
        try:
            ack = self._push(sink, events)
        except PushRejected as refused:
            # PERMANENT. Retrying cannot help, and requeueing it would head-of-line block this
            # run's telemetry for the rest of a twelve-hour grid on one bad payload, so the batch
            # is dropped and the reason said out loud every time rather than once per run: each
            # rejection is a distinct bug, and the sink's message names the field that caused it.
            log.warning(
                "run telemetry for %s: the platform refused %d event(s) with HTTP %s and they have "
                "been DROPPED (retrying cannot fix a refused payload): %s",
                self._external_id,
                len(events),
                refused.status_code,
                refused,
            )
            return self._take_resume_control()
        except Exception as exc:  # noqa: BLE001 - telemetry never propagates into a paid run
            # Transient: the platform, the network, or something unforeseen. Keep the events at the
            # FRONT so the next flush retries them in order; the bound is what limits the growth.
            self._warn(f"could not push {len(events)} event(s): {exc}")
            self._queue.extendleft(reversed(batch))
            self._drop_overflow()
            return self._take_resume_control()
        self._check_accepted(ack, batch)
        return self._take_resume_control() + ack.control

    def _push(self, sink: RunsSink, events: list[RunEvent]) -> PushAck:
        """Push one batch, restating the run's declaration once if that is what it was missing.

        The platform refuses any batch of a run it has not been told about, and that can happen
        even though this emitter declared the run: an outage can swallow the batch the declaration
        travelled in. Restating it costs nothing (its seq is derived, so a platform that already
        holds it discards the copy) and it is the difference between a run recovering and a run
        reporting nothing for the rest of its life.

        Raises:
            PushRejected: For any other permanent refusal, and for a second declaration failure.
        """
        try:
            return sink.push(self._external_id, events)
        except PushRejected as refused:
            meta = self._meta
            if meta is None or not _undeclared(refused) or any(event is meta for event in events):
                raise
            log.warning(
                "run telemetry for %s: the platform does not know this run, so its declaration is "
                "being restated and the batch retried once (%s)",
                self._external_id,
                refused,
            )
            return sink.push(self._external_id, [meta, *events])

    def _take_resume_control(self) -> tuple[ControlCommand, ...]:
        """Commands the resume probe pulled, handed over once.

        The probe is a run's first contact, so a stop issued before this process started is waiting
        there; dropping it would make a run ignore a command it had already been given.
        """
        pulled, self._pulled_at_resume = self._pulled_at_resume, ()
        return pulled

    def _check_accepted(self, ack: PushAck, batch: list[tuple[RunEvent, bool]]) -> None:
        """The collision tripwire, counted against ALLOCATED seqs only.

        An artifact-DERIVED seq the platform already holds is the resume converging on itself, and
        is exactly what is supposed to happen when a re-invocation re-states `run.meta` or a ledger
        line. An ALLOCATED seq it did not take is the real fault: another writer is numbering into
        this band and one of us is being discarded, which would otherwise read as a healthy run
        with a quietly incomplete cell matrix.
        """
        allocated = sum(1 for _event, derived in batch if not derived)
        sent = len(batch)
        if ack.accepted >= sent:
            return
        if self._resumed:
            # A resumed run re-states facts on purpose, and some of its allocated seqs may have been
            # used by the invocation it is continuing. Saying "collision" here on every ordinary
            # resume is how the same message gets ignored when it means real loss.
            log.info(
                "run telemetry for %s: %d of %d event(s) were already recorded, which is a resume "
                "converging on what the platform holds",
                self._external_id,
                sent - ack.accepted,
                sent,
            )
            return
        if ack.accepted >= allocated:
            log.info(
                "run telemetry for %s: %d of %d event(s) were already recorded, converging on a "
                "resume as intended (artifact-derived positions restate facts the platform holds)",
                self._external_id,
                sent - ack.accepted,
                sent,
            )
            return
        log.warning(
            "run telemetry for %s: only %d of %d freshly allocated event(s) were accepted "
            "(last_seq %d). Colliding seqs mean two writers share one band, so events are being "
            "silently discarded; check that concurrent processes were given disjoint --chunks "
            "ranges.",
            self._external_id,
            ack.accepted,
            allocated,
            ack.last_seq,
        )

    def _ack(self, control: ControlCommand, *, status: str, note: str | None = None) -> None:
        """Answer one command. A lost ack just means the panel asks again."""
        sink = self._sink
        if sink is None:
            return
        with self._quiet(f"could not ack control {control.id}"):
            sink.ack(self._external_id, control.id, status=status, note=note)

    def close(self) -> None:
        """Send whatever is queued and release the telemetry connection.

        Called once at the end of a run: a multi-arm process opens a sink per arm, and leaving them
        open holds an idle connection pool per finished run.
        """
        self._flush()
        sink = self._sink
        if sink is None:
            return
        with self._quiet("could not release the telemetry connection"):
            sink.close()

    def _warn(self, message: str) -> None:
        """One warning per run, then silence: a broken platform must not flood a run's log."""
        if self._warned:
            log.debug("run telemetry for %s: %s", self._external_id, message)
            return
        self._warned = True
        log.warning(
            "run telemetry for %s: %s. The run continues; later telemetry problems are logged at "
            "debug level only.",
            self._external_id,
            message,
        )


def _open(external_id: str, factory: SinkFactory | None, *, requested: bool) -> RunsSink | None:
    """Resolve emission for one run and say, once, which way it went.

    Anything missing (`--no-emit`, no credential, no organization, a client that will not build)
    turns emission off. None of them fails the run.
    """
    if not requested:
        log.info("run telemetry off for %s (--no-emit)", external_id)
        return None
    resolve = factory if factory is not None else runs_sink
    try:
        emitter = resolve()
    except Exception as exc:  # noqa: BLE001 - a client that cannot be built turns emission off
        log.info(
            "run telemetry off for %s: the platform client would not build (%s)", external_id, exc
        )
        return None
    if emitter is None:
        # `runs_sink` has already said which half was missing (credential, or organization).
        log.info("run telemetry off for %s; the run is unaffected", external_id)
        return None
    log.info("run telemetry on for %s", external_id)
    return emitter


class GridEmitter(_Reporter):
    """The tau grid runner's view of the runs surface: one arm, one platform run.

    Cells are buffered and sent on size or age, because a grid buys a cell every few tens of
    seconds for hours and one request per cell would be one request per episode.
    """

    def __init__(
        self,
        sink: RunsSink | None,
        *,
        external_id: str,
        band: int,
        arm: str,
        snapshot: Callable[[], GridSnapshot] | None = None,
        flush_cells: int = CELL_BATCH_CAP,
        flush_seconds: float = HEARTBEAT_INTERVAL_S,
        queue_limit: int = QUEUE_LIMIT,
        frontier: FrontierReader | None = None,
    ) -> None:
        """Build the emitter directly. Most callers want `create`, which resolves the sink."""
        # `top_band` is this writer's own band: its heartbeats and terminal status descend from that
        # ceiling, so a resume has to rebase that walk too or it re-issues the same seqs.
        super().__init__(
            sink, external_id, queue_limit=queue_limit, top_band=band, frontier=frontier
        )
        self._band = band
        self._arm = arm
        self._snapshot = snapshot
        self._flush_cells = flush_cells
        self._flush_seconds = flush_seconds
        self._buffered: dict[int, list[JsonObject]] = {}
        self._buffered_count = 0
        self._last_send = time.monotonic()
        self._last_heartbeat = 0.0
        self._stop_requested = False
        self._stop_control: list[ControlCommand] = []
        self._lock = Lock()

    @classmethod
    def create(
        cls,
        *,
        grid_relpath: str,
        arm: str,
        band: int,
        snapshot: Callable[[], GridSnapshot] | None = None,
        factory: SinkFactory | None = None,
        requested: bool = True,
        flush_cells: int = CELL_BATCH_CAP,
        flush_seconds: float = HEARTBEAT_INTERVAL_S,
        frontier: FrontierReader | None = None,
    ) -> GridEmitter:
        """Build the emitter for one arm, off unless a credential and an organization resolve.

        Args:
            grid_relpath: The grid directory relative to `.wmo` (e.g. `jt/grid-c2`), which with
                `arm` forms the run's stable external id.
            arm: Arm name.
            band: This process's run-level band. Pass `cell_band(first_owned_chunk)` so several
                processes on disjoint chunk ranges never share a band.
            snapshot: Whole-run progress and spend, re-read once per heartbeat.
            factory: Opens the transport; defaults to the saved platform credential.
            requested: False for `--no-emit`, which turns emission off without consulting
                credentials at all.
            flush_cells: Cells buffered before a send.
            flush_seconds: Age at which a partial buffer is sent anyway.
            frontier: Locates a resumed run's descending frontier; defaults to reading it from the
                platform. Only consulted when emission is on.
        """
        external_id = grid_arm_external_id(grid_relpath, arm)
        return cls(
            _open(external_id, factory, requested=requested),
            external_id=external_id,
            band=band,
            arm=arm,
            snapshot=snapshot,
            flush_cells=flush_cells,
            flush_seconds=flush_seconds,
            frontier=frontier if frontier is not None else platform_frontier,
        )

    @property
    def arm(self) -> str:
        """The arm this emitter reports.

        The name it was CONSTRUCTED with, not one parsed back out of the run id: an arm whose name
        contains a slash would split differently coming back, and this value is what the runner
        compares a ledger line's arm against before reporting it. A mismatch there would file a
        sibling arm's line under the wrong run at a colliding seq.
        """
        return self._arm

    @property
    def stop_requested(self) -> bool:
        """Whether the panel asked this run to stop. Checked by the runner at chunk boundaries."""
        return self._stop_requested

    def on_arm_start(self, *, cohort: JsonObject, world_model: str, created: str) -> None:
        """Declare the run. `run.meta` must be in a new run's first batch, so this flushes.

        Resumes first: an arm re-run after a stop has already reported, and numbering from the
        floor again would have every event discarded as a replay. `run.meta` then goes to its
        DERIVED position (band 0 seq 1), which is where a backfill of the same arm puts it, so the
        two converge on one declaration instead of writing two.
        """
        self._resume()
        self._emit_at(
            RUN_META_SEQ,
            RunEventType.RUN_META,
            created,
            {
                "kind": RunKind.GRID_ARM.value,
                "benchmark": world_model,
                "arm": self.arm,
                "world_model": world_model,
                "config": cohort,
                "started_at": created,
            },
        )
        self._handle(self._flush())

    def on_outcome(self, outcome: ScenarioOutcome, *, chunk: int) -> None:
        """One measured cell. Buffered, then sent on `flush_cells` cells or `flush_seconds`.

        The mapping is `wmo.optimize.telemetry.backfill.cell_payload`, deliberately the same
        function the backfill uses, so a cell reported live and the same cell replayed from its
        chunk file are one payload rather than two dialects that drift.
        """
        if not self.enabled:
            return
        with self._quiet("could not buffer a cell"), self._lock:
            self._buffered.setdefault(chunk, []).append(
                cell_payload(outcome.model_dump(mode="json"), chunk=chunk)
            )
            self._buffered_count += 1
        if self._due():
            self.send_cells()

    def _due(self) -> bool:
        with self._lock:
            return self._buffered_count >= self._flush_cells or (
                bool(self._buffered) and time.monotonic() - self._last_send >= self._flush_seconds
            )

    def send_cells(self) -> None:
        """Send every buffered cell, batched per chunk into that chunk's own band."""
        with self._lock:
            pending = self._buffered
            self._buffered = {}
            self._buffered_count = 0
            self._last_send = time.monotonic()
        if not pending:
            return
        ts = now_iso()
        for chunk, cells in sorted(pending.items()):
            band = cell_band(chunk)
            for start in range(0, len(cells), CELL_BATCH_CAP):
                self._emit(
                    band,
                    RunEventType.CELL_BATCH,
                    ts,
                    {"cells": cells[start : start + CELL_BATCH_CAP]},
                )
        self._heartbeat()
        self._handle(self._flush())

    def on_ledger_line(self, line: JsonObject, *, ts: str, position: int) -> None:
        """One appended ledger line, verbatim, plus the heartbeat its arrival earns.

        Every ledger line goes through one place in the runner (`GridState.append`), which is why
        this single hook covers chunks, retries, calibration, merges, and stops.

        Args:
            line: The stamped ledger line, passed through untouched.
            ts: The line's own timestamp.
            position: Its 1-based position among the ARM's ledger lines, which is what the seq is
                derived from. Fixed the moment the line is appended, so this process, a sibling
                process, and a later backfill all compute the same seq for the same line, and the
                platform keeps one copy of it rather than three.
        """
        self.send_cells()
        self._emit_at(ledger_walk_seq(position), LEDGER_LINE, ts, dict(line))
        self._heartbeat(force=True)
        self._handle(self._flush())

    def on_status(self, status: RunStatus, *, error: str | None = None) -> None:
        """The run's terminal transition. Sends buffered cells first, so none are lost.

        A status this build cannot name is skipped rather than sent (`_status_value`), but the run
        still ENDED, so everything else here happens anyway. In particular a pending `stop` is acked
        either way: the command WAS honored, and leaving it pending would show a stop as in-flight
        forever on a run that has already finished.
        """
        self.send_cells()
        finished_at = now_iso()
        value = _status_value(status)
        if value is not None:
            payload: JsonObject = {"status": value, "finished_at": finished_at}
            if error is not None:
                payload["error"] = error
            self._emit_from_top(self._band, RunEventType.RUN_STATUS, finished_at, payload)
        self._flush()
        for control in self._stop_control:
            self._ack(control, status=ACK_DONE, note=f"run {status}")
        self._stop_control = []

    def _heartbeat(self, *, force: bool = False) -> None:
        """A whole-run snapshot, read from the shared ledger rather than local counters."""
        if self._snapshot is None or not self.enabled:
            return
        if not force and time.monotonic() - self._last_heartbeat < self._flush_seconds:
            return
        snapshot: GridSnapshot | None = None
        with self._quiet("could not read a progress snapshot"):
            snapshot = self._snapshot()
        if snapshot is None:
            return
        self._last_heartbeat = time.monotonic()
        # From the TOP of the band: a heartbeat has no artifact position, so it cannot be
        # re-derived, and descending keeps it clear of the ascending derived walk below it.
        self._emit_from_top(
            self._band,
            RunEventType.HEARTBEAT,
            now_iso(),
            {
                "progress": {
                    "done": snapshot.done,
                    "total": snapshot.total,
                    "scored": snapshot.scored,
                },
                "spend": {
                    "candidate_usd": snapshot.candidate_usd,
                    "compressor_usd": snapshot.compressor_usd,
                    "wm_usd": snapshot.wm_usd,
                },
            },
        )

    def _handle(self, commands: tuple[ControlCommand, ...]) -> None:
        """Answer pulled commands: honor stop, decline what this runner does not own."""
        for control in commands:
            if control in self._stop_control:
                continue
            if control.command == CONTROL_STOP:
                self._stop_requested = True
                self._stop_control.append(control)
                self._ack(
                    control,
                    status=ACK_ACKED,
                    note="stopping at the next chunk boundary; finished chunks stay on disk",
                )
                log.warning(
                    "the platform asked run %s to stop; stopping at the next chunk boundary",
                    self.external_id,
                )
            elif control.command == CONTROL_RETRY_UNSCORED:
                self._ack(control, status=ACK_REJECTED, note=REJECT_RETRY_UNSCORED)
            elif control.command == CONTROL_FORCE_FROM_STAGE:
                self._ack(control, status=ACK_REJECTED, note=REJECT_FORCE_FROM_STAGE)
            else:
                self._ack(
                    control,
                    status=ACK_REJECTED,
                    note=f"the grid runner does not implement the '{control.command}' command",
                )


class PipelineEmitter(_Reporter):
    """`wmo optimize model`'s view of the runs surface: one world model, one staged run.

    Single-process, so it uses band 0 alone and its heartbeats are trivially whole-run.
    """

    def __init__(self, sink: RunsSink | None, *, external_id: str) -> None:
        """Build the emitter directly. Most callers want `create`, which resolves the sink."""
        super().__init__(sink, external_id)
        self._stages_done = 0

    @classmethod
    def create(
        cls,
        *,
        world_model: str,
        factory: SinkFactory | None = None,
        requested: bool = True,
    ) -> PipelineEmitter:
        """Build the emitter for one world model's pipeline, off unless a credential resolves."""
        external_id = pipeline_external_id(world_model)
        return cls(_open(external_id, factory, requested=requested), external_id=external_id)

    def start(self, *, world_model: str, config: JsonObject) -> None:
        """Declare the run and mark it running, continuing a previous invocation's numbering.

        `wmo optimize model` is resumable by design: the second invocation skips the stages whose
        artifacts are current and runs the rest. That makes a re-invocation the NORMAL case, so it
        resumes the seq walk from what the platform already holds rather than restarting at the
        floor and having every event discarded as a replay.
        """
        self._resume()
        started_at = now_iso()
        # Seq 1 is the declaration's, reserved. Without moving the ascending walk past it the first
        # ALLOCATED event (the running transition, below) takes seq 1 as well and the platform
        # discards it, so a fresh run never records that it started.
        self._bands.resume_at(RUN_LEVEL_BAND, RUN_META_SEQ + 1)
        self._emit_at(
            RUN_META_SEQ,
            RunEventType.RUN_META,
            started_at,
            {
                "kind": RunKind.PIPELINE.value,
                "benchmark": world_model,
                "world_model": world_model,
                "config": config,
                "started_at": started_at,
            },
        )
        self._emit(
            RUN_LEVEL_BAND, RunEventType.RUN_STATUS, started_at, {"status": RunStatus.RUNNING.value}
        )
        self._flush()

    def stage_running(self, stage: Stage) -> None:
        """A stage started: what makes the panel's stage table move during a long sweep."""
        started_at = now_iso()
        self._emit(
            RUN_LEVEL_BAND,
            RunEventType.STAGE_UPSERT,
            started_at,
            {"stage": stage.value, "status": "running", "started_at": started_at},
        )
        self._flush()

    def stage_skipped(self, stage: Stage, *, reason: str) -> None:
        """A stage the resume logic skipped, carrying the reason it printed.

        Reported rather than left blank: "skipped because the matrix is current" is the single most
        useful thing the panel can say about a re-run, and it is exactly what the terminal says.
        """
        self._emit(
            RUN_LEVEL_BAND,
            RunEventType.STAGE_UPSERT,
            now_iso(),
            {"stage": stage.value, "status": "skipped", "artifact": {"reason": reason}},
        )
        self._flush()

    def stage_completed(
        self, record: StageRecord, *, lifetime_candidate_usd: float, lifetime_wm_usd: float
    ) -> None:
        """A stage finished, with the fingerprint and artifact identity it recorded.

        Takes the `StageRecord` the manifest just persisted, so the event and the manifest cannot
        disagree about what the stage produced or what it cost. The lifetime figures are the
        manifest's `lifetime_split` legs.
        """
        self._stages_done += 1
        self._emit(
            RUN_LEVEL_BAND,
            RunEventType.STAGE_UPSERT,
            record.completed_at,
            {
                "stage": record.stage.value,
                "status": "completed",
                "fingerprint": dict(record.fingerprint),
                "spend": {
                    "candidate_usd": record.spend_usd,
                    "compressor_usd": record.compressor_spend_usd,
                    "wm_usd": record.world_model_spend_usd,
                },
                "completed_at": record.completed_at,
                "artifact": {
                    "artifact_path": record.artifact_path,
                    "artifact_identity": record.artifact_identity,
                },
            },
        )
        self.heartbeat(
            stage=record.stage,
            lifetime_candidate_usd=lifetime_candidate_usd,
            lifetime_wm_usd=lifetime_wm_usd,
        )

    def heartbeat(
        self, *, stage: Stage | None, lifetime_candidate_usd: float, lifetime_wm_usd: float
    ) -> None:
        """Where the pipeline is and what it has spent, at a stage boundary.

        The spend is the manifest's LIFETIME total — the number the run's own cap is checked
        against (`SpendLedger`) — reported as its two legs. The platform's run row stores the
        legs exactly as reported and totals them (`candidate_usd + wm_usd`), so folding both
        sides into `candidate_usd` (as an earlier version did) inflated the candidate leg,
        rendered the world-model leg as unreported, and made the panel's cells-vs-ledger
        reconciliation fire on every pipeline run.
        """
        self._emit(
            RUN_LEVEL_BAND,
            RunEventType.HEARTBEAT,
            now_iso(),
            {
                "progress": {
                    "done": self._stages_done,
                    "total": None,
                    "stage": stage.value if stage is not None else None,
                },
                "spend": {
                    "candidate_usd": lifetime_candidate_usd,
                    "wm_usd": lifetime_wm_usd,
                },
            },
        )
        self._flush()

    def finished(self, status: RunStatus, *, error: str | None = None) -> None:
        """The run's terminal transition: completed, failed, or stopped by the spend cap.

        A status this build cannot name is skipped rather than sent (`_status_value`); the flush
        still happens, so whatever the last stage queued is not stranded behind it.
        """
        finished_at = now_iso()
        value = _status_value(status)
        if value is not None:
            payload: JsonObject = {"status": value, "finished_at": finished_at}
            if error is not None:
                payload["error"] = error
            self._emit(RUN_LEVEL_BAND, RunEventType.RUN_STATUS, finished_at, payload)
        self._flush()
