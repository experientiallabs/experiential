"""Reading runs back: the same list, detail, cells, log, and tail the platform panel renders.

An agent driving a twelve-hour grid from a terminal should not have to open a browser to answer
"is it still running, how far in, what has it spent, and what failed". These are the org-scoped
read routes behind `wmo runs list/show/tail`, typed the way the server serves them.

Two coordinates travel with a run's log and they are not interchangeable:

- `pos` is the SERVER's arrival position. It is what every cursor, every page, and every SSE
  frame id orders by, and it is the only safe thing to resume from.
- `seq` is the EMITTER's idempotency key, banded per writer (`wmo.runtime.runs.schema`). A late
  write from a low band has a low seq and a high pos, so ordering by seq would permanently hide it.

A live run's reads deliberately stop about two seconds behind real time (the server's safe
frontier), which is why a tail can resume from `last_pos` without a gap: nothing can arrive behind
a position the server already served. The lag is released the moment the run goes terminal, and the
stream then closes on its own once drained, which is what makes `wmo runs tail` exit rather than
hang on a finished run.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections.abc import Iterator

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.core.types import JsonObject
from wmo.runtime.platform.client import PlatformClient, PlatformError, PlatformUnreachable
from wmo.runtime.platform.credentials import load_credentials
from wmo.runtime.runs.schema import is_terminal_status

log = logging.getLogger(__name__)

DEFAULT_PAGE = 50
"""Runs per list page, matching the route's own default."""

DEFAULT_EVENT_PAGE = 500
"""Events per log page, matching the route's own default."""

STREAM_TIMEOUT_S = 300.0
"""How long a tail waits on a silent stream before reconnecting. Comfortably past the 15s heartbeat
cadence, so a healthy quiet run is never mistaken for a dead connection."""

RECONNECT_DELAY_S = 2.0
"""Pause before a tail reconnects. Matches the server's safe-frontier lag, so a reconnect lands
where the next frame is about to be servable rather than spinning."""

RECONNECT_JITTER = 0.25
"""Fraction of the delay randomized. Several tails of one run (a person and a script, or a fleet of
agents) otherwise reconnect in lockstep after an outage and arrive as a thundering herd."""

MAX_RECONNECTS = 60
"""Reconnect attempts before a tail gives up and says so. At `RECONNECT_DELAY_S` that is two
minutes of a platform being unreachable, which is a real outage rather than a blip."""


def _resolve_org(client: PlatformClient, wanted: str) -> str:
    """Turn an organization id or slug into the id the routes take.

    A uuid is passed through untouched and costs no request. Anything else is looked up against
    the credential's own organizations by slug, then by name, so `WMO_PLATFORM_ORG=acme` and
    `--org acme` work as well as the uuid does.

    Raises:
        PlatformError: The name matches none of the organizations this credential can see, listing
            what it can.
    """
    try:
        uuid.UUID(wanted)
    except ValueError:
        pass
    else:
        return wanted
    orgs = client.whoami().orgs
    for org in orgs:
        if wanted in (org.slug, org.name):
            return org.id
    visible = ", ".join(f"{org.slug} ({org.name})" for org in orgs) or "none"
    msg = (
        f"no organization named {wanted!r} is visible to this credential; it can see: {visible}. "
        "Pass --org with one of those slugs, or a uuid."
    )
    raise PlatformError(msg)


class RunSummary(BaseModel):
    """One row of the run list, as the panel's table shows it.

    `extra="ignore"`, unlike most models in wmo: the shape belongs to the platform, and a field
    added there must not break every installed CLI.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    external_id: str
    kind: str
    status: str
    org_name: str | None = None
    benchmark: str | None = None
    arm: str | None = None
    progress: JsonObject = Field(default_factory=dict)
    candidate_usd: float | None = None
    compressor_usd: float | None = None
    wm_usd: float | None = None
    error: str | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    finished_at: str | None = None

    @property
    def spend_usd(self) -> float:
        """Everything this run has spent: candidate plus world model.

        `compressor_usd` is deliberately NOT added: it is a subset of the candidate side (a
        compressor's inference is folded into the cost of the arm it compressed), so adding it
        would bill it twice.
        """
        return (self.candidate_usd or 0.0) + (self.wm_usd or 0.0)


class StageRow(BaseModel):
    """One stage of a staged run."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    stage: str
    status: str
    fingerprint: JsonObject | None = None
    artifact: JsonObject | None = None
    candidate_usd: float | None = None
    compressor_usd: float | None = None
    wm_usd: float | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class CellStats(BaseModel):
    """One candidate model's cell rollup: how many ran, how many scored, how many failed."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    cell_count: int
    scored_count: int
    error_count: int
    unpriced_count: int = 0
    cost_usd_total: float | None = None
    reward_mean: float | None = None


class CellRow(BaseModel):
    """One measured cell."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cell_key: str
    scenario_id: str
    model: str
    episode: int
    chunk: int | None = None
    reward: float | None = None
    success: bool | None = None
    steps: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    cost_usd: float | None = None
    usage: JsonObject | None = None
    detail: JsonObject | None = None


class ControlRow(BaseModel):
    """One control command queued against a run."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    command: str
    args: JsonObject = Field(default_factory=dict)
    status: str
    note: str | None = None
    created_at: str | None = None
    acked_at: str | None = None
    resolved_at: str | None = None


class RunDetail(BaseModel):
    """One run with its stages, per-model cell rollup, and pending commands."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    run: RunSummary
    org: JsonObject | None = None
    event_count: int = 0
    stages: tuple[StageRow, ...] = ()
    cell_stats: tuple[CellStats, ...] = ()
    pending_control: tuple[ControlRow, ...] = ()


class EventRow(BaseModel):
    """One event out of a run's log, and the body of one SSE frame."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    pos: int
    seq: int
    type: str
    payload: JsonObject = Field(default_factory=dict)
    ts: str


class RunPage(BaseModel):
    """One keyset page of the run list.

    The cursor is two fields rather than one opaque string because that is what the route takes
    back: `cursor_ts` and `cursor_id`, echoed from the previous page's last row.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    runs: tuple[RunSummary, ...] = ()
    next_cursor: JsonObject | None = None


class CellPage(BaseModel):
    """One keyset page of a run's cells, `cell_key` ascending."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cells: tuple[CellRow, ...] = ()
    next_cursor_key: str | None = None


class EventPage(BaseModel):
    """One page of a run's log in arrival order, plus the cursor to resume from."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    events: tuple[EventRow, ...] = ()
    last_pos: int = 0


class RunsReader:
    """The org-scoped read surface for runs, over one authenticated platform client."""

    def __init__(self, client: PlatformClient, org_id: str) -> None:
        """Wrap an authenticated platform client, scoping every read to one organization."""
        self._client = client
        self._org_id = org_id

    @classmethod
    def open(
        cls, *, org: str | None = None, transport: httpx.BaseTransport | None = None
    ) -> RunsReader | None:
        """Build a reader from the saved credential, or None when this machine is not connected.

        None rather than an exception: "not logged in" is a state a CLI reports in one line with a
        next step, not a traceback.

        Args:
            org: An organization id OR slug, overriding the saved default (and `WMO_PLATFORM_ORG`).
                A slug is resolved against the credential's own organizations, because a slug is
                what a person reads off the platform's URL bar and a uuid is not something anyone
                has to hand.
            transport: Injected HTTP transport, for tests.
        """
        credentials = load_credentials()
        if not credentials.is_complete():
            return None
        if credentials.api_url is None or credentials.token is None:
            return None
        wanted = org or credentials.default_org
        if not wanted:
            return None
        client = PlatformClient(credentials.api_url, credentials.token, transport=transport)
        try:
            resolved = _resolve_org(client, wanted)
        except PlatformError:
            client.close()
            raise
        return cls(client, resolved)

    @property
    def org_id(self) -> str:
        """The organization every read is scoped to."""
        return self._org_id

    @property
    def client(self) -> PlatformClient:
        """The authenticated client behind these reads.

        Exposed so a command that both READS and WRITES a run (`wmo runs backfill`, which checks the
        run then pushes to it) can do both over one connection pool and close it once, instead of
        opening a second client for the write half.
        """
        return self._client

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()

    def __enter__(self) -> RunsReader:
        """Return the reader for use in a `with` block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the HTTP client on the way out, however the block ended."""
        self.close()

    def list_runs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        cursor_ts: str | None = None,
        cursor_id: str | None = None,
        limit: int = DEFAULT_PAGE,
    ) -> RunPage:
        """One page of the org's runs, newest first."""
        return RunPage.model_validate(
            self._client.list_org_runs(
                self._org_id,
                status=status,
                kind=kind,
                cursor_ts=cursor_ts,
                cursor_id=cursor_id,
                limit=limit,
            )
        )

    def get_run(self, external_id: str) -> RunDetail:
        """One run with its stages, cell rollup, and pending commands."""
        return RunDetail.model_validate(self._client.get_org_run(self._org_id, external_id))

    def list_cells(
        self,
        external_id: str,
        *,
        model: str | None = None,
        scored: bool | None = None,
        error: bool | None = None,
        cursor_key: str | None = None,
        limit: int = 100,
    ) -> CellPage:
        """One page of a run's measured cells.

        `scored=False` and `error=True` are different questions and compose: every errored cell is
        unscored, but an unscored cell with no error is one still in flight.
        """
        return CellPage.model_validate(
            self._client.list_org_run_cells(
                self._org_id,
                external_id,
                model=model,
                scored=scored,
                error=error,
                cursor_key=cursor_key,
                limit=limit,
            )
        )

    def list_events(
        self,
        external_id: str,
        *,
        after_pos: int = 0,
        limit: int = DEFAULT_EVENT_PAGE,
        tail: bool = False,
        event_type: str | None = None,
    ) -> EventPage:
        """One page of a run's raw log, in arrival order.

        Args:
            external_id: The run's name.
            after_pos: Exclusive `pos` cursor; ignored when `tail` is set.
            limit: Page size.
            tail: Open at the END of the log (the newest `limit` events, still ascending), which
                is what a live viewer wants for its first paint.
            event_type: Restrict to one event type, in the database rather than here.
        """
        return EventPage.model_validate(
            self._client.list_org_run_events(
                self._org_id,
                external_id,
                after_pos=after_pos,
                limit=limit,
                tail=tail,
                event_type=event_type,
            )
        )

    def request_control(
        self, external_id: str, command: str, args: JsonObject | None = None
    ) -> ControlRow:
        """Queue a control command for whichever process is feeding the run.

        Delivery is pull-based: the row stays pending until that process picks it up on its next
        push and acks it, and queueing does NOT change the run's status. A stop that never reaches
        its machine must not make the panel claim the run stopped.
        """
        payload = self._client.request_org_run_control(
            self._org_id, external_id, command=command, args=args
        )
        control = payload.get("control")
        if not isinstance(control, dict):
            msg = f"the platform accepted the {command} command but answered without a control row"
            raise PlatformError(msg)
        return ControlRow.model_validate(control)

    def tail(self, external_id: str, *, after_pos: int = 0) -> Iterator[EventRow]:
        """Yield a run's events as they arrive, resuming across reconnects.

        Ends when the server closes a drained terminal run's stream. A dropped connection on a
        live run is reconnected from the last `pos` seen, which is exactly the cursor the frame
        ids carry, so no event is skipped or repeated.

        Raises:
            PlatformUnreachable: The stream could not be re-established within `MAX_RECONNECTS`.
        """
        cursor = after_pos
        attempts = 0
        while True:
            served = 0
            try:
                for event in self._stream_once(external_id, cursor):
                    served += 1
                    attempts = 0
                    cursor = event.pos
                    yield event
            except (PlatformUnreachable, httpx.HTTPError) as error:
                attempts += 1
                self._give_up_after(external_id, attempts, error)
                log.debug("run stream for %s dropped (%s); reconnecting", external_id, error)
                self._pause()
                continue
            except PlatformError as error:
                # A 5xx is the platform having a bad minute, not a verdict about this run, so it is
                # retried like a dropped connection. Every other status (401, 403, 404) is an answer
                # and is raised: reconnecting cannot fix a credential or a run that is not there.
                if error.status_code is None or error.status_code < 500:
                    raise
                attempts += 1
                self._give_up_after(external_id, attempts, error)
                log.debug(
                    "run stream for %s failed with %s; retrying", external_id, error.status_code
                )
                self._pause()
                continue
            # A clean close means one of two things: the run is terminal and drained (done), or a
            # live run's connection ended between frames (resume). Asking the run itself is the
            # only way to tell them apart, and it costs one request per close.
            if self._terminal(external_id):
                # One more read AFTER the run went terminal: a live stream trails the safe frontier
                # by a couple of seconds, so the events that made it terminal (its last ledger line,
                # final heartbeat and the terminal status itself) are usually still behind the
                # frontier when the stream closes. Without this pass a tail routinely ends just
                # before the three events a watcher was waiting for.
                for event in self._drain(external_id, cursor):
                    cursor = event.pos
                    yield event
                return
            if served == 0:
                # A clean close that served nothing is not progress, so it counts against the
                # reconnect budget: otherwise a server closing immediately spins here forever and
                # presents to the user as a hang.
                attempts += 1
                self._give_up_after(external_id, attempts, None)
                self._pause()

    def _give_up_after(self, external_id: str, attempts: int, error: Exception | None) -> None:
        """Stop retrying once the budget is spent, saying what was tried.

        Raises:
            PlatformUnreachable: The stream could not be kept open within `MAX_RECONNECTS`.
        """
        if attempts <= MAX_RECONNECTS:
            return
        detail = f": {error}" if error is not None else " (the stream closed without any events)"
        msg = (
            f"the run stream for {external_id} could not be re-established after "
            f"{MAX_RECONNECTS} attempts{detail}"
        )
        raise PlatformUnreachable(msg) from error

    def _pause(self) -> None:
        """Wait out one safe-frontier lag, jittered so parallel tails do not reconnect in step."""
        time.sleep(RECONNECT_DELAY_S * (1.0 + random.uniform(0.0, RECONNECT_JITTER)))

    def _drain(self, external_id: str, after_pos: int) -> Iterator[EventRow]:
        """Page out whatever is left of a terminal run's log, which is served in full."""
        cursor = after_pos
        while True:
            page = self.list_events(external_id, after_pos=cursor)
            yield from page.events
            if page.last_pos <= cursor or not page.events:
                return
            cursor = page.last_pos

    def _stream_once(self, external_id: str, after_pos: int) -> Iterator[EventRow]:
        """One SSE connection's worth of frames, typed."""
        with self._client.stream_org_run_events(
            self._org_id, external_id, after_pos=after_pos, timeout_s=STREAM_TIMEOUT_S
        ) as frames:
            for frame in frames:
                try:
                    yield EventRow.model_validate(frame)
                except ValidationError:
                    # One malformed frame is not worth ending a twelve-hour follow over, and the
                    # cursor is unaffected: the next frame carries its own position.
                    log.warning(
                        "run stream for %s sent a frame this client does not understand; "
                        "skipping it",
                        external_id,
                    )

    def event_count(self, external_id: str) -> int:
        """Events the platform already holds for a run; 0 when it does not hold the run at all.

        The backfill guard's probe (`wmo.optimize.telemetry.backfill.ensure_backfillable`). It has
        to be a COUNT and not a cursor: the event page carries no `last_seq`, and a live
        run's `last_pos` trails real time by the server's safe frontier, so neither can
        answer "has anything landed for this run yet". An absent run answers 0, which
        reads as "new run, go ahead" rather than an error.
        """
        try:
            return self.get_run(external_id).event_count
        except PlatformError as error:
            if error.status_code == 404:
                return 0
            raise

    def _terminal(self, external_id: str) -> bool:
        """Whether the run is finished, which is what tells a closed stream from a dropped one.

        Load-bearing rather than cosmetic: the server closes a drained terminal run's stream on its
        own, so a closed connection is ambiguous and this is what disambiguates it. The one
        vocabulary lives in `schema.is_terminal_status`, whose membership test errs the right way:
        an unrecognized future status (paused, resuming) reads as NOT terminal, so a tail keeps
        following instead of silently ending a live run early.
        """
        return is_terminal_status(self.get_run(external_id).run.status)
