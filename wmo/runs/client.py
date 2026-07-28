"""The batched emitter that pushes one run's telemetry to the platform.

A `RunsEmitter` owns ONE run and its seq bands. Callers hand it events with the band
they are writing into — `RUN_LEVEL_BAND` for the run-level walk, `cell_band(chunk)`
for a chunk's cells — and it queues them, pushes in bounded batches, and answers with
the platform's acknowledgement. One emitter per run rather than one per band is
deliberate: a single `SeqBands` is what makes two writers' ranges provably disjoint,
and one flush can carry several bands in a single request.

Failure handling splits along one line, because the two halves need opposite
treatment. Transport trouble and 5xx are retried here with bounded backoff: the run's
work is expensive and a flaky network must not lose telemetry. A 4xx is PERMANENT —
retrying it forever would wedge a backfill and never succeed — so it surfaces as
`PushRejected` with the platform's own message. The emitter also pre-checks the size
caps that would produce a 422, so a caller normally never sees one.

Emitting is best-effort by design: telemetry must never take down the run it
describes. `flush` raises so a caller can decide, but `close` and the context manager
swallow a final failure after logging it.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from wmo.core.types import JsonObject
from wmo.platform.client import PlatformClient, PlatformError, PlatformUnreachable
from wmo.platform.credentials import load_credentials
from wmo.runs.schema import (
    MAX_CELLS_PER_EVENT,
    MAX_EVENTS_PER_BATCH,
    RUN_LEVEL_BAND,
    RUN_SEQ_BAND,
    RunEvent,
    SeqBands,
)

logger = logging.getLogger(__name__)

# Retry budget for transport failures. Deliberately short: a live hook is on the
# run's critical path, and telemetry that blocks progress is worse than telemetry
# that arrives on the next flush.
PUSH_ATTEMPTS = 3
PUSH_BACKOFF_SECONDS = 0.5

# Size caps mirrored from the platform, which refuses anything larger. Checked here
# so an oversized field is a loud local error naming the field rather than a 422
# after the events are already durable.
MAX_EVENT_PAYLOAD_BYTES = 960 * 1024
MAX_DOCUMENT_BYTES = 240 * 1024
MAX_SIDECAR_BYTES = 56 * 1024

# Payload fields carrying documents and sidecars, and the cap each one gets.
_DOCUMENT_FIELDS = ("config", "artifact", "detail")
_SIDECAR_FIELDS = ("fingerprint", "usage", "args")


class PushRejected(RuntimeError):
    """The platform refused a batch permanently; retrying cannot help.

    411 (no declared length), 413 (too large or too many events), and 422 (a payload
    field over its cap, or a malformed one) all land here. A caller logs and drops
    the batch; it must not loop.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        """Initialize with the platform's message and status."""
        super().__init__(message)
        self.status_code = status_code


class PushUnavailable(RuntimeError):
    """The platform could not be reached, or answered 5xx, after every retry."""


class ControlCommand(BaseModel):
    """One admin-issued command riding back on a push response."""

    model_config = ConfigDict(frozen=True)

    id: str
    command: str
    args: JsonObject = Field(default_factory=dict)


class PushAck(BaseModel):
    """What the platform answers to an accepted push."""

    model_config = ConfigDict(frozen=True)

    # NEWLY accepted seqs in this push. Compared against how many events the
    # emitter believed were new, this is the seq-collision tripwire.
    accepted: int
    # The run's high-water seq, so a restarted process resumes without replaying
    # its whole file. A resume mark, never a read cursor.
    last_seq: int
    control: tuple[ControlCommand, ...] = ()


class RunsTransport(Protocol):
    """The two platform calls a sink makes; `PlatformClient` satisfies it."""

    def push_run_events(
        self,
        org_id: str,
        external_id: str,
        *,
        emitter_id: str,
        events: Sequence[JsonObject],
    ) -> JsonObject:
        """Push one batch and return the raw acknowledgement."""
        ...

    def ack_run_control(
        self,
        org_id: str,
        external_id: str,
        control_id: str,
        *,
        status: str,
        note: str | None = None,
    ) -> JsonObject:
        """Answer one control command."""
        ...


class RunsSink:
    """The HTTP half of emitting: push events for a run, ack its control.

    The single transport seam both emitting paths share. A backfill wraps it in
    `RunsEmitter` for queueing; a live hook drives it directly, because a hook's
    queue policy is its own (bounded memory, drop oldest, never block the run)
    and differs from a backfill's (push everything, fail loudly).

    Stateless per run: the run is named per call, so one sink serves several.
    """

    def __init__(self, transport: RunsTransport, *, org_id: str, emitter_id: str) -> None:
        """Initialize the sink.

        Args:
            transport: Platform calls (a `PlatformClient`, or a fake in tests).
            org_id: Organization the runs belong to.
            emitter_id: This process's diagnostic id, sent with every push.
        """
        self._transport = transport
        self._org_id = org_id
        self._emitter_id = emitter_id

    @property
    def emitter_id(self) -> str:
        """The diagnostic id this sink stamps on every push."""
        return self._emitter_id

    def push(self, external_id: str, events: Sequence[RunEvent]) -> PushAck:
        """Push events for one run, splitting at the platform's per-request cap.

        Args:
            external_id: The run's stable name.
            events: Events to push, seqs already assigned.

        Returns:
            One acknowledgement for the whole call: `accepted` summed across the
            requests it took, `last_seq` the highest reported, `control` from the
            final response (control is run state, not per-request state).

        Raises:
            PushRejected: On a permanent 4xx; the batch is dropped.
            PushUnavailable: When the platform stayed unreachable.
        """
        accepted = 0
        last_seq = 0
        control: tuple[ControlCommand, ...] = ()
        for start in range(0, len(events), MAX_EVENTS_PER_BATCH):
            batch = events[start : start + MAX_EVENTS_PER_BATCH]
            for event in batch:
                _check_size(event)
            ack = self._push_once(external_id, batch)
            accepted += ack.accepted
            last_seq = max(last_seq, ack.last_seq)
            control = ack.control
        return PushAck(accepted=accepted, last_seq=last_seq, control=control)

    def probe(self, external_id: str) -> PushAck:
        """Ask the platform what it already holds for a run, without writing.

        A zero-event push is the probe: the ingest route answers `last_seq` and any
        pending control for a run it knows, and refuses one it does not (a new run's
        first batch must carry `run.meta`). That refusal is the "fresh run" answer,
        so it comes back as `last_seq` 0 rather than as an error.

        Args:
            external_id: The run's stable name.

        Returns:
            The run's resume mark and pending control; zeros for an unknown run.

        Raises:
            PushUnavailable: When the platform could not be reached.
        """
        try:
            return self._push_once(external_id, ())
        except PushRejected as refused:
            if refused.status_code == 422:
                return PushAck(accepted=0, last_seq=0)
            raise

    def ack(
        self, external_id: str, control_id: str, *, status: str, note: str | None = None
    ) -> None:
        """Answer one control command.

        Args:
            external_id: The run the command belongs to.
            control_id: The command's id, as it arrived on a push response.
            status: `acked`, `done`, or `rejected`.
            note: Why, for a rejection the operator will read.
        """
        self._transport.ack_run_control(
            self._org_id, external_id, control_id, status=status, note=note
        )

    def _push_once(self, external_id: str, batch: Sequence[RunEvent]) -> PushAck:
        """Push one request's worth, retrying transport failures only."""
        wire = [event.wire() for event in batch]
        for attempt in range(1, PUSH_ATTEMPTS + 1):
            try:
                raw = self._transport.push_run_events(
                    self._org_id, external_id, emitter_id=self._emitter_id, events=wire
                )
            except PlatformUnreachable as error:
                if attempt == PUSH_ATTEMPTS:
                    msg = f"run {external_id}: platform unreachable after {attempt} attempts"
                    raise PushUnavailable(msg) from error
                time.sleep(PUSH_BACKOFF_SECONDS * attempt)
            except PlatformError as error:
                status = error.status_code or 0
                if status >= 500:
                    if attempt == PUSH_ATTEMPTS:
                        msg = f"run {external_id}: platform failed with {status}"
                        raise PushUnavailable(msg) from error
                    time.sleep(PUSH_BACKOFF_SECONDS * attempt)
                    continue
                # 4xx: the batch is wrong, not the moment. Retrying cannot fix it.
                raise PushRejected(str(error), status_code=status) from error
            else:
                return PushAck.model_validate(raw)
        msg = f"run {external_id}: push exhausted its retries"
        raise PushUnavailable(msg)


def default_emitter_id() -> str:
    """Build this process's diagnostic id: `hostname:pid:uuid8`.

    Several processes may feed one run, so this identifies the feeder for
    debugging. It is never identity — the seq band is.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _payload_size(payload: JsonObject) -> int:
    """Measure a payload the way the platform's caps measure it."""
    return len(json.dumps(payload, ensure_ascii=False).encode())


def _check_size(event: RunEvent) -> None:
    """Reject an event the platform would refuse, naming the offending field.

    Raises:
        PushRejected: 422-equivalent, before anything leaves the machine.
    """
    where = f"the {event.type} event at seq {event.seq}"
    size = _payload_size(event.payload)
    if size > MAX_EVENT_PAYLOAD_BYTES:
        msg = f"payload in {where} is {size} bytes; the limit is {MAX_EVENT_PAYLOAD_BYTES}"
        raise PushRejected(msg, status_code=422)
    cells = event.payload.get("cells")
    if isinstance(cells, list) and len(cells) > MAX_CELLS_PER_EVENT:
        msg = f"{where} carries {len(cells)} cells; the limit is {MAX_CELLS_PER_EVENT}"
        raise PushRejected(msg, status_code=422)
    for field, limit in (
        *((name, MAX_DOCUMENT_BYTES) for name in _DOCUMENT_FIELDS),
        *((name, MAX_SIDECAR_BYTES) for name in _SIDECAR_FIELDS),
    ):
        value = event.payload.get(field)
        if value is None:
            continue
        field_size = _payload_size({field: value})
        if field_size > limit:
            msg = f"{field} in {where} is {field_size} bytes; the limit is {limit}"
            raise PushRejected(msg, status_code=422)


@dataclass(frozen=True)
class _Queued:
    """One queued event, and whether its seq was derived rather than allocated.

    A derived seq (an artifact position) is expected to collide on a resume — the
    fact already landed under that exact seq, which is the whole point. An allocated
    seq is not: a shortfall there means another writer is in this band. The
    distinction is what keeps the collision tripwire honest instead of crying wolf
    on every re-invocation.
    """

    event: RunEvent
    derived: bool


class RunsEmitter:
    """Queues one run's events and pushes them in bounded batches."""

    def __init__(
        self,
        sink: RunsSink,
        *,
        external_id: str,
        band_width: int = RUN_SEQ_BAND,
    ) -> None:
        """Initialize the emitter for one run.

        Args:
            sink: The transport seam to push through.
            external_id: The run's stable name.
            band_width: Seq band width; only override in tests.
        """
        self._sink = sink
        self._external_id = external_id
        self._bands = SeqBands(band_width=band_width)
        self._queue: deque[_Queued] = deque()
        self._last_ack: PushAck | None = None

    @property
    def external_id(self) -> str:
        """The run this emitter feeds."""
        return self._external_id

    @property
    def last_ack(self) -> PushAck | None:
        """The most recent successful acknowledgement, if any."""
        return self._last_ack

    @property
    def pending_control(self) -> tuple[ControlCommand, ...]:
        """Commands the platform handed back and nobody has acked yet."""
        return self._last_ack.control if self._last_ack is not None else ()

    def emit(
        self, band: int, event_type: str, ts: str, payload: BaseModel | JsonObject
    ) -> RunEvent:
        """Queue one event, taking its seq from the given band.

        Args:
            band: The writer's band (`RUN_LEVEL_BAND`, or `cell_band(chunk)`).
            event_type: Wire type; unknown types are legal and project nothing.
            ts: The artifact's or the moment's own timestamp, as a string.
            payload: A typed payload model, or a dict passed through verbatim.

        Returns:
            The queued event, seq assigned.

        Raises:
            PushRejected: If the payload exceeds a platform cap.
            SeqBandOverrun: If the band is exhausted.
        """
        return self._queue_event(self._bands.take(band), event_type, ts, payload, derived=False)

    def emit_from_top(
        self, band: int, event_type: str, ts: str, payload: BaseModel | JsonObject
    ) -> RunEvent:
        """Queue a live-only event, numbered DOWN from the top of its band.

        For facts with no artifact position — heartbeats, a live `run.status`. See
        `SeqBands.take_from_top`.
        """
        return self._queue_event(
            self._bands.take_from_top(band), event_type, ts, payload, derived=False
        )

    def emit_at(
        self, seq: int, event_type: str, ts: str, payload: BaseModel | JsonObject
    ) -> RunEvent:
        """Queue an event at a seq DERIVED from an artifact position.

        `run.meta` at `RUN_META_SEQ`, the Nth ledger line at `ledger_walk_seq(n)`.
        Marked derived, so a resume that re-emits it reports a converged no-op
        rather than tripping the collision alarm.

        Args:
            seq: The derived seq.
            event_type: Wire type.
            ts: The artifact's own timestamp.
            payload: A typed payload model, or a dict passed through verbatim.

        Returns:
            The queued event.
        """
        return self._queue_event(seq, event_type, ts, payload, derived=True)

    def resume(self) -> int:
        """Probe the run and continue band-0 numbering past what already landed.

        Without this, a re-invocation of a run that already holds events restarts at
        seq 1, the platform discards every event as a replay, and the run looks
        healthy while its new telemetry vanishes. Band 0 is only rebased when the
        resume mark sits inside it: a grid arm whose chunk bands pushed `last_seq`
        high derives its band-0 seqs from the ledger file instead, and rebasing
        would skip past them.

        Returns:
            The run's resume mark; 0 when the platform holds no such run.
        """
        ack = self._sink.probe(self._external_id)
        self._last_ack = ack
        if 0 < ack.last_seq < RUN_SEQ_BAND:
            self._bands.resume_at(RUN_LEVEL_BAND, ack.last_seq + 1)
        return ack.last_seq

    def _queue_event(
        self,
        seq: int,
        event_type: str,
        ts: str,
        payload: BaseModel | JsonObject,
        *,
        derived: bool,
    ) -> RunEvent:
        """Build, size-check, and queue one event."""
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        event = RunEvent(
            external_id=self._external_id,
            seq=seq,
            ts=ts,
            type=event_type,
            payload=body,
        )
        _check_size(event)
        self._queue.append(_Queued(event=event, derived=derived))
        return event

    def flush(self) -> PushAck | None:
        """Push every queued event, in batches the platform accepts.

        Returns:
            The last acknowledgement, or None when nothing was queued.

        Raises:
            PushRejected: On a permanent refusal; the batch is dropped.
            PushUnavailable: When the platform stayed unreachable.
        """
        if not self._queue:
            return None
        queued = list(self._queue)
        self._queue.clear()
        ack = self._sink.push(self._external_id, [item.event for item in queued])
        self._last_ack = ack
        self._report_acceptance(queued, ack)
        return ack

    def _report_acceptance(self, queued: Sequence[_Queued], ack: PushAck) -> None:
        """Log a shortfall as convergence or as collision, whichever it actually is.

        Derived seqs are artifact positions, so a resume re-deriving them SHOULD find
        them already held — that is convergence working, and reporting it as an error
        would train operators to ignore the one message that matters. A shortfall
        below the number of freshly ALLOCATED events is the real alarm: those seqs
        came from this process's own allocator, so nothing but another writer in the
        same band can already hold them.
        """
        fresh = sum(1 for item in queued if not item.derived)
        replayed = len(queued) - fresh
        if ack.accepted < fresh:
            logger.error(
                "run %s: pushed %d freshly numbered events but only %d were accepted — "
                "another writer is using this seq band, and telemetry is being silently "
                "discarded",
                self._external_id,
                fresh,
                ack.accepted,
            )
        elif replayed and ack.accepted < len(queued):
            logger.info(
                "run %s: %d of %d events were already held (re-derived artifact positions "
                "converging on a resume, as intended)",
                self._external_id,
                len(queued) - ack.accepted,
                len(queued),
            )

    def ack(self, control: ControlCommand, *, status: str, note: str | None = None) -> None:
        """Answer one control command.

        Args:
            control: The command as it arrived on a push response.
            status: `acked`, `done`, or `rejected`.
            note: Why, for a rejection the operator will read.
        """
        self._sink.ack(self._external_id, control.id, status=status, note=note)

    def close(self) -> None:
        """Flush what is queued, swallowing a final failure after logging it.

        Telemetry must never be the reason a finished run reports an error, so the
        last flush is best-effort.
        """
        try:
            self.flush()
        except (PushRejected, PushUnavailable):
            logger.warning(
                "run %s: %d events were not delivered", self._external_id, len(self._queue)
            )

    def __enter__(self) -> RunsEmitter:
        """Return the emitter for use in a `with` block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Flush on the way out, best-effort."""
        self.close()


def runs_sink(org_id: str | None = None, *, emitter_id: str | None = None) -> RunsSink | None:
    """Build a sink from this machine's saved credential, or None if it has none.

    Never raises: an unauthenticated machine is the ordinary case for a local run,
    and telemetry is an extra rather than a requirement. Callers log one line and
    carry on.

    Args:
        org_id: Organization to push to; defaults to the saved default org.
        emitter_id: Override this process's diagnostic id.

    Returns:
        A ready sink, or None when no credential or organization is configured.
    """
    credentials = load_credentials()
    if not credentials.is_complete() or credentials.api_url is None or credentials.token is None:
        logger.info("platform credential not found; run telemetry is off")
        return None
    org = org_id or credentials.default_org
    if not org:
        logger.info("no default organization saved; run telemetry is off")
        return None
    return RunsSink(
        PlatformClient(credentials.api_url, credentials.token),
        org_id=org,
        emitter_id=emitter_id or default_emitter_id(),
    )


def open_emitter(
    external_id: str, *, org_id: str | None = None, emitter_id: str | None = None
) -> RunsEmitter | None:
    """Open a queueing emitter for one run, or None when this machine cannot push.

    Args:
        external_id: The run's stable name.
        org_id: Organization to push to; defaults to the saved default org.
        emitter_id: Override this process's diagnostic id.

    Returns:
        A ready emitter, or None when no credential or organization is configured.
    """
    sink = runs_sink(org_id, emitter_id=emitter_id)
    if sink is None:
        return None
    return RunsEmitter(sink, external_id=external_id)
