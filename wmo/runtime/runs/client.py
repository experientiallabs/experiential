"""The transport that pushes one organization's run telemetry to the platform.

`RunsSink` is the whole surface: push a run's events, probe what the platform already
holds, ack a control command. It is stateless per run (the run is named per call), so
one sink serves a pipeline and every arm of a grid. Queueing, banding, and the
never-block-the-run policy belong to the callers, because a live hook and a backfill
want opposite things from a queue: a hook bounds memory and drops the oldest rather
than stall a paid run, while a backfill pushes everything and fails loudly.

Failure handling splits along one line, because the halves need opposite treatment.
Transport trouble and 5xx are retried here with bounded backoff: the run's work is
expensive and a flaky network must not cost its telemetry. A 4xx is PERMANENT (a
retry loop would wedge a backfill and never succeed), so it surfaces as
`PushRejected` carrying the platform's own message. Sizes are pre-checked against the
platform's caps, so a caller normally never sees the 422 those would cause.

Timeouts are deliberately short (`TELEMETRY_TIMEOUT_SECONDS`): emitting sits on a
paid run's critical path, so a black-holing platform must cost seconds, not minutes.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from wmo.common.core.types import JsonObject
from wmo.runtime.platform.client import PlatformClient, PlatformError, PlatformUnreachable
from wmo.runtime.platform.credentials import load_credentials
from wmo.runtime.runs.schema import (
    MAX_CELLS_PER_EVENT,
    MAX_EVENTS_PER_BATCH,
    RunEvent,
)

logger = logging.getLogger(__name__)

# Retry budget for transport failures. Deliberately short: a live hook is on the
# run's critical path, and telemetry that blocks progress is worse than telemetry
# that arrives on the next flush.
PUSH_ATTEMPTS = 3
PUSH_BACKOFF_SECONDS = 0.5

# Request timeout for telemetry, deliberately far below the registry client's
# patient default. Emitting sits on a paid run's critical path: at the registry's
# 120s, one flush against a black-holing platform would stall the run for minutes
# across retries. Ten seconds is generous for a batch push and bounds the worst
# case to well under a minute.
TELEMETRY_TIMEOUT_SECONDS = 10.0

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

    def close(self) -> None:
        """Release the connection pool."""
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
        pending control for a run it knows, and refuses one for a run it does not (a
        new run's first batch must carry `run.meta`). That refusal is the "fresh run"
        answer, so it comes back as `last_seq` 0 rather than as an error.

        A caller uses the answer to continue numbering past what already landed;
        without it, a re-invocation restarts at the band floor and the platform
        discards every event as a replay.

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

    def close(self) -> None:
        """Release the transport's connection pool.

        `runs_sink` builds a `PlatformClient` per sink, and a long process opens one
        per run (an arm each, for a multi-arm grid), so nothing closing them leaves an
        idle pool per finished run for the life of the process.

        Part of the `RunsTransport` contract rather than sniffed for with `getattr`:
        a capability the sink depends on belongs in the Protocol, where a transport
        that cannot satisfy it fails at the type boundary instead of being silently
        skipped at runtime. The cost is one no-op on each test double, which is
        cheaper than a permanent conditional in the production path.
        """
        self._transport.close()

    def __enter__(self) -> RunsSink:
        """Return the sink for use in a `with` block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the transport on the way out, however the block ended."""
        self.close()

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
    debugging. It is never identity: the seq band is.
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
    if isinstance(cells, list):
        if len(cells) > MAX_CELLS_PER_EVENT:
            msg = f"{where} carries {len(cells)} cells; the limit is {MAX_CELLS_PER_EVENT}"
            raise PushRejected(msg, status_code=422)
        # The server checks a cell's `detail` and `usage` PER CELL, inside `cells`,
        # never at the top of the payload where a cell.batch has neither. Checking
        # only the top level let an oversized cell through to a 422 that drops the
        # WHOLE flush, including the ledger line and heartbeat riding with it.
        for cell in cells:
            if isinstance(cell, dict):
                _check_fields(cell, where=f"cell {cell.get('cell_key')} in {where}")
    _check_fields(event.payload, where=where)


def _check_fields(payload: JsonObject, *, where: str) -> None:
    """Reject an oversized document or sidecar field, naming it.

    Raises:
        PushRejected: 422-equivalent, before anything leaves the machine.
    """
    for field, limit in (
        *((name, MAX_DOCUMENT_BYTES) for name in _DOCUMENT_FIELDS),
        *((name, MAX_SIDECAR_BYTES) for name in _SIDECAR_FIELDS),
    ):
        value = payload.get(field)
        if value is None:
            continue
        field_size = _payload_size({field: value})
        if field_size > limit:
            msg = f"{field} in {where} is {field_size} bytes; the limit is {limit}"
            raise PushRejected(msg, status_code=422)


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
        PlatformClient(credentials.api_url, credentials.token, timeout=TELEMETRY_TIMEOUT_SECONDS),
        org_id=org,
        emitter_id=emitter_id or default_emitter_id(),
    )
