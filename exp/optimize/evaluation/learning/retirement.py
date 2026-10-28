"""Retired learning traces: traces a memory will never learn, recorded with why.

A trace whose request the gateway refuses deterministically would otherwise block a learning
pass, and so ``exp eval --audit``, forever: every ``--retry-incomplete`` resends the identical
request. Each such refusal is recorded here as evidence, keyed by the memory lineage, the trace,
its fidelity run and the digest of the refused request body. After ``RETIREMENT_FAILURES``
recorded refusals of the identical request, the pass retires the trace and continues with the
next one. Only codes registered in
:data:`exp.simulation.evaluation.workflow.DETERMINISTIC_REFUSALS` qualify; transport outages,
timeouts and unknown outcomes stay resumable and are never retired automatically. An operator
can also retire a named trace explicitly through the ``repair-learning`` continuation
(:mod:`exp.optimize.evaluation.learning.repair`).

A retirement never edits memory. It is a separate immutable record under the lineage lock, so
the memory lineage, the versions published before it and the identity of every learned trace
stay unchanged. A retired trace is never scored and never silently dropped: every learning
report and audit checkpoint counts it as not learned and names it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

from filelock import Timeout
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models.unknown_outcome import MAXIMUM_REDISPATCHES
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity, saved_fidelity_report
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.evaluation.workflow import deterministic_refusal
from exp.simulation.world_model.experience_memory import memory_head, memory_lock

# The original dispatch plus two resends: a refusal that repeats three times on the identical
# request is treated as deterministic. A resumed request may be re-dispatched at most
# MAXIMUM_REDISPATCHES times, so the bound must stay reachable.
RETIREMENT_FAILURES = 3
if not 1 < RETIREMENT_FAILURES <= 1 + MAXIMUM_REDISPATCHES:
    raise ValueError("retirement bound must be reachable within the allowed re-dispatches")
# At most this share of a lineage's traces (and at least one trace) is retired automatically.
# More identical refusals than that look systemic (a provider or model setting), not one bad
# trace, so the pass stops instead of quietly retiring the rest. Operator retirements are not
# counted.
AUTOMATIC_RETIREMENT_SHARE_PERCENT = 10
REFUSAL_TYPE = "trace-learning-refusal"
RETIREMENT_TYPE = "trace-learning-retirement"
Cause = Literal["deterministic-refusal", "operator"]


class SystemicRefusalError(ValueError):
    """Deterministic refusals exceed the automatic-retirement bound: the pass stops.

    The refusal that tripped the bound is recorded; its trace stays unretired and resumable.
    """


def automatic_retirement_limit(traces: int) -> int:
    """Automatic retirements allowed in a lineage over ``traces`` learning traces."""
    return max(1, traces * AUTOMATIC_RETIREMENT_SHARE_PERCENT // 100)


class RefusalObservation(ContractModel):
    """One recorded deterministic refusal of a learning trace's request.

    Attributes:
        lineage: Memory lineage that was learning the trace.
        trace_id: Refused trace.
        run: Fidelity registration whose request was refused.
        request_sha256: Digest of the exact refused request body.
        status_code: HTTP status of the refusal.
        error_code: Registered refusal code.
        ordinal: Position among refusals of this identical request, from 1.
        observed_at: When the refusal was recorded.
    """

    lineage: str
    trace_id: str
    run: str
    request_sha256: Sha256
    status_code: int
    error_code: str
    ordinal: int = Field(ge=1)
    observed_at: datetime


class RetiredTrace(ContractModel):
    """A retired trace as an audit receipt records it.

    Attributes:
        trace_id: Retired trace.
        cause: ``deterministic-refusal`` (automatic) or ``operator``.
        reason: Refusal code with its count, or the operator's explanation.
        record: The immutable retirement record with its full evidence.
    """

    trace_id: str
    cause: Cause
    reason: str
    record: ArtifactInput


class TraceRetirement(ContractModel):
    """One trace a memory lineage will never learn, and why.

    Attributes:
        lineage: Memory lineage the trace is retired from.
        trace_id: Retired trace.
        cause: ``deterministic-refusal`` after the registered bound, or ``operator``.
        reason: Refusal code with its count, or the operator's explanation.
        run: Fidelity registration that stopped, or None when the trace never ran.
        refusals: Recorded refusals of the identical request, in order; may be empty for an
            operator retirement.
        head_version: Memory version when the trace was retired; 0 before any publication.
    """

    lineage: str
    trace_id: str
    cause: Cause
    reason: str = Field(min_length=1, max_length=4000)
    run: str | None = None
    refusals: tuple[RefusalObservation, ...] = ()
    head_version: int = Field(ge=0)


def record_refusal(
    project: ProjectStore,
    *,
    lineage: str,
    trace_id: str,
    run: str,
    error: ProviderTransportError,
    now: datetime,
    code_revision: str,
) -> tuple[RefusalObservation, ...]:
    """Record one deterministic refusal and return every refusal of that identical request.

    Args:
        project: Project owning the memory lineage.
        lineage: Memory lineage learning the trace.
        trace_id: Refused trace.
        run: Fidelity registration whose request was refused.
        error: The refusal, with its status, registered code and request digest.
        now: Aware observation time.
        code_revision: Recording implementation identity.

    Returns:
        Recorded refusals of the identical request in this run, oldest first, this one last.
    """
    if error.status_code is None or error.error_code is None or error.request_sha256 is None:
        raise ValueError("only an identified provider refusal can be recorded")
    with memory_lock(project.artifacts, lineage):
        prior = refusals(project, lineage, trace_id, run, error.request_sha256)
        observation = RefusalObservation(
            lineage=lineage,
            trace_id=trace_id,
            run=run,
            request_sha256=error.request_sha256,
            status_code=error.status_code,
            error_code=error.error_code,
            ordinal=len(prior) + 1,
            observed_at=now,
        )
        project.artifacts.write_json(
            artifact_id=_refusal_id(lineage, trace_id, run, error.request_sha256, len(prior) + 1),
            artifact_type=REFUSAL_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=now, code_revision=code_revision
            ),
            files={"refusal.json": observation.model_dump(mode="json")},
        )
    return (*prior, observation)


def refusals(
    project: ProjectStore, lineage: str, trace_id: str, run: str, request_sha256: str
) -> tuple[RefusalObservation, ...]:
    """Recorded refusals of one identical request in one run, oldest first."""
    rows: list[RefusalObservation] = []
    while True:
        identity = _refusal_id(lineage, trace_id, run, request_sha256, len(rows) + 1)
        if not project.artifacts.exists(identity):
            return tuple(rows)
        rows.append(
            RefusalObservation.model_validate_json(
                project.artifacts.read_bytes(identity, "refusal.json")
            )
        )


def trace_refusals(
    project: ProjectStore, lineage: str, trace_id: str
) -> tuple[RefusalObservation, ...]:
    """Every recorded refusal of one trace's requests in this lineage, oldest first."""
    rows = [
        RefusalObservation.model_validate_json(
            project.artifacts.read_bytes(identity, "refusal.json")
        )
        for identity in project.artifacts.list_ids(artifact_type=REFUSAL_TYPE)
    ]
    return tuple(
        sorted(
            (row for row in rows if row.lineage == lineage and row.trace_id == trace_id),
            key=lambda row: (row.observed_at, row.ordinal),
        )
    )


def retire(
    project: ProjectStore,
    retirement: TraceRetirement,
    *,
    now: datetime,
    code_revision: str,
    unconsumed: Sha256 | None = None,
    automatic_limit: int | None = None,
) -> ArtifactInput:
    """Record one retirement; an identical retirement replays, a different one is refused.

    Args:
        project: Project owning the memory lineage.
        retirement: What is retired and why.
        now: Aware recording time.
        code_revision: Recording implementation identity.
        unconsumed: Consumed-trajectory digest of the trace, checked against the memory head
            under the lineage lock so a trace is never both learned and retired.
        automatic_limit: For an automatic retirement, how many the lineage may hold; checked
            under the lineage lock.

    Returns:
        Reference to the immutable retirement record.

    Raises:
        ValueError: The trace is already learned, or already retired for a different reason.
        SystemicRefusalError: An automatic retirement would exceed ``automatic_limit``.
    """
    with memory_lock(project.artifacts, retirement.lineage):
        if automatic_limit is not None and not project.artifacts.exists(
            retirement_id(retirement.lineage, retirement.trace_id)
        ):
            earlier = _automatic(project, retirement.lineage)
            if len(earlier) >= automatic_limit:
                traces = ", ".join([*(row.trace_id for row in earlier), retirement.trace_id])
                raise SystemicRefusalError(
                    f"refusals look systemic: trace {retirement.trace_id} hit "
                    f"{retirement.reason}, and {len(earlier)} automatic retirement(s) "
                    f"already reached the bound of {automatic_limit} for memory "
                    f"{retirement.lineage} (affected traces: {traces}). The trace was not "
                    "retired and stays resumable. Check the provider or model settings (for "
                    "example the output allowance or reasoning effort) and resume with "
                    "--retry-incomplete, or retire specific traces explicitly with the "
                    "repair-learning retire-trace resolution"
                )
        head = memory_head(project.artifacts, retirement.lineage)
        if unconsumed is not None and head is not None and unconsumed in head.consumed:
            raise ValueError(
                f"trace {retirement.trace_id} is already learned into memory "
                f"{retirement.lineage}; a learned trace cannot be retired"
            )
        identity = retirement_id(retirement.lineage, retirement.trace_id)
        if project.artifacts.exists(identity):
            saved = _read(project, identity)
            if saved != retirement:
                raise ValueError(
                    f"trace {retirement.trace_id} is already retired from memory "
                    f"{retirement.lineage} ({saved.cause}: {saved.reason})"
                )
        else:
            references = (
                artifact_input(
                    project.artifacts.read(
                        _refusal_id(
                            row.lineage, row.trace_id, row.run, row.request_sha256, row.ordinal
                        )
                    ).manifest
                )
                for row in retirement.refusals
            )
            inputs = tuple(sorted(set(references), key=lambda ref: ref.artifact_id))
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type=RETIREMENT_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=now, code_revision=code_revision, inputs=inputs
                ),
                files={"retirement.json": retirement.model_dump(mode="json")},
            )
        return artifact_input(project.artifacts.read(identity).manifest)


def settle_refusal(
    project: ProjectStore,
    lineage: str,
    trace_id: str,
    prepared: PreparedFidelity,
    error: BaseException,
    *,
    traces: int,
    now: datetime,
    code_revision: str,
    retry_incomplete: bool,
) -> bool | None:
    """Record a deterministic refusal of an unsaved learning run; retire its trace at the bound.

    Args:
        project: Project owning the run and the memory lineage.
        lineage: Memory lineage the run learns into.
        trace_id: Trace the run learns.
        prepared: The learning run that stopped.
        error: What stopped it.
        traces: Learning traces in the lineage's scope, which set the automatic bound.
        now: Aware recording time.
        code_revision: Recording implementation identity.
        retry_incomplete: The pass may resend the identical request in place.

    Returns:
        True when the trace is now retired, False when the identical request may be resent in
        place, and None when the stop must propagate unchanged: any other stop, a run that
        already saved its report, or a refusal below the bound without a resumption.

    Raises:
        SystemicRefusalError: Retiring this trace would exceed
            :func:`automatic_retirement_limit`; nothing more is retired.
    """
    refusal = deterministic_refusal(error)
    if refusal is None or saved_fidelity_report(project, prepared) is not None:
        return None
    rows = record_refusal(
        project,
        lineage=lineage,
        trace_id=trace_id,
        run=prepared.id,
        error=refusal,
        now=now,
        code_revision=code_revision,
    )
    if len(rows) < RETIREMENT_FAILURES:
        return False if retry_incomplete else None
    head = memory_head(project.artifacts, lineage)
    retire(
        project,
        TraceRetirement(
            lineage=lineage,
            trace_id=trace_id,
            cause="deterministic-refusal",
            reason=f"{refusal.error_code} (HTTP {refusal.status_code}) on {len(rows)} "
            "identical requests",
            run=prepared.id,
            refusals=rows,
            head_version=0 if head is None else head.version,
        ),
        now=now,
        code_revision=code_revision,
        automatic_limit=automatic_retirement_limit(traces),
    )
    return True


def retirement_id(lineage: str, trace_id: str) -> str:
    """Identity of a trace's retirement record in one memory lineage."""
    return stable_id(RETIREMENT_TYPE, {"lineage": lineage, "trace": trace_id})


@contextmanager
def trace_held(project: ProjectStore, lineage: str, trace_id: str) -> Iterator[None]:
    """Hold a trace while it is learned or retired, so a retirement never interleaves with learning.

    A learning pass holds it from its retirement check until the trace's last attempt is saved
    and published; an operator retirement holds it from its checks until its record is written.
    Neither waits for the other: whichever comes second is refused and can be retried.

    Raises:
        ValueError: Another process is learning or retiring the trace.
    """
    lock = CoordinationLock(
        project.artifacts.project_directory / (retirement_id(lineage, trace_id) + ".lock"),
        timeout=0,
    )
    try:
        held = lock.acquire()
    except Timeout as error:
        raise ValueError(
            f"trace {trace_id} is being learned or retired by another process in memory "
            f"{lineage}; run the command again once it stops"
        ) from error
    with held:
        yield


def retirement_of(project: ProjectStore, lineage: str, trace_id: str) -> TraceRetirement | None:
    """The trace's retirement from this lineage, or None when it is not retired."""
    identity = retirement_id(lineage, trace_id)
    return _read(project, identity) if project.artifacts.exists(identity) else None


def retired_traces(
    project: ProjectStore, lineage: str, trace_ids: tuple[str, ...]
) -> tuple[RetiredTrace, ...]:
    """The retired traces among ``trace_ids`` in their order, as an audit receipt records them.

    Args:
        project: Project owning the memory lineage.
        lineage: Memory lineage.
        trace_ids: Traces in the lineage's scope, in learning order.

    Returns:
        One row per retired trace.
    """
    rows = []
    for trace_id in trace_ids:
        identity = retirement_id(lineage, trace_id)
        if project.artifacts.exists(identity):
            saved = _read(project, identity)
            rows.append(
                RetiredTrace(
                    trace_id=trace_id,
                    cause=saved.cause,
                    reason=saved.reason,
                    record=artifact_input(project.artifacts.read(identity).manifest),
                )
            )
    return tuple(rows)


def _automatic(project: ProjectStore, lineage: str) -> tuple[TraceRetirement, ...]:
    """Automatic retirements recorded in one lineage, in identity order."""
    rows = (
        _read(project, identity)
        for identity in project.artifacts.list_ids(artifact_type=RETIREMENT_TYPE)
    )
    return tuple(
        row for row in rows if row.lineage == lineage and row.cause == "deterministic-refusal"
    )


def _read(project: ProjectStore, identity: str) -> TraceRetirement:
    """Read one retirement record."""
    return TraceRetirement.model_validate_json(
        project.artifacts.read_bytes(identity, "retirement.json")
    )


def _refusal_id(lineage: str, trace_id: str, run: str, request_sha256: str, ordinal: int) -> str:
    """Identity of the ``ordinal``-th refusal of one identical request in one run."""
    return stable_id(
        REFUSAL_TYPE,
        {
            "lineage": lineage,
            "trace": trace_id,
            "run": run,
            "request": request_sha256,
            "ordinal": ordinal,
        },
    )
