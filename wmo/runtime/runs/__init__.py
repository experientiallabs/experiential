"""Run transport and event contracts for the platform runs surface.

`schema` holds the wire contract and the seq-band allocator every writer shares,
`client` owns the transport that pushes it, and `reader` owns the org-scoped views
behind `wmo runs`. Optimization-specific event derivation lives in
`wmo.optimize.telemetry`.
"""

from wmo.runtime.runs.client import (
    ControlCommand,
    PushAck,
    PushRejected,
    PushUnavailable,
    RunsSink,
    RunsTransport,
    default_emitter_id,
    runs_sink,
)
from wmo.runtime.runs.schema import (
    CELL_BATCH_CAP,
    LEDGER_LINE,
    LOG_LINE,
    MAX_CELLS_PER_EVENT,
    MAX_EVENTS_PER_BATCH,
    RUN_LEVEL_BAND,
    RUN_META_SEQ,
    RUN_SEQ_BAND,
    TERMINAL_STATUSES,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    SeqBandOverrun,
    SeqBands,
    cell_band,
    grid_arm_external_id,
    is_terminal_status,
    ledger_walk_seq,
    pipeline_external_id,
)

__all__ = [
    "CELL_BATCH_CAP",
    "ControlCommand",
    "LEDGER_LINE",
    "LOG_LINE",
    "MAX_CELLS_PER_EVENT",
    "MAX_EVENTS_PER_BATCH",
    "RUN_LEVEL_BAND",
    "RUN_META_SEQ",
    "RUN_SEQ_BAND",
    "RunEvent",
    "RunEventType",
    "RunKind",
    "RunStatus",
    "PushAck",
    "PushRejected",
    "PushUnavailable",
    "RunsSink",
    "RunsTransport",
    "SeqBandOverrun",
    "TERMINAL_STATUSES",
    "SeqBands",
    "cell_band",
    "default_emitter_id",
    "grid_arm_external_id",
    "is_terminal_status",
    "ledger_walk_seq",
    "pipeline_external_id",
    "runs_sink",
]
