"""Run telemetry: the events wmo pushes to the platform's runs surface (D-RUNS v1).

`schema` holds the wire contract and the seq-band allocator every writer shares,
`client` the transport that pushes it, `backfill` the mapping from artifacts already
on disk to the same events live emission produces, and `reader` the org-scoped views
behind `wmo runs`.
"""

from wmo.runs.backfill import (
    BackfillRefused,
    cell_payload,
    ensure_backfillable,
    grid_arm_events,
    optimize_events,
)
from wmo.runs.client import (
    ControlCommand,
    PushAck,
    PushRejected,
    PushUnavailable,
    RunsSink,
    RunsTransport,
    default_emitter_id,
    runs_sink,
)
from wmo.runs.schema import (
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
    "BackfillRefused",
    "RunsSink",
    "RunsTransport",
    "SeqBandOverrun",
    "TERMINAL_STATUSES",
    "SeqBands",
    "cell_band",
    "cell_payload",
    "default_emitter_id",
    "ensure_backfillable",
    "grid_arm_events",
    "grid_arm_external_id",
    "is_terminal_status",
    "ledger_walk_seq",
    "optimize_events",
    "pipeline_external_id",
    "runs_sink",
]
