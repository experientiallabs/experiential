"""The D-RUNS v1 wire contract and the seq-band allocator every writer shares.

One wmo run (a grid arm, an optimize pipeline, a build, a distill job, a research
probe) has a platform counterpart row identified by `(org, external_id)`. Progress
travels as a per-run sequence of events; the platform stores them keyed on
`(run_id, seq)` and discards a seq it already holds, which is what makes a replayed
backfill free.

That idempotency key is also a hazard, and the reason `SeqBands` exists. Several
processes feed one run (a chunked grid arm), and the platform cannot tell one
writer's fresh event from another writer's replay: if two processes number from the
same base, the loser's events are silently discarded and the run looks healthy with
a quietly short cell matrix. So every writer owns a disjoint band, and the band
assignment is fixed (not merely disjoint) so that a backfill and the live hooks
derive the SAME seq for the same fact and converge instead of double-counting:

    band 0            the run-level walk (run.meta, ledger.line, the final
                      heartbeat, run.status) plus in-flight partial-file cells
    band k + 1        completed chunk k's cell events, first seq
                      (k + 1) * RUN_SEQ_BAND + 1, cells in chunk-file line order

Pipeline runs use band 0 alone.

Live emission and backfill converge by deriving the SAME seq for the same fact,
rather than by partitioning who may write where. Three cases, and which one applies
depends only on whether the fact has a position in an artifact:

- Facts with a deterministic artifact position take that position, from either path.
  `run.meta` is band-0 seq 1. The Nth `ledger.line` (1-based, among the arm's own
  ledger lines) is band-0 seq `1 + N`, which is exactly where backfill's walk puts
  it, and any process can compute it after appending, because its line has a fixed
  file index. Cells take their chunk's band, which is a function of the chunk alone.
  All of these are idempotent across processes and across replays: re-deriving one
  produces the same seq, so the platform discards it as the duplicate it is.
- Live-only facts (heartbeats, a live `run.status`) have no artifact position, so
  they take DESCENDING seqs from the top of the writing process's first owned chunk
  band. Descending from the top keeps them clear of the ascending walk below, and
  staying inside an owned band keeps two processes from colliding. A resume may reuse
  a seq for a newer snapshot; that is harmless, because projection is guarded by
  event clock rather than by log position, so the newer snapshot either wins on ts or
  was already superseded.
- A resumed run whose band-0 walk is not artifact-derived (a pipeline re-invocation)
  rebases its band-0 numbering to `last_seq + 1`, learned by probing the run. Without
  that, a re-run restarts at seq 1 and every event it emits is discarded as a replay.

The one remaining asymmetry is why `wmo runs backfill` refuses a run that already
holds events unless forced: a backfill re-walks the artifact positions, which is
idempotent, but it ALSO re-derives the final heartbeat and status, and those are
live-only facts whose seqs it cannot know, so it would add a second copy under
band-0 seqs and double-count the panel's spend curve.

Timestamps are carried verbatim as the strings the artifacts hold. Two reasons, both
load-bearing: the platform's projection guards compare event clocks and a
re-serialized timestamp that differs by a digit is a different clock, and a backfill
must reproduce `expected-events.jsonl` byte for byte. Never invent a clock here: no
`now()`, no reformatting.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wmo.config import ARTIFACT_DIR
from wmo.core.types import JsonObject

# Width of one writer's seq band. Defined ONCE, here: the backfill command and
# every live hook import it, because two writers disagreeing about the width
# produces overlapping bands and silently discarded events.
RUN_SEQ_BAND = 100_000

# The band carrying the run-level walk. Completed chunk k takes band k + 1.
RUN_LEVEL_BAND = 0

# Cells per `cell.batch` event. The platform refuses an event carrying more than
# MAX_CELLS_PER_EVENT, so this stays an order of magnitude below it: the batch size
# is a shape choice, not a limit to sit against.
CELL_BATCH_CAP = 100

# Platform-side limits, mirrored so the emitter chunks correctly instead of
# discovering them as 4xx. Both are permanent refusals, never retryable.
MAX_EVENTS_PER_BATCH = 500
MAX_CELLS_PER_EVENT = 1000


class RunKind(StrEnum):
    """What kind of wmo work a run records; the platform's vocabulary is closed."""

    GRID_ARM = "grid_arm"
    PIPELINE = "pipeline"
    BUILD = "build"
    DISTILL = "distill"
    RESEARCH = "research"


class RunStatus(StrEnum):
    """A run's lifecycle, mirroring the platform's closed vocabulary."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# The statuses that mean no writer is left. Named as a set rather than inverted
# from `running`, for the reason `is_terminal_status` explains.
TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.STOPPED.value}
)


def is_terminal_status(status: str) -> bool:
    """Whether a run has no writer left, so nothing more will arrive for it.

    Load-bearing rather than cosmetic: the platform closes a drained terminal run's
    SSE tail on its own, which makes a closed connection ambiguous (finished, or
    dropped mid-run), and this is what a reader disambiguates it with.

    Deliberately a membership test against the KNOWN terminal statuses, not
    `status != "running"`. The two agree today and diverge the moment the platform
    adds a status that means still-writing (paused, resuming): inverting `running`
    would call that terminal, and a tail would exit early and silently drop the rest
    of the run. An unrecognized status therefore reads as NOT terminal, which errs
    toward a stream that stays open too long, which is recoverable, over one that
    ends early, which is not.

    Args:
        status: A run's status as the platform reported it.

    Returns:
        True when the status is a known terminal one.
    """
    return status in TERMINAL_STATUSES


class RunEventType(StrEnum):
    """Event types the platform projects onto its derived tables.

    The vocabulary is open: any other type is stored and streamed but projects
    nothing, so `ledger.line` and `log.line` are deliberately absent. They are
    log-only, and the spend curve reads them straight off the event log.
    """

    RUN_META = "run.meta"
    RUN_STATUS = "run.status"
    HEARTBEAT = "heartbeat"
    STAGE_UPSERT = "stage.upsert"
    CELL_BATCH = "cell.batch"


# Log-only event types: stored and streamed by the platform, projected onto
# nothing. Named here so the emitter, the backfill, and the fixtures cannot drift
# on a spelling that only fails as a silently unprojected event.
LEDGER_LINE = "ledger.line"
LOG_LINE = "log.line"


# `run.meta` is the first fact of the band-0 walk, from either path.
RUN_META_SEQ = 1


def ledger_walk_seq(position: int) -> int:
    """The band-0 seq for the Nth `ledger.line`, matching backfill's walk exactly.

    Live emission and backfill must agree on this without talking to each other,
    which is why it is derived from the ledger FILE rather than from a counter: a
    line's index is fixed the moment it is appended, so any process (and a later
    backfill) computes the same seq for it.

    Args:
        position: The line's 1-based position among the arm's own ledger lines.

    Returns:
        Its band-0 seq: `run.meta` holds 1, so the Nth line holds `1 + N`.

    Raises:
        ValueError: If `position` is not a 1-based position.
    """
    if position < 1:
        msg = f"ledger position must be 1-based, got {position}"
        raise ValueError(msg)
    return RUN_META_SEQ + position


class SeqBandOverrun(RuntimeError):
    """Raised when a writer exhausts its seq band.

    Loud by design: continuing would walk into the next band's range, where the
    platform would accept the events as another writer's and discard that writer's
    real ones as duplicates.
    """


class RunEvent(BaseModel):
    """One event, carrying the run it belongs to.

    `external_id` rides along as an internal carrier so a walk can hold several
    runs' events in one list, but it is NOT part of a pushed event: the ingest
    route takes the run in its URL path and the body is
    `{"emitter_id": ..., "events": [{seq, ts, type, payload}]}`. `wire()` is the
    pushed shape; `jsonl_row()` is the fixture/dry-run shape that names the run.

    `ts` is a string, not a datetime: it is the artifact's own timestamp passed
    through untouched (see the module docstring).
    """

    model_config = ConfigDict(frozen=True)

    external_id: str
    seq: int = Field(ge=1)
    ts: str
    type: str
    payload: JsonObject = Field(default_factory=dict)

    def wire(self) -> JsonObject:
        """Return the event as it appears inside a push body's `events` list."""
        return {"seq": self.seq, "ts": self.ts, "type": self.type, "payload": self.payload}

    def jsonl_row(self) -> JsonObject:
        """Return the dry-run/fixture line, which names its run.

        Key order matches `d-runs-fixtures/expected-events.jsonl`, because the
        backfill's determinism test compares against that file directly.
        """
        return {"external_id": self.external_id, **self.wire()}


class SeqBands:
    """Hands out per-run seqs inside disjoint bands, one band per writer.

    Bands are allocated lazily and independently: asking for band 3 does not
    consume anything from bands 0 through 2, so a backfill that walks chunks out
    of order still produces the same seqs as the live hooks did.
    """

    def __init__(self, band_width: int = RUN_SEQ_BAND) -> None:
        """Initialize the allocator.

        Args:
            band_width: Seqs per band; only override in tests exercising overrun.
        """
        self._band_width = band_width
        self._next: dict[int, int] = {}
        self._next_from_top: dict[int, int] = {}

    @property
    def band_width(self) -> int:
        """Seqs available to each writer."""
        return self._band_width

    def band_start(self, band: int) -> int:
        """Return the first seq belonging to a band."""
        return band * self._band_width + 1

    def band_end(self, band: int) -> int:
        """Return the last seq belonging to a band."""
        return (band + 1) * self._band_width

    def take(self, band: int) -> int:
        """Consume and return the next seq in a band, ascending from its floor.

        Args:
            band: Writer's band (0 is the run-level walk; chunk k uses k + 1).

        Returns:
            The next unused seq in that band.

        Raises:
            SeqBandOverrun: If the band is exhausted, or if the ascending walk has
                met the descending one growing down from the top.
        """
        seq = self._next.get(band, self.band_start(band))
        ceiling = self._next_from_top.get(band, self.band_end(band))
        if seq > self.band_end(band) or seq > ceiling:
            msg = (
                f"seq band {band} is exhausted at {self.band_end(band)}: "
                f"{self._band_width} events is the per-writer limit, and continuing "
                f"would collide with band {band + 1}"
            )
            raise SeqBandOverrun(msg)
        self._next[band] = seq + 1
        return seq

    def take_from_top(self, band: int) -> int:
        """Consume and return the next seq in a band, DESCENDING from its ceiling.

        For facts with no artifact position (a heartbeat, a live `run.status`),
        which therefore cannot be re-derived. Descending from the top keeps them
        clear of the ascending, artifact-derived walk growing up from the floor, so
        one band safely carries both without either needing to know the other's
        count.

        Args:
            band: Writer's band.

        Returns:
            The next unused seq from the top of that band.

        Raises:
            SeqBandOverrun: If the descending walk has met the ascending one.
        """
        seq = self._next_from_top.get(band, self.band_end(band))
        floor = self._next.get(band, self.band_start(band))
        if seq < self.band_start(band) or seq < floor:
            msg = (
                f"seq band {band} is exhausted from the top at {self.band_start(band)}: "
                f"its descending live-only writes have met the ascending walk"
            )
            raise SeqBandOverrun(msg)
        self._next_from_top[band] = seq - 1
        return seq

    def resume_at(self, band: int, next_seq: int) -> None:
        """Rebase a band's ascending walk so it continues past what already landed.

        A re-invocation that restarts numbering at the band floor has every event
        discarded as a replay, because `(run, seq)` is the platform's idempotency
        key, so the run looks healthy while its new telemetry silently vanishes. This
        only ever moves the cursor FORWARD: a stale, lower resume mark cannot rewind
        a walk that has already gone further.

        Args:
            band: Writer's band.
            next_seq: The first seq to hand out (typically `last_seq + 1`).
        """
        current = self._next.get(band, self.band_start(band))
        self._next[band] = max(current, next_seq)

    def resume_from_top(self, band: int, next_seq: int) -> None:
        """Rebase a band's DESCENDING walk so it continues below what already landed.

        The twin of `resume_at`, and needed for the same reason: a re-invocation that restarts the
        descending walk at the band ceiling re-issues the seqs its previous invocation used, and the
        platform discards every one as a replay. That loss is quieter than the ascending kind and
        worse in one specific way: a run's terminal `run.status` descends, so a resumed run whose
        status is discarded stays `running` on the panel forever.

        Only ever moves the cursor DOWN, so a stale higher mark cannot rewind a walk that has
        already descended further.

        Args:
            band: Writer's band.
            next_seq: The first seq to hand out (typically the run's high-water minus one).
        """
        current = self._next_from_top.get(band, self.band_end(band))
        self._next_from_top[band] = min(current, next_seq)


def cell_band(chunk: int) -> int:
    """Return the band carrying a completed chunk's cell events.

    Chunk 0 takes band 1, leaving band 0 to the run-level walk (which also owns
    in-flight partial cells, because a chunk with no ledger line yet has no band
    of its own).
    """
    return chunk + 1


def grid_arm_external_id(grid_relpath: str, arm: str) -> str:
    """Build a grid arm's run name from its grid directory and arm.

    Args:
        grid_relpath: Grid directory path relative to `.wmo` (e.g. `jt/grid-c2`).
        arm: Arm name.

    Returns:
        The run's stable external id (e.g. `jt/grid-c2/identity`).
    """
    return f"{grid_relpath.strip('/')}/{arm}"


def pipeline_external_id(model: str) -> str:
    """Build an optimize pipeline's run name from its world-model name."""
    return f"{model}/optimize"


def grid_relpath(grid_dir: Path) -> str:
    """Derive a grid directory's `grid_arm_external_id` prefix: its path under `.wmo`.

    Every writer and every later backfill has to land on the SAME string for one directory, or
    they describe two different runs, so the derivation lives here beside the naming it feeds.
    A directory outside `.wmo` (a scratch grid, a smoke) falls back to its own name, which is
    still one stable name per directory.

    Args:
        grid_dir: The grid directory, absolute or relative.

    Returns:
        The run-name prefix (e.g. `jt/grid-c2`).
    """
    resolved = Path(grid_dir).resolve()
    parts = resolved.parts
    if ARTIFACT_DIR in parts:
        index = len(parts) - 1 - parts[::-1].index(ARTIFACT_DIR)
        return "/".join(parts[index + 1 :]) or resolved.name
    return resolved.name
