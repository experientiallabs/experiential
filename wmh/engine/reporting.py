"""Progress reporting for the build pipeline.

The build emits coarse lifecycle events (ingest, split, index, optimize, persist) plus fine-grained
GEPA rollout events to a `BuildReporter`. The engine depends only on this protocol — never on rich —
so the pipeline stays headless and testable. The CLI supplies a rich-backed reporter
(`wmh.cli.ui.RichBuildReporter`); everything else uses `NullReporter`.
"""

from __future__ import annotations

from typing import Protocol


class BuildReporter(Protocol):
    """Sink for build-pipeline progress events. All methods are best-effort and side-effecting."""

    def ingest_done(self, traces: int, steps: int) -> None:
        """Called once after traces are ingested + normalized."""
        ...

    def split_done(self, train: int, val: int, test: int) -> None:
        """Called once after the train/val/test split (counts are traces)."""
        ...

    def index_done(self, steps: int) -> None:
        """Called once after the retrieval index is built over the replay buffer."""
        ...

    def optimize_start(self, budget: int) -> None:
        """Called once before GEPA begins, with the rollout budget."""
        ...

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        """Called as GEPA consumes its rollout budget. `score` is the mean rollout score so far."""
        ...

    def activity(self, line: str) -> None:
        """Called with one line of GEPA narration (proposals, selection scores, judge notes)."""
        ...

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        """Called once after GEPA finishes."""
        ...


class NullReporter:
    """A reporter that does nothing — the default for library/test callers."""

    def ingest_done(self, traces: int, steps: int) -> None: ...
    def split_done(self, train: int, val: int, test: int) -> None: ...
    def index_done(self, steps: int) -> None: ...
    def optimize_start(self, budget: int) -> None: ...
    def rollout(self, done: int, budget: int, score: float | None) -> None: ...
    def activity(self, line: str) -> None: ...

    def optimize_done(
        self, held_out_accuracy: float, frontier_size: int, rollouts: int
    ) -> None: ...
