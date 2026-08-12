"""Tests for `NullReporter`, the default sink the headless build pipeline calls into.

The `BuildReporter` protocol gets no test of its own: asserting which methods it declares restates
the declaration. What can actually break is the default implementation drifting from it, because
the pipeline calls these by name with keywords, so a missing or renamed parameter surfaces as a
`TypeError` mid-build, after the expensive stages have already run. The one real implementation is
covered beside itself in `wmo/cli/ui_test.py`.
"""

from __future__ import annotations

import inspect

from wmo.common.observability.reporting import BuildReporter, NullReporter

#: Read off the protocol rather than hand-copied, so a new event is checked the moment it is added.
_EVENTS = tuple(name for name in vars(BuildReporter) if not name.startswith("_"))


def test_the_default_reporter_accepts_every_event_exactly_as_declared() -> None:
    for name in _EVENTS:
        implementation = getattr(NullReporter, name, None)
        assert implementation is not None, f"NullReporter is missing {name}"
        assert inspect.signature(implementation) == inspect.signature(getattr(BuildReporter, name))


def test_a_whole_build_lifecycle_through_the_default_reporter_does_nothing() -> None:
    # A library caller passes no reporter, so every event lands here. Swallowing them must be
    # total: one event that raised would fail a build that asked for no progress output at all.
    reporter: BuildReporter = NullReporter()

    reporter.ingest_done(traces=2, steps=9)
    reporter.split_done(train=1, val=1, test=1)
    reporter.index_done(steps=9)
    reporter.optimize_start(budget=4)
    reporter.rollout(done=1, budget=4, score=None)
    reporter.rollout(done=2, budget=4, score=0.5)
    reporter.activity("proposed a new prompt")

    assert reporter.optimize_done(held_out_accuracy=0.5, frontier_size=2, rollouts=4) is None
