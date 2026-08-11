"""Tests for the build-progress reporting seam and its no-op default.

`BuildReporter` is the only thing the headless build pipeline knows about its UI, so what matters
is that the shipped implementations cover the whole protocol: a missing method surfaces as an
`AttributeError` mid-build, after the expensive stages have already run.
"""

from __future__ import annotations

import inspect

from wmo.common.observability.reporting import BuildReporter, NullReporter

#: Every event the build pipeline emits, in the order the lifecycle reaches them.
_PROTOCOL_METHODS = (
    "ingest_done",
    "split_done",
    "index_done",
    "optimize_start",
    "rollout",
    "activity",
    "optimize_done",
)


def test_the_protocol_is_exactly_the_build_lifecycle() -> None:
    # Pinned against the declaration itself: a new event added to the protocol without a stub for
    # it here would otherwise leave the NullReporter/RichBuildReporter checks below silently
    # narrower than the contract.
    declared = sorted(name for name in vars(BuildReporter) if not name.startswith("_"))

    assert declared == sorted(_PROTOCOL_METHODS)


def test_null_reporter_implements_every_protocol_method_with_a_matching_signature() -> None:
    for name in _PROTOCOL_METHODS:
        implementation = getattr(NullReporter, name, None)
        assert implementation is not None, f"NullReporter is missing {name}"
        assert inspect.signature(implementation) == inspect.signature(getattr(BuildReporter, name))


def test_rich_build_reporter_implements_every_protocol_method() -> None:
    # The CLI's reporter is the one real implementation; the pipeline calls these by name, so a
    # protocol method it forgot would only fail during an actual build.
    from wmo.cli.ui import RichBuildReporter

    missing = [name for name in _PROTOCOL_METHODS if not hasattr(RichBuildReporter, name)]
    assert not missing, f"RichBuildReporter is missing {missing}"


def test_null_reporter_swallows_every_event() -> None:
    reporter: BuildReporter = NullReporter()

    reporter.ingest_done(traces=2, steps=9)
    reporter.split_done(train=1, val=1, test=1)
    reporter.index_done(steps=9)
    reporter.optimize_start(budget=4)
    reporter.rollout(done=1, budget=4, score=None)
    reporter.rollout(done=2, budget=4, score=0.5)
    reporter.activity("proposed a new prompt")

    assert reporter.optimize_done(held_out_accuracy=0.5, frontier_size=2, rollouts=4) is None
