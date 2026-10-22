"""A report covers a fixed number of rounds of one climb."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.hillclimb.report import ClimbReportSpec, preflight_climb_report
from exp.optimize.evaluation.hillclimb.tests.climb_test import climb_fixture


def test_report_identity_ignores_its_deadline_and_needs_decided_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without kept edits no final comparison is planned; undecided rounds refuse."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    now = datetime.now(UTC)
    first = ClimbReportSpec(climb=spec, rounds=0, deadline=now)
    assert (
        first.identity()
        == first.model_copy(update={"deadline": now + timedelta(days=1)}).identity()
    )
    assert preflight_climb_report(project, first) is False
    with pytest.raises(ValueError, match="round 1 of this climb has no saved decision"):
        preflight_climb_report(project, ClimbReportSpec(climb=spec, rounds=1, deadline=now))
    assert not actor.requests
