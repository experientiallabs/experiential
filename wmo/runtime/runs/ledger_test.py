"""Tests for the shared grid-ledger schema.

These assert the properties the two readers DEPEND on, not the field list: that an
unrecognized key makes a line non-conforming (which is what makes "did the runner count
this?" decidable), and that the optional fields really are optional (so an older
writer's line still counts and does not renumber everything after it).
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue, ValidationError

from wmo.core.types import JsonObject
from wmo.runtime.runs.ledger import Calibration, LedgerLine


def _line(**overrides: JsonValue) -> JsonObject:
    """One conforming ledger line, as the runner writes them."""
    line: JsonObject = {
        "event": "chunk",
        "arm": "identity",
        "chunk": 0,
        "ts": "2026-07-27T09:00:00+00:00",
        "tip_sha": "abc123",
        "max_steps": 20,
        "episodes": 2,
    }
    line.update(overrides)
    return line


def test_an_unrecognized_key_makes_a_line_non_conforming() -> None:
    """The forbid rule is the whole contract: it decides what counts as a position.

    A newer writer adding a field would otherwise be counted by one reader and skipped
    by another, and every seq after that line would differ between the two.
    """
    assert LedgerLine.model_validate(_line()).chunk == 0

    with pytest.raises(ValidationError):
        LedgerLine.model_validate(_line(surprise="from a newer writer"))


def test_missing_required_identity_fields_are_refused() -> None:
    """A line with no clock or no provenance cannot anchor a seq or a cohort."""
    for missing in ("ts", "tip_sha", "arm", "event"):
        incomplete = _line()
        del incomplete[missing]
        with pytest.raises(ValidationError):
            LedgerLine.model_validate(incomplete)


def test_the_spend_and_count_fields_default_to_zero() -> None:
    """An event that spent nothing omits the legs rather than writing zeros.

    They have to stay optional: a `stop` or `merge` line carries no cells and no
    dollars, and refusing it would drop a line the runner counts.
    """
    line = LedgerLine.model_validate(_line(event="stop"))

    assert (line.cells, line.scored) == (0, 0)
    assert (line.candidate_usd, line.compressor_usd, line.wm_usd) == (0.0, 0.0, 0.0)
    assert line.chunk == 0
    assert line.note == ""
    assert line.calibration is None


def test_a_calibration_block_is_validated_too() -> None:
    """A nested block with a stray key must fail the whole line, not pass silently.

    Otherwise the two readers could disagree about a line whose top level looks fine.
    """
    calibration = {
        "sample_size": 8,
        "sample_tokens_raw": 4096,
        "endpoint_aggressiveness": 0.5,
        "endpoint_achieved_ratio": 0.62,
        "searched": [[0.4, 0.7], [0.5, 0.62]],
        "chosen_aggressiveness": 0.5,
        "chosen_achieved_ratio": 0.62,
        "tolerance": 0.02,
        "measured_at": "2026-07-27T08:00:00+00:00",
        "tip_sha": "abc123",
    }

    parsed = LedgerLine.model_validate(_line(event="calibration", calibration=calibration))
    assert isinstance(parsed.calibration, Calibration)
    assert parsed.calibration.searched == [(0.4, 0.7), (0.5, 0.62)]

    with pytest.raises(ValidationError):
        LedgerLine.model_validate(
            _line(event="calibration", calibration={**calibration, "extra": 1})
        )
