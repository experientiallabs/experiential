"""Tests for the sweep's crash-safe sidecar: what survives a kill, and what is refused.

The sweep-level behaviour (resume, remeasure, the matrix the two halves assemble into) lives in
`sweep_test.py`; these pin the file format itself, where the failure modes are damage and
mistaken identity rather than measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.routing.outcomes import ScenarioOutcome
from wmo.optimize.routing.sweep_partial import (
    PARTIAL_FORMAT_VERSION,
    PartialHeader,
    PartialSweepError,
    PartialWriter,
    PlanIdentity,
    partial_path,
    read_partial,
)


def _identity(**overrides: object) -> PlanIdentity:
    fields: dict[str, object] = {
        "pool": "poolsha",
        "scenarios": ("s1", "s2"),
        "episodes": 1,
        "max_steps": 20,
        "history_chars": 2000,
        "compression": "raw text (no compression)",
    }
    return PlanIdentity.model_validate(fields | overrides)


def _row(scenario: str, *, model: str = "cheap", episode: int = 0) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario, task=f"task {scenario}", model=model, episode=episode, reward=0.5
    )


def test_the_sidecar_sits_beside_the_matrix_it_protects() -> None:
    assert partial_path(Path("/tmp/run/matrix.json")).name == "matrix.json.partial.jsonl"


def test_rows_written_are_rows_read_back(tmp_path: Path) -> None:
    path = partial_path(tmp_path / "matrix.json")
    with PartialWriter(path, _identity()) as writer:
        writer.append(_row("s1"))
        writer.append(_row("s2"))
    assert [row.scenario_id for row in read_partial(path, _identity())] == ["s1", "s2"]


def test_reopening_appends_rather_than_starting_over(tmp_path: Path) -> None:
    # This IS the resume: the second attempt adds to what the first one paid for.
    path = partial_path(tmp_path / "matrix.json")
    with PartialWriter(path, _identity()) as first:
        first.append(_row("s1"))
    with PartialWriter(path, _identity()) as second:
        second.append(_row("s2"))
    assert len(read_partial(path, _identity())) == 2
    assert path.read_text(encoding="utf-8").count("\n") == 3  # one header, two rows


def test_a_missing_sidecar_is_simply_nothing_to_resume(tmp_path: Path) -> None:
    assert read_partial(partial_path(tmp_path / "matrix.json"), _identity()) == []


def test_a_torn_last_line_costs_that_cell_and_nothing_else(tmp_path: Path) -> None:
    # A kill between the write and the flush truncates the line being written. That cell is
    # unmeasured and gets bought again; every completed cell before it still counts.
    path = partial_path(tmp_path / "matrix.json")
    with PartialWriter(path, _identity()) as writer:
        writer.append(_row("s1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"scenario_id": "s2", "task": "half a ro')
    rows = read_partial(path, _identity())
    assert [row.scenario_id for row in rows] == ["s1"]


def test_damage_in_the_middle_is_refused_because_the_log_is_then_unknown(tmp_path: Path) -> None:
    path = partial_path(tmp_path / "matrix.json")
    with PartialWriter(path, _identity()) as writer:
        writer.append(_row("s1"))
        writer.append(_row("s2"))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = "{not json at all"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(PartialSweepError, match="damaged in the middle"):
        read_partial(path, _identity())


def test_a_file_this_module_did_not_write_is_refused_by_its_first_line(tmp_path: Path) -> None:
    path = partial_path(tmp_path / "matrix.json")
    path.write_text('{"some": "other jsonl file"}\n', encoding="utf-8")
    with pytest.raises(PartialSweepError, match="does not start with a partial-sweep header"):
        read_partial(path, _identity())


def test_a_newer_format_says_which_build_can_finish_that_sweep(tmp_path: Path) -> None:
    path = partial_path(tmp_path / "matrix.json")
    header = PartialHeader(version=PARTIAL_FORMAT_VERSION + 1, identity=_identity())
    path.write_text(header.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(PartialSweepError, match="format v"):
        read_partial(path, _identity())


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"pool": "different"}, "candidate pool changed"),
        ({"scenarios": ("s1",)}, "scenario cut changed"),
        ({"episodes": 3}, "episodes per cell changed"),
        ({"max_steps": 40}, "step budget changed"),
        ({"history_chars": 500}, "observation window changed"),
        ({"compression": "compressor 'truncate' version 1 at aggressiveness 0.5"}, "arm changed"),
    ],
)
def test_every_cohort_pin_is_a_refusal_that_names_itself(
    tmp_path: Path, overrides: dict[str, object], expected: str
) -> None:
    # Each of these makes the rows a different arm. Merging any of them into one matrix would be
    # a comparison nobody measured, so the refusal names the pin that moved.
    path = partial_path(tmp_path / "matrix.json")
    with PartialWriter(path, _identity()) as writer:
        writer.append(_row("s1"))
    with pytest.raises(PartialSweepError, match=expected):
        read_partial(path, _identity(**overrides))


def test_the_identity_digest_is_stable_and_sensitive() -> None:
    assert _identity().digest == _identity().digest
    assert _identity().digest != _identity(episodes=2).digest
    assert _identity().mismatch(_identity()) is None


def test_discarding_removes_the_file_the_matrix_replaced(tmp_path: Path) -> None:
    path = partial_path(tmp_path / "matrix.json")
    writer = PartialWriter(path, _identity())
    writer.append(_row("s1"))
    assert path.is_file()
    writer.discard()
    assert not path.exists()
    writer.discard()  # idempotent: a second finish must not explode
