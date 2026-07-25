"""Tests for deriving a graded test-pass score from harbor's per-trial CTRF reports.

Every fixture here is written in the exact shape `pytest 8.4.1` + `pytest-json-ctrf 0.3.5` produced
for the 48-episode TerminalBench-2 probe (`_write_ctrf`), and the two probe-replication tests pin
the graded rate that probe's real 46 reports produce, so the parser cannot quietly regress to
reporting the binary reward.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from wmh.distill.rollouts import rollout_stats
from wmh.distill.tokens import assemble_trial_records
from wmh.evals.harbor.ctrf import read_trial_graded_tests
from wmh.harness.scoring import GradedTests, ScoreCell

# -- realistic CTRF fixtures -----------------------------------------------------------------------


def _write_ctrf(
    trial_dir: Path,
    *,
    passed: int = 0,
    failed: int = 0,
    skipped: int = 0,
    pending: int = 0,
    other: int = 0,
    summary: bool = True,
    itemized: bool = True,
    summary_override: dict[str, int] | None = None,
) -> Path:
    """Write one trial's CTRF report in harbor's real pytest-json-ctrf shape.

    Args:
        trial_dir: The harbor trial dir; the report lands in its `verifier/` subdir.
        passed: Tests that passed.
        failed: Tests that failed (each carrying the `trace`/`message` pytest-json-ctrf adds).
        skipped: Tests reported `skipped`.
        pending: Tests reported `pending`.
        other: Tests reported `other`.
        summary: Include the `results.summary` object.
        itemized: Include the `results.tests` array.
        summary_override: Replace the derived summary counts (to fixture a self-contradicting
            report).

    Returns:
        The report path.
    """
    tests: list[dict[str, object]] = []
    for status, count in (
        ("passed", passed),
        ("failed", failed),
        ("skipped", skipped),
        ("pending", pending),
        ("other", other),
    ):
        for index in range(count):
            entry: dict[str, object] = {
                "name": f"test_outputs.py::test_{status}_{index}",
                "status": status,
                "duration": 0.0004,
                "retries": 0,
                "file_path": "test_outputs.py",
            }
            if status == "failed":
                entry["raw_status"] = "call_failed"
                entry["trace"] = "E       AssertionError: Expected 'flag{...}'"
                entry["message"] = "The test failed in the call phase due to an assertion error"
            tests.append(entry)
    counts = {
        "tests": passed + failed + skipped + pending + other,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pending": pending,
        "other": other,
    }
    results: dict[str, object] = {"tool": {"name": "pytest", "version": "8.4.1"}}
    if summary:
        results["summary"] = {
            **(summary_override if summary_override is not None else counts),
            "start": 1784951738.4598818,
            "stop": 1784951738.5418391,
        }
    if itemized:
        results["tests"] = tests
    path = trial_dir / "verifier" / "ctrf.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path


# -- the graded score itself -----------------------------------------------------------------------


def test_a_partial_pass_scores_the_fraction_of_resolved_tests(tmp_path: Path) -> None:
    # gcode-to-text on the real probe: the file existed, its contents were wrong. Binary calls this
    # a total failure; graded calls it half done.
    _write_ctrf(tmp_path, passed=1, failed=1)

    breakdown = read_trial_graded_tests(tmp_path)

    assert breakdown == GradedTests(passed=1, resolved=2, unresolved=0)
    assert breakdown is not None
    assert breakdown.score == 0.5


def test_a_full_pass_and_a_full_fail_bracket_the_binary_reward(tmp_path: Path) -> None:
    full_pass = tmp_path / "pass"
    full_fail = tmp_path / "fail"
    _write_ctrf(full_pass, passed=6)
    _write_ctrf(full_fail, failed=6)

    passing = read_trial_graded_tests(full_pass)
    failing = read_trial_graded_tests(full_fail)

    assert passing is not None and passing.score == 1.0
    assert failing is not None and failing.score == 0.0
    # The invariant that keeps graded honest against the headline: a trial the benchmark calls a
    # pass scores exactly 1.0, so graded can never contradict the binary verdict, only refine it.
    assert passing.resolved == 6


def test_a_single_test_task_stays_exactly_as_binary_as_its_reward(tmp_path: Path) -> None:
    # 8 of the probe's 12 tasks ship exactly one test, so graded buys them nothing. Encoded so the
    # coarseness is a tested property, not a footnote.
    failed_dir = tmp_path / "one-failed"
    passed_dir = tmp_path / "one-passed"
    _write_ctrf(failed_dir, failed=1)
    _write_ctrf(passed_dir, passed=1)

    failing = read_trial_graded_tests(failed_dir)
    passing = read_trial_graded_tests(passed_dir)

    assert failing is not None and (failing.resolved, failing.score) == (1, 0.0)
    assert passing is not None and (passing.resolved, passing.score) == (1, 1.0)


# -- absent and unusable reports -------------------------------------------------------------------


def test_a_missing_report_is_none_never_a_zero(tmp_path: Path) -> None:
    # The live case: a verifier that timed out graded nothing and wrote nothing. Scoring it 0.0
    # would report an unknown outcome as a definite failure.
    assert read_trial_graded_tests(tmp_path) is None
    assert read_trial_graded_tests(tmp_path / "does-not-exist") is None


def test_an_unparseable_report_is_none_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    truncated = tmp_path / "truncated"
    (truncated / "verifier").mkdir(parents=True)
    (truncated / "verifier" / "ctrf.json").write_text('{"results": {"summ', encoding="utf-8")
    foreign = tmp_path / "foreign"
    (foreign / "verifier").mkdir(parents=True)
    (foreign / "verifier" / "ctrf.json").write_text('{"totals": {"ok": 3}}', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="wmh.evals.harbor.ctrf"):
        assert read_trial_graded_tests(truncated) is None
        assert read_trial_graded_tests(foreign) is None

    assert caplog.text.count("does not parse as a CTRF document") == 2


def test_a_report_whose_tests_all_lack_a_verdict_records_no_graded_score(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _write_ctrf(empty)
    skipped_only = tmp_path / "skipped-only"
    skipped_only.mkdir()
    _write_ctrf(skipped_only, skipped=3)

    with caplog.at_level(logging.WARNING, logger="wmh.evals.harbor.ctrf"):
        assert read_trial_graded_tests(empty) is None
        assert read_trial_graded_tests(skipped_only) is None

    assert caplog.text.count("no test that returned a verdict") == 2


# -- the skipped/pending/other decision ------------------------------------------------------------


def test_unresolved_tests_leave_the_denominator_instead_of_counting_as_failures(
    tmp_path: Path,
) -> None:
    """A skipped, pending, or `other` test is a statement about the grader, not the agent.

    It also does not block the benchmark's own binary pass (a pytest suite exits 0 with skips), so
    counting it as a failure would put the graded score BELOW a trial the benchmark calls a pass.
    """
    _write_ctrf(tmp_path, passed=1, failed=1, skipped=2, pending=1, other=1)

    breakdown = read_trial_graded_tests(tmp_path)

    assert breakdown == GradedTests(passed=1, resolved=2, unresolved=4)
    assert breakdown is not None
    assert breakdown.score == 0.5  # not 1/6


def test_a_pytest_pass_with_skips_still_scores_a_perfect_graded_one(tmp_path: Path) -> None:
    _write_ctrf(tmp_path, passed=2, skipped=3)

    breakdown = read_trial_graded_tests(tmp_path)

    assert breakdown is not None
    assert breakdown.score == 1.0
    assert breakdown.unresolved == 3


# -- summary vs itemized statuses ------------------------------------------------------------------


def test_the_summary_counts_are_read_first(tmp_path: Path) -> None:
    _write_ctrf(tmp_path, passed=5, failed=1, itemized=False)

    assert read_trial_graded_tests(tmp_path) == GradedTests(passed=5, resolved=6)


def test_itemized_statuses_are_the_fallback_when_no_summary_exists(tmp_path: Path) -> None:
    _write_ctrf(tmp_path, passed=1, failed=1, skipped=1, summary=False)

    assert read_trial_graded_tests(tmp_path) == GradedTests(passed=1, resolved=2, unresolved=1)


def test_a_summary_contradicting_its_own_tests_loses_to_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A summary count is derived; an itemized status is the primitive fact. Disagreement is loud.
    _write_ctrf(
        tmp_path,
        passed=1,
        failed=1,
        summary_override={
            "tests": 2,
            "passed": 2,
            "failed": 0,
            "skipped": 0,
            "pending": 0,
            "other": 0,
        },
    )

    with caplog.at_level(logging.WARNING, logger="wmh.evals.harbor.ctrf"):
        breakdown = read_trial_graded_tests(tmp_path)

    assert breakdown == GradedTests(passed=1, resolved=2)
    assert "disagrees with itself" in caplog.text


def test_an_unrecognized_itemized_status_is_unresolved_not_a_failure(tmp_path: Path) -> None:
    path = _write_ctrf(tmp_path, passed=1, summary=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = payload["results"]["tests"]
    tests.append({"name": "test_outputs.py::test_odd", "status": "flaked"})
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_trial_graded_tests(tmp_path) == GradedTests(passed=1, resolved=1, unresolved=1)


# -- the 48-episode probe, replicated ------------------------------------------------------------

_PROBE_TRIALS: dict[str, list[tuple[int, int, int] | None]] = {
    # (binary reward, passing tests, resolved tests) per trial, transcribed from the 46 real
    # ctrf.json reports of `.wmh/distill-runs/probe-scaffold/harbor/step-0000`; None is one of the
    # 2 trials whose verifier timed out, leaving neither a reward nor a report.
    "break-filter-js-from-html": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
    "configure-git-webserver": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (1, 1, 1)],
    "count-dataset-tokens": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
    "dna-insert": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
    "filter-js-from-html": [None, (0, 1, 2), None, (0, 1, 2)],
    "fix-git": [(1, 2, 2), (1, 2, 2), (1, 2, 2), (0, 1, 2)],
    "gcode-to-text": [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)],
    "git-multibranch": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
    "log-summary-date-ranges": [(1, 2, 2), (0, 0, 2), (1, 2, 2), (1, 2, 2)],
    "openssl-selfsigned-cert": [(1, 6, 6), (1, 6, 6), (0, 5, 6), (0, 2, 6)],
    "regex-log": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (1, 1, 1)],
    "sqlite-db-truncate": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
}


def _probe_cells(root: Path) -> list[ScoreCell]:
    """The probe's 48 trials as score cells, each reading a materialized CTRF report."""
    cells: list[ScoreCell] = []
    for task_id, trials in _PROBE_TRIALS.items():
        for attempt, entry in enumerate(trials, 1):
            trial_dir = root / f"{task_id}__{attempt}"
            trial_dir.mkdir(parents=True)
            if entry is None:
                # The verifier died: a stand-in 0.0 reward, no report, excluded from both rates.
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        reward=0.0,
                        passed=False,
                        artifact_dir=str(trial_dir),
                        note="infra-failure: VerifierTimeoutError; no verifier evidence",
                        infra_failed=True,
                        tests=read_trial_graded_tests(trial_dir),
                    )
                )
                continue
            reward, passing, resolved = entry
            _write_ctrf(trial_dir, passed=passing, failed=resolved - passing)
            cells.append(
                ScoreCell(
                    task_id=task_id,
                    attempt=attempt,
                    reward=float(reward),
                    passed=reward == 1,
                    artifact_dir=str(trial_dir),
                    note="completed",
                    tests=read_trial_graded_tests(trial_dir),
                )
            )
    return cells


def test_the_probe_graded_rate_reproduces_its_measured_values(tmp_path: Path) -> None:
    """The regression pin: the probe's own 46 reports, end to end into `RolloutStats`.

    If the parser ever collapses back to the binary reward, `graded_solve_rate` falls to 0.2174 and
    this fails.
    """
    cells = _probe_cells(tmp_path / "job")
    records = assemble_trial_records(cells, tmp_path / "sinks")

    stats = rollout_stats(records)

    assert (stats.trials, stats.executed_trials, stats.infra_failed_trials) == (48, 46, 2)
    # The two ungradeable trials are out of the graded denominator too, not zeros inside it.
    assert stats.graded_trials == 46
    assert round(stats.solve_rate, 4) == 0.2174  # 10/46 binary
    assert round(stats.graded_solve_rate, 4) == 0.3188
    assert stats.graded_solve_rate > stats.solve_rate
    # 9 of 46 gradeable trials (19.6%) scored reward 0 while passing at least one test: the entire
    # signal binary throws away on this task mix.
    hidden = [
        record
        for record in records
        if not record.infra_failed
        and record.reward == 0.0
        and record.tests is not None
        and record.tests.passed > 0
    ]
    assert len(hidden) == 9
    assert round(len(hidden) / stats.graded_trials, 3) == 0.196
    # Coarse by construction: 1 to 6 tests per trial, so most scores are 0, 1/2, or 1.
    resolved = {record.tests.resolved for record in records if record.tests is not None}
    assert (min(resolved), max(resolved)) == (1, 6)


def test_the_probe_graded_rate_moves_two_dead_tasks_off_zero(tmp_path: Path) -> None:
    """Per-task resolution, which is where the statistical power comes from.

    A task at a flat 0.00 or 1.00 binary rate can never register an improvement; graded moves
    `filter-js-from-html` and `gcode-to-text` to 0.50, the most informative position there is.
    """
    cells = _probe_cells(tmp_path / "job")
    records = assemble_trial_records(cells, tmp_path / "sinks")
    gradeable = [record for record in records if not record.infra_failed]

    binary: dict[str, float] = {}
    graded: dict[str, float] = {}
    for task_id in _PROBE_TRIALS:
        task_records = [record for record in gradeable if record.task_id == task_id]
        binary[task_id] = sum(record.passed for record in task_records) / len(task_records)
        scores = [record.graded_score for record in task_records if record.graded_score is not None]
        graded[task_id] = sum(scores) / len(scores)

    assert (binary["filter-js-from-html"], graded["filter-js-from-html"]) == (0.0, 0.5)
    assert (binary["gcode-to-text"], graded["gcode-to-text"]) == (0.0, 0.5)
    assert binary["openssl-selfsigned-cert"] == 0.5
    assert round(graded["openssl-selfsigned-cert"], 4) == 0.7917
    assert binary["fix-git"] == 0.75
    assert graded["fix-git"] == 0.875
    # Informative = neither floored nor saturated, so an effect could show at all.
    informative_binary = sum(1 for rate in binary.values() if 0.0 < rate < 1.0)
    informative_graded = sum(1 for rate in graded.values() if 0.0 < rate < 1.0)
    assert (informative_binary, informative_graded) == (5, 7)
    assert len(binary) == 12
