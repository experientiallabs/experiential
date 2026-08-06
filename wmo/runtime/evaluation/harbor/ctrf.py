"""Read one harbor trial's CTRF test report into a graded test-pass breakdown.

Harbor's pytest verifiers run under `pytest-json-ctrf` and write a CTRF (Common Test Report
Format) document to `<trial_dir>/verifier/ctrf.json` beside the binary `reward.txt`. That report
is the only place a trial's per-test outcomes survive, so it is where the resolution a binary
reward discards comes from (see `wmo.runtime.harness.scoring.GradedTests` for why that resolution
matters and how coarse it is).

The shape this parses, confirmed against all 46 reports of the 48-episode TerminalBench-2 probe
(`pytest 8.4.1` + `pytest-json-ctrf 0.3.5`), which were byte-identical in structure:

```json
{"results": {"tool": {...},
             "summary": {"tests": 2, "passed": 1, "failed": 1, "skipped": 0,
                         "pending": 0, "other": 0, "start": ..., "stop": ...},
             "tests": [{"name": "test_outputs.py::test_hello_file_exists", "status": "passed",
                        ...}, ...]}}
```

`results.summary` is the authoritative aggregate and is read first; `results.tests[]` statuses are
the fallback when no summary is present, and the tiebreaker when the two disagree (an itemized
status is a primitive fact, a summary count is derived). Nothing here fabricates a score: a
missing, unreadable, or empty report yields None, which callers must exclude from graded rates
rather than average in as 0.0.

Multi-step harbor trials relocate their verifier dir to `steps/<step>/verifier/`, so they record no
graded score today (None, never a zero). Every TerminalBench-2 task WMO distills on is single-step.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.runtime.harness.scoring import GradedTests

logger = logging.getLogger(__name__)

CTRF_REPORT_FILENAME = "ctrf.json"
"""Harbor's per-trial CTRF report, written by the verifier into the trial's `verifier/` dir."""

_PASSED_STATUS = "passed"
_FAILED_STATUS = "failed"
_RESOLVED_STATUSES = frozenset({_PASSED_STATUS, _FAILED_STATUS})
"""CTRF statuses that carry a verdict. `skipped`, `pending`, and `other` deliberately do not:
they say the grader never ran the test, which is not the agent's failure and does not stop the
benchmark's own binary pass either (pytest exits 0 with skips)."""


class _CtrfTest(BaseModel):
    """One entry of `results.tests[]`; only its status is load-bearing here."""

    model_config = ConfigDict(extra="ignore")

    status: str = ""


class _CtrfSummary(BaseModel):
    """The `results.summary` aggregate; counts are optional so a partial summary degrades."""

    model_config = ConfigDict(extra="ignore")

    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    skipped: int = Field(default=0, ge=0)
    pending: int = Field(default=0, ge=0)
    other: int = Field(default=0, ge=0)


class _CtrfResults(BaseModel):
    """The `results` object: the summary plus the itemized tests."""

    model_config = ConfigDict(extra="ignore")

    summary: _CtrfSummary | None = None
    tests: list[_CtrfTest] = Field(default_factory=list)


class _CtrfReport(BaseModel):
    """The CTRF document root."""

    model_config = ConfigDict(extra="ignore")

    results: _CtrfResults


def read_trial_graded_tests(trial_dir: Path) -> GradedTests | None:
    """The per-test breakdown one harbor trial's verifier recorded, when it recorded one.

    Args:
        trial_dir: The harbor trial directory (a `ScoreCell.artifact_dir`).

    Returns:
        The trial's `GradedTests`, or None when no graded score exists for it: no report file,
        an unreadable or malformed one, or a report in which no test returned a verdict. None is
        never a 0.0 and callers must keep it out of graded denominators rather than count it as a
        failure, the same rule that keeps an ungradeable trial out of `solve_rate`.
    """
    path = trial_dir / "verifier" / CTRF_REPORT_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Overwhelmingly the ordinary case (no report was written), so this is not a warning: the
        # verifier-failure diagnosis belongs to the scorer's `infra_failed` note.
        logger.debug("no readable CTRF report at %s; this trial records no graded score", path)
        return None
    try:
        report = _CtrfReport.model_validate_json(text)
    except ValidationError:
        logger.warning(
            "CTRF report %s does not parse as a CTRF document; this trial records no graded "
            "score (its binary reward is unaffected)",
            path,
            exc_info=True,
        )
        return None
    return _breakdown(report.results, path)


def _breakdown(results: _CtrfResults, path: Path) -> GradedTests | None:
    """One report's counts, preferring itemized statuses when they contradict the summary."""
    from_summary = _from_summary(results.summary)
    from_tests = _from_tests(results.tests)
    if from_summary is not None and from_tests is not None and from_summary != from_tests:
        logger.warning(
            "CTRF report %s disagrees with itself: summary says %s, its %d itemized test(s) say "
            "%s; using the itemized statuses, which are the primitive fact",
            path,
            from_summary,
            len(results.tests),
            from_tests,
        )
        return from_tests
    if from_summary is not None:
        return from_summary
    if from_tests is None:
        logger.warning(
            "CTRF report %s carries no test that returned a verdict (no summary counts and no "
            "passed/failed entries); this trial records no graded score",
            path,
        )
    return from_tests


def _from_summary(summary: _CtrfSummary | None) -> GradedTests | None:
    """The breakdown `results.summary` states, or None when it states no verdict counts."""
    if summary is None or summary.passed is None or summary.failed is None:
        return None
    resolved = summary.passed + summary.failed
    if resolved == 0:
        return None
    return GradedTests(
        passed=summary.passed,
        resolved=resolved,
        unresolved=summary.skipped + summary.pending + summary.other,
    )


def _from_tests(tests: list[_CtrfTest]) -> GradedTests | None:
    """The breakdown the itemized `results.tests[]` statuses state, or None when none has a verdict.

    An unrecognized status counts as unresolved rather than as a failure: it means this parser does
    not know what the grader said, which is not evidence against the agent.
    """
    passed = sum(1 for test in tests if test.status == _PASSED_STATUS)
    resolved = sum(1 for test in tests if test.status in _RESOLVED_STATUSES)
    if resolved == 0:
        return None
    return GradedTests(passed=passed, resolved=resolved, unresolved=len(tests) - resolved)
