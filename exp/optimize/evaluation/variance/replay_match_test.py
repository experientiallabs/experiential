"""A replayed step matches its recording on substance; only the undecided go to the judge."""

import pytest

from exp.optimize.evaluation.variance.bootstrap_agents import ReplayJudgement
from exp.optimize.evaluation.variance.bootstrap_capture import (
    _execute_retrying,
    judge_undecided,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import ReplayStep, RoleDraw
from exp.optimize.evaluation.variance.replay_match import match_step

RECORDED = (
    ";; bridge result\nstatus: ok\nPython 3.13.5\nTraceback (most recent call last):\n"
    '  File "<string>", line 1, in <module>\n    import yaml; print(yaml.__version__)\n'
    "    ^^^^^^^^^^^\nModuleNotFoundError: No module named 'yaml'\n"
)
REPLAYED = (
    ";; bridge result\nstatus: ok\nPython 3.13.5\nTraceback (most recent call last):\n"
    '  File "<string>", line 1, in <module>\n    import yaml\n'
    "ModuleNotFoundError: No module named 'yaml'\n"
)
LISTING = (
    "total 16\ndrwxrwxrwx 3 node 986 4096 Aug 24 05:56 .\n"
    "-rw-rw-r-- 1 node 986 3508 Aug 24 05:56 notes.txt\n"
    "/bin/bash: line 1: pytest: command not found\nCommand exited with code {code}"
)


def test_bash_output_that_differs_only_in_detail_passes_after_normalisation() -> None:
    """An abridged traceback with the same failure is the same outcome, not a mismatch."""
    match = match_step(RECORDED, False, REPLAYED, False)
    assert (match.decision, match.rule) == ("pass", "normalised")
    assert match.similarity is not None and match.similarity >= 0.6
    assert match_step(RECORDED, False, RECORDED, False).rule == "exact"


def test_a_recorded_exit_127_replayed_as_exit_0_fails() -> None:
    """Different stated exit codes fail however similar the rest of the output is."""
    match = match_step(LISTING.format(code=127), True, LISTING.format(code=0), False)
    assert match.decision == "fail"
    assert (match.recorded_exit, match.replayed_exit) == (127, 0)


def test_a_borderline_step_goes_to_the_judge_and_only_it() -> None:
    """Exit code stated on one side only is undecided; the scripted judge settles that step."""
    replayed = LISTING.format(code=127).replace("\nCommand exited with code 127", "")
    undecided = match_step(LISTING.format(code=127), True, replayed, False)
    assert undecided.decision == "undecided"
    rows = (
        _row("trace:t:span:a", match_step(RECORDED, False, REPLAYED, False)),
        _row("trace:t:span:b", undecided),
    )
    asked: list[int] = []

    def judge(index: int, step: dict) -> tuple[RoleDraw, ReplayJudgement | None]:
        asked.append(index)
        assert step["recorded_exit"] == 127 and step["replayed_exit"] is None
        draw = RoleDraw(role="replay_judge", variant="same_outcome", round=0)
        return draw, ReplayJudgement(same_outcome=True, reason="pytest is missing in both")

    settled = judge_undecided(rows, judge)
    assert asked == [1]
    assert [row.rule for row in settled] == ["normalised", "judge"]
    assert all(row.agrees for row in settled) and settled[1].judge is not None


def test_a_judge_without_an_answer_fails_the_step() -> None:
    """No usable judgement is a failed step, recorded with the judge's failure."""
    replayed = LISTING.format(code=127).replace("\nCommand exited with code 127", "")
    row = _row("trace:t:span:b", match_step(LISTING.format(code=127), True, replayed, False))
    draw = RoleDraw(role="replay_judge", variant="same_outcome", round=0, failure="no answer")
    (settled,) = judge_undecided((row,), lambda index, step: (draw, None))
    assert not settled.agrees and settled.rule is None and "no answer" in settled.reason


def test_an_empty_simulator_reply_is_retried_once_then_raised() -> None:
    """One retry with the same action; a second unusable reply propagates to be refused."""
    from exp.runtime.models.providers.errors import ProviderRetryableResponseError

    class Session:
        def __init__(self, failures: int) -> None:
            self.failures, self.calls = failures, 0

        def execute(self, action: object) -> str:
            self.calls += 1
            if self.calls <= self.failures:
                raise ProviderRetryableResponseError("empty reply")
            return "ok"

    once = Session(1)
    assert _execute_retrying(once, object()) == "ok" and once.calls == 2  # type: ignore[arg-type]
    twice = Session(2)
    with pytest.raises(ProviderRetryableResponseError):
        _execute_retrying(twice, object())  # type: ignore[arg-type]
    assert twice.calls == 2


def _row(locator: str, match: object) -> ReplayStep:
    """A replayed step carrying a match, as ``replay_prefix`` builds it."""
    from exp.optimize.evaluation.variance.bootstrap_capture import UNDECIDED

    decision = match.decision  # type: ignore[attr-defined]
    return ReplayStep(
        locator=locator,
        action={"name": "bash", "arguments": {}},
        recorded="r",
        recorded_error=False,
        replayed="p",
        replayed_error=False,
        agrees=decision == "pass",
        rule=match.rule,  # type: ignore[attr-defined]
        similarity=match.similarity,  # type: ignore[attr-defined]
        recorded_exit=match.recorded_exit,  # type: ignore[attr-defined]
        replayed_exit=match.replayed_exit,  # type: ignore[attr-defined]
        reason=(UNDECIDED if decision == "undecided" else "") + match.reason,  # type: ignore[attr-defined]
    )
