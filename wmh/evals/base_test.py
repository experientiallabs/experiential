"""Protocol-conformance tests: the real report/eval classes must satisfy the base protocols.

The `EvalResult`/`Evaluation` protocols are the seam `wmh eval --mode ...` dispatches through;
a report class silently drifting out of conformance (renaming `headline`, dropping `summary`)
would break the CLI for that mode without any type error at the definition site. These tests pin
the contract with `runtime_checkable` isinstance checks against the shipped implementations.
"""

from __future__ import annotations

from wmh.evals.base import EvalResult, Evaluation
from wmh.evals.closed_loop import ClosedLoopEval, ClosedLoopReport
from wmh.evals.open_loop import EvalReport, OpenLoopEval


def test_open_loop_report_satisfies_eval_result() -> None:
    report = EvalReport(overall_fidelity=0.5, overall_std=0.1, total_steps=10, total_invalid=1)
    assert isinstance(report, EvalResult)
    assert report.headline == 0.5
    assert "fidelity=0.500" in report.summary()


def test_closed_loop_report_satisfies_eval_result() -> None:
    report = ClosedLoopReport(success_rate=0.75, k=2)
    assert isinstance(report, EvalResult)
    assert report.headline == 0.75
    assert isinstance(report.summary(), str)


def test_both_eval_classes_satisfy_evaluation_protocol() -> None:
    assert issubclass(OpenLoopEval, Evaluation)
    assert issubclass(ClosedLoopEval, Evaluation)


def test_non_conforming_object_is_rejected() -> None:
    class NotAResult:
        pass

    assert not isinstance(NotAResult(), EvalResult)
    assert not isinstance(NotAResult(), Evaluation)
