"""Probe-coordinate grading tells known outcomes apart."""

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.grading_controls import EXPECTED, run_controls

STATE: JsonObject = {
    "Workspace filesystem": {"files": {"/workspace/test_a.py": "frozen"}},
    "Tests": {"status": "red"},
}
PROBES = {
    "status": "/Tests/status",
    "test_file": "/Workspace filesystem/files/~1workspace~1test_a.py",
}
VERIFIER: JsonObject = {
    "description": "The suite turns green, a.py exists and the frozen test is unchanged.",
    "outcome_schema": {},
    "allowed_tools": ["bash"],
    "required_paths": [
        "/final_state/Tests/status",
        "/final_state/Workspace filesystem/files//workspace/a.py",
    ],
    "rules": [{"operator": "equals", "path": "/final_state/Tests/status", "value": "green"}],
    "preserved_paths": ["/Workspace filesystem/files/~1workspace~1test_a.py"],
}


def test_every_control_receives_its_expected_grade() -> None:
    """Correct passes; no-op and wrong content fail; unobserved is unassessable; gone fails."""
    results = {row.control: row for row in run_controls(VERIFIER, STATE, PROBES)}
    assert set(results) == set(EXPECTED)
    for row in results.values():
        assert row.ok, (row.control, row.status, row.reason)
    assert "absent" in results["observed_absence"].reason
    assert results["missing_observation"].status == "unassessable"


def test_a_verifier_reading_outside_the_workflow_cannot_be_shown_correct() -> None:
    """A rule on a path no workflow holds makes the start unpassable, and the controls say so."""
    broken = {**VERIFIER, "rules": [{"operator": "equals", "path": "/tests/passed", "value": 4}]}
    results = {row.control: row for row in run_controls(broken, STATE, PROBES)}
    assert results["correct"].status == "not_constructible" and not results["correct"].ok
