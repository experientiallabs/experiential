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


def test_a_rule_contradicting_a_preserved_path_leaves_no_passing_state() -> None:
    """Run 28's 86fd verifier: equals a literal on the probe it also protects; nothing passes."""
    contradictory = {
        **VERIFIER,
        "rules": [{"operator": "equals", "path": "/final_state/test_file", "value": "unchanged"}],
        "required_paths": ["/final_state/test_file"],
        "preserved_paths": ["/test_file"],
    }
    results = {row.control: row for row in run_controls(contradictory, STATE, PROBES)}
    assert results["correct"].status == "fail" and "protected state changed" in (
        results["correct"].reason
    )


def test_a_domain_keeps_its_initial_shape() -> None:
    """A filesystem domain must stay an object with its files map; values may change."""
    from exp.optimize.evaluation.variance.bootstrap_agents import shape_schema
    from exp.runtime.environments.world_responses import validate_state_projection

    schema = shape_schema(STATE["Workspace filesystem"])
    assert schema == {
        "type": "object",
        "required": ["files"],
        "properties": {"files": {"type": "object"}},
    }
    validate_state_projection({"fs": {"files": {"/workspace/new.py": "x"}}}, {"fs": schema})
    import pytest

    with pytest.raises(ValueError, match="transition refused"):
        validate_state_projection({"fs": "the whole filesystem as text"}, {"fs": schema})


def test_verifier_reads_the_simulator_cannot_be_trusted_to_hold_are_refused() -> None:
    """Run 30: a whole file demanded verbatim, and fields read inside a probe that starts null."""
    from types import SimpleNamespace
    from typing import cast

    import pytest

    from exp.optimize.evaluation.variance.bootstrap_agents import _check_verifier_reads
    from exp.simulation.evaluation.tasks import TaskVerifier

    state: JsonObject = {**STATE, "Tests": {"status": "red", "last_run": None}}
    probes = {**PROBES, "last_run": "/Tests/last_run"}

    from exp.optimize.evaluation.variance.bootstrap_contracts import StartDraft

    def draft(rules: list[JsonObject]) -> StartDraft:
        verifier = TaskVerifier.model_validate(
            {"description": "d", "outcome_schema": {}, "rules": rules}
        )
        stand_in = SimpleNamespace(initial_state=state, probe_paths=probes, verifier=verifier)
        return cast("StartDraft", stand_in)

    verbatim: JsonObject = {
        "operator": "equals",
        "path": "/final_state/test_file",
        "value": "x" * 500,
    }
    with pytest.raises(ValueError, match="verbatim"):
        _check_verifier_reads(draft([verbatim]))
    unshaped: JsonObject = {
        "operator": "equals",
        "path": "/final_state/last_run/exit_code",
        "value": 0,
    }
    with pytest.raises(ValueError, match="starts null"):
        _check_verifier_reads(draft([unshaped]))
    _check_verifier_reads(
        draft([{"operator": "equals", "path": "/final_state/status", "value": "green"}])
    )


def test_named_fields_keep_their_keys_and_file_maps_may_change() -> None:
    """Nested named fields are pinned; a files map keyed by paths may gain and lose files."""
    from exp.optimize.evaluation.variance.bootstrap_agents import shape_schema

    schema = shape_schema({"last_run": {"exit_code": None, "passed": 0}, "files": {"/a.py": "x"}})
    assert schema["required"] == ["files", "last_run"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    last_run = properties["last_run"]
    assert isinstance(last_run, dict) and last_run["required"] == ["exit_code", "passed"]
    assert properties["files"] == {"type": "object"}


def test_a_probed_file_stays_while_the_rest_of_its_files_map_may_change() -> None:
    """Run 30's simulator dropped a probed file while editing another; that is now refused."""
    from exp.optimize.evaluation.variance.bootstrap_agents import shape_schema
    from exp.runtime.environments.world_responses import validate_state_projection

    domain = {"files": {"/w/test.py": "t", "/w/a.py": "a"}}
    schema = shape_schema(domain, [["files", "/w/test.py"]])
    validate_state_projection(
        {"fs": {"files": {"/w/test.py": "t", "/w/b.py": "b"}}}, {"fs": schema}
    )
    import pytest

    with pytest.raises(ValueError, match="transition refused"):
        validate_state_projection({"fs": {"files": {"/w/a.py": "edited"}}}, {"fs": schema})


def test_a_positive_control_writes_a_file_as_the_files_map_holds_it() -> None:
    """The control builder once filled confcheck/__init__.py as nested objects, a shape no run
    produces; its witness now uses the map's flat key and refuses to nest inside a files map."""
    from exp.optimize.evaluation.variance.grading_controls import _set

    state: JsonObject = {
        "Workspace filesystem": {"files": {".teich-prompt.txt": "the task", "setup.py": "s"}}
    }
    probes = {
        "workspace_files": "/Workspace filesystem/files",
        "prompt_file": "/Workspace filesystem/files/.teich-prompt.txt",
    }
    verifier: JsonObject = {
        "description": "The package exists and the prompt is kept.",
        "outcome_schema": {},
        "required_paths": ["/final_state/workspace_files"],
        "rules": [
            {"operator": "exists", "path": "/final_state/workspace_files/confcheck/__init__.py"}
        ],
        "preserved_paths": ["/prompt_file"],
    }
    results = {row.control: row for row in run_controls(verifier, state, probes)}
    assert results["correct"].ok, results["correct"]
    files: JsonObject = {".teich-prompt.txt": "the task"}
    assert not _set({"files": files}, "/files/confcheck/__init__.py", "x")
    assert _set({"files": files}, "/files/confcheck~1__init__.py", "x")
    assert files["confcheck/__init__.py"] == "x"
    named: JsonObject = {"oracle": {"status": "red"}}
    assert _set(named, "/oracle/last_run/exit_code", 0)
    assert named == {"oracle": {"status": "red", "last_run": {"exit_code": 0}}}


def test_a_draft_reading_inside_a_files_map_off_its_key_style_is_refused() -> None:
    """A pointer normalisation cannot place in the map's key style refuses the draft."""
    from types import SimpleNamespace
    from typing import cast

    import pytest

    from exp.optimize.evaluation.variance.bootstrap_agents import _check_verifier_reads
    from exp.optimize.evaluation.variance.bootstrap_contracts import StartDraft
    from exp.simulation.evaluation.tasks import TaskVerifier

    state: JsonObject = {"fs": {"files": {"a.py": {"text": "x"}}}}
    probes = {"files": "/fs/files"}
    rule: JsonObject = {"operator": "exists", "path": "/final_state/files/pkg/b.py/text"}
    verifier = TaskVerifier.model_validate(
        {"description": "d", "outcome_schema": {}, "rules": [rule]}
    )
    stand_in = SimpleNamespace(initial_state=state, probe_paths=probes, verifier=verifier)
    with pytest.raises(ValueError, match="files map"):
        _check_verifier_reads(cast("StartDraft", stand_in))
