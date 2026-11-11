"""Verifier pointers authored in raw state coordinates are graded in probe coordinates."""

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.verifier_paths import project_verifier, projected, unplaced

STATE = {
    "Workspace filesystem": {"files": {"/workspace/test_a.py": "t", "/workspace/a.py": "a"}},
    "Python/Node runtime": {"interpreters": {"/opt/venv/bin/python": {"pytest": True}}},
}
PROBES = {"test_file": "/Workspace filesystem/files/~1workspace~1test_a.py"}


def test_a_raw_pointer_with_an_unescaped_key_moves_onto_its_covering_probe() -> None:
    """``files//workspace/test_a.py`` is the probed file; a preserved path drops its domain."""
    moved = projected(
        "/final_state/Workspace filesystem/files//workspace/test_a.py", STATE, dict(PROBES),
        rooted=True, add=False,
    )  # fmt: skip
    assert moved == "/final_state/test_file"
    assert projected("/files/~1workspace~1test_a.py", STATE, dict(PROBES), rooted=False, add=False)
    assert (
        projected("/files/~1workspace~1test_a.py", STATE, dict(PROBES), rooted=False, add=False)
        == "/test_file"
    )


def test_a_created_file_gets_a_probe_on_its_files_map_while_authoring() -> None:
    """A file the agent creates is observable through a probe added on the files map."""
    probes = dict(PROBES)
    verifier: JsonObject = {
        "required_paths": ["/final_state/Workspace filesystem/files//workspace/new.py"]
    }
    moved, notes = project_verifier(verifier, STATE, probes, add=True)
    assert probes["verifier_0"] == "/Workspace filesystem/files"
    assert moved["required_paths"] == ["/final_state/verifier_0/~1workspace~1new.py"]
    assert notes and unplaced(moved, probes) == []
    kept, _ = project_verifier(verifier, STATE, dict(PROBES), add=False)
    assert unplaced(kept, PROBES) == kept["required_paths"]


def test_pointers_already_in_probe_coordinates_and_other_roots_stay() -> None:
    """A probe-rooted pointer, the final answer and a nonsense path are not rewritten."""
    probes = dict(PROBES)
    for pointer in ("/final_state/test_file", "/final_answer", "/stop_reason"):
        assert projected(pointer, STATE, probes, rooted=True, add=True) == pointer
    assert projected("/tests/passed", STATE, probes, rooted=False, add=False) == "/tests/passed"
    assert unplaced({"required_paths": ["/tests/passed"]}, probes) == ["/tests/passed"]


FILES_STATE: JsonObject = {
    "Workspace filesystem": {"cwd": "/workspace", "files": {".teich-prompt.txt": "the task"}},
    "Tests": {"oracle": {}},
}
FILES_PROBES = {
    "workspace_files": "/Workspace filesystem/files",
    "prompt_file": "/Workspace filesystem/files/.teich-prompt.txt",
    "oracle": "/Tests/oracle",
}


def test_a_file_in_a_files_map_is_one_flat_key_never_nested_directories() -> None:
    """Run 31: rules read files/confcheck/__init__.py while the map held confcheck/__init__.py."""
    from exp.optimize.evaluation.variance.verifier_paths import keyed_by_paths, off_style

    assert keyed_by_paths({".teich-prompt.txt": "the task", "other_entries": [".pi"]})
    assert not keyed_by_paths({"oracle": {}}) and not keyed_by_paths({})
    nested = "/final_state/workspace_files/confcheck/__init__.py"
    flat = "/final_state/workspace_files/confcheck~1__init__.py"
    assert projected(nested, FILES_STATE, dict(FILES_PROBES), rooted=True, add=False) == flat
    assert projected(flat, FILES_STATE, dict(FILES_PROBES), rooted=True, add=False) == flat
    raw = "/final_state/Workspace filesystem/files/tests/test_rules.py"
    assert (
        projected(raw, FILES_STATE, dict(FILES_PROBES), rooted=True, add=False)
        == "/final_state/workspace_files/tests~1test_rules.py"
    )
    verifier: JsonObject = {
        "required_paths": [nested],
        "rules": [{"operator": "exists", "path": nested}],
        "preserved_paths": ["/prompt_file"],
    }
    moved, notes = project_verifier(verifier, FILES_STATE, dict(FILES_PROBES), add=False)
    assert moved["required_paths"] == [flat] and notes
    assert off_style(verifier, FILES_STATE, FILES_PROBES) == [nested, nested]
    assert off_style(moved, FILES_STATE, FILES_PROBES) == []


def test_a_pointer_the_state_cannot_place_in_its_key_style_is_left_for_refusal() -> None:
    """Object-valued file entries and empty objects do not say where a file key ends."""
    from exp.optimize.evaluation.variance.verifier_paths import map_style, off_style

    entries: JsonObject = {"a.py": {"text": "x", "mode": "644"}}
    assert map_style(entries, ["pkg", "b.py", "text"]) is None
    assert map_style(entries, ["a.py", "text"]) == ["a.py", "text"]
    assert map_style({}, ["last_run", "exit_code"]) is None
    assert map_style({"status": "red"}, ["last_run", "exit_code"]) == ["last_run", "exit_code"]
    assert map_style({}, ["passed"]) == ["passed"]
    deep: JsonObject = {"rules": [{"operator": "exists", "path": "/final_state/oracle/run/rc"}]}
    assert off_style(deep, FILES_STATE, FILES_PROBES) == ["/final_state/oracle/run/rc"]
    shallow: JsonObject = {"rules": [{"operator": "exists", "path": "/final_state/oracle/rc"}]}
    assert off_style(shallow, FILES_STATE, FILES_PROBES) == []


TASK_STATE: JsonObject = {
    "Workspace filesystem": {"files": {"engine.py": "e", "test_engine.py": "t"}},
    "Tests": {"test_run": {"status": "failed"}},
}
TASK_PROBES = {"files": "/Workspace filesystem/files", "tests": "/Tests"}


def test_a_garbled_file_key_is_refused_with_the_closest_grounded_names() -> None:
    """Run 32: engine~1py and test~1engine.py are well formed but name no file of the task."""
    from exp.optimize.evaluation.variance.verifier_paths import path_tokens, ungrounded

    verifier: JsonObject = {
        "required_paths": ["/final_state/files/engine~1py", "/final_state/files/engine.py"],
        "preserved_paths": ["/files/test~1engine.py"],
    }
    rows = ungrounded(verifier, TASK_STATE, TASK_PROBES, path_tokens("Fix engine.py."))
    assert len(rows) == 2
    assert "'engine/py'" in rows[0] and "'/final_state/files/engine.py'" in rows[0]
    assert "'test/engine.py'" in rows[1] and "'/files/test_engine.py'" in rows[1]


def test_initial_and_task_named_files_are_grounded() -> None:
    """An initial files-map key, a file the task asks to create and a /workspace/ path pass."""
    from exp.optimize.evaluation.variance.verifier_paths import path_tokens, ungrounded

    verifier: JsonObject = {
        "required_paths": ["/final_state/files/test_engine.py", "/final_state/files/cli.py"],
        "rules": [{"operator": "exists", "path": "/final_state/files/docs~1usage.md"}],
    }
    named = path_tokens("Create cli.py, then document it in /workspace/docs/usage.md.")
    assert ungrounded(verifier, TASK_STATE, TASK_PROBES, named) == []
    assert ungrounded(verifier, TASK_STATE, TASK_PROBES, path_tokens("Create xcli.py")) != []


def test_a_name_inside_a_longer_token_does_not_ground_a_file() -> None:
    """``engine.py`` in the text grounds engine.py, not a substring like ``gine.py``."""
    from exp.optimize.evaluation.variance.verifier_paths import path_tokens, ungrounded

    verifier: JsonObject = {"required_paths": ["/final_state/files/gine.py"]}
    assert ungrounded(verifier, TASK_STATE, TASK_PROBES, path_tokens("edit engine.py")) != []


def test_pointers_outside_a_files_map_are_not_checked() -> None:
    """Named fields, other workflow roots and unplaced pointers are left to other checks."""
    from exp.optimize.evaluation.variance.verifier_paths import ungrounded

    verifier: JsonObject = {
        "required_paths": ["/final_state/tests/test_run", "/final_answer", "/final_state/nope/x"],
        "rules": [{"operator": "equals", "path": "/final_state/tests/test_run/status"}],
        "preserved_paths": ["/files"],
    }
    assert ungrounded(verifier, TASK_STATE, TASK_PROBES, set()) == []
