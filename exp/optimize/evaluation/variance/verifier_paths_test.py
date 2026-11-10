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
