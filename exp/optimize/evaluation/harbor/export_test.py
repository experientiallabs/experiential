"""Known excluded calibration ancestry cannot qualify a Harbor delivery."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.analysis.source_access_test import _source
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.harbor.export import (
    export_harbor_task,
    verify_harbor_export,
    write_harbor_package,
)


def test_export_refuses_excluded_judgment_before_task_content(tmp_path: Path) -> None:
    """Excluded calibration ancestors refuse even if a derived report claims qualification."""
    project, _ = source_project(tmp_path)
    judgment = _source(project.artifacts, "unsafe-judgment")
    calibration = _source(project.artifacts, "calibration", (judgment,))
    exclude_source_from_qualification(
        project.artifacts,
        judgment,
        audit=_source(project.artifacts, "audit"),
        rationale="Proven unsafe redraw",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        export_harbor_task(
            project,
            calibration,
            tmp_path / "export",
            name="test/excluded",
            calibration=calibration,
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("dialogue", [True, False])
def test_instruction_routes_customer_messages_only_when_backend_records_them(
    tmp_path: Path, dialogue: bool
) -> None:
    """A dialogue backend's instruction names ``say``; a backend without a ledger never does."""
    package = write_harbor_package(
        tmp_path / "package",
        name="test/dialogue",
        instruction="Help the customer.",
        public_context={},
        binding={"backend": "test"},
        qualified=False,
        notice="Fixture backend.",
        dialogue=dialogue,
    )
    instruction = (package / "instruction.md").read_text()
    assert ("python /app/tools.py say 'TEXT'" in instruction) is dialogue
    assert "python /app/tools.py submit" in instruction
    assert verify_harbor_export(package)["backend"] == "test"


def test_a_dense_reward_definition_is_part_of_the_package_identity(tmp_path: Path) -> None:
    """The definition digest is bound and declared; its tags never enter the actor files."""
    package = write_harbor_package(
        tmp_path / "package",
        name="test/dense",
        instruction="Fix the parser.",
        public_context={},
        binding={"backend": "test"},
        qualified=False,
        notice="Fixture backend.",
        dense_reward="f" * 64,
    )
    binding = verify_harbor_export(package)
    assert binding["dense_reward"] == {
        "schema_version": "dense-reward-v1",
        "definition_sha256": "f" * 64,
    }
    assert 'dense_reward = "dense-reward-v1"' in (package / "task.toml").read_text()
    assert "modified_evaluation" not in (package / "environment/actor-context.json").read_text()
