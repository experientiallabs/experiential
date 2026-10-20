"""Executable package evidence stays private and the shared Harbor file inventory stays exact."""

import json
from pathlib import Path

import pytest

from exp.optimize.evaluation.executable_qualification import qualify_executable_task
from exp.optimize.evaluation.executable_qualification_test import qualified_fixture
from exp.optimize.evaluation.executable_task import export_executable_harbor
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.optimize.model.sft.training_test import _TIME


def test_executable_export_uses_common_client_without_private_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor image receives its prefix and interfaces, never the verifier or workbook store."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unqualified"):
        export_executable_harbor(project, task, tmp_path / "refused", name="test/workbook")
    qualification = qualify_executable_task(project, spec, created_at=_TIME, code_revision="test")
    output = export_executable_harbor(
        project, task, tmp_path / "package", name="test/workbook", qualification=qualification
    )
    binding = verify_harbor_export(output)
    assert binding["qualification"] == "qualified_observable_contract"
    context = json.loads((output / "environment/actor-context.json").read_text())
    assert context["actor_prefix"]
    assert "verifier" not in context and "world" not in context
    assert not tuple(output.rglob("*.xlsx"))
    assert "from a frozen checkpoint" in (output / "instruction.md").read_text()
    (output / "environment/actor-context.json").write_text("{}")
    with pytest.raises(ValueError, match="changed after export"):
        verify_harbor_export(output)
