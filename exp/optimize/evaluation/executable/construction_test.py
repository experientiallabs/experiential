"""Only explicitly pinned adapter code may turn asset references into runtime bindings."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.optimize.evaluation.executable.construction import bind_construction
from exp.optimize.evaluation.factory.scenario_search_test import workbook_search_fixture


def test_workbook_binding_changes_only_assets_and_probe_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """World generation cannot widen tool ownership, output paths or event policy."""
    project, search, _ = workbook_search_fixture(tmp_path, monkeypatch)
    first, second = search.candidates
    capture = first.scenario.capture
    assets = second.scenario.capture.sources[0]
    binding = bind_construction(project, capture, assets)
    assert binding.configuration == {
        **capture.configuration,
        "input_artifact": assets.model_dump(mode="json"),
    }
    assert binding.probe_sha256 == second.scenario.capture.probe_sha256
    assert binding.probe_sha256 != capture.probe_sha256


def test_runtime_binding_is_not_workbook_specific_and_checks_source_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted JSON-only binder uses the same seam; changed code is refused before import."""
    project, search, _ = workbook_search_fixture(tmp_path, monkeypatch)
    source = tmp_path / "json_runtime.py"
    source.write_text("""
from exp.optimize.evaluation.executable.construction import ConstructionBinding
from exp.common.core.artifacts import sha256_json
def bind_construction(project, configuration, assets):
    data = project.artifacts.read_bytes(assets.artifact_id, "assets/tables.json")
    return ConstructionBinding(
        configuration={"tables": data.decode()}, probe_sha256=sha256_json(configuration),
    )
""")
    capture = search.candidates[0].scenario.capture.model_copy(
        update={
            "runtime_file": source,
            "runtime_sha256": sha256_bytes(source.read_bytes()),
        }
    )
    assets = search.candidates[1].scenario.capture.sources[0]
    binding = bind_construction(project, capture, assets)
    assert (
        binding.configuration["tables"]
        == project.artifacts.read_bytes(assets.artifact_id, "assets/tables.json").decode()
    )
    source.write_text("raise AssertionError('must not execute unpinned code')")
    with pytest.raises(ValueError, match="runtime source changed"):
        bind_construction(project, capture, assets)
