"""``retire-checkpoint`` and ``collect-garbage`` through ``exp eval --continuations``."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.optimize.evaluation.state_retention import (
    RETIREMENT_TYPE,
    CheckpointRetirement,
    StateCollection,
)
from exp.optimize.evaluation.state_retention_test import _checkpointed, _save_referrer
from exp.runtime.sessions.errors import StateUnavailableError


def _run(tmp_path: Path, command: ContinuationCommand, *flags: str, exit_code: int = 0) -> str:
    """Run one continuation action on the fixture project and return its flattened output."""
    spec = tmp_path / (command.action + ".json")
    spec.write_text(command.model_dump_json())
    result = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--continuations", str(spec), *flags],
    )
    assert result.exit_code == exit_code, result.output
    return " ".join(result.output.replace("│", " ").split())


def test_operator_previews_retires_and_collects_checkpoint_history(tmp_path: Path) -> None:
    """Dry runs change nothing; retirement releases the pin; collection reclaims and reports."""
    project, store, checkpoint = _checkpointed(tmp_path)
    collect = ContinuationCommand(action="collect-garbage", collection=StateCollection())
    retire = ContinuationCommand(
        action="retire-checkpoint",
        retirement=CheckpointRetirement(checkpoint=checkpoint, reason="superseded"),
    )
    preview = _run(tmp_path, collect, "--dry-run")
    assert "Would reclaim" in preview and "1 of them belong to checkpoints nothing" in preview
    inventory = json.loads(preview[preview.index("{") :])
    assert inventory["pins"][0]["pin"]["owner_id"] == checkpoint.artifact_id
    assert inventory["pins"][0]["owner"] == "retirable"
    assert not (project.paths.runtime_directory / "state-retention").exists()
    assert "Retirable (dry run, not recorded): would release 1 pin(s)" in _run(
        tmp_path, retire, "--dry-run"
    )
    assert len(store.pins()) == 1 and not project.artifacts.list_ids(artifact_type=RETIREMENT_TYPE)
    referrer = _save_referrer(project, checkpoint, as_input=True)
    refused = _run(tmp_path, retire, exit_code=2)
    assert "still referenced by 1 saved artifact" in refused and referrer in refused
    assert len(store.pins()) == 1
    assert store.read_state(store.ref_at("source", 1), ("business_state", "count")) == 1


def test_a_retired_checkpoint_history_is_reclaimed_by_the_next_collection(
    tmp_path: Path,
) -> None:
    """After retirement the collection reclaims the history and saves its inventory."""
    project, store, checkpoint = _checkpointed(tmp_path)
    retire = ContinuationCommand(
        action="retire-checkpoint",
        retirement=CheckpointRetirement(checkpoint=checkpoint, reason="superseded"),
    )
    assert "Retired: released 1 pin(s)" in _run(tmp_path, retire)
    collected = _run(
        tmp_path, ContinuationCommand(action="collect-garbage", collection=StateCollection())
    )
    assert "Reclaimed" in collected and "0 pin(s) alone keep 0 node(s)" in collected
    assert "Report: " in collected
    (report,) = (project.paths.runtime_directory / "state-retention").glob("*/report.txt")
    saved = json.loads(report.read_text())
    assert saved["pins"] == [] and saved["stores"][0]["report"]["reclaimed_nodes"] > 0
    with pytest.raises(StateUnavailableError):
        store.read_state(store.ref_at("source", 1))
