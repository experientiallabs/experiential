"""``forget-summary`` checks eligibility on a dry run, records once and refuses what is unsafe."""

import json
from pathlib import Path

from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.optimize.evaluation.learning.forgetting import SummaryForget
from exp.optimize.evaluation.tests.memory_forget_test import LINEAGE, Compressor, Memory


def forget(tmp_path: Path, block: str, *flags: str, exit_code: int = 0) -> str:
    """Run one ``forget-summary`` action on the fixture project and return its output."""
    spec = tmp_path / "forget.json"
    spec.write_text(
        ContinuationCommand(
            action="forget-summary",
            forget=SummaryForget(lineage=LINEAGE, block=block, reason="summary is stale"),
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--continuations", str(spec), *flags],
    )
    assert result.exit_code == exit_code, result.output
    return " ".join(result.output.replace("│", " ").split())


def test_cli_forgets_a_built_summary_from_the_next_version(tmp_path: Path) -> None:
    """Dry run records nothing; the recorded forget replays; the next version applies it."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha claim")
    memory.learn("beta", "[unresolved] beta BAD-SUMMARY")
    assert "no built summary" in forget(tmp_path, "#2-3", exit_code=2)
    memory.maintain(Compressor())
    assert "aligned block" in forget(tmp_path, "#1-2", exit_code=2)
    dry = forget(tmp_path, "#2-3", "--dry-run")
    assert "Eligible (dry run, not recorded)" in dry
    assert not memory.store.list_ids(artifact_type="memory-forget")
    recorded = forget(tmp_path, "#2-3")
    assert "Recorded: forget #2-3 of memory forget-memory from version 3" in recorded
    decision = json.loads(recorded[recorded.index("{") :])
    assert decision["forgets"] == ["#2-3", "#0-3"] and decision["record"]
    assert forget(tmp_path, "#2-3") == recorded
    assert memory.learn("gamma", "[unresolved] gamma claim").forgotten == ("2-4", "0-4")
