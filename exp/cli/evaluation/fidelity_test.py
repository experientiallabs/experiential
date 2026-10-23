"""CLI replay completes interrupted memory publication without provider calls."""

from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    PreparedFidelity,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import saved_memory_receipt
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    predicted,
    trajectory,
)


def test_saved_cli_replay_finishes_memory_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interruption after report persistence is recovered without rebuying calls."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "publication-retry", "alpha", "learn")
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    wire = RoutedTransport(
        [
            predicted("publication-retry-0", "alpha-reply-0"),
            predicted("publication-retry-1", "alpha-reply-1"),
            judged(spec),
        ]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)

    def interrupted(store: ProjectStore, plan: PreparedFidelity, now: datetime) -> None:
        """Represent an interruption after the durable report, before publication."""
        raise RuntimeError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(fidelity, "consume_trajectory", interrupted)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            execute_fidelity_evaluation(project, prepared)
    assert saved_memory_receipt(project, prepared) is None
    calls = len(wire.requests), len(wire.decisions)
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    spec_path = tmp_path / "publication.json"
    spec_path.write_text(spec.model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "fidelity-test",
            "--root",
            str(tmp_path),
            "--fidelity",
            str(spec_path),
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["published"] is True
    assert (len(wire.requests), len(wire.decisions)) == calls
