"""Initial captures retain native bytes, exact chat instructions and unknown startup outcomes."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.capture import (
    CaptureDefinition,
    capture_initial_world,
    preflight_capture,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint


def capture_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, CaptureDefinition, ScriptedJsonTransport]:
    """Use a real XLSX-backed factory, with no actor dispatch during capture."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    checkpoint = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    definition = CaptureDefinition(
        task=checkpoint.task,
        sources=(job.study.checkpoint,),
        runtime_file=job.runtime_file,
        runtime_sha256=job.runtime_sha256,
        configuration=job.configuration,
        probe_sha256=job.study.probe_sha256,
        maximum_model_calls=job.maximum_model_calls,
        system_prompt="Operate only the declared workbook tools.",
    )
    return project, definition, transport


def test_initial_capture_replays_exact_world_without_loading_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture neither buys an actor draw nor changes the input workbook on replay."""
    project, definition, transport = capture_fixture(tmp_path, monkeypatch)
    reference = capture_initial_world(project, definition, code_revision="test")
    result = load_continuation_checkpoint(project.artifacts, reference)
    original = load_continuation_checkpoint(project.artifacts, definition.sources[0])
    assert result.world == original.world
    actor = ChatAgentRuntime(
        maximum_model_calls=definition.maximum_model_calls, system_prompt=definition.system_prompt
    )
    assert result.actor == actor.initial_checkpoint(result.task)
    assert actor.remaining_calls(result.actor) == definition.maximum_model_calls
    assert result.actor.messages[0].content == definition.system_prompt
    definition.runtime_file.unlink()
    assert capture_initial_world(project, definition, code_revision="test") == reference
    assert not transport.requests


def test_failed_factory_is_not_silently_initialized_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory may have changed external state before failing, so its intent blocks replay."""
    project, definition, transport = capture_fixture(tmp_path, monkeypatch)
    marker = tmp_path / "factory-invocations"
    definition.runtime_file.write_text(
        f"from pathlib import Path\np = Path({str(marker)!r})\n"
        "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
        "raise RuntimeError('startup failed after side effect')\n"
    )
    definition = definition.model_copy(
        update={"runtime_sha256": sha256_bytes(definition.runtime_file.read_bytes())}
    )
    preflight_capture(project, definition)
    assert not marker.exists()
    with pytest.raises(RuntimeError, match="startup failed"):
        capture_initial_world(project, definition, code_revision="test")
    with pytest.raises(ValueError, match="unfinished startup intent"):
        capture_initial_world(project, definition, code_revision="test")
    assert marker.read_text() == "x" and not transport.requests


def test_capture_rejects_held_out_before_factory_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untouched frozen source cannot make a held-out task available for development."""
    project, definition, transport = capture_fixture(tmp_path, monkeypatch)
    definition = definition.model_copy(
        update={"task": definition.task.model_copy(update={"partition": "held_out"})}
    )
    with pytest.raises(ValueError, match="requires fit"):
        capture_initial_world(project, definition, code_revision="test")
    assert not transport.requests
