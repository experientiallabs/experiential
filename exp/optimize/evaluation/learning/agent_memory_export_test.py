"""Agent memory as a markdown folder: accepted lessons only, idempotent, never over others."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.learning.agent_memory_contracts import AgentMemory, ScreenedLesson
from exp.optimize.evaluation.learning.agent_memory_export import (
    export_agent_memory,
    render_agent_memory,
)
from exp.optimize.evaluation.learning.agent_memory_requests import LessonKind, LessonScreening

REF = ArtifactInput(artifact_id="saved", sha256="0" * 64)


def screened(statement: str, kind: LessonKind, *, accepted: bool) -> ScreenedLesson:
    """One screened lesson cited by two traces."""
    return ScreenedLesson(
        kind=kind,
        statement=statement,
        applies_when="a task asks for it",
        evidence=("first note", "second note", "third note"),
        traces=("a", "b"),
        screening=LessonScreening(general=0.9, covered=0.0, supported=0.9),
        score=0.81 if accepted else 0.1,
        accepted=accepted,
    )


def memory(*lessons: ScreenedLesson) -> AgentMemory:
    """Saved-shape agent memory holding the given lessons."""
    return AgentMemory(
        schema_version=1,
        created_at=datetime(2026, 10, 24, tzinfo=UTC),
        code_revision="test",
        memory_id="memory",
        plan=REF,
        candidates=REF,
        outcomes={"PASS": 2, "FAIL": 1, "unknown": 0},
        lessons=lessons,
        failed=(),
    )


def test_only_accepted_lessons_are_exported_in_one_file_per_kind() -> None:
    """MEMORY.md indexes the kinds that have accepted lessons; screened-out ones stay saved."""
    files = render_agent_memory(
        memory(
            screened("Write to /out.", "task-contract", accepted=True),
            screened("Guess the format.", "task-contract", accepted=False),
            screened("Answer in bullets.", "user-preference", accepted=True),
        )
    )
    assert list(files) == ["MEMORY.md", "task-contracts.md", "user-preferences.md"]
    index = files["MEMORY.md"]
    assert "3 earlier attempts at similar tasks (2 passed, 1 failed, 0 with unknown" in index
    assert "- [Task contracts](task-contracts.md): 1 lessons" in index
    assert "Guess the format." not in "".join(files.values())
    contracts = files["task-contracts.md"]
    assert contracts.startswith("# Task contracts\n\n## Write to /out.\n")
    assert "- Evidence: first note second note\n" in contracts
    assert "Learned from 2 earlier attempts." in contracts


def test_memory_without_accepted_lessons_still_says_so() -> None:
    """A future agent always finds a nonempty MEMORY.md, even when nothing passed screening."""
    files = render_agent_memory(memory(screened("x", "pitfall", accepted=False)))
    assert list(files) == ["MEMORY.md"]
    assert "No lesson passed screening." in files["MEMORY.md"]


def test_export_is_idempotent_and_never_writes_over_other_files(tmp_path: Path) -> None:
    """The same memory rewrites nothing new; a folder holding anything else is refused."""
    saved = memory(screened("Write to /out.", "task-contract", accepted=True))
    output = tmp_path / "memory"
    written = export_agent_memory(saved, output)
    assert [path.name for path in written] == ["MEMORY.md", "task-contracts.md"]
    assert export_agent_memory(saved, output) == written
    (output / "notes.md").write_text("mine")
    with pytest.raises(ValueError, match="already holds other files"):
        export_agent_memory(saved, output)
    other = tmp_path / "other"
    other.mkdir()
    (other / "MEMORY.md").write_text("different")
    with pytest.raises(ValueError, match="already holds other files"):
        export_agent_memory(saved, other)
