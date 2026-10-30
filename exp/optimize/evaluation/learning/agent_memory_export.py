"""Agent memory as a markdown folder a future agent reads before it starts.

``MEMORY.md`` explains where the lessons come from and indexes one file per lesson kind; each
kind's file lists its accepted lessons, each with when it applies, its evidence and how many
earlier attempts it came from. Screened-out and unknown lessons are never exported: they stay in
the saved memory artifact with their citations and screening.
"""

from pathlib import Path

from exp.common.core.files import write_text_atomic
from exp.optimize.evaluation.learning.agent_memory_contracts import AgentMemory, ScreenedLesson
from exp.optimize.evaluation.learning.agent_memory_requests import KINDS, LessonKind

FILES: dict[LessonKind, tuple[str, str]] = {
    "task-contract": ("task-contracts.md", "Task contracts"),
    "user-preference": ("user-preferences.md", "User preferences"),
    "environment-rule": ("environment-rules.md", "Environment rules"),
    "procedure": ("procedures.md", "Procedures"),
    "pitfall": ("pitfalls.md", "Pitfalls"),
}


def export_agent_memory(memory: AgentMemory, output: Path) -> tuple[Path, ...]:
    """Write accepted lessons as ``MEMORY.md`` plus one markdown file per kind.

    Writing is idempotent: a folder that already holds exactly these files is left as is.

    Args:
        memory: Saved agent memory.
        output: Memory folder; absent, empty, or already holding exactly this memory.

    Returns:
        The written files, ``MEMORY.md`` first.

    Raises:
        ValueError: The folder holds anything else.
    """
    files = render_agent_memory(memory)
    written = tuple(output / name for name in files)
    present = sorted(output.rglob("*")) if output.exists() else []
    if present:
        names = {path.relative_to(output).as_posix() for path in present}
        if names != set(files) or any(
            not path.is_file() or path.read_text() != files[path.relative_to(output).as_posix()]
            for path in present
        ):
            raise ValueError(
                f"{output} already holds other files; choose an empty or new memory folder"
            )
        return written
    output.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        write_text_atomic(output / name, text)
    return written


def render_agent_memory(memory: AgentMemory) -> dict[str, str]:
    """Render accepted lessons as markdown files keyed by name, ``MEMORY.md`` first.

    Args:
        memory: Saved agent memory.

    Returns:
        File names to their text; a kind without accepted lessons has no file.
    """
    accepted = [lesson for lesson in memory.lessons if lesson.accepted]
    counts = memory.outcomes
    index = [
        "# Agent memory",
        "",
        f"Lessons learned from {sum(counts.values())} earlier attempts at similar tasks "
        f"({counts['PASS']} passed, {counts['FAIL']} failed, {counts['unknown']} with unknown "
        "outcome). Each lesson says when it applies and how many attempts it came from. Read "
        "the file for the kind of work before you start, and prefer your own observations "
        "when the environment contradicts a lesson.",
        "",
        "## Files",
        "",
    ]
    files: dict[str, str] = {}
    for kind in KINDS:
        rows = [lesson for lesson in accepted if lesson.kind == kind]
        if not rows:
            continue
        name, title = FILES[kind]
        index.append(f"- [{title}]({name}): {len(rows)} lessons")
        files[name] = "\n".join(
            [f"# {title}", "", *(line for row in rows for line in _lesson_lines(row)), ""]
        )
    if not files:
        index.append("No lesson passed screening.")
    top = sorted(accepted, key=lambda row: (-(row.score or 0) * len(row.traces), row.statement))
    if top:
        index.extend(["", "## Most supported lessons", ""])
        index.extend(f"- {row.statement} (when: {row.applies_when})" for row in top[:15])
    return {"MEMORY.md": "\n".join([*index, ""]), **files}


def _lesson_lines(lesson: ScreenedLesson) -> list[str]:
    """One lesson's markdown block."""
    return [
        f"## {lesson.statement}",
        "",
        f"- Applies when: {lesson.applies_when}",
        f"- Evidence: {' '.join(lesson.evidence[:2])}",
        f"- Learned from {len(lesson.traces)} earlier attempts.",
        "",
    ]
