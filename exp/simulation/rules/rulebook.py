"""Render a rule book as a directory of Markdown files, with a machine-readable copy beside it.

The rule store describes how an environment behaves and nothing else: no environment state is
ever written here. The layout is fixed so readers and diffs stay stable:

- ``README.md``: the index, with source count, rules by status, open questions and tool files.
- ``tools/<tool>.md``: one file per tool, holding its parsed contract, its rules grouped by
  kind with their evidence on both sides, and its open questions.
- ``environment.md``: invariants and questions that belong to no single tool.
- ``rules.json``: the whole book as JSON, written by :func:`write_rulebook`.

Rendering is deterministic: the same book always gives byte-identical files.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from exp.common.core.artifacts import JsonValue
from exp.simulation.rules.merge import KIND_ORDER, canonical_predicate
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook, RuleStatus, ToolContract

RULES_JSON = "rules.json"
"""File name of the machine-readable copy written beside the Markdown files."""

_STATUSES: tuple[RuleStatus, ...] = ("supported", "contradicted", "unknown")
_FILE_STEM_LIMIT = 80


def render_rulebook(book: RuleBook) -> dict[str, str]:
    """Render a rule book as Markdown files.

    Args:
        book: The rule book.

    Returns:
        File contents keyed by POSIX path relative to the rule store, in path order.
    """
    paths = tool_file_paths(_tool_names(book))
    files = {
        "README.md": _readme(book, paths),
        "environment.md": _environment(book),
    }
    for tool, path in paths.items():
        files[path] = _tool_file(book, tool)
    return dict(sorted(files.items()))


def rulebook_json(book: RuleBook) -> str:
    """Serialize a rule book as indented JSON with a trailing newline.

    Args:
        book: The rule book.

    Returns:
        JSON that ``RuleBook.model_validate_json`` reads back.
    """
    return book.model_dump_json(indent=2) + "\n"


def write_rulebook(book: RuleBook, directory: Path) -> tuple[Path, ...]:
    """Write the rendered rule store into a new directory.

    Every file is written into a staging directory beside the destination, which is then renamed
    into place, so a reader sees either no rule store or the complete one.

    Args:
        book: The rule book.
        directory: Destination directory; it must not exist yet. Its parent is created if needed.

    Returns:
        Paths of every written file, sorted.

    Raises:
        FileExistsError: ``directory`` already exists.
    """
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(
            f"rule store directory {directory} already exists. Choose a new directory; an "
            "existing rule store is never overwritten or merged into."
        )
    files = render_rulebook(book)
    files[RULES_JSON] = rulebook_json(book)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        for relative, text in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        staging.chmod(0o755)
        # A plain rename refuses a non-empty destination that appeared since the check above.
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(sorted(directory / relative for relative in files))


def tool_file_paths(tools: Sequence[str]) -> dict[str, str]:
    """Map tool names to safe relative Markdown paths under ``tools/``.

    Names are lowercased and reduced to ``[a-z0-9_.-]``. When several names reduce to the same
    file name, each of them gets a suffix from a digest of its exact name, so the mapping never
    depends on input order and never lets one tool overwrite another.

    Args:
        tools: Tool names.

    Returns:
        Paths keyed by tool name, sorted by tool name.
    """
    stems: dict[str, list[str]] = {}
    for tool in sorted(set(tools)):
        stems.setdefault(_file_stem(tool), []).append(tool)
    paths: dict[str, str] = {}
    for stem, names in stems.items():
        for name in names:
            if len(names) == 1:
                paths[name] = f"tools/{stem}.md"
            else:
                digest = hashlib.sha256(name.encode("utf-8"), usedforsecurity=False).hexdigest()
                paths[name] = f"tools/{stem}-{digest[:10]}.md"
    return dict(sorted(paths.items()))


def _file_stem(tool: str) -> str:
    """Filesystem-safe lowercase stem of a tool name."""
    stem = re.sub(r"[^a-z0-9_.-]+", "_", tool.lower()).strip("._-")
    return stem[:_FILE_STEM_LIMIT].strip("._-") or "tool"


def _tool_names(book: RuleBook) -> list[str]:
    """Every tool named by a contract, a rule or a question."""
    names = set(book.contracts)
    names.update(rule.tool for rule in book.rules if rule.tool is not None)
    names.update(question.tool for question in book.questions if question.tool is not None)
    return sorted(names)


def _code(text: str) -> str:
    """Inline code span that survives backticks inside the text."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    padded = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{padded}{fence}"


def _line(text: str) -> str:
    """Collapse free text to one Markdown line."""
    return " ".join(text.split())


def _locators(title: str, locators: Sequence[str]) -> list[str]:
    """A titled bullet list of locators, or a single line saying there are none."""
    if not locators:
        return [f"- {title}: none"]
    return [f"- {title}:", *(f"  - {_code(locator)}" for locator in locators)]


def _readme(book: RuleBook, paths: dict[str, str]) -> str:
    """The index file."""
    lines = [
        "# Rule book",
        "",
        "How the environment behaves, learned from traces. This store holds rules only, never",
        "environment state.",
        "",
        f"- Sources: {len(book.sources)} traces",
        f"- Rules: {len(book.rules)}",
        f"- Open questions: {len(book.questions)}",
        "",
        "## Rules by status",
        "",
    ]
    for status in _STATUSES:
        count = sum(1 for rule in book.rules if rule.status == status)
        lines.append(f"- {status}: {count}")
    lines += ["", "## Files", "", "- [environment.md](environment.md): tool-less invariants"]
    for tool, path in paths.items():
        count = sum(1 for rule in book.rules if rule.tool == tool)
        lines.append(f"- [{path}]({path}): {_code(tool)}, {count} rules")
    lines += ["", "## Open questions", ""]
    if not book.questions:
        lines.append("None.")
    for question in book.questions:
        where = "environment" if question.tool is None else _code(question.tool)
        lines.append(f"- {where}: {_line(question.question)}")
    return "\n".join(lines) + "\n"


def _environment(book: RuleBook) -> str:
    """The file for rules and questions that belong to no tool."""
    lines = ["# Environment", "", "Rules and questions that belong to no single tool.", ""]
    lines += _rules_section([rule for rule in book.rules if rule.tool is None])
    lines += _questions_section([question for question in book.questions if question.tool is None])
    return "\n".join(lines) + "\n"


def _tool_file(book: RuleBook, tool: str) -> str:
    """The file for one tool."""
    lines = [f"# Tool {_code(tool)}", ""]
    lines += _contract_section(book.contracts.get(tool))
    lines += _rules_section([rule for rule in book.rules if rule.tool == tool])
    lines += _questions_section([question for question in book.questions if question.tool == tool])
    return "\n".join(lines) + "\n"


def _contract_section(contract: ToolContract | None) -> list[str]:
    """Summary of a tool's parsed contract."""
    lines = ["## Contract", ""]
    if contract is None:
        return [*lines, "No contract was parsed for this tool.", ""]
    lines += [f"- Calls: {contract.calls}", f"- Failures: {contract.failures}"]
    keys = _schema_keys(contract.declared_schema)
    if keys is None:
        lines.append("- Declared schema: none")
    else:
        lines.append(f"- Declared schema keys: {', '.join(_code(key) for key in keys) or 'none'}")
    lines.append(
        "- Required arguments: "
        + (", ".join(_code(name) for name in contract.required_arguments) or "none")
    )
    if contract.argument_types:
        lines.append("- Argument types:")
        for name, types in sorted(contract.argument_types.items()):
            lines.append(f"  - {_code(name)}: {', '.join(types) or 'unknown'}")
    else:
        lines.append("- Argument types: none observed")
    lines += _locators("Result shapes", contract.result_shapes)
    lines += _locators("Error forms", contract.error_forms)
    lines += _locators("Examples", contract.examples)
    return [*lines, ""]


def _schema_keys(schema: dict[str, JsonValue] | None) -> list[str] | None:
    """Argument names a declared schema lists, or its top-level keys when it lists none."""
    if schema is None:
        return None
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return sorted(properties)
    return sorted(schema)


def _rules_section(rules: Sequence[Rule]) -> list[str]:
    """Rules grouped by kind."""
    lines = ["## Rules", ""]
    if not rules:
        return [*lines, "None.", ""]
    for kind in KIND_ORDER:
        members = [rule for rule in rules if rule.kind == kind]
        if not members:
            continue
        lines += [f"### {kind.capitalize()}", ""]
        for rule in members:
            lines += _rule_block(rule)
    return lines


def _rule_block(rule: Rule) -> list[str]:
    """One rule with its status, counts, predicate, evidence and witnesses."""
    lines = [
        f"#### {_code(rule.rule_id)}",
        "",
        _line(rule.statement),
        "",
        f"- Status: {rule.status}",
        f"- Support: {rule.support} trajectories",
        f"- Contradiction: {rule.contradiction} trajectories",
    ]
    lines += _locators("Evidence", rule.evidence)
    lines += _locators("Counter-evidence", rule.counter_evidence)
    if rule.witnesses:
        lines.append("- Witnesses:")
        for witness in rule.witnesses:
            verdict = "supports" if witness.supports else "refutes"
            lines.append(f"  - {witness.kind} {_code(witness.reference)}: {verdict}")
    else:
        lines.append("- Witnesses: none")
    if rule.weighing is not None:
        lines.append(f"- Weighing: {_line(rule.weighing.rationale)}")
        if rule.weighing.independence:
            lines.append(f"  - Independence: {_line(rule.weighing.independence)}")
    lines.append("")
    if rule.predicate is not None:
        text = canonical_predicate(rule.predicate)
        longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        lines += [f"{fence}json", text, fence, ""]
    return lines


def _questions_section(questions: Sequence[ResearchQuestion]) -> list[str]:
    """Open questions with why they matter and what raised them."""
    lines = ["## Open questions", ""]
    if not questions:
        return [*lines, "None.", ""]
    for question in questions:
        lines += [
            f"- {_line(question.question)}",
            f"  - Why it matters: {_line(question.why_it_matters)}",
        ]
        if question.evidence:
            lines.append("  - Evidence: " + ", ".join(_code(loc) for loc in question.evidence))
    return [*lines, ""]
