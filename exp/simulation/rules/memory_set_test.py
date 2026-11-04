"""The rule memory set: small Markdown files, per-file digests and versioned edits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.simulation.rules.memory_set import (
    CONTRADICTIONS_MD,
    DIGESTS_JSON,
    INDEX_MD,
    SET_JSON,
    UNKNOWNS_MD,
    MemorySet,
    Passage,
    RuleFile,
    add_rule,
    load_memory_set,
    memory_set_digest,
    memory_set_files,
    remove_rule,
    replace_rule,
    rule_sha256,
    save_memory_set,
)


def passage(locator: str, text: str = '{"action": {"name": "refund"}}') -> Passage:
    """An original passage with coordinates."""
    trace = locator.split(":")[1]
    return Passage(
        locator=locator, trace_id=trace, step=1, quote=text, source_sha256=sha256_json(text)
    )


def rule(
    rule_id: str,
    tool: str | None,
    truism: str,
    *,
    status: str = "supported",
    counter: tuple[Passage, ...] = (),
    questions: tuple[str, ...] = (),
) -> RuleFile:
    """One rule file."""
    return RuleFile.model_validate(
        {
            "rule_id": rule_id,
            "topic": tool or "environment",
            "tool": tool,
            "kind": "precondition" if tool else "invariant",
            "truism": truism,
            "reasoning": f"Because the environment checks it before acting: {truism}",
            "evidence": [passage(f"trace:t1:span:{rule_id}").model_dump(mode="json")],
            "counter_evidence": [p.model_dump(mode="json") for p in counter],
            "status": status,
            "status_rationale": "Weighed from the cited refusals.",
            "applicability": "Whenever the tool is called.",
            "open_questions": list(questions),
        }
    )


def sample_memory() -> MemorySet:
    """Three rules over two tools plus the environment."""
    return MemorySet(
        tools=("lookup", "refund"),
        rules=(
            rule("refund.limit", "refund", "A refund above the paid amount is refused."),
            rule(
                "refund.partial",
                "refund",
                "A partial refund is accepted.",
                status="contradicted",
                counter=(passage("trace:t2:span:c9"),),
            ),
            rule(
                "env.orders",
                None,
                "Order ids are strings.",
                status="unknown",
                questions=("Are numeric ids accepted?",),
            ),
        ),
        sources=("t1", "t2"),
    )


def test_memory_set_is_many_small_files_with_per_file_digests() -> None:
    """Each rule is its own file; index, per-tool, contradictions and unknowns sit beside them."""
    memory = sample_memory()
    files = memory_set_files(memory)

    assert set(files) == {
        INDEX_MD,
        CONTRADICTIONS_MD,
        UNKNOWNS_MD,
        SET_JSON,
        DIGESTS_JSON,
        "rules/refund/refund.limit.md",
        "rules/refund/refund.partial.md",
        "rules/environment/env.orders.md",
        "tools/lookup.md",
        "tools/refund.md",
    }
    limit = files["rules/refund/refund.limit.md"].decode()
    for section in ("## Why it holds", "## Evidence", "## Status rationale", "## Applicability"):
        assert section in limit
    assert "`trace:t1:span:refund.limit` (trace `t1`, step 1)" in limit
    assert "refund.partial" not in limit
    assert "refund.partial" in files[CONTRADICTIONS_MD].decode()
    unknowns = files[UNKNOWNS_MD].decode()
    assert "Order ids are strings." in unknowns and "Are numeric ids accepted?" in unknowns
    assert "coverage gap" in files["tools/lookup.md"].decode()
    index = files[INDEX_MD].decode()
    assert index.count("| [") == 3
    digests = json.loads(files[DIGESTS_JSON])
    assert set(digests["files"]) == set(files) - {DIGESTS_JSON}
    for path, digest in digests["files"].items():
        assert digest == sha256_json(files[path].decode())
    assert digests["digest"] == memory_set_digest(memory)


def test_edits_are_small_versioned_and_change_the_set_digest() -> None:
    """A patch bumps one file's version; a stale base or a duplicate is refused."""
    memory = sample_memory()
    current = memory.rule("refund.limit")
    assert current is not None
    base = rule_sha256(current)
    edited = replace_rule(memory, base, current.model_copy(update={"truism": "Refunds cap."}))

    changed = edited.rule("refund.limit")
    assert changed is not None and changed.version == 2 and changed.truism == "Refunds cap."
    assert memory_set_digest(edited) != memory_set_digest(memory)
    before, after = memory_set_files(memory), memory_set_files(edited)
    differing = {path for path in before if before[path] != after.get(path)}
    assert differing == {
        "rules/refund/refund.limit.md",
        "tools/refund.md",
        INDEX_MD,
        SET_JSON,
        DIGESTS_JSON,
    }
    with pytest.raises(ValueError, match="changed since"):
        replace_rule(edited, base, current)
    with pytest.raises(ValueError, match="already exists"):
        add_rule(edited, current)
    removed = remove_rule(edited, rule_sha256(changed), "refund.limit")
    assert removed.rule("refund.limit") is None


def test_saved_set_loads_verified_and_refuses_tampering(tmp_path: Path) -> None:
    """A saved set round-trips; a reference with another digest is refused."""
    project = ProjectStore(tmp_path, "memory")
    memory = sample_memory()
    reference = save_memory_set(
        project.artifacts, memory, artifact_id="memory-v1", inputs=(), code_revision="test"
    )
    assert load_memory_set(project.artifacts, reference) == memory
    again = save_memory_set(
        project.artifacts, memory, artifact_id="memory-v1", inputs=(), code_revision="test"
    )
    assert again == reference
    with pytest.raises(ValueError, match="differs"):
        save_memory_set(
            project.artifacts,
            memory.model_copy(update={"revision": 9}),
            artifact_id="memory-v1",
            inputs=(),
            code_revision="test",
        )
    with pytest.raises(ValueError, match="manifest digest changed"):
        load_memory_set(project.artifacts, reference.model_copy(update={"sha256": "0" * 64}))
