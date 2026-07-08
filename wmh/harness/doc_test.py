"""Tests for HarnessDoc: surface validation, hashing, derived views, document invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.harness.doc import (
    MAX_TURNS_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.skills import Skill


def _skill_surface(name: str = "count-words") -> Surface:
    skill = Skill(name=name, description="count words in a file", body="wc -w <path>")
    return Surface(id=f"skill:{name}", kind=SurfaceKind.SKILL, content=skill.to_markdown())


def test_baseline_is_valid_and_derives_defaults() -> None:
    doc = HarnessDoc.baseline("base")
    assert doc.max_turns() == 20
    assert doc.temperature() == 0.7
    assert "submit" in doc.tools()
    assert doc.system_prompt()  # non-empty
    assert doc.version == 0  # unsaved


def test_surface_id_must_match_kind() -> None:
    with pytest.raises(ValidationError, match="matching its kind"):
        Surface(id="skill:core", kind=SurfaceKind.PROMPT, content="x")
    with pytest.raises(ValidationError, match="matching its kind"):
        Surface(id="no-prefix", kind=SurfaceKind.PROMPT, content="x")
    with pytest.raises(ValidationError, match="matching its kind"):
        Surface(id="prompt:Not Kebab", kind=SurfaceKind.PROMPT, content="x")


def test_budget_is_enforced_at_construction() -> None:
    Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="ok", budget=10)
    with pytest.raises(ValidationError, match="over its budget"):
        Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="x" * 11, budget=10)


def test_doc_hash_is_content_and_order_independent() -> None:
    a = Surface(id="prompt:a", kind=SurfaceKind.PROMPT, content="A")
    b = Surface(id="prompt:b", kind=SurfaceKind.PROMPT, content="B")
    doc1 = HarnessDoc(name="x", surfaces=[a, b])
    doc2 = HarnessDoc(name="y", version=7, surfaces=[b, a])  # different name/version/order
    assert doc1.doc_hash == doc2.doc_hash  # identity is the surfaces, nothing else
    doc3 = HarnessDoc(name="x", surfaces=[a.model_copy(update={"content": "A2"}), b])
    assert doc3.doc_hash != doc1.doc_hash


def test_duplicate_surface_ids_rejected() -> None:
    a = Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="A")
    with pytest.raises(ValidationError, match="duplicate surface id"):
        HarnessDoc(name="x", surfaces=[a, a.model_copy(update={"content": "B"})])


def test_document_requires_a_prompt_surface() -> None:
    tools = Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit")
    with pytest.raises(ValidationError, match="prompt surface"):
        HarnessDoc(name="x", surfaces=[tools])


def test_invalid_derived_values_fail_at_construction() -> None:
    core = Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p")
    # tool policy without the required submit tool
    bad_tools = Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash")
    with pytest.raises(ValidationError, match="submit"):
        HarnessDoc(name="x", surfaces=[core, bad_tools])
    bad_turns = Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content="zero")
    with pytest.raises(ValidationError, match="integer"):
        HarnessDoc(name="x", surfaces=[core, bad_turns])
    bad_temp = Surface(id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content="9.5")
    with pytest.raises(ValidationError, match=r"\[0, 2\]"):
        HarnessDoc(name="x", surfaces=[core, bad_temp])


def test_skill_surface_slug_must_match_frontmatter() -> None:
    core = Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p")
    skill = Skill(name="other-name", description="d", body="b")
    mismatched = Surface(id="skill:my-skill", kind=SurfaceKind.SKILL, content=skill.to_markdown())
    with pytest.raises(ValidationError, match="must match"):
        HarnessDoc(name="x", surfaces=[core, mismatched])


def test_prompt_sections_join_in_id_order() -> None:
    doc = HarnessDoc(
        name="x",
        surfaces=[
            Surface(id="prompt:z-recovery", kind=SurfaceKind.PROMPT, content="RECOVER"),
            Surface(id="prompt:a-role", kind=SurfaceKind.PROMPT, content="ROLE"),
        ],
    )
    assert doc.system_prompt() == "ROLE\n\nRECOVER"


def test_runtime_reflects_document() -> None:
    doc = HarnessDoc.baseline("base")
    doc = HarnessDoc(name="base", surfaces=[*doc.surfaces, _skill_surface()])
    skills = doc.skills()
    assert [s.name for s in skills] == ["count-words"]
    assert doc.surface_hashes()["skill:count-words"]


def test_json_roundtrip_preserves_identity() -> None:
    doc = HarnessDoc(name="x", surfaces=[*HarnessDoc.baseline("x").surfaces, _skill_surface()])
    restored = HarnessDoc.model_validate_json(doc.model_dump_json())
    assert restored == doc
    assert restored.doc_hash == doc.doc_hash
