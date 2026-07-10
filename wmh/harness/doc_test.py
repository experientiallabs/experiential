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


def test_runtime_backend_selector() -> None:
    """Default backend is local (in-process); unknown backends are rejected."""
    import pytest

    from wmh.harness.code_runtime import CodeRuntime
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline
    from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, system: str, messages: list[Message], **k) -> Completion:  # noqa: ANN003
            raise NotImplementedError

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    from typing import cast

    from wmh.providers.base import Provider

    provider = cast("Provider", _P())

    # Default backend is local; a code:runtime doc runs in-process.
    coded = code_baseline("seed")
    assert isinstance(coded.runtime(provider), CodeRuntime)
    assert coded.surface(CODE_RUNTIME_ID) is not None

    # Unknown backends are rejected.
    with pytest.raises(ValueError, match="unknown backend"):
        coded.runtime(provider, backend="bogus")


def _pi_doc() -> HarnessDoc:
    from wmh.harness.doc import RUNTIME_KIND_ID

    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id="code:a", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"),
        ],
    )


def _stub_provider():  # noqa: ANN202 - returns the casted Provider protocol below
    from typing import cast

    from wmh.providers.base import Completion, Message, Provider, ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, system: str, messages: list[Message], **k) -> Completion:  # noqa: ANN003
            raise NotImplementedError

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    return cast("Provider", _P())


def test_runtime_e2b_backend_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend='e2b' on a pi-node doc: E2BPiRuntime, with the template flag beating the env."""
    from wmh.harness.pi_e2b import E2BPiRuntime, E2BSandboxPool

    provider = _stub_provider()
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "env-tmpl")
    # An explicit template (the --e2b-template flag) beats the env var: it pins the runtime's
    # private pool (env-var resolution is bootstrap-time, covered in pi_e2b_test).
    explicit = _pi_doc().runtime(provider, backend="e2b", e2b_template="flag-tmpl")
    assert isinstance(explicit, E2BPiRuntime)
    assert explicit._pool._template == "flag-tmpl"  # noqa: SLF001 - pins the flag > env precedence
    # A shared pool (a whole search's) is used as-is; the runtime never builds a private one.
    shared_pool = E2BSandboxPool()
    shared = _pi_doc().runtime(provider, backend="e2b", e2b_pool=shared_pool)
    assert isinstance(shared, E2BPiRuntime)
    assert shared._pool is shared_pool  # noqa: SLF001 - pins the pool passthrough


def test_runtime_e2b_backend_rejects_in_process_runtime_kinds() -> None:
    """backend='e2b' only moves a pi-node harness PROCESS; in-process kinds must raise."""
    from wmh.harness.doc import code_baseline

    provider = _stub_provider()
    with pytest.raises(ValueError, match="use backend='local'"):
        HarnessDoc.baseline("b").runtime(provider, backend="e2b")
    with pytest.raises(ValueError, match="use backend='local'"):
        code_baseline("c").runtime(provider, backend="e2b")
