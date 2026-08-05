"""Tests for HarnessDoc: surface validation, hashing, derived views, document invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.optimize.harness import doc as doc_module
from wmo.optimize.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    MAX_TURNS_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
    code_surface_id,
)
from wmo.optimize.harness.runtime import JSON_PROTOCOL_CLAUSE
from wmo.optimize.harness.skills import Skill


def _skill_surface(name: str = "count-words") -> Surface:
    skill = Skill(name=name, description="count words in a file", body="wc -w <path>")
    return Surface(id=f"skill:{name}", kind=SurfaceKind.SKILL, content=skill.to_markdown())


def test_baseline_is_valid_and_derives_defaults() -> None:
    doc = HarnessDoc.baseline("base")
    assert doc.max_turns() == 20
    assert doc.max_output_tokens() == 4096
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


def test_trailing_newline_never_slips_past_id_or_path_validation() -> None:
    # `$` in re.match also matches before a final newline; the validators must use fullmatch
    # or a newline-carrying surface passes here and fails only at render/reparse time.
    with pytest.raises(ValidationError, match="matching its kind"):
        Surface(id="prompt:core\n", kind=SurfaceKind.PROMPT, content="x")
    with pytest.raises(ValidationError, match="unsafe path"):
        Surface(
            id=code_surface_id("src/agent.ts"),
            kind=SurfaceKind.CODE,
            content="x",
            path="src/agent.ts\n",
        )


def test_budget_is_enforced_at_construction() -> None:
    Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="ok", budget=10)
    with pytest.raises(ValidationError, match="over its budget"):
        Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="x" * 11, budget=10)


@pytest.mark.parametrize("content", ["before\x00after", "before\ud800after"])
def test_surface_rejects_text_that_cannot_roundtrip_through_durable_storage(
    content: str,
) -> None:
    with pytest.raises(ValidationError, match="NUL|surrogate"):
        Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content=content)


def test_doc_hash_is_content_and_order_independent() -> None:
    a = Surface(id="prompt:a", kind=SurfaceKind.PROMPT, content="A")
    b = Surface(id="prompt:b", kind=SurfaceKind.PROMPT, content="B")
    doc1 = HarnessDoc(name="x", surfaces=[a, b])
    doc2 = HarnessDoc(name="y", version=7, surfaces=[b, a])  # different name/version/order
    assert doc1.doc_hash == doc2.doc_hash  # identity is the surfaces, nothing else
    doc3 = HarnessDoc(name="x", surfaces=[a.model_copy(update={"content": "A2"}), b])
    assert doc3.doc_hash != doc1.doc_hash


def test_doc_hash_includes_materialized_paths_and_ignores_display_metadata() -> None:
    # "src/worker.ts" and "src.worker.ts" both map (lossily) to id "code:src-worker-ts", so the
    # id and content alone cannot tell these two materializations apart; the hash must.
    first = HarnessDoc(
        name="first",
        version=1,
        surfaces=[
            Surface(id="prompt:main", kind=SurfaceKind.PROMPT, content="prompt"),
            Surface(
                id="code:src-worker-ts",
                kind=SurfaceKind.CODE,
                content="export const value = 1;",
                path="src/worker.ts",
            ),
        ],
    )
    renamed = first.model_copy(update={"name": "renamed", "version": 99})
    moved = HarnessDoc(
        name="moved",
        surfaces=[
            Surface(id="prompt:main", kind=SurfaceKind.PROMPT, content="prompt"),
            Surface(
                id="code:src-worker-ts",
                kind=SurfaceKind.CODE,
                content="export const value = 1;",
                path="src.worker.ts",
            ),
        ],
    )

    assert renamed.doc_hash == first.doc_hash
    assert moved.doc_hash != first.doc_hash
    # The legacy identity (pull compatibility only) predates path coverage: it cannot tell the
    # moved document apart, which is exactly why doc_hash now includes paths.
    assert first.legacy_doc_hash == moved.legacy_doc_hash
    assert first.legacy_doc_hash != first.doc_hash


def test_pathless_doc_hash_matches_its_legacy_hash() -> None:
    doc = HarnessDoc.baseline("base")
    assert doc.doc_hash == doc.legacy_doc_hash


def test_code_surface_path_must_derive_its_id() -> None:
    Surface(id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// ok")
    with pytest.raises(ValidationError, match="must use id 'code:src-agent-ts'"):
        Surface(id="code:other", kind=SurfaceKind.CODE, path="src/agent.ts", content="// no")


@pytest.mark.parametrize("path", ["src/agent_utils.ts", "Upper.ts", "src/a..b.ts"])
def test_code_surface_rejects_paths_outside_the_id_grammar(path: str) -> None:
    with pytest.raises(ValidationError, match="kebab-slug"):
        Surface(id="code:whatever", kind=SurfaceKind.CODE, path=path, content="x")


@pytest.mark.parametrize("path", ["./x.ts", "src/./x.ts", "src//x.ts", "/abs.ts", "a/../b.ts"])
def test_code_surface_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="canonical relative POSIX path"):
        Surface(id="code:x-ts", kind=SurfaceKind.CODE, path=path, content="x")


@pytest.mark.parametrize("path", ["doc.json", "aliases.toml"])
def test_code_surface_rejects_store_metadata_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="store metadata"):
        Surface(id=code_surface_id(path), kind=SurfaceKind.CODE, path=path, content="x")


def test_code_runtime_surface_must_be_pathless() -> None:
    # A file named exactly `runtime` would mint a pathful code:runtime that hijacks the
    # in-process runtime module and collides with the rendered runtime.py on reparse.
    with pytest.raises(ValidationError, match="must not carry a path"):
        Surface(id="code:runtime", kind=SurfaceKind.CODE, path="runtime", content="x")


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
    bad_output = Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="many")
    with pytest.raises(ValidationError, match="integer"):
        HarnessDoc(name="x", surfaces=[core, bad_output])
    zero_output = Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="0")
    with pytest.raises(ValidationError, match=">= 1"):
        HarnessDoc(name="x", surfaces=[core, zero_output])
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
    assembled = doc.assembled_prompt()
    assert "read_skill:" in assembled
    assert "wc -w <path>" not in assembled  # bodies remain progressive-disclosure only


def test_json_roundtrip_preserves_identity() -> None:
    doc = HarnessDoc(name="x", surfaces=[*HarnessDoc.baseline("x").surfaces, _skill_surface()])
    restored = HarnessDoc.model_validate_json(doc.model_dump_json())
    assert restored == doc
    assert restored.doc_hash == doc.doc_hash


def test_runtime_backend_selector() -> None:
    """Default backend is local (in-process); unknown backends are rejected."""
    import pytest

    from wmo.optimize.harness.code_runtime import CodeRuntime
    from wmo.optimize.harness.doc import CODE_RUNTIME_ID, code_baseline
    from wmo.providers.base import Completion, Message, ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, system: str, messages: list[Message], **k) -> Completion:  # noqa: ANN003
            raise NotImplementedError

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    from typing import cast

    from wmo.providers.base import Provider

    provider = cast("Provider", _P())

    # Default backend is local; a code:runtime doc runs in-process.
    coded = code_baseline("seed")
    assert isinstance(coded.runtime(provider), CodeRuntime)
    assert coded.surface(CODE_RUNTIME_ID) is not None

    # Unknown backends are rejected.
    with pytest.raises(ValueError, match="unknown backend"):
        coded.runtime(provider, backend="bogus")


def _pi_doc() -> HarnessDoc:
    from wmo.optimize.harness.doc import RUNTIME_KIND_ID

    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content="7"),
            Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="16384"),
            Surface(
                id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"
            ),
        ],
    )


def _stub_provider():  # noqa: ANN202 - returns the casted Provider protocol below
    from typing import cast

    from wmo.providers.base import Completion, Message, Provider, ProviderConfig, ProviderKind
    from wmo.utils.waterfall import ChatRequest, ChatResponse

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, system: str, messages: list[Message], **k) -> Completion:  # noqa: ANN003
            raise NotImplementedError

        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            del request
            return ChatResponse.model_validate(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    return cast("Provider", _P())


def test_runtime_e2b_backend_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend='e2b' on a pi-node doc: E2BPiRuntime, with the template flag beating the env."""
    from wmo.optimize.harness.pi_e2b import E2BPiRuntime, E2BSandboxPool

    provider = _stub_provider()
    monkeypatch.setenv("WMO_E2B_TEMPLATE", "env-tmpl")
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
    assert shared._max_turns == 7  # noqa: SLF001 - document parameter reaches the runner
    assert shared._max_output_tokens == 16384  # noqa: SLF001 - same agent model contract


def test_pi_e2b_runtime_inherits_temperature_and_skill_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wmo.optimize.harness.pi_e2b import E2BPiRuntime, E2BSandboxPool

    base = _pi_doc()
    doc = HarnessDoc(
        name="pi-with-skill",
        surfaces=[
            *base.surfaces,
            Surface(id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content="0.35"),
            _skill_surface(),
        ],
    )
    monkeypatch.delenv("WMO_E2B_TEMPLATE", raising=False)
    pool = E2BSandboxPool()
    runtime = doc.runtime(_stub_provider(), backend="e2b", e2b_pool=pool)

    assert isinstance(runtime, E2BPiRuntime)
    assert runtime._temperature == 0.35  # noqa: SLF001 - doc parameter reaches worker boundary
    assert runtime._skills.get("count-words") is not None  # noqa: SLF001 - body stays available
    assert {tool.name for tool in runtime._tools} >= {  # noqa: SLF001 - implicit runtime tool
        "bash",
        "submit",
        "read_skill",
    }
    assert "read_skill:" in runtime._system_prompt  # noqa: SLF001 - visible in pi's prompt
    pool.close()


def test_runtime_e2b_backend_rejects_in_process_runtime_kinds() -> None:
    """backend='e2b' only moves a pi-node harness PROCESS; in-process kinds must raise."""
    from wmo.optimize.harness.doc import code_baseline

    provider = _stub_provider()
    with pytest.raises(ValueError, match="use backend='local'"):
        HarnessDoc.baseline("b").runtime(provider, backend="e2b")
    with pytest.raises(ValueError, match="use backend='local'"):
        code_baseline("c").runtime(provider, backend="e2b")


def test_transport_retries_rejects_execution_modes_where_it_has_no_effect() -> None:
    provider = _stub_provider()
    with pytest.raises(ValueError, match="only to e2b pi-node"):
        _pi_doc().runtime(provider, transport_retries=0)
    with pytest.raises(ValueError, match="only to e2b pi-node"):
        HarnessDoc.baseline("b").runtime(provider, transport_retries=0)


def test_transport_retries_reaches_e2b_pi_runtime() -> None:
    from wmo.optimize.harness.pi_e2b import E2BPiRuntime, E2BSandboxPool

    pool = E2BSandboxPool()
    runtime = _pi_doc().runtime(
        _stub_provider(),
        backend="e2b",
        e2b_pool=pool,
        transport_retries=0,
    )

    assert isinstance(runtime, E2BPiRuntime)
    assert runtime._transport_retries == 0  # noqa: SLF001 - pins public policy passthrough
    pool.close()


def test_episode_timeout_reaches_every_pi_runtime_and_rejects_non_pi_runtimes() -> None:
    """The wall budget is pi-node policy on BOTH backends, not an e2b-only knob.

    It used to be rejected under `backend="local"`, so the SSH transport silently kept its
    hardcoded `timeout 300 node` and 31% of long TerminalBench-2 trials died on that clock while
    the configured budget was ignored.
    """
    from wmo.optimize.harness.pi_e2b import E2BPiRuntime
    from wmo.optimize.harness.pi_runtime import PiRuntime

    runtime = _pi_doc().runtime(
        _stub_provider(),
        backend="e2b",
        episode_timeout_s=12_000,
    )

    assert isinstance(runtime, E2BPiRuntime)
    assert runtime._episode_timeout_s == 12_000  # noqa: SLF001 - public policy passthrough
    runtime.close()

    local = _pi_doc().runtime(
        _stub_provider(),
        backend="local",
        episode_timeout_s=12_000,
    )
    assert isinstance(local, PiRuntime)
    assert local._episode_timeout_s == 12_000  # noqa: SLF001 - public policy passthrough

    with pytest.raises(ValueError, match="episode_timeout_s applies only to pi-node"):
        HarnessDoc.baseline().runtime(_stub_provider(), episode_timeout_s=12_000)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1.0])
def test_episode_timeout_validates_once_at_the_runtime_entry_point(value: object) -> None:
    from typing import cast

    with pytest.raises(ValueError, match="finite positive number"):
        _pi_doc().runtime(
            _stub_provider(),
            backend="e2b",
            episode_timeout_s=cast("float", value),
        )


# --- one tool protocol per runtime (audit defect 6) ----------------------------------------------
def test_structured_tool_runtimes_do_not_carry_the_json_action_clause() -> None:
    """The pi path passes STRUCTURED tool schemas; its renderer defines the calling convention.

    Shipping the JSON-action clause there declares a protocol nothing parses. Verified live in
    `super-topk/samples/step-0009.md`: the model received the tool schemas twice, in two notations,
    with two contradictory conventions, and 14.7% of Super trials spent at least one turn's
    reasoning on the JSON envelope (16% of the turns that hit the output cap contained such
    reasoning).
    """
    doc = HarnessDoc.baseline("base")

    json_runtime_prompt = doc.assembled_prompt()
    structured_prompt = doc.assembled_prompt(structured_tools=True)

    assert JSON_PROTOCOL_CLAUSE in json_runtime_prompt
    assert JSON_PROTOCOL_CLAUSE not in structured_prompt
    # Everything else is unchanged: same task guidance, same tool listing.
    assert "capable command-line agent" in structured_prompt
    assert "## Tools" in structured_prompt
    assert "call `submit` with your answer" in structured_prompt
    # No blank-line scar where the clause was.
    assert "\n\n\n" not in structured_prompt


def test_stripping_the_json_clause_is_a_no_op_on_an_edited_prompt() -> None:
    """An optimizer-edited prompt surface that never carried the clause is untouched."""
    doc = HarnessDoc(
        name="edited",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="Be a careful agent."),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
        ],
    )
    assert doc.assembled_prompt(structured_tools=True) == doc.assembled_prompt()


def test_every_pi_runtime_gets_the_structured_tool_prompt() -> None:
    """The three pi branches of runtime() must all opt in, or one path keeps the contradiction."""
    source = Path(doc_module.__file__).read_text(encoding="utf-8")
    assert source.count("self.assembled_prompt(skills, structured_tools=True)") == 3
    # CodeRuntime really does parse JSON via parse_tool_call, so it keeps the clause.
    assert source.count("system_prompt=self.assembled_prompt(skills),") == 1
