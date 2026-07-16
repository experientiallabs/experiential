"""Tests for the cross-session knowledge base."""

from __future__ import annotations

from pathlib import Path

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.knowledge import (
    GROUNDED_FILE,
    LEARNED_FILE,
    SEEDED_FILES,
    KnowledgeBase,
    seed_knowledge,
)
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


def test_missing_directory_is_empty_and_renders_nothing(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "does-not-exist")
    assert kb.is_empty
    assert kb.render() == ""


def test_render_concatenates_files_sorted_with_headers(tmp_path: Path) -> None:
    (tmp_path / "rules.md").write_text("- no refunds after 24h", encoding="utf-8")
    (tmp_path / "entities.md").write_text("- user u_kath", encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    text = kb.render()
    assert not kb.is_empty
    assert text.index("## entities.md") < text.index("## rules.md")  # sorted order
    assert "- user u_kath" in text and "- no refunds after 24h" in text


def test_render_respects_budget_with_loud_truncation(tmp_path: Path) -> None:
    (tmp_path / "rules.md").write_text("x" * 500, encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    text = kb.render(budget=100)
    assert len(text) <= 100
    assert "TRUNCATED" in text


def test_append_learned_adds_provenance_and_dedupes(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path)
    assert kb.append_learned("flight HAT-201 JFK->SFO exists", provenance="session abc123")
    assert not kb.append_learned("flight HAT-201 JFK->SFO exists", provenance="session zzz")
    content = (tmp_path / LEARNED_FILE).read_text(encoding="utf-8")
    assert content.count("HAT-201") == 1
    assert "session abc123" in content


def test_append_learned_refuses_past_cap(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path, append_cap=50)
    assert kb.append_learned("a" * 60, provenance="s1")  # first write always lands
    assert not kb.append_learned("another fact", provenance="s2")  # file now over cap


def test_grounded_cache_roundtrip(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path)
    assert kb.lookup_grounded("tomli_w python api") is None
    kb.append_grounded("tomli_w python api", "tomli_w.dump(obj, fh) writes TOML")
    hit = kb.lookup_grounded("tomli_w python api")
    assert hit is not None and "tomli_w.dump" in hit
    # Cached results are part of the rendered KB.
    assert "tomli_w.dump" in kb.render()


def test_write_and_read_files(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path)
    kb.write_file("rules.md", "- gate: auth required")
    assert kb.files() == {"rules.md": "- gate: auth required"}


class _SeedProvider:
    """Returns one canned extraction per call and counts calls."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.calls += 1
        return Completion(
            text=(
                f'{{"rules": "- gate: modifying a booking requires auth (call {self.calls})", '
                '"entities": "- flight HAT-201", "schemas": "- get_user -> {id, name}"}'
            )
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


def _trace(trace_id: str, n_steps: int = 2) -> Trace:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
            observation=Observation(content='{"id": "u1", "name": "kath"}'),
        )
        for _ in range(n_steps)
    ]
    return Trace(trace_id=trace_id, steps=steps)


def test_seed_knowledge_writes_seeded_files(tmp_path: Path) -> None:
    provider = _SeedProvider()
    kb = KnowledgeBase(tmp_path)
    seed_knowledge(kb, [_trace("t1"), _trace("t2")], provider)
    files = kb.files()
    for name in SEEDED_FILES:
        assert name in files, f"missing seeded file {name}"
    assert "HAT-201" in files["entities.md"]
    assert provider.calls >= 1


def test_seed_knowledge_respects_call_budget(tmp_path: Path) -> None:
    provider = _SeedProvider()
    kb = KnowledgeBase(tmp_path)
    traces = [_trace(f"t{i}", n_steps=40) for i in range(50)]  # far more text than one chunk
    seed_knowledge(kb, traces, provider, max_calls=3)
    assert provider.calls <= 3


def test_grounded_file_constant_is_markdown() -> None:
    assert GROUNDED_FILE.endswith(".md") and LEARNED_FILE.endswith(".md")


def test_append_learned_is_race_free_across_threads(tmp_path: Path) -> None:
    # FastAPI sync handlers run in a thread pool: two sessions stepping the same served model
    # race read-check-append. The lock must keep one entry per fact and lose no distinct facts.
    from concurrent.futures import ThreadPoolExecutor

    kb = KnowledgeBase(tmp_path / "knowledge")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: kb.append_learned("shared fact", provenance="s"), range(16)))
        list(pool.map(lambda i: kb.append_learned(f"fact {i}", provenance="s"), range(16)))
    learned = (tmp_path / "knowledge" / "learned.md").read_text(encoding="utf-8")
    assert learned.count("- shared fact") == 1  # no double-append under contention
    for i in range(16):
        assert f"- fact {i}" in learned  # no lost writes


def test_append_grounded_caches_a_racing_query_once(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    kb = KnowledgeBase(tmp_path / "knowledge")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: kb.append_grounded("pkg tomli_w", f"- result {i}"), range(12)))
    grounded = (tmp_path / "knowledge" / "grounded.md").read_text(encoding="utf-8")
    assert grounded.count("### query: pkg tomli_w") == 1  # first entry wins, no duplicates
