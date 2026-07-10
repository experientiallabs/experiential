"""Tests for body-mapping ingest: collection rules, slug/path mapping, doc assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.harness.doc import SurfaceKind
from wmh.harness.ingest import (
    BODYMAP_PATH,
    BodyMap,
    CollectedRepo,
    FileDoc,
    RepoFile,
    body_map,
    build_ingest_doc,
    collect_repo_files,
    sanitize_path,
    slug_for_path,
)
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)


class FakeProvider:
    """Replies with a canned harnessdoc JSON (or garbage when `garbage_for` matches)."""

    def __init__(self, garbage_for: set[str] | None = None) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="fake")
        self.calls: list[str] = []
        self._garbage_for = garbage_for or set()

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        user = messages[0].content
        self.calls.append(user)
        if "## File tree" in user:  # the overview call (checked first: it embeds every path)
            if "## File tree" in self._garbage_for:
                return Completion(text="not json at all", usage=TokenUsage(), model="fake")
            payload = {"overview": "A demo agent. Prompts in prompts/, loop in src/."}
        elif any(marker in user for marker in self._garbage_for):
            return Completion(text="not json at all", usage=TokenUsage(), model="fake")
        else:
            payload = {"role": "a mapped file", "doc": "Does a thing. Serves the agent."}
        return Completion(text=json.dumps(payload), usage=TokenUsage(), model="fake")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:  # pragma: no cover - protocol filler
        raise NotImplementedError


def _write(root: Path, rel: str, content: str | bytes) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")


def _demo_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root, "README.md", "# Demo agent\nDoes demo things.")
    _write(root, "src/agent.py", "def run():\n    return 'hi'\n")
    _write(root, "src/tools/search.py", "QUERY = 'x'\n")
    _write(root, "prompts/system.txt", "You are a demo agent.")
    _write(root, ".github/workflows/ci.yml", "on: push\n")
    _write(root, "node_modules/dep/index.js", "module.exports = 1")
    _write(root, "package-lock.json", "{}")
    _write(root, ".env", "SECRET=1")
    _write(root, "logo.png", b"\x89PNG\x00\x00binary")
    _write(root, "big.txt", "x" * 200_000)
    _write(root, ".gitignore", "ignored_dir/\n*.tmp\n")
    _write(root, "ignored_dir/inner.txt", "nope")
    _write(root, "scratch.tmp", "nope")
    return root


def test_collect_includes_zealously_and_skips_with_reasons(tmp_path: Path) -> None:
    collected = collect_repo_files(_demo_repo(tmp_path))
    included = [f.path for f in collected.files]
    assert included == sorted(included)  # deterministic order
    assert ".github/workflows/ci.yml" in included
    assert "README.md" in included
    assert "src/tools/search.py" in included
    assert ".gitignore" in included  # zealous: meta files are body too
    reasons = {s.path: s.reason for s in collected.skipped}
    assert "package-lock.json" in reasons
    assert reasons[".env"] == "may contain secrets"
    assert "binary" in reasons["logo.png"]
    assert "per-file cap" in reasons["big.txt"]
    assert "matched .gitignore" in reasons["ignored_dir/inner.txt"]
    assert "matched .gitignore" in reasons["scratch.tmp"]
    assert not any("node_modules" in p for p in included)


def test_collect_total_budget_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root, "a.txt", "x" * 60)
    _write(root, "b.txt", "y" * 60)
    collected = collect_repo_files(root, max_total_bytes=100)
    assert [f.path for f in collected.files] == ["a.txt"]
    assert collected.skipped[0].path == "b.txt"
    assert "total budget" in collected.skipped[0].reason


def test_collect_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        collect_repo_files(tmp_path / "missing")


def test_slug_for_path_normalizes_and_disambiguates() -> None:
    taken: set[str] = set()
    assert slug_for_path("README.md", taken) == "readme-md"
    assert slug_for_path("src/__init__.py", taken) == "src-init-py"
    first = slug_for_path("a/b.ts", taken)
    taken.add(first)
    second = slug_for_path("a/b_ts", taken)  # same kebab, different path
    assert first == "a-b-ts" and second.startswith("a-b-ts-") and second != first


def test_sanitize_path_handles_unrepresentable_segments() -> None:
    assert sanitize_path("app/[id]/page.tsx") == ("app/_id_/page.tsx", True)
    assert sanitize_path("src/agent.py") == ("src/agent.py", False)
    assert sanitize_path("weird/../escape.txt") == ("weird/_/escape.txt", True)


def test_body_map_survives_unusable_replies(tmp_path: Path) -> None:
    collected = collect_repo_files(_demo_repo(tmp_path))
    provider = FakeProvider(garbage_for={"prompts/system.txt"})
    mapping = body_map(collected, provider, name="demo")
    assert mapping.unmapped == ["prompts/system.txt"]
    kept = mapping.doc_for("prompts/system.txt")
    assert kept is not None and "unusable" in kept.doc
    mapped = mapping.doc_for("src/agent.py")
    assert mapped is not None and mapped.role == "a mapped file"
    assert "demo agent" in mapping.overview.lower()


def test_body_map_overview_fallback_on_unusable_reply(tmp_path: Path) -> None:
    collected = collect_repo_files(_demo_repo(tmp_path))
    mapping = body_map(collected, FakeProvider(garbage_for={"## File tree"}), name="demo")
    assert "ingested from an existing repo" in mapping.overview  # deterministic fallback


def test_build_ingest_doc_preserves_hierarchy_and_docs(tmp_path: Path) -> None:
    collected = collect_repo_files(_demo_repo(tmp_path))
    mapping = body_map(collected, FakeProvider(), name="demo")
    doc = build_ingest_doc("demo", collected, mapping, source="github.com/acme/demo")
    paths = {s.path for s in doc.code_files()}
    assert "src/tools/search.py" in paths  # hierarchy preserved
    assert BODYMAP_PATH in paths
    agent_surface = next(s for s in doc.code_files() if s.path == "src/agent.py")
    assert agent_surface.doc == "Does a thing. Serves the agent."
    assert agent_surface.kind is SurfaceKind.CODE
    assert "BODYMAP.md" in doc.system_prompt()
    bodymap = next(s for s in doc.code_files() if s.path == BODYMAP_PATH)
    assert "`src/agent.py` - a mapped file" in bodymap.content
    assert "## Skipped" in bodymap.content
    # The doc validates as a whole and round-trips through JSON with annotations intact.
    reloaded = type(doc).model_validate_json(doc.model_dump_json())
    assert reloaded.doc_hash == doc.doc_hash
    assert (
        next(s for s in reloaded.code_files() if s.path == "src/agent.py").doc == agent_surface.doc
    )


def test_build_ingest_doc_disambiguates_sanitized_collisions() -> None:
    collected = CollectedRepo(
        files=[
            RepoFile(path="a/[x].ts", content="one"),
            RepoFile(path="a/_x_.ts", content="two"),
        ],
        skipped=[],
    )
    mapping = BodyMap(
        overview="o",
        file_docs=[
            FileDoc(path="a/[x].ts", role="r", doc="d"),
            FileDoc(path="a/_x_.ts", role="r", doc="d"),
        ],
    )
    doc = build_ingest_doc("demo", collected, mapping, source="local")
    code_paths = [s.path for s in doc.code_files() if s.path != BODYMAP_PATH]
    assert len(set(code_paths)) == 2  # both kept, paths disambiguated
    original_notes = [s.doc for s in doc.code_files() if s.path != BODYMAP_PATH]
    assert any(note is not None and "Original path: a/[x].ts" in note for note in original_notes)


def test_build_ingest_doc_moves_reserved_render_paths_aside() -> None:
    collected = CollectedRepo(
        files=[RepoFile(path="SYSTEM.md", content="the repo's own system doc")],
        skipped=[],
    )
    mapping = BodyMap(overview="o", file_docs=[FileDoc(path="SYSTEM.md", role="r", doc="d")])
    doc = build_ingest_doc("demo", collected, mapping, source="local")
    paths = [s.path for s in doc.code_files()]
    assert "SYSTEM.md" not in paths  # the store's rendered SYSTEM.md owns that path
    moved = next(s for s in doc.code_files() if s.path not in {BODYMAP_PATH})
    assert moved.path.startswith("SYSTEM-") and moved.content == "the repo's own system doc"
    assert moved.doc is not None and "Original path: SYSTEM.md" in moved.doc
