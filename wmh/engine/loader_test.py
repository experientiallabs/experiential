"""Tests for the artifact-dir to live WorldModel loading seam."""

from __future__ import annotations

from pathlib import Path

import pytest

import wmh.engine.loader as loader_module
from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.core.types import Action, ActionKind
from wmh.engine.knowledge import KnowledgeBase
from wmh.engine.loader import load_world_model
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)


class _StubProvider:
    """A hermetic serve provider: no credentials, records the last rendered prompt."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        del system, temperature, max_tokens
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def _stub_serve_provider(monkeypatch: pytest.MonkeyPatch, reply: str) -> _StubProvider:
    """Pin the loader's provider construction to a hermetic stub (no backend credentials)."""
    provider = _StubProvider(reply)
    monkeypatch.setattr(loader_module, "provider_or_chain", lambda _config: provider)
    return provider


def test_load_world_model_threads_knowledge_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
            serve_provider=ProviderKind.BEDROCK,
            reasoning=True,
            knowledge=True,
        ),
        root=root,
    )
    override = tmp_path / "run-knowledge"
    KnowledgeBase(override).write_file("foo.md", "- override fact: seeded per run")
    provider = _stub_serve_provider(
        monkeypatch, '{"reasoning": "r", "output": "ok", "is_error": false}'
    )

    wm, returned = load_world_model(root, knowledge_dir=override)

    assert returned is provider
    assert wm.knowledge is not None
    assert "foo.md" in wm.knowledge.files()
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    assert "override fact: seeded per run" in (provider.last_user or "")


def test_load_world_model_without_knowledge_dir_uses_the_artifact_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
            serve_provider=ProviderKind.BEDROCK,
            reasoning=True,
            knowledge=True,
        ),
        root=root,
    )
    KnowledgeBase(ArtifactPaths(root).knowledge).write_file("rules.md", "- gate: auth first")
    _stub_serve_provider(monkeypatch, '{"reasoning": "r", "output": "ok", "is_error": false}')

    wm, _provider = load_world_model(root)

    assert wm.knowledge is not None
    assert wm.knowledge.directory == ArtifactPaths(root).knowledge
