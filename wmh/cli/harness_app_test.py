"""Tests for the `wmh harness` CLI: the ingest command wiring (fake provider, local repo)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmh.cli import app
from wmh.harness.store import HarnessStore
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, VerifyResult

harness_app_module = importlib.import_module("wmh.cli.harness_app")

runner = CliRunner()


class FakeProvider:
    """Canned harnessdoc/overview JSON for every mapping call."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="fake")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "## File tree" in messages[0].content:
            return Completion(text=json.dumps({"overview": "A tiny agent."}))
        return Completion(text=json.dumps({"role": "a file", "doc": "Maps a thing."}))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:  # pragma: no cover - protocol filler
        raise NotImplementedError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "my-agent"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# my agent", encoding="utf-8")
    (root / "src" / "loop.py").write_text("def run():\n    pass\n", encoding="utf-8")
    return root


def _fake_provider_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness_app_module, "_ingest_provider", lambda *args, **kwargs: FakeProvider()
    )


def test_ingest_local_repo_saves_champion(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_provider_factory(monkeypatch)
    root = tmp_path / "proj"
    result = runner.invoke(app, ["harness", "ingest", str(repo), "--root", str(root), "--yes"])
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output and "my-agent" in result.output
    doc = HarnessStore(str(root)).load("my-agent")
    assert {s.path for s in doc.code_files()} == {"BODYMAP.md", "README.md", "src/loop.py"}
    loop = next(s for s in doc.code_files() if s.path == "src/loop.py")
    assert loop.doc == "Maps a thing."


def test_ingest_rejects_existing_name(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_provider_factory(monkeypatch)
    root = tmp_path / "proj"
    first = runner.invoke(app, ["harness", "ingest", str(repo), "--root", str(root), "--yes"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["harness", "ingest", str(repo), "--root", str(root), "--yes"])
    assert second.exit_code == 2
    assert "already exists" in second.output


def test_ingest_ref_requires_url_source(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_provider_factory(monkeypatch)
    result = runner.invoke(
        app,
        ["harness", "ingest", str(repo), "--ref", "main", "--root", str(tmp_path / "p"), "--yes"],
    )
    assert result.exit_code == 2
    assert "git URL sources" in result.output


def test_ingest_default_name_from_url() -> None:
    assert harness_app_module._default_ingest_name("https://github.com/acme/demo.git") == "demo"
    assert harness_app_module._default_ingest_name("git@github.com:acme/demo.git") == "demo"
    assert harness_app_module._looks_like_git_url("https://github.com/acme/demo")
    assert not harness_app_module._looks_like_git_url("/tmp/somewhere")
