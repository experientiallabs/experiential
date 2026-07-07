"""Tests for Hub corpus publishing/fetching (hermetic — a stub stands in for the Hub API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from environment_capture import hub
from environment_capture.hub import CORPORA, fetch_corpus, push_corpus, repo_id_for


class _StubApi:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.uploaded: dict[str, bytes] = {}

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        self.created.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok}
        )

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None:
        content = (
            Path(path_or_fileobj).read_bytes()
            if isinstance(path_or_fileobj, str)
            else path_or_fileobj
        )
        self.uploaded[f"{repo_id}/{path_in_repo}"] = content

    def upload_folder(
        self,
        *,
        folder_path: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None:
        for file in sorted(Path(folder_path).rglob("*")):
            if file.is_file():
                rel = file.relative_to(folder_path)
                self.uploaded[f"{repo_id}/{path_in_repo}/{rel}"] = file.read_bytes()


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    monkeypatch.setattr(hub, "_data_root", lambda: tmp_path)
    return tmp_path


def _make_bench(data_root: Path, benchmark: str) -> None:
    bench = data_root / benchmark
    bench.mkdir()
    (bench / "traces.otel.jsonl").write_text('{"traceId": "t"}\n')
    for data_dir in CORPORA[benchmark].data_dirs:
        (bench / data_dir).mkdir()
        (bench / data_dir / "part.jsonl").write_text("x\n")


def test_push_uploads_corpus_and_card(data_root: Path) -> None:
    _make_bench(data_root, "bird-sql")
    api = _StubApi()

    url = push_corpus("bird-sql", api=api)

    repo_id = repo_id_for("bird-sql")
    assert url == f"https://huggingface.co/datasets/{repo_id}"
    assert api.created[0] == {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private": False,
        "exist_ok": True,
    }
    assert api.uploaded[f"{repo_id}/traces.otel.jsonl"] == b'{"traceId": "t"}\n'
    card = api.uploaded[f"{repo_id}/README.md"].decode()
    assert card.startswith("---\nlicense: cc-by-sa-4.0\n")  # tag must match upstream terms
    assert "bird-bench mini-dev" in card  # attribution rides the card
    # the data payload rides the same repo, under its dir names
    assert api.uploaded[f"{repo_id}/data/part.jsonl"] == b"x\n"
    assert api.uploaded[f"{repo_id}/gold/part.jsonl"] == b"x\n"
    assert api.uploaded[f"{repo_id}/schemas/part.jsonl"] == b"x\n"


def test_push_private_flag_reaches_create_repo(data_root: Path) -> None:
    _make_bench(data_root, "dabstep")
    api = _StubApi()
    push_corpus("dabstep", private=True, api=api)
    assert api.created[0]["private"] is True


def test_push_rejects_unpublishable_benchmark(data_root: Path) -> None:
    """appworld's license forbids plain-text redistribution — pushing it must be an error that
    says so, not a silent upload."""
    with pytest.raises(ValueError, match="appworld is local-only"):
        push_corpus("appworld", api=_StubApi())


def test_push_requires_a_local_corpus(data_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="capture one first"):
        push_corpus("gaia2", api=_StubApi())


def test_fetch_keeps_existing_local_corpus_unless_forced(
    data_root: Path,
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    """Local-first: a corpus grown by local capture waves must never be silently clobbered."""
    local = data_root / "gaia2" / "traces.otel.jsonl"
    local.parent.mkdir()
    local.write_text("local-waves\n")
    remote = tmp_path / "remote.jsonl"
    remote.write_text("published\n")
    snapshot = tmp_path / "snapshot"
    (snapshot / "data").mkdir(parents=True)
    (snapshot / "data" / "train.jsonl").write_text("tasks\n")
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **k: str(remote))
    monkeypatch.setattr(hub, "snapshot_download", lambda *a, **k: str(snapshot))

    assert fetch_corpus("gaia2") == local
    assert local.read_text() == "local-waves\n"  # kept
    assert (data_root / "gaia2" / "data" / "train.jsonl").read_text() == "tasks\n"  # materialized
    assert fetch_corpus("gaia2", force=True) == local
    assert local.read_text() == "published\n"  # explicitly overwritten

    # a plain re-fetch keeps everything (no clobber without force)
    (data_root / "gaia2" / "data" / "train.jsonl").write_text("local-edit\n")
    fetch_corpus("gaia2")
    assert (data_root / "gaia2" / "data" / "train.jsonl").read_text() == "local-edit\n"


def test_gaia2_card_carries_the_disclosures(data_root: Path) -> None:
    _make_bench(data_root, "gaia2")
    api = _StubApi()
    push_corpus("gaia2", api=api)
    card = " ".join(api.uploaded[f"{repo_id_for('gaia2')}/README.md"].decode().split())
    assert "not comparable to the official leaderboard" in card
    assert "models not be trained on evaluation data" in card


def test_every_committed_corpus_is_publishable_or_documented_local_only() -> None:
    """Manifest coverage: every benchmark dir with a committed corpus must either be in the
    publish manifest or be appworld (the documented local-only exception)."""
    root = hub._data_root()
    dirs = {p.parent.name for p in root.glob("*/traces.otel.jsonl")}
    if not dirs:  # standalone package install: data dirs don't ship
        pytest.skip("no sibling benchmark data dirs")
    assert dirs - set(CORPORA) <= {"appworld"}
