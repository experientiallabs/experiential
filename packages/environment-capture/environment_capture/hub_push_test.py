"""Tests for the Hub write side (hermetic — a stub stands in for HfApi)."""

from __future__ import annotations

from pathlib import Path

import pytest

from environment_capture import hub
from environment_capture.hub import CORPORA, repo_id_for
from environment_capture.hub_push import push_corpus


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
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hub, "_data_root", lambda: tmp_path)
    return tmp_path


def _make_bench(data_root: Path, benchmark: str) -> None:
    bench = data_root / benchmark
    bench.mkdir()
    (bench / "traces.otel.jsonl").write_text('{"traceId": "t"}\n')
    for data_dir in CORPORA[benchmark].data_dirs:
        (bench / data_dir).mkdir()
        (bench / data_dir / "part.jsonl").write_text("x\n")


def test_push_uploads_corpus_data_dirs_and_card(data_root: Path) -> None:
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


def test_gaia2_card_carries_the_disclosures(data_root: Path) -> None:
    _make_bench(data_root, "gaia2")
    api = _StubApi()
    push_corpus("gaia2", api=api)
    card = " ".join(api.uploaded[f"{repo_id_for('gaia2')}/README.md"].decode().split())
    assert "not comparable to the official leaderboard" in card
    assert "models not be trained on evaluation data" in card


def test_every_declared_data_dir_has_a_card_blurb() -> None:
    """A new data_dir with no blurb would KeyError at push time; catch it at test time."""
    from environment_capture.hub_push import DIR_BLURBS

    declared = {d for spec in CORPORA.values() for d in spec.data_dirs}
    assert declared <= set(DIR_BLURBS), f"missing blurbs: {declared - set(DIR_BLURBS)}"
