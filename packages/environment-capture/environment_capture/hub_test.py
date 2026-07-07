"""Tests for the stdlib read core: listing, fetching, progress, atomicity (no network)."""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable
from pathlib import Path

import pytest

from environment_capture import hub
from environment_capture.hub import CORPORA, fetch_corpus, published_corpora, repo_id_for


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hub, "_data_root", lambda: tmp_path)
    return tmp_path


def _fake_hub(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    """Stand in for the Hub REST API: a tree listing plus resolve-URL streaming."""

    def http_json_page(url: str, *, token: str | None) -> tuple[object, None]:
        assert "/api/datasets/" in url and "/tree/main?recursive=true" in url
        listing = [
            {"type": "file", "path": path, "size": len(content)}
            for path, content in files.items()
        ]
        return listing, None

    def stream_to(
        url: str, dest: Path, *, token: str | None, chunk_done: Callable[[int], None]
    ) -> int:
        remote_path = urllib.parse.unquote(url.split("/resolve/main/", 1)[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = files[remote_path]
        dest.write_bytes(content)
        chunk_done(len(content))
        return len(content)

    monkeypatch.setattr(hub, "_http_json_page", http_json_page)
    monkeypatch.setattr(hub, "_stream_to", stream_to)


def test_fetch_downloads_corpus_and_data_dirs_with_one_progress_bar(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(
        monkeypatch,
        {
            "traces.otel.jsonl": b"spans\n",
            "data/train.jsonl": b"tasks\n",
            "gold/t1.json": b"{}",
        },
    )
    progress: list[tuple[int, int]] = []

    path = fetch_corpus(
        "continual-learning", on_progress=lambda done, total: progress.append((done, total))
    )

    assert path == data_root / "continual-learning" / "traces.otel.jsonl"
    assert path.read_bytes() == b"spans\n"
    assert (data_root / "continual-learning" / "data" / "train.jsonl").read_bytes() == b"tasks\n"
    assert (data_root / "continual-learning" / "gold" / "t1.json").read_bytes() == b"{}"
    # one monotone bar over the WHOLE bundle: total constant, done reaches it
    total = 6 + 6 + 2
    assert progress == [(6, total), (12, total), (14, total)]


def test_fetch_keeps_existing_local_files_unless_forced(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local-first: a corpus grown by local capture waves must never be silently clobbered."""
    _fake_hub(monkeypatch, {"traces.otel.jsonl": b"published\n", "data/train.jsonl": b"tasks\n"})
    bench = data_root / "gaia2"
    (bench / "data").mkdir(parents=True)
    (bench / "traces.otel.jsonl").write_text("local-waves\n")
    (bench / "data" / "train.jsonl").write_text("local-edit\n")

    fetch_corpus("gaia2")
    assert (bench / "traces.otel.jsonl").read_text() == "local-waves\n"  # kept
    assert (bench / "data" / "train.jsonl").read_text() == "local-edit\n"  # kept

    fetch_corpus("gaia2", force=True)
    assert (bench / "traces.otel.jsonl").read_text() == "published\n"
    assert (bench / "data" / "train.jsonl").read_text() == "tasks\n"


def test_fetch_with_dest_writes_only_the_corpus_file(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_hub(monkeypatch, {"traces.otel.jsonl": b"spans\n", "data/train.jsonl": b"tasks\n"})
    dest = tmp_path / "elsewhere" / "corpus.jsonl"
    assert fetch_corpus("gaia2", dest=dest) == dest
    assert dest.read_bytes() == b"spans\n"
    assert not (data_root / "gaia2" / "data").exists()


def test_fetch_unknown_benchmark_names_the_available_ones(data_root: Path) -> None:
    with pytest.raises(ValueError, match="no published corpus"):
        fetch_corpus("nope")


def test_stream_to_is_atomic(tmp_path: Path) -> None:
    """The real streamer writes a .part sibling and renames over — a partial download must
    never be mistaken for a complete corpus by a concurrent reader."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (3 * 1024))
    dest = tmp_path / "out" / "corpus.jsonl"
    seen: list[int] = []

    hub._stream_to(source.as_uri(), dest, token=None, chunk_done=seen.append)

    assert dest.read_bytes() == b"x" * (3 * 1024)
    assert not dest.with_name(dest.name + ".part").exists()
    assert sum(seen) == 3 * 1024


def test_published_corpora_maps_repos_to_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [
        {"id": "experiential-labs/wmh-gaia2-traces", "lastModified": "2026-07-07T06:00:00.000Z"},
        {
            "id": "experiential-labs/wmh-bird-sql-traces",
            "lastModified": "2026-07-05T00:00:00.000Z",
        },
        {"id": "experiential-labs/unrelated-dataset", "lastModified": "2026-07-06T00:00:00.000Z"},
        {"id": "experiential-labs/wmh-not-a-benchmark-traces", "lastModified": ""},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    published = published_corpora()
    assert [(c.benchmark, c.last_modified) for c in published] == [
        ("gaia2", "2026-07-07"),
        ("bird-sql", "2026-07-05"),
    ]
    assert published[0].repo_id == repo_id_for("gaia2")


def test_every_committed_corpus_is_publishable_or_documented_local_only() -> None:
    """Manifest coverage: every benchmark dir with a local corpus must either be in the
    publish manifest or be appworld (the documented local-only exception)."""
    root = hub._data_root()
    dirs = {p.parent.name for p in root.glob("*/traces.otel.jsonl")}
    if not dirs:  # standalone package install: data dirs don't ship
        pytest.skip("no sibling benchmark data dirs")
    assert dirs - set(CORPORA) <= {"appworld"}


def test_fetch_resumes_missing_files_inside_an_existing_dir(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted fetch that materialized only part of a data dir picks up the missing
    files on re-run — dir presence alone must not mean 'complete'."""
    _fake_hub(
        monkeypatch,
        {
            "traces.otel.jsonl": b"spans\n",
            "data/train.jsonl": b"tasks\n",
            "data/test.jsonl": b"held-out\n",
        },
    )
    bench = data_root / "gaia2"
    (bench / "data").mkdir(parents=True)
    (bench / "traces.otel.jsonl").write_text("local\n")
    (bench / "data" / "train.jsonl").write_text("already-here\n")

    fetch_corpus("gaia2")
    assert (bench / "data" / "train.jsonl").read_text() == "already-here\n"  # kept
    assert (bench / "data" / "test.jsonl").read_bytes() == b"held-out\n"  # resumed


def test_fetch_names_a_repo_missing_its_corpus(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch, {"data/train.jsonl": b"tasks\n"})
    with pytest.raises(ValueError, match="never pushed"):
        fetch_corpus("gaia2")


def test_gitignore_covers_every_declared_data_dir() -> None:
    """The package .gitignore must shadow CORPORA's data_dirs: a spec dir with no matching
    ignore pattern means `git add -A` can commit license-restricted payload."""
    gitignore = hub._data_root() / ".gitignore"
    if not gitignore.exists():  # standalone package install
        pytest.skip("no package .gitignore shipped")
    patterns = {
        line.strip() for line in gitignore.read_text().splitlines() if line.strip().startswith("*/")
    }
    assert "*/traces.otel.jsonl" in patterns
    declared = {d for spec in CORPORA.values() for d in spec.data_dirs}
    missing = {d for d in declared if f"*/{d}/" not in patterns}
    assert not missing, f"data dirs with no ignore pattern (license-leak risk): {missing}"


def test_license_tags_match_the_provenance_readmes() -> None:
    """CorpusSpec.license_id is what gets published on the dataset card; it must agree with the
    license each benchmark README records (INTEGRATION.md non-negotiable #3)."""
    human = {
        "cc-by-nc-4.0": ("CC BY-NC",),
        "cc-by-sa-4.0": ("CC BY-SA",),
        "cc-by-4.0": ("CC BY 4.0", "CC-BY-4.0"),
        "mit": ("MIT",),
        "apache-2.0": ("Apache",),
    }
    root = hub._data_root()
    checked = 0
    for spec in CORPORA.values():
        readme = root / spec.benchmark / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        assert any(marker in text for marker in human[spec.license_id]), (
            f"{spec.benchmark}: card would publish {spec.license_id} but its README never "
            f"mentions {human[spec.license_id]} — fix whichever is wrong before pushing"
        )
        checked += 1
    if not checked:  # standalone package install
        pytest.skip("no benchmark READMEs shipped")


def test_published_corpora_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """An org with more datasets than one page must not hide corpora beyond page 1."""
    pages = {
        "page1": (
            [{"id": "experiential-labs/wmh-gaia2-traces", "lastModified": "2026-07-07T00:00:00Z"}],
            "page2",
        ),
        "page2": (
            [
                {
                    "id": "experiential-labs/wmh-bird-sql-traces",
                    "lastModified": "2026-07-06T00:00:00Z",
                }
            ],
            None,
        ),
    }

    def page(url: str, *, token: str | None) -> tuple[object, str | None]:
        key = "page2" if url == "page2" else "page1"
        return pages[key]

    monkeypatch.setattr(hub, "_http_json_page", page)
    assert [c.benchmark for c in published_corpora()] == ["gaia2", "bird-sql"]
