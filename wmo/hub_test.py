"""Tests for the vendored Hub read core: listing, fetching, atomicity (no network).

This is the copy `pip install world-model-optimizer` actually runs, so it carries its own
coverage rather than leaning on the upstream distribution's. Nothing here imports
`environment_capture`, and nothing under `wmo/` does either.

There used to be a drift test diffing this file's shared regions against the in-repo
`packages/environment-capture/` copy. That directory is gone, so the check could only ever
skip; upstream now diverges by release, not by hand-mirrored edit.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from http.client import HTTPMessage
from pathlib import Path

import pytest

from wmo import hub
from wmo.hub import (
    CORPORA,
    CorpusRepoUnavailable,
    candidate_repo_ids,
    downloadable_benchmarks,
    fetch_corpus,
    published_corpora,
)


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hub, "_data_root", lambda: tmp_path)
    return tmp_path


@dataclass
class _HubCalls:
    """The repo ids the fake Hub was asked for, in order (one entry per request)."""

    trees: list[str] = field(default_factory=list)
    resolves: list[str] = field(default_factory=list)


def _fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    live_repos: Container[str] | None = None,
    missing_code: int = 404,
    claimed_sizes: dict[str, int] | None = None,
) -> _HubCalls:
    """Stand in for the Hub REST API: a tree listing plus resolve-URL streaming.

    Args:
        monkeypatch: The patcher used to swap the module's HTTP seams.
        files: Repo path -> content, served by every live repo.
        live_repos: Repo ids that resolve; every other id answers ``missing_code``. ``None``
            (the default) means every id resolves.
        missing_code: Status the Hub returns for a repo id outside ``live_repos``.
        claimed_sizes: Repo path -> the size the TREE advertises, when it should disagree with
            the bytes served (how a truncated transfer is detected).

    Returns:
        A live record of the repo ids requested, so a test can assert the request COUNT and
        not just the downloaded bytes.
    """
    calls = _HubCalls()
    sizes = claimed_sizes or {}

    def http_json_page(url: str, *, token: str | None) -> tuple[object, None]:
        assert "/api/datasets/" in url and "/tree/main?recursive=true" in url
        repo_id = url.split("/api/datasets/", 1)[1].split("/tree/", 1)[0]
        calls.trees.append(repo_id)
        if live_repos is not None and repo_id not in live_repos:
            raise urllib.error.HTTPError(url, missing_code, "not found", HTTPMessage(), None)
        listing = [
            {"type": "file", "path": path, "size": sizes.get(path, len(content))}
            for path, content in files.items()
        ]
        return listing, None

    def stream_to(
        url: str,
        dest: Path,
        *,
        token: str | None,
        chunk_done: Callable[[int], None],
        expect_bytes: int | None = None,
    ) -> int:
        calls.resolves.append(url.split("/datasets/", 1)[1].split("/resolve/", 1)[0])
        remote_path = urllib.parse.unquote(url.split("/resolve/main/", 1)[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = files[remote_path]
        chunk_done(len(content))
        # The real streamer verifies the advertised size BEFORE renaming its `.part` over
        # `dest`, so a short transfer leaves `dest` exactly as it was. A fake that wrote first
        # would hide precisely the data-loss case this seam exists to prevent.
        if expect_bytes is not None and len(content) != expect_bytes:
            raise OSError(
                f"downloaded {len(content)} bytes but the Hub tree lists {expect_bytes} — "
                f"truncated transfer; {dest} was left as it was, re-run the fetch"
            )
        dest.write_bytes(content)
        return len(content)

    monkeypatch.setattr(hub, "_http_json_page", http_json_page)
    monkeypatch.setattr(hub, "_stream_to", stream_to)
    return calls


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


_PREBUILT = {
    "traces.otel.jsonl": b"spans\n",
    "models/tau-bench/config.toml": b'serve_provider = "bedrock"\n',
    "models/tau-bench/card.json": b'{"name": "tau-bench"}',
    "models/tau-bench/metrics.json": b"{}",
    "models/tau-bench/prompts/base.txt": b"you are a backend\n",
    "models/tau-bench/index/embeddings.npy": b"\x93NUMPY-ish",
    "evals/default.toml": b'title = "Tau Bench default replay"\nfiles = ["../traces.otel.jsonl"]\n',
}


def test_fetch_downloads_the_prebuilt_model_and_eval_suites(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle is not just traces: the world model built from them and its suites ride along.

    tau-bench declares no ``data_dirs``, which is the point: the artifact dirs are a property of
    every bundle, not of a corpus spec, so a benchmark with no upstream payload still gets them.
    """
    _fake_hub(monkeypatch, _PREBUILT)

    fetch_corpus("tau-bench")

    bench = data_root / "tau-bench"
    for remote_path, content in _PREBUILT.items():
        assert (bench / remote_path).read_bytes() == content


def test_fetch_places_artifacts_where_existing_discovery_looks(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote layout mirrors the local one, so nothing downstream needs a Hub special case.

    This asserts against the REAL resolvers rather than restating paths: the store that every
    read command walks (`<data root>/<benchmark>/models/<name>/`) and the suite glob
    (`<root>/*/evals/*.toml`). Renaming either layout on the Hub breaks here instead of at a
    user's first `wmo eval`.
    """
    from wmo.config.store import WorldModelStore
    from wmo.engine.eval_suites import discover_eval_suites

    _fake_hub(monkeypatch, _PREBUILT)

    fetch_corpus("tau-bench")

    assert WorldModelStore(data_root / "tau-bench").list_names() == ["tau-bench"]
    suites = discover_eval_suites(data_root)
    assert [suite.id for suite in suites] == ["tau-bench/default"]
    # The suite's relative `files` resolve because corpus and suites land in one benchmark dir.
    assert suites[0].resolve_files() == [data_root / "tau-bench" / "traces.otel.jsonl"]


def test_fetch_keeps_a_local_artifact_unless_forced(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local-first covers artifacts too: a model retuned in place is not published-over."""
    _fake_hub(monkeypatch, _PREBUILT)
    config = data_root / "tau-bench" / "models" / "tau-bench" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('serve_provider = "retuned-locally"\n')

    fetch_corpus("tau-bench")
    assert config.read_text() == 'serve_provider = "retuned-locally"\n'

    fetch_corpus("tau-bench", force=True)
    assert config.read_bytes() == _PREBUILT["models/tau-bench/config.toml"]


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


def test_fetch_with_dest_writes_only_the_corpus_file(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit `dest` asks for one file, so neither data nor artifact dirs are materialized
    (they have nowhere sensible to go: their layout is relative to the benchmark dir)."""
    _fake_hub(
        monkeypatch,
        {
            "traces.otel.jsonl": b"spans\n",
            "data/train.jsonl": b"tasks\n",
            "models/gaia2/config.toml": b"top_k = 5\n",
            "evals/default.toml": b"seed = 0\n",
        },
    )
    dest = tmp_path / "elsewhere" / "corpus.jsonl"
    assert fetch_corpus("gaia2", dest=dest) == dest
    assert dest.read_bytes() == b"spans\n"
    assert not (data_root / "gaia2" / "data").exists()
    assert not (data_root / "gaia2" / "models").exists()
    assert not (data_root / "gaia2" / "evals").exists()


def test_fetch_fails_plainly_when_the_repo_does_not_resolve(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One canonical name per benchmark: an unresolvable repo is an error, not a hunt.

    The wmh-/wmo- fallback this test used to pin was dropped when the Hub repos were
    renamed (2026-08-03); the Hub redirects the legacy ids, so the fallback's job moved
    to the Hub itself and a miss here means the dataset genuinely is not published.
    """
    (canonical,) = candidate_repo_ids("gaia2")
    calls = _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n"},
        live_repos=set(),
    )

    with pytest.raises(CorpusRepoUnavailable):
        fetch_corpus("gaia2")

    assert calls.trees == [canonical]

def test_fetch_asks_once_when_the_canonical_repo_resolves(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolving canonical id is the only lookup: one name, one tree call."""
    (canonical,) = candidate_repo_ids("gaia2")
    calls = _fake_hub(monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos={canonical})

    fetch_corpus("gaia2")

    assert calls.trees == [canonical]


@pytest.mark.parametrize("code", [404, 401])
def test_fetch_names_every_repo_id_it_tried_when_none_resolve(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """A miss on BOTH names is a real error (404 anonymously, 401 with a token attached), and
    it has to say which ids were looked for or the user cannot tell what to publish."""
    calls = _fake_hub(
        monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos=set(), missing_code=code
    )

    with pytest.raises(CorpusRepoUnavailable) as caught:
        fetch_corpus("gaia2")

    assert isinstance(caught.value, urllib.error.HTTPError)  # front-ends still catch it
    assert caught.value.code == code
    assert caught.value.attempted == candidate_repo_ids("gaia2")
    assert all(repo_id in str(caught.value) for repo_id in candidate_repo_ids("gaia2"))
    assert calls.trees == list(candidate_repo_ids("gaia2"))


def test_fetch_does_not_try_another_name_on_a_non_missing_hub_error(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate limiting is the Hub misbehaving, not the wrong repo name: surface it as-is rather
    than burning a second request and reporting it as an unpublished corpus."""
    calls = _fake_hub(
        monkeypatch, {"traces.otel.jsonl": b"spans\n"}, live_repos=set(), missing_code=429
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch_corpus("gaia2")

    assert not isinstance(caught.value, CorpusRepoUnavailable)
    assert caught.value.code == 429
    assert calls.trees == [candidate_repo_ids("gaia2")[0]]


def test_fetch_names_a_repo_missing_its_corpus(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_hub(monkeypatch, {"data/train.jsonl": b"tasks\n"})
    with pytest.raises(ValueError, match="never pushed"):
        fetch_corpus("gaia2")


def test_fetch_unknown_benchmark_names_the_available_ones(data_root: Path) -> None:
    with pytest.raises(ValueError, match="no published corpus"):
        fetch_corpus("nope")


def test_fetch_rejects_a_transfer_the_tree_says_is_short(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short read is a corrupt corpus, not a small one.

    It must RAISE, with a type `wmo download` can tell apart from an outage (it catches
    `OSError` to keep one bad bundle from stranding a batch), and it must leave nothing behind:
    a fetch skips any path that exists, so a truncated file at the destination would make the
    error's own "re-run the fetch" remedy answer `kept local` over a corpus missing most of its
    spans, at exit 0.
    """
    _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n"},
        claimed_sizes={"traces.otel.jsonl": 4096},
    )
    with pytest.raises(OSError, match="truncated transfer") as caught:
        fetch_corpus("gaia2")
    assert not isinstance(caught.value, urllib.error.URLError)  # not mistaken for an outage
    assert "traces.otel.jsonl" in str(caught.value)  # a bundle is many files; name the one
    corpus = data_root / "gaia2" / "traces.otel.jsonl"
    assert not corpus.exists()  # nothing to make the next fetch think it is done
    assert not corpus.with_name(corpus.name + ".part").exists()


def test_a_truncated_forced_refresh_keeps_the_corpus_it_was_replacing(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--force` over a good local corpus must not be able to destroy it.

    The size is checked BEFORE the `.part` is renamed over the destination, so a short transfer
    is a no-op. Checking after would already have clobbered a valid corpus with a truncated one
    — and then deleting that to keep the re-run honest turns a failed refresh into data loss.
    """
    _fake_hub(
        monkeypatch,
        {"traces.otel.jsonl": b"spans\n"},
        claimed_sizes={"traces.otel.jsonl": 4096},
    )
    corpus = data_root / "gaia2" / "traces.otel.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("the corpus that was already here\n")

    with pytest.raises(OSError, match="truncated transfer"):
        fetch_corpus("gaia2", force=True)

    assert corpus.read_text() == "the corpus that was already here\n"
    assert not corpus.with_name(corpus.name + ".part").exists()


def test_unknown_benchmark_never_offers_an_unpublished_name(data_root: Path) -> None:
    # Offering a name the Hub can only answer 401 for sends the user down a dead end, so the
    # "available:" list is the published subset, not the whole registry.
    unpublished = sorted(name for name, spec in CORPORA.items() if not spec.published)
    assert unpublished, "this test is meaningless once every registered corpus is published"
    with pytest.raises(ValueError) as caught:
        fetch_corpus("nope")
    for name in unpublished:
        assert name not in str(caught.value)
        assert name not in downloadable_benchmarks()


def test_fetch_of_an_unpublished_benchmark_fails_offline_and_says_why(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No Hub round trip at all: a registered-but-unpushed bundle is knowable locally, so the
    # user gets the reason instead of a 401 they cannot act on.
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not touch the Hub for an unpublished benchmark")

    monkeypatch.setattr(hub, "_resolve_repo", explode)
    with pytest.raises(ValueError, match="has not been published to the Hub yet"):
        fetch_corpus("kimi-gui-control")


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
        {"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T06:00:00.000Z"},
        {
            "id": "experiential-labs/wmo-bird-sql-traces",
            "lastModified": "2026-07-05T00:00:00.000Z",
        },
        {"id": "experiential-labs/unrelated-dataset", "lastModified": "2026-07-06T00:00:00.000Z"},
        {"id": "experiential-labs/wmo-not-a-benchmark-traces", "lastModified": ""},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    published = published_corpora()
    assert [(c.benchmark, c.last_modified) for c in published] == [
        ("gaia2", "2026-07-07"),
        ("bird-sql", "2026-07-05"),
    ]
    assert published[0].repo_id == candidate_repo_ids("gaia2")[0]


def test_published_corpora_ignores_legacy_prefixed_listings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-rename, a wmh-prefixed row in the org listing is stale and never offered.

    Offering it would make the picker advertise a name fetch no longer tries; the Hub
    redirect covers old CLIENTS, not old listings.
    """
    listing = [
        {"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-08-03T00:00:00.000Z"},
        {"id": "experiential-labs/wmh-bird-sql-traces", "lastModified": "2026-07-05T00:00:00.000Z"},
        {"id": "experiential-labs/unrelated-dataset", "lastModified": "2026-07-06T00:00:00.000Z"},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    published = published_corpora()
    assert [(c.benchmark, c.repo_id) for c in published] == [
        ("gaia2", "experiential-labs/wmo-gaia2-traces"),
    ]

def test_published_corpora_lists_a_double_published_benchmark_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-rename both repo names can exist at once; the picker shows one row per benchmark,
    under the canonical id (which is also the one a fetch resolves first)."""
    listing = [
        {"id": "experiential-labs/wmh-gaia2-traces", "lastModified": "2026-07-01T00:00:00.000Z"},
        {"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T00:00:00.000Z"},
    ]
    monkeypatch.setattr(hub, "_http_json_page", lambda url, *, token: (listing, None))

    assert [(c.benchmark, c.repo_id) for c in published_corpora()] == [
        ("gaia2", candidate_repo_ids("gaia2")[0])
    ]


def test_published_corpora_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """An org with more datasets than one page must not hide corpora beyond page 1."""
    pages = {
        "page1": (
            [{"id": "experiential-labs/wmo-gaia2-traces", "lastModified": "2026-07-07T00:00:00Z"}],
            "page2",
        ),
        "page2": (
            [
                {
                    "id": "experiential-labs/wmo-bird-sql-traces",
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


def test_data_root_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env override wins; otherwise bundles land under the CWD, never inside site-packages.

    There is no checkout special case any more: `packages/environment-capture/` is gone, so a
    checkout and an installed wheel resolve identically. That makes the root CWD-relative, which
    is the property worth pinning: `wmo download` and every later command that reads the bundle
    have to run from the same directory, or `ENVCAP_DATA_ROOT` has to name a fixed path.
    """
    monkeypatch.setenv("ENVCAP_DATA_ROOT", str(tmp_path / "override"))
    assert hub._data_root() == tmp_path / "override"

    monkeypatch.delenv("ENVCAP_DATA_ROOT")
    monkeypatch.chdir(tmp_path)
    assert hub._data_root() == tmp_path / "environment-capture-data"

    # Where the module itself lives is irrelevant, so a pip user's bundles land in their project
    # rather than inside site-packages.
    site = tmp_path / "venv" / "site-packages" / "wmo"
    site.mkdir(parents=True)
    monkeypatch.setattr(hub, "__file__", str(site / "hub.py"))
    assert hub._data_root() == tmp_path / "environment-capture-data"

    # It tracks the CWD, not the repo: run from elsewhere and the root moves with you.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert hub._data_root() == elsewhere / "environment-capture-data"
