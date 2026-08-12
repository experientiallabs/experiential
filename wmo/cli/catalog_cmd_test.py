"""Benchmark download command tests."""

# ruff: noqa: F403, F405
from wmo.cli.cli_fixtures_test import *


def test_download_fetches_named_benchmarks(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fetched: list[tuple[str, bool]] = []

    def fake_fetch(name: str, *, force: bool = False, on_progress=None) -> Path:  # noqa: ANN001
        fetched.append((name, force))
        return tmp_path / name / "traces.otel.jsonl"

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fake_fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "bird-sql", "dabstep", "--force"])
    assert result.exit_code == 0, result.output
    assert fetched == [("bird-sql", True), ("dabstep", True)]
    assert "fetched" in result.output


def test_download_all_expands_to_the_published_list(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # `all` means "everything actually on the Hub" (live list), not the static registry - a
    # registry entry that isn't published yet would 404.
    fetched: list[str] = []
    published = [SimpleNamespace(benchmark=n, last_modified=None) for n in ("a-bench", "b-bench")]
    monkeypatch.setattr("wmo.simulation.hub.published_corpora", lambda: published)
    monkeypatch.setattr(
        "wmo.simulation.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "all"])
    assert result.exit_code == 0, result.output
    assert fetched == ["a-bench", "b-bench"]


def test_download_multi_skips_a_404_and_fetches_the_rest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # One unpublished dataset must not abort the remaining downloads (it used to kill `all`
    # mid-loop, alphabetically stranding everything after the 404).
    import urllib.error

    fetched: list[str] = []

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name == "broken":
            raise urllib.error.HTTPError("https://hub/x", 404, "nf", None, None)  # ty: ignore[invalid-argument-type]
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "a-bench", "broken", "z-bench"])
    assert fetched == ["a-bench", "z-bench"]  # kept going past the 404
    assert result.exit_code != 0  # ...but the failure is still reported at the end
    assert "broken" in result.output


def test_download_all_offline_skips_the_unpublished_and_still_succeeds(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Offline, `all` falls back to the local registry. That registry names bundles registered
    # here so the write side knows how to publish them but never pushed, and the Hub can only
    # answer 401 for those - which used to turn an otherwise complete `wmo download all` into a
    # failed command over something the user cannot act on. The fallback is the published
    # subset, and it says what it dropped.
    import urllib.error

    from wmo.simulation.hub import CORPORA, downloadable_benchmarks

    unpublished = sorted(n for n, spec in CORPORA.items() if not spec.published)
    assert unpublished, "this test is meaningless once every registered corpus is published"
    fetched: list[str] = []

    def no_catalogue(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("wmo.simulation.hub.published_corpora", no_catalogue)
    monkeypatch.setattr(
        "wmo.simulation.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "all"])
    assert result.exit_code == 0, result.output  # no failure over an unpushed registry entry
    assert fetched == downloadable_benchmarks()
    for name in unpublished:
        assert name not in fetched
        assert name in result.output  # the narrowing is announced, never silent


def test_download_multi_keeps_going_past_a_truncated_transfer(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # A file still short after `fetch_corpus`'s own per-file retries raises OSError, which used
    # to escape the loop's per-item handling and kill the command - so a bundle the Hub served
    # badly stranded every benchmark queued behind it, exactly like the 404 above once did.
    fetched: list[str] = []

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name == "short":
            raise OSError("traces.otel.jsonl: downloaded 6 bytes but the Hub tree lists 4096")
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "a-bench", "short", "z-bench"])
    assert fetched == ["a-bench", "z-bench"]  # kept going past the short transfer
    assert result.exit_code != 0  # ...but the failure is still reported at the end
    assert "short" in result.output


def test_download_of_one_bundle_reports_a_truncated_transfer_as_a_failure(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Alone it is a runtime failure, not a usage error: the name was fine, the transfer was not.
    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        raise OSError("traces.otel.jsonl: 6 bytes, tree lists 4096 - truncated transfer")

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "dabstep"])
    assert result.exit_code == 1
    assert "truncated transfer" in result.output
    assert "Invalid value" not in result.output


def test_download_multi_reports_an_unknown_name_without_stranding_the_rest(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Same defect class, decided offline before the network is touched: one bad name in a
    # hand-typed list used to abort the command before the good ones were attempted.
    fetched: list[str] = []

    from wmo.simulation.hub import CORPORA

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name not in CORPORA:
            raise ValueError(f"{name!r} has no published corpus (available: dabstep)")
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "nope", "dabstep"])
    assert fetched == ["dabstep"]
    assert result.exit_code != 0
    assert "nope" in result.output


def test_download_failure_names_every_repo_id_it_tried(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # A fetch tries more than one dataset repo name (the wmh -> wmo rename), so a bare "404"
    # cannot be acted on: the report must say which ids were looked for. The CLI reads that off
    # plain HTTPError attributes rather than the subclass, so the report survives any fetcher
    # that raises a stock HTTPError.
    import urllib.error
    from http.client import HTTPMessage

    from wmo.simulation.hub import CorpusRepoUnavailable, candidate_repo_ids

    attempts = [
        (repo_id, urllib.error.HTTPError(f"https://hub/{repo_id}", 404, "nf", HTTPMessage(), None))
        for repo_id in candidate_repo_ids("dabstep")
    ]

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        raise CorpusRepoUnavailable(name, "main", attempts)

    monkeypatch.setattr("wmo.simulation.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "dabstep"])
    assert result.exit_code != 0
    for repo_id in candidate_repo_ids("dabstep"):
        assert repo_id in result.output


def test_download_unknown_benchmark_is_a_usage_error(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "nope"])
    assert result.exit_code != 0
    assert "no published corpus" in result.output


def test_download_picker_lists_published_and_fetches_choice(
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    from wmo.simulation.hub import PublishedCorpus

    published = [
        PublishedCorpus(
            benchmark="gaia2",
            repo_id="experiential-labs/wmo-gaia2-traces",
            last_modified="2026-07-06",
        )
    ]
    fetched: list[str] = []
    monkeypatch.setattr("wmo.simulation.hub.published_corpora", lambda: published)
    monkeypatch.setattr(
        "wmo.simulation.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.simulation.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download"], input="1\n")
    assert result.exit_code == 0, result.output
    assert fetched == ["gaia2"]
    assert "not downloaded" in result.output  # picker showed local status
