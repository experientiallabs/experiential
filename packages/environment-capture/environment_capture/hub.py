"""Fetch and list trace-corpus data bundles from the Hugging Face Hub (stdlib-only).

Every publishable benchmark's bundle — the trace corpus plus its task data / gold / evidence
dirs — lives in a dataset repo under the org. This module is the READ core shared by every
front-end (`wmh download`, the serving API's trace-download endpoint, `python -m
environment_capture.hub fetch`): plain-HTTP against the Hub's public REST API, so it needs no
extra dependency and no token for public repos (pass ``token`` for private ones). Uploading
lives in `environment_capture.hub_push` (requires the ``fetch`` extra).

Bundles are local-first: capture writes into the benchmark dir, nothing here deletes local
files, and fetching never overwrites an existing file unless forced. Downloads stream to a
``.part`` sibling and are atomically renamed, so a failed fetch never looks like a corpus.

Usage (from the repo root):
    uv run wmh download                                              # interactive picker
    uv run python -m environment_capture.hub fetch dabstep           # skip if already present
    uv run python -m environment_capture.hub fetch all --force       # overwrite local copies
    uv run python -m environment_capture.hub push bird-sql           # see hub_push
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_ORG = "experiential-labs"
_CORPUS_FILE = "traces.otel.jsonl"
_HUB = "https://huggingface.co"
_CHUNK_BYTES = 1 << 20

# on_progress(bytes_done, bytes_total): called after every streamed chunk, across ALL files in
# the fetch (front-ends render one bar for the whole bundle).
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class CorpusSpec:
    """One publishable corpus: where it lives locally and how its dataset card reads."""

    benchmark: str  # dir name under packages/environment-capture/
    license_id: str  # Hub license identifier (must match the upstream terms)
    upstream: str  # attribution line for the dataset card
    description: str  # one-sentence environment summary
    extra_terms: str = ""  # disclosures that ride below the boilerplate
    # Data payload dirs published alongside the trace corpus (task indexes, gold sidecars,
    # evidence docs, ...). Same license as the corpus — they ARE the upstream-derived data.
    data_dirs: tuple[str, ...] = ()


# The publishable corpora. appworld is deliberately ABSENT: its protected data may only be
# redistributed in encrypted form (plain-text posting is disallowed), so that corpus stays
# local-only — see packages/environment-capture/appworld/README.md § License.
CORPORA: dict[str, CorpusSpec] = {
    spec.benchmark: spec
    for spec in (
        CorpusSpec(
            benchmark="financebench",
            data_dirs=("data", "gold", "corpus"),
            license_id="cc-by-nc-4.0",
            upstream="PatronusAI/financebench (CC BY-NC 4.0)",
            description=(
                "Financial-document QA over real SEC-filing evidence excerpts: the agent greps "
                "a workspace of evidence docs plus distractors and submits a final answer."
            ),
        ),
        CorpusSpec(
            benchmark="bird-sql",
            data_dirs=("data", "gold", "schemas"),
            license_id="cc-by-sa-4.0",
            upstream="bird-bench mini-dev (CC BY-SA 4.0)",
            description=(
                "Text-to-SQL over real SQLite databases: the agent explores a copy of the "
                "task's database and schema, then submits a SQL query."
            ),
        ),
        CorpusSpec(
            benchmark="continual-learning",
            data_dirs=("data", "gold"),
            license_id="cc-by-4.0",
            upstream="Continual Learning Bench (CC BY 4.0)",
            description=(
                "Database-exploration QA over a large, deliberately obfuscated SQLite product "
                "database: the agent maps cryptic tables to answer catalog questions."
            ),
        ),
        CorpusSpec(
            benchmark="dabstep",
            data_dirs=("data", "gold", "datafiles"),
            license_id="cc-by-4.0",
            upstream="adyen/DABstep (CC BY 4.0)",
            description=(
                "Data-analysis QA over a shared payments dataset and a business-rules manual, "
                "answered with pandas in a Python shell."
            ),
        ),
        CorpusSpec(
            benchmark="crmarena",
            data_dirs=("data", "gold"),
            license_id="cc-by-nc-4.0",
            upstream="Salesforce CRMArena (CC BY-NC 4.0)",
            description=(
                "Professional CRM analytics over a realistic Salesforce org snapshot: case "
                "routing, handle-time analytics, and entity disambiguation via SQL."
            ),
        ),
        CorpusSpec(
            benchmark="gaia2",
            data_dirs=("data",),
            license_id="cc-by-4.0",
            upstream="meta-agents-research-environments/gaia2 (CC-BY-4.0, attribution to Meta)",
            description=(
                "A stateful multi-app simulated world (Contacts, Email, Calendar, Shopping, "
                "...): the agent drives app tools with Python against live scenario state."
            ),
            extra_terms=(
                "Rewards in this corpus come from a deterministic STRUCTURAL grader "
                "(oracle-action matching), not GAIA2's official judge — scores are not "
                "comparable to the official leaderboard. GAIA2's authors ask that models not "
                "be trained on evaluation data; the test split carries that request."
            ),
        ),
        CorpusSpec(
            benchmark="tau-bench",
            license_id="mit",
            upstream="sierra-research/tau2-bench (MIT)",
            description=(
                "Customer-service tool-agent episodes from the real tau2-bench harness "
                "(airline/retail/telecom domains)."
            ),
        ),
        CorpusSpec(
            benchmark="terminal-tasks",
            license_id="apache-2.0",
            upstream="terminal-bench (Apache 2.0)",
            description=(
                "Computer-use agent runs in real terminal containers: bash commands and their "
                "true outputs from live task environments."
            ),
        ),
        CorpusSpec(
            benchmark="swe-bench",
            license_id="mit",
            upstream="princeton-nlp/SWE-bench Verified + mini-swe-agent (MIT)",
            description=(
                "Software-engineering agent runs from real SWE-bench Verified instances: shell "
                "exploration and repo edits inside per-instance Docker images."
            ),
        ),
    )
}


@dataclass(frozen=True)
class PublishedCorpus:
    """One live dataset repo under the org, mapped back to its benchmark."""

    benchmark: str
    repo_id: str
    last_modified: str  # ISO date, "" when the Hub omits it


def repo_id_for(benchmark: str) -> str:
    """The dataset repo backing one benchmark's corpus."""
    return f"{_ORG}/wmh-{benchmark}-traces"


def corpus_path(benchmark: str) -> Path:
    """The canonical local path of the benchmark's trace corpus (whether or not it exists yet).

    This is the "is it local, and where" resolver every front-end shares: check
    ``corpus_path(b).exists()`` before deciding to download or to serve from disk.
    """
    return _data_root() / benchmark / _CORPUS_FILE


def published_corpora(*, token: str | None = None) -> list[PublishedCorpus]:
    """The org's live corpus datasets (Hub REST API), newest first, mapped to benchmark names.

    Only repos that follow the ``wmh-<benchmark>-traces`` convention AND appear in the local
    manifest are returned — those are the ones ``fetch_corpus`` knows where to place.
    """
    listing = _http_json(f"{_HUB}/api/datasets?author={_ORG}&limit=100", token=token)
    published: list[PublishedCorpus] = []
    for entry in listing if isinstance(listing, list) else []:
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("id", ""))
        name = repo_id.removeprefix(f"{_ORG}/")
        if not (name.startswith("wmh-") and name.endswith("-traces")):
            continue
        benchmark = name.removeprefix("wmh-").removesuffix("-traces")
        if benchmark not in CORPORA:
            continue
        modified = str(entry.get("lastModified") or "")
        published.append(
            PublishedCorpus(benchmark=benchmark, repo_id=repo_id, last_modified=modified[:10])
        )
    published.sort(key=lambda c: c.last_modified, reverse=True)
    return published


def fetch_corpus(
    benchmark: str,
    *,
    dest: Path | None = None,
    force: bool = False,
    token: str | None = None,
    revision: str = "main",
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download the benchmark's corpus AND published data dirs into place; returns the corpus path.

    Local-first: existing files/dirs are kept unless ``force=True`` — fetching must never
    silently clobber a corpus that local capture waves have grown past the published one.
    With an explicit ``dest`` only the corpus file is written (no data dirs).
    ``on_progress(bytes_done, bytes_total)`` fires per streamed chunk across the whole bundle.
    """
    spec = CORPORA.get(benchmark)
    if spec is None:
        publishable = ", ".join(sorted(CORPORA))
        raise ValueError(f"{benchmark!r} has no published corpus (available: {publishable})")
    repo_id = repo_id_for(benchmark)
    target = dest or corpus_path(benchmark)

    # Work list: (remote path, local target, size). Sizes come from the tree API so the total
    # is known up front and one progress bar can cover the whole bundle.
    work: list[tuple[str, Path, int]] = []
    if not target.exists() or force:
        (corpus_entry,) = [
            e for e in _repo_tree(repo_id, revision, token=token) if e[0] == _CORPUS_FILE
        ] or [(_CORPUS_FILE, 0)]
        work.append((_CORPUS_FILE, target, corpus_entry[1]))
    if dest is None:
        pending = [
            d for d in spec.data_dirs if force or not (_data_root() / benchmark / d).is_dir()
        ]
        for data_dir in pending:
            for remote_path, size in _repo_tree(repo_id, revision, subpath=data_dir, token=token):
                local = _data_root() / benchmark / remote_path
                work.append((remote_path, local, size))

    total = sum(size for _, _, size in work)
    done = 0
    for remote_path, local, _size in work:
        url = f"{_HUB}/datasets/{repo_id}/resolve/{revision}/{remote_path}"

        def chunk_done(n: int) -> None:
            nonlocal done
            done += n
            if on_progress is not None:
                on_progress(done, total)

        _stream_to(url, local, token=token, chunk_done=chunk_done)
    return target


def _data_root() -> Path:
    """The sibling benchmark data dirs (packages/environment-capture/<benchmark>/)."""
    return Path(__file__).resolve().parents[1]


def _request(url: str, token: str | None) -> urllib.request.Request:
    headers = {"User-Agent": "environment-capture/hub"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)  # noqa: S310 - https-only constants


def _http_json(url: str, *, token: str | None) -> object:
    with urllib.request.urlopen(_request(url, token), timeout=60) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _repo_tree(
    repo_id: str, revision: str, *, subpath: str = "", token: str | None
) -> list[tuple[str, int]]:
    """(path, size) for every FILE under ``subpath`` in the dataset repo."""
    suffix = f"/{subpath}" if subpath else ""
    url = f"{_HUB}/api/datasets/{repo_id}/tree/{revision}{suffix}?recursive=true"
    listing = _http_json(url, token=token)
    files: list[tuple[str, int]] = []
    for entry in listing if isinstance(listing, list) else []:
        if isinstance(entry, dict) and entry.get("type") == "file":
            size = entry.get("size")
            files.append(
                (str(entry.get("path", "")), size if isinstance(size, int) else 0)
            )
    return files


def _stream_to(
    url: str, dest: Path, *, token: str | None, chunk_done: Callable[[int], None]
) -> None:
    """Stream ``url`` to ``dest`` atomically: write a ``.part`` sibling, then rename over.

    A partially-downloaded corpus must never be mistaken for a complete one by a concurrent
    reader (the serving API polls the target path for byte progress).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(_request(url, token), timeout=300) as response:  # noqa: S310
        with part.open("wb") as sink:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                chunk_done(len(chunk))
    os.replace(part, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="Create/update dataset repo(s) from local corpora.")
    push.add_argument("benchmark", help=f"Benchmark name, or 'all' ({', '.join(sorted(CORPORA))})")
    push.add_argument("--private", action="store_true", help="Create the repo(s) private.")

    fetch = sub.add_parser("fetch", help="Download full data bundles into the benchmark dirs.")
    fetch.add_argument("benchmark", help="Benchmark name, or 'all'")
    fetch.add_argument(
        "--force", action="store_true", help="Overwrite existing local corpus/data files."
    )

    args = parser.parse_args()
    if args.command == "push":
        # The write path is the one place that needs huggingface_hub (the `fetch` extra); the
        # import lives at the dispatch so plain fetch installs never pay for it. A missing
        # extra fails loudly right here with the install hint.
        try:
            from environment_capture import hub_push
        except ModuleNotFoundError as error:  # pragma: no cover - exercised only without extra
            raise SystemExit(
                "pushing needs huggingface_hub: install the extra "
                "(`pip install 'environment-capture[fetch]'` / `uv sync --extra dev`)"
            ) from error
    names = sorted(CORPORA) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        if args.command == "push":
            url = hub_push.push_corpus(name, private=args.private)
            print(f"pushed {name} -> {url}")
        else:
            existing = corpus_path(name).exists()
            path = fetch_corpus(name, force=args.force)
            state = "kept local" if existing and not args.force else "fetched"
            print(f"{state} {name} -> {path}")


if __name__ == "__main__":
    main()
