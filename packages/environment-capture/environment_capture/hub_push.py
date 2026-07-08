"""Publish trace-corpus data bundles to the Hugging Face Hub (the WRITE side of `hub`).

Split from `environment_capture.hub` so the read path stays dependency-free: this module needs
``huggingface_hub`` (the ``fetch`` extra) and a token with write access (``hf auth login`` or
``HF_TOKEN``). Updating a published bundle is just pushing again — the Hub records each push as
a commit, so history is kept and downloads always see the latest data.

Usage (from the repo root):
    uv run python -m environment_capture.hub_push bird-sql          # create/update, public
    uv run python -m environment_capture.hub_push all --private
"""

from __future__ import annotations

import argparse
from typing import Protocol

from huggingface_hub import HfApi

from environment_capture.hub import _CORPUS_FILE, CORPORA, CorpusSpec, corpus_path, repo_id_for


class HubApi(Protocol):
    """The slice of ``HfApi`` this module uses (injectable for tests)."""

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> None: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None: ...

    def upload_folder(
        self,
        *,
        folder_path: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None: ...


# One blurb per publishable data dir name; a test asserts every declared CorpusSpec.data_dir
# has an entry, so a new dir can't silently publish with a generic card line.
DIR_BLURBS: dict[str, str] = {
    "data": "task index (train/test splits: prompts + task metadata)",
    "gold": "per-task gold sidecars (graders read these; never staged into agent workspaces)",
    "corpus": "evidence documents the tasks are answered from",
    "datafiles": "shared context files (manual, datasets) staged into agent workspaces",
    "schemas": "database DDL per task database",
}


def _dataset_card(spec: CorpusSpec) -> str:
    """The dataset card (README.md with Hub YAML frontmatter) for one corpus."""
    extra = f"\n{spec.extra_terms}\n" if spec.extra_terms else ""
    data_dir_lines = "".join(
        f"- `{d}/` — {DIR_BLURBS[d]}\n" for d in spec.data_dirs
    )
    return f"""---
license: {spec.license_id}
pretty_name: "{spec.benchmark} agent-environment traces (world-model-harness)"
language:
- en
tags:
- agent-trajectories
- world-models
- llm-environments
---

# {spec.benchmark} — real agent-environment traces

{spec.description}

Every trace is a REAL run: an LLM agent stepping against the actual benchmark environment, with
each transition (tool call → true environment observation) recorded as OpenTelemetry GenAI spans
(`{_CORPUS_FILE}`, one span per line). Captured by
[world-model-harness](https://github.com/experientiallabs/world-model-harness)'s
`environment-capture` package, which also holds the adapter, capture scripts, and per-corpus
provenance: see
[`packages/environment-capture/{spec.benchmark}/`](https://github.com/experientiallabs/world-model-harness/tree/main/packages/environment-capture/{spec.benchmark}).

## License and attribution

Derived from **{spec.upstream}**; this corpus is redistributed under the same terms
(`{spec.license_id}`). The trace text embeds task data and environment output from the upstream
benchmark — keep this attribution if you redistribute.
{extra}
## Contents

- `traces.otel.jsonl` — the trace corpus (OTel GenAI spans, one JSON object per line)
{data_dir_lines}
## Using it

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "{repo_id_for(spec.benchmark)}", "{_CORPUS_FILE}", repo_type="dataset"
)
```

or, from a world-model-harness checkout:

```bash
uv run wmh download {spec.benchmark}
```
"""


def push_corpus(
    benchmark: str,
    *,
    private: bool = False,
    token: str | None = None,
    api: HubApi | None = None,
) -> str:
    """Create/update the benchmark's dataset repo from the local bundle; returns the repo URL.

    Re-pushing after local capture waves is the update path: the Hub records each push as a
    commit, so history is kept and downloads always see the latest corpus.
    """
    spec = CORPORA.get(benchmark)
    if spec is None:
        publishable = ", ".join(sorted(CORPORA))
        raise ValueError(
            f"{benchmark!r} is not a publishable corpus (available: {publishable}). "
            "appworld is local-only: its license forbids plain-text redistribution."
        )
    corpus = corpus_path(benchmark)
    if not corpus.exists():
        raise FileNotFoundError(
            f"no local corpus at {corpus}; capture one first (see the benchmark README)"
        )
    hub = api or HfApi(token=token)
    repo_id = repo_id_for(benchmark)
    hub.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    hub.upload_file(
        path_or_fileobj=str(corpus),
        path_in_repo=_CORPUS_FILE,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"update {benchmark} corpus",
    )
    for data_dir in spec.data_dirs:
        local_dir = corpus.parent / data_dir
        if not local_dir.is_dir():
            raise FileNotFoundError(
                f"declared data dir {local_dir} is missing; fetch or rebuild it before pushing"
            )
        hub.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=data_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"update {benchmark} {data_dir}/",
        )
    hub.upload_file(
        path_or_fileobj=_dataset_card(spec).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"update {benchmark} dataset card",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def add_hub_args(parser: argparse.ArgumentParser) -> None:
    """Wire the optional post-capture Hub push into a capture script's CLI."""
    parser.add_argument(
        "--push-hub",
        action="store_true",
        help="After capture, push the corpus to its Hub dataset repo (needs a write token via "
        "`hf auth login` or HF_TOKEN). The local file always stays.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the dataset repo private (matters on the first push only).",
    )


def push_after_capture(benchmark: str, *, enabled: bool, private: bool) -> None:
    """The capture scripts' post-run hook: push when ``--push-hub`` was passed, else no-op."""
    if not enabled:
        return
    url = push_corpus(benchmark, private=private)
    print(f"pushed corpus -> {url}")


def main() -> None:
    """CLI: publish/update dataset repo(s) from local bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", help=f"Benchmark name, or 'all' ({', '.join(sorted(CORPORA))})"
    )
    parser.add_argument("--private", action="store_true", help="Create the repo(s) private.")
    args = parser.parse_args()
    names = sorted(CORPORA) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        url = push_corpus(name, private=args.private)
        print(f"pushed {name} -> {url}")


if __name__ == "__main__":
    main()
