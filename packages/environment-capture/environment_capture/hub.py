"""Publish and fetch trace corpora on the Hugging Face Hub.

Corpora are local-first: capture always writes `traces.otel.jsonl` into the benchmark dir, and
nothing here deletes it. This module adds the sharing layer on top — push a corpus to a dataset
repo (public or private), fetch the full set back when you need more than a committed sample.
Updating a corpus is just pushing again: the Hub versions every upload as a commit.

Requires the ``fetch`` extra (``environment-capture[fetch]``) and a Hub token with write access
(``hf auth login`` or the ``HF_TOKEN`` env var; fetching public datasets needs no token).

Usage (from the repo root):
    uv run python -m environment_capture.hub push bird-sql          # create/update, public
    uv run python -m environment_capture.hub push all --private
    uv run python -m environment_capture.hub fetch dabstep          # skip if already present
    uv run python -m environment_capture.hub fetch all --force      # overwrite local copies
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download

_ORG = "experiential-labs"
_CORPUS_FILE = "traces.otel.jsonl"


class HubApi(Protocol):
    """The slice of ``HfApi`` this module uses (injectable for tests)."""

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> object: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> object: ...


@dataclass(frozen=True)
class CorpusSpec:
    """One publishable corpus: where it lives locally and how its dataset card reads."""

    benchmark: str  # dir name under packages/environment-capture/
    license_id: str  # Hub license identifier (must match the upstream terms)
    upstream: str  # attribution line for the dataset card
    description: str  # one-sentence environment summary
    extra_terms: str = ""  # disclosures that ride below the boilerplate


# The publishable corpora. appworld is deliberately ABSENT: its protected data may only be
# redistributed in encrypted form (plain-text posting is disallowed), so that corpus stays
# local-only — see packages/environment-capture/appworld/README.md § License.
CORPORA: dict[str, CorpusSpec] = {
    spec.benchmark: spec
    for spec in (
        CorpusSpec(
            benchmark="financebench",
            license_id="cc-by-nc-4.0",
            upstream="PatronusAI/financebench (CC BY-NC 4.0)",
            description=(
                "Financial-document QA over real SEC-filing evidence excerpts: the agent greps "
                "a workspace of evidence docs plus distractors and submits a final answer."
            ),
        ),
        CorpusSpec(
            benchmark="bird-sql",
            license_id="cc-by-sa-4.0",
            upstream="bird-bench mini-dev (CC BY-SA 4.0)",
            description=(
                "Text-to-SQL over real SQLite databases: the agent explores a copy of the "
                "task's database and schema, then submits a SQL query."
            ),
        ),
        CorpusSpec(
            benchmark="continual-learning",
            license_id="cc-by-4.0",
            upstream="Continual Learning Bench (CC BY 4.0)",
            description=(
                "Database-exploration QA over a large, deliberately obfuscated SQLite product "
                "database: the agent maps cryptic tables to answer catalog questions."
            ),
        ),
        CorpusSpec(
            benchmark="dabstep",
            license_id="cc-by-4.0",
            upstream="adyen/DABstep (CC BY 4.0)",
            description=(
                "Data-analysis QA over a shared payments dataset and a business-rules manual, "
                "answered with pandas in a Python shell."
            ),
        ),
        CorpusSpec(
            benchmark="crmarena",
            license_id="cc-by-nc-4.0",
            upstream="Salesforce CRMArena (CC BY-NC 4.0)",
            description=(
                "Professional CRM analytics over a realistic Salesforce org snapshot: case "
                "routing, handle-time analytics, and entity disambiguation via SQL."
            ),
        ),
        CorpusSpec(
            benchmark="gaia2",
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


def repo_id_for(benchmark: str) -> str:
    """The dataset repo backing one benchmark's corpus."""
    return f"{_ORG}/wmh-{benchmark}-traces"


def _data_root() -> Path:
    """The sibling benchmark data dirs (packages/environment-capture/<benchmark>/)."""
    return Path(__file__).resolve().parents[1]


def _dataset_card(spec: CorpusSpec) -> str:
    """The dataset card (README.md with Hub YAML frontmatter) for one corpus."""
    extra = f"\n{spec.extra_terms}\n" if spec.extra_terms else ""
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
## Using it

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "{repo_id_for(spec.benchmark)}", "{_CORPUS_FILE}", repo_type="dataset"
)
```

or, from a world-model-harness checkout:

```bash
uv run python -m environment_capture.hub fetch {spec.benchmark}
```
"""


def push_corpus(
    benchmark: str,
    *,
    private: bool = False,
    token: str | None = None,
    api: HubApi | None = None,
) -> str:
    """Create/update the benchmark's dataset repo from the local corpus; returns the repo URL.

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
    corpus = _data_root() / benchmark / _CORPUS_FILE
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
    hub.upload_file(
        path_or_fileobj=_dataset_card(spec).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"update {benchmark} dataset card",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def fetch_corpus(
    benchmark: str,
    *,
    dest: Path | None = None,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download the benchmark's full corpus into its data dir (or ``dest``); returns the path.

    Local-first: an existing file is kept unless ``force=True`` — fetching must never silently
    clobber a corpus that local capture waves have grown past the published one.
    """
    if benchmark not in CORPORA:
        publishable = ", ".join(sorted(CORPORA))
        raise ValueError(f"{benchmark!r} has no published corpus (available: {publishable})")
    target = dest or _data_root() / benchmark / _CORPUS_FILE
    if target.exists() and not force:
        return target
    downloaded = hf_hub_download(
        repo_id_for(benchmark), _CORPUS_FILE, repo_type="dataset", token=token
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, target)
    return target


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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="Create/update dataset repo(s) from local corpora.")
    push.add_argument("benchmark", help=f"Benchmark name, or 'all' ({', '.join(sorted(CORPORA))})")
    push.add_argument("--private", action="store_true", help="Create the repo(s) private.")

    fetch = sub.add_parser("fetch", help="Download full corpora into the benchmark data dirs.")
    fetch.add_argument("benchmark", help="Benchmark name, or 'all'")
    fetch.add_argument(
        "--force", action="store_true", help="Overwrite an existing local corpus file."
    )

    args = parser.parse_args()
    names = sorted(CORPORA) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        if args.command == "push":
            url = push_corpus(name, private=args.private)
            print(f"pushed {name} -> {url}")
        else:
            existing = (_data_root() / name / _CORPUS_FILE).exists()
            path = fetch_corpus(name, force=args.force)
            state = "kept local" if existing and not args.force else "fetched"
            print(f"{state} {name} -> {path}")


if __name__ == "__main__":
    main()
