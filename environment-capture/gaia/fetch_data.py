"""Materialize the offline-solvable slice of GAIA's validation split into a local data dir.

GAIA (``gaia-benchmark/GAIA``) is GATED on HuggingFace and its terms forbid resharing the
validation/test set in a crawlable format, so NOTHING this script writes is committed — the whole
``environment-capture/gaia/{data,datafiles,gold}`` tree is gitignored and rebuilt from the gated
source on each machine. Run it after ``huggingface-cli login`` with an account that has ACCEPTED
the dataset's gating terms (visit the dataset page and click "Agree and access repository");
without acceptance every download 403s.

Scope (documented here, applied below):
- **validation split only** — its gold answers are public, so we can grade locally; the test
  split has hidden answers and is never materialized.
- **levels 1 and 2** — level 3 is out of scope (a strong jump in required autonomy).
- **has one attachment** — we capture agents that solve a question FROM its file; question with no
  file typically need live web, and the workspace has no network.
- **offline-solvable file type** — a bash + ``python3`` agent can parse spreadsheets, PDFs, CSVs,
  documents, code, and structured text, but cannot transcribe audio or see raster images offline,
  so ``.mp3``/``.wav``/``.m4a`` and ``.png``/``.jpg``/``.jpeg``/``.gif``/``.mov``/``.mp4`` are
  skipped (recorded in the run summary).
- **no live-web tool** — when a task's ``Annotator Metadata`` lists a web/search/browser tool, the
  question needs the network we don't have, so it is skipped.

Tasks are re-keyed ``gaia-{split}-{i}`` (the original GAIA uuid rides in ``data.gaia_task_id``) and
seeded-split ~70/30 into train/test. Gold answers land in ``gold/<task_id>.json`` (never staged
into an agent workspace); attachments land in ``datafiles/<file_name>``.

Usage (from the repo root):
    uv sync --extra fetch                                  # pulls huggingface_hub + pyarrow
    uv run python environment-capture/gaia/fetch_data.py   # writes data/, datafiles/, gold/
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

_HERE = Path(__file__).parent
_REPO_ID = "gaia-benchmark/GAIA"
_METADATA_PATH = "2023/validation/metadata.parquet"

_IN_SCOPE_LEVELS = {"1", "2"}
# File types a text-only bash/python3 agent cannot interpret offline (no vision, no transcription).
_UNSOLVABLE_EXTS = {"mp3", "wav", "m4a", "png", "jpg", "jpeg", "gif", "mov", "mp4"}
# Substrings that mark an annotator tool as needing the live web the workspace does not have.
_WEB_TOOL_MARKERS = ("web browser", "search engine", "internet", "google search", "web search")


def _needs_web(annotator_metadata: dict[str, object] | None) -> bool:
    """True when the task's annotated tools include a web/search/browser tool."""
    if not annotator_metadata:
        return False
    tools = annotator_metadata.get("Tools")
    text = str(tools).lower() if tools is not None else ""
    return any(marker in text for marker in _WEB_TOOL_MARKERS)


def _in_scope(row: dict[str, object]) -> bool:
    level = str(row.get("Level", ""))
    file_name = str(row.get("file_name", "") or "")
    if level not in _IN_SCOPE_LEVELS or not file_name:
        return False
    if file_name.rsplit(".", 1)[-1].lower() in _UNSOLVABLE_EXTS:
        return False
    annotator = row.get("Annotator Metadata")
    return not _needs_web(annotator if isinstance(annotator, dict) else None)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Seed for the train/test shuffle")
    parser.add_argument(
        "--train-frac", type=float, default=0.7, help="Fraction of in-scope tasks used for train"
    )
    args = parser.parse_args()

    metadata_file = hf_hub_download(_REPO_ID, _METADATA_PATH, repo_type="dataset")
    rows = pq.read_table(metadata_file).to_pylist()
    in_scope = [r for r in rows if _in_scope(r)]
    in_scope.sort(key=lambda r: str(r["task_id"]))
    random.Random(args.seed).shuffle(in_scope)

    n_train = round(len(in_scope) * args.train_frac)
    splits = {"train": in_scope[:n_train], "test": in_scope[n_train:]}

    for directory in ("data", "datafiles", "gold"):
        target = _HERE / directory
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()

    downloaded = 0
    for split, split_rows in splits.items():
        records: list[dict[str, object]] = []
        for index, row in enumerate(split_rows):
            task_id = f"gaia-{split}-{index}"
            file_name = str(row["file_name"])
            source = hf_hub_download(_REPO_ID, str(row["file_path"]), repo_type="dataset")
            shutil.copy(source, _HERE / "datafiles" / file_name)
            downloaded += 1
            records.append(
                {
                    "task_id": task_id,
                    "prompt": str(row["Question"]),
                    "data": {
                        "file_name": file_name,
                        "level": str(row["Level"]),
                        "gaia_task_id": str(row["task_id"]),
                    },
                }
            )
            (_HERE / "gold" / f"{task_id}.json").write_text(
                json.dumps({"answer": str(row["Final answer"])}, ensure_ascii=False),
                encoding="utf-8",
            )
        _write_jsonl(_HERE / "data" / f"{split}.jsonl", records)

    skipped = len(rows) - len(in_scope)
    print(
        f"materialized {len(in_scope)} in-scope validation tasks "
        f"({len(splits['train'])} train / {len(splits['test'])} test, {downloaded} attachments) "
        f"from {len(rows)} total; skipped {skipped} out-of-scope (level 3 / no file / "
        f"audio-image / web-tool). data+datafiles+gold are gitignored (GAIA gating)."
    )


if __name__ == "__main__":
    main()
