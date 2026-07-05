"""Download CRMArena's official local org database and task file from the upstream sources.

The org records the agent queries live in ``crm.db`` — a ~8 MB SQLite dump of the CRMArena
Salesforce org that the official repo ships under ``local_data/crmarena_data.db``. It is gitignored
here (a binary blob) and fetched by this script so a fresh clone is runnable. The upstream task file
(the 1170 tasks with their gold answers, from the HuggingFace dataset) is fetched too and consumed
by ``build_split.py`` to (re)generate the committed train/test split. Stdlib-only (urllib).

Usage (from the repo root):
    uv run python packages/environment-capture/crmarena/fetch_data.py        # crm.db only
    uv run python packages/environment-capture/crmarena/fetch_data.py --all  # + task file + schema
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

_HERE = Path(__file__).parent

# The official SQLite dump of the CRMArena org, from the SalesforceAIResearch/CRMArena repo.
_DB_URL = "https://raw.githubusercontent.com/SalesforceAIResearch/CRMArena/main/local_data/crmarena_data.db"
# The upstream task set (queries + gold answers + per-task domain instructions) and object schema.
_HF_BASE = "https://huggingface.co/datasets/Salesforce/CRMArena/resolve/main"
_TASKS_URL = f"{_HF_BASE}/crmarena_w_metadata.json"
_SCHEMA_URL = f"{_HF_BASE}/schema_with_dependencies.json"


def _download(url: str, dest: Path) -> None:
    print(f"fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https GitHub/HuggingFace hosts
    print(f"  wrote {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also fetch crmarena_tasks.json + schema.json (needed to rebuild the split)",
    )
    args = parser.parse_args()

    _download(_DB_URL, _HERE / "crm.db")
    if args.all:
        _download(_TASKS_URL, _HERE / "crmarena_tasks.json")
        _download(_SCHEMA_URL, _HERE / "schema.json")
    print(f"done -> {_HERE}")


if __name__ == "__main__":
    main()
