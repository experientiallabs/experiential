"""Fetch the shared products database from the TRUE upstream (Continual Learning Bench on HF).

The task questions + gold sidecars are committed; only the ~400 MB shared SQLite ``products.db``
is fetched on demand (it is gitignored). It is pulled from the upstream HuggingFace dataset
``continual-learning-benchmark/continual-learning-bench-data`` (CC BY 4.0), subset
``database_exploration``, and written read-only (0444) so concurrent capture runs cannot mutate or
corrupt the single shared copy.

Gated behind ``--confirm`` and needs the optional ``huggingface_hub`` dependency, so it never runs
in CI.

Usage:
    uv run python environment-capture/continual-learning/fetch_data.py --confirm
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

_HF_REPO = "continual-learning-benchmark/continual-learning-bench-data"
_SUBSET = "database_exploration"
_DB_FILENAME = "products.db"
_HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm", action="store_true", help="Required: acknowledge the ~400 MB download"
    )
    parser.add_argument(
        "--out",
        default=str(_HERE / "datafiles" / _DB_FILENAME),
        help="Where to write the shared products.db (gitignored)",
    )
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("refusing to download ~400 MB without --confirm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(_HF_REPO, f"{_SUBSET}/{_DB_FILENAME}", repo_type="dataset")
    # Copy out of the HF cache so the read-only chmod does not fight hub cache management.
    if out.exists():
        out.chmod(0o644)
    shutil.copyfile(cached, out)
    out.chmod(0o444)
    size_mb = os.path.getsize(out) / 1e6
    print(f"fetched {_HF_REPO}:{_SUBSET}/{_DB_FILENAME} -> {out} ({size_mb:.0f} MB, read-only)")


if __name__ == "__main__":
    main()
