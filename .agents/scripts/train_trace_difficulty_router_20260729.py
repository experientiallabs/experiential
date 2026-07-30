"""Train a transfer difficulty model from a bounded Open-SWE-Traces sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fsspec
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HF_BASE = "https://huggingface.co/datasets"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default="nvidia/Open-SWE-Traces")
    parser.add_argument("--config", default="openhands")
    parser.add_argument("--split", default="minimax_m25")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def _url(repo: str, config: str, split: str, shard: int) -> str:
    prefix = f"{split}_{config}_trajectories"
    return f"{HF_BASE}/{repo}/resolve/main/data/{prefix}/train-{shard:05d}-of-00020.parquet"


def _first_user(trajectory: list[dict[str, Any]]) -> str:
    for step in trajectory:
        if step.get("role") == "user" and isinstance(step.get("content"), str):
            return step["content"]
    return ""


def _extract_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in range(args.shards):
        url = _url(args.repo, args.config, args.split, shard)
        with fsspec.open(url, "rb", block_size=1024 * 1024, cache_type="readahead") as handle:
            table = parquet.ParquetFile(handle).read(
                columns=["instance_id", "repo", "language", "trajectory", "resolved"],
                use_threads=False,
            )
        for row in table.to_pylist():
            label = row.get("resolved")
            if label not in (0, 1):
                continue
            trajectory = row.get("trajectory") or []
            prompt = _first_user(trajectory)
            rows.append(
                {
                    "instance_id": row.get("instance_id"),
                    "repo": row.get("repo"),
                    "language": row.get("language") or "unknown",
                    "prompt": prompt,
                    "trajectory_steps": len(trajectory),
                    "tool_calls": sum(len(step.get("tool_calls") or []) for step in trajectory),
                    "prompt_chars": len(prompt),
                    "resolved": int(label),
                }
            )
            if len(rows) >= args.max_rows:
                return rows
    return rows


def main() -> None:
    args = _parser().parse_args()
    if args.shards < 1 or args.max_rows < 100:
        raise ValueError("--shards must be positive and --max-rows must be at least 100")
    rows = _extract_rows(args)
    if len({row["resolved"] for row in rows}) < 2:
        raise ValueError("trace sample has only one resolved class")
    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=[rows[index]["resolved"] for index in indices],
    )
    text = TfidfVectorizer(
        max_features=30_000,
        min_df=2,
        ngram_range=(1, 2),
        strip_accents="unicode",
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("prompt", text, "prompt"),
            ("language", OneHotEncoder(handle_unknown="ignore"), ["language"]),
            (
                "numeric",
                StandardScaler(with_mean=False),
                ["trajectory_steps", "tool_calls", "prompt_chars"],
            ),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocess),
            ("classifier", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    fields = ("prompt", "language", "trajectory_steps", "tool_calls", "prompt_chars")
    x = pd.DataFrame([{key: row[key] for key in fields} for row in rows])
    y = np.asarray([row["resolved"] for row in rows])
    model.fit(x.iloc[train_idx], y[train_idx])
    probability = model.predict_proba(x.iloc[test_idx])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    metadata = {
        "dataset": {
            "repo": args.repo,
            "config": args.config,
            "split": args.split,
            "shards": args.shards,
            "rows_used": len(rows),
            "positive_rate": float(y.mean()),
        },
        "features": [
            "prompt_tfidf",
            "language_one_hot",
            "trajectory_steps",
            "tool_calls",
            "prompt_chars",
        ],
        "seed": args.seed,
        "heldout_metrics": {
            "accuracy": float(accuracy_score(y[test_idx], prediction)),
            "roc_auc": float(roc_auc_score(y[test_idx], probability)),
        },
        "model": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
