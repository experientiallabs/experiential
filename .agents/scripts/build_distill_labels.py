"""Rung B label generation: self-distill keep/drop labels from a frontier compressor.

The LLMLingua-2 recipe on OUR transcripts: a frontier model compresses each
live-structure segment under a preserve-all-information instruction; the output is
required to be an exact word-subsequence of the input (extractive, delete-only); greedy
alignment turns it into per-word keep/drop labels for training the 177M scorer.
Non-subsequence outputs are DISCARDED and counted (the discard rate is part of the
deliverable, and a high rate is evidence against the teacher, not silently patched).

SPEND GATE: this script refuses to run without --approved-cap-usd, which must match the
master-approved figure recorded in DECISIONS.md (2026-07-27 C1 round 2 entry). It meters
input/output tokens per call at list price and hard-stops at the cap.

Usage:
    uv run python .agents/scripts/build_distill_labels.py --pilot          # 20 segments
    uv run python .agents/scripts/build_distill_labels.py --approved-cap-usd 30
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from wmo.core.types import Message
from wmo.providers.pool import load_pool, pool_provider

log = logging.getLogger("distill_labels")

DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
SEGMENTS_GLOB = "live-segments-*.jsonl"
OUT_DIR = DATA_ROOT / "cache/distill-labels"
LABEL_MODEL = "gpt-5.5"
PRICE_IN, PRICE_OUT = 5.0, 30.0  # USD per Mtok, list
CAP_PER_CORPUS = 2000
WORD_RE = re.compile(r"\S+\s*")
SEED = 0

INSTRUCTION = (
    "You compress context for an AI agent. Delete words whose removal loses no "
    "information the agent could need to act: keep every number, identifier, file path, "
    "code token, error message, entity name, and factual statement; drop filler, "
    "pleasantries, boilerplate, and redundancy. HARD RULE: the output must be an exact "
    "subsequence of the input's whitespace-delimited words (delete whole words only; "
    "never rewrite, reorder, merge, or add words). Output ONLY the compressed text."
)


def align_labels(raw: str, compressed: str) -> list[int] | None:
    """Greedy subsequence alignment: 1 = keep. None if not a word-subsequence."""
    raw_words = [w.strip() for w in WORD_RE.findall(raw)]
    out_words = [w.strip() for w in WORD_RE.findall(compressed)]
    labels = [0] * len(raw_words)
    i = 0
    for w in out_words:
        while i < len(raw_words) and raw_words[i] != w:
            i += 1
        if i == len(raw_words):
            return None
        labels[i] = 1
        i += 1
    return labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="20 segments, ~$0.10, pipeline check")
    ap.add_argument("--approved-cap-usd", type=float, default=None)
    ap.add_argument("--model", default=LABEL_MODEL)
    args = ap.parse_args()
    if not args.pilot and args.approved_cap_usd is None:
        raise SystemExit(
            "full label generation needs --approved-cap-usd matching the master-approved "
            "figure in DECISIONS.md (see 2026-07-27 C1 round 2 entry); or run --pilot"
        )
    cap = 0.15 if args.pilot else args.approved_cap_usd

    pool = load_pool()
    provider = pool_provider(pool.entry(args.model))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    spent = 0.0
    for path in sorted((DATA_ROOT / "cache").glob(SEGMENTS_GLOB)):
        corpus = path.stem.replace("live-segments-", "")
        segments = [
            (row["trace_id"], i, seg)
            for row in map(json.loads, path.open())
            for i, seg in enumerate(row["segments"])
        ]
        rng.shuffle(segments)
        if args.pilot:
            segments = segments[:5]
        else:
            segments = segments[:CAP_PER_CORPUS]
        out_path = OUT_DIR / f"labels-{corpus}{'-pilot' if args.pilot else ''}.jsonl"
        done: set[tuple[str, int]] = set()
        if out_path.exists():
            done = {(r["trace_id"], r["segment_index"]) for r in map(json.loads, out_path.open())}
        n_discarded = 0
        with out_path.open("a") as f:
            for trace_id, seg_idx, segment in segments:
                if (trace_id, seg_idx) in done:
                    continue
                completion = provider.complete(
                    INSTRUCTION,
                    [Message(role="user", content=segment)],
                    temperature=0.0,
                    max_tokens=4096,
                )
                usage = completion.usage
                spent += (
                    usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT
                ) / 1e6
                labels = align_labels(segment, completion.text)
                if labels is None:
                    n_discarded += 1
                else:
                    f.write(
                        json.dumps(
                            {
                                "corpus": corpus,
                                "trace_id": trace_id,
                                "segment_index": seg_idx,
                                "segment": segment,
                                "labels": labels,
                                "teacher": args.model,
                                "keep_fraction": sum(labels) / max(1, len(labels)),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if spent > cap:
                    raise SystemExit(f"spend ${spent:.2f} exceeded cap ${cap:.2f}; halting")
        log.info("%s: wrote %s, discarded(non-subsequence)=%d, spend so far $%.2f", corpus, out_path.name, n_discarded, spent)
    log.info("total spend $%.2f", spent)


if __name__ == "__main__":
    main()
