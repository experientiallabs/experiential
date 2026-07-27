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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wmo.providers.base import Message
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
WINDOW_WORDS = 400  # matches the scorer's chunk budget; keeps teacher output << max_tokens
MAX_WINDOWS_PER_SEGMENT = 12  # bounds the 115k-char terminal outlier

INSTRUCTIONS = {
    # Strict preservation: the first pilot measured keep_fraction 0.94-1.0 with this on
    # real trace segments, i.e. near-incompressible under a preserve-ALL standard.
    "strict": (
        "You compress context for an AI agent. Delete words whose removal loses no "
        "information the agent could need to act: keep every number, identifier, file path, "
        "code token, error message, entity name, and factual statement; drop filler, "
        "pleasantries, boilerplate, and redundancy. HARD RULE: the output must be an exact "
        "subsequence of the input's whitespace-delimited words (delete whole words only; "
        "never rewrite, reorder, merge, or add words). Output ONLY the compressed text."
    ),
    # Paper-style (LLMLingua-2's stance): push for brevity, keep the essentials.
    "aggressive": (
        "Compress the text by deleting as many words as possible while keeping every "
        "number, identifier, file path, code token, error message, entity name, and any "
        "word needed to preserve the facts. Grammar and fluency do not matter; telegraphic "
        "output is good. HARD RULE: the output must be an exact subsequence of the input's "
        "whitespace-delimited words (delete whole words only; never rewrite, reorder, "
        "merge, or add words). Output ONLY the compressed text."
    ),
}


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
    ap.add_argument("--instruction", choices=["strict", "aggressive"], default="strict")
    ap.add_argument("--corpora", default=None, help="comma list; default all")
    args = ap.parse_args()
    if not args.pilot and args.approved_cap_usd is None:
        raise SystemExit(
            "full label generation needs --approved-cap-usd matching the master-approved "
            "figure in DECISIONS.md (see 2026-07-27 C1 round 2 entry); or run --pilot"
        )
    cap = 0.60 if args.pilot else args.approved_cap_usd
    instruction = INSTRUCTIONS[args.instruction]

    pool = load_pool()
    provider = pool_provider(pool.entry(args.model))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    spent = 0.0
    wanted = args.corpora.split(",") if args.corpora else None
    for path in sorted((DATA_ROOT / "cache").glob(SEGMENTS_GLOB)):
        corpus = path.stem.replace("live-segments-", "")
        if wanted and corpus not in wanted:
            continue
        # Teacher sees the episode task as relevance context (labels become task-informed
        # for the domain); the STUDENT still sees only the segment at inference, so the
        # trained scorer stays query-agnostic and append-stable. Caught by the pilot:
        # without task context the teacher labels payload content (a movie script the
        # task operates on) as droppable junk.
        segments = [
            (row["trace_id"], i, seg, row["segments"][0])
            for row in map(json.loads, path.open())
            for i, seg in enumerate(row["segments"])
        ]
        # Longest-first: compression value concentrates where the token mass is, and the
        # first pilot showed short dense segments label as keep-everything (median
        # keep_fraction 1.0 on sub-200-char segments), which trains nothing. The rng
        # shuffle stays as the tiebreaker for equal lengths.
        rng.shuffle(segments)
        segments.sort(key=lambda t: len(t[2]), reverse=True)
        if args.pilot:
            segments = segments[:5]
        else:
            segments = segments[:CAP_PER_CORPUS]
        suffix = ("-pilot" if args.pilot else "") + (f"-{args.instruction}" if args.instruction != "strict" else "")
        out_path = OUT_DIR / f"labels-{corpus}{suffix}.jsonl"
        done: set[tuple[str, int]] = set()
        if out_path.exists():
            done = {(r["trace_id"], r["segment_index"]) for r in map(json.loads, out_path.open())}
        n_discarded = 0
        with out_path.open("a") as f:
            jobs: list[tuple] = []
            for trace_id, seg_idx, segment, task in segments:
                if (trace_id, seg_idx) in done:
                    continue
                # One teacher call per word-window: a segment longer than max_tokens
                # would otherwise truncate the teacher's output and mislabel the tail
                # as drop (caught by the terminal-tasks pilot, keep_fraction 0.0).
                words_ws = WORD_RE.findall(segment)
                windows = [
                    "".join(words_ws[i : i + WINDOW_WORDS])
                    for i in range(0, len(words_ws), WINDOW_WORDS)
                ][:MAX_WINDOWS_PER_SEGMENT]
                jobs.extend(
                    (trace_id, seg_idx, win_idx, window, task)
                    for win_idx, window in enumerate(windows)
                )

            lock = threading.Lock()
            state = {"spent_local": 0.0, "discarded": 0, "over_cap": False}

            def label_one(job):  # noqa: ANN001, ANN202
                trace_id, seg_idx, win_idx, window, task = job
                with lock:
                    if state["over_cap"]:
                        return
                user_content = (
                    f"TASK CONTEXT (for judging relevance only; never copy it into the "
                    f"output): {task[:1200]}\n\nTEXT TO COMPRESS:\n{window}"
                )
                completion = provider.complete(
                    instruction,
                    [Message(role="user", content=user_content)],
                    temperature=0.0,
                    max_tokens=2048,
                )
                usage = completion.usage
                cost = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1e6
                labels = align_labels(window, completion.text)
                with lock:
                    state["spent_local"] += cost
                    if spent + state["spent_local"] > cap:
                        state["over_cap"] = True
                    if labels is None:
                        state["discarded"] += 1
                        return
                    f.write(
                        json.dumps(
                            {
                                "corpus": corpus,
                                "trace_id": trace_id,
                                "segment_index": seg_idx,
                                "window_index": win_idx,
                                "segment": window,
                                "labels": labels,
                                "teacher": args.model,
                                "instruction": args.instruction,
                                "keep_fraction": sum(labels) / max(1, len(labels)),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()

            with ThreadPoolExecutor(max_workers=8) as pool_exec:
                list(pool_exec.map(label_one, jobs))
            spent += state["spent_local"]
            n_discarded += state["discarded"]
            if state["over_cap"]:
                raise SystemExit(f"spend ${spent:.2f} reached cap ${cap:.2f}; halting")
        log.info("%s: wrote %s, discarded(non-subsequence)=%d, spend so far $%.2f", corpus, out_path.name, n_discarded, spent)
    log.info("total spend $%.2f", spent)


if __name__ == "__main__":
    main()
