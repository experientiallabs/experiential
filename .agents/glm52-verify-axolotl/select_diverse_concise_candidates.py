#!/usr/bin/env python3
"""Select concise, diverse SFT candidates while excluding evaluation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
HEX_DIRECTORY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EvaluationPrompt:
    """One held-out benchmark instruction used only for exclusion."""

    benchmark: str
    task_name: str
    path: str
    sha256: str
    normalized: str
    shingles: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    """Compact selection metadata for one source trajectory."""

    row_index: int
    rollout_id: str
    task_id: str
    manifest_order: int
    replica: int
    prompt_sha256: str
    prompt_characters: int
    transcript_characters: int
    message_count: int
    n_student_tokens: int
    n_supervised_tokens: int
    nearest_evaluation_benchmark: str | None
    nearest_evaluation_task: str | None
    nearest_evaluation_similarity: float


def normalize_text(value: str) -> str:
    """Normalize text for conservative prompt-overlap checks."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def word_shingles(value: str, size: int = 5) -> frozenset[str]:
    """Return fixed-size word shingles for near-duplicate detection."""
    words = value.split()
    if len(words) < size:
        return frozenset({value}) if value else frozenset()
    return frozenset(
        " ".join(words[index : index + size]) for index in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Return Jaccard similarity for two shingle sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def task_name_for_instruction(path: Path) -> str:
    """Resolve task name for both Harbor and TBLite directory layouts."""
    parent = path.parent
    return parent.parent.name if HEX_DIRECTORY.fullmatch(parent.name) else parent.name


def load_evaluation_prompts(roots: list[Path]) -> list[EvaluationPrompt]:
    """Load every instruction.md below labeled benchmark roots."""
    prompts: list[EvaluationPrompt] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"evaluation instruction root is missing: {root}")
        benchmark = root.parent.name if root.name == "dataset" else root.name
        paths = sorted(root.rglob("instruction.md"))
        if not paths:
            raise ValueError(f"no instruction.md files below {root}")
        for path in paths:
            payload = path.read_bytes()
            normalized = normalize_text(payload.decode("utf-8"))
            prompts.append(
                EvaluationPrompt(
                    benchmark=benchmark,
                    task_name=task_name_for_instruction(path),
                    path=str(path),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    normalized=normalized,
                    shingles=word_shingles(normalized),
                )
            )
    return prompts


def first_user_prompt(messages: list[dict[str, object]]) -> str:
    """Return the first textual user message in a source transcript."""
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise ValueError("source transcript has no textual user message")


def nearest_evaluation(
    task_id: str,
    prompt: str,
    evaluation_prompts: list[EvaluationPrompt],
    *,
    similarity_threshold: float,
) -> tuple[bool, EvaluationPrompt | None, float, str | None]:
    """Return whether a task overlaps held-out evaluation material."""
    normalized = normalize_text(prompt)
    prompt_shingles = word_shingles(normalized)
    normalized_task_id = task_id.rsplit("/", 1)[-1].casefold()
    best: EvaluationPrompt | None = None
    best_similarity = 0.0
    for evaluation in evaluation_prompts:
        if normalized_task_id == evaluation.task_name.casefold():
            return True, evaluation, 1.0, "task_name_match"
        if normalized == evaluation.normalized:
            return True, evaluation, 1.0, "exact_prompt_match"
        if min(len(normalized), len(evaluation.normalized)) >= 40 and (
            normalized in evaluation.normalized or evaluation.normalized in normalized
        ):
            return True, evaluation, 1.0, "prompt_containment"
        similarity = jaccard(prompt_shingles, evaluation.shingles)
        if similarity > best_similarity:
            best = evaluation
            best_similarity = similarity
    if best_similarity >= similarity_threshold:
        return True, best, best_similarity, "near_duplicate_prompt"
    return False, best, best_similarity, None


def percentile(values: list[int], quantile: float) -> int:
    """Return a nearest-rank integer percentile."""
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def summarize(values: list[int]) -> dict[str, int | float]:
    """Summarize a nonempty integer series."""
    return {
        "min": min(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
    }


def select_stratified(candidates: list[Candidate], count: int, seed: int) -> list[Candidate]:
    """Select evenly across prompt-length ranks with deterministic randomness."""
    if count <= 0 or count > len(candidates):
        raise ValueError("count must be positive and no larger than candidate count")
    ordered = sorted(candidates, key=lambda item: (item.prompt_characters, item.row_index))
    randomizer = random.Random(seed)
    selected: list[Candidate] = []
    selected_rows: set[int] = set()
    bins = min(8, count)
    base_per_bin, extra = divmod(count, bins)
    for bin_index in range(bins):
        start = len(ordered) * bin_index // bins
        end = len(ordered) * (bin_index + 1) // bins
        choices = ordered[start:end]
        randomizer.shuffle(choices)
        target = base_per_bin + (1 if bin_index < extra else 0)
        for candidate in choices[:target]:
            selected.append(candidate)
            selected_rows.add(candidate.row_index)
    if len(selected) < count:
        remainder = [item for item in ordered if item.row_index not in selected_rows]
        randomizer.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    return sorted(selected, key=lambda item: item.row_index)


def build_candidates(
    corpus: Path,
    evaluation_prompts: list[EvaluationPrompt],
    *,
    max_supervised_tokens: int,
    target_supervised_tokens: int,
    similarity_threshold: float,
) -> tuple[list[Candidate], Counter[str], int]:
    """Build one concise representative per eligible non-evaluation task."""
    per_task: dict[str, list[Candidate]] = defaultdict(list)
    exclusion_counts: Counter[str] = Counter()
    total_rows = 0
    with corpus.open(encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            total_rows += 1
            row = json.loads(raw_line)
            if row["student_truncated_at_train_axis"] or row["teacher_truncated_at_train_axis"]:
                exclusion_counts["truncated_at_train_axis"] += 1
                continue
            supervised_tokens = int(row["n_supervised_tokens"])
            if supervised_tokens > max_supervised_tokens:
                exclusion_counts["above_supervised_token_cap"] += 1
                continue
            messages = json.loads(row["message_log_json"])
            prompt = first_user_prompt(messages)
            task_id = str(row["task_id"])
            overlaps, nearest, similarity, reason = nearest_evaluation(
                task_id,
                prompt,
                evaluation_prompts,
                similarity_threshold=similarity_threshold,
            )
            if overlaps:
                exclusion_counts[str(reason)] += 1
                continue
            normalized_prompt = normalize_text(prompt)
            per_task[task_id].append(
                Candidate(
                    row_index=row_index,
                    rollout_id=str(row["rollout_id"]),
                    task_id=task_id,
                    manifest_order=int(row["manifest_order"]),
                    replica=int(row["replica"]),
                    prompt_sha256=hashlib.sha256(normalized_prompt.encode()).hexdigest(),
                    prompt_characters=len(normalized_prompt),
                    transcript_characters=len(str(row["message_log_json"])),
                    message_count=len(messages),
                    n_student_tokens=int(row["n_student_tokens"]),
                    n_supervised_tokens=supervised_tokens,
                    nearest_evaluation_benchmark=nearest.benchmark if nearest else None,
                    nearest_evaluation_task=nearest.task_name if nearest else None,
                    nearest_evaluation_similarity=round(similarity, 6),
                )
            )

    representatives: list[Candidate] = []
    seen_prompt_hashes: set[str] = set()
    for task_id in sorted(per_task):
        choices = sorted(
            per_task[task_id],
            key=lambda item: (
                abs(item.n_supervised_tokens - target_supervised_tokens),
                item.n_supervised_tokens,
                item.row_index,
            ),
        )
        choice = choices[0]
        if choice.prompt_sha256 in seen_prompt_hashes:
            exclusion_counts["duplicate_prompt_across_task_ids"] += len(choices)
            continue
        seen_prompt_hashes.add(choice.prompt_sha256)
        representatives.append(choice)
        exclusion_counts["nonrepresentative_replica"] += len(choices) - 1
    return representatives, exclusion_counts, total_rows


def main() -> int:
    """Select and record a deterministic, held-out-safe candidate set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--evaluation-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--max-supervised-tokens", type=int, default=8000)
    parser.add_argument("--target-supervised-tokens", type=int, default=5000)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite selection outputs")
    if not 0 < args.similarity_threshold <= 1:
        raise ValueError("similarity threshold must be in (0, 1]")

    evaluation_prompts = load_evaluation_prompts(args.evaluation_root)
    candidates, exclusions, total_rows = build_candidates(
        args.corpus,
        evaluation_prompts,
        max_supervised_tokens=args.max_supervised_tokens,
        target_supervised_tokens=args.target_supervised_tokens,
        similarity_threshold=args.similarity_threshold,
    )
    selected = select_stratified(candidates, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(json.dumps(asdict(candidate), sort_keys=True) + "\n")

    manifest = {
        "schema": "glm52-diverse-concise-candidate-selection-v1",
        "source_corpus": str(args.corpus),
        "source_corpus_sha256": args.source_sha256,
        "source_rows": total_rows,
        "evaluation_prompt_count": len(evaluation_prompts),
        "evaluation_benchmark_counts": dict(
            sorted(Counter(prompt.benchmark for prompt in evaluation_prompts).items())
        ),
        "evaluation_instruction_set_sha256": hashlib.sha256(
            "\n".join(sorted(prompt.sha256 for prompt in evaluation_prompts)).encode()
        ).hexdigest(),
        "selection": {
            "seed": args.seed,
            "count": len(selected),
            "unique_tasks": len({item.task_id for item in selected}),
            "unique_prompt_hashes": len({item.prompt_sha256 for item in selected}),
            "max_supervised_tokens": args.max_supervised_tokens,
            "target_supervised_tokens": args.target_supervised_tokens,
            "near_duplicate_jaccard_threshold": args.similarity_threshold,
            "one_representative_per_task": True,
            "prompt_length_strata": 8,
        },
        "eligible_unique_task_representatives": len(candidates),
        "row_exclusion_counts": dict(sorted(exclusions.items())),
        "selected_row_indices": [item.row_index for item in selected],
        "selected_task_ids": [item.task_id for item in selected],
        "selected_supervised_tokens": summarize([item.n_supervised_tokens for item in selected]),
        "selected_student_tokens": summarize([item.n_student_tokens for item in selected]),
        "selected_prompt_characters": summarize([item.prompt_characters for item in selected]),
        "maximum_selected_evaluation_similarity": max(
            item.nearest_evaluation_similarity for item in selected
        ),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "evaluation_prompts_are_excluded_not_used_for_training": True,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("%s", json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
