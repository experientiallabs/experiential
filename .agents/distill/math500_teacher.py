"""Measure the GLM-5.2 teacher's MATH-500 pass@1, for the same grader as the student.

The distillation gate is a fraction of the teacher's solve rate, so the teacher
number has to come from the SAME extraction and normalization as the student's
or the ratio is meaningless. This imports both from `math500_baseline` rather
than restating them.

Generation goes through Fireworks chat completions (the teacher only needs to
generate here; teacher-forced scoring is a different route, see
`wmh.distill.xtoken.prompt_logprobs`).

Usage:
    uv run python .agents/distill/math500_teacher.py --n 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from math500_baseline import (
    DATASETS,
    SYSTEM_PROMPT,
    answers_match,
    extract_boxed,
    load_problems,
)

logger = logging.getLogger("math500-teacher")

FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/glm-5p2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="math500", choices=DATASETS)
    parser.add_argument("--n", type=int, default=100, help="0 means the whole set")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit(
            "FIREWORKS_API_KEY is not set; source platform/.env.local before running"
        )
    problems = load_problems(args.n, args.dataset)
    logger.info("loaded %d %s problems", len(problems), args.dataset)

    def run(item: tuple[int, dict[str, str]]) -> dict[str, object]:
        index, row = item
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["problem"]},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        request = urllib.request.Request(
            FIREWORKS_CHAT_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            logger.warning("problem %d failed: %r", index, exc)
            return {"index": index, "gold": row["answer"], "predicted": None,
                    "correct": False, "error": repr(exc), "completion_tokens": 0}
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
        predicted = extract_boxed(text)
        correct = answers_match(row["answer"], predicted)
        return {
            "index": index,
            "gold": row["answer"],
            "predicted": predicted,
            "correct": correct,
            "completion_tokens": (payload.get("usage") or {}).get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, enumerate(problems)))

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r.get("error"))
    no_answer = sum(1 for r in results if r["predicted"] is None and not r.get("error"))
    truncated = sum(1 for r in results if r.get("finish_reason") == "length")
    out_tokens = sum(int(r["completion_tokens"]) for r in results)
    standard_error = (correct / total * (1 - correct / total) / total) ** 0.5

    logger.info("")
    logger.info("model:            %s", args.model)
    logger.info("problems:         %d (temperature %.1f)", total, args.temperature)
    logger.info("pass@1:           %.1f%%  (SE %.1fpp)", 100 * correct / total, 100 * standard_error)
    logger.info("truncated:        %d", truncated)
    logger.info("no boxed answer:  %d", no_answer)
    logger.info("request errors:   %d", errors)
    logger.info("output tokens:    %d  (approx $%.2f at $4.40/Mtok)", out_tokens, out_tokens * 4.4e-6)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
