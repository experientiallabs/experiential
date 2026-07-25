"""Measure the GLM-5.2 teacher's MATH-500 pass@1, for the same grader as the student.

The distillation gate is a fraction of the teacher's solve rate, so the teacher
number has to come from the SAME extraction and normalization as the student's
or the ratio is meaningless. This imports both from `math500_baseline` rather
than restating them.

Generation goes through a chat-completions endpoint. The teacher only needs to
GENERATE here, so evals default to the Azure AI Foundry deployment, which
proxies the same weights (its responses report
`accounts/fireworks/models/glm-5p2`) and is a resource we can burn freely.
Fireworks is reserved for TRAINING tokens, where its unique `echo` prompt-logprob
surface is required (see `wmh.distill.xtoken.prompt_logprobs`); spending eval
tokens there would eat the training budget for no benefit.

Usage:
    uv run python .agents/distill/math500_teacher.py --dataset aime --n 0
    uv run python .agents/distill/math500_teacher.py --provider fireworks --n 10
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
FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p2"

PROVIDERS = ("azure", "fireworks")
"""Where eval generation goes. `azure` is the default and is free to burn."""

# Fireworks output price, used only to report what a fireworks-provider run cost
# against the training budget.
FIREWORKS_OUTPUT_USD_PER_MTOK = 4.40


def resolve_provider(provider: str, model: str | None) -> tuple[str, str, dict[str, str]]:
    """The (url, model, headers) for one provider, read from the environment.

    Args:
        provider: One of `PROVIDERS`.
        model: Explicit model or deployment override; None uses the provider default.

    Returns:
        The chat-completions URL, the model id to send, and the auth headers.

    Raises:
        SystemExit: If the provider's credentials are absent, naming the env var.
    """
    if provider == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise SystemExit(
                "FIREWORKS_API_KEY is not set; source platform/.env.local, or use the "
                "default --provider azure so training budget is not spent on evals"
            )
        return (
            FIREWORKS_CHAT_URL,
            model or FIREWORKS_MODEL,
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    key = os.environ.get("AZURE_FOUNDRY_API_KEY")
    deployment = model or os.environ.get("AZURE_FOUNDRY_GLM52_DEPLOYMENT")
    missing = [
        name
        for name, value in (
            ("AZURE_FOUNDRY_ENDPOINT", endpoint),
            ("AZURE_FOUNDRY_API_KEY", key),
            ("AZURE_FOUNDRY_GLM52_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not set; source platform/.env.local before running"
        )
    assert endpoint is not None and key is not None and deployment is not None  # noqa: S101
    return (
        endpoint.rstrip("/") + "/chat/completions",
        deployment,
        {"Content-Type": "application/json", "api-key": key, "Authorization": f"Bearer {key}"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="azure", choices=PROVIDERS)
    parser.add_argument("--model", default=None, help="model/deployment override")
    parser.add_argument("--dataset", default="math500", choices=DATASETS)
    parser.add_argument("--n", type=int, default=100, help="0 means the whole set")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    url, model_id, headers = resolve_provider(args.provider, args.model)
    logger.info("provider %s -> %s (model %s)", args.provider, url, model_id)
    problems = load_problems(args.n, args.dataset)
    logger.info("loaded %d %s problems", len(problems), args.dataset)

    def run(item: tuple[int, dict[str, str]]) -> dict[str, object]:
        index, row = item
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["problem"]},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers
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
    logger.info("provider/model:   %s / %s", args.provider, model_id)
    logger.info("problems:         %d (temperature %.1f)", total, args.temperature)
    logger.info("pass@1:           %.1f%%  (SE %.1fpp)", 100 * correct / total, 100 * standard_error)
    logger.info("truncated:        %d", truncated)
    logger.info("no boxed answer:  %d", no_answer)
    logger.info("request errors:   %d", errors)
    if args.provider == "fireworks":
        logger.info(
            "output tokens:    %d  (approx $%.2f against the TRAINING budget)",
            out_tokens,
            out_tokens * FIREWORKS_OUTPUT_USD_PER_MTOK / 1e6,
        )
    else:
        logger.info("output tokens:    %d  (azure, not billed to the training budget)", out_tokens)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
