"""Run a benchmark fidelity eval through a Bedrock FallbackProvider chain (4-8 -> 4-7).

Workaround for evals dying on single ServiceUnavailableException blips: the provider layer
deliberately disables botocore retries (FallbackProvider owns failover), but `wmh eval run`
constructs a bare provider. This script mirrors the suite runner with a resilient chain.

Usage: uv run python .agents/scripts/eval_with_fallback.py <suite-root> <benchmark>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from wmh.engine.eval import evaluate_files
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.optimize.judge import RubricJudge
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.fallback import FallbackProvider
from wmh.providers.registry import get_provider
from wmh.retrieval.embedders import HashingEmbedder


class RetryingProvider:
    """Retry ANY provider error with backoff — outer loop around the fallback chain, for nights
    when every Bedrock model flaps at once. Complete() only; config passthrough for metering."""

    def __init__(self, inner, attempts: int = 6, backoff_s: float = 15.0) -> None:
        self._inner = inner
        self._attempts = attempts
        self._backoff_s = backoff_s
        self.config = inner.config

    def complete(self, *args, **kwargs):
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                return self._inner.complete(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - deliberate outer resilience loop
                last = error
                time.sleep(self._backoff_s * (attempt + 1))
        raise last  # type: ignore[misc]


def main() -> None:
    root, bench = Path(sys.argv[1]), sys.argv[2]
    chain = FallbackProvider(
        [
            get_provider(
                ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region="us-east-1")
            )
            for model in ("us.anthropic.claude-opus-4-8", "us.anthropic.claude-opus-4-7")
        ]
    )
    resilient = RetryingProvider(chain)
    report = evaluate_files(
        [root / bench / "traces.otel.jsonl"],
        BASE_ENV_PROMPT,
        resilient,
        RubricJudge(resilient),
        embedder=HashingEmbedder(dim=512),
        train_split=0.7,
        top_k=5,
        sample_turns="all",
        seed=0,
    )
    entry = next(iter(report.per_file.values()))
    flagged = [
        r
        for r in entry.results
        if r.is_error_actual is not None and r.is_error_predicted is not None
    ]
    err_acc = (
        sum(1 for r in flagged if r.is_error_actual == r.is_error_predicted) / len(flagged)
        if flagged
        else None
    )
    out = {
        "benchmark": bench,
        "fidelity": round(report.overall_fidelity, 4),
        "std": round(report.overall_std, 4),
        "n_steps": report.total_steps,
        "error_flag_accuracy": round(err_acc, 4) if err_acc is not None else None,
    }
    print(json.dumps(out))
    Path(f"/tmp/fallback-eval-{bench}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
