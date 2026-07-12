#!/usr/bin/env python
"""Contamination probe — does the world model reproduce benchmark outputs from PRETRAINING alone?

Zero-shot (no retrieved examples, no trajectory history) each held-out step is predicted from the
action only, then scored for exact/near match and RubricJudge factuality. If a model had memorized a
benchmark it would reproduce outputs here; near-zero recall refutes memorization. tau-bench
(synthetic, randomly generated per instance) is the uncontaminated control; swe-bench (public repos)
is the most contamination-susceptible. Results recorded in `contam.log`.

PROVENANCE: the numbers in `contam.log` were produced BEFORE #83 (rubric-v1, the unweighted mean of
five dimensions). Main is rubric-v2 (factuality-weighted headline + validity + middle truncation),
which scores strictly lower on identical predictions — re-running this WILL produce lower numbers.
Also note main relocated benchmark suites to packages/environment-capture/ (EXAMPLES_ROOT below).

    AWS_PROFILE=default AWS_REGION=us-west-2 uv run python \
        .agents/docs/research/rag_opt_results/contamination_probe.py
"""

import random
import re

import numpy as np

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.research.scaling_split import partition_corpus
from wmh.retrieval import HashingEmbedder

EXAMPLES_ROOT = "packages/environment-capture"  # was "examples/" pre-monorepo (#100/#115)
MODEL = "us.anthropic.claude-sonnet-4-6"

llm = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=MODEL, region="us-west-2"))
judge = RubricJudge(llm)
adapter = get_adapter("otel-genai")
emb = HashingEmbedder(dim=512)


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def trig(a: str, b: str) -> float:
    va, vb = np.array(emb.embed([a or " ", b or " "]))
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom > 0 else 0.0


for suite in ["tau-bench", "terminal-tasks", "swe-bench"]:
    resolved = resolve_eval_suite(suite, EXAMPLES_ROOT)
    traces = [t for f in resolved.resolve_files() for t in adapter.from_file(str(f))]
    split = partition_corpus(traces)
    steps = [st for t in split.test for st in t.steps if (st.observation.content or "").strip()]
    random.seed(4)
    sample = random.sample(steps, min(30, len(steps)))
    ex = nr = fc = 0.0
    for st in sample:
        # ZERO-SHOT: no history, no demos — pure pretraining recall of this step's output.
        pred = predict_observation(
            llm, BASE_ENV_PROMPT, st.task, st.state_before, st.action, demos=[], history=None
        )
        ex += norm(pred.content) == norm(st.observation.content)
        nr += trig(pred.content, st.observation.content) > 0.95
        fc += judge.score(pred, st.observation, st).dimensions.get("factuality", 0.0)
    n = len(sample)
    print(f"{suite:15} zero-shot: exact={ex / n:.0%} near={nr / n:.0%} factuality={fc / n:.2f} (n={n})")
