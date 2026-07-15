"""Old-vs-new judge on identical real predictions: the fidelity-regression check.

Samples held-out steps from the bundled corpora, generates each world-model prediction ONCE
(zero-shot, teacher-forced history, cached to --cache), then scores the same (predicted, actual,
step) triples with the pre-overhaul judge (prompt + unweighted mean + no truncation, snapshotted
verbatim from git 404749b) and the overhauled `RubricJudge`. Reports mean shift, rank agreement,
and the biggest per-step deltas so shifts can be verified to concentrate on low-factuality steps.

Usage:
    uv run python .agents/scripts/run_judge_regression.py \
        --cache .agents/docs/research/raw/judge-regression-preds.json \
        --out .agents/docs/research/raw/judge-regression.json
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import Action, Observation, Step, Trace
from wmh.engine.build import split_traces
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import RUBRIC_DIMENSIONS, JudgeResult, RubricJudge, _clamp
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Message, Provider

CORPORA = {
    "tau-bench": "examples/tau-bench/traces.otel.jsonl",
    "terminal-tasks": "examples/terminal-tasks/traces.otel.jsonl",
    "swe-bench": "examples/swe-bench/traces.otel.jsonl",
}

# --- The OLD judge, snapshotted verbatim from git 404749b (pre-overhaul) -------------------------

OLD_JUDGE_SYSTEM = """You grade a world model that simulates an environment for an AI agent. You
see the agent's action, the ACTUAL observation the real environment returned, and a PREDICTED
observation the world model generated. Score the prediction's fidelity to the actual observation on
five independent dimensions, each from 0.0 to 1.0:

- format: same shape/structure/encoding the environment uses (JSON shape, field names, exit status).
- factuality: conveys the same outcome, errors, and SALIENT data the agent would act on.
- consistency: coherent with the action and the environment's established behavior.
- realism: looks like a real response this environment would emit (not an explanation or apology).
- quality: overall, how usable this prediction is as a stand-in for the real observation.

CONTENT TYPE matters — infer it from the action and observation:
- DETERMINISTIC / computed content (a file's contents via `cat`, a command's computed stdout, a
  lookup of state that exists) MUST match the actual values to score high on factuality — wrong
  computed values are unambiguously wrong even if well-formatted.
- VOLATILE / incidental content (PIDs, timestamps, random ids, ordering of unordered output) should
  be judged on plausibility and format only — a different-but-plausible value is fine.

Respond with ONLY a JSON object, no prose:
{"format": <0..1>, "factuality": <0..1>, "consistency": <0..1>, "realism": <0..1>,
 "quality": <0..1>, "critique": "<one or two sentences: what matched, what diverged>"}"""


class _OldRawRubric(BaseModel):
    format: float = 0.0
    factuality: float = 0.0
    consistency: float = 0.0
    realism: float = 0.0
    quality: float = 0.0
    critique: str = ""


def _old_observation_payload(observation: Observation, *, empty_sentinel: str) -> dict[str, object]:
    return {
        "is_error": observation.is_error,
        "content_length": len(observation.content),
        "content": observation.content,  # old judge: no truncation
        "empty_content": observation.content == "",
        "empty_sentinel": empty_sentinel if observation.content == "" else None,
    }


def _old_build_prompt(predicted: Observation, actual: Observation, context: Step) -> str:
    action = context.action
    action_desc = action.name or action.content or "(none)"
    actual_payload = _old_observation_payload(actual, empty_sentinel="<EMPTY_ACTUAL_OBSERVATION>")
    predicted_payload = _old_observation_payload(predicted, empty_sentinel="<EMPTY_PREDICTION>")
    return (
        f"AGENT ACTION ({action.kind.value}): {action_desc}\n"
        f"ACTION ARGUMENTS: {json.dumps(action.arguments, sort_keys=True, default=str)}\n\n"
        "ACTUAL OBSERVATION JSON:\n"
        f"{json.dumps(actual_payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "PREDICTED OBSERVATION JSON:\n"
        f"{json.dumps(predicted_payload, ensure_ascii=False, sort_keys=True)}\n"
    )


class OldRubricJudge:
    """Pre-overhaul judge: unweighted mean, silent 0.0 defaults, no truncation, no retry."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        user = _old_build_prompt(predicted, actual, context)
        completion = self._provider.complete(
            OLD_JUDGE_SYSTEM, [Message(role="user", content=user)], temperature=0.0, max_tokens=512
        )
        raw = extract_json_object(completion.text)
        if raw is not None:
            try:
                parsed = _OldRawRubric.model_validate_json(raw)
            except ValidationError:
                parsed = None
            if parsed is not None:
                dims = {d: _clamp(getattr(parsed, d)) for d in RUBRIC_DIMENSIONS}
                return JudgeResult(
                    score=sum(dims.values()) / len(dims),
                    critique=parsed.critique.strip(),
                    dimensions=dims,
                )
        return JudgeResult(score=0.0, critique="Unparseable (old judge).")


# --- sampling + prediction cache ------------------------------------------------------------------


def _sample_steps(seed: int, per_corpus: int) -> list[dict[str, object]]:
    """Seeded sample of held-out steps per corpus (+ the 2 longest terminal-task observations,
    which exercise the new judge's truncation path on real data)."""
    adapter = get_adapter("otel-genai")
    rng = random.Random(seed)
    sampled: list[dict[str, object]] = []
    for name, path in CORPORA.items():
        traces: list[Trace] = adapter.from_file(path)
        _, holdout = split_traces(traces, 0.7)
        pool = [
            (trace, i)
            for trace in holdout
            for i in range(len(trace.steps))
        ]
        picks = rng.sample(pool, min(per_corpus, len(pool)))
        if name == "terminal-tasks":
            longest = sorted(pool, key=lambda p: len(p[0].steps[p[1]].observation.content))[-2:]
            picks.extend(p for p in longest if p not in picks)
        for trace, i in picks:
            sampled.append({"corpus": name, "trace": trace, "index": i})
    return sampled


def _predict(
    items: list[dict[str, object]], model: str, region: str | None, concurrency: int
) -> None:
    provider = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region))

    def _one(item: dict[str, object]) -> None:
        trace: Trace = item["trace"]  # type: ignore[assignment]
        i: int = item["index"]  # type: ignore[assignment]
        step = trace.steps[i]
        predicted = predict_observation(
            provider,
            BASE_ENV_PROMPT,
            step.task,
            step.state_before,
            step.action,
            demos=[],
            history=trace.steps[:i],
        )
        item["predicted"] = predicted

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(_one, items))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--wm-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--region", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-corpus", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cache", required=True, help="Prediction cache JSON (reused if present).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        print(f"loaded {len(rows)} cached predictions from {cache}")
    else:
        items = _sample_steps(args.seed, args.per_corpus)
        print(f"sampled {len(items)} steps; predicting once with {args.wm_model} ...")
        _predict(items, args.wm_model, args.region, args.concurrency)
        rows = []
        for item in items:
            trace: Trace = item["trace"]  # type: ignore[assignment]
            i: int = item["index"]  # type: ignore[assignment]
            step = trace.steps[i]
            predicted: Observation = item["predicted"]  # type: ignore[assignment]
            rows.append(
                {
                    "corpus": item["corpus"],
                    "trace_id": trace.trace_id,
                    "index": i,
                    "task": step.task,
                    "action": step.action.model_dump(mode="json"),
                    "actual": step.observation.model_dump(mode="json"),
                    "predicted": predicted.model_dump(mode="json"),
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote prediction cache -> {cache}")

    judge_provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=args.judge_model, region=args.region)
    )
    old_judge = OldRubricJudge(judge_provider)
    new_judge = RubricJudge(judge_provider)

    def _score(row: dict[str, object]) -> dict[str, object]:
        action = Action.model_validate(row["action"])
        actual = Observation.model_validate(row["actual"])
        predicted = Observation.model_validate(row["predicted"])
        step = Step(action=action, observation=actual, task=row.get("task"))
        old = old_judge.score(predicted, actual, step)
        new = new_judge.score(predicted, actual, step)
        return {
            "corpus": row["corpus"],
            "trace_id": row["trace_id"],
            "index": row["index"],
            "action": action.name or action.content or "(none)",
            "len_actual": len(actual.content),
            "old_score": old.score,
            "old_dims": old.dimensions,
            "new_score": new.score,
            "new_dims": new.dimensions,
            "new_valid": new.valid,
            "old_critique": old.critique,
            "new_critique": new.critique,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        scored = list(pool.map(_score, rows))

    valid = [s for s in scored if s["new_valid"]]
    old_scores = [s["old_score"] for s in valid]
    new_scores = [s["new_score"] for s in valid]
    n = len(valid)
    mean_old, mean_new = fmean(old_scores), fmean(new_scores)

    def _spearman(a: list[float], b: list[float]) -> float:
        def ranks(v: list[float]) -> list[float]:
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            j = 0
            while j < len(order):
                k = j
                while k + 1 < len(order) and v[order[k + 1]] == v[order[j]]:
                    k += 1
                avg = (j + k) / 2 + 1
                for m in range(j, k + 1):
                    r[order[m]] = avg
                j = k + 1
            return r

        ra, rb = ranks(a), ranks(b)
        ma, mb = fmean(ra), fmean(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
        da = sum((x - ma) ** 2 for x in ra) ** 0.5
        db = sum((y - mb) ** 2 for y in rb) ** 0.5
        return num / (da * db) if da and db else 0.0

    rho = _spearman(old_scores, new_scores)
    print(f"\nn={n} valid steps ({len(scored) - n} judge-invalid under the new judge)")
    print(f"mean fidelity: old={mean_old:.3f}  new={mean_new:.3f}  shift={mean_new - mean_old:+.3f}")
    print(f"spearman rank agreement: {rho:.3f}")
    per_corpus: dict[str, list[dict[str, object]]] = {}
    for s in valid:
        per_corpus.setdefault(str(s["corpus"]), []).append(s)
    for name, group in sorted(per_corpus.items()):
        o = fmean(float(g["old_score"]) for g in group)
        nw = fmean(float(g["new_score"]) for g in group)
        print(f"  {name:16} n={len(group):3}  old={o:.3f}  new={nw:.3f}  shift={nw - o:+.3f}")

    print("\nlargest per-step shifts (new - old), with new-judge factuality:")
    for s in sorted(valid, key=lambda s: abs(float(s["new_score"]) - float(s["old_score"])))[-8:]:
        delta = float(s["new_score"]) - float(s["old_score"])
        fact = dict(s["new_dims"]).get("factuality")  # type: ignore[arg-type]
        print(
            f"  {delta:+.3f}  old={s['old_score']:.2f} new={s['new_score']:.2f} "
            f"fact={fact} len={s['len_actual']:6} {s['corpus']}/{s['action']}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "judge_model": args.judge_model,
                    "wm_model": args.wm_model,
                    "seed": args.seed,
                    "mean_old": mean_old,
                    "mean_new": mean_new,
                    "spearman": rho,
                    "steps": scored,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
