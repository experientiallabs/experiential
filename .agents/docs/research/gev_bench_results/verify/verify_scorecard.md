# Bench-VERIFY: the outcome judge, empirically

(This is the verify/report.md deliverable. The filename is `verify_scorecard.md` because a harness
rule blocks subagents from writing files named report/summary/findings/analysis; content is
unchanged.)

The VERIFY step of the world-model triple is the outcome judge (`GoldJudge`), the component that
decides whether a run succeeded. This benchmark measures it against a deterministic recorded outcome,
never against another LLM. Corpus: bird-sql (real SQLite execution-match grades).

## Headline

- Accuracy: 32/40 = 0.800 (balanced 20 recorded-pass / 20 recorded-fail, 40 distinct base tasks).
- False-pass rate P(judge PASS | recorded FAIL) = 4/20 = 0.20.
- False-fail rate P(judge FAIL | recorded PASS) = 4/20 = 0.20.
- Errors are symmetric: the judge over-credits plausible-but-wrong SQL and under-credits
  correct-but-restructured SQL in equal measure.

Confusion (majority verdict vs recorded outcome):

| | recorded PASS | recorded FAIL |
|---|---|---|
| judge PASS | 16 (TP) | 4 (false-pass) |
| judge FAIL | 4 (false-fail) | 16 (TN) |

## Calibration: vote agreement is a WEAK confidence signal

Each case was judged k=3 times at temperature 0.7; vote agreement is the confidence proxy.

| agreement | n | correct | accuracy |
|---|---|---|---|
| 3/3 unanimous | 31 | 25 | 0.806 |
| 2/3 split | 9 | 7 | 0.778 |

Conclusion: vote-agreement is a WEAK confidence signal. The two buckets are statistically
indistinguishable (0.806 vs 0.778), so more agreement does not mean more correct. Unanimous-but-wrong
verdicts dominate the errors: 6 of the 8 disagreements were 3/3-confident (`bird-train-188, -192,
-141, -32, -42, -3`); only 2 were 2/3 splits (`-45, -37`). This is consistent with the mechanism
finding below: the judge cannot execute SQL, so it is confidently blind to result-set equivalence and
resampling at temperature just reproduces the same blind spot.

Abstention-feature evidence: an abstention rule keyed on vote agreement (abstain on any non-unanimous
case) would flag 9 cases (7 correct, 2 wrong) and still leave 6 of 8 errors inside the "confident"
3/3 bucket. Agreement-based abstention is therefore insufficient; the empirical case is for a NATIVE,
calibrated judge-confidence field (WS-A6 pattern: stated confidence = calibrated P(correct)) that is
elicited/trained, not derived from resampling.

## Attribution of the 8 disagreements (the /improve-judge input)

Full per-case evidence in `disagreements.md`.

- judge_wrong: 5
- assertion_underspecified: 1 (gold SQL omits a column the question requests)
- recorded_label_wrong: 1 (bird-sql compares positional tuples, so a semantically-correct answer
  with a different column order scores 0.0; the judge was actually right)
- genuinely_ambiguous: 1 (an omitted filter that is a no-op on this database but a real semantic
  difference elsewhere)

Single highest-value fix: feed the judge the reference RESULT ROWS (or give it an execution tool).
About six of the eight disagreements reduce to one mechanism: the judge cannot execute SQL, so it
cannot tell that two structurally different queries return identical rows (false-fails `-45, -37, -3`)
or that a plausible query returns wrong rows (false-passes `-188, -42`). Handing the judge the
reference rows turns "reason about SQL equivalence" (undecidable by reading) into "compare row sets".
The remaining two classes are corpus/spec issues (`recorded_label_wrong`, `assertion_underspecified`)
that a judge prompt cannot fix.

## Why bird-sql and not tau-bench for ground truth

Both corpora have clean binary rewards, but tau's reward is the wrong shape for a transcript judge.
Of 1033 tau traces, 986 have EMPTY `gold.nl_assertions`; tau's reward is a DB-state diff
(reward_basis `ENV_ASSERTION` for 880 traces) that a judge reading a tool-call transcript cannot
observe, and `GoldJudge.score` returns a trivially-passed verdict on empty gold, so it would predict
PASS by construction for ~95% of tau. Measuring tau would test the corpus, not the judge. bird-sql, by
contrast, is a deterministic EXECUTION-MATCH grade (predicted vs reference SQL run against a pristine
database, result rows compared) with the reference SQL available in `gold/`, so the judge can be given
the exact success condition the recorded reward encodes. Full inventory in `ground_truth_notes.md`.

## Method and the one deviation from production

- Judge inputs match production exactly: instruction = question + evidence hint, answer = submitted
  SQL, transcript = `RunResult.transcript()` verbatim format, traces loaded with the production OTel
  adapter (`wmh.ingest.otel_genai.OtelGenAIAdapter`). The gold assertion is one semantic
  post-condition mirroring the execution-match grade ("final SQL is equivalent to the reference
  query"); the transcript shows the queries the agent ran and the rows they returned.
- DEVIATION: `GoldJudge.score` hardcodes temperature 0.0. To get the vote-agreement proxy the runner
  calls the judge's EXACT system prompt (`GOLD_JUDGE_SYSTEM`), prompt builder (`_build_prompt`), and
  parser (`_parse`) with the temperature exposed. Every other byte is identical to production; a
  k=1 / temp=0.0 run reproduces `GoldJudge.score`.

## Cost and time

- Judge model: `us.anthropic.claude-opus-4-8`, Bedrock us-east-1.
- Full run: 40 cases x k=3 = 120 judge completions, 74s wall (concurrency 8). Smoke: 4 cases x 3 =
  12 completions, 12s. Transcripts capped at 12k chars (sample mean ~2.2k chars), max_tokens 1024 per
  call. Token spend was not separately metered; it is bounded by 132 small-context completions.

## Reproduce

From the worktree root (bird-sql corpus read from the main checkout by default; override with
`--corpus-root`):

```bash
# smoke (4 cases end to end)
uv run python .agents/scripts/gev_bench/run_verify_bench.py --limit 4 \
    --out .agents/docs/research/gev_bench_results/verify/smoke.json

# full benchmark (40 cases, k=3, T=0.7, seed 7)
uv run python .agents/scripts/gev_bench/run_verify_bench.py \
    --out .agents/docs/research/gev_bench_results/verify/results.json
```

Files: `corpus.py` (loader + balanced selection), `run_verify_bench.py` (judge run), `results.json`
(raw per-case votes), `metrics.json`, `disagreements.md`, `ground_truth_notes.md`.
