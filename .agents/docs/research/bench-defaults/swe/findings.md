# bench-defaults/swe: the real SWE-bench evidence behind the product's swe default

**Lane:** bench-defaults/swe · **Cohort:** `swe-cachefix-8bd6c3a11dea` · **Endpoint:** `0bbfb3cc-5bf7-4261-852c-e9a079fc0f32` (admin org)

Every number here is reproducible from `.agents/docs/research/bench-defaults/swe/numbers.json` plus the `bench-defaults/swe:` entries in `DECISIONS.md`. Nothing is hand-computed.

## Headline

**The swe default is a pinned `opus-5`, measured 36.3% cheaper than the `fable-5` anchor with a quality difference of -2.5 points that this sample size does not resolve.**

| | value |
| --- | --- |
| accuracy | 0.850 (anchor 0.875) |
| cost per run | $0.7771 (anchor $1.2196) |
| savings | **36.3%** |
| paired N | 20 instances, both episodes |
| quality delta | -2.5 pts, 95% CI **-10.0 to +5.0**, sign test **p = 1.00** |
| cost delta per completed task | -34.4%, 95% CI -46.5 to -22.9 |
| latency, per-episode model seconds | p50 131.0s vs anchor 173.6s |
| provenance | `real_episode` |
| judge | SWE-bench test suite (FAIL_TO_PASS + PASS_TO_PASS), deterministic |

The cost saving is statistically resolved; its CI excludes zero. The quality difference is NOT: the paired CI spans zero and the sign test is p = 1.00. Copy must therefore say "36.3% measured savings, quality difference -2.5pt unresolved at this sample size", never "matches" or "equal quality".

## What was measured

640 real SWE-bench Verified episodes: 20 pinned instances x 2 episodes x 16 candidates. Each cell is a real mini-swe-agent 2.4.6 episode inside that instance's own SWE-bench Docker image, using the benchmark's canonical `swebench.yaml` tool-calling config. Every model call went through wmo's provider layer via a loopback proxy with one alias per cell, so usage is exact per cell and every price comes from the cohort's pinned copy of `pool-17.toml`, never a harness cost field.

636 scored, 4 unscored, $241.39. The anchor `fable-5` has ZERO unscored cells, so all 15 comparison arms carry the full paired N=20 against it.

The pin is deterministic and unstratified: sort the corpus's 255 instance ids by `sha256("swe-defaults-v1:" + instance_id)` ascending, take the first 20. All 20 are present in `princeton-nlp/SWE-Bench_Verified`. Repo mix 15 django / 3 astropy / 2 matplotlib, against a corpus that is itself 231/22/2.

The verifier was validated before any model ran: the official swebench 4.1.0 harness on the GOLD patch resolved `django__django-15280` 1/1 in 25.5 seconds, reusing pulled instance images. That is the same code path every cell's score comes from.

## The router null result

The fitted kNN router routes NOTHING at its balanced default and therefore saves nothing. On the fit's own 6-scenario held-out band the best dial position buys -3.4%. Under leave-one-out refit across all 20 scenarios (every scenario held out once, guard and fallback rediscovered per fold):

    rung                 dial   quality vs anchor   cost      routed mix
    quality-max          0.00   -5.0 pts            -10.2%    fable-5 18, opus-5 2
    balanced (default)   0.25   -5.0 pts            -10.2%    fable-5 18, opus-5 2
    cost saver           0.50   -12.5 pts           -11.4%    fable-5 16, opus-4-8 2, opus-5 2
    deep saver           0.75   -12.5 pts           -13.2%    fable-5 14, opus-4-8 4, opus-5 2
    max savings          1.00   -12.5 pts           -13.2%    fable-5 14, opus-4-8 4, opus-5 2

Pinning a single model dominates every one of those rows (full paired N=20):

    candidate        quality vs anchor   cost/completed task   cost delta   cost CI95
    opus-5           -2.5 pts            $0.9142               -34.4%       -46.5 to -22.9
    opus-4-8         -10.0 pts           $0.4046               -71.0%       -80.9 to -48.7
    gpt-5.6-sol      -17.5 pts           $0.2798               -79.9%       -88.6 to -60.3
    gpt-5.6-terra    -27.5 pts           $0.1277               -90.8%       -93.8 to -83.5
    gpt-5.6-luna     -37.5 pts           $0.0574               -95.9%       -97.5 to -91.6
    qwen3.5-9b       -85.0 pts           $3.7641               +170.1%      -30.1 to +434.8

MECHANISM: SWE-bench cost is dominated by turn count, because every turn replays the whole transcript. Re-routing two instances of twenty barely moves the total, while choosing a model that needs fewer turns moves all of it. This is a finding about where the product's value sits on this benchmark, not a defect in the router, and it is why cost per COMPLETED task is the right primary metric.

Honesty about the pinned choice too: even opus-4-8's -10.0-point gap is unresolved at N=20 (CI -27.5 to +2.5, p = 0.625). The only quality difference this cohort RESOLVES is qwen3.5-9b's collapse (CI -95.0 to -72.5, p < 0.0001). opus-5 was chosen on the cost CI, not on a quality point estimate.

## qwen3.5-9b: the cheapest tokens are the most expensive per completed task

`qwen3.5-9b` has the cheapest tokens in the pool ($0.10/$0.15 per Mtok) and the HIGHEST cost per completed task in the cohort: $3.7641, which is 2.7x the anchor's $1.3938. It solved 1 of 38 scored instances, hit the 75-call cap on 31, and roughly half its billed calls produced turns the harness rejected outright (measured: 75 calls made, 38 assistant turns kept on `django__django-13343`). Its input tokens grew normally to 22-28k, so this is not a context ceiling: it simply fails to emit parseable actions about half the time.

Cost per episode would have made this arm look ten times cheap. Cost per completed task convicts it. Any max-savings rung built on it would route to a candidate that finishes nothing. This is the sharpest of the three benchmarks showing the same shape.

## Per-arm cap-hit table

97 of 636 scored cells (15%) hit the 75-call pin, and the concentration is monotone in price:

    candidate         cells   solve   capped   cap%
    qwen3.5-9b           38    0.03       31    82%
    haiku-4-5            40    0.42       18    45%
    kimi-k2.6            40    0.42       15    38%
    glm-5.2              39    0.54        8    21%
    kimi-k3              39    0.64        8    21%
    sonnet-5             40    0.68        5    12%
    deepseek-v4-pro      40    0.60        4    10%
    qwen3.6-27b          40    0.55        4    10%
    opus-5               40    0.85        3     8%
    fable-5              40    0.88        1     2%
    gpt-5.5              40    0.68        0     0%
    gpt-5.6-sol          40    0.70        0     0%
    gpt-5.6-terra        40    0.60        0     0%
    gpt-5.6-luna         40    0.50        0     0%
    gpt-5.4-mini         40    0.40        0     0%
    opus-4-8             40    0.78        0     0%

The pin is uniform in calls but not in effect. The frontier arms finish inside it (fable-5's fastest solve is 7 calls) while the cheap arms are still working when it stops them. Raising the cap would help the cheap arms and nobody else, so these numbers are CONSERVATIVE for the frontier candidates and HARSH on the cheap ones.

## Failure taxonomy, with counts

Binding rule applied throughout: candidate-caused termination is a SCORED failure (reward 0.0, harness exit status on the row); infrastructure is UNSCORED, excluded from every statistic and reported.

    class                                                        disposition                count
    resolved by the test suite                                   reward 1.0                   371
    unresolved / empty patch / budget exhausted                  reward 0.0 (candidate)       265
    wall-deadline timeout (glm-5.2, qwen3.5-9b, kimi-k3)         unscored (infra)               3
    transient provider fault, no terminal status                 unscored (infra)               1
    transient provider fault, episode DID reach terminal status  rescued and scored             3
    provider content-filter refusal (gpt-5.5, django-11790)      retried, did not reproduce     1

REFUSAL DISPOSITION: gpt-5.5 returned 400 `invalid_prompt` ("flagged as potentially violating our usage policy") on a Django issue. Per the retry-decides ruling it was re-bought; the refusal did NOT recur, so it became a genuine measured cell (reward 0.0 in 21 calls) and gpt-5.5 keeps its full N=20. Had it reproduced it would have been reclassified as a scored 0 with a refusal flag, on the reasoning that a deterministic refusal is a production defect of that arm rather than an infrastructure hole.

THE RESCUED THREE: "first provider fault leaves the cell unscored" is right when the fault ends the episode and too blunt when it does not. OpenRouter occasionally returns a truncated JSON body, litellm retries it, and the episode reaches a terminal status anyway. `repair_swe_transients.py` decided per cell from the artifacts: terminal status means score it (two empty-patch budget exhaustions scored 0.0, one 507-char patch handed to the official verifier and returned unresolved); no status means leave it unscored. No episode was re-run.

## Cost provenance

- METRIC: cache-adjusted effective cost per COMPLETED task, from `wmo.optimize.scorecard` and nowhere else. Unscored spend is excluded from the ratio and reported separately.
- PRICES: the cohort's pinned copy of `pool-17.toml`, snapshotted into the grid directory. All 16 rows were audited before the buy (DECISIONS price-audit entry). Anthropic cache WRITES are modeled at 1.25x input from the catalog, correcting a stale pool comment claiming they were not.
- CACHE CREDIT IS REAL AND MEASURED: a live episode served 98% of its input tokens from cache by step 30. Both reporting locations are read (Anthropic's top-level `cache_read_input_tokens`, and OpenAI-compatible backends' nested `prompt_tokens_details.cached_tokens`).
- GRID SPEND: $241.39 across both episodes, matching the platform's stored run to the cent ($241.3931). Lane total $263.52 including a discarded first cohort ($21.21) and smoke ($0.91).
- ANALYSIS SPEND: 530 embedding calls, about $0.0015 at 3-large list price.

## Binding caveats

1. QUALITY RESOLUTION. The headline quality difference (-2.5 pts) is UNRESOLVED at N=20: CI -10.0 to +5.0, p = 1.00. Copy says "quality difference -2.5pt unresolved at this sample size".
2. 75-STEP DISCLOSURE. The step limit is pinned at 75 against mini-swe-agent's shipped 250, applied uniformly. Measured justification: a probe (haiku-4-5, django__django-15280) spent 217 calls and 15.4 minutes to submit one patch. These solve rates are NOT comparable to published SWE-bench numbers, and 15% of cells hit the cap, concentrated in the cheap arms.
3. ARM-COUNT FOOTNOTE. swe measures 16 arms, the full pool, nothing pinned out: SWE-bench has no user simulator and no LLM judge. tau measures 15; the difference is gpt-5.4-mini, tau's user simulator and swe's candidate.
4. The gpt-5.6 family ran with reasoning OFF (`reasoning_effort="none"`), the only tool-calling configuration reachable via chat completions, which is what the product serves.
5. ONE MACHINE, UNDER EMULATION. SWE-bench publishes x86_64 images and this box is arm64, so latency is honest as a comparison between arms and is not a production-latency claim.
6. TWO SAVINGS NUMBERS EXIST AND MUST NOT BE CONFLATED. 36.3% is the benchmark number over 640 episodes and is the SWE-bench claim. The live savings card read 45.35% over 7 short chat requests whose token mix has nothing to do with SWE-bench episodes.

## Traps and product bugs found, with shas

    finding                                                          disposition
    AnthropicProvider had NO complete_chat, so no Anthropic          implemented with cache breakpoints + 14 tests: d2e7f854,
    candidate could run a tool-calling harness (the same gap         restored as 0573809e after a sibling's --amend dropped it;
    behind the serving endpoint's Anthropic refusal)                 ty-clean in 422a7f13
    cache-token reader missed the OpenAI-compatible NESTED field,     3766f31f; first cohort abandoned with its $21.21 bill,
    pricing ~95%-cached prompts at full rate for 8 of 16 arms         per-call usage now persisted so repricing never needs a
    (gpt-5.6-sol $1.46 -> $0.31 on the same episode, 4.7x)            re-buy
    `steps` counted transcript turns, not billed calls                c366ba09; 12 of 41 cells repaired offline by
    (qwen3.5-9b showed 38 against a 75 cap)                           repair_swe_steps.py, no re-buy
    exhausted step budget was scored as UNSCORED, deleting the        dc918379
    weakest arms' worst rows
    litellm message/request debris reaching strict upstreams          4f7f89df, 2eb63a7b, 9ea76928
    (glm-5.2 400s on provider_specific_fields); gpt-5.6 refuses
    tools unless reasoning_effort="none"
    a grid could not name the endpoint it is evidence for, so run     2a840689
    history never reached the product object
    verifier lost a race with the grid's own containers; 1500s        000b48bd
    deadline capped the FRONTIER arms and left the anchor
    unscored on an instance
    transient-fault cells that completed anyway were discarded        f1bfa322
    wmo push dropped knowledge/ and auto_fidelity.json, so the        master's e36d2ef8; round trip now byte-identical
    hosted model could not serve its own measured-best reason+kb
    push digests were not content-addressed (gzip mtimes)             master's f57c1854
    installing a current wmo policy BROKE serving (HTTP 500): the     master's platform 6663bf83; serving now PROJECTS the policy
    platform validated the raw dump with the old wmh schema           onto the pinned wheel's schema, stored artifact stays true
    route report silently skipped the pareto write for STATIC         tau's 9ccbcca3; curve now emits over all 20 scenarios,
    policies, so the pinned endpoint shipped no curve                 recommending opus-5
    the model dir still held the kNN artifacts while the endpoint     closed by writing the pinned artifacts INTO the model dir;
    carried the pin, so `wmo serve` and GET /config would have        adopted as a program rule
    served the router and a 6-scenario curve

OPERATIONAL TRAPS worth not repeating: `wmo runs backfill` REFUSES to replay a live-emitted run without `--force`, and this lane's periodic loop silently no-op'd for seven hours while the panel sat at 7 cells; the monitor printed the TAIL of that refusal, which is an error box's bottom border, and read as success. Never let a monitor's success line be the last line of output. The forced replay was exact: 76 new, 6 already recorded, spend equal to the ledger to the cent.

## Reproduction

    uv run python .agents/scripts/run_swe_grid.py --episodes 2 --concurrency 12
    uv run --extra viz python .agents/docs/research/corners/common/build_corners.py \
        --lens bench-defaults-swe --anchor fable-5 --loo

Artifacts: matrix, per-cell records and ledger at `.wmo/jt/bench-defaults/swe/swe-cachefix-8bd6c3a11dea/`; `policy.json`, `report.json`, `pareto.json` at `.wmo/models/swe-bench/`; `numbers.json` and both figures at `.agents/docs/research/bench-defaults/swe/`.
