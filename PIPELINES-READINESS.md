# Pipelines readiness (simulation/dv-pipelines-ready)

**What runs.** Four starts run in sequence. Build and rules replay from bootstrap. Then traceml, error-mining, families and records run over 5 tiers (G=16, J=C=8, M=8, horizon H=20, up to 20 points per tier). Worst-case calls per start:

| Stage | Actors | deepseek-judge | glm-flash | clef |
|---|---|---|---|---|
| traceml | 1.6k | – | 2.4k | ~45k bound, ~20k real |
| error-mining | 6.4k | 10.6k | 3.5k (2.4k of it blast) | – |
| families (5) | ~15k | ~10k | ~5k | – |

That is about 100k per start (~45k real) and about 400k for 4 starts. Every model is unpriced, so this does not fit the 100k per-operation cap. Run one start per operation, or re-invoke after the resumable stop.

**Error states and the positions before them.** TraceML tags every attempt step with error-taxonomy-v1 and the point bar. Selectors and critics pick the positions before the errors. M continuations measure each position. Blast radius adds a 3-stance severity review and decision variance. Outputs:
- blast ranking: an `error-points` artifact (`error-mining.outputs.blast`)
- curriculum: a difficulty-curriculum artifact plus one schedule per tier
- Harbor environments: `hf-agents/runtime/decision-variance/<records-id>/<tier>-<point>-<scenario>`

**Fixed (9 commits, one intent test each):**
1. The authoring packet named all 1,039 traces, over the 128 KB limit. It now names at most 32.
2. Groups kept the start's 64-attempt ceiling, which left most attempts outcome-unknown. The ceiling is now G×H.
3. Selectors, critics and continuations ran serially. They now run concurrently within admission, and stop at the first failure.
4. Blast reviews now run concurrently, and one ValueError leaves that point unscored instead of aborting.
5. One failing start blocked the rest. Now it is recorded and the others continue.
6. Group judging, and all tiers' traceml groups, now run concurrently. An over-long attempt is clipped instead of refused.
7. 429 handling:
   - Gateway completions (actors, critics, judges) gave up after 3 quick tries (2 s backoff cap). Each lost call became an outcome-unknown branch or an unscored judgment. They now get 8 extra paced 429 retries (1→30 s backoff, Retry-After as the floor, within the deadline).
   - `decide()` now retries 429/502/503/504 the same way.

**Admission today.** Gateway roles are admitted as provider `openai-compatible` (default 8, 4 per alias), not as `endpoint:api.experientiallabs.ai`. Clef decisions use `endpoint:127.0.0.1` (48), and some of them reach the gateway through the round-robin.

Recommended additions to `provider-admission.toml`:
```
breaker_threshold = 20
[providers."openai-compatible"]
max_in_flight = 12
```
Plus per-alias `max_in_flight` of 6 for the actors, `deepseek-judge` and `glm-flash`. Gateway clef traffic shares the same key, so keep the total under ~16.

**Risks, ranked:**
1. Blast radius is not cross-tier: each tier continues only its own checkpoints, so `failing_share` is 0 or 1.
2. The breaker opens after 5 consecutive 429s and then raises ProviderAdmissionRefused, which stops the run (resumably). Raise `breaker_threshold`.
3. Wall clock: branches, families and mining tiers are still serial.
4. `continuation_physical_attempts` is 64, but 8 continuations can need 160.
5. live_next4's `RefuseRun` must allow the blast-radius and curriculum slots.
