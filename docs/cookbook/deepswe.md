# Cookbook: DeepSWE, credential-free routing on published long-horizon SWE trials

The routing walk with no API in the loop at all: the evidence is a benchmark someone else
already paid for, and the embedder is a small model running in-process. DeepSWE v1.1
(deepswe.datacurve.ai) publishes a dense per-trial table: 50 model-and-effort configs on the
one mini-swe-agent scaffold, 113 long-horizon SWE tasks (median 61 agent steps), roughly four
trials per cell with graded fail-to-pass rewards and measured USD costs. Converting it buys
routing supervision without spending a GPU-hour or an API dollar.

| Step | Command | Artifact |
|---|---|---|
| 1 | `wmo optimize route convert-deepswe <source> --embedding-cache <qwen3.json> --out bundle/` | `matrix.json` + `task_embeddings.npy` + `scenario_groups.json` |
| 2 | `wmo optimize route fit bundle/matrix.json --kind knn --fallback claude-opus-5@high --embedder local` | `policy.json` + bank sidecar |
| 3 | `wmo research deepswe-holdout bundle/` | the repo-grouped holdout table below |
| 4 | `reproduce run deepswe-coding113` | the bit-exact verdict on the published bundle |

## The conversion and its gate

`convert-deepswe` keeps the 41 configs whose models the price table covers (the OpenAI and
Anthropic families; the other 9 vendors would enter unpriced and poison every cost number) and
refuses to write anything unless every published config's pass@1 reproduces exactly from the
raw trials. The converted matrix reproduces the pre-split lab's numbers to the digit: strongest
arm `claude-opus-5@high` at graded 0.955, pass@1 0.729, $6.09/task over all 113 tasks. The
bundle is a build output published to Hugging Face
(`experiential-labs/wmo-deepswe-coding113`), never committed to git.

## The local embedder

Queries and tasks embed with `EmbedderSpec(kind="local")`: Qwen3-Embedding-0.6B in-process,
via MLX on Apple silicon or torch on CUDA/CPU (`wmo.providers.local_embed`, the `local`
extra). Weights download from Hugging Face once; after that the whole routing path (embed,
retrieve, decide) is offline and credential-free. Fits and reproductions replay the bundle's
RECORDED vectors (`task_embeddings.npy`) instead of re-embedding, which is what makes the
reproduction bit-exact; live re-embedding reproduces the neighbor structure, not the bits.

## The grouped split

DeepSWE tasks from one repository share code (38 of the 113 tasks share a repository with
another), so this benchmark's fit/report split is by REPOSITORY
(`split_router_scenarios_grouped`): the held-out claim is about repositories the fit never
saw. An ungrouped split quietly inflates coding-router evals with same-repo near-duplicates.

## Measured result (2026-07-31)

`wmo research deepswe-holdout` over 6 salted repo-grouped splits, guard pinned to
`claude-opus-5@high`, recorded vectors throughout:

| dial | median cost ratio | median graded delta |
|---|---|---|
| 0.25 (default) | 0.97x | -0.001 |
| 0.50 | 1.26x | -0.009 |
| 0.75 | 1.34x | -0.009 |
| 1.00 | 1.40x | -0.007 |

Read the first row as the guard doing its job on a near-parity pool: the strongest arm is
also the guard baseline, the champion optimizes quality first, and at the default dial it
holds that arm's quality while spending about the same. The savings end tops out at 1.40x.
The pre-split coding-router lab, on this same matrix under its own protocol (cheapest arm
whose predicted solve odds clear a threshold, 6 seeded 80/20 repo splits), measured a 3.18x
median cost ratio (range 0.90x to 5.25x) at graded delta -0.015. That gap is a finding about
DECISION RULES, not a broken port: the threshold rule abandons the priciest arm aggressively
and accepts the quality variance; the guarded champion refuses a switch without paired
evidence, which buys safety and pays for it in forgone savings on pools whose arms cluster
near parity. Re-measure both sides with `wmo research deepswe-holdout` before quoting either
number on a different pool.

## Reproduce it

```bash
# in the research repo: github.com/experientiallabs/research
uv run reproduce run deepswe-coding113
```

Downloads the pinned bundle, replays the exact grouped-split fit and report offline, and
compares against the published row (quality parity at $5.04/task routed vs $5.40 baseline on
the held-out repositories). Exit code 0 is REPRODUCED, 4 is DIVERGED.
