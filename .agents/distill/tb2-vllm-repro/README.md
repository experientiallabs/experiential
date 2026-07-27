# Reproducing a Tinker-trained LoRA off Tinker, on local vLLM

Takes an on-policy-distilled LoRA checkpoint produced on Tinker, merges it into a full
HuggingFace model, serves base and merged side by side in vLLM, and evaluates both on the
TerminalBench-2 holdout through harbor's `terminus-2` agent with E2B sandboxes.

The point is to answer a question Tinker itself cannot: **does the trained policy survive
leaving the training service?** Tinker's sampler and our vLLM are different inference stacks,
so a lift that only exists on Tinker is not a property of the weights.

Run of record: student `Qwen/Qwen3.5-9B`, teacher `Qwen/Qwen3.6-27B`, LoRA rank 32,
sampler `pi-step-0004-b1604882` from `qwen-train-v3`. Numbers live in Notion, not here.

## Two findings worth not rediscovering

**vLLM cannot serve this adapter as a LoRA — you must merge it.** Tinker trains with
`train_unembed=True`, so the adapter carries `unembed_tokens.lora_A/B`, which the export
remaps to `lm_head`. vLLM only admits LoRA on embedding modules when the model class declares
a non-empty `embedding_modules`, and `Qwen3_5ForConditionalGeneration` declares `{}` — so
`LoRAModel.from_local_checkpoint` rejects the adapter outright. Independently, the adapter
stores split `linear_attn.in_proj_q/k/v` while the base and vLLM use a fused `in_proj_qkv`,
which is not exactly representable as a rank-32 LoRA on the fused tensor. Merging sidesteps
both: `merge_strategy="auto"` concatenates the split deltas into the fused weight.

**A merge can silently no-op, so assert coverage rather than eyeballing the output.**
`verify_merge.py` checks that the set of base tensors that actually changed equals the set the
adapter targets, after applying the split→fused and unembed→lm_head remaps. Sampling a few
tensor names by hand is not enough: an obvious pick like `layers.0.*` hits `input_layernorm`,
which `all-linear` never targets and which is *correctly* unchanged. A green "weights differ"
from the wrong tensor is worse than no check.

## Sequence

```bash
# 1. Export + merge  (CPU-only, ~19 GB out; output_path must not exist)
python export_and_merge.py \
  --tinker-path 'tinker://<uuid>:train:0/sampler_weights/<name>' \
  --base-model  /scratch/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
  --output      /scratch/repro-tb2/qwen35-9b-distill-v3

# 2. Prove the merge landed before spending sandbox budget
python verify_merge.py \
  --base /scratch/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
  --merged /scratch/repro-tb2/qwen35-9b-distill-v3 \
  --adapter /scratch/repro-tb2/adapter-raw

# 3. Serve both arms, one model per GPU (TP=1 each so the arms run concurrently)
./serve_arms.sh

# 4. Confirm the arms are not the same model
python tripwire.py            # greedy decode, must differ on every prompt

# 5. Evaluate  (arm-base.yaml is the canonical config; derive the other arm from it)
harbor run -c arm-base.yaml --yes
harbor run -c arm-distill.yaml --yes

# 6. Score, paired over tasks
python score.py --base-jobs <dir> --distill-jobs <dir>
```

## Condition an attrition rate on winnability before calling it a bias

`score.py` reports errored counts per arm, and it is tempting to read a large asymmetry as a
large bias. Don't, without checking *where* the losses fall. On this run the distilled arm
errored on 55.1% of trials against the base arm's 14.0% — every one of them an
`AgentTimeoutError` at the episode wall, in both arms, with no other mechanism. That looks like
it must dominate the result. It did not: **68 of those 75 timeouts landed on the 12 tasks that
score 0.000 in both arms**, so the truncation happened where there was nothing to win.
Conditioning on winnability, the distilled arm lost at most 7 trials and the base arm 1, worth
roughly +0.04 on the paired delta rather than the "this is only a floor" reading the raw rate
invites.

The same crosstab is what separates capability from speed. A task where one arm times out 8/8
and the other 4/8 produces a solve-rate delta that is substantially "finishes inside the
budget", not "solves better" — a real advantage for a deployed agent, but a different claim.
Cross timeouts against per-task rates before attributing any mover.

## Environment notes

These bit a real run; they are cheap to pre-empt.

- vLLM's flashinfer sampling kernels are JIT-compiled at startup and need **both** `ninja` and
  the CUDA **dev** headers. A box can have a working CUDA runtime and still be missing
  `curand.h` (`libcurand-dev-13-0`). Put `$CUDA_HOME/bin` on `PATH` for the server process.
  Do not reach for `enforce_eager` or a triton fallback to make the error go away — that
  changes throughput, not the missing toolchain.
- **Serve with headroom above the agent's context budget, or compaction dies silently.** vLLM
  rejects a request when `prompt + max_tokens > max_model_len`, so setting `max_model_len` to
  the agent's `context_budget_tokens + max_tokens` exactly leaves no room: terminus-2's
  proactive summarization sends slightly *more* than the budget and gets a 400. litellm
  swallows the body, so the only symptom is `Error in proactively summarizing:` with an empty
  message, followed by `Even fallback chat failed:` — and the run continues at full speed with
  compaction disabled. Trials still complete and `exception_info` stays null, so nothing marks
  the results as invalid. Reproducing a Tinker run is where this bites hardest: Tinker's 65,530
  ceiling forces budgets that sum to just under it, and copying that sum to `max_model_len`
  reproduces the ceiling without reproducing the behaviour. Serve well above it (we used 81,920
  against a 53,240 + 12,288 envelope) and leave the agent's own budget untouched.
- Harbor task names are namespaced: `terminal-bench/<task>`, not `<task>`. An unprefixed
  filter raises rather than silently selecting a subset, but assert the resolved count anyway.
- Concurrent harbor runs can `rmtree` each other's task cache. Give each arm its own
  `download_dir` and `jobs_dir`, pre-warmed with `install_only: true`.
- Keep `n_concurrent_trials` at or below the server's `--max-num-seqs`. Oversubscribing raises
  per-token latency, which pushes episodes into the wall clock and biases the result toward
  whichever arm happens to be faster.
