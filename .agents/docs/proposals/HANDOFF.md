# Transfer prompt — WM scenario-mining → filtered-BC ablation (assume no prior context)

> **STATUS 2026-07-04: EXPERIMENT COMPLETE.** The ablation ran end-to-end; the hypothesis was
> NOT supported — bc-random (36.5% success / 0.581 pass-rate) ≥ bc-mined (22.2% / 0.488) ≥/≈
> base (27.0% / 0.425) on the 21 held-out scenarios, k=3, gpt-5.4 WM, Opus 4.8 judge; no paired
> comparison reaches p<0.05 at n=21. Full numbers + verdict in the journal
> (`~/Documents/claas-verl/experiments/tau/07_03_2026_wm-mined-scenario-distill.md`) and the
> PR #81 comment. Everything below is kept for provenance.

You are taking over an autonomous experiment mid-flight. Read this whole file, verify the live
state yourself (don't trust it blindly — a background job may have finished or died), then
continue. **Everything you need is on disk; nothing critical is only in chat memory.**

## 1. The one-sentence goal

Prove (or disprove) that **scenarios mined from traces are more valuable than randomly-chosen
traces for training an agent** — via a clean ablation: filtered behavior cloning of a base
Qwen3.5-9B on MINED vs RANDOM scenario pools, everything else held equal. This is the *decisive*
experiment the PR needs; earlier work only showed the pipeline runs end-to-end.

## 2. Where things live

- **Repo (worktree):** `/Users/admin/Documents/experientiallabs/world-model-harness-scenario-construction`
  branch `feature/scenario-set-construction`, PR #81 (`gh pr view 81`). HEAD = `5a6e881`.
  Run `uv run ruff check . && uv run ty check && uv run pytest -q` before any commit (repo rule;
  AGENTS.md governs). `.agents/` is the disposable workspace — experiment scripts + results live
  under `.agents/scripts/` and `.agents/docs/research/distill/`.
- **Experiment journal:** `~/Documents/claas-verl/experiments/tau/07_03_2026_wm-mined-scenario-distill.md`
  (append results here; claas-verl is a separate git repo — the RL training stack).
- **GPU box:** `azureuser@4.154.170.26` (NOT .179 — that's a typo in an earlier instruction),
  2×H100. **GPU 0 is the user's `qwen35pilot` tmux — DO NOT TOUCH.** GPU 1 is ours. Real disk is
  `/mnt/azureuser` (251 GB); `/data` is an empty decoy; root has ~5 GB free (LoRA adapters only).
- **Credentials:** map is in your memory file
  `~/.claude/projects/-Users-admin-Documents-experientiallabs/memory/platform-secrets-env-local.md`.
  Never print key values. The scripts load keys themselves (see `collect_teacher.py::_load_gemini_key`).

## 3. The approved model stack (the user was very specific — do not substitute)

- **World model = `gpt-5.4`** via Azure AI Foundry (`foundry("gpt-5.4")` in
  `.agents/scripts/collect_teacher.py`). Endpoint + key auto-loaded.
- **Judge = Claude Opus 4.8** via AWS Bedrock, profile `claas-bedrock`, **region us-east-2 only**
  (`opus_judge()` helper). Opus is access-denied in us-west-2 and not deployed on Foundry.
- **Student = Qwen3.5-9B**, served by vLLM on the box, port 8001 base + LoRA adapters, tunneled to
  localhost:18002. `wmh scenarios` mining synthesis also uses gpt-5.4 with Opus as validation judge.
- Forbidden per user: Nova (called "shit"), and don't silently swap in Kimi-2.5/DeepSeek-v3.2/
  gpt-5-mini — those were rejected. If gpt-5.4 or Opus is unreachable, STOP and tell the user;
  do not substitute.

## 4. What is DONE (committed)

1. **`wmh/scenarios/` mining package** (the actual PR deliverable): facets → cluster → hybrid
   select → synthesize → **inline checklist validation** (back-agreement against the source trace,
   regen once, drop on repeat fail — folded INTO `build_scenario_set`, so an invalid scenario
   never leaves the build; `wmh scenarios verify` remains only for extrinsic WM-solvability). 573
   tests pass, ruff+ty clean.
2. **`BedrockProvider`** gained a Converse path (for Kimi/DeepSeek) and Nova path — general, keep.
3. **An earlier teacher-based distillation** (now superseded by this ablation) showed base→SFT
   +14.3pts on a WM eval but was confounded (privileged teacher hint = re-rendered traces). Its
   artifacts remain in `distill/` (teacher_episodes.jsonl, eval_student-*.json). The clarity
   report artifact and PR comments already scope those honestly.
4. **Both ablation pools built** (uncommitted, in `distill/`): `bc_pool_mined.json` (60 scenarios,
   21 clusters) and `bc_pool_random.json` (60 scenarios, uniform draw). 3/60 source-trace overlap
   — correctly distinct. Same gpt-5.4 synthesis + Opus validation for both; only SELECTION differs.

## 5. What is RUNNING right now (updated 2026-07-04, post-restart)

The first mined-arm run was killed at the user's request before it wrote output; everything was
restarted from scratch on a clean serving stack.

- **BC collection on the MINED arm (restart):** `uv run python .agents/scripts/collect_bc.py
  --pool bc_pool_mined.json --out bc_mined.jsonl --samples 6` (bash task `bp92fsa4n`). It writes
  `distill/bc_mined.jsonl` only at the END. The base student self-rolls each scenario in the
  gpt-5.4 WM; Opus judges vs the checklist; passing episodes (with the student's own `<think>`
  reasoning) are kept.
- **Student vLLM (ours, new):** box tmux `wmh_student_bc`, GPU 1, port **8010**, serving base
  `Qwen/Qwen3.5-9B` with `--reasoning-parser qwen3 --dtype bfloat16 --max-model-len 32768`.
  Launch needs `PATH=/mnt/azureuser/venvs/vllm/bin:$PATH` (ninja for GDN kernels) and
  `HF_HOME=/mnt/azureuser/hf_cache`. Log `~/qwen35_bc_vllm.log` on the box. Tunnel
  `ssh -N -L 18002:localhost:8010 azureuser@4.154.170.26` → localhost:18002.
  **The `--reasoning-parser qwen3` flag is required** — without it `message.reasoning` is empty
  and think-text leaks into `content`, corrupting SFT targets.
- **Casualty of the kill:** the user's pilot APIServer (was pid 2140398, port 8001) is DEAD; its
  orphaned EngineCore (pid 2140894) still holds 81 GB on GPU 0 inside the user's `qwen35pilot`
  tmux. NOT touched (user's territory) — user informed. Port 8001 is free but avoid it to prevent
  confusion with the pilot.
- **Box disk warning:** `/mnt` is 100% full (~2 GB free). Adapters (~200 MB each) still fit but
  clean up before anything bigger.

## 6. Immediate next steps (in order)

1. **Wait for `beqh99d2c` to finish**, then audit: `bc_mined.jsonl` should have ~40–80 episodes,
   each with `<think>` blocks and valid tool calls. Rule 12: actually read one episode.
2. **Run the RANDOM arm identically:** `uv run python .agents/scripts/collect_bc.py --pool
   bc_pool_random.json --out bc_random.jsonl --samples 6`. (Run after mined to avoid doubling
   Foundry rate-limit load — or in parallel if limits allow.)
3. **Train two LoRAs on GPU 1**, one per arm, IDENTICAL hyperparams:
   `.agents/scripts/train_sft_box.py` (TRL, r=32, 3 epochs, ~23 min each). Ship each `bc_*.jsonl`
   to `/mnt/azureuser/wmh_distill/` and run in tmux on the box. Venv:
   `/mnt/azureuser/venvs/wmh-distill/bin/python` (needs `ninja` on PATH for Qwen3.5 GDN+LoRA — a
   known gotcha). Save adapters `adapter_bc_mined` and `adapter_bc_random`.
4. **Serve base + both adapters** from one vLLM (`--enable-lora --lora-modules
   bc-mined=... bc-random=... --max-lora-rank 32`), PATH must include the venv bin (ninja).
5. **3-way eval on the held-out set** `distill/eval_pool.json` (21 scenarios), 3 passes each,
   Opus judge, gpt-5.4 WM: `.agents/scripts/eval_student.py --endpoint http://localhost:18002/v1
   --model {Qwen/Qwen3.5-9B | bc-mined | bc-random} --label {base|bc-mined|bc-random}
   --wm-model foundry:gpt-5.4 --judge-model opus-judge --passes 3`. Report paired per-scenario
   deltas + per-domain (eval pool is telecom-heavy — read non-telecom columns).
6. **The decisive number:** does bc-mined beat bc-random? If yes, mining earns its keep for
   training. If no or within noise, say so plainly (like the earlier selection-vs-random
   correction that retracted an over-claim — the user values that honesty). Write it up in the
   journal + a PR comment; do NOT overclaim.

## 7. Landmines already hit (don't repeat)

- **The checklist judge must see the agent's FINAL message** and needs `max_tokens≥8192` (reasoning
  judges truncate verdicts → silent failures). Both fixed in `wmh/scenarios/verification.py`; the
  BC/eval scripts already append the final message as a judged step.
- **Verify model availability in the right region/profile before declaring anything unavailable** —
  check ALL the `.env.local` files (memory has the map). The user corrected me twice for
  substituting models and for missing credentials on disk.
- Student generation needs `max_tokens=10240` (Qwen3.5 reasoning eats the budget → 0-step episodes).
- `world_model.frozen()` context wraps all eval/collection rollouts so generated steps never
  pollute the WM's retrieval index — keep using it (the scripts already do).

## 8. Commit hygiene

Commit script/result changes to the wmh worktree as you go (they're under `.agents/`, exempt from
the gate but still committed). Keep `wmh/` changes gate-clean. Push to PR #81's branch. End commit
messages with `Co-Authored-By: Claude <noreply@anthropic.com>`.
