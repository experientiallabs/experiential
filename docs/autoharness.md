# Auto-harness: a managed agent runtime that evolves against the world model

`wmh.agent` is the **build-agent path**: an automatic, managed agent harness whose evolution is
steered by closed-loop evals against the world model. It closes the loop the rest of `wmh` leaves
open — it *produces* the traces `wmh build` consumes, and it *consumes* the world model those traces
build, using it as a cheap simulator to score and evolve harness variants.

```
        ┌─────────────── collect (real, E2B) ───────────────┐
        │                                                    ▼
   HarnessSpec ──> AgentRuntime ──> E2BEnvironment ──> Trace (OTel JSONL)
        ▲                                                    │
        │                                                    ▼
   evolve (mutate)                                      wmh build
        │                                                    │
        │                                                    ▼
   ClosedLoopReport <── GoldJudge <── AgentRuntime <── WorldModelEnvironment (WorldModel.step)
        └───────────────── eval (simulated) ─────────────────┘
```

The real environment (E2B) and the simulated one (the world model) sit behind **one interface**
(`AgentEnvironment`), so the *same fixed agent loop* runs both ways. That is what makes the search
sound: the runtime is fixed and only the `HarnessSpec` varies, so any score delta is attributable to
the harness, and "does the sim rank harnesses like reality?" is a matter of swapping the environment,
not the eval (see [`closed_loop.md`](./closed_loop.md), [`sim_real_agreement.md`](./sim_real_agreement.md)).

## Components

| Module | Role |
|---|---|
| `spec.py` | `HarnessSpec` — the **evolvable artifact** (system prompt, tool set, seed skills, loop knobs) + lineage. The AlphaEvolve "EVOLVE-BLOCK". |
| `tools.py` | The pi-style minimal tool surface: `bash`, `read_file`, `write_file` (env) + `save_skill`, `read_skill`, `submit` (harness). |
| `skills.py` | `SkillLibrary` — the agent's self-written skills, persisted as `SKILL.md` files, injected by **progressive disclosure** (names up front, bodies on demand). |
| `runtime.py` | `AgentRuntime` — the fixed, owned while-loop (12-factor "own your control flow") that drives any environment. |
| `environment.py` | `E2BEnvironment` (real microVM) and `WorldModelEnvironment` (`WorldModel.step`) behind one interface. |
| `capture.py` | Runs → `Trace` → OTel GenAI JSONL, gold-stamped, so captures feed `wmh build` directly. |
| `collect.py` | Real trace collection: run each task in a fresh E2B sandbox, capture. |
| `gold.py` | `GoldJudge` — semantic gold-assertion checking (fuzzy post-conditions, not brittle rules). |
| `closed_loop.py` | `evaluate_with_env` (env-agnostic core) + `evaluate_closed_loop` — the **fitness function**: k=3 passes per task against the world model → success rate + per-task Pareto vector. |
| `real_loop.py` | `evaluate_real` — the same scoring core against real E2B (the validation counterpart of the sim fitness fn). |
| `evolve.py` | `HarnessArchive` + `evolve` — the meta-loop: DGM parent selection, GEPA reflective mutation, instance-level Pareto retention. |
| `agreement.py` | `sim_real_agreement` — does the world model rank harnesses like reality? Outcome-agreement confusion + Spearman rank correlation. See [`sim_real_validation.md`](./sim_real_validation.md). |

CLI: `wmh agent collect | eval | evolve | gate | verify`. `gate` oracle-checks a task suite in real
E2B; `verify` runs the sim-real agreement study over an evolve archive — the number that says whether
sim-driven evolution transfers.

## What we stole, and from where

The design is a deliberate synthesis of the auto-/meta-harness literature (see the research memo):

- **DGM** (Sakana, [2505.22954](https://arxiv.org/abs/2505.22954)) — an **archive** of every scored
  variant with **stepping-stone parent selection** (∝ fitness, discounted by children), not greedy
  hill-climbing; fully **auditable lineage** (`spec.parent`) so any score jump is traceable. DGM's
  cautionary tale (it fabricated test logs, then stripped the detectors) is why the judge runs
  *outside* the variant's sandbox and success is checked against held gold, never self-reported.
- **ADAS** ([2408.08435](https://arxiv.org/abs/2408.08435)) — variants carry a **name + motivation**,
  so the archive doubles as legible design memory the meta-agent reads when proposing the next mutation.
- **AlphaEvolve** ([2506.13131](https://arxiv.org/abs/2506.13131)) / **OpenEvolve** — a **scoped
  evolvable artifact** (the `HarnessSpec`, not arbitrary code) and the **artifacts feedback channel**:
  the mutation prompt gets *why* the parent failed (worst tasks + unmet assertions), not just a scalar.
- **GEPA** ([2507.19457](https://arxiv.org/abs/2507.19457)) — **reflective mutation** over failure
  transcripts, and an **instance-level Pareto frontier** (`HarnessArchive.pareto_names`) so a variant
  that wins on *some* task subset survives an aggregate metric.
- **Voyager** ([2305.16291](https://arxiv.org/abs/2305.16291)) + **Anthropic Agent Skills** — a
  **self-written skill library** of reusable, human-auditable `SKILL.md` units, surfaced by
  **progressive disclosure** to keep always-loaded context small.
- **pi** (badlogic) — a **minimal core**: ~a handful of tools and a short system prompt; everything
  else is composed with `bash`. Token budget is a design gate.
- **Ralph** / **12-factor agents** — an **owned while-loop** with a machine-checkable **stop
  condition** (`submit` or a turn cap), not a framework hiding the control flow.
- **WebArena / OSWorld / AgentRewardBench** — tasks as **(setup, instruction, programmatic
  post-condition)** triples, with the post-condition judged **semantically** (rule-only checks
  under-report success).
- **WMA / Qwen-AgentWorld** — the **decoupled-simulator** pattern: the world model is a swappable env
  backend, so closed-loop eval is nearly free compared to snapshot-and-reset of a real environment.

## Usage

```bash
# 1. Collect real traces by running the baseline agent in E2B sandboxes (needs $E2B_API_KEY).
uv sync --extra e2b
export E2B_API_KEY=...
wmh agent collect tasks.jsonl --out .wmh/traces/agent.otel.jsonl --provider bedrock

# 2. Build a world model from the agent's own behavior.
wmh build --name shell --file .wmh/traces/agent.otel.jsonl

# 3. Closed-loop eval a harness variant against it (k=3).
wmh agent eval tasks.jsonl --name shell

# 4. Evolve harness variants, steered by closed-loop score deltas.
wmh agent evolve tasks.jsonl --name shell --generations 8 \
    --out best_harness.json --archive .wmh/archive.json
```

A task file is JSONL of `TaskSpec` (`task_id`, `instruction`, `gold` assertions, optional real-env
`setup`); see `examples/agent_tasks.jsonl`.

## Determinism & cost

Per repo rules there is no wall-clock or RNG entropy: parent selection is seeded (a hash of the
generation index), and captured OTel spans use a fixed tick, so a run is reproducible and captures are
byte-identical. Every closed-loop metric is the mean of **k=3** passes (never single-pass), matching
the project's eval-reporting convention. Collection is the only step that spends real sandbox time;
eval and evolution run entirely against the world model.
