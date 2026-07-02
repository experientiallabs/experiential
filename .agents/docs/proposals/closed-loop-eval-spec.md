---
source: https://app.notion.com/38e0f8b3f591817bbc64ed9b22061d86
area: Research
status: Future direction
migrated: 2026-07-02
---

# Closed-loop evaluation (future direction)

Today the harness evaluates the world model **open-loop** (`wmh eval`, `wmh/engine/replay.py`):
each held-out step is replayed teacher-forced — feed the *real recorded* `(state, action)`, predict
the observation, score it against the *real recorded* observation. Nothing the model generates feeds
forward, so it is perfectly repeatable per step and isolates per-step fidelity. This doc specifies
the **closed-loop** eval we have deliberately deferred.

## What closed-loop is

A live agent runs a task **against the world model as its environment**. The agent emits a tool
call; the **world-model LLM** answers it (instead of the real environment); the agent sees that
predicted observation, updates its context, and acts again — until it submits or hits a turn cap.
Then we score **task success**, not per-step fidelity.

It is the literal "Docker as an LLM" promise: *would running my eval against the simulated
environment reach the same verdict as the real one?*

## Why it's separate from open-loop (and harder)

- **Not repeatable past step 0.** The moment the world model emits an observation that differs from
  the recording, the live agent reacts differently and the trajectory diverges. There is no
  ground-truth observation to compare each step against — only the *final outcome* is checkable.
- **Needs a live agent LLM in the loop.** Open-loop replays recorded actions; closed-loop must
  generate the agent's next action from the (now simulated) history, so it needs the agent model
  *and* a faithful reconstruction of the agent's prompt at each turn.
- **The agent is FIXED.** We are testing the *world model*, not the agent. The agent model, its
  system prompt, and its tool schema are held constant (reconstructed from the trace); the only
  substitution is real-env → world-model for tool results. Any divergence is attributable to the
  world model alone.

## What this requires that we don't have yet

1. **Per-step agent-prompt capture in traces.** Open-loop only needs `(state, action, observation)`.
   Closed-loop needs each step to also carry the agent's full turn context: the agent system prompt,
   the tool schema, and the running message list the agent saw. This is an additive extension to the
   trace schema (e.g. an optional `agent_context` on `Step` / in `Trace.metadata`) that the
   trace-capture pipeline must emit. Old open-loop traces remain valid (they just can't drive
   closed-loop).
2. **An agent driver.** Reconstruct the agent's prompt from captured context, call the agent via the
   `Provider` interface, parse its tool call, route it to the world model, append the predicted
   observation, repeat. Terminate on the benchmark's submit signal (each benchmark defines its own
   submit convention) or a turn cap (Qwen-AgentWorld used 50).
3. **Success detection against gold.** Traces already carry the task's **gold assertions** in
   `Trace.metadata` (the trace-capture pipeline stores them now for exactly this). Score the agent's
   final submission against gold with an LLM judge that *checks the assertions* (semantic, not
   brittle path-equality), yielding pass/fail.

## Metrics

- **Simulated task-success rate**: fraction of tasks the fixed agent passes (vs gold) when run
  against the world model.
- **Outcome agreement**: does the simulated run pass **iff** the recorded real run passed? Reported
  as agreement rate (and a 2×2 confusion: sim-pass/real-pass etc.). This is the headline closed-loop
  validity number we can compute *without* re-running the real environment — the recorded trace's
  pass/fail is the real outcome.
- (The stronger *policy-rank* validity question — does the sim rank multiple agents the same as the
  real env — is its own research direction: see [`sim_real_agreement.md`](./sim_real_agreement.md).)

## Reuse from the open-loop path

- `predict_observation` (`wmh/optimize/gepa.py`) is already the single rollout primitive and already
  accepts `temperature`; closed-loop calls it per agent tool call.
- The world model's session machinery (`WorldModel.step`, scratchpad state, retrieval) already
  advances state across a session — closed-loop is essentially "let an agent drive `WorldModel.step`
  in a loop and judge the end state," rather than replaying recorded steps.
- The `RubricJudge` pattern (reference-grounded, structured output) extends naturally to a
  gold-assertion judge.

## Sketch

```
run_closed_loop(task, agent_provider, world_model, gold, max_turns=50) -> ClosedLoopResult:
    session = world_model.new_session(task)
    history = reconstruct_agent_context(task)        # from captured agent prompt + tool schema
    for _ in range(max_turns):
        action = agent_provider -> parse tool call from agent's next turn given `history`
        if is_submit(action): break
        obs = world_model.step(session.id, action)   # world-model LLM answers the tool call
        history.append(action, obs)
    passed = gold_judge(session final state / submission, gold)   # vs gold assertions
    return ClosedLoopResult(passed=passed, turns=..., transcript=...)
```

Status: **not implemented.** Open-loop (`wmh eval`) is the shipping eval; this is the next layer.
