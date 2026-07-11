# Closed-loop evaluation (`wmh eval --mode closed-loop`)

Open-loop eval (`wmh eval <files>`) replays recorded steps teacher-forced and scores per-step
reconstruction fidelity. Closed-loop is the other half: a **live agent** runs a task with the world
model as its environment — the agent emits a tool call, the world-model LLM answers it, the agent
reacts, until it submits or hits a turn cap — and we score **task success**, not per-step fidelity.
It answers the literal "Docker as an LLM" question: *would my eval reach the same verdict against
the simulated environment as against the real one?*

## Running it

```bash
wmh eval tasks.jsonl --mode closed-loop --name <world-model> --k 3 --out sim_report.json
```

- `tasks.jsonl` — one task per line: `{"task_id": ..., "instruction": ..., "gold": ["...", ...]}`.
  `gold` is a list of plain-English assertions that define success (post-conditions on the final
  state, checked semantically).
- The **agent is fixed** (a minimal 4-tool loop: `bash`, `read_file`, `write_file`, `submit`) so any
  score movement is attributable to the world model, not the agent.
- Every task runs **k=3 passes** (the repo's eval-reporting convention); the score is the fraction of
  passes whose transcript satisfies every gold assertion, judged by an LLM judge that never trusts
  the agent's own claim of success.

### In-process Python harnesses

A path-less `code:runtime` surface is Python executed with the host process's privileges. Closed-loop
eval and harness search refuse it by default. After reviewing locally authored code, acknowledge
that trust decision explicitly:

```bash
wmh eval tasks.jsonl --mode closed-loop --harness reviewed-code --allow-unsafe-code
wmh harness create evolved --tasks tasks.jsonl --seed reviewed-code --allow-unsafe-code
```

`wmh pull` never saves registry harnesses containing `code:runtime`. The registry's document hash
checks transport integrity, not publisher authenticity, so the unsafe-code flag is intentionally
not a way to opt pulled code back in.

## Comparing two reports (`wmh eval agreement`)

```bash
wmh eval agreement sim_report.json real_report.json
```

Compares two saved closed-loop reports task-by-task: a 2×2 confusion of pass/fail verdicts, the
outcome-agreement rate, and the aggregate success gap. The canonical use is world-model vs a real
environment — the closed-loop validity check ("does the simulator reach the same verdicts as
reality?"). The second report can come from any execution backend that emits the same
`ClosedLoopReport` JSON; none ships in this repo (real execution is the platform's job).

The cell to watch is **A-pass & B-FAIL** with A = the world model: tasks the simulator credits that
reality fails. Anything optimizing against the simulator (e.g. harness search) would chase exactly
those.

## Where the pieces live

`wmh/evals/`: `base.py` (the general `Evaluation`/`EvalResult` interface), `open_loop.py`
(teacher-forced replay fidelity — the default `wmh eval` mode), `closed_loop.py` (k-pass live
scoring + `WorldModelEnvironment`), `gold.py` (gold-assertion judge), `agreement.py`
(report-vs-report comparison), `tasks.py` (task specs).

`wmh/harness/`: `runtime.py` (the fixed agent loop), `code_runtime.py` (explicitly trusted local
Python loops), `environment.py` (the `AgentEnvironment` seam the loop drives), `tools.py` (the tool
registry).
