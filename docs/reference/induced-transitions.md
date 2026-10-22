# Induced transition programs

Induced transition programs turn the deterministic part of a simulated world into exact Python,
learned from fit traces. Each program covers one tool: it takes the tool's visible state and the
call's arguments and returns the reply and the next state. A program serves only the inputs it
was validated on. Every other call goes to the grounded simulator.

This addresses two measured limits of the grounded simulator. Without the starting state it
invents content. Simulated episodes also run slower than native ones. Where a program applies,
its reply is exact and comes from a local process rather than a model call.

## Workflow

Both operations are `exp eval --continuations` actions:

```bash
exp eval PROJECT --root ROOT --continuations induce.json --dry-run
exp eval PROJECT --root ROOT --continuations induce.json --yes --non-interactive
exp eval PROJECT --root ROOT --continuations validate.json --non-interactive
```

### Inducing programs

`action: "induce-transitions"` takes an `induction: InductionSpec`, defined in
`exp/optimize/evaluation/induction/service.py`. The spec holds:

- `manifest`: the world manifest of the simulated world. It is the same manifest as the grounded
  runtime definition the programs will be composed with.
- `initial_state`: the declared starting value of the simulated domains. Use `null` when the
  starting world is unknown.
- `inducer_alias`, `inducer_model` and `catalog_sha256`: the model that proposes programs, and
  the catalog it comes from.
- `settings`: the per-tool limits, as `InductionSettings`:
  - `maximum_attempts`: proposals per tool;
  - `minimum_support`: exact fit steps a program needs before it is accepted;
  - `maximum_prompt_characters`: evidence per request;
  - `maximum_feedback_failures`: failing steps quoted back when a program is re-asked;
  - output tokens and temperature.
- `limits`: the `SandboxLimits` that programs are validated and executed under.
- `maximum_episodes`, `maximum_physical_attempts` and `deadline`.

The episodes come from the completed build's fit traces, in dataset order. Each episode holds
every recorded exchange, including the public prefix. It also holds the trace package's
version-0 conditioning instance facts, which every program receives read-only as `facts`.

Held-out traces are never read. The request for one tool carries:

- the tool's schema and declared effects;
- the effects of every other tool;
- the episodes that call the tool.

`--dry-run` stops after these checks, with no credential and no call:

- the catalog and model;
- the admitted transport;
- whether the fit evidence includes an inducible tool;
- the attempt ceiling.

It prints the episode count, the tools that can be induced and the most proposals it could
make. A paid run then asks for spend consent before resolving credentials. The estimate prices
`maximum_physical_attempts` requests at full input capacity.

Every draw saves a pre-dispatch intent and then the raw response. After each draw the run saves
the parsed proposal, and after each joint validation round it saves the round's report. A saved
induction replays with no consent, credential or call. An intent without a response is an
unknown outcome and is never redrawn.

The printed summary gives each tool's verdict:

- `status` and `reason`;
- `exact`, `evaluable` and `unverifiable` step counts;
- the `induced_tools` and `induced_domains` that induced code owns.

### Re-validating programs

`action: "validate-transitions"` takes a `transition_validation: TransitionValidationSpec`. It
replays the saved programs against the build's current fit evidence, which suits a rebuild that
added traces. It needs no provider and is deterministic. `minimum_support` can be raised for
this run only.

The saved result includes `unchanged_evidence` and a new coverage. With unchanged evidence the
report equals the induction's. New evidence can only keep or shrink what induced code owns.

### Using the programs as a world backend

The built-in runtime file `exp/optimize/evaluation/executable/induced.py` exports
`continuation_runtime`. Its configuration is `InducedRuntimeDefinition`:
`{"grounded": GroundedRuntimeDefinition, "coverage": ArtifactInput}`. `coverage` names a saved
induction or a saved re-validation, which must have been validated under the grounded
definition's manifest.

Use the runtime file in a continuation `job`, a `capture`, or `compare-environments` (with the
grounded runtime as the reference) like any trusted runtime. Its cost inventory is the grounded
predictor's allowance, because induced calls make no provider request. Per-trial provider grants
(`trial_runtime`) are not composed.

## Validation rules

`validate_programs` in `exp/simulation/induction/validation.py` replays each fit episode in
order from its starting state. One exact state is threaded through the calls. A program runs on
each step of its tool whose read domains are known. The step is exact only when all of these
hold:

- the reply equals the recorded reply under strict JSON equality;
- the error flag matches;
- every recorded post-step state matches;
- only the tool's declared write domains changed;
- every domain still satisfies its declared state schema.

When the state a step reads is unknown, the step is `unverifiable`. That happens when the
episode has no starting state for it, or when an earlier step of a tool without an accepted
program, or a failing step, wrote it. Unverifiable steps count neither for nor against a program.
Recorded post-step state makes a domain known again.

Validation is joint and conservative:

- A program with any failing step is rejected.
- After each rejection the replay runs again without the rejected program, until a round rejects
  nothing. One wrong write could otherwise make another program's read look exact.
- Programs with fewer exact steps than `minimum_support` are rejected.
- Ownership then follows the world kernel's rule that no tool spans owners. A domain is induced
  only when every simulated tool that touches it has an accepted program.

Every proposal is retained with its validation report. Each report gives each program's verdict
and up to twenty failing steps, with the expected and actual replies. When a tool is re-asked,
the model sees its previous program and those failures.

## Execution

Programs are model-written code, so they never run inside Experiential's process.
`ProgramSandbox` (`exp/simulation/induction/sandbox.py`) owns one long-lived worker,
`exp/simulation/induction/runner.py`.

The worker runs under these conditions:

- It starts without site packages and without the script directory on its import path.
- It runs in a private temporary directory, with an environment built from scratch and a fixed
  hash seed.
- Before it reads any program it sets its own limits on CPU time, address space and file size,
  and it cannot open new file descriptors. Even an escaped program can open no file, socket or
  pipe.
- Every request has a wall-clock deadline. A worker that misses it, crashes or breaks protocol is
  killed with its process group. Its outcome is `failed`.

Programs must stay within a Python subset:

- Imports are limited to the allowlisted modules: `collections`, `copy`, `functools`,
  `itertools`, `json`, `math`, `re` and `string`. Each is a view of its public, non-module
  attributes.
- Plain functions and constant assignments are allowed.
- Classes, `global`, `with`, `yield`, bare `except`, dunder names, underscore attributes,
  frame and traceback attributes, and `str.format` are not allowed.
- Only the listed safe builtins exist.
- A program defines `transition(state, args, facts)`. It edits `state` in place and returns the
  reply. It raises `ToolError(reply)` for an error reply and `OutOfDomain(reason)` for an input
  it does not model.

## Serving inside the validated domain

`InducedWorldFactory` (`exp/simulation/induction/backend.py`) is one simulated-side backend in
the world kernel. The grounded predictor backend holds the single JSON state and the call
history.

A call to an induced tool runs as code only when it stays inside the program's validated domain.
Its argument names must match a validated call, and every program line it runs must also have
run on a validated step. The reply and new state are then adopted into the predictor's state and
history, so later predictions see them.

Every other call goes to the predictor. That includes:

- calls with new argument names;
- calls that take a new code path;
- calls the program refuses;
- calls on which the program fails.

When the predictor executes a call that writes an induced tool's domains, that tool and every
tool sharing those domains hand them to the predictor for the rest of the episode. Induced
domains are therefore only ever written by validated code. The induced partition is part of
every image, so checkpoints, rollbacks, forks and restores keep the exact induced part.

The world stays the simulated owner. Replies carry `transition: "induced"` or
`transition: "predicted"` in their metadata. Trajectory export never treats either as native
evidence.

## Measuring the effect

`compare_world_replay` (`exp/optimize/evaluation/induction/comparison.py`) replays fidelity
`ReplayCase` values through two or more world arms, for example induced code with the predictor
against the predictor alone. A held-out panel the audit owner selected and reserved can be passed
unchanged.

Each arm opens a fresh world per case and applies the public prefix without scoring it. It then
executes each measured call, and reports for that arm:

- exact replies under strict JSON equality, broken down by the fidelity owner's reply kinds;
- how many calls induced code answered;
- per-episode wall time;
- the exact rate per lineage, with repeats averaged first.

It also reports the paired difference between the first two arms, by lineage. World and provider
failures are raised rather than scored. Reports contain measurements only.

## Limits

- Programs see only fit evidence. The Python subset and the resource limits contain mistakes,
  but they are not a security boundary against a hostile local user.
- Facts at later state versions are not checked as recorded state. Only recorded post-step state
  is compared.
- The validated domain is syntactic: validated argument names and code paths. A program that
  takes a validated path on unseen values can still be wrong where fit evidence never tested it.
- The comparison is a library call. The induce and validate actions never read held-out traces.
