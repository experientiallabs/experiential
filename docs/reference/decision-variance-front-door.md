# Decision-variance front door

Point the decision-variance pipeline at a trace file and run everything after it with one
command and one consent: the project build, decision tagging, tag clusters, environment
families, per-family rules, registered starts, and the six-stage pipeline for every start.
Every step resumes from its saved artifacts.

## Run it

Write a `FrontDoorSpec` (`exp/optimize/evaluation/variance/frontdoor.py`) and pass it to the
`decision-variance` continuation action as `front_door`:

```json
{"action": "decision-variance",
 "decision_variance": {"front_door": "front-door.json", "execute": false},
 "spend": {"ceiling_usd": 50.0, "unpriced_attempts": 100000}}
```

```bash
exp eval PROJECT --root ROOT --continuations plan.json
```

The spec has four parts:

- `traces`: the trace file (`path`, `source`, optional gateway `identity`). Omit it to use the
  project's completed build as it is.
- `preparation`: a `BootstrapPreparationSpec` (variance build with environment families and
  a compatibility reviewer, bootstrap roles, finite limits).
- `pipeline`: a `PipelineTemplate`, every pipeline setting except the build and the start;
  `mining.job` and `mining.bootstrap` stay empty because each registered start fills them.
- `slots`: settings for stages other owners add (see below).

The default output is the dry run: every phase with its state, then the completed work, the
remaining priced calls, the unpriced components (including phases that can only be priced once
an earlier phase exists) and the state machines below. With `"execute": true` the action asks
consent once and runs or resumes.

## Phases

1. `project-build`: `exp build` runs non-interactively in a subprocess over the trace file. It
   keeps its own embedding preflight and stops instead of exceeding its configured budget. A
   `front-door-source` record binds the trace file's digest to the build it produced, so a
   rerun over the same file skips it.
2. `bootstrap-preparation`: `run_bootstrap_preparation` creates the start-bootstrap batch from
   the build: tags, clusters, the family review and per-family scopes with probe routing,
   family rules learned once and frozen into each family runtime, and one authored, reviewed,
   captured and validated start per selected probe task. Nothing is registered by hand.
3. `pipeline <task>`: one `run_decision_variance` per registered start, with the saved build
   pinned and `mining.bootstrap` bound to that start. The rules stage reuses the family's
   frozen rules with no call.

The actor horizon of each start defaults to the full length of its longest source trajectory
(`JobTemplate.maximum_model_calls = None`, within `horizon_ceiling`), so each pipeline plan
prices the real horizon.

## One cumulative ledger

One consent gives one dollar ceiling and one unpriced-attempt allowance for the registration.
`PhaseLedger` (`frontdoor_ledger.py`, saved as `<project runtime>/front-door/<spec>/ledger.json`)
is never reset and never split per tier or start. Before a phase starts, its bound from its
owner's plan of the work it will run is reserved; a bound that does not fit what is left is
refused before the phase starts (`stopped_at_cap`). After the phase, its measured usage
replaces the reservation. Before each phase the run shows its dry-run view: completed work,
the phase's remaining calls, its unpriced components and what is left of the ledger.

## Self-hosted Clef

Clef is the default decision alias (`DEFAULT_DECISION_ALIAS`) for tagging, TraceML, rule
memory and recall, error tags and behavioral differences, with no fallback. Serve it with
SGLang on the SystemOne wire and reach it through a loopback tunnel:

```toml
[connections.clef-local]
provider = "typesafe"
base_url = "http://127.0.0.1:30001/v1"
self_hosted = true

[models.clef]
connection = "clef-local"
model = "Cloudflare/clef"
revision = "cfddda043dd190ef3b01cff449d5ee8cff1e8ec6"
billing_source = "customer_managed"
```

`self_hosted` is accepted only for the `typesafe` provider and only with a loopback base URL;
plain HTTP to any other host, or a custom origin without the flag, is refused. Every model on
the connection must pin its weights `revision`. The connection identity omits the port, so
moving a tunnel does not change the model snapshot or any cache key.

## State machines

Each phase and the run move only through these tables (`frontdoor_states.py`); every move is
checked, an illegal one is refused, and each is appended to
`<project runtime>/front-door/<spec>/transitions.jsonl`.

Run:

```mermaid
stateDiagram-v2
    [*] --> planned
    planned --> consented: consented
    planned --> replaying: replay
    consented --> running: started
    replaying --> finished: finished
    replaying --> failed: raised
    running --> finished: finished
    running --> stopped: stopped
    running --> failed: raised
    failed --> [*]
    finished --> [*]
    stopped --> [*]
```

Phase:

```mermaid
stateDiagram-v2
    [*] --> blocked
    [*] --> done
    [*] --> planned
    blocked --> planned: priced
    blocked --> done: replayed
    planned --> approved: approved
    planned --> stopped_at_cap: refused
    planned --> done: replayed
    approved --> running: started
    running --> done: finished
    running --> stopped_at_cap: stopped
    running --> failed: raised
    stopped_at_cap --> planned: replanned
    failed --> planned: replanned
    done --> [*]
```

Bootstrap task status (`bootstrap_states.py`, journaled beside the batch manifest):

```mermaid
stateDiagram-v2
    [*] --> failed
    [*] --> refused
    [*] --> registered
    registered --> failed: failed
    refused --> failed: failed
    failed --> failed: failed
    failed --> registered: registered
    failed --> refused: refused
```

## Slots for other stages

`frontdoor_slots.py` names where new stages join, each with a fixed position and the owner
entry points it calls. A configured slot without a registered `SlotExecutor` is refused at
planning, before any call.

| Slot | Position | Owner entry points |
|---|---|---|
| `tag-banks` | after the families review | `corpus_tag_bank`, `authored_bank_for_view` |
| `task-types` | after the families review | `task_types`, `task_type_family` |
| `rule-admission` | after preparation | `MemorySetSettings.admission`, `memory_stage_costs`, `run_rules_stage`, `load_guard_table`, `guard_rulebook`, `check_proposed_transition` |
| `state-rag` | after preparation | `plan_state_rag_calls`, `GroundedRuntimeDefinition.state_memory` (`StateMemoryPolicy`), `GroundedWorldFactory`, `GroundedTransitionEngine.prepare` |
| `blast-radius` | after the pipelines | `MiningSettings.qualification` (`PointQualification`): `from_bank`, `bar_instruction`, `check_picks`, `schedule_points`, `order_by_blast` |
| `curriculum` | after the pipelines | `judged_attempts`, `ActorLadder`, `difficulty_tiers`, `merge_curriculum`, `persist_difficulty_curriculum`, `curriculum_plan` |
| `dense-hybrid` | after the pipelines | `dense_hybrid`, `behavior_adversarial`, `adversarial_rubric` |

`after-families` slots run once the preparation's variance build has saved the reviewed
families, before any start is authored; the preparation then replays that build.
`MiningSettings.qualification` is refused at planning until its owner's per-phase planner
lands, so a registration never silently falls back to the per-selector point cap.
