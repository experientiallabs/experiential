# W16 router and sandbox evidence

W16 verifies the current public router workflow and the text-versus-sandbox comparison path with
deterministic injected clients. The evidence invokes no hosted provider, E2B, Tinker, credential,
or paid service. It does not claim provider quality or cloud-environment parity.

## Router workflow

`wmo/workflow/router_evidence_test.py` drives `wmo.compose_router` from 100 normalized traces.
The build produces 50 fit tasks and 20 sealed held-out tasks. A reviewed rubric and calibration,
two frozen candidates, one pricing snapshot, ten production overlaps, and one frozen embedder feed
the real text `WorldModelSimulator`, W10 fit lock and held-out report, and W11 HTTP runtime.

The exact deterministic run retains these denominators:

- 150 planned candidate-task cells, with 10 observed fidelity cells and 140 simulated cells
- 40 held-out report rows, with zero failed, not-run, or missing-cost rows
- 150 persisted judgments under one workflow ceiling of 200
- one shared simulation ceiling of $2.00 and exactly $0.00 observed fake-client spend

The test crashes once after the fit lock and once after durable report and telemetry creation. Resume
and exact replay add no model, world-model, judge, approval, or telemetry delivery. Two HTTP turns
with one episode ID retain one hashed episode identity and one sticky routed alias without exposing
the raw episode ID.

## Text and local process comparison

`wmo/simulation/comparison_evidence_test.py` runs both the real text simulator and the bounded
Darwin `LocalProcessEnvironmentRuntime` through `SandboxSimulator`. Two exact post-lock pairs are
retained: both are paired, one is usable, and one records an explicit malformed sandbox response.
The report therefore keeps one sandbox failure in the denominator instead of silently dropping it.
Replay performs zero additional model, world-model, or local-process dispatches and leaves no
workspace process directory. Terminal agreement is structural, not a task-quality judgment.

Run the evidence with:

```console
uv run pytest -q wmo/workflow/router_evidence_test.py \
  wmo/simulation/comparison_evidence_test.py
```

The local process case is intentionally skipped outside Darwin because that runtime's containment
contract is Darwin-only. Release tests separately require every public W16 owner in the wheel and
resolve the public APIs without importing test modules.
