# Retired learning traces

Learning (`exp build` and the fit-only learning of `exp eval PROJECT --audit N`) visits traces in
canonical dataset order. A trace whose request fails stops the pass; `--retry-incomplete`
resumes it in place. Most failures are outages and stay resumable forever. One kind does not
clear on its own: the gateway refuses the exact request it was sent. Retirement lets such a pass
finish, with the trace recorded as not learned.

## Automatic retirement

A failure qualifies only when it is a definite HTTP response whose `error.code` is registered
as a deterministic refusal (currently `invalid_provider_stream`: the provider's stream for that
request could not be encoded as a valid reply) and the client identified the refused request
body by digest. Each qualifying failure is saved as a `trace-learning-refusal` record: memory
lineage, trace, fidelity run, request digest, status, code and time.

- The first refusal stops the pass as before. The plain command never resends it.
- `--retry-incomplete` resends the identical request in place. After three recorded refusals of
  that identical request (the original and two resends), the trace is retired and the pass
  continues with the next trace.
- At most 10% of a lineage's learning traces (and at least one) are retired automatically.
  Past that bound the refusals look systemic, so the pass stops instead: the message names the
  code and the affected traces, the trace that tripped the bound keeps its recorded refusals and
  stays resumable, and nothing more is retired. Check the provider or model settings (for
  example the output allowance or reasoning effort), then resume with `--retry-incomplete`, or
  retire specific traces explicitly (below). Operator retirements do not count toward the bound.
- Transport outages, timeouts, overload (`gateway_overloaded`), conflicts, unknown outcomes and
  any unregistered code are never retired automatically. They stay resumable.

## Operator retirement

A trace an outage keeps blocking can be retired by name with the `repair-learning`
continuation, without a model call:

```json
{"action": "repair-learning",
 "learning_repair": {"resolution": "retire-trace", "scope": "fit",
                     "trace_id": "TRACE", "reason": "why it is retired"}}
```

```bash
exp eval PROJECT --root ROOT --continuations retire.json --dry-run
exp eval PROJECT --root ROOT --continuations retire.json --yes --non-interactive
```

`scope` is `fit` (the audit's memory, the default) or `all` (the build's memory). The trace must
be a learning trace of that scope, not learned, without a saved measurement, and not running in
another process. A held-out trace is never learned, so it cannot be retired. Refusals already
recorded for the trace are attached as evidence. Retiring it again with the same reason replays.

## What a retirement changes

A `trace-learning-retirement` record names the lineage, trace, cause (`deterministic-refusal` or
`operator`), reason, stopped run, refusal evidence and the memory version at retirement. It never
edits memory: the lineage identity and every published version stay the same, and held-out
reservations are unchanged. Later passes skip the trace without calls.

A retired trace is never scored and never dropped. The learning curve counts it among traces not
learned (`N trace(s) not learned (K retired, unscored)`), the learning output names each retired
trace with its cause and reason, and the audit checkpoint lists it in `retired` beside
`fit_traces_learned`.
