# Gateway guardrails

## One policy engine and request lifecycle

Every guardrail uses `GuardrailPolicy`, the same policy store, and a request-owned
`GuardrailSession` created by `GuardrailEngine`. Scope determines which checks apply;
it does not select a different execution pipeline. A policy without organization
and identity IDs applies to all identities and must be protected. Identity assignments
add checks and cannot replace or disable platform checks.

```python
engine = GuardrailEngine(
    store=MappingGuardrailStore((
        GuardrailPolicy(
            policy_id="criminal-abuse",
            revision="policy-v1:detector-revision",
            protected=True,
            input_execution="parallel",
            checks=(GuardrailCheck(
                check_id="criminal-abuse-input",
                capability="content_safety",
                stage="input",
                action="block",
                adapter_id="criminal-abuse",
                timeout_ms=1000,
            ),),
        ),
    )),
    client=DirectClassifierClient(classifier_registry),
    monotonic=time.monotonic,
)
control = NativeControlPlane(components, guardrails=engine)
```

The gateway resolves `policies_for(organization_id, identity_id)` once for admission.
The session captures every applicable policy, the original deadline, approval state,
and output requirements. Continuation expansion, rewrites, gateway search rounds,
response delivery, cancellation, and settlement all use that session. No later
lookup changes a request's policy midway through execution.

Policies run in resolved scope order, with platform policies before identity policies.
Within each input or output stage, checks run in authored order, so earlier privacy
rewrites feed later checks. Each check retains the subject it accepted or produced.
If a later rewrite changes that subject, the earlier check independently validates
the final result; conflicting rewrites fail closed even if they restore the original
text. Checks that already accepted the final result are not repeated. Recovered
plaintext reasoning and gateway-expanded conversation context are inspected before
the next provider dispatch. Exact duplicate subjects share work only within one
request. All policies use the same classifier executor, bounds, verdict actions,
and content-free decision recorder. Deterministic native detectors are an execution
strategy of that pipeline, with the same input and completion contracts.

`max_request_bytes` bounds the complete in-memory request, including fields omitted
from provider serialization, before native or async input inspection. A no-copy
character bound rejects obviously oversized input before projection; fitting candidates
receive an exact UTF-8 JSON size check. Tool examples, allowed callers, private tool
identifiers and search context cannot escape this bound. This does not expand the
generic HTTP classifier's declared wire projection or certify its coverage.

Embeddings, image generation, and native Decisions cannot yet use that normalized
inspection contract. An applicable enforcing platform or identity policy rejects those
endpoints before request acceptance, reservation, or provider dispatch. Identities
with no applicable policy retain access to those surfaces.

## Observation and controlled rollout

`GuardrailPolicy(mode="observe", protected=True, checks=(input_check,))` evaluates
the same input contract and authored action without blocking, rewriting, withholding,
cancelling generation, or waiving the customer's normal charge. `mode="enforce"`
is the default. Observation supports read-only input checks; configuration rejects
output checks and `modify` actions rather than promising incomplete observation.
Protected policy ownership remains independent of rollout mode. A host selects the
mode from authenticated organization scope, then the session freezes it with the
rest of the policy. Customers cannot choose the operator's mode.

The shared engine admits observations through the same bounded classifier executor
abstraction with a separate worker pool and adapter quarantine from enforcement.
It retains at most 8 jobs and 8 MiB of complete subjects by default, configurable
through `max_observations` and `max_observation_bytes`. Full queues, byte limits, or
expired request deadlines produce `skipped`, never an uninspected `allow`. A cold,
starting or unavailable scheduler produces `unavailable` and releases its unused job
and byte reservation. Optional admission can start the shared callback daemon, but
does not wait for its startup lock or event-loop readiness; later requests can observe
once that same loop is ready. Exact
duplicate subjects share work only within their request. While observation admission
remains open, each distinct authenticated context, including recovered history, private
provider fields and gateway tool results, is considered separately. Fields excluded from provider serialization still count toward
subject identity and retained bytes. Observer capacity or quarantine cannot consume
the engine's enforcement capacity or quarantine its enforcing calls.

All observing policies share one detached deep snapshot and one complete subject
encoding per admission boundary. Nested caller-owned containers are copied before
hashing, byte accounting and asynchronous inspection, so later caller mutations cannot
change an admitted subject or its deduplication identity.
The owner grants at most one nonblocking preparation permit, separate from retained
inspection jobs. New sessions reject before projection or hashing when inspection
capacity is full, admission is closed, or the original deadline has expired. Rejection
at admission has no oversized classification because the content was not projected.
A session with prior fingerprints may use a free permit to deduplicate an equal complete subject
even with full inspection capacity or an expired deadline. Concurrent preparation never
waits: `preparation_unavailable` ends that session's optional coverage and records one
`check=None, skipped` admission outcome per observing policy. This can include an equal
subject whose identity could not be compared while another preparation held the permit;
it is neither a classifier verdict nor evidence of a distinct missed subject.
Size-walk, snapshot, encoding or hashing failures record `check=None, unavailable` coverage
without exposing input or exception diagnostics, and serving continues.

Within the permit, a cheap character-count bound rejects clearly oversized subjects
without serializing their text. That rejection closes optional observation for this
request and records one coverage-admission outcome per observing policy: `unsupported`
above the policy bound, otherwise `skipped` above the owner's byte capacity. Later contexts
are not observed after terminal admission failure. The permit remains held until all
projection and hash temporaries are released, including failure paths, and shutdown
does not release a permit still held by a request thread. Preparation admits one candidate
under the owner's conservative size lower bound, in addition to retained-job bytes.
Canonical JSON can expand that single candidate before the exact byte check. These limits
do not claim an exact process memory ceiling.

An admitted observation is engine-owned and may finish after its request completes;
ordinary settlement does not cancel it and bias observations toward fast classifiers.
Its original absolute request deadline and per-check timeout still apply. Hosts call
`engine.close(timeout_seconds=...)` after serving drains and before stopping their
recorder. Shutdown stops observation admission, drains within that budget, and cancels
remaining work cooperatively. Each owned isolation loop closes its shared HTTP classifier
client on that loop before destroying it. Slow cleanup keeps its owning worker alive
without extending the host's shutdown budget; injected clients remain caller-owned.
A classifier that ignores cancellation retains its job
and byte reservation until it actually exits; timeout or shutdown cannot admit more
subjects against that occupied capacity. Work that completes within the drain budget
retains its actual verdict; cancellation is requested only when that budget expires.
Observation may still incur classifier cost, so it does not certify that admission had
no paid effects.

`_record_observation(policy, check, outcome, latency_seconds)` is a content-free host
recording seam separate from `_record`, which continues to report enforcing actions.
`GuardrailOutcome` distinguishes `allow`, `flagged`, `unsupported`, `uncertain`,
`unavailable`, `timeout`, and `skipped`. `check.action` reports the authored action
that would apply under enforcement. Policy IDs and revisions identify rollout state;
subject text, replacements, and detector diagnostics never enter either recorder.
An outcome reports the configured adapter's verdict for its declared input projection,
not an independent certification that every in-memory field was inspected. Complete-context
policies must bind an adapter with explicit coverage checks through the shared engine.
Hosts must count observation outcomes separately from actual customer blocks.
An outcome with `check=None` records incomplete observation admission or coverage,
not a classifier verdict. Keep these coverage counts separate from per-check verdict
denominators; they do not count every distinct missed subject.
Recording runs on one dedicated daemon worker. Only immutable policy/check metadata,
outcome and elapsed time enter its bounded queue; request text and callback closures
never enter that queue. `max_observation_records` defaults to 128 queued plus active
records. A saturated or closed sink drops metadata without delaying admission;
`engine.observation_recording_dropped` counts those losses and
`engine.observation_recording_failed` counts sink exceptions. Hosts should expose both
counters. The native `/metrics.json` snapshot exposes both values in `control_plane`;
`/metrics` exports the unlabeled counters
`exp_gateway_observation_recording_dropped_total` and
`exp_gateway_observation_recording_failed_total`.
The inspection owner and metadata recorder share the same overall `close` budget.
At shutdown expiry queued metadata is discarded and counted, while at most one stalled
sink call can remain alive holding content-free metadata. Cancellation callbacks may
arrive after the recorder closes: their attempted `skipped` record then increments the
dropped counter instead of delivering an outcome. Shutdown therefore does not promise
delivery of every `skipped` outcome or extend its deadline to wait for a late callback.
These loss counters are process-local, like the other gateway counters.

The local CLI closes its engine after native serving ends, including readiness-only
checks and startup failures, with the configured graceful timeout for observation cleanup.
The owned Python gateway shares its shutdown deadline between joining the native server
and draining its engine, then releases the ledger and virtual key.

Unsupported surfaces remain available to observation-only callers and record
`unsupported`. Hosts with additional surfaces, such as provider batches, use
`engine.record_unsupported_observations(policies)` for that coverage accounting and
only refuse the surface when an applicable policy has `mode="enforce"`.

## Input inspection alongside generation

`input_execution="parallel"` allows input classification to overlap generation.
Overlap requires every applicable input policy to select parallel execution and
no input check to modify the request. Any privacy rewrite keeps the combined input
chain before dispatch. The execution mode never adds an output check: the example
above only classifies the prompt.

Host-managed generation on direct aliases may start while approval is pending.
Native Chat, Responses, and Messages hold all response content until approval,
including reasoning and client function calls. Replay publication and continuation
retention also wait. Approval releases the pending prefix and resumes normal
streaming, without waiting for full completion. Function declarations supplied by
clients do not execute remote tools.

A block, classifier error, or timeout cancels the provider request and discards its
response. The trusted `input_guardrail_denied` settlement detail requires zero
customer charge while retaining available provider usage. Missing final usage uses
the existing estimated or unknown-cost accounting contract. The platform bears
provider work that already occurred; cancellation cannot undo it.

Provider-native or provider-server tools, gateway web or tool search, explicit cache
operations, service tiers, and routes with customer-managed credentials require
approval before dispatch. These are execution constraints of the same session.
Customers cannot choose the operator policy's mode or forge its settlement detail.

Project aliases complete input inspection before learned routing. Selection may
send the initial user message to a separately billed embedding provider, so it
cannot overlap pending approval. Responses continuations retain their selection
episode and inspect expanded history before routing.

When provider first-token latency exceeds classifier latency, overlap can hide most
of the inspection delay. Otherwise the first response content waits for the check.
Measure this with the actual detector and workload. Local scripted classifiers
validate gateway ordering and overhead, not GPU inference latency or model quality.

## Output policies

Configured output checks use the same session and engine. Blocking and model-backed
output checks buffer a complete normalized completion. Pure deterministic redaction
may stream a proven-safe prefix when the request shape permits it. There is no
incremental model-output classifier API. Input-only policies do not buffer complete
responses or invoke output classifiers.

The shared completion projection includes assistant text, refusal text, readable
reasoning, retrieved content, and complete tool arguments. Checks inspect original
output and any rewrite before release. A text rewrite suppresses alternate output
channels, and replacing a typed refusal converts its refusal terminal to completion.
Other provider failures remain failures. Tool arguments are inspected but cannot be rewritten. Generated
gateway-owned tool calls are checked before execution. Gateway-generated metadata
and the provider answer are inspected together as one completion, with one combined
response-size limit. Output policies buffer requests with gateway web or tool search,
including deterministic redactors. Terminal responses are checked while the same
accounting entry still owns the session. A rewrite that would change or remove generated
metadata fails closed because protocol encoders cannot apply it safely.

## Failure, replay, and integration

A confirmed violation returns a sanitized guardrail failure. Protected classifier
outages and timeouts return `unavailable`, without exposing detector diagnostics or
claiming that content violated policy. Queues and inflight work remain bounded;
request removal cancels its pending inspection. Unprotected identity checks may
observe and continue after uncertain classifier results. Platform policies are
always protected. An adapter requiring complete-context coverage raises
`ClassifierCoverageError()` when its declared inspection contract cannot cover the context
or modality; enforcement returns a nonretryable
`unsupported_capability` error, with no failover and no content-violation accusation.
The input subject bound has the same coverage semantics. `ClassifierUncertainError()`
distinguishes an inconclusive result from transport or runtime failure in observation;
protected enforcement continues to return `unavailable` for either case.

Replay keys include the applicable enforcing policies, execution settings, and revision.
Adding or revising observation-only policies leaves replay unchanged. Promoting an
observer to enforcement changes replay identity. Only deterministic detector rules
referenced by applicable enforcing checks enter that digest.

Version 0.7.166 intentionally establishes a new guarded replay contract. Guarded keys
accepted on 0.7.161 return `409` after upgrading, even when the authored policy is
otherwise unchanged. There is no legacy-fingerprint translation. Finish in-flight
requests before the cutover and use new keys for new operations.
A keyed admission compares its frozen policy revision with the replay claim before
inspection, acceptance, search, or model dispatch. A reload between claim and admission
returns `409`; retrying takes a new snapshot. Change `revision` whenever the detector
or external rollout configuration changes.
Policies are immutable; compose a new engine snapshot when changing adapters.
Default inspection bounds are 1 MiB, and the same absolute request deadline applies
throughout inspection and provider execution.

Startup requires `GUARDRAIL_CONTRACT_VERSION=3`. Publish coordinated engine and
native packages before a downstream host updates its exact release pins. There
are no synchronous adapter wrappers, alternate mandatory-policy entry points, or
package-version fallbacks.

A host ledger must implement `finish_request(..., web_search_requests=...)` for
completed gateway searches when admission fails before any model attempt, including
guardrail, routing, capability, dispatch-construction, and cache-binding failures.
Persist that meter atomically with the request failure, with zero customer charge;
it is provider expense evidence, not a model dispatch. Never certify such work as
having no paid effects. The local ledger stores the meter on `gateway_requests`.
Accounting retains failed terminal writes for its existing retry sweep and refuses
new paid admission work on every native endpoint until the retained request writes commit.
Hosted ledgers must
adopt this contract before enabling the coordinated release.

This integration covers native Chat Completions, Responses (including WebSocket
admission), and Messages. Native Decisions rejects identities with applicable policies
because its decision payload does not implement the inspection contract.
Other unsupported surfaces and modalities require explicit
host fencing before enforcement is enabled. Encrypted reasoning and signatures
remain opaque. This machinery supplies neither a hosted detector nor evidence of
model quality, GPU latency, or production rollout.

## Identity policy configuration

An optional identity policy uses the same schema with both `organization_id` and
`identity_id` set. An unassigned identity still receives every platform policy.
Only a request with no applicable policies takes the unguarded path, with no
classifier call, buffering, or guardrail callback. Local file configuration and
hosted policy stores must implement the same scoped policy-store contract.

## Classifier adapters

Python owns policy lookup and replaceable adapters. Adapters are reached only
through an injected in-process client. That client cannot recurse through
`POST /v1/chat/completions` or `POST /v1/responses`. A hosted detector must
use its own transport, not the public gateway.

Capability kinds name the inspection job, not a vendor:

- `pii`
- `secret_leakage`
- `prompt_injection`
- `content_safety`

`prompt_injection` is input-only.

Built-in adapter kinds:

- `regex`: local RE2 matching and deterministic text replacement. Configure
  custom expressions or the `email`, `credit_card`, and `api_key` built-in
  families. Card candidates require a valid Luhn checksum. Space- or hyphen-separated
  digit groups are inspected for adjacent cards; overlapping valid spans are
  redacted together. A valid card followed by another numeric group is still
  redacted, while uninterrupted numbers longer than 19 digits are not split. These rules find
  specific patterns, not all personal information or every secret format.
- `keyword`: coarse, test-oriented needle matching. Case-folded substrings of
  message text, completion text, or tool-call arguments. This is not a
  production prompt-injection or content-safety classifier.
- `http_json`: production async adapter that POSTs a strict inspect contract to
  a dedicated classifier endpoint. It never calls `POST /v1/chat/completions`
  or `POST /v1/responses`. Operators bind replaceable hosted adapters,
  including a hosted PII redactor, by `adapter_id`.

Other adapters may still be injected in code when composing a `GuardrailEngine`.

### Regex rules

For example, redact emails, card candidates, and an internal account format
before provider dispatch and before delivering the completion:

```json
{
  "adapters": [{
    "adapter_id": "sensitive-patterns",
    "kind": "regex",
    "builtin_patterns": ["email", "credit_card"],
    "patterns": ["ACCT-[0-9]{8}"],
    "replacement": "[REDACTED]"
  }],
  "policies": [{
    "policy_id": "redact-patterns",
    "organization_id": "organization-one",
    "identity_id": "identity-one",
    "protected": true,
    "checks": [
      {"check_id": "redact-input", "capability": "pii", "stage": "input",
       "action": "modify", "adapter_id": "sensitive-patterns", "timeout_ms": 250},
      {"check_id": "redact-output", "capability": "pii", "stage": "output",
       "action": "modify", "adapter_id": "sensitive-patterns", "timeout_ms": 250}
    ]
  }]
}
```

`block` refuses a match instead of replacing it. Replacements are literal
strings, not capture-group substitutions. Overlapping spans are combined
before replacement. Zero-length matches do not flag text. Tool arguments are
inspected but never rewritten: a flagged input tool argument under `modify`
is refused, and output modifications with tool calls are blocked.

RE2 does not support backreferences or look-around. Each custom pattern is
limited to 1,024 UTF-8 bytes, with at most 32 patterns per adapter and 256 KiB
of compile memory per expression. Each inspected text is limited to 1 MiB
and 4,096 matches or card candidates across its patterns. Exceeding these bounds is classifier
uncertainty, governed by the policy's fail-closed setting. Use a model-backed
PII detector for contextual entities such as names and addresses.

Each check has an action (`allow`, `modify`, `block`, `error`), a per-check
timeout, and an adapter identity. `modify` may rewrite request messages or
completion text. Tool-call arguments are never rewritten; a `modify` action on
a completion that contains tool calls becomes `block`.

Protected identities fail closed on adapter timeout, missing adapter, oversized
payload, or any other classifier uncertainty. Non-protected identities skip a
failed check and continue the remaining chain.

## Latency

With `input_execution="before_dispatch"`, input inspection follows continuation
expansion and delays dispatch. Parallel input inspection uses the overlap rules above. Output enforcement for a protected identity delays the
first visible byte until the winning completion is buffered and the output
chain returns.

The standard pack is seven ordered checks: four on input and three on output.
Each check has its own timeout. The conservative default is 250 ms per check,
not 250 ms for the whole chain. Worst-case classifier waiting is the sum of
the configured check timeouts, then capped by the remaining request deadline.
Shared keep-alive on `http_json` removes connection setup from later inspects.
It does not remove classifier inference latency.

Production adapters are async. Each inspect runs on a bounded isolation
worker with its own event loop, so a classifier that blocks before its first
await cannot freeze the caller's timeout. The caller waits on a cross-thread
future and returns at the tighter of the check timeout and the remaining
request deadline. Isolation workers stay occupied until that invocation
actually exits. An abandoned adapter is quarantined until every abandoned
inspect for it finishes. Further calls to that adapter fail immediately and
do not start another worker. Other adapters keep any remaining isolation
workers. `http_json` still reuses one keep-alive client per isolation loop.
Native callbacks submit enforcement onto one shared daemon loop so a Rust
worker can return while an abandoned inspect still occupies an isolation
worker. Adapters implement the async contract directly. Request bounds count the
complete in-memory classifier subject, including provider fields excluded from wire
serialization. Response bounds count the serialized completion, including context
and tool-call arguments.

## Privacy

Logs and durable state record only content-free decision metadata: policy,
organization, identity, check, capability, action, and latency. Raw prompts,
completions, detector payloads, and replacements are not logged or persisted.
Replacements exist only in memory for the remainder of the request.

## Configuration

Place an optional file at `ROOT/gateway/guardrails.json`. Missing files leave
every organization and identity unguarded. A valid file whose `policies` list
is empty is also unguarded: the loader returns no engine. There is no global
switch that turns classifiers on for every identity.

## Interactive setup

Setup Gateway shows `Guardrails: Off` on a first run. Pressing Enter accepts
that default and does not create `gateway/guardrails.json`. The selected
identity stays on the unguarded hot path.

Choosing `edit` can opt the selected identity into the standard pack. The
prompts stay short: enable or leave off, then a dedicated classifier URL, then
an optional bearer credential environment-variable name. Setup never asks for
or stores the credential value. An enabled choice authors the documented
standard preset for the local organization and that identity, with
`protected` true and the 250 ms default timeout. One `http_json` adapter is
bound to all four capabilities because the outbound inspect contract includes
`capability`.

Reconfiguration keeps the current file unless the operator changes it. A
hand-authored policy that is not the setup-owned standard pack is shown as
`Custom/preserved`. Setup will not replace it unless the operator types
`replace`. Other identities, policies, and adapters stay unchanged.

Hand-author `ROOT/gateway/guardrails.json` when you need extra adapters,
timeouts, identities, or checks that setup does not own. That file is the
advanced escape hatch. Missing files, and valid files with no policies, remain
the unguarded path. Setup-owned adapter and policy IDs are deterministic
`stable_id` values over organization and identity. They do not concatenate
those fragments, so hyphenated pairs cannot collide.

If the configured path is a symlink, setup writes through it and never
removes the link. Disabling the last setup-owned pack writes an empty valid
document through that symlink so the link and target stay usable. A regular
file created solely by setup is deleted when it becomes empty.

## Standard preset

The `standard` preset is opt-in per organization and identity. It is never
implied. The operator must name `preset: standard`, set an explicit
`protected` boolean (`true` fail-closed or `false` fail-open), and bind an
`adapter_id` for every capability. Load fails closed on a malformed preset, an
empty or null `preset`, a missing `protected` field, a missing capability
binding, a duplicate check ID, or a `preset` key combined with a `checks` key
(including an empty check list). `capability_adapters` is rejected whenever
`preset` is absent, including an empty map.

Expanded standard order, with a conservative 250 ms default timeout per check:

1. input `pii`: modify (redact)
2. input `secret_leakage`: modify (redact)
3. input `prompt_injection`: block
4. input `content_safety`: block
5. output `pii`: modify (redact)
6. output `secret_leakage`: modify (redact)
7. output `content_safety`: block

Timeouts stay configurable. `timeout_ms` sets the default. `timeouts` overrides
individual checks by check ID (`standard-input-pii`) or `stage.capability`
(`input.pii`).

```json
{
  "adapters": [
    {
      "adapter_id": "hosted-pii",
      "kind": "http_json",
      "url": "https://classifier.example.invalid/v1/inspect",
      "bearer_env": "CLASSIFIER_BEARER"
    },
    {
      "adapter_id": "hosted-secrets",
      "kind": "http_json",
      "url": "https://classifier.example.invalid/v1/inspect"
    },
    {
      "adapter_id": "hosted-injection",
      "kind": "http_json",
      "url": "https://classifier.example.invalid/v1/inspect"
    },
    {
      "adapter_id": "hosted-safety",
      "kind": "http_json",
      "url": "https://classifier.example.invalid/v1/inspect"
    }
  ],
  "policies": [
    {
      "policy_id": "standard-member",
      "organization_id": "organization-one",
      "identity_id": "identity-one",
      "protected": true,
      "preset": "standard",
      "revision": "detector-rollout-v1",
      "timeout_ms": 250,
      "capability_adapters": {
        "pii": "hosted-pii",
        "secret_leakage": "hosted-secrets",
        "prompt_injection": "hosted-injection",
        "content_safety": "hosted-safety"
      }
    }
  ]
}
```

`bearer_env` is the name of a process environment variable. The document never
stores a credential. The adapter reads that name at inspect time and sends a
bearer header. Missing values are classifier uncertainty.

Local keyword and regex input checks inspect canonical message text, including the
generated tool-error prefix, tool arguments, function definitions and examples,
tool call/result identifiers, names and namespaces, structured
output schemas, and normalized gateway web/tool-search declarations. Recovered or sealed reasoning, capture-only reasoning,
multimodal attachments, verbatim native message blocks, context management, raw provider
thinking/output configuration, nonempty Messages safeguards, opaque provider tool-caller objects, and provider-owned
native/server tools return unsupported coverage. Regex rewrites only message text; matches in tool arguments or immutable
tool/schema/search context are refused under modify. Exact flattened text blocks and inert
cache/replay metadata remain supported for inspection. Cached block text is inspectable,
but a match is refused under local modify because changing only canonical text cannot
safely rewrite the provider block. Tool-error text is also immutable under modify:
rewriting the body cannot remove the typed error flag that generates its prefix.

The `http_json` request includes `capability`, `stage`, `action`, `check_id`,
and exactly one subject: `request` or `completion`. The request subject is the
compact deterministic JSON of the public `GatewayRequest` projection. Fields marked
`exclude=True`, including private reasoning, native provider blocks and configuration,
and tool examples, are intentionally absent. A hosted modifier may rewrite the visible
subject while the gateway restores unchanged hidden replay authority; that restoration
does not inspect the hidden content. An `allow` verdict covers the transmitted projection
only. Generic `http_json` cannot implement mandatory complete-context criminal inspection
when private carriers matter. Bind a coverage-aware adapter through the same engine for
that requirement. This release does not expand the HTTP wire contract.

The response
must validate as `ClassifierVerdict`. A non-`modify` check must not return a
replacement. A flagged `modify` verdict must contain exactly the
stage-appropriate replacement (`replacement_messages` on input,
`replacement_text` on output). Unflagged verdicts cannot include replacements.
Both replacement types together, wrong-stage replacements, non-2xx statuses,
malformed JSON, oversized bodies, and other contract drift become classifier
uncertainty. Redirects are disabled. The outer `GuardrailEngine` still owns
per-check and request deadlines.

Hand-authored checks remain available when a preset is not used:

```json
{
  "adapters": [
    {
      "adapter_id": "keyword-safety",
      "kind": "keyword",
      "needles": ["example-disallowed-phrase"]
    }
  ],
  "policies": [
    {
      "policy_id": "strict-member",
      "organization_id": "organization-one",
      "identity_id": "identity-one",
      "protected": true,
      "max_request_bytes": 1048576,
      "max_response_bytes": 1048576,
      "checks": [
        {
          "check_id": "input-safety",
          "capability": "content_safety",
          "stage": "input",
          "action": "block",
          "timeout_ms": 250,
          "adapter_id": "keyword-safety"
        }
      ]
    }
  ]
}
```

Interactive setup can author the standard pack for one identity. Bind
additional adapters in process when composing a `GuardrailEngine`, or
hand-author `guardrails.json` for identities and checks that setup does not
own.

## Inappropriate-content use

Content-safety checks can block or redact disallowed text in prompts and
completions. They are not a substitute for provider-side safety systems, legal
review, or human moderation. Keyword adapters are coarse and test-oriented. A
`modify` action needs the adapter to supply a replacement; otherwise the check
errors. Protected streaming may add classifier latency before the first token.

## Limitations

- Guardrails do not change routing policy, budgets, or catalog snapshots.
- Classifiers must not call the public gateway. Recursion fails closed.
- At most 32 isolation workers may run async classifier inspects for one
  limiter. Additional inspects wait only until their remaining timeout, then
  fail closed without starting another worker. A worker occupied by an
  abandoned inspect is retained until that invocation exits. An adapter that
  was abandoned stays quarantined until every abandoned inspect for it
  finishes.
- Oversized tool arguments count against the response byte bound and are
  blocked, not rewritten.
