# Gateway guardrails

Identity-scoped guardrails inspect a request after authentication and, when
configured, inspect the winning completion before any caller byte is delivered.
They are default-off. Lookup is by authenticated `organization_id` plus
`identity_id`. A pair with no assigned policy keeps the existing gateway hot
path: no classifier call, no stream buffering, and no extra native callback.

## Data flow

1. Authenticate the virtual key and expand an optional Responses continuation.
2. Look up at most one immutable policy by organization and identity. Missing
   policies stop here.
3. Run the input chain once. The validated or transformed canonical request is
   reused for route resolution, provider preflight, acceptance, and every
   waterfall attempt.
4. Dispatch the provider waterfall. Provider failures remain failover-eligible.
   A guardrail block, error, or protected-identity fail-closed outcome is
   terminal and does not advance the waterfall.
5. Buffer the winning normalized completion when the policy has output checks.
   Run the output chain once, including tool-call arguments. Then encode,
   remember a Responses continuation, and publish replay.
6. Return a sanitized OpenAI-shaped error on failure. Partial blocked output is
   never exposed.

The native data plane follows the same order. Input enforcement runs after
Responses continuation expansion and before route resolution, provider
preflight, ledger acceptance, and attempt start. A block never reaches routing.
When admission sets `output_guardrail`, Rust buffers the completion and calls
`enforce_output` once. Unguarded admissions omit that flag and never invoke
the callback.

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

- `keyword`: coarse, test-oriented needle matching. Case-folded substrings of
  message text, completion text, or tool-call arguments. This is not a
  production prompt-injection or content-safety classifier.
- `http_json`: production async adapter that POSTs a strict inspect contract to
  a dedicated classifier endpoint. It never calls `POST /v1/chat/completions`
  or `POST /v1/responses`. Operators bind replaceable hosted adapters,
  including a hosted PII redactor, by `adapter_id`.

Other adapters may still be injected in code when composing a `GuardrailEngine`.

Each check has an action (`allow`, `modify`, `block`, `error`), a per-check
timeout, and an adapter identity. `modify` may rewrite request messages or
completion text. Tool-call arguments are never rewritten; a `modify` action on
a completion that contains tool calls becomes `block`.

Protected identities fail closed on adapter timeout, missing adapter, oversized
payload, or any other classifier uncertainty. Non-protected identities skip a
failed check and continue the remaining chain.

## Latency

Input enforcement is on the request critical path after continuation expansion
and before dispatch. Output enforcement for a protected identity delays the
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
worker. Leftover synchronous test adapters, when still needed, run only
through a private bounded compatibility wrapper. Exhaustion of that wrapper
cannot take async capacity from healthy adapters. Request bounds count the
compact JSON request subject sent to classifiers, including tool definitions,
structured schemas, and metadata. Response bounds count completion text and
tool-call arguments.

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

The `http_json` request includes `capability`, `stage`, `action`, `check_id`,
and exactly one subject: `request` or `completion`. The request subject is the
compact deterministic JSON of the canonical `GatewayRequest`. The response
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
