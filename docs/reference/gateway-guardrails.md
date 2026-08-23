# Gateway guardrails

Identity-scoped guardrails inspect a request after authentication and, when
configured, inspect the winning completion before any caller byte is delivered.
They are default-off. An identity with no assigned policy keeps the existing
gateway hot path: no classifier call, no stream buffering, and no extra native
callback.

## Data flow

1. Authenticate the virtual key and expand an optional Responses continuation.
2. Look up at most one immutable policy by authenticated identity. Missing
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

The native data plane follows the same order. Input enforcement runs after the
engine-ownership decision (so escalated requests are not inspected twice) and
before ledger acceptance. When admission sets `output_guardrail`, Rust buffers
the completion and calls `enforce_output` once. Unguarded admissions omit that
flag and never invoke the callback.

## Classifier adapters

Python owns policy lookup and replaceable adapters. Adapters are reached only
through an injected in-process client. That client cannot recurse through
`POST /v1/chat/completions` or `POST /v1/responses`. A hosted detector must
use its own transport, not the public gateway.

Capability kinds name the inspection job, not a vendor:

- `pii`
- `prompt_injection`
- `content_safety`

The only built-in adapter kind is `keyword`: operator-authored needles compared
as case-folded substrings. Other adapters are injected in code.

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
chain returns. Per-check timeouts are bounded by the remaining request
deadline. Request and response content are bounded by `max_request_bytes` and
`max_response_bytes` on the policy.

## Privacy

Logs and durable state record only content-free decision metadata: policy,
identity, check, capability, action, and latency. Raw prompts, completions,
detector payloads, and replacements are not logged or persisted. Replacements
exist only in memory for the remainder of the request.

## Configuration

Place an optional file at `ROOT/gateway/guardrails.json`. Missing files leave
every identity unguarded.

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

There is no management UI. Bind additional adapters in process when composing
a `GuardrailEngine`.

## Inappropriate-content use

Content-safety checks can block or redact disallowed text in prompts and
completions. They are not a substitute for provider-side safety systems, legal
review, or human moderation. Keyword adapters are coarse. A `modify` action
needs the adapter to supply a replacement; otherwise the check errors.
Protected streaming may add classifier latency before the first token.

## Limitations

- Guardrails do not change routing policy, budgets, or catalog snapshots.
- Native multi-deployment pools still escalate to the python engine, which
  owns waterfall accounting and runs input enforcement once.
- Classifiers must not call the public gateway. Recursion fails closed.
- Oversized tool arguments count against the response byte bound and are
  blocked, not rewritten.
