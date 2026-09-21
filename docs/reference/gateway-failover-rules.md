# Per-rung conditional failover (`failover_only_on`)

A deployment may carry
`GatewayDeploymentCapabilities.failover_only_on`, a set of failure tokens: the failover-eligible
failure classes by wire name (`throttled`, `timeout`, `transport`, `provider_internal`,
`provider_quota`, `provider_authentication`, `provider_not_found`, `unavailable`, `empty_completion`,
`malformed_response`, `guardrail`), `refusal` for any provider refusal, or `refusal:<reason>` for
one bounded category (`cyber_policy`, `cbrn`, `content_policy`, `recitation`, `data_inspection`,
`unspecified`; the closed list is `exp.common.models.failover_tokens.FAILOVER_TOKENS`, mirrored by
the data plane's `waterfall/fallback_rules.rs`). Such a rung is FAILOVER-ONLY: it is never chosen
for a request's first dial (nor reached by a sideways shed or a budget skip) and is claimed as a
successor only when the failure the ladder is walking from spells one of its tokens — and then even
for a refusal on an alias revision that did not enable refusal failover, because the rung's own
opt-in is the caller's remedy (a customer's OpenAI key enrolled in a trusted-access program taking
over exactly the requests the house rung refused under `refusal:cyber_policy`). A rung with no set is
unrestricted and behaves exactly as before; unrestricted rungs keep their order and the route policy
still governs advancing onto them. Refusal DELTAS carry no reason, so a rule rung downstream causes
them to be withheld only when its set accepts an unnamed refusal (`refusal` or `refusal:unspecified`).
The wire fact rides each route entry as `failover_only_on` (null when unrestricted); the control
plane chooses the depth (`native_fallback_rules.py`) and records the reason on the attempt as
`fallback_reason = failover_only_on:<token>` (the failure's precise token); the data plane decides
whether a successor exists before asking, and refuses a reservation that violates the rules (a first
dial or an unmatched successor on a rule rung) with an internal error rather than dialing it. A route
whose EVERY admitted rung carries a set fails closed at admission with a named internal error
(`FailoverRulesError`), never an exhausted ladder that dialed nothing.

## Authoring

Set `failover_only_on` on the deployment's `GatewayDeploymentCapabilities` (the platform projects it
from the lane's dispatch facts, beside `time_to_first_byte_base_seconds`). Tokens are validated
against the closed `FailoverToken` literal, so a misspelled or never-failover class (`invalid_request`,
`quota_exceeded`) is refused at the typed boundary. An empty set is a rung that is never dialed at all.

| Set | First dial | Successor to `refusal:cyber_policy` | Successor to `throttled` |
|---|---|---|---|
| (none) | yes | only with the alias revision's refusal failover | yes (policy) |
| `["refusal:cyber_policy"]` | never | yes, regardless of the revision's opt-in | never |
| `["refusal"]` | never | yes | never |
| `["throttled", "provider_internal"]` | never | never | yes |

## Ledger

An attempt reserved on a rule rung as a successor carries `fallback_reason =
failover_only_on:<token>`, where the token is the FAILURE's precise spelling (`refusal:cyber_policy`
even when the rung's set is the bare `refusal`). First dials, same-rung redials, and unrestricted
rungs keep the route's own `fallback_reason`.

## Failure modes

- Every admitted rung restricted: admission raises `FailoverRulesError`, answered as the gateway's
  internal error and logged with the exception type, before any reservation.
- The control plane reserves a rule rung for a first dial or for a failure its set does not name:
  the data plane settles the attempt `failed` / `internal` ("failover-only rung reserved outside
  its rules") and answers 500 without dialing the rung.
- A rung restricted to one refusal category never takes an unnamed refusal (`refusal:unspecified`),
  and refusal text streamed by the provider is withheld for a rule rung downstream only when that
  rung accepts unnamed refusals.

A rung at its per-worker in-flight bound is a different kind of bypass — a policy shed, not a
failure — and what happens when every rung is shed is described in
[gateway-lane-saturation.md](gateway-lane-saturation.md).
