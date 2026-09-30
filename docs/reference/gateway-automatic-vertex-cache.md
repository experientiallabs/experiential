# Automatic Vertex prefix caching

`NativeControlPlane(components, automatic_cache=host)` opts a serving operator
into automatic explicit caching. It is disabled by default and mutually exclusive
with `explicit_cache`. Platform is the first scheduled consumer: its host owns
key-scoped policies, operator funding and cross-worker claims in Postgres.

Google's implicit cache is not guaranteed. A native request with shared text and
a different suffix can miss even when an identical repeat hits. This integration
does not change implicit-cache accounting or claim that every request is eligible.

The planner accepts plain text requests with leading system instructions followed
by user messages. It considers up to eight whole-message prefixes, leaves the last
user message untouched, and caps resource bodies at 256 KiB. Explicit caller
checkpoints, tools, assistant history, media and native carriers are left alone.
Suffixes and generation controls, including response schemas, stay on generation;
they do not change the cached prefix's identity. Partial-message matches are not
selected. Small or one-off prompts continue normally without a cache create.

The host receives only bounded prefix hashes and sizes. It selects a repeated
candidate under the exact tenant/key/deployment/credential/endpoint scope. Only
after the selected attempt has started may it durably reserve money and return
one creator. The native transport measures that prefix with Vertex `countTokens`;
a count failure or a result below the configured minimum records `not_created`
and retains the original generation. This proves no create HTTP was sent. It is
distinct from an ambiguous create timeout, which records `unknown` and retains
the reservation. The engine neither steals nor retries an uncertain create.

Creation uses the admitted endpoint, authenticated headers and an absolute expiry
no more than 300 seconds after reservation. US and EU jurisdictional endpoints
must match their exact location. A verified project-ID-to-number mapping binds
returned resource names; a prefix in another project or location is refused.
The host acknowledges accounting before the continuation can use the resource.
Expired permissions fall back to the original generation. A host accounting
failure stops the attempt rather than granting unrecorded cache authority.

The host must revoke reuse as well as creation when policy, credentials, plan or
ZDR requirements change. A cache remains provider-held data until its short TTL
expires; do not opt ZDR requests into it. Prompt text stays in request memory and
the admitted provider region. Never publish customer captures as test fixtures.

The native wheel exports `AUTOMATIC_VERTEX_CACHE_CONTRACT_VERSION = 1`. Consumers
must require this and the Python constructor contract together before enabling
the feature. Test with fictional prefixes, changed suffixes and changed schemas,
plus concurrent claims, below-minimum counts, expiry and ambiguous provider I/O.
