# Provider tariff evidence

This directory retains the exact public source entities used by built-in provider tariffs.
Artifacts are compressed with deterministic gzip headers. Each tariff records both the SHA-256
of the compressed package artifact and the SHA-256 of the decoded source entity. Catalog lookup
verifies both digests before returning a tariff.

The source URL, publication metadata, HTTP content encoding, semantic media type, and exact claim
bindings remain in the tariff provenance. Filenames are content addressed by the decoded entity
digest. These files contain public pricing and model documentation only; they must never contain
credentials, signed URLs, account identifiers, prompts, or benchmark task content.

`provider_cost_meter()` is the supported construction boundary for a paid provider meter. It reads
only retained package resources or artifact bytes explicitly supplied by its caller, verifies the
encoded and decoded digests, interprets the registered source coordinates, and returns a receipt
bound to the complete route, price ceiling, and provenance claim. It never fetches evidence from
the network and never accepts provider credentials. Direct construction of an evidence receipt is
not a substitute for this verification step.

Built-in Bedrock routes require the exact source set registered for that route. The verifier checks
the retained catalog publication metadata, model route, billing mode, region, SKU, usage dimension,
unit, effective date, and price. A known but unrelated catalog source is rejected.

The generic HTML binding check proves only that an exact bound value occurs in the digest-pinned
snapshot; it is not a general DOM-coordinate evaluator. Built-in Bedrock semantic sources also pass
a profile-specific structural check that requires one model row in the expected pricing table,
binds its input and output columns to the exact pricing-expression identifiers, and joins those
identifiers to exact JSON catalog meters. A new HTML evidence profile must add an equivalent
structural verifier and must not rely on value presence alone.

Azure routes are caller-supplied because their deployment identity is account-specific. Capture and
retain exactly three JSON responses: the Azure resource account, its deployment, and the bounded
Azure Retail Prices response. The ARM reads may use normal request-header authentication, but
credentials must not enter the retained URL, query, response, or receipt. Pass the evidence bytes
by source ID through `evidence_artifacts`. Use
`azure_provider_cost_meter_from_evidence()` to derive the route bindings, billing meters, exact
price floor, and verified meter directly from those responses. A caller may supply a higher
conservative price ceiling, but cannot lower the evidence-derived price floor.
The offline verifier joins the endpoint, account location, deployment name and ETag, immutable model
version, no-auto-upgrade setting, deployment SKU, retail service, region, meter and SKU IDs, token
dimensions, units, effective date, and prices. Missing, extra, paginated, non-USD, discounted, or
digest-mismatched evidence fails closed.

The receipt protects a trusted experiment builder from accidental drift and replay. It is not
provider-signed evidence and does not defend against hostile in-process Python code. Paid
orchestration must therefore expose the verified meter or its serialized frozen policy, not a
public path for callers to fabricate receipts.
