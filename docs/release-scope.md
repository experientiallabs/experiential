# Release scope

This release supports the current source and one wheel with either core dependencies or the
optional `sft` dependency extra on their documented local paths. It claims only behavior exercised
on the exact release checkout.

## Supported and verified

- Root CLI commands are exactly `build`, `config`, `optimize`, and `run`; optimizer commands are
  exactly `router` and `model`.
- Public Python exposes provider-free build, explicit router composition, frozen router load and
  HTTP application, structural text-versus-sandbox comparison, and managed SFT composition.
- W16 router evidence uses 100 normalized traces, 50 fit tasks, 20 held-out tasks, 140 planned
  cells, 130 deterministic text simulations, and 140 deterministic judgments under one finite
  simulation and judgment budget. Observed hosted-service spend is exactly $0.00.
- W16 sandbox evidence compares two exact post-lock text and Darwin local-process pairs. It retains
  one malformed sandbox failure in the denominator and claims structural terminal agreement only.
- Exact-checkout CI supplies the full 40-hex Git revision, recursively verifies every evidence
  artifact and manifest input, and publishes machine-readable JSON plus JUnit evidence.

## Explicitly excluded

- No paid E2B or Harbor cloud smoke ran. The repository verifies the optional `bounded-close-v1`
  Harbor lifecycle and ledger with injected fakes, but makes no cloud cleanup, provider-quality, or
  environment-parity claim.
- No real Tinker training ran. Managed SFT remains fail-closed unless its immutable configuration
  has a finite positive ceiling and its backend supplies a conservative full-schedule estimate.
- No trained-versus-base behavioral comparison ran because this release produced no paid training
  artifact. It makes no trained-model quality-improvement claim.
- No hosted model, judge, embedding, telemetry, environment, credential, or `.env` path was used by
  release evidence. The deterministic W16 evidence reports exactly $0.00 observed service spend.

These exclusions are product boundaries, not evidence that the corresponding hosted services are
unsafe or unsupported forever. Any future claim requires separately authorized, finite-budget,
denominator-preserving evidence.
