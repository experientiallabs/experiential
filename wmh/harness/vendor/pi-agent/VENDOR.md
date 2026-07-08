# Vendored: pi agent (`@earendil-works/pi-agent-core`)

This directory is a **byte-exact copy** of the `packages/agent` subtree of the pi agent, vendored
into this repo so the harness search runs the real agent's source rather than a re-implementation.

| | |
|---|---|
| Upstream | https://github.com/earendil-works/pi |
| Package | `@earendil-works/pi-agent-core` |
| Tag | `v0.80.3` |
| Commit (the pin) | `a23abe4a695df8b69b613f73e9fdda2a8af894d4` |
| Vendored subtree | `packages/agent/` → this directory (`wmh/harness/vendor/pi-agent/`) |
| Files | 56 (byte-identical to upstream `packages/agent`) |
| Upstream license | MIT (© 2025 Mario Zechner) |

The tag is a convenience label; **the commit SHA is the pin**. `wmh/harness/vendor/vendor_pi.sh` fetches upstream
at that SHA, re-materializes this tree, and regenerates `wmh/harness/vendor/manifest.sha256`
(the per-file integrity ledger). Re-running it must produce zero diff against the committed copy —
that is how anyone re-verifies this vendoring from scratch.

## License

The `packages/agent` package carries no license file of its own upstream; pi is MIT-licensed at the
repository root, so `LICENSE` here is a verbatim copy of the upstream **root** `LICENSE` at the
pinned commit. See it for the full MIT text and copyright.

## Do not edit in place

These files are upstream bytes and must stay that way — `wmh/harness/vendor/vendor_pi.sh` and the checksum gate
both assume byte-identity with upstream. The harness *searches over* this source by loading it into
`code:` surfaces (`wmh/harness/pi_vendor.py`) and mutating those surfaces through audited
`HarnessDelta`s; the mutations live in stored `HarnessDoc` versions, never as edits to this tree.

## What consumes it

`wmh/harness/pi_vendor.py` is the only reader: `pi_agent_code_surfaces()` loads the 25 runnable
`src/**/*.ts` files (vitest `*.test.ts` specs excluded) into `code:` surfaces, which
`wmh/harness/pi_runtime.py` materializes to run pi headless against the world model. The full
package is vendored on disk; only `src/` is surfaced.
