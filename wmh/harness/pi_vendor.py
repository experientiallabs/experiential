"""The vendored pi agent, and the seam that turns it into a searchable harness.

`wmh/harness/vendor/pi-agent/` is a byte-exact copy of `packages/agent` from
earendil-works/pi at v0.80.3 (commit a23abe4a695df8b69b613f73e9fdda2a8af894d4). The pin, the
license attribution, and the integrity ledger live beside it: `vendor/pi-agent/VENDOR.md`,
`vendor/pi-agent/LICENSE`, and `vendor/manifest.sha256` (regenerate/verify with
`wmh/harness/vendor/vendor_pi.sh`).

This module is the ONLY place wmh reads that tree, and it reads it straight from disk:
`pi_agent_code_surfaces()` loads pi's own TypeScript source into `code:` surfaces so the
meta-agent searches over the real agent's source, and `PiRuntime` materializes those surfaces to
run pi headless. Nothing here fetches pi over the network or from a scratch checkout — the
committed vendored copy is the sole source of truth. The whole 56-file package is vendored on disk
(byte-checked against upstream); the 25 runnable `src/**/*.ts` files become the searchable
surfaces (fixtures, docs, and the package's own vitest specs are vendored but not surfaced).
"""

from __future__ import annotations

from pathlib import Path

# code_surface_id lives with the Surface grammar in doc.py; imported (and re-exported) here for
# the existing pi-vendor call sites.
from wmh.harness.doc import Surface, SurfaceKind, code_surface_id

# The committed, byte-exact vendored copy (see VENDOR.md for the upstream pin).
PI_AGENT_ROOT = Path(__file__).parent / "vendor" / "pi-agent"
# pi's runnable TypeScript source — the harness the meta-agent searches over.
_SOURCE_GLOB = "src/**/*.ts"


def pi_agent_source_paths() -> list[Path]:
    """Every runnable pi source file under the vendored tree, sorted, excluding vitest specs."""
    return sorted(
        p
        for p in PI_AGENT_ROOT.glob(_SOURCE_GLOB)
        if p.is_file() and not p.name.endswith(".test.ts")
    )


def pi_agent_code_surfaces() -> list[Surface]:
    """pi's vendored source as `code:` surfaces, each carrying its path under the package root.

    Read straight from `PI_AGENT_ROOT` (the committed copy) — never from a network fetch or a
    scratch checkout. Raises if the vendored tree is missing so a broken vendoring fails loudly
    instead of silently running an empty harness.
    """
    paths = pi_agent_source_paths()
    if not paths:
        raise FileNotFoundError(
            f"no pi source under {PI_AGENT_ROOT}; is the vendored copy present? "
            "regenerate with wmh/harness/vendor/vendor_pi.sh"
        )
    surfaces: list[Surface] = []
    for p in paths:
        rel = p.relative_to(PI_AGENT_ROOT).as_posix()
        surfaces.append(
            Surface(
                id=code_surface_id(rel),
                kind=SurfaceKind.CODE,
                path=rel,
                content=p.read_text(encoding="utf-8"),
            )
        )
    return surfaces
