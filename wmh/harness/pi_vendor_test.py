"""Proves the pi harness is seeded from the committed vendored copy — not a scratch dir or fetch.

These tests touch only the filesystem: if they pass, pi's source is present under
`wmh/harness/vendor/pi-agent/` and that is where the surfaces come from. No network, no ssh, no
node — a genuine vendoring is a static-tree property.
"""

from __future__ import annotations

from wmh.harness.doc import SurfaceKind
from wmh.harness.pi_vendor import (
    PI_AGENT_ROOT,
    pi_agent_code_surfaces,
    pi_agent_source_paths,
)


def test_vendored_copy_is_present_and_committed() -> None:
    assert PI_AGENT_ROOT.is_dir(), f"vendored pi missing at {PI_AGENT_ROOT}"
    # provenance artifacts ship with the source
    assert (PI_AGENT_ROOT / "VENDOR.md").is_file()
    assert (PI_AGENT_ROOT / "LICENSE").is_file()
    assert (PI_AGENT_ROOT / "package.json").is_file()
    # the path lives inside the repo, not /tmp or a home-dir checkout
    assert "vendor/pi-agent" in PI_AGENT_ROOT.as_posix()


def test_pin_recorded_in_vendor_md() -> None:
    text = (PI_AGENT_ROOT / "VENDOR.md").read_text(encoding="utf-8")
    assert "a23abe4a695df8b69b613f73e9fdda2a8af894d4" in text
    assert "v0.80.3" in text
    assert "earendil-works/pi" in text


def test_surfaces_come_from_the_vendored_tree() -> None:
    surfaces = pi_agent_code_surfaces()
    # the 25 runnable src/*.ts files (vitest specs excluded)
    assert len(surfaces) == 25
    for s in surfaces:
        assert s.kind is SurfaceKind.CODE
        assert s.path is not None and s.path.startswith("src/")
        # content is exactly the byte content of the file under the vendored root
        on_disk = (PI_AGENT_ROOT / s.path).read_text(encoding="utf-8")
        assert s.content == on_disk
    # entrypoint is present — proves we surfaced the real agent, not a stub
    assert any(s.path == "src/agent-loop.ts" for s in surfaces)
    # ids are unique and stable
    ids = [s.id for s in surfaces]
    assert len(set(ids)) == len(ids)
    assert "code:src-agent-loop-ts" in ids


def test_source_paths_resolve_under_the_vendored_root() -> None:
    for p in pi_agent_source_paths():
        # every source path is physically inside the committed vendored tree
        assert PI_AGENT_ROOT in p.parents
