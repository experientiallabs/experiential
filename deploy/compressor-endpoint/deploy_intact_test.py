"""Regression test for the deploy that silently rotated the pinned TLS certificate.

`deploy.sh` once ran `rsync --delete-excluded` against a filter that named four files and
excluded everything else, so every deploy DELETED the generated trees from the service root:
`tls/` (regenerating the self-signed certificate that every client pins as its entire trust
store, breaking them all until they re-pulled), `venv/` (multi-GB torch, rebuilt each time), and
`hf/` (the model weights).

The bug is one rsync flag, so the test drives the real rsync command out of `deploy.sh` against
a populated fake service root on the local filesystem and asserts those trees survive
byte-for-byte. No box, no ssh: the flag is the thing under test.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DEPLOY_SH = HERE / "deploy.sh"

# The trees bootstrap.sh generates ON the box. A deploy must never touch them.
GENERATED_TREES = ("tls", "venv", "hf")


def _rsync_command() -> list[str]:
    """The rsync invocation deploy.sh actually runs, read out of the script itself.

    Parsed rather than duplicated so the test cannot drift into asserting a command the script
    no longer runs.
    """
    script = DEPLOY_SH.read_text()
    match = re.search(r"^rsync (.*?)(?=\n\n|\n[a-z])", script, re.MULTILINE | re.DOTALL)
    if match is None:
        pytest.fail("could not find the rsync invocation in deploy.sh")
    joined = match.group(0).replace("\\\n", " ")
    return joined.split()


def _digest(root: Path) -> dict[str, str]:
    """Content hash of every file under `root`, keyed by relative path."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not available")
def test_deploy_rsync_leaves_generated_trees_intact(tmp_path: Path) -> None:
    """A deploy against a populated service root must not disturb tls/, venv/, or hf/."""
    destination = tmp_path / "service-root"
    for tree in GENERATED_TREES:
        (destination / tree).mkdir(parents=True)
    (destination / "tls" / "cert.pem").write_text("PINNED CERTIFICATE")
    (destination / "tls" / "key.pem").write_text("PRIVATE KEY")
    (destination / "venv" / "bin").mkdir()
    (destination / "venv" / "bin" / "python").write_text("#!/bin/sh")
    (destination / "hf" / "weights.bin").write_text("0" * 128)
    (destination / "server.py").write_text("stale")
    before = _digest(destination)

    command = _rsync_command()
    # The real command ends in `"${HOST}:${ROOT}/"`; retarget it at a local directory so the
    # test exercises the flags without needing the box.
    command = [part for part in command if not part.startswith('"${HOST}')]
    resolved = [
        part.replace('"${HERE}/', str(HERE) + "/").rstrip('"') if "${HERE}" in part else part
        for part in command
    ]
    subprocess.run([*resolved, f"{destination}/"], check=True, capture_output=True)

    after = _digest(destination)
    for tree in GENERATED_TREES:
        kept = {path: digest for path, digest in after.items() if path.startswith(f"{tree}/")}
        original = {path: digest for path, digest in before.items() if path.startswith(f"{tree}/")}
        assert kept == original, f"deploy modified or deleted {tree}/ on the box"
    # And it did do its actual job.
    assert after["server.py"] != before["server.py"]


def test_deploy_never_uses_a_delete_flag() -> None:
    """The flag that caused the incident stays out, whatever else the command grows.

    Belt and braces next to the behavioral test above: a future edit that reintroduces
    `--delete` or `--delete-excluded` deletes the certificate again, and the behavioral test
    only catches it if the trees happen to be modelled here.
    """
    command = " ".join(_rsync_command())
    assert "--delete" not in command, (
        "deploy.sh must not pass any --delete flag: the destination holds tls/, venv/, and hf/, "
        "which are generated on the box. Deleting tls/ regenerates the self-signed certificate "
        "that every client pins."
    )
