"""Tests for virtual-key material and versioned fingerprint protection."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from wmo.runtime.gateway.auth import (
    FingerprintPepperFile,
    GatewayAuthError,
    fingerprint_virtual_key,
    issue_key_material,
    key_prefix,
)


def test_pepper_is_mode_0600_and_rotation_retains_old_fingerprint_keys(tmp_path: Path) -> None:
    """Pepper rotation changes future HMACs without invalidating retained versions."""
    path = tmp_path / "state" / "gateway-key-pepper.json"
    pepper_file = FingerprintPepperFile(path)
    first = pepper_file.current()
    _, raw_key = issue_key_material()
    first_fingerprint = fingerprint_virtual_key(raw_key, first)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert pepper_file.rotate() == 2
    assert pepper_file.current().version == 2
    assert fingerprint_virtual_key(raw_key, pepper_file.key(1)) == first_fingerprint
    assert fingerprint_virtual_key(raw_key, pepper_file.current()) != first_fingerprint


def test_pepper_rejects_group_readable_and_symlink_state(tmp_path: Path) -> None:
    """Permission drift and link substitution fail closed."""
    path = tmp_path / "pepper.json"
    pepper_file = FingerprintPepperFile(path)
    pepper_file.current()
    path.chmod(0o640)

    with pytest.raises(GatewayAuthError, match="mode-0600"):
        pepper_file.current()

    target = tmp_path / "target.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(GatewayAuthError, match="regular"):
        pepper_file.current()


def test_virtual_key_has_256_random_bits_and_parseable_prefix() -> None:
    """Issued raw keys carry a non-secret lookup prefix and a 32-byte random secret."""
    prefix, raw_key = issue_key_material()

    assert key_prefix(raw_key) == prefix
    assert len(raw_key.removeprefix(f"wmo_vk_{prefix}_")) >= 43
    with pytest.raises(GatewayAuthError, match="invalid"):
        key_prefix("not-a-gateway-key")
