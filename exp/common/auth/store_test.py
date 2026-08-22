"""Tests for the user-only provider credential store."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from exp.common.auth.store import (
    ProviderAuthStore,
    ProviderAuthStoreError,
    StoredCredentialBinding,
    StoredCredentialEndpointMismatch,
)

_BINDING = StoredCredentialBinding(provider="openai-compatible", endpoint_sha256="a" * 64)
_OTHER_BINDING = StoredCredentialBinding(provider="openai-compatible", endpoint_sha256="b" * 64)

_SECRET = "sk-store-test-secret-value"
_OTHER = "sk-other-connection-secret"


def _store(path: Path) -> ProviderAuthStore:
    """Return a store bound to one temporary credential file.

    Args:
        path: Destination ``auth.json`` path.

    Returns:
        Store that reads and writes ``path``.
    """
    return ProviderAuthStore(path)


def test_put_is_visible_to_a_fresh_store_instance(tmp_path: Path) -> None:
    """A later resolver instance reads the key written by an earlier instance."""
    path = tmp_path / "auth.json"
    _store(path).put("openai", _SECRET)

    assert _store(path).get("openai") == _SECRET
    assert _store(path).get("missing") is None


def test_put_survives_a_fresh_python_process(tmp_path: Path) -> None:
    """A child process reading the same file sees the persisted credential."""
    path = tmp_path / "auth.json"
    _store(path).put("openai", _SECRET)
    script = (
        "from pathlib import Path\n"
        "from exp.common.auth.store import ProviderAuthStore\n"
        f"store = ProviderAuthStore(Path({str(path)!r}))\n"
        "raise SystemExit(0 if store.get('openai') == "
        f"{_SECRET!r} else 1)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )

    assert completed.returncode == 0, completed.stderr
    assert _SECRET not in completed.stdout
    assert _SECRET not in completed.stderr


def test_same_provider_connections_stay_distinct(tmp_path: Path) -> None:
    """Two OpenAI connections keep independent stored credentials."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("openai", _SECRET)
    store.put("openai-work", _OTHER)

    fresh = _store(path)
    assert fresh.get("openai") == _SECRET
    assert fresh.get("openai-work") == _OTHER
    assert fresh.connection_ids() == ("openai", "openai-work")


def test_file_and_parent_permissions_are_restrictive(tmp_path: Path) -> None:
    """The credential file is 0600 and its parent directory is 0700."""
    path = tmp_path / "exp" / "auth.json"
    _store(path).put("openai", _SECRET)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_atomic_replacement_keeps_the_previous_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rename leaves the previous complete document in place."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("openai", _SECRET)

    def _fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        """Refuse the final rename after the staging file has been written.

        Args:
            source: Staging path.
            target: Destination credential path.
        """
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.put("openai", _OTHER)

    assert json.loads(path.read_text(encoding="utf-8"))["openai"]["key"] == _SECRET
    assert _store(path).get("openai") == _SECRET


def test_repr_and_str_never_include_secret_values(tmp_path: Path) -> None:
    """Store representations name the path and never the credential."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("openai", _SECRET)

    assert _SECRET not in repr(store)
    assert _SECRET not in str(store)
    assert "ProviderAuthStore" in repr(store)


def test_malformed_file_fails_with_a_recoverable_error(tmp_path: Path) -> None:
    """Corrupt JSON fails closed and names the recovery command without quoting secrets."""
    path = tmp_path / "auth.json"
    path.write_text(f'{{"openai": {_SECRET!r}}}', encoding="utf-8")

    with pytest.raises(ProviderAuthStoreError, match="malformed") as captured:
        _store(path).get("openai")

    message = str(captured.value)
    assert "exp config providers" in message
    assert str(path) in message
    assert _SECRET not in message
    assert _SECRET not in repr(captured.value)


def test_logout_removes_only_the_selected_connection(tmp_path: Path) -> None:
    """Removing one stored credential leaves every other connection intact."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("openai", _SECRET)
    store.put("anthropic", _OTHER)

    assert store.remove("openai") is True
    assert store.remove("openai") is False
    fresh = _store(path)
    assert fresh.get("openai") is None
    assert fresh.get("anthropic") == _OTHER


def test_bound_key_is_rejected_for_a_different_endpoint(tmp_path: Path) -> None:
    """A stored key is not returned when the connection now names another endpoint."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("acme", _SECRET, binding=_BINDING)

    assert store.get("acme", binding=_BINDING) == _SECRET
    with pytest.raises(StoredCredentialEndpointMismatch, match="does not match") as captured:
        store.get("acme", binding=_OTHER_BINDING)

    assert _SECRET not in str(captured.value)
    assert store.get("acme") == _SECRET


def test_unbound_record_is_rejected_when_a_binding_is_required(tmp_path: Path) -> None:
    """OpenCode-shaped records without endpoint identity cannot be used at a new endpoint."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("acme", _SECRET)

    with pytest.raises(StoredCredentialEndpointMismatch, match="does not match") as captured:
        store.get("acme", binding=_BINDING)

    assert _SECRET not in str(captured.value)
    assert store.get("acme") == _SECRET


def test_put_preserves_other_connection_bindings(tmp_path: Path) -> None:
    """Writing one connection leaves another connection's endpoint binding intact."""
    path = tmp_path / "auth.json"
    store = _store(path)
    store.put("acme", _SECRET, binding=_BINDING)
    store.put("other", _OTHER)

    fresh = _store(path)
    assert fresh.get("acme", binding=_BINDING) == _SECRET
    with pytest.raises(StoredCredentialEndpointMismatch):
        fresh.get("acme", binding=_OTHER_BINDING)


def test_replace_fsyncs_the_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful replace persists the parent directory entry after the rename."""
    path = tmp_path / "exp" / "auth.json"
    flushed: list[Path] = []

    def _record(directory: Path) -> None:
        """Record the directory durability call.

        Args:
            directory: Parent directory of the replaced credential file.
        """
        flushed.append(directory)

    monkeypatch.setattr("exp.common.auth.store.fsync_directory_best_effort", _record)
    _store(path).put("openai", _SECRET)

    assert flushed == [path.parent]
    assert _store(path).get("openai") == _SECRET


def test_missing_file_is_an_empty_store(tmp_path: Path) -> None:
    """Absence is not an error; the store behaves as empty."""
    store = _store(tmp_path / "missing" / "auth.json")

    assert store.get("openai") is None
    assert store.connection_ids() == ()
    assert store.remove("openai") is False


def test_symlink_destination_is_rejected(tmp_path: Path) -> None:
    """A credential path that is a symlink fails closed instead of following the link."""
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path = tmp_path / "auth.json"
    path.symlink_to(target)

    with pytest.raises(ProviderAuthStoreError, match="malformed"):
        _store(path).get("openai")
