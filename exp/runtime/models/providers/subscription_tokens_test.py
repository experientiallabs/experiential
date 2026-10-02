"""Tests for the store-backed plan token source."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.runtime.models.providers.subscription_tokens import (
    ACCESS_TOKEN_REFRESH_AHEAD_SECONDS,
    StoredSubscriptionTokenSource,
    SubscriptionSignInError,
)

_NOW_MS = 1_800_000_000_000
_BINDING = StoredCredentialBinding(provider="anthropic", endpoint_sha256="e" * 64)


def _source(
    tmp_path: Path, tokens: StoredOAuthTokens | None, refreshed: list[StoredOAuthTokens]
) -> tuple[StoredSubscriptionTokenSource, ProviderAuthStore]:
    """Return a source over a temporary store whose refresher records each rotation."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    if tokens is not None:
        store.put_oauth("plan", tokens, binding=_BINDING)

    def refresh(previous: StoredOAuthTokens) -> StoredOAuthTokens:
        rotated = StoredOAuthTokens(
            access_token=f"access-{len(refreshed) + 1}",
            refresh_token=f"refresh-{len(refreshed) + 1}",
            expires_at_ms=_NOW_MS + 7_200_000,
            account_id=previous.account_id,
        )
        refreshed.append(rotated)
        return rotated

    source = StoredSubscriptionTokenSource(
        store=store,
        connection_id="plan",
        binding=_BINDING,
        refresher=refresh,
        re_sign_in_hint="exp config gateway provider add plan --replace",
        clock_ms=lambda: _NOW_MS,
    )
    return source, store


def test_a_fresh_sign_in_is_served_without_a_refresh(tmp_path: Path) -> None:
    """No rotation happens while the access token has life left."""
    rotations: list[StoredOAuthTokens] = []
    fresh = StoredOAuthTokens(access_token="a", refresh_token="r", expires_at_ms=_NOW_MS * 2)
    source, _store = _source(tmp_path, fresh, rotations)

    assert source.current() == fresh
    assert rotations == []


def test_a_near_expiry_sign_in_rotates_once_and_persists_before_use(tmp_path: Path) -> None:
    """The rotated pair replaces the stored one, and a second read reuses it."""
    rotations: list[StoredOAuthTokens] = []
    near = StoredOAuthTokens(
        access_token="a",
        refresh_token="r",
        expires_at_ms=_NOW_MS + int(ACCESS_TOKEN_REFRESH_AHEAD_SECONDS * 1_000) - 1,
        account_id="acct",
    )
    source, store = _source(tmp_path, near, rotations)

    first = source.current()
    second = source.current()

    assert len(rotations) == 1
    assert first == second == rotations[0]
    assert first.account_id == "acct"
    assert store.get_oauth("plan", binding=_BINDING) == first


def test_a_missing_sign_in_names_the_re_sign_in_command(tmp_path: Path) -> None:
    """An unsigned connection fails with the command that fixes it."""
    source, _store = _source(tmp_path, None, [])

    with pytest.raises(SubscriptionSignInError, match="provider add plan --replace"):
        source.current()


def test_a_refresh_another_process_landed_first_is_used_not_repeated(tmp_path: Path) -> None:
    """The locked re-check sees a pair a peer rotated after our unlocked read and spends nothing."""
    rotations: list[StoredOAuthTokens] = []
    near = StoredOAuthTokens(
        access_token="a",
        refresh_token="r",
        expires_at_ms=_NOW_MS + int(ACCESS_TOKEN_REFRESH_AHEAD_SECONDS * 1_000) - 1,
    )
    peer, _store = _source(tmp_path, near, rotations)

    class _PeerRefreshesAfterRead(ProviderAuthStore):
        """A second handle on the same file whose first read is followed by the peer's refresh."""

        raced = False

        def get_oauth(
            self, connection_id: str, *, binding: StoredCredentialBinding | None = None
        ) -> StoredOAuthTokens | None:
            tokens = super().get_oauth(connection_id, binding=binding)
            if not self.raced:
                self.raced = True
                peer.current()
            return tokens

    spent: list[str] = []

    def must_not_refresh(previous: StoredOAuthTokens) -> StoredOAuthTokens:
        spent.append(previous.refresh_token)
        raise AssertionError("the peer's pair was fresh; no second refresh may run")

    ours = StoredSubscriptionTokenSource(
        store=_PeerRefreshesAfterRead(tmp_path / "auth.json"),
        connection_id="plan",
        binding=_BINDING,
        refresher=must_not_refresh,
        re_sign_in_hint="exp config gateway provider add plan --replace",
        clock_ms=lambda: _NOW_MS,
    )

    assert ours.current() == rotations[0]
    assert len(rotations) == 1
    assert spent == []


def test_an_unusable_credential_file_surfaces_as_a_sign_in_error(tmp_path: Path) -> None:
    """A malformed file or an API key under the name is a credential error, not a store error.

    Gateway admission narrows past a rung only on credential errors, so a store failure must
    not escape as a bare ``ProviderAuthStoreError``.
    """
    source, store = _source(tmp_path, None, [])
    store.put("plan", "sk-api-key", binding=_BINDING)

    with pytest.raises(SubscriptionSignInError, match="API key"):
        source.current()

    store.path.write_text("not json", encoding="utf-8")
    with pytest.raises(SubscriptionSignInError, match="cannot be read"):
        source.current()
