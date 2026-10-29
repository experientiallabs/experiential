"""Plan sign-in tokens as a dispatch credential: the source protocol and the local store source.

A plan connection (``ConnectionConfig.subscription``) dispatches on an OAuth access token that
expires within hours and whose refresh token rotates on every use. The client never holds a
static key; it asks a :class:`SubscriptionTokenSource` for a bearer at each physical dispatch.

Two sources satisfy the protocol. :class:`StoredSubscriptionTokenSource` reads and rewrites the
user-only credential file, which is what the local gateway and the CLI use. A hosted embedder
supplies its own through :class:`SubscriptionTokenSourceFactory` (the runtime catalog's seam), so
the sign-in can live in the embedder's secret store and a refreshed pair is written back there.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from exp.common.auth import (
    ProviderAuthStore,
    ProviderAuthStoreError,
    StoredCredentialBinding,
    StoredOAuthTokens,
)
from exp.common.core.locks import FileLockTimeout
from exp.common.models import ConnectionConfig
from exp.runtime.models.credentials import ModelCredentialError

logger = logging.getLogger(__name__)

ACCESS_TOKEN_REFRESH_AHEAD_SECONDS = 300.0
"""Refresh when the access token has this little life left, so no dispatch rides an expiry."""

TokenRefresher = Callable[[StoredOAuthTokens], StoredOAuthTokens]
"""Rotate one sign-in through its provider's refresh grant; the old refresh token is spent."""


class SubscriptionSignInError(ModelCredentialError):
    """A plan sign-in could not be established, read, or refreshed."""


class SubscriptionTokenSource(Protocol):
    """Hands out a bearer good for at least the refresh-ahead window."""

    @property
    def connection_id(self) -> str:
        """Return the connection this source mints for."""
        ...

    def current(self) -> StoredOAuthTokens:
        """Return the current sign-in, refreshed and persisted first when it was about to expire.

        Raises:
            SubscriptionSignInError: No sign-in is stored, or the refresh was refused.
        """
        ...


class SubscriptionTokenSourceFactory(Protocol):
    """Builds the token source for one plan connection (the hosted embedder's seam)."""

    def __call__(
        self, *, connection_id: str, connection: ConnectionConfig
    ) -> SubscriptionTokenSource:
        """Return the source for ``connection_id``.

        Args:
            connection_id: Exact catalog or gateway connection name.
            connection: Its secret-free metadata; ``subscription`` names the plan kind.

        Returns:
            A source whose ``current`` mints bearers for this connection.
        """
        ...


class StoredSubscriptionTokenSource:
    """Mint bearers for one connection from its sign-in in the user-only credential file.

    Every read goes to the file, so a sign-in refreshed by another process is picked up at
    once. A refresh happens ahead of expiry under the file's cross-process lock and the
    rotated pair is persisted before any dispatch uses it, so a burst of dispatches across
    processes spends the refresh token once.
    """

    def __init__(
        self,
        *,
        store: ProviderAuthStore,
        connection_id: str,
        binding: StoredCredentialBinding,
        refresher: TokenRefresher,
        re_sign_in_hint: str,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        """Bind one connection's stored sign-in.

        Args:
            store: Credential store holding the sign-in.
            connection_id: Exact connection name used as the store key.
            binding: Endpoint identity the stored sign-in must match.
            refresher: The plan kind's refresh grant.
            re_sign_in_hint: The command that signs the connection in again, named in errors.
            clock_ms: Unix-millisecond clock, injectable for deterministic tests.
        """
        self._store = store
        self._connection_id = connection_id
        self._binding = binding
        self._refresher = refresher
        self._hint = re_sign_in_hint
        self._clock_ms = clock_ms if clock_ms is not None else _unix_ms
        self._lock = threading.Lock()

    @property
    def connection_id(self) -> str:
        """Return the connection this source mints for."""
        return self._connection_id

    def current(self) -> StoredOAuthTokens:
        """Return a sign-in good for the refresh-ahead window, refreshing and persisting first.

        Raises:
            SubscriptionSignInError: No sign-in is stored, the credential file cannot be used
                (unreadable, bound to another endpoint, an API key under this name, or its
                lock held too long), or the refresh was refused. Store failures surface as this
                credential error so gateway admission narrows past the rung, exactly as an
                API-key connection's unusable credential file does.
        """
        try:
            return self._current()
        except (ProviderAuthStoreError, FileLockTimeout) as exc:
            raise SubscriptionSignInError(
                f"the plan sign-in for connection {self._connection_id!r} cannot be read: {exc}"
            ) from exc

    def _current(self) -> StoredOAuthTokens:
        """Read, and when due refresh, the stored sign-in; store failures propagate.

        Raises:
            SubscriptionSignInError: No sign-in is stored, or the refresh was refused.
            ProviderAuthStoreError: The credential file cannot be used.
            FileLockTimeout: Another holder kept the credential file lock too long.
        """
        with self._lock:
            tokens = self._store.get_oauth(self._connection_id, binding=self._binding)
            if tokens is None:
                raise self._missing()
            if not self._due(tokens):
                return tokens
            # The refresh token is single-use and other processes (a second gateway
            # worker, the CLI) share the file, so the re-check, the grant, and the write
            # run under the store's cross-process lock: a waiter finds the pair the
            # first refresher wrote instead of spending the old refresh token again.
            rotated: list[StoredOAuthTokens] = []

            def refresh(stored: StoredOAuthTokens) -> StoredOAuthTokens | None:
                """Rotate ``stored`` when it is still due under the lock, else keep it."""
                if not self._due(stored):
                    return None
                rotated.append(self._refresher(stored))
                return rotated[-1]

            current = self._store.refresh_oauth(
                self._connection_id, binding=self._binding, refresh=refresh
            )
            if current is None:
                raise self._missing()
            if rotated:
                logger.info("refreshed the plan sign-in for connection %r", self._connection_id)
            return current

    def _due(self, tokens: StoredOAuthTokens) -> bool:
        """Whether the pair expires inside the refresh-ahead window."""
        return tokens.expires_within(ACCESS_TOKEN_REFRESH_AHEAD_SECONDS, now_ms=self._clock_ms())

    def _missing(self) -> SubscriptionSignInError:
        """The error for a connection with no stored sign-in."""
        return SubscriptionSignInError(
            f"no plan sign-in is stored for connection {self._connection_id!r}; run '{self._hint}'"
        )


def _unix_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1_000)
