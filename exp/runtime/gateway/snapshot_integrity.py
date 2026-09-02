"""Write-side guard that refuses to pin a self-inconsistent catalog snapshot.

Root-cause prevention for the persistent-hydration incident: an alias revision
must never pin a normalized snapshot whose stored content does not hash to its
pinned digest, because a same-version mismatch is unservable and 503s every
alias on it. This verifies content against digest at the activation seam.
"""

from __future__ import annotations

import logging
from pathlib import Path

from exp.common.models.gateway_catalog import read_pinned_normalized_snapshot

_logger = logging.getLogger(__name__)


def refuse_self_inconsistent_snapshot(
    state_dir: Path, snapshot_ref: str, catalog_sha256: str
) -> None:
    """Refuse to pin a snapshot whose stored content does not match its digest.

    When the ``<sha>.json`` file is present under ``state_dir``, its bytes must
    parse and hash to ``catalog_sha256`` (a cross-version snapshot is tolerated
    exactly as the serving reader tolerates it); a same-version mismatch or a
    malformed file fails the activation loudly so a self-inconsistent snapshot
    never becomes an alias authority. When the file is not local (a pin whose
    content lives on another node), it cannot be verified here, so this flags and
    proceeds rather than blocking a legitimate activation.

    Args:
        state_dir: The gateway state directory the reference is relative to.
        snapshot_ref: Relative reference to the stored normalized snapshot.
        catalog_sha256: The digest the activation is about to pin for it.

    Raises:
        ValueError: The reference escapes gateway state, or a locally present
            snapshot's content does not match its pinned digest.
    """
    root = state_dir.resolve()
    snapshot_path = (root / snapshot_ref).resolve()
    if not snapshot_path.is_relative_to(root):
        raise ValueError("catalog snapshot reference escapes gateway state")
    try:
        data = snapshot_path.read_bytes()
    except OSError:
        _logger.warning(
            "gateway snapshot content unverifiable at activation: the pinned "
            "snapshot file is not present on this node; pinning without a content check"
        )
        return
    try:
        read_pinned_normalized_snapshot(data, catalog_sha256)
    except ValueError as exc:
        _logger.error(
            "gateway refused a self-inconsistent catalog snapshot at activation: "
            "stored content does not match its pinned digest (%s)",
            type(exc).__name__,
        )
        raise ValueError(
            "catalog snapshot content does not match its pinned digest; refusing to pin"
        ) from exc
