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
    exactly as the serving reader tolerates it); a same-version mismatch, a
    malformed file, or a present-but-unreadable file fails the activation loudly
    so a self-inconsistent (or unverifiable) snapshot never becomes an alias
    authority. Only a genuinely ABSENT file (a pin whose content lives on another
    node) cannot be verified here, so that case flags and proceeds rather than
    blocking a legitimate cross-node activation.

    Args:
        state_dir: The gateway state directory the reference is relative to.
        snapshot_ref: Relative reference to the stored normalized snapshot.
        catalog_sha256: The digest the activation is about to pin for it.

    Raises:
        ValueError: The reference escapes gateway state, or a locally present
            snapshot is unreadable or its content does not match its pinned
            digest.
    """
    root = state_dir.resolve()
    reference_path = root / snapshot_ref
    snapshot_path = reference_path.resolve()
    if not snapshot_path.is_relative_to(root):
        raise ValueError("catalog snapshot reference escapes gateway state")
    try:
        data = snapshot_path.read_bytes()
    except FileNotFoundError as exc:
        # A broken symlink (or any dangling entry) at the reference is a PRESENT
        # local filesystem object, not a genuine absence: it fails closed like an
        # unreadable file rather than being waved through as a remote-node pin.
        if reference_path.is_symlink():
            _logger.error(
                "gateway refused a catalog snapshot reference that is a broken symlink"
            )
            raise ValueError(
                "catalog snapshot path is a broken symlink; refusing to pin"
            ) from exc
        # Genuinely absent: a pin whose content lives on another node. It cannot
        # be verified here, so flag and proceed rather than block a legitimate
        # cross-node activation (topology-agnostic).
        _logger.warning(
            "gateway snapshot content unverifiable at activation: the pinned "
            "snapshot file is not present on this node; pinning without a content check"
        )
        return
    except OSError as exc:
        # Present but unreadable (a permission or partial/corrupt-write fault):
        # NOT the remote-node topology case, so fail the activation closed rather
        # than pin a snapshot whose content was never verified.
        _logger.error(
            "gateway refused an unreadable local catalog snapshot at activation (%s)",
            type(exc).__name__,
        )
        raise ValueError("catalog snapshot is present but unreadable; refusing to pin") from exc
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
