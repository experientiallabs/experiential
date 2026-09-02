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


def _has_symlink_component(root: Path, reference: Path) -> bool:
    """Whether any path component from ``root`` up to ``reference`` is a symlink.

    A symlink anywhere on the reference path (the file itself or any parent
    directory) makes a read failure a present-but-unusable local entry rather
    than a genuine remote-node absence, so the caller must fail closed.
    """
    current = reference
    while current != root:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            # Reached the filesystem root without meeting ``root``: treat as
            # suspect rather than a clean absence.
            return True
        current = parent
    return False


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
    except OSError as exc:
        # The ONLY tolerated failure is a genuine remote-node absence: the
        # reference resolves through a real directory tree (no symlink at ANY
        # component) and simply names a file not present on this node. Anything
        # else -- a broken/dangling symlink at the reference or any parent, a
        # non-regular entry, a permission or partial-read fault -- is a present-
        # but-unusable LOCAL reference, so it fails closed rather than pinning a
        # snapshot whose content was never verified.
        if (
            isinstance(exc, FileNotFoundError)
            and not _has_symlink_component(root, reference_path)
            and reference_path.parent.is_dir()
        ):
            _logger.warning(
                "gateway snapshot content unverifiable at activation: the pinned "
                "snapshot file is not present on this node; pinning without a content check"
            )
            return
        _logger.error(
            "gateway refused an unusable local catalog snapshot reference at activation (%s)",
            type(exc).__name__,
        )
        raise ValueError(
            "catalog snapshot reference is not a readable local file; refusing to pin"
        ) from exc
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
