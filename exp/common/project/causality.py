"""Causal order between saved artifacts: a later artifact proves it follows an earlier one.

A manifest names each artifact it consumed by identity and canonical manifest SHA-256. That
digest covers the earlier artifact's files and envelope, so a writer can only know it after the
earlier artifact was saved: a citation is evidence of order that no clock can fake, and that clock
skew between processes or hosts cannot reverse. Ordering rules therefore ask whether the later
artifact cites the earlier one, directly or through the inputs it cites, and every
``created_at`` stays informational.
"""

from collections import deque

from exp.common.core.artifacts import ArtifactInput
from exp.common.project.manifests import artifact_input
from exp.common.project.store import ArtifactStore


def cites(store: ArtifactStore, later: str, earlier: ArtifactInput) -> bool:
    """Whether ``later`` cites the exact saved ``earlier`` artifact through its manifest inputs.

    Only citations that still match a saved manifest are followed, so a reference to a changed
    or missing artifact proves nothing.

    Args:
        store: Artifact store holding both artifacts.
        later: Artifact ID of the artifact that must follow.
        earlier: Exact reference (ID and manifest digest) of the artifact that must precede it.

    Returns:
        True when a chain of verified citations leads from ``later`` to ``earlier``.
    """
    if not store.exists(earlier.artifact_id) or not store.exists(later):
        return False
    if artifact_input(store.read(earlier.artifact_id).manifest) != earlier:
        return False
    seen = {later}
    queue = deque((later,))
    while queue:
        for reference in store.read(queue.popleft()).manifest.inputs:
            if reference == earlier:
                return True
            if reference.artifact_id in seen or not store.exists(reference.artifact_id):
                continue
            if artifact_input(store.read(reference.artifact_id).manifest) != reference:
                continue
            seen.add(reference.artifact_id)
            queue.append(reference.artifact_id)
    return False


def require_cites(
    store: ArtifactStore, later: str, earlier: ArtifactInput, *, refusal: str
) -> None:
    """Refuse unless ``later`` cites the exact saved ``earlier`` artifact.

    Args:
        store: Artifact store holding both artifacts.
        later: Artifact ID of the artifact that must follow.
        earlier: Exact reference of the artifact that must precede it.
        refusal: What went wrong and what to do, for the error.

    Raises:
        ValueError: No verified citation chain leads from ``later`` to ``earlier``.
    """
    if not cites(store, later, earlier):
        raise ValueError(refusal)
