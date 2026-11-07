"""Approved refined prompts applied to every draw through the shared author-response owner.

The offline Astra refinement freezes each human-approved prompt as an immutable
``astra-frozen-prompt`` artifact (target, version, the exact code text it replaces, the
approved text). :func:`overlay_request` applies the current version of every approved prompt to
a request before it is keyed: each system message that holds an approved prompt's code text
exactly once gets the approved text instead, and the draw identity gains a suffix naming the
applied versions, so the draw is bought anew under the approved text and replays exactly after
that. A changed code default no longer matches and is left alone. With nothing approved, or for
the refinement's own draws (``astra-prompt-...``, which compare candidates against the code
text), the request and identity are unchanged.

This module reads artifacts only; it imports no model client and holds no closed-model settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from exp.common.core.artifacts import sha256_json
from exp.common.models import ModelMessage, ModelRequest
from exp.common.project import ArtifactStore

FROZEN_TYPE = "astra-frozen-prompt"
"""Artifact type of one approved, frozen prompt version (``frozen.json``)."""

OWN_PREFIX = "astra-prompt-"
"""Identity prefix of the refinement's own draws, which never take an approved prompt."""


@dataclass(frozen=True)
class Override:
    """The current approved version of one target.

    Attributes:
        target_id: The target.
        role: Its pipeline role.
        version: Its version.
        base_text: The code text it replaces.
        text: The approved text.
        sha256: Digest of the approved text.
    """

    target_id: str
    role: str
    version: int
    base_text: str
    text: str
    sha256: str


def approved_overrides(store: ArtifactStore) -> tuple[Override, ...]:
    """The current version of every approved prompt whose base digest is intact."""
    latest: dict[str, Override] = {}
    for artifact_id in store.list_ids(artifact_type=FROZEN_TYPE):
        row = json.loads(store.read_bytes(artifact_id, "frozen.json"))
        if sha256_json(row["base_text"]) != row["base_sha256"]:
            continue
        found = Override(
            target_id=row["target_id"],
            role=row["role"],
            version=int(row["version"]),
            base_text=row["base_text"],
            text=row["text"],
            sha256=row["sha256"],
        )
        if found.target_id not in latest or latest[found.target_id].version < found.version:
            latest[found.target_id] = found
    return tuple(sorted(latest.values(), key=lambda row: row.target_id))


def overlay_request(
    store: ArtifactStore, request: ModelRequest, identity: str
) -> tuple[ModelRequest, str, tuple[str, ...]]:
    """Apply every approved prompt to a request's system messages before it is keyed.

    Returns:
        The request, the draw identity (suffixed when anything applied) and the applied
        ``<target>@<version>`` labels.
    """
    if identity.startswith(OWN_PREFIX):
        return request, identity, ()
    rows = approved_overrides(store)
    if not rows:
        return request, identity, ()
    applied: list[str] = []
    messages: list[ModelMessage] = []
    for message in request.messages:
        content = message.content
        if message.role in ("system", "developer") and isinstance(content, str):
            for row in rows:
                if content.count(row.base_text) == 1:
                    content = content.replace(row.base_text, row.text, 1)
                    applied.append(f"{row.target_id}@{row.version}")
            message = message.model_copy(update={"content": content})
        messages.append(message)
    if not applied:
        return request, identity, ()
    labels = tuple(sorted(applied))
    suffix = sha256_json(list(labels))[:16]
    return (
        request.model_copy(update={"messages": tuple(messages)}),
        f"{identity}-refined-{suffix}",
        labels,
    )
