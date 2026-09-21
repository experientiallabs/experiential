"""Incremental redaction of a streamed completion by a deterministic adapter.

A deterministic redactor decides about a prefix of a completion without
seeing the rest of it, so a streamed completion does not have to be buffered
before the caller receives its first byte. This module owns the release rule
that makes that safe: the adapter names how much of the buffered tail is
still undecided, everything before that point is redacted and released, and
the tail stays buffered until more text arrives or the stream ends.

The caller (the native data plane) owns the buffer. Every function here is
pure, so a release decision can run on any control-plane worker thread
without per-request state.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from exp.common.core.artifacts import ContractModel


@runtime_checkable
class StreamingRedactor(Protocol):
    """A deterministic detector that can decide about a prefix of its subject.

    An adapter implements this when its verdict on a prefix can never be
    changed by text that arrives later, except through a match that straddles
    the prefix boundary. :meth:`release_boundary` names exactly how much of
    the tail such a match could still occupy.
    """

    def release_boundary(self, text: str) -> int:
        """Return how many leading characters of ``text`` are settled.

        Args:
            text: The buffered completion tail, from the last released
                character to the newest delta.

        Returns:
            The count of leading characters no later text can change.
        """
        ...

    def redact(self, text: str) -> tuple[bool, str]:
        """Return whether ``text`` matched and its fully redacted form.

        Args:
            text: A settled prefix of the completion.

        Returns:
            The flag and the redacted text.
        """
        ...


@runtime_checkable
class StreamableClassifier(Protocol):
    """A classifier adapter that can offer a deterministic streaming redactor."""

    def stream_redactor(self) -> StreamingRedactor | None:
        """Return the deterministic redactor, or ``None`` when not streamable."""
        ...


class StreamSegment(ContractModel):
    """One incremental decision over the buffered tail of a streamed completion.

    ``release`` is redacted text the caller may send immediately. ``pending``
    is the tail the caller must keep buffered and present again with the next
    delta. Neither field is ever logged or persisted.
    """

    release: str = ""
    pending: str = ""
    flagged: bool = False


def release_segment(
    *,
    redactor: StreamingRedactor,
    pending: str,
    final: bool,
) -> StreamSegment:
    """Split one buffered tail into a redacted release and the tail to keep.

    A final segment releases everything, because no further text can extend a
    match. Otherwise the adapter's own boundary decides, so a match split
    across delta boundaries is redacted exactly as it would have been in one
    buffered completion.

    Args:
        redactor: The deterministic adapter bound by the output check.
        pending: Buffered completion tail, oldest character first.
        final: Whether the provider stream has ended.

    Returns:
        The redacted release, the tail to keep buffered, and the flag.

    Raises:
        ValueError: The adapter refused the subject, for instance because a
            bound was exceeded. The caller fails the request closed.
    """
    boundary = len(pending) if final else redactor.release_boundary(pending)
    boundary = max(0, min(boundary, len(pending)))
    if boundary == 0:
        return StreamSegment(release="", pending=pending, flagged=False)
    flagged, release = redactor.redact(pending[:boundary])
    return StreamSegment(release=release, pending=pending[boundary:], flagged=flagged)
