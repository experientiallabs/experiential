"""Reasoning display: whether a rung returns its model's reasoning text to callers.

Every rung returns the readable reasoning its provider streams (plaintext
reasoning, thinking, and reasoning summaries) as display copy beside the
content, unless the rung is stamped ``reasoning_output_hidden`` or the operator
kill switch withholds it everywhere. Display never changes replay or the sealed
reasoning carrier.
"""

from __future__ import annotations

from exp.common.models.model import ModelCapabilities
from exp.runtime.models.providers.reasoning_compat import reasoning_display_enabled


def reasoning_output_hidden(capabilities: ModelCapabilities | None) -> bool:
    """Return whether one rung withholds its reasoning text from the caller.

    Args:
        capabilities: The rung's declared capabilities, or ``None`` when undeclared.

    Returns:
        ``True`` when the kill switch is set or the rung opts out.
    """
    if not reasoning_display_enabled():
        return True
    return capabilities is not None and capabilities.reasoning_output_hidden
