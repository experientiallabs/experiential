"""Human-readable rendering of the canonical rubric axis contract."""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import wrap

from exp.common.judging.rubric import Rubric, RubricDimension

_MIN_WIDTH = 40


def render_rubric_table(
    dimensions: Sequence[RubricDimension] | Rubric,
    *,
    width: int = 80,
) -> str:
    """Render every axis, range, meaning, and score anchor for a narrow terminal.

    Args:
        dimensions: Ordered rubric axes, or a rubric that owns them.
        width: Available terminal columns. Values below 40 still wrap readably.

    Returns:
        A stacked table that stays readable on narrow terminals.
    """
    axes = dimensions.dimensions if isinstance(dimensions, Rubric) else tuple(dimensions)
    usable = max(_MIN_WIDTH, width)
    lines = ["Rubric"]
    if not axes:
        lines.append("  (no axes)")
        return "\n".join(lines)
    for index, axis in enumerate(axes, start=1):
        if index > 1:
            lines.append("")
        lines.extend(_render_axis(axis, index=index, width=usable))
    return "\n".join(lines)


def _render_axis(axis: RubricDimension, *, index: int, width: int) -> list[str]:
    """Render one axis as wrapped label, range, meaning, and score rows.

    Args:
        axis: Canonical rubric axis.
        index: One-based position in the ordered rubric.
        width: Available columns for wrapping.

    Returns:
        Display lines for this axis.
    """
    indent = 2
    body_width = max(24, width - indent)
    prefix = " " * indent
    lines = [
        f"{index}. {index_label(axis)}",
        f"{prefix}Range: {axis.min_score}-{axis.max_score}",
    ]
    lines.extend(_wrapped("Meaning", axis.description, prefix=prefix, width=body_width))
    lines.append(f"{prefix}Scores:")
    for anchor in axis.anchors:
        lines.extend(
            _wrapped(
                str(anchor.score),
                anchor.description,
                prefix=prefix + "  ",
                width=max(20, body_width - 2),
            )
        )
    return lines


def index_label(axis: RubricDimension) -> str:
    """Return the stable ID and human label for one axis."""
    return f"{axis.dimension_id}  {axis.name}"


def _wrapped(label: str, text: str, *, prefix: str, width: int) -> list[str]:
    """Wrap one labeled field so it stays inside ``width`` columns.

    Args:
        label: Field name or score printed before the first line.
        text: Plain-language value to wrap.
        prefix: Leading indent shared by every wrapped line.
        width: Maximum characters after ``prefix``.

    Returns:
        One or more wrapped display lines.
    """
    heading = f"{label}: "
    available = max(12, width - len(heading))
    pieces = wrap(text, width=available) or [""]
    lines = [f"{prefix}{heading}{pieces[0]}"]
    follow = " " * len(heading)
    lines.extend(f"{prefix}{follow}{piece}" for piece in pieces[1:])
    return lines
