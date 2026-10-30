"""State selectors preserve ambiguity and missing-evidence boundaries."""

import pytest

from exp.runtime.environments.world_state_paths import (
    ArrayLength,
    EmptyArrayDefault,
    ProjectItems,
    SelectItem,
    SliceItems,
    state_at,
)


def test_empty_array_view_never_fills_unknown_state_or_aliases_the_default() -> None:
    """Only a positively observed empty array receives the declared virtual contents."""
    view = EmptyArrayDefault(empty_array_default=([None],))
    result = state_at([], (view,), {})
    assert result == [[None]]
    assert isinstance(result, list)
    inner = result[0]
    assert isinstance(inner, list)
    inner[0] = 99
    result.append("changed")
    assert state_at([], (view,), {}) == [[None]]
    assert state_at([[42]], (view,), {}) == [[42]]
    for unknown in (None, {}, "unknown"):
        with pytest.raises(ValueError, match="observed array"):
            state_at(unknown, (view,), {})


def test_duplicate_selection_is_not_first_match_wins() -> None:
    """Ambiguous entity identity must refuse instead of hiding a second matching entity."""
    with pytest.raises(ValueError, match="exactly one"):
        state_at([{"id": 1}, {"id": 1}], (SelectItem(select="id", argument="id"),), {"id": 1})


@pytest.mark.parametrize("start,count", [(True, 1), (0, True), (-1, 1), (0, 3), (0, 0)])
def test_slices_never_clamp_or_accept_boolean_indices(start: int, count: int) -> None:
    """Range errors cannot become plausible partial successful observations."""
    with pytest.raises(ValueError, match="bounds"):
        state_at([1, 2], (SliceItems(slice_start="s", slice_count="n"),), {"s": start, "n": count})


def test_declared_pagination_and_descriptions_use_actual_state() -> None:
    """A description page preserves names and dimensions without inventing missing rows."""
    page = SliceItems(
        slice_start="offset", slice_count="limit", default_start=0, default_count=32, clamp=True
    )
    projection = ProjectItems(
        fields={"name": ("name",), "size": ("rows", ArrayLength(length=True))}
    )
    assert state_at(
        [{"name": "A", "rows": [1, 2]}, {"name": "B", "rows": [3]}],
        (page, projection),
        {"offset": 1},
    ) == [{"name": "B", "size": 1}]
    assert state_at([1], (page,), {"offset": 9}) == []
    with pytest.raises(ValueError, match="unavailable"):
        state_at([{"name": "A"}], (projection,), {})
