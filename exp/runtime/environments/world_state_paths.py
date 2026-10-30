"""Finite argument-bound selection of private state for tool reply consistency."""

from copy import deepcopy
from typing import Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, canonical_json_bytes


class ArgumentKey(ContractModel):
    """Resolve an exact path segment from an action argument.

    Attributes:
        argument: Required string key or integer index in the current call arguments.
    """

    argument: str = Field(min_length=1)


class SelectItem(ContractModel):
    """Select exactly one list member by an argument's equality to its field.

    Attributes:
        select: Object field whose value identifies the member.
        argument: Required argument supplying the exact match value.
    """

    select: str = Field(min_length=1)
    argument: str = Field(min_length=1)


class SliceItems(ContractModel):
    """Select an exact range without Python's silent bounds clamping.

    Attributes:
        slice_start: Argument supplying the range's first index.
        slice_count: Argument supplying the positive requested length.
        default_start: Explicit tool default when the start argument is absent.
        default_count: Explicit tool default when the count argument is absent.
        offset: Index origin, zero or one.
        each: Apply this slice to each child list, for rectangular projections.
        clamp: Explicit pagination contract that permits a shorter page or an empty past-end
            page. False for exact ranges; never inferred from the available array length.
    """

    slice_start: str = Field(min_length=1)
    slice_count: str = Field(min_length=1)
    default_start: StrictInt | None = None
    default_count: StrictInt | None = Field(default=None, ge=1)
    offset: int = Field(default=0, ge=0, le=1, strict=True)
    each: bool = Field(default=False, strict=True)
    clamp: bool = Field(default=False, strict=True)


class ArrayLength(ContractModel):
    """Read the known length of a materialized array.

    Attributes:
        length: Literal operation discriminator; unavailable arrays are never zero.
    """

    length: Literal[True]


class EmptyArrayDefault(ContractModel):
    """A declared view of a known empty array, without filling absent or unknown state.

    Attributes:
        empty_array_default: Explicit nonempty contents exposed only when the observed array
            is empty, such as a spreadsheet's virtual blank cell. Never changes stored state.
    """

    empty_array_default: tuple[JsonValue, ...] = Field(min_length=1)


class ProjectItems(ContractModel):
    """Render declared fields from each selected object without generating new facts.

    Attributes:
        fields: Output names mapped to exact relative paths, optionally ending in array length.
    """

    fields: dict[str, tuple[StrictStr | StrictInt | ArrayLength | EmptyArrayDefault, ...]] = Field(
        min_length=1, max_length=32
    )


class ArgumentValue(ContractModel):
    """Read an argument value with an explicitly declared default.

    Attributes:
        argument_value: Argument field copied without conversion.
        default: Tool-declared value when the field is absent.
    """

    argument_value: str = Field(min_length=1)
    default: JsonValue


class PageCursor(ContractModel):
    """Compute the next cursor for explicitly bounded offset pagination.

    Attributes:
        page_cursor: Clamp-enabled page selection applied to the current array. The result
            is the next start index, or null when the page reaches or exceeds the array end.
    """

    page_cursor: SliceItems

    @model_validator(mode="after")
    def bounded_page(self) -> Self:
        """Reject an incoherent pagination contract before any provider or tool call."""
        if not self.page_cursor.clamp or self.page_cursor.each or self.page_cursor.offset:
            raise ValueError("page cursor requires clamped zero-based pagination")
        return self


type StatePathPart = (
    StrictStr
    | StrictInt
    | ArgumentKey
    | SelectItem
    | SliceItems
    | ArrayLength
    | ProjectItems
    | ArgumentValue
    | PageCursor
    | EmptyArrayDefault
)


def resolve_key(part: str | int | ArgumentKey, arguments: JsonObject) -> str | int:
    """Resolve a path key, refusing missing arguments and boolean indices."""
    value = arguments.get(part.argument) if isinstance(part, ArgumentKey) else part
    if type(value) not in (str, int):
        raise ValueError("state binding requires a string or integer path argument")
    assert isinstance(value, (str, int))
    return value


def _slice(value: JsonValue, part: SliceItems, arguments: JsonObject) -> JsonValue:
    """Refuse unknown bounds or out-of-range requests rather than inventing a smaller result."""
    start = arguments.get(part.slice_start, part.default_start)
    count = arguments.get(part.slice_count, part.default_count)
    if type(start) is not int or type(count) is not int:
        raise ValueError("state slice requires known integer bounds")
    assert isinstance(start, int) and isinstance(count, int)
    start -= part.offset
    if (
        not isinstance(value, list)
        or start < 0
        or count < 1
        or (not part.clamp and start + count > len(value))
    ):
        raise ValueError("state slice exceeds existing bounds; success cannot be clamped")
    return value[start : start + count]


def state_at(value: JsonValue, path: tuple[StatePathPart, ...], arguments: JsonObject) -> JsonValue:
    """Resolve a finite read without fetching data, executing code or inferring unknown values."""
    for part in path:
        if isinstance(part, EmptyArrayDefault):
            if not isinstance(value, list):
                raise ValueError("empty-array view requires an observed array, not unknown state")
            if not value:
                value = deepcopy(list(part.empty_array_default))
        elif isinstance(part, ArgumentValue):
            value = arguments.get(part.argument_value, part.default)
        elif isinstance(part, PageCursor):
            page = part.page_cursor
            selected = _slice(value, page, arguments)
            assert isinstance(value, list) and isinstance(selected, list)
            start = arguments.get(page.slice_start, page.default_start)
            assert isinstance(start, int)
            end = start + len(selected)
            value = end if end < len(value) else None
        elif isinstance(part, ProjectItems):
            if not isinstance(value, list):
                raise ValueError("state projection requires an observed list")
            value = [
                {name: state_at(item, selected, {}) for name, selected in part.fields.items()}
                for item in value
            ]
        elif isinstance(part, ArrayLength):
            if not isinstance(value, list):
                raise ValueError("state length requires a known array")
            value = len(value)
        elif isinstance(part, SelectItem):
            if part.argument not in arguments or not isinstance(value, list):
                raise ValueError("state selection requires a known argument and list")
            matches = [
                item
                for item in value
                if isinstance(item, dict)
                and part.select in item
                and canonical_json_bytes(item[part.select])
                == canonical_json_bytes(arguments[part.argument])
            ]
            if len(matches) != 1:
                raise ValueError("state selection requires exactly one matching member")
            value = matches[0]
        elif isinstance(part, SliceItems):
            if part.each:
                if not isinstance(value, list):
                    raise ValueError("state slice requires a list of rows")
                value = [_slice(item, part, arguments) for item in value]
            else:
                value = _slice(value, part, arguments)
        else:
            key = resolve_key(part, arguments)
            if isinstance(value, dict) and isinstance(key, str) and key in value:
                value = value[key]
            elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
                value = value[key]
            else:
                raise ValueError(
                    "reply/state contract path is unavailable; provide the declared state"
                )
    return value
