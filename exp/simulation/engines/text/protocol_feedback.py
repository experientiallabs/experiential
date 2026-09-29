"""Bounded, content-free schema diagnostics for private world-model correction prompts."""

from pydantic import ValidationError

_TRANSITION_FIELDS = frozenset({"message", "tool_results", "state", "terminal"})
_TOOL_RESULT_FIELDS = frozenset({"call_id", "content", "is_error"})
_ERROR_TYPES = frozenset(
    {
        "extra_forbidden",
        "missing",
        "model_type",
        "dict_type",
        "list_type",
        "tuple_type",
        "string_type",
        "string_unicode",
        "string_too_short",
        "bool_type",
        "bool_parsing",
        "json_type",
        "invalid_key",
        "recursion_loop",
    }
)


def transition_validation_feedback(error: ValidationError) -> str:
    """Describe at most eight schema errors without echoing generated content.

    Only explicitly recognized error codes and schema-owned field names are returned. Unknown
    keys, dynamic state paths, custom validator messages, values, contexts and URLs are excluded.
    Locations have at most four segments; list indices above 9999 are anonymized. The complete
    feedback is bounded to 900 characters, including a capped omitted-error count.

    Args:
        error: Strict transition validation failure, retained unchanged as the parser cause.

    Returns:
        A content-free schema summary for the existing private correction request.
    """
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    details = [
        f"{entry['type'] if entry['type'] in _ERROR_TYPES else 'validation_error'} "
        f"at {_schema_path(entry['loc'])}"
        for entry in errors[:8]
    ]
    omitted = len(errors) - len(details)
    if omitted:
        details.append(f"additional_errors={omitted if omitted <= 9999 else '>9999'}")
    return ("Schema errors: " + "; ".join(details))[:900]


def _schema_path(location: tuple[int | str, ...]) -> str:
    """Return a bounded location made only of known schema fields and list indices."""
    if not location:
        return "$"
    field = location[0]
    if field not in _TRANSITION_FIELDS:
        return "$.<extra-field>"
    path = f"$.{field}"
    if len(location) == 1:
        return path
    if field != "tool_results":
        return path + ".<field>"
    index = location[1]
    path += f"[{index}]" if type(index) is int and 0 <= index <= 9999 else "[<index>]"
    if len(location) < 3:
        return path
    name = location[2]
    path += f".{name}" if name in _TOOL_RESULT_FIELDS else ".<extra-field>"
    return path + (".<field>" if len(location) > 3 else "")
