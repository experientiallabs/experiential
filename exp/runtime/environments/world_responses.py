"""Declared successful JSON reply contracts, independent of a backend's state authority."""

import json
from typing import Literal, Protocol, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import StrictInt, StrictStr

from exp.common.core.artifacts import ContractModel, JsonObject, canonical_json_bytes
from exp.common.core.json_schema import expand_local_references
from exp.common.models import ToolCall
from exp.common.traces.ingest.json_strict import reject_duplicate_json_keys
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_state_paths import (
    ArgumentKey,
    StatePathPart,
    resolve_key,
    state_at,
)


@runtime_checkable
class StateProjection(Protocol):
    """Backend-owned private state projection, independent of the returned tool reply."""

    def state_projection(self) -> JsonObject:
        """Read current state without changing it or consulting model-authored replies."""
        ...


class ReplyStateBinding(ContractModel):
    """One declared equality between a JSON reply field and actual backend state.

    Attributes:
        reply_path: Exact object keys or list indices within the decoded reply.
        state_path: Path within the backend's private state projection, with optional
            argument keys, unique-member selectors and exact slices. Missing state refuses.
        view: State before or after this tool call, before subsequent scheduled events.
    """

    reply_path: tuple[StrictStr | StrictInt | ArgumentKey, ...]
    state_path: tuple[StatePathPart, ...]
    view: Literal["before", "after"] = "after"


def validate_reply_state(
    observation: Observation,
    bindings: tuple[ReplyStateBinding, ...],
    before: JsonObject,
    after: JsonObject,
    *,
    action: ToolCall | None = None,
    domains: frozenset[str] | None = None,
) -> None:
    """Check declared relationships only, without promoting coherent state to native truth."""
    if not bindings or observation.is_error:
        return
    validate_reply(observation, {})
    reply = json.loads(observation.content)
    arguments = {} if action is None else action.arguments
    for binding in bindings:
        if domains is not None:
            first = binding.state_path[0] if binding.state_path else None
            if (
                not isinstance(first, (str, int, ArgumentKey))
                or resolve_key(first, arguments) not in domains
            ):
                raise ValueError("reply state binding exceeds the tool's declared domains")
        actual = state_at(reply, binding.reply_path, arguments)
        expected = state_at(
            before if binding.view == "before" else after, binding.state_path, arguments
        )
        if canonical_json_bytes(actual) != canonical_json_bytes(expected):
            field = json.dumps([resolve_key(part, arguments) for part in binding.reply_path])
            raise ValueError(
                f"tool reply field {field} contradicts its declared {binding.view} state "
                "binding; recompute that field from the specified state path; transition refused"
            )


def validate_reply_schema(schema: JsonObject) -> None:
    """Refuse malformed or externally resolved schemas before opening an executable world."""
    expanded = expand_local_references(schema)
    try:
        Draft202012Validator.check_schema(expanded)
    except SchemaError as exc:
        raise ValueError("invalid declared tool reply schema") from exc


def validate_state_projection(state: JsonObject, schemas: dict[str, JsonObject]) -> None:
    """Refuse unavailable or malformed domain state without coercing values or inventing facts."""
    for name, schema in schemas.items():
        if name not in state:
            raise ValueError(f"state schema requires a projection for domain {name!r}")
        validator = Draft202012Validator(expand_local_references(schema))
        failure = next(validator.iter_errors(state[name]), None)
        if failure is not None:
            path = "/".join(str(part) for part in failure.absolute_path)
            raise ValueError(
                f"state violates schema for domain {name!r} at /{path} "
                f"({failure.validator}); transition refused"
            )


def validate_reply(observation: Observation, schema: JsonObject | None) -> None:
    """Validate successful wire JSON before committing state; error observations stay errors.

    This checks representation only, never factual correctness. A tool without a declared
    schema retains its existing text contract. No coercion, field extraction, repair or
    provider retry happens here. Refusal leaves raw backend evidence available to its owner.
    """
    if schema is None or observation.is_error:
        return
    try:
        value = json.loads(
            observation.content,
            parse_constant=_nonfinite,
            object_pairs_hook=reject_duplicate_json_keys,
        )
        canonical_json_bytes(value)  # Reject numeric overflow as well as explicit NaN tokens.
    except ValueError as exc:
        raise ValueError("successful tool reply is not JSON required by its reply schema") from exc
    validator = Draft202012Validator(expand_local_references(schema))
    if not validator.is_valid(value):
        raise ValueError("successful tool reply violates its declared reply schema")


def _nonfinite(value: str) -> None:
    """Refuse non-JSON numeric tokens that Python's decoder otherwise accepts."""
    raise ValueError(f"nonfinite JSON token {value}")
