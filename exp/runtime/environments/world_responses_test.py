"""Reply contracts reject shape drift without converting evidence or retrying predictions."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import ToolEffectPolicy
from exp.runtime.environments.world_responses import validate_reply


def test_argument_bound_reply_checks_exact_slice_and_refuses_clamping() -> None:
    """A declared rectangular read must equal current typed cells at the requested bounds."""
    from exp.runtime.environments.world_responses import ReplyStateBinding, validate_reply_state

    binding = ReplyStateBinding.model_validate(
        {
            "reply_path": ["rows"],
            "state_path": [
                {"argument": "source"},
                "sheets",
                {"select": "name", "argument": "sheet"},
                "rows",
                {"slice_start": "start", "slice_count": "count", "default_start": 1, "offset": 1},
                {
                    "slice_start": "column",
                    "slice_count": "columns",
                    "default_start": 1,
                    "default_count": 1,
                    "offset": 1,
                    "each": True,
                },
            ],
        }
    )
    state: JsonObject = {
        "file": {"sheets": [{"name": "Data", "rows": [["header", 0], ["value", 1]]}]}
    }
    action = ToolCall(
        name="read", call_id="a", arguments={"source": "file", "sheet": "Data", "count": 2}
    )
    validate_reply_state(
        Observation(content='{"rows":[["header"],["value"]]}'),
        (binding,),
        state,
        state,
        action=action,
        domains=frozenset({"file"}),
    )
    with pytest.raises(ValueError, match="contradicts"):
        validate_reply_state(
            Observation(content='{"rows":[[0],[1]]}'),
            (binding,),
            state,
            state,
            action=action,
            domains=frozenset({"file"}),
        )
    action = action.model_copy(update={"arguments": {**action.arguments, "count": 3}})
    with pytest.raises(ValueError, match="bounds"):
        validate_reply_state(
            Observation(content='{"rows":[["header"],["value"]]}'),
            (binding,),
            state,
            state,
            action=action,
            domains=frozenset({"file"}),
        )
    validate_reply_state(
        Observation(content="out of bounds", is_error=True),
        (binding,),
        state,
        state,
        action=action,
        domains=frozenset({"file"}),
    )


def test_argument_bound_reply_cannot_read_an_undeclared_domain() -> None:
    """Dynamic path selection does not bypass the manifest's authority boundary."""
    from exp.runtime.environments.world_responses import ReplyStateBinding, validate_reply_state

    binding = ReplyStateBinding.model_validate(
        {"reply_path": [], "state_path": [{"argument": "source"}]}
    )
    with pytest.raises(ValueError, match="declared domains"):
        validate_reply_state(
            Observation(content="42"),
            (binding,),
            {"secret": 42},
            {"secret": 42},
            action=ToolCall(name="read", call_id="a", arguments={"source": "secret"}),
            domains=frozenset({"public"}),
        )


@pytest.mark.parametrize(
    "reply",
    [
        '{"first":{"value":"2022-12-21"},"rows":1}',
        "not JSON",
        '{"first":NaN,"rows":1}',
        '{"first":1e10000,"rows":1}',
        '{"first":{},"first":1,"rows":1}',
    ],
)
def test_typed_scalar_reply_refuses_wrappers_and_non_json(reply: str) -> None:
    """The observed workbook shape miss is rejected, not repaired into an apparent hit."""
    schema: JsonObject = {
        "type": "object",
        "required": ["first", "rows"],
        "properties": {
            "first": {"type": ["string", "number", "boolean", "null"]},
            "rows": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="reply.*schema"):
        validate_reply(Observation(content=reply), schema)
    validate_reply(Observation(content='{"first":"2022-12-21","rows":1}'), schema)
    validate_reply(Observation(content="tool is unavailable", is_error=True), schema)
    validate_reply(Observation(content="unstructured text"), None)


def test_reply_schema_refuses_external_resolution_before_execution() -> None:
    """A manifest cannot cause a validator to fetch a remote schema or recursive definition."""
    with pytest.raises(ValueError, match="local schema"):
        ToolEffectPolicy(name="read", reply_schema={"$ref": "https://example.org/schema"})
    with pytest.raises(ValueError):
        ToolEffectPolicy(name="read", reply_schema={"type": "not-a-json-type"})


@pytest.mark.parametrize("backend", ["workbook", "records"])
@pytest.mark.parametrize("valid", [False, True])
def test_native_backends_check_reply_against_independent_state(backend: str, valid: bool) -> None:
    """Both actual files and actual SQLite rows enforce the shared relation contract."""
    from exp.runtime.environments.tests.world_conformance_test import ticket_case, workbook_case
    from exp.runtime.environments.world_responses import ReplyStateBinding
    from exp.runtime.environments.worlds import ExecutableWorldRuntime

    case = workbook_case() if backend == "workbook" else ticket_case()
    binding = (
        ReplyStateBinding(
            reply_path=("written", "output.xlsx", "sha256"),
            state_path=("output.xlsx" if valid else "input.xlsx", "sha256"),
        )
        if backend == "workbook"
        else ReplyStateBinding(
            reply_path=("agent",), state_path=("tickets", "rows", 1 if valid else 0, 5)
        )
    )
    manifest = case.runtime.manifest.model_copy(
        update={
            "tools": tuple(
                row.model_copy(update={"reply_bindings": (binding,)})
                if row.name == case.first.name
                else row
                for row in case.runtime.manifest.tools
            )
        }
    )
    runtime = ExecutableWorldRuntime(manifest, case.runtime.factory)
    with runtime.open(case.task) as world:
        before = world.checkpoint()
        if valid:
            assert not world.execute(case.first).is_error
        else:
            with pytest.raises(ValueError, match="contradicts"):
                world.execute(case.first)
            assert world.checkpoint() == before
