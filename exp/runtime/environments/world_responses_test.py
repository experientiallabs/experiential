"""Reply contracts reject shape drift without converting evidence or retrying predictions."""

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import ToolEffectPolicy
from exp.runtime.environments.world_responses import (
    ReplyStateBinding,
    printed_test_run_inconsistency,
    read_test_summaries,
    validate_reply,
    validate_reply_state,
)


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


PYTEST_PASS = (
    "============================= test session starts ==============================\n"
    "collected 4 items\n\ntest_smoothing.py ....                [100%]\n\n"
    "============================== 4 passed in 0.02s ==============================="
)
PYTEST_FAIL = (
    "test_engine.py .F                                         [100%]\n"
    "FAILED test_engine.py::test_cli - AssertionError\n"
    "========================= 1 failed, 1 passed in 0.09s ==========================\n"
    "Command exited with code 1"
)
SUITE = "/opt/venv/bin/python -m pytest test_smoothing.py"
WRITES = frozenset({"tests", "files"})


def bash(command: str) -> ToolCall:
    """One shell action."""
    return ToolCall(name="bash", call_id="call", arguments={"command": command})


def suites(status: str) -> JsonObject:
    """Domains with one oracle suite keyed by its file, as a world author declares it."""
    return {
        "tests": {"suites": {"test_smoothing.py": {"status": status, "tests": 4}}},
        "files": {"files": {}},
    }


def receipt_bindings() -> tuple[ReplyStateBinding, ...]:
    """A frozen suite command whose structured result binds status, counts and exit code."""
    scope = {"argument": "command", "equals": SUITE}
    return tuple(
        ReplyStateBinding.model_validate(
            {"reply_path": [field], "state_path": ["tests", "run", field], "command": scope}
        )
        for field in ("status", "passed", "failed", "exit_code")
    )


def run_state(status: str, passed: int, failed: int, code: int | None) -> JsonObject:
    """Committed test state recording one run of the registered suite."""
    run = {"status": status, "passed": passed, "failed": failed, "exit_code": code}
    return {"tests": {"run": run}}


def receipt(status: str, passed: int, failed: int, code: int) -> Observation:
    """A structured reply of the registered suite command."""
    return Observation(
        content=json.dumps(
            {"status": status, "passed": passed, "failed": failed, "exit_code": code}
        )
    )


def test_declared_receipt_accepts_a_coherent_passing_run() -> None:
    """A registered suite's result recorded exactly as the reply states commits."""
    after = run_state("passed", 4, 0, 0)
    validate_reply_state(
        receipt("passed", 4, 0, 0), receipt_bindings(), {}, after, action=bash(SUITE)
    )


def test_declared_receipt_refuses_a_pass_left_not_run() -> None:
    """The reply says the registered suite passed while its state stayed not_run: refused."""
    after = run_state("not_run", 0, 0, None)
    with pytest.raises(ValueError, match=r'\["status"\] contradicts'):
        validate_reply_state(
            receipt("passed", 4, 0, 0), receipt_bindings(), {}, after, action=bash(SUITE)
        )


def test_declared_receipt_refuses_a_failure_recorded_as_passed() -> None:
    """The reply says the registered suite failed while its state says passed: refused."""
    after = run_state("passed", 4, 0, 0)
    with pytest.raises(ValueError, match="contradicts its declared after state binding"):
        validate_reply_state(
            receipt("failed", 3, 1, 1), receipt_bindings(), {}, after, action=bash(SUITE)
        )


def test_declared_receipt_leaves_other_commands_alone() -> None:
    """No test command, another suite or an echoed summary is outside the frozen scope."""
    after = run_state("not_run", 0, 0, None)
    for command in ("ls -F", "pytest my_check.py", 'echo "4 passed in 0.01s"', SUITE + " -q"):
        validate_reply_state(
            Observation(content=PYTEST_PASS), receipt_bindings(), {}, after, action=bash(command)
        )


def test_declared_write_byte_count_refuses_another_size() -> None:
    """A write result schema declaring UTF-8 bytes must report the committed file's size."""
    binding = ReplyStateBinding.model_validate(
        {
            "reply_path": ["bytes"],
            "state_path": ["files", "files", {"argument": "path"}, {"utf8_length": True}],
        }
    )
    content = "caf\u00e9\n" * 3
    action = ToolCall(
        name="write_file", call_id="call", arguments={"path": "a.py", "content": content}
    )
    after: JsonObject = {"files": {"files": {"a.py": content}}}
    size = len(content.encode())
    validate_reply_state(
        Observation(content=json.dumps({"bytes": size})), (binding,), {}, after, action=action
    )
    for wrong in (len(content), 3742):
        with pytest.raises(ValueError, match="contradicts its declared after state binding"):
            validate_reply_state(
                Observation(content=json.dumps({"bytes": wrong})),
                (binding,),
                {},
                after,
                action=action,
            )


def test_declared_append_byte_count_binds_the_written_argument() -> None:
    """Append semantics bind the byte count to the appended content, not the whole file."""
    binding = ReplyStateBinding.model_validate(
        {
            "reply_path": ["bytes"],
            "state_path": [{"argument_value": "content", "default": ""}, {"utf8_length": True}],
        }
    )
    action = ToolCall(name="append", call_id="c", arguments={"path": "a", "content": "\u00e9x"})
    validate_reply_state(Observation(content='{"bytes": 3}'), (binding,), {}, {}, action=action)
    with pytest.raises(ValueError, match="contradicts"):
        validate_reply_state(Observation(content='{"bytes": 2}'), (binding,), {}, {}, action=action)


def finding(action: ToolCall, reply: str, after: JsonObject) -> str | None:
    """The diagnostic for one undeclared text reply."""
    return printed_test_run_inconsistency(action, Observation(content=reply), after, WRITES)


def test_text_summary_contradicting_state_is_only_described() -> None:
    """Without a declared receipt the contradiction is a diagnostic string, never a refusal."""
    found = finding(bash(SUITE), PYTEST_PASS, suites("not_run"))
    assert found is not None
    assert "4 passed" in found and "suites/test_smoothing.py" in found and "'not_run'" in found
    assert finding(bash(SUITE), PYTEST_PASS, suites("passed")) is None
    failing = finding(bash("pytest test_engine.py"), PYTEST_FAIL, run_state("passed", 2, 0, 0))
    assert failing is not None and "status 'passed'" in failing


def test_text_diagnostic_reads_only_the_command_argument() -> None:
    """Written content, logs read by other tools and non-runner commands are never parsed."""
    write = ToolCall(
        name="write_file",
        call_id="call",
        arguments={"path": "t.py", "content": "import pytest\n"},
    )
    assert finding(write, PYTEST_PASS, suites("not_run")) is None
    assert finding(bash("cat pytest.log"), PYTEST_PASS, suites("not_run")) is None


def test_text_diagnostic_skips_partial_repeated_and_unrelated_runs() -> None:
    """Selected subsets, several summaries and the actor's own test file imply no record."""
    for command, reply in (
        (SUITE + " -k endpoints", PYTEST_PASS),
        ("pytest test_smoothing.py::test_endpoints", PYTEST_PASS),
        ("pytest", "1 passed, 3 deselected in 0.01s"),
        ("for i in 1 2; do pytest -q; done", "1 passed in 0.4s\n1 passed in 0.4s"),
        ("pytest my_check.py", "2 passed in 0.01s"),
    ):
        assert finding(bash(command), reply, suites("not_run")) is None


def test_text_diagnostic_defers_to_a_declared_receipt() -> None:
    """A tool declaring a command-scoped receipt is governed by it, not by the text reading."""
    observation = Observation(content=PYTEST_PASS)
    found = printed_test_run_inconsistency(
        bash(SUITE), observation, suites("not_run"), WRITES, receipt_bindings()
    )
    assert found is None


def test_summaries_are_read_from_standard_pytest_and_unittest_lines_only() -> None:
    """The text reading recognizes the standard summary lines and nothing looser."""
    (quiet,) = read_test_summaries("17 passed in 0.34s")
    assert (quiet.passed, quiet.failed, quiet.outcome) == (17, 0, "passed")
    (failing,) = read_test_summaries(PYTEST_FAIL)
    assert (failing.passed, failing.failed, failing.exit_code) == (1, 1, 1)
    (errors,) = read_test_summaries("==== 1 failed, 2 errors, 3 passed, 1 warning in 1.2s ====")
    assert (errors.passed, errors.failed) == (3, 3)
    (unit,) = read_test_summaries("....F\nRan 5 tests in 0.001s\n\nFAILED (failures=1, skipped=1)")
    assert (unit.passed, unit.failed, unit.outcome) == (3, 1, "failed")
    assert read_test_summaries("all 4 tests passed") == ()
    assert read_test_summaries("4 tests collected in 0.01s") == ()
