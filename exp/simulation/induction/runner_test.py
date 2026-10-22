"""The program subset check and request execution of the standalone worker."""

import pytest

from exp.simulation.induction.contracts import ProgramOutcome
from exp.simulation.induction.runner import ProgramRefused, check_source, run_request

GOOD = """
import json

LIMIT = 3


def helper(rows):
    return len(rows)


def transition(state, args, facts):
    rows = state["items"]
    if args.get("fail"):
        raise ToolError({"error": "refused"})
    if args.get("unknown"):
        raise OutOfDomain("unknown input")
    rows.append(args["name"])
    return {"count": helper(rows), "dump": json.dumps(rows), "limit": LIMIT}
"""


def _run(
    source: str,
    state: dict[str, object] | None = None,
    args: dict[str, object] | None = None,
    steps: int = 10_000,
) -> ProgramOutcome:
    """Run one request in process, as the worker would."""
    return ProgramOutcome.model_validate(
        run_request(
            {
                "source": source,
                "state": state if state is not None else {"items": []},
                "args": args if args is not None else {"name": "a"},
                "facts": {},
                "maximum_steps": steps,
            },
            {},
        )
    )


def test_a_program_in_the_subset_replies_edits_state_and_reports_its_lines() -> None:
    """The entry function edits state in place and every executed line is reported."""
    outcome = _run(GOOD)
    assert outcome.status == "reply"
    assert outcome.reply == {"count": 1, "dump": '["a"]', "limit": 3}
    assert outcome.state == {"items": ["a"]}
    assert outcome.lines and all(isinstance(line, int) for line in outcome.lines)


def test_error_replies_and_domain_refusals_are_distinct_outcomes() -> None:
    """ToolError is an error reply with state; OutOfDomain is a refusal without a reply."""
    error = _run(GOOD, args={"name": "a", "fail": True})
    assert error.status == "error_reply" and error.reply == {"error": "refused"}
    refused = _run(GOOD, args={"name": "a", "unknown": True})
    assert refused.status == "out_of_domain" and refused.detail == "unknown input"
    assert set(refused.lines) != set(_run(GOOD).lines)


def test_the_step_budget_stops_a_runaway_program() -> None:
    """An infinite loop fails once it runs more lines than allowed."""
    source = "def transition(state, args, facts):\n    while True:\n        state['x'] = 1\n"
    outcome = _run(source, steps=500)
    assert outcome.status == "failed" and "step budget" in str(outcome.detail)


def test_unavailable_builtins_fail_without_reaching_the_host() -> None:
    """Reflective and I/O builtins are simply absent from the program namespace."""
    for body in ("open('/etc/passwd')", "eval('1')", "getattr(state, 'keys')", "type(state)"):
        outcome = _run(f"def transition(state, args, facts):\n    return {body}\n")
        assert outcome.status == "failed" and "NameError" in str(outcome.detail)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\ndef transition(state, args, facts):\n    return 1\n", "import of"),
        ("from json import dumps\ndef transition(state, args, facts):\n    return 1\n", "Import"),
        ("def transition(state, args, facts):\n    return state.__class__\n", "attribute"),
        ("def transition(state, args, facts):\n    return state._x\n", "attribute"),
        ("def transition(state, args, facts):\n    return __builtins__\n", "name"),
        ("def transition(state, args, facts):\n    yield 1\n", "Yield"),
        ("class A:\n    pass\ndef transition(state, args, facts):\n    return 1\n", "Class"),
        (
            "def transition(state, args, facts):\n    try:\n        return 1\n    except:\n"
            "        return 2\n",
            "bare except",
        ),
        ("def transition(state, args, facts):\n    return '{0.x}'.format(state)\n", "format"),
        ("def transition(state, args):\n    return 1\n", "exactly"),
        ("def other(state, args, facts):\n    return 1\n", "must define"),
        ("print(1)\ndef transition(state, args, facts):\n    return 1\n", "top-level"),
        ("def transition(state, args, facts)\n", "does not parse"),
    ],
)
def test_constructs_outside_the_subset_are_refused_before_execution(
    source: str, message: str
) -> None:
    """The static check names the refused construct."""
    with pytest.raises(ProgramRefused, match=message):
        check_source(source)
    assert _run(source).status == "refused"


def test_imported_modules_expose_only_public_non_module_attributes() -> None:
    """An allowlisted module is a view, so its own imports such as codecs are unreachable."""
    outcome = _run("import json\ndef transition(state, args, facts):\n    return json.codecs\n")
    assert outcome.status == "failed" and "AttributeError" in str(outcome.detail)
