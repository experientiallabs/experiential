"""The program subset check and request execution of the standalone worker."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from exp.simulation.induction import runner
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


SECRET = "fixture-secret-outside-the-sandbox"


def secret_file(directory: Path) -> Path:
    """Create the host file a contained program must never read, holding :data:`SECRET`."""
    path = directory / "secret.txt"
    path.write_text(SECRET, encoding="utf-8")
    return path


def formatter_escape(path: Path) -> str:
    """Return the reproduced escape to the real builtins, reading ``path`` if it succeeds.

    Attribute paths written as strings never reach the static check, so every entry point that
    walks them must be unreachable.
    """
    return f"""
import json
import string


def transition(state, args, facts):
    field = "0.__init__.__globals__[__builtins__][open]"
    opener, _ = string.Formatter().get_field(field, (json.JSONEncoder,), {{}})
    return opener({str(path)!r}).read()
"""


@pytest.mark.parametrize(
    "body",
    [
        "    return string.Formatter().vformat('{0.__class__}', (state,), {})",
        "    return string.Template('$x').substitute(x=1)",
        "    return json.JSONEncoder",
        "    return copy.dispatch_table",
        "    return format(state, '')",
    ],
)
def test_string_driven_attribute_walks_are_unreachable(body: str) -> None:
    """No allowed object resolves attribute names from a string, so none reaches the host."""
    source = (
        f"import copy\nimport json\nimport string\ndef transition(state, args, facts):\n{body}\n"
    )
    check_source(source)
    outcome = _run(source)
    assert outcome.status == "failed"
    assert "AttributeError" in str(outcome.detail) or "NameError" in str(outcome.detail)


def test_the_formatter_escape_to_the_real_builtins_fails(tmp_path: Path) -> None:
    """The reproduced escape passes the static check but finds no ``string.Formatter``."""
    source = formatter_escape(secret_file(tmp_path))
    check_source(source)
    outcome = _run(source)
    assert outcome.status == "failed" and "Formatter" in str(outcome.detail)
    assert SECRET not in outcome.model_dump_json()


@pytest.mark.parametrize(
    "body",
    [
        "    return '{0.__class__}'.format(state)",
        "    return '{x.__class__}'.format_map({'x': state})",
        "    return str.format('{0.__class__}', state)",
        "    return f'{state.__class__}'",
        "    return (x for x in state).gi_frame",
    ],
)
def test_format_methods_and_dunder_traversal_are_refused_statically(body: str) -> None:
    """Format entry points and dunder or frame attributes are refused before execution."""
    with pytest.raises(ProgramRefused, match="attribute"):
        check_source(f"def transition(state, args, facts):\n{body}\n")


def test_module_views_expose_exactly_the_listed_names() -> None:
    """Each view carries the curated names only, so nothing else of the module is reachable."""
    for name, (_, exported) in runner.ALLOWED_MODULES.items():
        assert sorted(vars(runner._module_view(name))) == sorted(exported)


def _under_limits(probe: str, fixture: Path) -> str:
    """Run one probe after the real worker limits are installed and return its output.

    The probe reads the path of an existing fixture file as ``sys.argv[2]``.
    """
    script = (
        "import errno, importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('runner', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._install_limits(512 * 1024 * 1024, 10)\n"
        f"{probe}"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        (sys.executable, "-S", "-c", script, str(Path(runner.__file__)), str(fixture)),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_installed_limits_leave_no_free_descriptor_slot(tmp_path: Path) -> None:
    """Even the real ``open`` fails once limits are installed: every slot below the cap is held."""
    probe = (
        "try:\n"
        "    open(sys.argv[2])\n"
        "except OSError as error:\n"
        "    print(error.errno == errno.EMFILE)\n"
    )
    assert _under_limits(probe, secret_file(tmp_path)) == "True"


def test_the_descriptor_limit_is_documented_as_a_backstop_only(tmp_path: Path) -> None:
    """Closing a held descriptor frees a slot, so the docs promise no escaped-code containment.

    Code that escaped the subset with ``os.close`` could open a file in the freed slot; the
    module documentation therefore names the language subset as the containment.
    """
    probe = "os.close(0)\nprint(open(sys.argv[2]).read())\n"
    assert _under_limits(probe, secret_file(tmp_path)) == SECRET
    documentation = " ".join(str(runner.__doc__).split())
    assert "code that escaped the subset could close one and reuse its slot" in documentation
    assert "even if it escaped" not in documentation


COUNTER = """
SEEN = []


def transition(state, args, facts):
    SEEN.append(1)
    return len(SEEN)
"""


def test_module_level_values_do_not_persist_across_requests() -> None:
    """Each request runs in fresh globals, so a module-level list starts empty every time."""
    cache: dict[str, types.CodeType] = {}
    request = {"source": COUNTER, "state": {}, "args": {}, "facts": {}, "maximum_steps": 1000}
    replies = [run_request(dict(request), cache)["reply"] for _ in range(3)]
    assert replies == [1, 1, 1]
    assert set(cache) == {COUNTER}


@pytest.mark.parametrize(
    "statement",
    ["json.dumps = len", "del json.dumps", "json.dumps += 1", "state.items = 1"],
)
def test_attribute_assignment_and_deletion_are_refused(statement: str) -> None:
    """A program cannot rebind another object's attribute, so module views stay unchanged."""
    source = f"import json\ndef transition(state, args, facts):\n    {statement}\n    return 1\n"
    with pytest.raises(ProgramRefused, match="assigning or deleting attribute"):
        check_source(source)


def test_module_views_are_fresh_for_every_import() -> None:
    """Two imports never share a view object, even of the same module."""
    assert runner._module_view("json") is not runner._module_view("json")


@pytest.mark.parametrize(
    ("used", "hard", "soft"),
    [
        (0.2, None, 2),
        (100.3, None, 102),
        (0.2, 5, 2),
        (3.0, 5, 4),
        (4.5, 5, None),
        (4.0, 4, None),
    ],
)
def test_the_cpu_soft_limit_gives_each_request_its_full_budget_or_none(
    used: float, hard: int | None, soft: int | None
) -> None:
    """A request gets one full budget past the CPU already used, never a hard-limit remainder."""
    assert runner._cpu_soft_limit(used, 1, hard) == soft
