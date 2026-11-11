"""Declared successful JSON reply contracts, independent of a backend's state authority.

Reply bindings declare equalities between reply fields and backend state. A binding may be
limited to one frozen invocation (``CommandScope``), which is the contract for a registered
suite's structured test receipt, and a state path may end in a UTF-8 byte length, which is the
contract for a write tool's declared byte count.

``printed_test_run_inconsistency`` is a diagnostic for simulated tools without such a contract:
when the action's command invokes pytest or unittest and the text reply prints exactly one
standard summary of a whole run, it describes a test-run record the tool may write that
contradicts it.
It never refuses, changes state or changes a grade, and never treats a printed summary as
evidence that a suite ran or passed.

Test-run records are recognized structurally: an object whose ``status`` holds a test outcome
word (``passed``, ``failed``, ``not_run`` and close variants), or one carrying both ``passed`` and
``failed``. A call selecting part of a suite (``-k``, ``file.py::test``, ``--lf``, deselected
tests) or printing several summaries is not checked. A record keyed by a file name is bound only
when the action or reply names that file, so an actor's own test file never contradicts the
oracle suite. One bound record must agree on status, passed and failed counts and exit code;
several bound records must include one with the printed outcome.
"""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import Field, StrictInt, StrictStr

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, canonical_json_bytes
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

Outcome = Literal["passed", "failed", "not_run"]

_RUNNER = re.compile(r"(?<![\w.-])(?:pytest|py\.test|unittest)(?![\w.-])")
_PYTEST_COUNT = r"\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|rerun)"
_PYTEST_SUMMARY = re.compile(
    rf"^=*\s*((?:{_PYTEST_COUNT})(?:, {_PYTEST_COUNT})*)\s+in\s+[\d.]+s"
    r"(?:\s+\([^)]*\))?\s*=*$"
)
_SELECTION = re.compile(r"(?<!\S)(?:-k|--deselect|--lf|--last-failed)(?:\s|=|$)|\.py::")
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in [\d.]+s$")
_UNITTEST_RESULT = re.compile(r"^(OK|FAILED)(?: \(([^)]*)\))?$")
_EXIT = re.compile(r"(?:Command exited with code|exit (?:code|status)[:=]?)\s*(-?\d+)\b", re.I)
_STATUS_WORDS: Mapping[str, Outcome] = {
    **dict.fromkeys(("passed", "pass", "ok", "success", "succeeded", "green"), "passed"),
    **dict.fromkeys(("failed", "fail", "failure", "error", "errored", "red"), "failed"),
    **dict.fromkeys(("not_run", "not run", "notrun", "never_run", "unrun", "pending"), "not_run"),
}


@runtime_checkable
class StateProjection(Protocol):
    """Backend-owned private state projection, independent of the returned tool reply."""

    def state_projection(self) -> JsonObject:
        """Read current state without changing it or consulting model-authored replies."""
        ...


class CommandScope(ContractModel):
    """Limit a reply binding to one declared, frozen invocation of a tool.

    A registered test receipt binds only the exact test command of its suite, so another
    command (or another suite) printing a similar summary never satisfies or breaks it.

    Attributes:
        argument: Action argument holding the invocation, such as ``command``.
        equals: The exact frozen invocation, including the suite file.
    """

    argument: str = Field(min_length=1)
    equals: str = Field(min_length=1)


class ReplyStateBinding(ContractModel):
    """One declared equality between a JSON reply field and actual backend state.

    Attributes:
        reply_path: Exact object keys or list indices within the decoded reply.
        state_path: Path within the backend's private state projection, with optional
            argument keys, unique-member selectors, exact slices and lengths. Missing state
            refuses.
        view: State before or after this tool call, before subsequent scheduled events.
        command: Optional frozen invocation this binding is limited to, for command-scoped
            structured receipts such as a registered suite's test result (status, passed and
            failed counts, exit code). Other invocations skip the binding.
    """

    reply_path: tuple[StrictStr | StrictInt | ArgumentKey, ...]
    state_path: tuple[StatePathPart, ...]
    view: Literal["before", "after"] = "after"
    command: CommandScope | None = Field(default=None, exclude_if=lambda value: value is None)

    def applies(self, action: ToolCall | None) -> bool:
        """Whether this binding governs the action: unscoped, or the exact frozen invocation."""
        if self.command is None:
            return True
        return action is not None and action.arguments.get(self.command.argument) == (
            self.command.equals
        )


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
    bindings = tuple(binding for binding in bindings if binding.applies(action))
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


@dataclass(frozen=True)
class TestRunReceipt:
    """The outcome a reply states for one test run.

    Attributes:
        passed: Tests reported passing.
        failed: Tests reported failing, errors included.
        outcome: ``passed`` with no failure and at least one pass, ``failed`` with any failure,
            None when nothing passed or failed (for example only skipped tests).
        exit_code: The exit code the reply states, when it states one for its only run.
        partial: The summary reports deselected tests, so the run covered part of a suite.
        summary: The summary line the receipt was read from.
    """

    __test__ = False

    passed: int
    failed: int
    outcome: Literal["passed", "failed"] | None
    exit_code: int | None
    partial: bool
    summary: str


def read_test_summaries(text: str) -> tuple[TestRunReceipt, ...]:
    """Read every standard pytest or unittest summary of a reply, in order.

    Args:
        text: The reply text.
    """
    lines = [line.strip() for line in text.splitlines()]
    found: list[tuple[int, int, bool, str]] = []
    for index, line in enumerate(lines):
        if match := _PYTEST_SUMMARY.match(line):
            words = {
                word: int(number)
                for number, word in (part.split(" ", 1) for part in match.group(1).split(", "))
            }
            failed = words.get("failed", 0) + words.get("error", 0) + words.get("errors", 0)
            found.append((words.get("passed", 0), failed, "deselected" in words, line))
        elif match := _UNITTEST_RAN.match(line):
            result = next((r for r in lines[index + 1 :] if _UNITTEST_RESULT.match(r)), None)
            verdict = None if result is None else _UNITTEST_RESULT.match(result)
            if verdict is None:
                continue
            detail = {
                key.strip(): int(value)
                for key, _, value in (
                    item.partition("=") for item in (verdict.group(2) or "").split(",") if item
                )
                if value.strip().isdigit()
            }
            failed = (
                detail.get("failures", 0)
                + detail.get("errors", 0)
                + detail.get("unexpected successes", 0)
            )
            skipped = detail.get("skipped", 0) + detail.get("expected failures", 0)
            passed = max(int(match.group(1)) - failed - skipped, 0)
            found.append((passed, failed, False, f"{line} {result}"))
    exits = _EXIT.findall(text)
    return tuple(
        TestRunReceipt(
            passed=passed,
            failed=failed,
            outcome="failed" if failed else "passed" if passed else None,
            exit_code=int(exits[-1]) if exits and len(found) == 1 else None,
            partial=partial,
            summary=summary,
        )
        for passed, failed, partial, summary in found
    )


def printed_test_run_inconsistency(
    action: ToolCall,
    observation: Observation,
    after: JsonObject,
    writes: frozenset[str],
    bindings: tuple[ReplyStateBinding, ...] = (),
) -> str | None:
    """Describe a printed test summary that the committed test state contradicts; diagnostic.

    This reads undeclared text, so it never refuses, never changes state and never changes a
    grade; a caller records the finding as a world-consistency diagnostic. Enforcement belongs
    to declared command-scoped reply bindings (``validate_reply_state``), and a tool declaring
    one is skipped here. A printed summary is never treated as evidence a suite ran or passed.

    Args:
        action: The simulated action; only its ``command`` or ``cmd`` string is read.
        observation: The proposed reply, an error reply included.
        after: Every domain the proposal would commit.
        writes: Domains the tool may write; only these can record a run of this call.
        bindings: The tool's declared reply bindings.

    Returns:
        The finding, or None when nothing applies or the state agrees.
    """
    if any(binding.command is not None for binding in bindings):
        return None
    command = next(
        (
            value
            for key in ("command", "cmd")
            if isinstance(value := action.arguments.get(key), str)
        ),
        None,
    )
    if command is None or not _RUNNER.search(command) or _SELECTION.search(command):
        return None
    receipts = read_test_summaries(observation.content)
    if len(receipts) != 1 or receipts[0].outcome is None or receipts[0].partial:
        return None
    receipt = receipts[0]
    written: JsonObject = {name: after[name] for name in sorted(writes) if name in after}
    mentioned = " ".join((command, observation.content))
    bound = [
        (path, record)
        for name, value in written.items()
        for path, record in _records(value, (name,))
        if len(path) == 1
        or not _named(path[-1])
        or path[-1] in mentioned
        or path[-1].rsplit("/", 1)[-1] in mentioned
    ]
    reported = f"{receipt.summary!r} ({receipt.outcome})"
    if len(bound) == 1:
        path, record = bound[0]
        problem = _disagreement(record, receipt)
        if problem is not None:
            return (
                f"reply prints the test summary {reported} but the committed test state at "
                f"{'/'.join(path)} records {problem}"
            )
    elif bound and not any(_outcome(r.get("status")) == receipt.outcome for _, r in bound):
        places = ", ".join("/".join(path) for path, _ in bound)
        return (
            f"reply prints the test summary {reported} but no committed test state among "
            f"{places} records that outcome"
        )
    return None


def _strings(value: JsonValue) -> Iterator[str]:
    """Yield every string inside a JSON value.

    Args:
        value: Any JSON value.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _records(
    value: JsonValue, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], JsonObject]]:
    """Yield every test-run record inside one domain value, with its path.

    Args:
        value: A domain value or a node inside one.
        path: The node's path from the domain name.
    """
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _records(item, (*path, str(index)))
        return
    if not isinstance(value, dict):
        return
    status = value.get("status")
    if (isinstance(status, str) and _outcome(status) is not None) or (
        "passed" in value and "failed" in value
    ):
        yield path, value
    for key, item in value.items():
        yield from _records(item, (*path, key))


def _outcome(status: JsonValue) -> Outcome | None:
    """Classify a recorded status as a test outcome; None for any other word.

    Args:
        status: A recorded ``status`` value.
    """
    if not isinstance(status, str):
        return None
    return _STATUS_WORDS.get(status.strip().lower().replace("-", "_"))


def _named(key: str) -> bool:
    """Whether a record's key names a file or path rather than a role such as ``last_run``.

    Args:
        key: The record's own key.
    """
    return "/" in key or bool(re.search(r"\.[A-Za-z0-9]+$", key))


def _count(value: JsonValue) -> int | None:
    """Read a recorded test count: an integer, or the length of a list of test names.

    Args:
        value: A recorded ``passed`` or ``failed`` value.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return None


def _disagreement(record: JsonObject, receipt: TestRunReceipt) -> str | None:
    """Name the first field of one bound record that contradicts the receipt; None if none.

    Args:
        record: A test-run record bound to this run.
        receipt: The reply's test-run summary.
    """
    if "status" in record:
        status = _outcome(record["status"])
        if receipt.outcome is not None and status is not None and status != receipt.outcome:
            return f"status {record['status']!r} where the reply reports {receipt.outcome!r}"
    for field, reported in (("passed", receipt.passed), ("failed", receipt.failed)):
        if field not in record:
            continue
        value = record[field]
        recorded = _count(value)
        if (value is None and reported) or (recorded is not None and recorded != reported):
            return f"{field} {value!r} where the reply reports {reported}"
    if "exit_code" in record and receipt.outcome is not None:
        code = record["exit_code"]
        if isinstance(code, bool) or not isinstance(code, int | None):
            return None
        if receipt.exit_code is not None and code != receipt.exit_code:
            return f"exit_code {code!r} where the reply reports {receipt.exit_code}"
        if code is None or (code == 0) != (receipt.outcome == "passed"):
            return f"exit_code {code!r} for a run the reply reports {receipt.outcome!r}"
    return None
