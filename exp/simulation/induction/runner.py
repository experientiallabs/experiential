"""Standalone worker that runs induced transition programs in a restricted Python subset.

The parent starts this file as ``python -S -s -P runner.py MEMORY_BYTES CPU_SECONDS`` with an
environment it builds from scratch (only a fixed hash seed and locale), no site packages, no
script-directory import path and a private working directory. The worker refuses to start with
randomized hashing, so set and dict iteration order is reproducible across runs. Before it reads any
program the worker imports the allowlisted modules, then installs kernel resource limits on
itself: CPU time, address space, no file creation and no free file-descriptor slot, so a program
can open no file, socket or pipe even if it escaped the language subset. It then speaks one JSON
line per request on standard input and output.

A program is one Python module that defines ``transition(state, args, facts)``. ``state`` maps
the call's visible state domains to JSON values and is edited in place; ``args`` are the call's
JSON arguments; ``facts`` maps starting instance-fact keys to values (read-only knowledge of
the starting world). The function returns the JSON reply, raises ``ToolError(reply)`` for an
error reply, or raises ``OutOfDomain(reason)`` for an input it does not model.

The static check (:func:`check_source`) admits a small subset: top-level imports of allowlisted
modules, each a view of the listed names in :data:`ALLOWED_MODULES` only, plain function
definitions and constant assignments; no classes, ``global``,
``with``, ``yield``, bare ``except``, dunder names or underscore attributes, and none of the
reflective builtins. Every executed program line is reported, so callers can bound a program to
the code paths that validation actually exercised.

This file imports only the standard library: it must run with no site packages.
"""

from __future__ import annotations

import ast
import builtins
import collections
import copy
import functools
import itertools
import json
import math
import os
import re
import string
import sys
import types
from collections.abc import Callable
from typing import cast

# Identity of the code that decides how programs execute, in place of its source text. Bump it
# when behavior changes; exp/tests/code_protocols_test.py guards the covered source.
RUNNER_PROTOCOL = "induced-program-runner-v2"
ENTRYPOINT = "transition"
PROGRAM_FILENAME = "<induced-program>"
MAXIMUM_SOURCE_CHARACTERS = 20_000
# The names each allowlisted module exposes: pure functions, container types and immutable
# constants. Nothing listed resolves attribute names from a string (``string.Formatter`` and
# ``string.Template`` are absent for that reason), so the static check sees every attribute a
# program reaches, and nothing listed holds state shared by programs (``copy.dispatch_table`` is
# absent). Anything not listed, including every submodule, is unreachable.
ALLOWED_MODULES: dict[str, tuple[types.ModuleType, tuple[str, ...]]] = {
    "collections": (collections, ("Counter", "OrderedDict", "defaultdict", "deque")),
    "copy": (copy, ("copy", "deepcopy")),
    "functools": (functools, ("cmp_to_key", "partial", "reduce")),
    "itertools": (
        itertools,
        (
            "accumulate",
            "batched",
            "chain",
            "combinations",
            "combinations_with_replacement",
            "compress",
            "count",
            "cycle",
            "dropwhile",
            "filterfalse",
            "groupby",
            "islice",
            "pairwise",
            "permutations",
            "product",
            "repeat",
            "starmap",
            "takewhile",
            "tee",
            "zip_longest",
        ),
    ),
    "json": (json, ("dumps", "loads")),
    "math": (
        math,
        (
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "cbrt",
            "ceil",
            "comb",
            "copysign",
            "cos",
            "cosh",
            "degrees",
            "dist",
            "e",
            "erf",
            "erfc",
            "exp",
            "exp2",
            "expm1",
            "fabs",
            "factorial",
            "floor",
            "fmod",
            "frexp",
            "fsum",
            "gamma",
            "gcd",
            "hypot",
            "inf",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "isqrt",
            "lcm",
            "ldexp",
            "lgamma",
            "log",
            "log10",
            "log1p",
            "log2",
            "modf",
            "nan",
            "nextafter",
            "perm",
            "pi",
            "pow",
            "prod",
            "radians",
            "remainder",
            "sin",
            "sinh",
            "sqrt",
            "sumprod",
            "tan",
            "tanh",
            "tau",
            "trunc",
            "ulp",
        ),
    ),
    "re": (
        re,
        (
            "ASCII",
            "DOTALL",
            "IGNORECASE",
            "MULTILINE",
            "VERBOSE",
            "A",
            "I",
            "M",
            "S",
            "X",
            "compile",
            "escape",
            "findall",
            "finditer",
            "fullmatch",
            "match",
            "search",
            "split",
            "sub",
            "subn",
        ),
    ),
    "string": (
        string,
        (
            "ascii_letters",
            "ascii_lowercase",
            "ascii_uppercase",
            "capwords",
            "digits",
            "hexdigits",
            "octdigits",
            "printable",
            "punctuation",
            "whitespace",
        ),
    ),
}
SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "ord",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "KeyError",
    "IndexError",
    "TypeError",
    "ZeroDivisionError",
    "StopIteration",
    "True",
    "False",
    "None",
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "format",
        "format_map",
        "gi_code",
        "gi_frame",
        "gi_yieldfrom",
        "ag_frame",
        "mro",
        "tb_frame",
        "tb_next",
        "with_traceback",
    }
)
_FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


class ProgramRefused(ValueError):
    """A program is outside the admitted language subset; it never executes."""


class ToolError(Exception):
    """Raised by a program to return an error reply; ``args[0]`` is the JSON reply."""


class OutOfDomain(Exception):
    """Raised by a program for an input it does not model; ``args[0]`` names the reason."""


class _BudgetExceeded(BaseException):
    """The program executed more lines than its step budget allows."""


def check_source(source: str) -> ast.Module:
    """Parse one program and refuse anything outside the admitted subset.

    Args:
        source: Complete program text.

    Returns:
        The parsed module.

    Raises:
        ProgramRefused: The text is too long, does not parse, or uses a refused construct.
    """
    if len(source) > MAXIMUM_SOURCE_CHARACTERS:
        raise ProgramRefused(f"program exceeds {MAXIMUM_SOURCE_CHARACTERS} characters")
    try:
        tree = ast.parse(source, filename=PROGRAM_FILENAME)
    except SyntaxError as error:
        raise ProgramRefused(f"program does not parse: {error.msg} (line {error.lineno})") from None
    entry = None
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            unknown = [alias.name for alias in statement.names if alias.name not in ALLOWED_MODULES]
            if unknown:
                raise ProgramRefused(
                    f"import of {unknown!r} is refused; allowed: {sorted(ALLOWED_MODULES)!r}"
                )
        elif isinstance(statement, ast.FunctionDef):
            if statement.name == ENTRYPOINT:
                entry = statement
        elif isinstance(statement, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in statement.targets):
                raise ProgramRefused("top-level assignments may only bind plain names")
        elif not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            raise ProgramRefused(
                f"top-level {type(statement).__name__} is refused; define functions, constants "
                "and allowlisted imports only"
            )
    if entry is None:
        raise ProgramRefused(f"program must define {ENTRYPOINT}(state, args, facts)")
    arguments = entry.args
    if (
        [argument.arg for argument in arguments.args] != ["state", "args", "facts"]
        or arguments.vararg
        or arguments.kwarg
        or arguments.kwonlyargs
        or arguments.posonlyargs
        or arguments.defaults
    ):
        raise ProgramRefused(f"{ENTRYPOINT} must take exactly (state, args, facts)")
    for node in ast.walk(tree):
        _check_node(node, top_level=node in tree.body)
    return tree


def _check_node(node: ast.AST, *, top_level: bool) -> None:
    """Refuse one construct that could reach outside the program's own values."""
    if isinstance(node, _FORBIDDEN_NODES):
        raise ProgramRefused(f"{type(node).__name__} is refused in transition programs")
    if isinstance(node, ast.Import) and not top_level:
        raise ProgramRefused("imports are allowed only at the top of the program")
    if isinstance(node, ast.FunctionDef) and node.decorator_list:
        raise ProgramRefused("decorators are refused in transition programs")
    if isinstance(node, ast.ExceptHandler) and node.type is None:
        raise ProgramRefused("bare except is refused; name the exception types")
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        raise ProgramRefused(f"name {node.id!r} is refused in transition programs")
    if isinstance(node, ast.Attribute) and (
        node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES
    ):
        raise ProgramRefused(f"attribute {node.attr!r} is refused in transition programs")
    if isinstance(node, ast.arg) and node.arg.startswith("__"):
        raise ProgramRefused(f"argument {node.arg!r} is refused in transition programs")


def _restricted_import(
    name: str,
    globals_: object = None,
    locals_: object = None,
    fromlist: object = (),
    level: int = 0,
) -> types.SimpleNamespace:
    """Return a public-attribute view of one allowlisted module, never the module itself."""
    del globals_, locals_, fromlist
    if level or name not in ALLOWED_MODULES:
        raise ImportError(f"module {name!r} is not available to transition programs")
    return _module_view(name)


@functools.cache
def _module_view(name: str) -> types.SimpleNamespace:
    """Expose only the listed names of an allowlisted module."""
    module, names = ALLOWED_MODULES[name]
    return types.SimpleNamespace(**{key: getattr(module, key) for key in names})


def _namespace() -> dict[str, object]:
    """Return fresh globals whose builtins hold only the safe names and the two signals."""
    safe: dict[str, object] = {
        name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)
    }
    safe.update(
        {"__import__": _restricted_import, "ToolError": ToolError, "OutOfDomain": OutOfDomain}
    )
    return {"__builtins__": safe, "__name__": "induced_program"}


def load_program(source: str) -> Callable[[object, object, object], object]:
    """Check, compile and execute one program's top level, returning its entry function.

    Raises:
        ProgramRefused: The program fails the static check or its top level fails.
    """
    tree = check_source(source)
    namespace = _namespace()
    try:
        exec(compile(tree, PROGRAM_FILENAME, "exec"), namespace)  # noqa: S102 - checked subset
    except Exception as error:  # noqa: BLE001 - reported as a refused program
        raise ProgramRefused(f"program top level failed: {type(error).__name__}") from None
    entry = namespace.get(ENTRYPOINT)
    if not callable(entry):
        raise ProgramRefused(f"program must define {ENTRYPOINT}(state, args, facts)")
    return cast("Callable[[object, object, object], object]", entry)


def run_request(
    request: dict[str, object], cache: dict[str, Callable[[object, object, object], object]]
) -> dict[str, object]:
    """Execute one request and describe its outcome as one JSON-ready object.

    Args:
        request: ``source``, ``state``, ``args``, ``facts`` and ``maximum_steps``.
        cache: Loaded programs keyed by exact source text.

    Returns:
        ``{"status": "reply" | "error_reply", "reply", "state", "lines"}`` on success, or
        ``{"status": "out_of_domain" | "failed" | "refused", "detail", "lines"}``.
    """
    source = request["source"]
    steps = request["maximum_steps"]
    if not isinstance(source, str) or not isinstance(steps, int):
        return {"status": "refused", "detail": "malformed request", "lines": []}
    try:
        entry = cache.get(source) or load_program(source)
    except ProgramRefused as error:
        return {"status": "refused", "detail": str(error), "lines": []}
    if len(cache) < 64:
        cache[source] = entry
    state, args, facts = request["state"], request["args"], request["facts"]
    lines: set[int] = set()
    counter = [0]
    exhausted = [False]

    def local(frame: types.FrameType, event: str, argument: object) -> object:
        """Record one executed program line and enforce the step budget."""
        del argument
        if event == "line":
            lines.add(frame.f_lineno)
            counter[0] += 1
            if counter[0] > steps:
                exhausted[0] = True
                raise _BudgetExceeded
        return local

    def tracer(frame: types.FrameType, event: str, argument: object) -> object:
        """Trace only frames of the program itself, never the standard library."""
        del event, argument
        return local if frame.f_code.co_filename == PROGRAM_FILENAME else None

    outcome: dict[str, object]
    sys.settrace(tracer)
    try:
        reply = entry(state, args, facts)
        outcome = {"status": "reply", "reply": reply, "state": state}
    except ToolError as error:
        outcome = {
            "status": "error_reply",
            "reply": error.args[0] if error.args else None,
            "state": state,
        }
    except OutOfDomain as error:
        outcome = {"status": "out_of_domain", "detail": str(error.args[0] if error.args else "")}
    except _BudgetExceeded:
        outcome = {"status": "failed", "detail": f"step budget of {steps} lines exhausted"}
    except Exception as error:  # noqa: BLE001 - a program failure is a reported outcome
        outcome = {"status": "failed", "detail": f"{type(error).__name__}: {error}"[:500]}
    finally:
        sys.settrace(None)
    if exhausted[0]:
        outcome = {"status": "failed", "detail": f"step budget of {steps} lines exhausted"}
    outcome["lines"] = sorted(lines)
    return outcome


def _install_limits(memory_bytes: int, cpu_seconds: int) -> None:
    """Limit CPU, memory, file size and descriptors before any program text is read."""
    import resource  # noqa: PLC0415 - POSIX-only; the runner is started only on POSIX hosts

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    memory_limited = False
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (memory_bytes, memory_bytes))
            memory_limited = True
        except (OSError, ValueError):
            continue
    if not memory_limited:
        raise SystemExit(71)
    # The listing itself holds one descriptor that it closes again, so keep only the descriptors
    # still open afterwards. They must be exactly 0..n-1: a free slot below the limit would let
    # an escaped program open a file.
    listed = sorted(int(name) for name in os.listdir("/dev/fd") if name.isdigit())
    held = [descriptor for descriptor in listed if _is_open(descriptor)]
    if held != list(range(len(held))):
        raise SystemExit(73)
    resource.setrlimit(resource.RLIMIT_NOFILE, (len(held), len(held)))


def _is_open(descriptor: int) -> bool:
    """Return whether one descriptor number is currently open in this process."""
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


def main() -> None:
    """Install limits, announce readiness, then answer one JSON line per request."""
    if len(sys.argv) != 3:
        raise SystemExit(64)
    if sys.flags.hash_randomization:
        raise SystemExit(72)
    _install_limits(int(sys.argv[1]), int(sys.argv[2]))
    for name in ALLOWED_MODULES:
        _module_view(name)
    output = sys.stdout
    output.write(json.dumps({"ready": True, "protocol": RUNNER_PROTOCOL}) + "\n")
    output.flush()
    cache: dict[str, Callable[[object, object, object], object]] = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = run_request(request, cache)
            text = json.dumps(response, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError, KeyError) as error:
            text = json.dumps(
                {"status": "failed", "detail": f"unencodable outcome: {error}"[:500], "lines": []}
            )
        output.write(text + "\n")
        output.flush()


if __name__ == "__main__":
    main()
