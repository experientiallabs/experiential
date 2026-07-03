# Agent guide — llm-waterfall

A stateless, provider-agnostic LLM client that sends each call down an ordered waterfall of
backends, failing over only on capacity errors (throttling / 5xx / timeouts) and propagating real
client errors immediately. Every call returns which backend served it, token usage, and USD cost.

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format .
uv run ty check
uv run pytest -q
```

## Python

- Every Python file must have a module docstring.
- Write Google-style docstrings for all classes and functions with significant logic. Use plain
  one-line docstrings for simple/self-explanatory classes and functions.

## Rules

1. **Clean tree before every commit.** Run `uv run ruff check .` and `uv run ty check` over the
   **whole project** and fix **every** error before committing — including errors you don't think
   you introduced. A commit must never add to or leave behind lint/type errors.

2. **Tests live inline next to the code.** A module `foo.py` is tested by `foo_test.py` in the same
   directory. There is no top-level `tests/` directory. Pytest is configured
   (`python_files = ["*_test.py"]`) to discover these. Tests are excluded from the wheel.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use pydantic's `JsonValue`, not `Any`.

4. **Keep imports explicit and fail-fast.** Imports at module scope, with exactly one exception:
   provider SDKs (`boto3`, `openai`, `anthropic`) are optional extras and are imported lazily
   inside each adapter's lock-guarded client constructor so the package imports and constructs with
   zero SDKs installed. Never catch `ImportError` to fall back to alternate behavior — a missing
   SDK raises with the extra name to install.

5. **Design every public surface from the perspective of a dev using it.** Write the call site
   first, then implement. Public surfaces stay small, composable, and explicitly typed. Error
   messages are part of the interface: a failure a user can hit must say what went wrong *and*
   what to do about it.

6. **Tests come before functionality.** Write the failing test first, watch it fail, then
   implement until it passes. Bug fixes are no exception: capture the repro as a test first.
   Never weaken a test merely to get green.

7. **Verify end-to-end before claiming done.** Unit tests passing is necessary, not sufficient —
   drive a real chain against a live backend and confirm observed behavior.

8. **Build for now, not for the future.** No speculative generality, no unused hooks, no
   backwards-compatibility shims pre-1.0. Make breaking changes freely; migrate every caller in
   the same change; keep exactly one way to do each thing.

9. **The classifier is the product's core contract.** Capacity classification prefers structured
   error codes; message-substring matching stays conservative (transport phrases only). Never
   match generic tokens like "429"/"503"/"capacity" against raw messages — a bad request whose
   message contains them must propagate, not fail over. Every classifier change needs a test.

10. **No global state.** No module-level mutable registries, no env-var reads/writes at call time,
    no singletons. Configuration flows through constructor arguments; the only mutable state is
    each adapter's lazily-built SDK client, guarded by a lock.
