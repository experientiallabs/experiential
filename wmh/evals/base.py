"""The general evaluation interface: configured at construction, one `run()` to a result.

Open-loop and closed-loop evaluation answer the same question — "how good is this world model?" —
through genuinely different inputs and metrics: open-loop replays recorded trace steps
teacher-forced and scores per-step reconstruction fidelity; closed-loop runs a live agent with the
world model as its environment and scores end-to-end task success. Forcing those inputs through
one signature would produce a union-typed blob, so the interface deliberately unifies only what is
truly common:

- an `Evaluation` is a fully configured measurement — every mode-specific input (trace files,
  task specs, the agent runtime, k) is bound at construction;
- `run()` executes it and returns an `EvalResult`: a pydantic report that can print a one-line
  `summary()` and expose a single `headline` score in [0, 1] (fidelity or task success).

`wmh eval --mode open-loop|closed-loop` selects the implementation; any future backend (e.g. a
real-environment closed loop) plugs in as one more implementation of the same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EvalResult(Protocol):
    """What every evaluation returns: a serializable report with a comparable headline."""

    @property
    def headline(self) -> float:
        """The one-number score in [0, 1] (open-loop: fidelity; closed-loop: success rate)."""
        ...

    def summary(self) -> str:
        """One human-readable line describing the result."""
        ...

    def model_dump_json(self, *, indent: int | None = None) -> str:  # satisfied by pydantic
        ...


@runtime_checkable
class Evaluation(Protocol):
    """A configured evaluation of a world model. Construct with mode-specific inputs, then run."""

    def run(self) -> EvalResult: ...
