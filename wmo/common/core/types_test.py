"""Tests for the core data types."""

from __future__ import annotations

import wmo
from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Session, Step, Trace


def test_types_instantiate() -> None:
    action = Action(kind=ActionKind.TOOL_CALL, name="cd", arguments={"path": "/tmp"})
    obs = Observation(content="", is_error=False)
    step = Step(action=action, observation=obs, state_before=EnvState(), task="poke around")
    trace = Trace(trace_id="t1", steps=[step], source="file:demo.jsonl")
    session = Session(id="s1", task="poke around")
    assert trace.steps[0].action.name == "cd"
    assert session.history == []


def test_every_name_the_package_root_promises_is_importable() -> None:
    """`wmo.__all__` is the documented import path for these types (README quickstart).

    `wmo/__init__.py` has no suite of its own (AGENTS.md rule 2 exempts it), and the failure mode
    lives here: a type renamed or moved in this module leaves a name in `__all__` that
    `from wmo import X` raises AttributeError on, while `import wmo` still succeeds.
    """
    missing = [name for name in wmo.__all__ if not hasattr(wmo, name)]

    assert not missing, f"names in wmo.__all__ that no longer resolve: {missing}"
    assert wmo.ActionKind is ActionKind
