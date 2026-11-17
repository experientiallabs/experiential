"""Durable world sessions publish exactly what the full-state envelope implementation published.

The reference below rebuilds the committed envelope the way the full-state store did: the old
generic-delta reducer computes each business state, and the history and transition records are
appended with their complete ``before`` and ``after`` states in the original key order. Every
revision the incremental store publishes must equal it byte for byte, every dispatched request
must carry the reference state and history, replays must not dispatch, and the committed history
must rebuild from the event log alone, including across a store restart.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.sqlite.connection import close_idle_connections
from exp.runtime.sessions.nodes import declared_text
from exp.runtime.sessions.store import SessionStore
from exp.simulation.engines.text.generic_delta_test import reference_apply_delta
from exp.simulation.world_model.durable_test import action, setup


def _edits(rng: random.Random, items: list[JsonValue], label: str) -> list[JsonObject]:
    """Draw schema-valid generic edits over the fixture's ``items`` and ``owner`` fields.

    Args:
        rng: Seeded generator.
        items: Current reference items.
        label: Distinct new item text.

    Returns:
        Ordered generic delta ops.
    """
    ops: list[JsonObject] = [{"op": "append", "path": ["items"], "value": label}]
    if items and rng.random() < 0.5:
        ops.append({"op": "delete", "path": ["items", rng.randrange(len(items))]})
    if items and rng.random() < 0.4:
        ops.append({"op": "set", "path": ["items", 0], "value": f"{label}-first"})
    if rng.random() < 0.3:
        ops.append({"op": "copy", "from": ["items", 0], "path": ["owner"]})
    if rng.random() < 0.3:
        ops.append({"op": "remove_value", "path": ["items"], "value": label})
    return ops


def _payload(body: JsonObject) -> JsonObject:
    """Decode the predictor payload of one dispatched request."""
    messages = body["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    payload = json.loads(str(messages[1]["content"]))
    assert isinstance(payload, dict)
    return payload


def test_sparse_sessions_publish_the_full_state_envelope_exactly(tmp_path: Path) -> None:
    """Twelve randomized steps, a restart and replays match the full-state construction."""
    session, _, requests, time = setup(tmp_path)
    rng = random.Random(5)
    deltas: list[list[JsonObject]] = []
    original = session._complete

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return the scripted proposal with this step's randomized edits."""
        proposal = original(body, identity)
        effect = proposal["effect"]
        assert isinstance(effect, dict)
        effect["edits"] = copy.deepcopy(deltas[-1])
        return proposal

    session._complete = complete
    initial = session.committed_record().state
    state = initial["business_state"]
    history = initial["history"]
    assert isinstance(state, dict) and isinstance(history, list)
    transitions: list[JsonValue] = []
    for step in range(12):
        if step == 6:
            close_idle_connections()
            session._store = SessionStore(tmp_path / "world.sqlite", clock=lambda: time[0])
        items = state["items"]
        assert isinstance(items, list)
        deltas.append(_edits(rng, items, f"item-{step}"))
        observation = session.execute(action(step))
        payload = _payload(requests[-1])
        assert declared_text(payload["state"]) == declared_text(state)
        stored = session.committed_record().state
        stored_transitions = stored["transitions"]
        assert isinstance(stored_transitions, list) and isinstance(stored_transitions[-1], dict)
        record = stored_transitions[-1]
        after = reference_apply_delta(state, deltas[-1])
        actions: list[JsonValue] = [action(step).model_dump(mode="json")]
        transitions.append(
            {
                "operation_id": f"call-{step}",
                "actions": actions,
                "replies": record["replies"],
                "before": state,
                "after": after,
                "observations": record["observations"],
                "evidence": record["evidence"],
                "proposal": record["proposal"],
            }
        )
        history = [*history, {"actions": actions, "observations": record["observations"]}]
        reference: JsonObject = {
            "business_state": after,
            "history": history,
            "transitions": transitions,
        }
        assert declared_text(stored) == declared_text(reference)
        assert [observation.model_dump(mode="json")] == record["observations"]
        assert session.state == after
        state = after
    dispatched = len(requests)
    assert session.execute(action(3)) == session.execute(action(3))
    assert len(requests) == dispatched
    refs = session._store.verify_history("world")
    assert len(refs) == 13
    fourth = transitions[4]
    assert isinstance(fourth, dict)
    assert session._store.read_state(refs[4], ("business_state",)) == fourth["before"]
