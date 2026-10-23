"""The grounded transition protocol carries compact state edits."""

import pytest
from pydantic import ValidationError

from exp.simulation.world_model.grounded_session import _GroundedJsonProposal

OBSERVATION = {"call_id": "c", "reply": {"id": "a"}, "encoding": "json"}


def test_grounded_proposals_carry_state_edits_not_full_state() -> None:
    """Re-writing the whole state each turn dominated output; only edits are accepted."""
    edits = {"kind": "edits", "edits": [{"op": "set", "path": ["orders", "a"], "value": 1}]}
    proposal = _GroundedJsonProposal.model_validate(
        {"observations": [OBSERVATION], "effect": edits}
    )
    assert proposal.next_state({"orders": {"b": 2}}) == {"orders": {"a": 1, "b": 2}}
    with pytest.raises(ValidationError):
        _GroundedJsonProposal.model_validate(
            {"observations": [OBSERVATION], "effect": {"kind": "replace", "state": {}}}
        )
