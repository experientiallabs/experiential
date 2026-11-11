"""Evidence-tool boundaries preserve provenance without leaking event content into inventories."""

import json

from exp.optimize.evaluation.variance.judge_reading import event_inventory, evidence_scope
from exp.optimize.evaluation.variance.judge_test import _evidence


def test_inventory_preserves_provenance_and_counts_encoded_content_without_revealing_it() -> None:
    """Content bytes describe the complete Unicode value, while the inventory keeps no content."""
    event = _evidence().events[3].model_copy(update={"content": {"private_receipt": "réglé"}})
    row = event_inventory(event)
    assert isinstance(row, dict)
    assert "content" not in row and "private_receipt" not in json.dumps(row)
    assert row["content_bytes"] == len(json.dumps(event.content, sort_keys=True).encode())
    assert row["event_id"] == "e3" and row["sequence"] == 3
    assert row["owner"] == "world" and row["scope"] == "scored"
    assert row["source"] == event.source


def test_reading_scope_does_not_assign_an_imposed_action_to_the_actor() -> None:
    """Scope describes chronological coverage separately from actor responsibility."""
    scope = evidence_scope(_evidence())
    assert scope["context_event_ids"] == ["e0", "e1"]
    assert scope["scored_event_ids"] == ["e2", "e3", "e4"]
    assert scope["experiment_imposed_event_ids"] == ["e2"]
    assert scope["actor_scored_event_ids"] == ["e4"]
    assert scope["scored_start"] == "e2"
