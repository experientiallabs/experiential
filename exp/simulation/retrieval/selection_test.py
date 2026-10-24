"""Raw evidence selection preserves source eligibility and current-packet identity."""

import pytest

from exp.common.models import ToolCall
from exp.simulation.retrieval.selection import HashingOriginals, SelectionQuery, selected_evidence


def test_real_local_ranking_changes_delivered_originals_and_never_caches_stale_packets() -> None:
    """Run the actual maintained hashing encoder and inspect delivered evidence receipts."""
    query = SelectionQuery(
        action=ToolCall(call_id="lookup", name="calendar", arguments={"event": "meeting"}),
        state={},
        history=(),
    )
    policy = HashingOriginals(top_k=1)
    catalog = {
        "calendar": {"tool": "calendar", "event": "meeting"},
        "mail": {"tool": "mail", "message": "newsletter"},
    }
    delivered, first = selected_evidence(policy, query, catalog)
    assert len(delivered) == 1 and set(delivered) <= set(catalog)
    changed, second = selected_evidence(policy, query, {"new": {"tool": "calendar"}})
    assert changed == {"new": {"tool": "calendar"}}
    assert first["catalog_sha256"] != second["catalog_sha256"]
    assert first["ordered_content_sha256"] != second["ordered_content_sha256"]


@pytest.mark.parametrize("top_k", [True, 0, -1])
def test_invalid_selection_count_is_rejected_before_embedding(top_k: int) -> None:
    """Reject values that cannot define the intended result inventory."""
    with pytest.raises(ValueError, match="positive integer"):
        HashingOriginals(top_k=top_k)
