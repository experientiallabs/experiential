"""Compatibility review draws retain exact model authority across replay."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.simulation.mining.environment_families_test import (
    SNAPSHOT,
    ScriptedReviewer,
    answer,
    order_trace,
)
from exp.simulation.mining.environment_review import (
    CompatibilityReviewer,
    CompatibilitySpec,
    reconcile,
    review_variant,
)


@pytest.mark.parametrize("field", ["model_id", "capabilities_sha256", "connection_sha256"])
def test_review_and_reconcile_isolate_changed_reviewer_snapshots(field: str) -> None:
    """Changed reviewers buy fresh draws; unchanged reviewers replay every saved turn."""

    def compatible(members: frozenset[str], _variant: str) -> JsonObject:
        """Answer the single observed environment with cited compatible evidence."""
        return answer("compatible", members)

    script = ScriptedReviewer(compatible)
    traces = (order_trace("o1"),)
    spec = CompatibilitySpec(min_reviews=1)
    saved_identities: set[str] = set()
    models = (SNAPSHOT, SNAPSHOT.model_copy(update={field: "c" * 64}))
    for model in models:

        def draw(
            request: ModelRequest, *, identity: str, selected: ModelSnapshot = model
        ) -> ModelResponse:
            """Serve the exact selected snapshot from the shared replay cache."""
            return script(request, identity=identity).model_copy(update={"model": selected})

        reviewer = CompatibilityReviewer(draw=draw, model=model)
        reviewed = review_variant(
            reviewer, traces, variant="contract_first", spec=spec, scope="one-group"
        )
        reconciled = reconcile(reviewer, traces, (reviewed,), spec=spec, scope="one-group")
        assert reviewed.answer is not None and reconciled.answer is not None
        identities = set((*reviewed.identities, *reconciled.identities))
        assert identities.isdisjoint(saved_identities)
        saved_identities.update(identities)
        calls = len(script.requests)
        assert (
            review_variant(reviewer, traces, variant="contract_first", spec=spec, scope="one-group")
            == reviewed
        )
        assert reconcile(reviewer, traces, (reviewed,), spec=spec, scope="one-group") == reconciled
        assert len(script.requests) == calls
    assert len(script.requests) == 6


def test_a_large_group_is_shown_as_tool_set_buckets_and_split_by_bucket() -> None:
    """Over max_listed_members, members are bucketed by declared tools; splits name buckets."""
    from exp.simulation.mining.environment_families_test import file_trace
    from exp.simulation.mining.environment_review import group_packet, validate_review

    traces = [order_trace(f"o{i:02d}") for i in range(6)] + [file_trace(f"f{i:02d}") for i in range(3)]
    packet, shown, buckets = group_packet(traces, max_listed=4)
    assert "members" not in packet
    assert [row["members"] for row in packet["member_buckets"]] == [6, 3]
    assert buckets == {"b00": tuple(f"o{i:02d}" for i in range(6)), "b01": ("f00", "f01", "f02")}
    assert group_packet(traces, max_listed=None)[2] == {}
    members = sorted(t.trace_id for t in traces)
    split = {
        "verdict": "split",
        "rationale": "chat has no tools; edit has a file surface",
        "subgroups": [
            {"members": ["b00"], "rationale": "no tools", "evidence": [sorted(shown)[0]]},
            {"members": ["b01"], "rationale": "file edits", "evidence": [sorted(shown)[0]]},
        ],
        "evidence": [sorted(shown)[0]],
    }
    answer = validate_review(split, members, shown, buckets=buckets)
    assert [row.members for row in answer.subgroups] == [buckets["b00"], buckets["b01"]]
    split["subgroups"][1]["members"] = ["b00"]
    with pytest.raises(ValueError, match="every bucket id exactly once"):
        validate_review(split, members, shown, buckets=buckets)
