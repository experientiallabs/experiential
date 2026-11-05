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
    validate_review,
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


def _split_groups(groups: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {
        "verdict": "split",
        "rationale": "Different tool contracts require separate environments.",
        "evidence": ["trace:a"],
        "subgroups": [
            {"members": list(group), "rationale": "Shared tool contracts.", "evidence": ["trace:a"]}
            for group in groups
        ],
    }


@pytest.mark.parametrize(
    "groups",
    [
        (("a",), ("b",)),
        (("trace:a",), ("trace:b",)),
        (("a",), ("trace:b",)),
    ],
)
def test_split_resolves_exact_registered_member_locators(
    groups: tuple[tuple[str, ...], ...],
) -> None:
    reviewed = validate_review(_split_groups(groups), ("a", "b"), frozenset({"trace:a", "trace:b"}))
    assert tuple(row.members for row in reviewed.subgroups) == (("a",), ("b",))


@pytest.mark.parametrize(
    "groups",
    [
        (("a", "trace:a"), ("b",)),
        (("trace:unknown",), ("trace:b",)),
        (("trace:a:span:step",), ("trace:b",)),
        (("trace:trace:a",), ("trace:b",)),
        (("a",), ("a",)),
    ],
)
def test_split_member_aliases_still_require_an_exact_partition(
    groups: tuple[tuple[str, ...], ...],
) -> None:
    with pytest.raises(ValueError, match="partition exactly"):
        validate_review(_split_groups(groups), ("a", "b"), frozenset({"trace:a", "trace:b"}))


def test_split_prefers_explicit_member_ids_over_locator_aliases() -> None:
    reviewed = validate_review(
        _split_groups((("a",), ("trace:a",))),
        ("a", "trace:a"),
        frozenset({"trace:a", "trace:trace:a"}),
    )
    assert tuple(row.members for row in reviewed.subgroups) == (("a",), ("trace:a",))
