"""Tests for the request log's human-readable cluster labels."""

from __future__ import annotations

from wmo.optimize.cluster_labels import label_clusters, majority_prefix, tokenize


def test_tokenize_drops_stop_words_and_short_tokens() -> None:
    assert tokenize("You are the user who should cancel an order") == ["cancel", "order"]


def test_tokenize_splits_escaped_newlines() -> None:
    # JSON-encoded task payloads carry literal "\n", which would otherwise fuse the words on
    # either side into a pseudo-word like "nyou".
    assert "nyou" not in tokenize("refund the order\\nYou are helpful")
    assert "refund" in tokenize("refund the order\\nYou are helpful")


def test_labels_are_distinctive_not_merely_frequent() -> None:
    # The whole point of c-TF-IDF over raw counts: "order" is in both clusters, so it must not
    # be what either is called.
    labels = label_clusters(
        [
            ["cancel my order please", "cancel the pending order", "order cancellation refund"],
            ["exchange my order camera", "exchange the camera zoom", "camera exchange order"],
        ],
        max_terms=2,
    )
    assert "cancel" in labels[0]
    assert "camera" in labels[1] or "exchange" in labels[1]
    assert labels[0] != labels[1]


def test_max_terms_bounds_the_label() -> None:
    labels = label_clusters([["alpha beta gamma delta epsilon zeta"]], max_terms=3)
    assert len(labels[0].split()) == 3


def test_a_cluster_with_no_content_words_gets_an_empty_label() -> None:
    # Nothing honest to call it, so it keeps the empty label it would have had anyway.
    assert label_clusters([["the and for"], ["cancel order refund"]])[0] == ""


def test_one_label_per_cluster_in_order() -> None:
    assert len(label_clusters([["cancel order"], ["camera exchange"], ["flight booking"]])) == 3


def test_a_tie_breaks_alphabetically_not_by_encounter_order() -> None:
    # A label lands in a persisted artifact, so two fits of one matrix must agree. "alpha" and
    # "beta" have identical frequency and identical document frequency here, so the ONLY thing
    # that can order them is the tie-break; feeding the two texts in either order must still
    # name the cluster "alpha". Comparing one input against itself would pass on any
    # implementation, tie-break or not, which is what this test used to do.
    assert label_clusters([["alpha beta", "beta alpha"]], max_terms=1) == ["alpha"]
    assert label_clusters([["beta alpha", "alpha beta"]], max_terms=1) == ["alpha"]


def test_majority_prefix_wins_when_ids_carry_one() -> None:
    assert majority_prefix(["tau-bench:1", "tau-bench:2", "mmlu:3"]) == "tau-bench"


def test_majority_prefix_is_empty_for_prefix_less_ids() -> None:
    # What a corpus built from real traces looks like: scenarios keyed by trace hash.
    assert majority_prefix(["9f2a1c", "0bb31d"]) == ""
