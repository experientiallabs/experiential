"""Data-chosen clustering of the tag space, its replay and admission by cluster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exp.common.project import ProjectStore
from exp.simulation.mining.tag_clusters import (
    CLUSTERS_TYPE,
    ClusterAdmissionSpec,
    ClusteringSpec,
    TagClustering,
    admit_by_cluster,
    cluster_tags,
    load_tag_clustering,
    persist_tag_clustering,
    tag_space,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow

PROTOTYPES = (
    (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
)


def planted(
    sizes: tuple[int, ...] = (20, 14, 8), *, noise: float = 0.05, view: str = "task", seed: int = 3
) -> tuple[TagMatrix, dict[str, int]]:
    """A matrix of soft answers drawn around planted prototypes, and each trace's prototype."""
    rng = np.random.default_rng(seed)
    rows: list[TagRow] = []
    truth: dict[str, int] = {}
    for group, size in enumerate(sizes):
        for index in range(size):
            trace = f"g{group}-{index:02d}"
            bits = np.array(PROTOTYPES[group], dtype=float)
            flips = rng.random(bits.size) < noise
            values = np.abs(bits - flips) * 0.84 + 0.08 + rng.normal(0.0, 0.02, bits.size)
            rows.append(
                TagRow(
                    trace_id=trace,
                    unit_hash=f"h-{trace}",
                    values=tuple(float(min(1.0, max(0.0, v))) for v in values),
                )
            )
            truth[trace] = group
    matrix = TagMatrix(
        bank_id="bank",
        bank_version=1,
        question_ids=tuple(f"q{i}" for i in range(12)),
        view=view,
        rows=tuple(rows),
    )
    return matrix, truth


def test_the_planted_cluster_count_is_chosen_by_held_out_likelihood() -> None:
    """Three planted groups yield three pure clusters, chosen among stable counts."""
    matrix, truth = planted()
    clustering = cluster_tags([matrix], ClusteringSpec(max_clusters=8))

    assert len(clustering.clusters) == 3
    (selected,) = [score for score in clustering.scores if score.selected]
    assert selected.stable and selected.occupied == 3
    assert [c.size for c in clustering.clusters] == [20, 14, 8]
    for cluster in clustering.clusters:
        assert len({truth[trace] for trace in cluster.members}) == 1
    largest = clustering.clusters[0]
    assert set(largest.fired) == {"task:q0", "task:q1", "task:q2"}
    assert set(largest.distinctive) == set(largest.fired)
    assert largest.exemplars == largest.members[:3]
    assert "fewest occupied" in clustering.reason
    assert {s.clusters for s in clustering.scores} == set(range(1, 9))


def test_bic_selects_the_planted_count_on_confident_answers() -> None:
    """The BIC criterion is available and agrees on clearly separated groups."""
    matrix, _ = planted(sizes=(30, 30, 30), noise=0.0)
    clustering = cluster_tags([matrix], ClusteringSpec(max_clusters=6, criterion="bic"))
    assert len(clustering.clusters) == 3 and "lowest BIC" in clustering.reason


def test_a_structureless_space_stays_one_cluster() -> None:
    """Uniform noise gives no stable, better-than-one split."""
    rng = np.random.default_rng(0)
    rows = tuple(
        TagRow(trace_id=f"t{i:02d}", unit_hash="h", values=tuple(float(v) for v in rng.random(10)))
        for i in range(60)
    )
    matrix = TagMatrix(
        bank_id="b",
        bank_version=1,
        question_ids=tuple(f"q{i}" for i in range(10)),
        view="task",
        rows=rows,
    )
    assert len(cluster_tags([matrix]).clusters) == 1


def test_tag_space_joins_views_and_drops_partially_tagged_traces() -> None:
    """Columns carry their view; a trace missing from one matrix is dropped, not imputed."""
    task, _ = planted(sizes=(3, 3, 3))
    errors, _ = planted(sizes=(3, 3, 3), view="errors", seed=9)
    errors = errors.model_copy(update={"rows": errors.rows[1:]})
    traces, dropped, columns, values = tag_space([task, errors])
    assert dropped == ("g0-00",) and "g0-00" not in traces
    assert columns[0] == "task:q0" and columns[12] == "errors:q0" and values.shape == (8, 24)
    with pytest.raises(ValueError, match="repeat columns"):
        tag_space([task, task])
    authored = task.model_copy(
        update={"bank_id": "authored", "question_ids": tuple(f"a{q}" for q in task.question_ids)}
    )
    _, _, joined, both = tag_space([task, authored])
    assert joined[0] == "task:q0" and joined[12] == "task:aq0" and both.shape == (9, 24)
    only, _, _, _ = tag_space([task], trace_ids={"g1-00", "g2-02"})
    assert only == ("g1-00", "g2-02")


def test_clustering_is_deterministic_and_replays_from_its_artifact(tmp_path: Path) -> None:
    """The same space clusters identically, and the saved record loads back unchanged."""
    matrix, _ = planted()
    first = cluster_tags([matrix])
    second = cluster_tags([matrix])
    assert first == second and first.identity() == second.identity()
    store = ProjectStore(tmp_path, "case").artifacts
    reference = persist_tag_clustering(store, first, code_revision="test")
    assert persist_tag_clustering(store, second, code_revision="test") == reference
    assert store.read(reference.artifact_id).manifest.artifact_type == CLUSTERS_TYPE
    assert load_tag_clustering(store, reference) == first


def _clustering() -> tuple[TagClustering, dict[str, int]]:
    """The planted clustering and its truth."""
    matrix, truth = planted()
    return cluster_tags([matrix], ClusteringSpec(max_clusters=6)), truth


def test_admission_reads_only_eligible_traces_and_records_why() -> None:
    """Admission keeps to the eligible role, follows cluster order and explains each trace."""
    clustering, truth = _clustering()
    eligible = {trace for trace in truth if not trace.endswith(("00", "01"))}
    admission = admit_by_cluster(
        clustering,
        eligible=eligible,
        held_out={"held-1"},
        names={"c02": "rare cluster"},
    )
    assert set(admission.trace_ids()) == eligible
    assert [q.eligible for q in admission.quotas] == [18, 12, 6]
    row = next(r for r in admission.admitted if r.cluster_id == "c02")
    assert row.cluster_name == "rare cluster" and "c02 (rare cluster)" in row.reason
    assert admission.named and not admission.refused
    assert admission.clustering == clustering.identity()


def test_admission_shares_a_total_evenly_and_honours_caps_and_exclusions() -> None:
    """A total is shared in turn, so the rare cluster is admitted whole before others are capped."""
    clustering, _ = _clustering()
    admission = admit_by_cluster(
        clustering,
        ClusterAdmissionSpec(total=21),
        eligible=set(clustering.trace_ids),
        held_out=(),
    )
    assert [q.admitted for q in admission.quotas] == [7, 7, 7]
    capped = admit_by_cluster(
        clustering,
        ClusterAdmissionSpec(total=30, per_cluster=12, exclude_clusters=("c01",)),
        eligible=set(clustering.trace_ids),
        held_out=(),
    )
    assert [q.admitted for q in capped.quotas] == [12, 0, 8]
    assert capped.quotas[1].reason == "excluded by the admission spec"
    refused = capped.refused
    assert any("per-cluster cap" in why for why in refused.values())
    assert any("c01 excluded" in why for why in refused.values())
    small = admit_by_cluster(
        clustering,
        ClusterAdmissionSpec(minimum_cluster_size=10),
        eligible=set(clustering.trace_ids),
        held_out=(),
    )
    assert small.quotas[2].admitted == 0 and "below the minimum" in small.quotas[2].reason


def test_admission_refuses_held_out_traces_unknown_names_and_empty_results() -> None:
    """A clustering holding a held-out trace is refused outright."""
    clustering, _ = _clustering()
    with pytest.raises(ValueError, match="held out"):
        admit_by_cluster(clustering, eligible=clustering.trace_ids, held_out={"g0-03"})
    with pytest.raises(ValueError, match="unknown clusters"):
        admit_by_cluster(clustering, eligible=(), held_out=(), names={"c99": "x"})
    with pytest.raises(ValueError, match="admitted no trace"):
        admit_by_cluster(clustering, eligible={"outside"}, held_out=())
