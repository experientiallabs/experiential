"""Test 1 of scenario-set verification: task-identity recovery on labeled trace corpora.

Benchmarks with known task identities (tau2-bench stamps `domain`/`task_id` into trace metadata)
let us grade the front half of the construction pipeline directly: strip the labels, run
facet extraction -> embedding -> clustering, and measure how well the discovered clusters recover
the true task structure (cluster purity and Adjusted Rand Index). If Clio-style facets can't
recover task identity on a corpus where tasks are cleanly defined, nothing downstream matters.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from wmh.core.types import Trace


class RecoveryReport(BaseModel):
    """Clustering-vs-ground-truth agreement for one corpus."""

    n_traces: int
    n_clusters: int
    n_true_labels: int
    purity: float  # fraction of traces whose cluster's majority label is their own
    adjusted_rand_index: float  # chance-corrected pair agreement, 1.0 = identical partitions


def ground_truth_labels(traces: list[Trace]) -> list[str]:
    """Per-trace task-identity labels from benchmark metadata (`domain`/`task_id`).

    Traces without both keys get the label "unknown" — callers should check how many that is
    before trusting the report.
    """
    labels: list[str] = []
    for trace in traces:
        domain = trace.metadata.get("domain")
        task_id = trace.metadata.get("task_id")
        if isinstance(domain, str) and task_id is not None:
            labels.append(f"{domain}/{task_id}")
        else:
            labels.append("unknown")
    return labels


def recovery_report(cluster_labels: list[int], true_labels: list[str]) -> RecoveryReport:
    """Grade a clustering against ground-truth task identities (purity + ARI)."""
    if len(cluster_labels) != len(true_labels):
        raise ValueError(f"{len(cluster_labels)} cluster labels but {len(true_labels)} true labels")
    if not cluster_labels:
        raise ValueError("cannot grade an empty clustering")
    contingency = _contingency(cluster_labels, true_labels)
    n = len(cluster_labels)
    purity = float(contingency.max(axis=1).sum()) / n
    return RecoveryReport(
        n_traces=n,
        n_clusters=contingency.shape[0],
        n_true_labels=contingency.shape[1],
        purity=purity,
        adjusted_rand_index=_adjusted_rand_index(contingency),
    )


def _contingency(cluster_labels: list[int], true_labels: list[str]) -> np.ndarray:
    cluster_ids = sorted(set(cluster_labels))
    truth_ids = sorted(set(true_labels))
    cluster_index = {label: i for i, label in enumerate(cluster_ids)}
    truth_index = {label: i for i, label in enumerate(truth_ids)}
    table = np.zeros((len(cluster_ids), len(truth_ids)), dtype=np.int64)
    for cluster_label, true_label in zip(cluster_labels, true_labels, strict=True):
        table[cluster_index[cluster_label], truth_index[true_label]] += 1
    return table


def _adjusted_rand_index(contingency: np.ndarray) -> float:
    """ARI from a contingency table (Hubert & Arabie 1985)."""
    n = int(contingency.sum())
    sum_cells = float((_choose2(contingency)).sum())
    sum_rows = float(_choose2(contingency.sum(axis=1)).sum())
    sum_cols = float(_choose2(contingency.sum(axis=0)).sum())
    total_pairs = float(_choose2(np.asarray([n]))[0])
    if total_pairs == 0.0:
        return 0.0
    expected = sum_rows * sum_cols / total_pairs
    maximum = (sum_rows + sum_cols) / 2.0
    if maximum == expected:  # a single cluster and a single label: partitions are trivially equal
        return 1.0
    return (sum_cells - expected) / (maximum - expected)


def _choose2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return values * (values - 1.0) / 2.0
