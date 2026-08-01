"""Tests for the external-only SWE-rebench effort fitter."""

from __future__ import annotations

import coding_model_router_swerebench_fit as fit
import numpy as np
from coding_model_router_codeforces_fit import ARMS, Data


def _source() -> fit.SourceData:
    task_ids = [f"task-{index}" for index in range(10)]
    groups = [f"repo-{index // 2}" for index in range(10)]
    rewards = np.zeros((10, len(ARMS), fit.ATTEMPTS), dtype=np.float64)
    costs = np.empty_like(rewards)
    for arm in range(len(ARMS)):
        rewards[:, arm, :] = float(arm >= 2)
        costs[:, arm, :] = 0.01 * (arm + 1)
    data = Data(
        task_ids=task_ids,
        groups=groups,
        texts=[f"repository={groups[index]}\nfix bug {index}" for index in range(10)],
        structural=np.asarray(
            [[float(index), *([1.0] * 14)] for index in range(10)]
        ),
        rewards=rewards.mean(axis=2),
        costs=costs.mean(axis=2),
    )
    return fit.SourceData(
        data=data,
        raw_rewards=rewards,
        raw_costs=costs,
        languages=["Python"] * 10,
        repositories=groups,
    )


def test_candidate_grid_is_complete_and_has_no_similarity_floor() -> None:
    candidates = fit.candidate_grid()
    assert len(candidates) == 1_389
    assert len({candidate.key for candidate in candidates}) == len(candidates)
    knn = [candidate for candidate in candidates if candidate.family == "knn"]
    assert {candidate.rag_num for candidate in knn} == {8, 16, 32, 64}
    assert {candidate.z for candidate in knn} == {0.0, 0.5, 1.0, 1.645, 2.0}
    assert {candidate.pick_lam for candidate in knn} == {0.0, 0.01, 0.02, 0.03}
    assert {candidate.config()["rag_thres"] for candidate in knn} == {0.0}


def test_grouped_folds_have_zero_repository_overlap() -> None:
    source = _source()
    for train, test in fit._folds(source.data):
        assert set(np.asarray(source.data.groups)[train]).isdisjoint(
            set(np.asarray(source.data.groups)[test])
        )


def test_direct_full_fit_router_returns_a_frozen_effort() -> None:
    source = _source()
    candidate = fit.Candidate(
        family="direct",
        order=0,
        dim=512,
        alpha=1.0,
        threshold=0.02,
    )
    route = fit._fit_text_router(
        source,
        candidate,
        label_rewards=source.data.rewards,
    )
    choice = route(
        {
            "repository": "heldout/repo",
            "language": "Python",
            "prompt": "Fix the failing parser test.",
        }
    )
    assert 0 <= choice < len(ARMS)


def test_within_repository_permutation_never_crosses_groups() -> None:
    source = _source()
    labels = source.data.rewards.copy()
    for index in range(len(labels)):
        labels[index] = index
    permutable = fit.SourceData(
        data=Data(
            task_ids=source.data.task_ids,
            groups=source.data.groups,
            texts=source.data.texts,
            structural=source.data.structural,
            rewards=labels,
            costs=source.data.costs,
        ),
        raw_rewards=source.raw_rewards,
        raw_costs=source.raw_costs,
        languages=source.languages,
        repositories=source.repositories,
    )
    shuffled = fit._permuted_labels(permutable)
    for index, group in enumerate(source.data.groups):
        allowed = {
            float(member)
            for member, candidate_group in enumerate(source.data.groups)
            if candidate_group == group
        }
        assert float(shuffled[index, 0]) in allowed
