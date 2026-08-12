"""Tests for the closed-loop outcome matrix types (the routing optimizer's training data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.providers.base import ProviderKind, TokenUsage
from wmo.common.providers.pool import PoolEntry
from wmo.optimize.routing.outcomes import (
    OutcomeMatrix,
    ScenarioOutcome,
    split_router_scenarios,
    split_router_scenarios_grouped,
)


def _outcome(
    scenario_id: str, model: str, *, reward: float = 0.5, episode: int = 0
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        task="do the thing",
        model=model,
        episode=episode,
        reward=reward,
        success=reward >= 0.5,
        critique="ok",
        steps=3,
        stop_reason="agent_done",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost_usd=0.01,
        call_seconds=[0.2, 0.3, 0.25],
        replies=["{}", "{}", '{"done": true}'],
    )


def _matrix() -> OutcomeMatrix:
    entries = [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
    ]
    outcomes = [
        _outcome("s1", "fable-5", reward=0.9),
        _outcome("s2", "fable-5", reward=0.7),
        _outcome("s1", "haiku-4-5", reward=0.4),
        _outcome("s2", "haiku-4-5", reward=0.8),
    ]
    return OutcomeMatrix(pool=entries, outcomes=outcomes)


def test_matrix_accessors() -> None:
    matrix = _matrix()
    assert matrix.model_names() == ["fable-5", "haiku-4-5"]
    assert matrix.scenario_ids() == ["s1", "s2"]
    assert matrix.mean_reward("fable-5") == pytest.approx(0.8)
    assert matrix.mean_reward("haiku-4-5") == pytest.approx(0.6)
    assert [o.model for o in matrix.for_scenario("s1")] == ["fable-5", "haiku-4-5"]


def test_w05_outcome_row_fixture_preserves_evaluation_evidence() -> None:
    """Map current `OutcomeMatrix` and `ScenarioOutcome` to approved evaluation dataset rows.

    The row preserves task, candidate, repeat, reward, usage, cost, latency, stop reason, and
    replies. Current main has no evaluation protocol, judgment, or source-run identity fields.
    """
    row = ScenarioOutcome(
        scenario_id="scenario-w05-refund",
        task="Refund order A-42",
        model="haiku-4-5",
        episode=0,
        reward=0.75,
        success=True,
        critique="The order was refunded.",
        steps=1,
        stop_reason="agent_done",
        usage=TokenUsage(input_tokens=12, output_tokens=8),
        cost_usd=0.0012,
        call_seconds=[0.24],
        replies=['{"status":"refunded"}'],
    )
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
            PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
        ],
        outcomes=[row],
    )

    assert [(entry.name, entry.kind.value, entry.model) for entry in matrix.pool] == [
        ("fable-5", "anthropic", "claude-fable-5"),
        ("haiku-4-5", "anthropic", "claude-haiku-4-5"),
    ]
    assert row.model_dump(mode="json") == {
        "scenario_id": "scenario-w05-refund",
        "task": "Refund order A-42",
        "model": "haiku-4-5",
        "episode": 0,
        "reward": 0.75,
        "success": True,
        "critique": "The order was refunded.",
        "steps": 1,
        "stop_reason": "agent_done",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
        },
        "cost_usd": 0.0012,
        "call_seconds": [0.24],
        "replies": ['{"status":"refunded"}'],
        "error": None,
        "remeasured": False,
        "tokens_in_raw": 0,
        "tokens_in_compressed": 0,
        "compressor_id": "",
        "compressor_version": "",
        "aggressiveness": 0.0,
        "compressor_latency_s": 0.0,
        "compressor_cost_usd": 0.0,
    }
    assert OutcomeMatrix.model_validate_json(matrix.model_dump_json()) == matrix


def test_router_split_is_deterministic_disjoint_and_order_preserving() -> None:
    ids = [f"scenario-{index}" for index in range(10)]
    split = split_router_scenarios(ids)

    assert len(split.fit_ids) == 7
    assert len(split.report_ids) == 3
    assert set(split.fit_ids).isdisjoint(split.report_ids)
    assert [sid for sid in ids if sid in split.fit_ids] == list(split.fit_ids)
    reordered = list(reversed(ids))
    rerun = split_router_scenarios(reordered)
    assert set(rerun.fit_ids) == set(split.fit_ids)
    assert set(rerun.report_ids) == set(split.report_ids)


def test_router_split_refuses_a_claim_with_no_holdout() -> None:
    with pytest.raises(ValueError, match="at least 2 scenarios"):
        split_router_scenarios(["only-one"])


def test_matrix_round_trips_through_json(tmp_path: Path) -> None:
    matrix = _matrix()
    path = tmp_path / "outcomes.json"
    matrix.save(path)
    loaded = OutcomeMatrix.load(path)
    assert loaded == matrix


def test_mean_reward_unknown_model_errors() -> None:
    with pytest.raises(KeyError, match="fable-5"):
        _matrix().mean_reward("nope")


def test_outcomes_must_name_pool_models() -> None:
    # A ghost model used to reach the fitter and die on a bare KeyError; the matrix names it.
    matrix = _matrix()
    with pytest.raises(ValueError, match="ghost-model"):
        OutcomeMatrix(
            pool=matrix.pool,
            outcomes=[*matrix.outcomes, _outcome("s1", "ghost-model")],
        )


def test_measured_compression_reads_the_arm_off_the_scored_rows() -> None:
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(
                scenario_id="s1",
                task="t",
                model="a",
                reward=1.0,
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            )
        ],
    )
    config = matrix.measured_compression()
    assert config is not None
    assert config.compressor_id == "truncate"
    assert config.aggressiveness == 0.5


def test_a_matrix_with_no_compression_fields_reads_as_the_uncompressed_arm() -> None:
    # Every matrix captured before D-COMPRESS existed, which must keep fitting exactly as before.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0)],
    )
    assert matrix.measured_compression() is None


def test_a_matrix_that_mixes_arms_refuses_to_name_one() -> None:
    # Two arms in one file: the rows are not comparable, so no single policy can be fitted
    # from them and picking a winner here would hide that.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0),
            ScenarioOutcome(
                scenario_id="s2",
                task="t",
                model="a",
                reward=1.0,
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            ),
        ],
    )
    with pytest.raises(ValueError, match="mixes compression configs"):
        matrix.measured_compression()


def test_an_unscored_row_does_not_decide_the_arm() -> None:
    # An unscored episode produced no reward, so whatever it ran under cannot bias a fit.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0),
            ScenarioOutcome(
                scenario_id="s2",
                task="t",
                model="a",
                reward=None,
                error="provider throttled",
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            ),
        ],
    )
    assert matrix.measured_compression() is None


# --- grouped router split ------------------------------------------------------------------
def test_grouped_split_never_lets_a_group_straddle_fit_and_report() -> None:
    ids = [f"repo{i % 5}::t{i}" for i in range(20)]
    groups = {sid: sid.split("::")[0] for sid in ids}
    split = split_router_scenarios_grouped(ids, groups)
    fit_groups = {groups[sid] for sid in split.fit_ids}
    report_groups = {groups[sid] for sid in split.report_ids}
    assert fit_groups.isdisjoint(report_groups)
    assert split.fit_ids and split.report_ids
    assert sorted(split.fit_ids + split.report_ids) == sorted(ids)


def test_grouped_split_is_deterministic_and_order_preserving() -> None:
    ids = [f"g{i % 7}-{i}" for i in range(28)]
    groups = {sid: sid.split("-")[0] for sid in ids}
    first = split_router_scenarios_grouped(ids, groups)
    again = split_router_scenarios_grouped(list(reversed(ids)), groups)
    # Same membership regardless of row order; each call preserves ITS caller's order.
    assert set(first.fit_ids) == set(again.fit_ids)
    assert list(first.fit_ids) == [sid for sid in ids if sid in set(first.fit_ids)]
    assert list(again.fit_ids) == [sid for sid in reversed(ids) if sid in set(first.fit_ids)]


def test_grouped_split_targets_the_fit_fraction_by_scenario_count() -> None:
    ids = [f"g{i % 10}-{i}" for i in range(100)]
    groups = {sid: sid.split("-")[0] for sid in ids}
    split = split_router_scenarios_grouped(ids, groups)
    # Groups are 10 scenarios each, so the fit side lands exactly on the 70% target here.
    assert len(split.fit_ids) == 70


def test_grouped_split_salt_changes_membership_deterministically() -> None:
    ids = [f"g{i % 6}-{i}" for i in range(30)]
    groups = {sid: sid.split("-")[0] for sid in ids}
    plain = split_router_scenarios_grouped(ids, groups)
    salted = [split_router_scenarios_grouped(ids, groups, salt=str(s)) for s in range(4)]
    assert all(
        set(one.fit_ids) == set(two.fit_ids)
        for one, two in zip(
            salted,
            [split_router_scenarios_grouped(ids, groups, salt=str(s)) for s in range(4)],
            strict=True,
        )
    )
    # At least one salt must move the partition, or salting buys no independent splits.
    assert any(set(one.fit_ids) != set(plain.fit_ids) for one in salted)


def test_grouped_split_keeps_a_report_side_even_when_one_group_dominates() -> None:
    ids = [f"big-{i}" for i in range(30)] + ["small-a", "small-b"]
    groups = {sid: sid.split("-")[0] for sid in ids}
    split = split_router_scenarios_grouped(ids, groups)
    assert split.fit_ids and split.report_ids
    fit_groups = {groups[sid] for sid in split.fit_ids}
    assert fit_groups.isdisjoint({groups[sid] for sid in split.report_ids})


def test_grouped_split_requires_a_group_for_every_scenario() -> None:
    with pytest.raises(ValueError, match="no group"):
        split_router_scenarios_grouped(["a", "b"], {"a": "g1"})


def test_grouped_split_refuses_a_single_group() -> None:
    with pytest.raises(ValueError, match="at least 2 groups"):
        split_router_scenarios_grouped(["a", "b"], {"a": "g", "b": "g"})


def test_grouped_split_refuses_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        split_router_scenarios_grouped(["a", "a"], {"a": "g"})
