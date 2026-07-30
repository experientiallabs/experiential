"""Tests for the teacher-search gate.

Pure offline arithmetic over synthetic outcome matrices: no provider, no judge, no spend. The two
shapes that matter are the ones this gate was written from, tau-bench's (peer models, no gap) and
TerminalBench-2's (+27 points, a real gap), so both are reproduced here as fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.teacher import select_teacher
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry, Tier


def _entry(name: str, *, per_mtok: float, tier: Tier = "open") -> PoolEntry:
    """A priced pool entry. `per_mtok` sets both rates, so the list ladder is just that number."""
    return PoolEntry(
        name=name,
        kind=ProviderKind.OPENAI,
        model=f"{name}-runtime",
        tier=tier,
        input_per_mtok=per_mtok,
        output_per_mtok=per_mtok,
    )


def _rows(
    model: str, rewards: Mapping[str, float | None], *, cost: float = 0.01, tokens: int = 100
) -> list[ScenarioOutcome]:
    """One episode per scenario for `model`; a None reward is an unscored (failed) episode."""
    return [
        ScenarioOutcome(
            scenario_id=sid,
            task=f"task for {sid}",
            model=model,
            reward=reward,
            success=reward is not None and reward >= 0.5,
            usage=TokenUsage(input_tokens=tokens, output_tokens=tokens // 2),
            cost_usd=cost,
            error=None if reward is not None else "sandbox timeout",
        )
        for sid, reward in rewards.items()
    ]


def _scenarios(n: int) -> list[str]:
    return [f"s{i:02d}" for i in range(n)]


def _flat(rewards: float, n: int) -> dict[str, float | None]:
    return dict.fromkeys(_scenarios(n), rewards)


def _tau_matrix() -> OutcomeMatrix:
    """tau-bench's measured shape: the 27B is +1.6 points and K3 is well BELOW the 9B student.

    (The measured K3 row was -11.7 points; this fixture puts it at -12.5, which is the same
    verdict for the same reason and keeps the rewards to two decimals.)

    Rewards alternate around each model's mean so the paired differences carry real spread, which
    is what a matrix from a live sweep looks like and what the interval is estimated from.
    """
    ids = _scenarios(20)
    student = {sid: 0.60 if i % 2 else 0.50 for i, sid in enumerate(ids)}
    peer = {sid: student[sid] + (-0.18 if i < 6 else 0.10) for i, sid in enumerate(ids)}
    weak = {sid: student[sid] - (0.20 if i % 2 else 0.05) for i, sid in enumerate(ids)}
    return OutcomeMatrix(
        pool=[
            _entry("qwen3-9b", per_mtok=0.10),
            _entry("qwen3.6-27b", per_mtok=1.15),
            _entry("kimi-k3", per_mtok=9.00),
        ],
        outcomes=[
            *_rows("qwen3-9b", student, cost=0.002),
            *_rows("qwen3.6-27b", peer, cost=0.020),
            *_rows("kimi-k3", weak, cost=0.200),
        ],
    )


def _tb2_matrix(*, n: int = 12) -> OutcomeMatrix:
    """TerminalBench-2's measured shape: a real gap over the 9B, and two models that clear it.

    The 27B is +27 points at $1.15/Mtok list; K3 is +30 points at $9.00. K3 has the bigger gain
    and the 27B keeps 90% of it, so at the default 0.8 sufficiency the cheap model is the teacher.
    """
    ids = _scenarios(n)
    student = {sid: 0.20 if i % 2 else 0.24 for i, sid in enumerate(ids)}
    teacher = {
        sid: value + (0.30 if i % 2 else 0.24) for i, (sid, value) in enumerate(student.items())
    }
    frontier = {
        sid: value + (0.34 if i % 2 else 0.26) for i, (sid, value) in enumerate(student.items())
    }
    return OutcomeMatrix(
        pool=[
            _entry("qwen3-9b", per_mtok=0.10),
            _entry("qwen3.6-27b", per_mtok=1.15),
            _entry("kimi-k3", per_mtok=9.00),
        ],
        outcomes=[
            *_rows("qwen3-9b", student, cost=0.002),
            *_rows("qwen3.6-27b", teacher, cost=0.020),
            *_rows("kimi-k3", frontier, cost=0.200),
        ],
    )


def test_tau_shape_finds_no_teacher_and_refuses_to_distill() -> None:
    """The measured tau verdict: peers, not teachers, so the training run is never authorized."""
    verdict = select_teacher(_tau_matrix())

    assert verdict.decision == "do_not_distill"
    assert verdict.student == "qwen3-9b"  # the cheapest measured model, unnamed by the caller
    assert verdict.teacher is None
    assert not verdict.should_distill
    leader = verdict.gains[0]
    assert leader.model == "qwen3.6-27b"
    assert leader.mean_gain == pytest.approx(0.016, abs=5e-4)  # the measured +1.6 points
    assert not leader.clears_gap
    # K3 below the student is a NEGATIVE gain, not a missing row: the table has to show it.
    assert verdict.gains[-1].model == "kimi-k3"
    assert verdict.gains[-1].mean_gain < 0.0


def test_tb2_shape_distills_and_picks_the_cheapest_sufficient_teacher() -> None:
    """The measured TB2 verdict: a gap exists, and the $1.15 model teaches it, not the $9.00 one."""
    verdict = select_teacher(_tb2_matrix())

    assert verdict.decision == "distill"
    assert verdict.student == "qwen3-9b"
    assert verdict.teacher == "qwen3.6-27b"
    assert [row.model for row in verdict.gains] == ["kimi-k3", "qwen3.6-27b"]
    best, chosen = verdict.gains[0], verdict.gains[1]
    assert best.model == "kimi-k3"  # the biggest gain is NOT the teacher
    assert best.clears_gap and chosen.clears_gap
    assert chosen.mean_gain == pytest.approx(0.27)


def test_a_bigger_gain_wins_when_the_cheap_model_falls_below_sufficiency() -> None:
    """Sufficiency is a real bar: a cheap model that keeps too little of the gain is not enough."""
    verdict = select_teacher(_tb2_matrix(), sufficiency=0.95)

    assert verdict.decision == "distill"
    assert verdict.teacher == "kimi-k3"  # 0.27 / 0.30 = 90%, under the 95% asked for


def test_a_gain_whose_interval_includes_zero_is_not_a_gap() -> None:
    """A large point gain over wildly inconsistent scenarios does not authorize spending."""
    ids = _scenarios(10)
    student = _flat(0.40, 10)
    noisy = {sid: (1.00 if i % 2 else 0.10) for i, sid in enumerate(ids)}
    matrix = OutcomeMatrix(
        pool=[_entry("small", per_mtok=0.10), _entry("erratic", per_mtok=1.15)],
        outcomes=[*_rows("small", student), *_rows("erratic", noisy)],
    )

    verdict = select_teacher(matrix)

    row = verdict.gains[0]
    assert row.mean_gain == pytest.approx(0.15)  # a point gain half again over the bar
    assert row.ci_low is not None and row.ci_high is not None
    assert row.ci_low < 0.0 < row.ci_high
    assert not row.clears_gap
    assert verdict.decision == "do_not_distill"
    assert "interval includes zero" in verdict.reason


def test_a_thin_matrix_says_insufficient_rather_than_yes() -> None:
    """Three scenarios cannot authorize a training run, however large the apparent gain."""
    verdict = select_teacher(_tb2_matrix(n=3))

    assert verdict.decision == "insufficient_evidence"
    assert verdict.teacher is None
    assert "below the 8 this gate requires" in verdict.reason
    assert verdict.gains[0].n_scenarios == 3


def test_a_thin_matrix_is_still_decidable_when_the_caller_lowers_the_bar() -> None:
    """`min_scenarios` is the caller's, so a deliberately small probe can still return a verdict."""
    verdict = select_teacher(_tb2_matrix(n=3), min_scenarios=3)

    assert verdict.decision == "distill"
    assert verdict.teacher == "qwen3.6-27b"


def test_no_shared_scenario_is_insufficient_evidence() -> None:
    """Two models measured on disjoint scenarios were never compared, whatever their means say."""
    matrix = OutcomeMatrix(
        pool=[_entry("small", per_mtok=0.10), _entry("big", per_mtok=1.15)],
        outcomes=[
            *_rows("small", dict.fromkeys(_scenarios(4), 0.2)),
            *_rows("big", {f"other-{i}": 0.9 for i in range(4)}),
        ],
    )

    verdict = select_teacher(matrix)

    assert verdict.decision == "insufficient_evidence"
    assert verdict.gains == []
    assert verdict.unmeasured_models == ["big"]
    assert "shares a scored scenario" in verdict.reason


def test_unscored_episodes_are_excluded_from_both_sides() -> None:
    """An infrastructure failure is not a reward of 0, so it narrows the pairing instead."""
    ids = _scenarios(10)
    student = _flat(0.30, 10)
    teacher: dict[str, float | None] = {sid: (None if i < 3 else 0.90) for i, sid in enumerate(ids)}
    matrix = OutcomeMatrix(
        pool=[_entry("small", per_mtok=0.10), _entry("big", per_mtok=1.15)],
        outcomes=[*_rows("small", student), *_rows("big", teacher)],
    )

    verdict = select_teacher(matrix)

    row = verdict.gains[0]
    assert row.n_scenarios == 7  # the three unscored episodes are not zeros
    assert row.mean_gain == pytest.approx(0.60)
    assert row.student_mean_reward == pytest.approx(0.30)
    assert verdict.n_scenarios == 10  # the student itself was scored on all ten


def test_unpriced_rows_fall_back_to_the_list_price_ladder() -> None:
    """A matrix whose episodes recorded $0 still has a ladder: the pool's published prices."""
    ids = _scenarios(10)
    student = _flat(0.20, 10)
    cheap = {sid: 0.60 for sid in ids}
    dear = {sid: 0.62 for sid in ids}
    matrix = OutcomeMatrix(
        pool=[
            _entry("self-hosted-small", per_mtok=0.10),
            _entry("cheap-teacher", per_mtok=1.15),
            _entry("dear-teacher", per_mtok=9.00),
        ],
        outcomes=[
            *_rows("self-hosted-small", student, cost=0.0),
            *_rows("cheap-teacher", cheap, cost=0.0),
            *_rows("dear-teacher", dear, cost=0.0),
        ],
    )

    verdict = select_teacher(matrix)

    assert verdict.price_basis == "list"
    assert verdict.student == "self-hosted-small"
    assert verdict.teacher == "cheap-teacher"  # 0.40 keeps 97% of the 0.42 best gain
    assert verdict.gains[0].price == pytest.approx(18.0)  # the $9.00/$9.00 entry, summed


def test_the_measured_ladder_is_used_when_every_model_completed_paid_tasks() -> None:
    """With real spend and real completions on the rows, the ladder is dollars per completed task.

    Note what that requires, and why the TB2 fixture above does NOT get there: cost per completed
    task is undefined for a model that completed nothing, so a workload the student fails outright
    is priced from the pool's list rates instead.
    """
    ids = _scenarios(10)
    matrix = OutcomeMatrix(
        pool=[
            # List order would put `dear-per-mtok` last; its measured cost per completed task
            # puts it second, so this also pins WHICH ladder decided the teacher.
            _entry("small", per_mtok=0.10),
            _entry("dear-per-mtok", per_mtok=9.00),
            _entry("cheap-per-mtok", per_mtok=1.15),
        ],
        outcomes=[
            *_rows("small", dict.fromkeys(ids, 0.55), cost=0.002),
            *_rows("dear-per-mtok", dict.fromkeys(ids, 0.95), cost=0.010),
            *_rows("cheap-per-mtok", dict.fromkeys(ids, 0.94), cost=0.050),
        ],
    )

    verdict = select_teacher(matrix)

    assert verdict.price_basis == "measured"
    assert verdict.student == "small"
    assert verdict.teacher == "dear-per-mtok"  # $0.010 a completed task against $0.050
    chosen = next(row for row in verdict.gains if row.model == verdict.teacher)
    assert chosen.price == pytest.approx(0.010)


def test_an_open_teacher_is_preferred_over_a_cheaper_frontier_one() -> None:
    """The existence proof may be frontier; the model trained ON should be one you may train on."""
    ids = _scenarios(10)
    student = _flat(0.20, 10)
    matrix = OutcomeMatrix(
        pool=[
            _entry("small", per_mtok=0.10),
            _entry("frontier-cheap", per_mtok=1.00, tier="frontier"),
            _entry("open-dearer", per_mtok=2.00, tier="open"),
        ],
        outcomes=[
            *_rows("small", student, cost=0.002),
            *_rows("frontier-cheap", dict.fromkeys(ids, 0.60), cost=0.010),
            *_rows("open-dearer", dict.fromkeys(ids, 0.58), cost=0.020),
        ],
    )

    verdict = select_teacher(matrix)

    assert verdict.decision == "distill"
    assert verdict.teacher == "open-dearer"  # dearer, but 0.38 keeps 95% of the 0.40 best gain
    assert "open weights" in verdict.reason


def test_a_frontier_teacher_is_chosen_when_no_open_model_is_sufficient() -> None:
    """And the reason says so, so the licensing question lands before the training spend."""
    ids = _scenarios(10)
    student = _flat(0.20, 10)
    matrix = OutcomeMatrix(
        pool=[
            _entry("small", per_mtok=0.10),
            _entry("frontier-strong", per_mtok=9.00, tier="frontier"),
            _entry("open-weak", per_mtok=1.00, tier="open"),
        ],
        outcomes=[
            *_rows("small", student, cost=0.002),
            *_rows("frontier-strong", dict.fromkeys(ids, 0.90), cost=0.200),
            *_rows("open-weak", dict.fromkeys(ids, 0.35), cost=0.010),
        ],
    )

    verdict = select_teacher(matrix)

    assert verdict.teacher == "frontier-strong"  # open-weak keeps 21% of the gain, under 80%
    assert "check that provider's terms" in verdict.reason


def test_the_student_can_be_named_when_the_cheapest_model_is_not_the_one_being_trained() -> None:
    """A distillation trains a specific model; the default is a convenience, not a constraint."""
    verdict = select_teacher(_tb2_matrix(), student="qwen3.6-27b")

    assert verdict.student == "qwen3.6-27b"
    assert verdict.decision == "do_not_distill"  # K3 is only +3 points over the 27B
    assert [row.model for row in verdict.gains] == ["kimi-k3", "qwen3-9b"]


def test_naming_a_model_the_matrix_never_scored_says_which_ones_it_did() -> None:
    matrix = _tb2_matrix()
    with pytest.raises(ValueError, match="no pool model named 'ghost'"):
        select_teacher(matrix, student="ghost")


def test_a_matrix_with_one_measured_model_cannot_be_probed() -> None:
    """A gap is a comparison: refuse rather than report 'no teacher' from a pool of one."""
    matrix = OutcomeMatrix(
        pool=[_entry("small", per_mtok=0.10), _entry("big", per_mtok=1.15)],
        outcomes=[
            *_rows("small", _flat(0.3, 4)),
            *_rows("big", dict.fromkeys(_scenarios(4), None)),
        ],
    )

    with pytest.raises(ValueError, match="at least two models with scored episodes"):
        select_teacher(matrix)


def test_thresholds_that_would_make_the_verdict_meaningless_are_rejected() -> None:
    matrix = _tb2_matrix()
    with pytest.raises(ValueError, match="min_gap must be a positive fraction"):
        select_teacher(matrix, min_gap=0.0)
    with pytest.raises(ValueError, match="min_scenarios must be at least 2"):
        select_teacher(matrix, min_scenarios=1)
    with pytest.raises(ValueError, match="sufficiency must be in"):
        select_teacher(matrix, sufficiency=1.5)


def test_the_do_not_distill_reason_reads_as_product_copy() -> None:
    """The reason string is printed verbatim by the CLI and by the orchestrator, so pin it."""
    verdict = select_teacher(_tau_matrix())

    assert verdict.reason == (
        "DO NOT DISTILL: no model in this matrix beats 'qwen3-9b' by the 10.0 points this gate "
        "requires. The best is 'qwen3.6-27b' at +1.6 points on 20 shared scenarios (95% CI -4.2 "
        "to +7.4 points). 'qwen3-9b' already serves this workload and earns its traffic on "
        "price, so training has no teacher to learn from and this stage skips with zero spend."
    )


def test_the_distill_reason_names_the_gap_the_teacher_and_the_price() -> None:
    verdict = select_teacher(_tb2_matrix())

    assert verdict.reason == (
        "DISTILL: 'kimi-k3' beats 'qwen3-9b' by +30.0 points on 12 shared scenarios (95% CI "
        "+27.6 to +32.4 points), clearing the 10.0-point bar this gate requires, so there is "
        "something to teach. The cheapest sufficient teacher is 'qwen3.6-27b' at $2.30 per 1M "
        "tokens (list, input + output), keeping 90% of the best measured gain (the bar is 80%). "
        "It is open weights, which is the tier this gate prefers for a data teacher: a frontier "
        "model can prove the gap exists, but you have to be allowed to train on what the teacher "
        "writes."
    )


def test_price_tie_break_judges_candidates_over_their_shared_scenarios_only() -> None:
    """An easy private subset must not crown the weaker of two equally cheap students.

    `flattered` is scored on the shared band at 0.4 plus a private band of gimmes at 1.0, so
    its unpaired mean (0.7) beats `honest`'s (0.6) - but on the scenarios BOTH were scored on,
    `honest` is plainly better. The tie-break must read the shared band and pick `honest`;
    the old unpaired mean picked `flattered` (the review finding this test pins).
    """
    shared = {f"s{i:02d}": None for i in range(8)}
    # flattered: 0.45 on the 8 shared scenarios (below the 0.5 completion bar) plus 8 private
    # gimmes at 1.0 -> unpaired mean 0.725, 8 completions over 16 cheap rows = $0.02/task.
    flattered_rewards: dict[str, float | None] = {sid: 0.45 for sid in shared}
    flattered_rewards |= {f"easy{i}": 1.0 for i in range(8)}
    # honest: 0.6 on the same 8 shared scenarios -> unpaired mean 0.6 (LOWER than flattered's),
    # 8 completions over 8 rows at $0.02 = the same $0.02/task, so the two students tie on the
    # measured ladder and only the tie-break separates them.
    honest_rewards: dict[str, float | None] = {sid: 0.6 for sid in shared}
    teacher_rewards = dict.fromkeys(list(shared) + [f"easy{i}" for i in range(8)], 0.95)

    matrix = OutcomeMatrix(
        pool=[
            _entry("flattered", per_mtok=1.0),
            _entry("honest", per_mtok=1.0),
            _entry("teacher", per_mtok=9.0),
        ],
        outcomes=(
            _rows("flattered", flattered_rewards, cost=0.01)
            + _rows("honest", honest_rewards, cost=0.02)
            + _rows("teacher", teacher_rewards, cost=0.05)
        ),
    )
    verdict = select_teacher(matrix)
    assert verdict.student == "honest"
