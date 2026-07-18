import pytest

from wmh.serving.public_play import (
    PublicPlayLimiter,
    PublicPlayLimitError,
    PublicPlayLimits,
)


def test_charges_steps_against_global_ceiling() -> None:
    limiter = PublicPlayLimiter(PublicPlayLimits(max_total_steps=2, max_steps_per_session=10))
    limiter.open_session("m", "s1")
    limiter.charge_step("s1")
    limiter.charge_step("s1")
    with pytest.raises(PublicPlayLimitError):
        limiter.charge_step("s1")
    assert limiter.snapshot()["total_steps"] == 2


def test_per_session_step_cap_is_independent() -> None:
    limiter = PublicPlayLimiter(PublicPlayLimits(max_total_steps=100, max_steps_per_session=1))
    limiter.open_session("m", "s1")
    limiter.open_session("m", "s2")
    limiter.charge_step("s1")
    with pytest.raises(PublicPlayLimitError):
        limiter.charge_step("s1")
    # A different session still has its own budget.
    limiter.charge_step("s2")


def test_session_ceiling() -> None:
    limiter = PublicPlayLimiter(PublicPlayLimits(max_sessions=1))
    limiter.open_session("m", "s1")
    with pytest.raises(PublicPlayLimitError):
        limiter.open_session("m", "s2")


def test_unknown_session_is_refused() -> None:
    limiter = PublicPlayLimiter()
    with pytest.raises(PublicPlayLimitError):
        limiter.charge_step("nope")


def test_model_for_session_roundtrips() -> None:
    limiter = PublicPlayLimiter()
    limiter.open_session("tau-bench", "s1")
    assert limiter.model_for_session("s1") == "tau-bench"
    assert limiter.model_for_session("missing") is None
