"""Deterministic exact traces for bounded canonical model-reference traversal."""

import pytest

from exp.common.models.gateway_chains import (
    GatewayDeploymentRung as D,
)
from exp.common.models.gateway_chains import (
    GatewayModelChain,
    ModelChainConfigurationError,
    expand_model_chain,
)
from exp.common.models.gateway_chains import (
    GatewayModelReferenceRung as M,
)


def chain(model: str, *rungs: str) -> GatewayModelChain:
    """Build an authored chain using arrows only inside readable test fixtures."""
    return GatewayModelChain(
        model_id=model,
        pool_id=f"pool-{model}",
        revision="rev1",
        rungs=tuple(M(model_id=r[1:]) if r.startswith(">") else D(deployment_id=r) for r in rungs),
    )


def test_reciprocal_preserves_child_and_parent_suffix_trace() -> None:
    """A reciprocal edge is a visible no-op, not an execution loop."""
    result = expand_model_chain(
        "a", {"a": chain("a", "a1", ">b", "a2"), "b": chain("b", "b1", ">a", "b2")}
    )
    assert [s.deployment_ids for s in result.segments] == [("a1",), ("b1",), ("b2",), ("a2",)]
    assert [(e.reason, e.model_id, e.cursor) for e in result.events] == [
        ("model_entered", "a", 0),
        ("model_entered", "b", 1),
        ("model_reference_already_visited", "a", 2),
        ("traversal_exhausted", "a", 4),
    ]
    assert result.segments[2].ancestry == ("a", "b")
    assert result.segments[3].rung_positions == (2,)


@pytest.mark.parametrize("length", [1, 2, 3, 16])
def test_cycles_of_every_supported_length_enter_each_model_once(length: int) -> None:
    """Self and long cycles terminate by canonical identity, not depth truncation."""
    chains = {f"m{i}": chain(f"m{i}", f"d{i}", f">m{(i + 1) % length}") for i in range(length)}
    result = expand_model_chain("m0", chains)
    assert result.visited_model_ids == tuple(f"m{i}" for i in range(length))
    assert result.examined_rungs == length * 2
    assert result.events[-2].reason == "model_reference_already_visited"


def test_bounds_are_explicit_errors_not_truncation() -> None:
    """Distinct canonical-model and examined-rung limits are independently enforced."""
    chains = {f"m{i}": chain(f"m{i}", f"d{i}", f">m{(i + 1) % 17}") for i in range(17)}
    with pytest.raises(ModelChainConfigurationError, match="16"):
        expand_model_chain("m0", chains)
    with pytest.raises(ModelChainConfigurationError, match="256"):
        expand_model_chain("a", {"a": chain("a", *(f"d{i}" for i in range(257)))})
    assert (
        expand_model_chain("a", {"a": chain("a", *(f"d{i}" for i in range(256)))}).examined_rungs
        == 256
    )


@pytest.mark.parametrize("length", [16, 17])
def test_unavailable_terminal_still_counts_toward_canonical_bound(length: int) -> None:
    """An empty unavailable child is visited even though it cannot emit a physical stage."""
    chains = {f"m{i}": chain(f"m{i}", f"d{i}", f">m{i + 1}") for i in range(length - 1)}
    model = f"m{length - 1}"
    chains[model] = GatewayModelChain(
        model_id=model, pool_id="absent", revision="r", available=False
    )
    if length == 17:
        with pytest.raises(ModelChainConfigurationError, match="16 canonical"):
            expand_model_chain("m0", chains)
    else:
        result = expand_model_chain("m0", chains)
        assert result.visited_model_ids == tuple(f"m{i}" for i in range(length))
        assert result.events[-2].reason == "model_unavailable"
        assert len(result.segments) == length - 1


@pytest.mark.parametrize("direct_count", [255, 256])
def test_unavailable_reference_still_spends_examined_rung_budget(direct_count: int) -> None:
    """The reference itself counts even when its empty target contributes no leaves."""
    root = chain("a", *(f"d{i}" for i in range(direct_count)), ">b")
    child = GatewayModelChain(model_id="b", pool_id="absent", revision="r", available=False)
    if direct_count == 256:
        with pytest.raises(ModelChainConfigurationError, match="256"):
            expand_model_chain("a", {"a": root, "b": child})
    else:
        result = expand_model_chain("a", {"a": root, "b": child})
        assert result.examined_rungs == 256
        assert result.visited_model_ids == ("a", "b")


def test_invalid_chain_and_unavailable_override_fail_closed() -> None:
    """First actual rung and one-reference rules hold even before expansion."""
    with pytest.raises(ValueError, match="first"):
        chain("a", ">a")
    with pytest.raises(ValueError, match="one model"):
        chain("a", "a1", ">b", ">c")
    with pytest.raises(ModelChainConfigurationError, match="unavailable"):
        expand_model_chain("a", {"a": chain("a", "a1", ">missing")})
    unavailable = GatewayModelChain(model_id="a", pool_id="pool-a", revision="r", available=False)
    result = expand_model_chain("a", {"a": unavailable})
    assert not result.segments
    assert [event.reason for event in result.events] == [
        "model_entered",
        "model_unavailable",
        "traversal_exhausted",
    ]
