"""Coverage ownership rules and validated input domains."""

import pytest

from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.induction.contracts import (
    InducedCoverage,
    ProgramDomain,
    ProgramOutcome,
    TransitionProgram,
)

MANIFEST = WorldManifest(
    domains=(StateDomain(name="a", owner="simulated"), StateDomain(name="b", owner="simulated")),
    tools=(
        ToolEffectPolicy(name="write_a", reads=("a",), writes=("a",)),
        ToolEffectPolicy(name="copy", reads=("a",), writes=("b",)),
    ),
)
DOMAIN = ProgramDomain(argument_keys=(("id",),), lines=(2, 3))
PROGRAM = TransitionProgram(
    tool="write_a", source="def transition(state, args, facts):\n    return 1\n"
)


def test_coverage_refuses_a_partition_where_a_tool_spans_owners() -> None:
    """Owning ``a`` while ``copy`` is predicted would let copy span induced and predicted."""
    with pytest.raises(ValueError, match="spans induced and predicted"):
        InducedCoverage(
            manifest=MANIFEST,
            programs=(PROGRAM,),
            domains={"write_a": DOMAIN},
            induced_tools=("write_a",),
            induced_domains=("a",),
        )


def test_coverage_needs_one_program_and_domain_per_induced_tool() -> None:
    """A tool without its program or validated domain cannot be induced."""
    with pytest.raises(ValueError, match="one accepted program"):
        InducedCoverage(manifest=MANIFEST, induced_tools=("write_a",), induced_domains=("a", "b"))
    empty = InducedCoverage(manifest=MANIFEST)
    with pytest.raises(KeyError):
        empty.program("write_a")


def test_a_domain_admits_only_validated_argument_names_and_lines() -> None:
    """New arguments, new code paths and non-replies are all outside the domain."""
    reply = ProgramOutcome(status="reply", reply=1, state={}, lines=(2,))
    assert DOMAIN.admits({"id": 1}, reply)
    assert not DOMAIN.admits({"id": 1, "extra": 2}, reply)
    assert not DOMAIN.admits({"id": 1}, reply.model_copy(update={"lines": (2, 4)}))
    assert not DOMAIN.admits({"id": 1}, ProgramOutcome(status="out_of_domain", lines=(2,)))
