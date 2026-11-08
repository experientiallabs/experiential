"""Mostly covered end to end by ``pipeline_test.py``; the source cap is pinned here."""

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance.pipeline_inputs import WORLD_SOURCE_LIMIT, world_sources
from exp.optimize.evaluation.variance.pipeline_spec import BuildOutputs


def _ref(name: str) -> ArtifactInput:
    return ArtifactInput(artifact_id=name, sha256="0" * 64)


def test_a_thousand_trace_family_names_a_bounded_source_list_led_by_the_start() -> None:
    """A c00-sized family (1,039 world-learning traces) must not put every trace in the packet.

    Naming all of them (twice, with the citable locators) outgrows the authors' input limit, so
    the start's own source traces lead and the list stops at the cap.
    """
    traces = tuple(f"t{index:04d}" for index in range(1039))
    build = BuildOutputs(
        report=_ref("r"),
        exposure=_ref("e"),
        world_model=_ref("w"),
        trace_dataset=_ref("d"),
        rules_traces=traces,
    )
    sources = world_sources(build, ("t0900", "not-world-learning", "t0900", "t0005"))
    assert len(sources) == WORLD_SOURCE_LIMIT
    assert [row.trace_id for row in sources[:3]] == ["t0900", "t0005", "t0000"]
    assert len({row.source_id for row in sources}) == WORLD_SOURCE_LIMIT
