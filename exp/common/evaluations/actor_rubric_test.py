"""Actor judging has a separate provisional identity and the maintained wire contract."""

from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import default_judge, judge_wire_schema


def test_actor_rubric_is_distinct_and_uses_the_shared_evaluation_protocol() -> None:
    """A new authored rubric cannot inherit reconstruction calibration or criteria."""
    config = actor_judge("fixture")
    assert config.calibration == "provisional"
    assert config.definition.revision != default_judge("fixture").definition.revision
    wire = judge_wire_schema(config, 2)
    assert wire["type"] == "object"
    assert "unsupported" in config.prompt
    assert config.per_turn is True
