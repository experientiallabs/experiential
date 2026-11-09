"""Bootstrap contracts are exercised by the real scripted flows in bootstrap_test.py."""


def test_a_long_source_trace_opens_clipped_with_an_inventory_count() -> None:
    """A 300-step session with large outputs opens small; the full trace stays readable."""
    import json

    from exp.optimize.evaluation.variance.bootstrap_agents import (
        OPENING_STEPS_HEAD,
        OPENING_STEPS_TAIL,
        _inventory_summary,
        _opening_view,
    )

    view = {
        "trace_id": "t",
        "steps": [
            {"step": i, "action": {"name": "bash", "arguments": {"cmd": "x" * 5000}}}
            for i in range(300)
        ],
    }
    opened = _opening_view(view)
    assert len(opened["steps"]) == OPENING_STEPS_HEAD + OPENING_STEPS_TAIL + 1
    assert opened["steps"][OPENING_STEPS_HEAD]["omitted_steps"] == 240
    assert len(json.dumps(opened)) < 200_000
    summary = _inventory_summary(frozenset({"trace:t", "trace:t:span:1", "rule:r1"}))
    assert summary["counts"] == {"rule": 1, "trace": 2}
