"""Fit traces as induction episodes, with starting facts and the recorded error convention."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.induction.episodes import fit_episodes


def _with_facts(index: int) -> JsonObject:
    """One retail episode whose package supplies a starting fact and an error reply."""
    row = episode(index)
    row["facts"] = {
        "schema_version": 1,
        "facts": [
            {
                "key": f"orders/#W{1000 + index}/status",
                "value": "pending",
                "visibility": "conditioning",
                "provenance": {
                    "adapter": "fixture-v1",
                    "source": "fixture",
                    "validation": {"status": "validated", "method": "fixture snapshot"},
                    "state_version": 0,
                },
            },
            {
                "key": "later",
                "value": 1,
                "visibility": "conditioning",
                "provenance": {
                    "adapter": "fixture-v1",
                    "source": "fixture",
                    "validation": {"status": "validated", "method": "fixture snapshot"},
                    "state_version": 2,
                },
            },
        ],
    }
    exchanges = row["exchanges"]
    assert isinstance(exchanges, list)
    exchanges.append(("get_order_details", {"order_id": "#W9"}, {"is_error": True, "error": "no"}))
    return row


def test_episodes_carry_every_exchange_starting_facts_and_error_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefix exchanges are kept, later-version facts are not, and error replies are flagged."""
    rows = [_with_facts(index) for index in range(USERS)]
    _, project, _ = build_project(tmp_path, monkeypatch, SimulatedProvider(rows), episodes=rows)
    episodes = fit_episodes(
        project,
        tools=("find_user_id_by_email", "get_order_details"),
        initial_state={"orders": {}},
        maximum_episodes=3,
    )
    assert len(episodes) == 3
    first = episodes[0]
    assert [step.tool for step in first.steps][0] == "find_user_id_by_email"
    assert len(first.steps) == 5 and first.steps[-1].is_error
    assert not first.steps[0].is_error
    assert list(first.facts) == [next(key for key in first.facts if key.startswith("orders/"))]
    assert first.initial_state == {"orders": {}}
    assert fit_episodes(project, tools=("unknown",), initial_state=None, maximum_episodes=3) == ()
