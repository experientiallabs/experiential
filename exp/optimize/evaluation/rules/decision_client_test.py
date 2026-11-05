"""Shared System One factory rejects a transport without operation accounting."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.rules.decision_client import rule_memory_decisions
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings
from exp.optimize.evaluation.rules.rule_tags import RuleTaggingSettings
from exp.optimize.evaluation.rules.stage import RulesProfile
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


@pytest.mark.parametrize("profile", ["jev_memory", "trajectory_tags"])
def test_system_one_refuses_unaccounted_transport_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: RulesProfile
) -> None:
    """Even a correctly configured native alias cannot buy through a naked transport."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    assert spec.bootstrap.rules is not None
    stage = spec.bootstrap.rules.model_copy(
        update={
            "profile": profile,
            "memory": MemorySetSettings(decision_alias="jev"),
            "tagging": RuleTaggingSettings(decision_alias="jev"),
        }
    )
    wire = ScriptedJsonTransport()
    transport = AdmittedJsonTransport(
        wire,
        ledger=DispatchLedger(tmp_path / "dispatch.sqlite", 1, sha256_json("test")),
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="operation spend ceiling"):
        rule_memory_decisions(project, stage, transport)
    assert not wire.requests
