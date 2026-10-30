"""Measure later predictions exposed to claims without turning that association into truth."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.evaluations.reconstruction import JudgeNotRun, default_judge, evaluate
from exp.common.evaluations.reconstruction_test import evidence, references
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.memory_outcomes import (
    MemoryOutcomeReport,
    record_memory_outcomes,
)


def test_build_records_only_preexisting_claims_against_later_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual build dreams after prediction; its own new claim cannot explain that trace."""
    from exp.optimize.evaluation.audit.panel_test import build_project, episode
    from exp.optimize.evaluation.fidelity.dreaming_test import DreamProvider
    from exp.optimize.evaluation.learning.traces import learning_plan
    from exp.runtime.models.providers.transport import JsonHttpResponse
    from exp.simulation.world_model.experience_memory import memory_head

    class ClaimReader(DreamProvider):
        """Expose a checked claim in the real recall wire so its later use is observable."""

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: JsonObject,
            timeout_seconds: float,
        ) -> JsonHttpResponse:
            """Rank claims first while leaving the build and predictor execution unchanged."""
            response = super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
            state = payload.get("state")
            records = state.get("records") if isinstance(state, dict) else None
            if url.endswith("/systemone") and isinstance(records, list):
                answers: JsonObject = {"dependency": {"type": "choice", "choice": "tool behavior"}}
                for index, row in enumerate(records):
                    record = row.get("record") if isinstance(row, dict) else None
                    score = (
                        2 if isinstance(record, dict) and record.get("kind") == "wiki-claim" else 0
                    )
                    for axis in ("field", "history"):
                        answers[f"{axis}_{index}"] = {"type": "score", "score": score}
                return JsonHttpResponse(status_code=200, body={"answers": answers})
            return response

    episodes = [episode(index) for index in range(2)]
    provider = ClaimReader(episodes, learnable=True)
    _, project, _ = build_project(
        tmp_path, monkeypatch, provider, "--dream-every", "1", learn=True, episodes=episodes
    )
    ledgers = [
        MemoryOutcomeReport.model_validate_json(project.artifacts.read_bytes(key, "outcomes.json"))
        for key in project.artifacts.list_ids(artifact_type="memory-prediction-outcomes")
    ]
    assert all(row.pinned_memory is not None for row in ledgers)
    learning = [
        row
        for row in ledgers
        if row.pinned_memory is not None and row.pinned_memory["mode"] == "learn"
    ]
    assert len(learning) == 2
    by_version = {
        row.pinned_memory["version"]: row for row in learning if row.pinned_memory is not None
    }
    assert not by_version[0].outcomes
    # The first claim also saw retrieved evidence from the second task, so provenance hides it
    # during that task's learning pass. The later disclosed retention test may read both claims.
    assert not by_version[1].outcomes
    retention = [
        row
        for row in ledgers
        if row.pinned_memory is not None and row.pinned_memory["mode"] == "frozen"
    ]
    assert retention and any(row.outcomes for row in retention)
    for outcome in retention[0].outcomes:
        record = json.loads(project.artifacts.read_bytes(outcome.claim.artifact_id, "record.json"))
        assert record["kind"] == "wiki-claim"
        assert "not native fact" in record["authority"]
        assert outcome.exact_reply is not None
    head = memory_head(project.artifacts, learning_plan(project).lineage)
    assert head is not None and head.version == 2
    calls = len(provider.requests)
    for row in ledgers:
        ref = record_memory_outcomes(
            project.artifacts,
            row.evaluation.artifact_id.removesuffix("-report"),
            created_at=datetime.now(UTC),
            code_revision="test",
        )
        assert ref is not None
    assert len(provider.requests) == calls
    assert memory_head(project.artifacts, head.lineage) == head


def test_outcome_ledger_rejects_a_claim_that_differs_from_the_delivered_record(
    tmp_path: Path,
) -> None:
    """A good prediction cannot be attributed to a substituted or revised claim text."""
    project = ProjectStore(tmp_path, "memory")
    now = datetime.now(UTC)
    claim: JsonObject = {"id": "claim", "kind": "wiki-claim", "claim": "Historical hypothesis."}
    project.artifacts.write_json(
        artifact_id="claim",
        artifact_type="memory-dream",
        envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="test"),
        files={"record.json": claim},
    )
    source = evidence().model_copy(
        update={
            "audit_metadata": {
                "memory_claim_deliveries": [
                    {
                        "step": 0,
                        "operation_id": "read-0",
                        "request_sha256": "a" * 64,
                        "claims": [
                            {"id": "claim", "sha256": sha256_json({**claim, "claim": "Different."})}
                        ],
                    }
                ]
            }
        }
    )

    def no_judge(body: JsonObject, identity: str) -> JsonObject:
        """Keep this reference check entirely mechanical."""
        del body, identity
        raise JudgeNotRun("Mechanical fixture")

    evaluate(
        source,
        references(),
        judge=default_judge("fixture"),
        complete=no_judge,
        artifacts=project.artifacts,
        artifact_id="case-report",
        clock=lambda: now,
        code_revision="test",
    )
    with pytest.raises(ValueError, match="differs from its original"):
        record_memory_outcomes(project.artifacts, "case", created_at=now, code_revision="test")
    assert not project.artifacts.exists("case-memory-outcomes")
