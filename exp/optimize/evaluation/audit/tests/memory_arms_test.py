"""Memory-usefulness arms: bare, guidance and earlier checkpoints through ``exp eval --audit``.

The arms share one build, one fit-only memory and one panel of held-out cases; their specs differ
only in what the arm names. Guidance is temporary: it reaches predictions of its own session,
never sees a recorded reply of the replayed trajectory, and is never published to memory.
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel import MemoryProfile, PlannedPanel, plan_held_out_panel
from exp.optimize.evaluation.audit.panel_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    _arr,
    _at,
    _obj,
    _tool_reply,
    _user_content,
    build_project,
    episode,
    held_out_command,
    learn_fit,
)
from exp.optimize.evaluation.audit.receipts import audit_receipts, receipt_panel
from exp.optimize.evaluation.learning.traces import fit_cases, learning_plan
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest
from exp.simulation.world_model.experience_memory import memory_head
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact
from exp.simulation.world_model.session_guidance import GUIDANCE_TYPE

NOTE = "replies list matching records as JSON"


class GuidanceProvider(SimulatedProvider):
    """The simulated provider, also answering session-guidance proposals with one cited note."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer a guidance request; route everything else as recorded truth."""
        if url.endswith("/chat/completions") and "tools" in payload:
            schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
            if "guidance" in schema:
                self.requests.append(RecordedRequest(url, dict(headers), payload))
                request = _user_content(payload)
                originals = request["retrieved_originals"]
                assert isinstance(originals, list)
                basis = [str(_obj(row)["transition_id"]) for row in originals][:1]
                action = _obj(request["current_action"])
                tool = str(_obj(_at(action, "tool_calls", 0))["name"])
                return _tool_reply(
                    "world-v1",
                    "submit_proposal",
                    {"guidance": [{"tool": tool, "note": NOTE, "basis": basis}]},
                )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def guidance_requests(self) -> list[JsonObject]:
        """Every guidance request body's user content."""
        return [
            _user_content(request.payload)
            for request in self.requests
            if request.url.endswith("/chat/completions")
            and "guidance"
            in json.dumps(_at(request.payload, "tools", 0, "function", "parameters", "properties"))
        ]


def _project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, ProjectStore, GuidanceProvider]:
    """A built project whose fit-only memory is learned, before any audit."""
    provider = GuidanceProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    return root, store, provider


def _plan(
    store: ProjectStore, profile: MemoryProfile, checkpoint: int | None = None
) -> PlannedPanel:
    """Plan one arm's panel without registering it."""
    return plan_held_out_panel(
        store,
        2,
        profile="memory",
        code_revision=REVISION,
        clock=lambda: datetime(2026, 10, 23, tzinfo=UTC),
        memory_profile=profile,
        checkpoint=checkpoint,
    )


def test_arms_differ_only_in_what_they_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same cases and specs; bare swaps the simulator, guidance adds its policy, memory differs."""
    _, store, _ = _project(tmp_path, monkeypatch)
    arms = {name: _plan(store, name) for name in ("bare", "none", "guidance", "learned")}
    assert len({tuple(panel.lineages) for panel in arms.values()}) == 1
    varying = {"world_model", "memory", "guidance", "maximum_physical_attempts", "run_id"}
    base = [spec.model_dump(mode="json", exclude=varying) for spec in arms["none"].specs]
    for panel in arms.values():
        assert [spec.model_dump(mode="json", exclude=varying) for spec in panel.specs] == base
    assert all(spec.memory is None and spec.guidance is None for spec in arms["none"].specs)
    assert all(spec.memory is None and spec.guidance for spec in arms["guidance"].specs)
    assert all(spec.memory is not None for spec in arms["learned"].specs)
    bare = load_grounded_world_model_artifact(store.artifacts, arms["bare"].world_model)
    hybrid = load_grounded_world_model_artifact(store.artifacts, arms["none"].world_model)
    assert bare.original_evidence == "none" and hybrid.original_evidence == "retrieved"
    assert arms["none"].world_model == arms["guidance"].world_model == arms["learned"].world_model
    assert arms["bare"].checkpoint is not None and arms["bare"].checkpoint.memory == "bare"


def test_bare_arm_retrieves_no_original_and_reads_no_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory-first arm's predictions carry no grounded example and no memory channel."""
    root, _, provider = _project(tmp_path, monkeypatch)
    before = len(provider.requests)
    output = held_out_command(root, 2, "--audit-memory", "bare")
    assert "retrieve no recorded originals" in output
    served = provider.predictions()[-6:]
    assert len(provider.requests) > before and served
    for prediction in served:
        assert not prediction["grounded_examples"]
        assert "memory_observations" not in prediction and "session_guidance" not in prediction
    assert not any(r.url.endswith("/embeddings") for r in provider.requests[before:])


def test_guidance_is_session_scoped_temporary_and_never_sees_this_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notes reach later steps of their own session only; nothing is published; replays."""
    root, store, provider = _project(tmp_path, monkeypatch)
    plan = learning_plan(store, scope="fit")
    head = memory_head(store.artifacts, plan.lineage)
    output = held_out_command(root, 2, "--audit-memory", "guidance")
    assert "temporary session guidance" in output
    guidance = provider.guidance_requests()
    predictions = [row for row in provider.predictions() if "session_guidance" in row]
    assert guidance and len(guidance) == len(predictions)
    for request in guidance:
        assert set(request) == {
            "instruction",
            "current_action",
            "tools",
            "retrieved_originals",
            "original_contexts",
            "session_guidance",
        }, "no state, history, facts, memory or recorded reply of this trajectory"
    for prediction in predictions:
        assert "memory_observations" not in prediction
        notes = prediction["session_guidance"]
        assert isinstance(notes, list) and notes
        assert all("invented" not in json.dumps(_obj(note)["basis"]) for note in notes)
    # A session's first prediction carries only its own note; later ones accumulate.
    sizes = [len(_arr(row["session_guidance"])) for row in predictions]
    assert sizes[0] == 1 and max(sizes) > 1
    assert memory_head(store.artifacts, plan.lineage) == head, "guidance is never published"
    records = [
        json.loads(store.artifacts.read_bytes(identity, "guidance.json"))
        for identity in store.artifacts.list_ids(artifact_type=GUIDANCE_TYPE)
    ]
    assert len(records) == len(predictions)
    assert all(row["persisted_to_memory"] is False for row in records)
    bought = len(provider.requests)
    replay = held_out_command(root, 2, "--audit-memory", "guidance")
    assert "Saved results" in replay or "no new model calls" in replay
    assert len(provider.requests) == bought


def test_an_earlier_checkpoint_is_audited_by_consumed_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--audit-checkpoint K reads version K; a missing version and misuse are refused."""
    root, store, _ = _project(tmp_path, monkeypatch)
    fit = len(fit_cases(store))
    early = _plan(store, "learned", checkpoint=2)
    assert early.checkpoint is not None
    assert (early.checkpoint.version, early.checkpoint.fit_traces_learned) == (2, 2)
    assert all(spec.memory and spec.memory.checkpoint == 2 for spec in early.specs)
    final = _plan(store, "learned")
    assert final.checkpoint is not None and final.checkpoint.version == fit
    assert early.identity != final.identity
    with pytest.raises(ValueError, match="does not exist yet"):
        _plan(store, "learned", checkpoint=fit + 1)
    output = held_out_command(root, 2, "--audit-checkpoint", "2")
    assert "Held-out simulation fidelity" in output and "Complete" in output
    receipts = audit_receipts(store)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.request.checkpoint == 2
    assert receipt.checkpoint is not None and receipt.checkpoint.version == 2
    registered = receipt_panel(store, receipt)
    assert registered.prepared
    assert all(
        row.spec.memory is not None and row.spec.memory.checkpoint == 2
        for row in registered.prepared
    )
    plan = learning_plan(store, scope="fit")
    report = store.paths.runtime_directory / "audits" / f"{plan.lineage}-v2" / "report.txt"
    assert "Report:" in output
    assert "memory v2" in report.read_text()
    misuse = held_out_command(root, 2, "--audit-checkpoint", "0", exit_code=2)
    assert "--audit-checkpoint" in misuse
