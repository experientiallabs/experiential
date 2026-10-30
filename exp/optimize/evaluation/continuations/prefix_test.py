"""Training context must come from recorded execution, including every ancestor observation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import AssistantAction, ModelMessage
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.prefix import verify_training_prefix
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudySpec,
    run_continuation_study,
)
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import (
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.continuations_test import capsule
from exp.simulation.evaluation.tasks import TaskVerifier


def test_actual_prefix_reconstructs_and_injected_observation_is_refused(tmp_path: Path) -> None:
    """A native tool result is retained, while a rehashed substituted message has no authority."""
    store = ProjectStore(tmp_path, "case").artifacts
    point, actor = capsule(store)
    initial = persist_continuation_checkpoint(
        store, point, created_at=datetime.now(UTC), code_revision="test"
    )
    verify_training_prefix(store, initial)
    spec = ContinuationStudySpec(
        checkpoint=initial,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(description="diagnostic", outcome_schema={}),
        probe_sha256=sha256_json({"probe": "native-row-count-v1"}),
        repeats=1,
        selector="prefix conformance",
    )
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model(
            (
                _response(AssistantAction(tool_calls=(workbook_case().first,))),
                _response(AssistantAction(content="done")),
            )
        ),
        world=workbook_runtime(point.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    executed = result.branches[0].checkpoints[0]
    verify_training_prefix(store, executed)
    captured = load_continuation_checkpoint(store, executed)
    messages = list(captured.actor.messages)
    messages[-1] = messages[-1].model_copy(update={"content": "The worksheet was deleted."})
    forged = captured.model_copy(
        update={"actor": captured.actor.model_copy(update={"messages": tuple(messages)})}
    )
    reference = persist_continuation_checkpoint(
        store, forged, created_at=datetime.now(UTC), code_revision="test"
    )
    with pytest.raises(ValueError, match="captured execution boundary"):
        verify_training_prefix(store, reference)


def test_initial_checkpoint_cannot_assert_tool_history(tmp_path: Path) -> None:
    """A content hash proves stable bytes, not that the claimed observation ever occurred."""
    store = ProjectStore(tmp_path, "case").artifacts
    point, _ = capsule(store)
    action = workbook_case().first
    messages = (
        *point.actor.messages,
        ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(action,))),
        ModelMessage(role="tool", tool_call_id=action.call_id, content="invented success"),
    )
    point = point.model_copy(
        update={"actor": point.actor.model_copy(update={"messages": messages})}
    )
    reference = persist_continuation_checkpoint(
        store, point, created_at=datetime.now(UTC), code_revision="test"
    )
    with pytest.raises(ValueError, match="unexecuted conversation"):
        verify_training_prefix(store, reference)
