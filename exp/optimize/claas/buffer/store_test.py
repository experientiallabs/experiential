"""Transactional queue, feedback, rejection, and recovery regression tests."""

from pathlib import Path

import pytest

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.models import AssistantAction
from exp.optimize.claas.buffer.store import ExperienceBuffer
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.training_contracts import (
    TrainingBatch,
    TrainingCheckpoint,
    TrainingExample,
    TrainingResult,
)
from exp.optimize.claas.training_contracts_test import example, spec


def item(identity: str = "one", *, policy: str = "policy-0", exact: bool = True) -> TrainingExample:
    """Create a uniquely named exact example without a model or provider."""
    source = example(policy=policy, exact=exact)
    return source.model_copy(
        update={
            "experience": source.experience.model_copy(
                update={
                    "experience_id": identity,
                    "response_id": identity,
                }
            )
        }
    )


def result(batch: TrainingBatch, root: Path, *, step: int = 1) -> TrainingResult:
    """Make an orchestration receipt; this fixture does not execute optimizer training."""
    revision = f"policy-{step}"
    return TrainingResult(
        checkpoint=TrainingCheckpoint(
            scope=spec().scope,
            adapter_id=spec().adapter_id,
            policy_revision=revision,
            policy_history=(revision, batch.expected_policy_revision),
            step=step,
            path=str(root / revision),
            manifest_sha256="0" * 64,
        ),
        consumed_experience_ids=tuple(value.experience.experience_id for value in batch.examples),
        metrics={"loss": 0.5},
    )


def test_all_or_none_import_and_feedback_bounds(tmp_path: Path) -> None:
    """An oversized import or feedback update leaves every prior row unchanged."""
    limits = RunConfiguration(maximum_buffer_records=1, maximum_record_bytes=4096)
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), limits)
    with pytest.raises(ValueError, match="full"):
        store.import_examples((item("one"), item("two")))
    assert store.status().ready == 0
    store.import_examples((item(),))
    store.import_examples((item(),))
    assert store.status().ready == 1
    with pytest.raises(ValueError, match="conflicts"):
        store.feedback("one", scalar_reward=None, text_feedback="changed")
    batch = store.lease()
    assert batch is not None
    assert batch.examples == (item(),)


def test_generated_response_feedback_replay_and_frozen_lease(tmp_path: Path) -> None:
    """Request replay is exact and consumed feedback cannot be silently rewritten."""
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), RunConfiguration())
    request = GenerationRequest(request_id="request", model="tiny-model", prompt="Question")
    tokens = item().experience.exact_tokens
    assert tokens is not None
    generation = GenerationResult(
        response_id="response",
        action=AssistantAction(content="Answer"),
        exact_tokens=tokens,
        raw_text="Answer",
    )
    store.record_generation(request, generation)
    assert store.status().pending_feedback == 1
    assert store.replay(request) == generation
    with pytest.raises(ValueError, match="different generation"):
        store.replay(request.model_copy(update={"prompt": "Other"}))
    store.feedback("response", scalar_reward=None, text_feedback="Correct")
    assert store.status().ready == 1
    batch = store.lease()
    assert batch is not None
    store.feedback("response", scalar_reward=None, text_feedback="Correct")
    with pytest.raises(ValueError, match="frozen"):
        store.feedback("response", scalar_reward=1, text_feedback=None)
    store.acknowledge(batch, result(batch, tmp_path))
    assert store.status().consumed == 1
    store.feedback("response", scalar_reward=None, text_feedback="Correct")


def test_inflight_survives_reopen_and_ack_is_atomic(tmp_path: Path) -> None:
    """A crash leaves the identical optimization ID and examples eligible for safe replay."""
    path = tmp_path / "queue.sqlite"
    store = ExperienceBuffer(path, spec(), RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    resumed = ExperienceBuffer(path, spec(), RunConfiguration())
    assert resumed.inflight() == batch
    assert resumed.lease() == batch
    completed = result(batch, tmp_path)
    with pytest.raises(ValueError, match="leased batch"):
        resumed.acknowledge(
            batch, completed.model_copy(update={"consumed_experience_ids": ("wrong",)})
        )
    assert resumed.status().inflight == 1
    assert resumed.checkpoint() is None
    resumed.acknowledge(batch, completed)
    resumed.acknowledge(batch, completed)
    assert resumed.inflight() is None
    assert resumed.checkpoint() == completed.checkpoint
    assert resumed.status().consumed == 1


def test_unsupported_and_stale_examples_remain_visible(tmp_path: Path) -> None:
    """Missing exact evidence and obsolete policies are retained as explicit rejections."""
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), RunConfiguration())
    store.import_examples((item("no-tokens", exact=False), item("old", policy="unknown")))
    status = store.status()
    assert status.rejected == 2
    assert len(status.rejection_reasons) == 2
    assert store.lease() is None


def test_partial_batches_obey_token_limit_and_snapshot(tmp_path: Path) -> None:
    """A drain admits only snapshot IDs and never truncates an original example."""
    store = ExperienceBuffer(
        tmp_path / "queue.sqlite",
        spec().model_copy(update={"max_batch_tokens": 4}),
        RunConfiguration(),
    )
    store.import_examples((item("one"), item("two")))
    batch = store.lease(allowed_ids=("two",))
    assert batch is not None
    assert [value.experience.experience_id for value in batch.examples] == ["two"]
    assert store.status().ready == 1


def test_feedback_growth_is_atomic_and_scope_cannot_partially_import(tmp_path: Path) -> None:
    """Invalid feedback and a late foreign import leave earlier records untouched."""
    store = ExperienceBuffer(
        tmp_path / "queue.sqlite", spec(), RunConfiguration(maximum_record_bytes=2048)
    )
    source = item().model_copy(update={"text_feedback": None})
    store.import_examples((source,))
    before = store.status()
    with pytest.raises(ValueError, match="maximum_record_bytes"):
        store.feedback("one", scalar_reward=None, text_feedback="x" * 2048)
    assert store.status() == before
    foreign = example(application="other")
    with pytest.raises(ValueError, match="scope"):
        store.import_examples((item("new"), foreign))
    assert store.status() == before


def test_full_recipe_identity_is_required_on_reopen(tmp_path: Path) -> None:
    """The queue cannot silently resume against another tokenizer or training objective."""
    path = tmp_path / "queue.sqlite"
    ExperienceBuffer(path, spec(), RunConfiguration())
    with pytest.raises(ValueError, match="recipe differs"):
        ExperienceBuffer(
            path, spec().model_copy(update={"objective": "reinforce"}), RunConfiguration()
        )


@pytest.mark.parametrize("objective", ["sdpo", "reinforce", "hybrid"])
def test_complementary_feedback_waits_for_the_required_signal(
    tmp_path: Path, objective: str
) -> None:
    """Partial feedback stays pending and later nonconflicting signals complete readiness."""
    recipe = spec().model_copy(update={"objective": objective})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    request = GenerationRequest(request_id="request", model="tiny-model", prompt="Question")
    tokens = item().experience.exact_tokens
    assert tokens is not None
    store.record_generation(
        request,
        GenerationResult(
            response_id="response",
            action=AssistantAction(content="answer"),
            exact_tokens=tokens,
            raw_text="answer",
        ),
    )
    first_scalar = objective != "reinforce"
    store.feedback(
        "response",
        scalar_reward=1.0 if first_scalar else None,
        text_feedback=None if first_scalar else "Correct",
    )
    assert store.status().pending_feedback == 1
    assert store.lease() is None
    store.feedback(
        "response",
        scalar_reward=None if first_scalar else 1.0,
        text_feedback="Correct" if first_scalar else None,
    )
    batch = store.lease()
    assert batch is not None
    assert batch.examples[0].scalar_reward == 1.0
    assert batch.examples[0].text_feedback == "Correct"


def test_receipt_space_is_reserved_before_leasing(tmp_path: Path) -> None:
    """A full buffer rejects dispatch while leaving every input ready and unconsumed."""
    path = tmp_path / "queue.sqlite"
    limits = RunConfiguration(maximum_update_receipt_bytes=1024)
    store = ExperienceBuffer(path, spec(), limits)
    store.import_examples((item(),))
    batch_bytes = len(
        TrainingBatch(
            batch_id="0" * 32,
            expected_policy_revision="policy-0",
            examples=(item(),),
        )
        .model_dump_json()
        .encode()
    )
    cap = store.status().retained_bytes + batch_bytes + 2 * 1024 - 1
    constrained = ExperienceBuffer(
        path,
        spec(),
        limits.model_copy(update={"maximum_buffer_bytes": cap}),
    )
    with pytest.raises(ValueError, match="buffer is full"):
        constrained.lease()
    assert constrained.inflight() is None
    assert constrained.status().ready == 1
    assert constrained.status().consumed == 0


def test_completed_update_fits_its_frozen_receipt_reservation(tmp_path: Path) -> None:
    """After dispatch, acknowledgement uses reserved bytes rather than needing extra capacity."""
    path = tmp_path / "queue.sqlite"
    limits = RunConfiguration(maximum_update_receipt_bytes=1024)
    store = ExperienceBuffer(path, spec(), limits)
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    retained = store.status().retained_bytes
    resumed = ExperienceBuffer(
        path,
        spec(),
        limits.model_copy(update={"maximum_buffer_bytes": retained}),
    )
    completed = result(batch, tmp_path)
    resumed.acknowledge(batch, completed)
    assert resumed.status().retained_bytes < retained
    assert resumed.status().consumed == 1
    assert resumed.checkpoint() == completed.checkpoint


def test_oversized_receipt_preserves_unacknowledged_lease(tmp_path: Path) -> None:
    """A runtime violating its output cap cannot consume inputs or publish partial metadata."""
    store = ExperienceBuffer(
        tmp_path / "queue.sqlite",
        spec(),
        RunConfiguration(maximum_update_receipt_bytes=1024),
    )
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    oversized = result(batch, tmp_path).model_copy(update={"metrics": {"x" * 1024: 0.0}})
    before = store.status()
    with pytest.raises(ValueError, match="reserved byte limit"):
        store.acknowledge(batch, oversized)
    assert store.status() == before
    assert store.inflight() == batch
    assert store.checkpoint() is None


def test_pending_feedback_uses_latest_policy_eligibility(tmp_path: Path) -> None:
    """Late feedback explicitly rejects an obsolete rollout without losing its original record."""
    recipe = spec().model_copy(update={"max_policy_lag": 0})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    pending = item("pending").model_copy(update={"text_feedback": None})
    store.import_examples((pending, item("trained")))
    batch = store.lease()
    assert batch is not None
    store.acknowledge(batch, result(batch, tmp_path))
    store.feedback("pending", scalar_reward=None, text_feedback="Correct")
    assert store.status().pending_feedback == 0
    assert store.status().rejected == 1
    assert store.status().consumed == 1
    assert store.lease() is None
