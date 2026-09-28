"""Transactional queue, feedback, rejection, and recovery regression tests."""

import shutil
from pathlib import Path

import pytest

from exp.common.claas.batches import TrainingSubmission
from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.claas.learning import FeedbackSubmission
from exp.common.core.artifacts import sha256_json
from exp.common.models import AssistantAction
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    checkpoint_receipt,
    recover_training_result,
)
from exp.optimize.claas.backends.checkpoints_test import checkpoint
from exp.optimize.claas.buffer.store import ExperienceBuffer
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingExample,
    TrainingResult,
)
from exp.optimize.claas.training_contracts_test import example, spec


def record_pending(store: ExperienceBuffer, identity: str, *, policy: str = "policy-0") -> None:
    """Retain an original sample without assigning its eventual cohort reward."""
    tokens = item(identity, policy=policy).experience.exact_tokens
    assert tokens is not None
    store.record_generation(
        GenerationRequest(request_id=identity, model="tiny-model", prompt="question"),
        GenerationResult(
            response_id=identity,
            action=AssistantAction(content="answer"),
            exact_tokens=tokens,
            raw_text="answer",
        ),
    )


def submission(
    *identities: str, batch_id: str = "batch-1", policy: str = "policy-0"
) -> TrainingSubmission:
    """Select explicit original response IDs with already-attributed centered rewards."""
    return TrainingSubmission(
        batch_id=batch_id,
        expected_policy_revision=policy,
        feedback=tuple(
            FeedbackSubmission(response_id=identity, reward=0.5) for identity in identities
        ),
    )


def test_explicit_batch_rejection_is_all_or_none_and_retry_is_immutable(tmp_path: Path) -> None:
    """A late bad response rolls back earlier feedback; exact retries survive policy advance."""
    recipe = spec().model_copy(update={"objective": "reinforce"})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    for identity in ("one", "two", "unrelated"):
        record_pending(store, identity)
    before = store.status()
    with pytest.raises(ValueError, match="unknown response"):
        store.lease_submission(submission("one", "missing"))
    assert store.status() == before and store.inflight() is None
    assert store.batch_status("batch-1") is None
    selected = submission("two", "one")
    accepted = store.lease_submission(selected)
    assert accepted.response_ids == ("two", "one") and accepted.state == "pending"
    assert store.status().pending_feedback == 1
    assert store.lease_submission(selected) == accepted
    with pytest.raises(ValueError, match="another batch is pending"):
        store.lease_submission(submission("one", batch_id="overlap"))
    with pytest.raises(ValueError, match="different immutable"):
        store.lease_submission(submission("one", "two"))
    batch = store.inflight()
    assert batch is not None
    store.acknowledge(batch, result(batch, tmp_path, recipe=recipe))
    completed = store.lease_submission(selected)
    assert completed.state == "completed" and completed.result is not None
    assert completed.batch_sha256 == accepted.batch_sha256
    assert store.checkpoint() is not None
    with pytest.raises(ValueError, match="stale"):
        store.lease_submission(submission("unrelated", batch_id="stale"))
    with pytest.raises(ValueError, match="exact current policy"):
        store.lease_submission(submission("unrelated", batch_id="old-sample", policy="policy-1"))


@pytest.mark.parametrize("limit", ["max_batch_tokens", "max_batch_examples"])
def test_explicit_batch_never_silently_splits_selected_responses(
    tmp_path: Path, limit: str
) -> None:
    """A whole selection exceeding either update bound preserves all original pending feedback."""
    recipe = spec().model_copy(
        update={"objective": "reinforce", limit: 4 if limit == "max_batch_tokens" else 1}
    )
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    for identity in ("one", "two"):
        record_pending(store, identity)
    before = store.status()
    with pytest.raises(ValueError, match=limit):
        store.lease_submission(submission("one", "two"))
    assert store.status() == before
    assert store.inflight() is None and store.batch_status("batch-1") is None


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


def result(
    batch: TrainingBatch,
    root: Path,
    *,
    step: int = 1,
    recipe: ClaasTrainingSpec | None = None,
    history: tuple[str, ...] | None = None,
    metrics: dict[str, float] | None = None,
) -> TrainingResult:
    """Write a hash-complete inert checkpoint without executing optimizer training."""
    recipe = recipe or spec()
    revision = f"policy-{step}"
    directory = root / f"claas-{revision}"
    checkpoint(directory)
    manifest_path = directory / "manifest.json"
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes()).model_copy(
        update={
            "spec": recipe,
            "policy_revision": revision,
            "parent_policy_revision": batch.expected_policy_revision,
            "policy_history": (history or (revision, batch.expected_policy_revision))[
                : recipe.max_policy_lag + 1
            ],
            "step": step,
            "batch_id": batch.batch_id,
            "batch_sha256": sha256_json(batch),
            "consumed_experience_ids": tuple(
                value.experience.experience_id for value in batch.examples
            ),
            "metrics": metrics if metrics is not None else {"loss": 0.5},
        }
    )
    manifest_path.write_text(manifest.model_dump_json())
    return TrainingResult(
        checkpoint=checkpoint_receipt(directory, manifest),
        consumed_experience_ids=manifest.consumed_experience_ids,
        metrics=manifest.metrics,
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
    oversized = result(batch, tmp_path, metrics={"x" * 1024: 0.0})
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
    store.acknowledge(batch, result(batch, tmp_path, recipe=recipe))
    store.feedback("pending", scalar_reward=None, text_feedback="Correct")
    assert store.status().pending_feedback == 0
    assert store.status().rejected == 1
    assert store.status().consumed == 1
    assert store.lease() is None


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_committed_checkpoint_recovery_acknowledges_each_policy_lag_once(
    tmp_path: Path, lag: int
) -> None:
    """Recover immutable native-shaped state without another optimizer or stale-parent history."""
    recipe = spec().model_copy(update={"max_policy_lag": lag})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    for step in (1, 2):
        store.import_examples((item(f"step-{step}", policy=f"policy-{step - 1}"),))
        batch = store.lease()
        assert batch is not None
        saved = result(
            batch,
            tmp_path,
            step=step,
            recipe=recipe,
            history=tuple(f"policy-{index}" for index in range(step, -1, -1)),
        )
        recovered = recover_training_result(tmp_path, recipe, batch, "main")
        assert recovered is not None
        assert recovered == saved
        assert len(saved.checkpoint.policy_history) == min(step + 1, lag + 1)
        reopened = ExperienceBuffer(store.path, recipe, RunConfiguration())
        assert reopened.inflight() == batch
        reopened.acknowledge(batch, recovered)
        reopened.acknowledge(batch, recovered)
        assert reopened.inflight() is None
        assert reopened.status().consumed == step
        assert reopened.checkpoint() == saved.checkpoint
        status = reopened.batch_status(batch.batch_id)
        assert status is not None and status.result == saved


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("parent_policy_revision", "wrong-parent"),
        ("batch_id", "wrong-batch"),
        ("batch_sha256", "0" * 64),
        ("metrics", {"loss": 2.0}),
        ("consumed_experience_ids", ("wrong-response",)),
    ],
)
def test_acknowledgement_binds_verified_manifest_to_exact_lease(
    tmp_path: Path, field: str, changed: object
) -> None:
    """Even a rehashed structural manifest cannot substitute another parent, batch, or result."""
    recipe = spec().model_copy(update={"max_policy_lag": 0})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    saved = result(batch, tmp_path, recipe=recipe)
    path = Path(saved.checkpoint.path) / "manifest.json"
    manifest = CheckpointManifest.model_validate_json(path.read_bytes()).model_copy(
        update={field: changed}
    )
    path.write_text(manifest.model_dump_json())
    substituted = saved.model_copy(update={"checkpoint": checkpoint_receipt(path.parent, manifest)})
    before = store.status()
    with pytest.raises(ValueError, match="leased batch"):
        store.acknowledge(batch, substituted)
    assert store.status() == before
    assert store.inflight() == batch
    assert store.checkpoint() is None


def test_acknowledgement_rejects_changed_checkpoint_payload(tmp_path: Path) -> None:
    """A matching receipt cannot acknowledge corrupted optimizer bytes."""
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    saved = result(batch, tmp_path)
    optimizer = Path(saved.checkpoint.path) / "verl/actor/optim_world_size_1_rank_0.pt"
    optimizer.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="missing or changed"):
        store.acknowledge(batch, saved)
    assert store.inflight() == batch and store.checkpoint() is None


def test_acknowledgement_cannot_skip_a_checkpoint_step(tmp_path: Path) -> None:
    """Even a matching manifest and parent cannot jump over an unacknowledged update."""
    recipe = spec().model_copy(update={"max_policy_lag": 0})
    store = ExperienceBuffer(tmp_path / "queue.sqlite", recipe, RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    saved = result(batch, tmp_path, step=2, recipe=recipe)
    with pytest.raises(ValueError, match="advance exactly one"):
        store.acknowledge(batch, saved)
    assert store.inflight() == batch and store.checkpoint() is None


def test_positive_lag_retains_its_immediate_parent(tmp_path: Path) -> None:
    """Manifest parent validation cannot authorize a contradictory retained ancestry."""
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    saved = result(batch, tmp_path, history=("policy-1", "foreign"))
    with pytest.raises(ValueError, match="leased batch"):
        store.acknowledge(batch, saved)
    assert store.inflight() == batch and store.checkpoint() is None


@pytest.mark.parametrize("unavailable", ["missing", "corrupt"])
def test_completed_acknowledgement_replays_without_checkpoint_files(
    tmp_path: Path, unavailable: str
) -> None:
    """An exact completed retry returns durable evidence without reauthorizing absent artifacts."""
    store = ExperienceBuffer(tmp_path / "queue.sqlite", spec(), RunConfiguration())
    store.import_examples((item(),))
    batch = store.lease()
    assert batch is not None
    saved = result(batch, tmp_path)
    store.acknowledge(batch, saved)
    completed = store.batch_status(batch.batch_id)
    before = store.path.read_bytes()
    checkpoint_root = Path(saved.checkpoint.path)
    if unavailable == "missing":
        shutil.rmtree(checkpoint_root)
    else:
        (checkpoint_root / "manifest.json").write_text("corrupted")
    store.acknowledge(batch, saved)
    assert store.path.read_bytes() == before
    assert store.batch_status(batch.batch_id) == completed
    with pytest.raises(ValueError, match="different acknowledgement"):
        store.acknowledge(batch, saved.model_copy(update={"metrics": {"loss": 99.0}}))
    assert store.path.read_bytes() == before
