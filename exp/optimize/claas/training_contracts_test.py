"""Token integrity, application isolation, and explicit objective contract tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.claas import ClaasScope, ExactTokenEvidence, Experience, ExperienceProvenance
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingExample,
    TrainingJob,
    next_policy_revision,
    validate_training_batch,
)


def example(
    *, policy: str = "policy-0", exact: bool = True, application: str = "app"
) -> TrainingExample:
    """Construct exact action evidence without any provider or model call."""
    return TrainingExample(
        experience=Experience(
            experience_id="experience-1",
            response_id="response-1",
            scope=ClaasScope(user_id="user", application_id=application),
            protocol="chat_completions",
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
            request={},
            response={},
            provenance=ExperienceProvenance(
                source_kind="simulation",
                source_id="simulation-1",
                model_id="tiny-model",
                model_revision="base-1",
                policy_revision=policy,
            ),
            exact_tokens=ExactTokenEvidence(
                model_id="tiny-model",
                model_revision="base-1",
                policy_revision=policy,
                tokenizer_id="tiny-tokenizer",
                tokenizer_revision="tokenizer-1",
                prompt_token_ids=(1, 2),
                response_token_ids=(3, 4),
                response_logprobs=(-2.0, -2.0),
                sampling_temperature=1.0,
                sampling_top_p=1.0,
                sampling_top_k=None,
            )
            if exact
            else None,
        ),
        scalar_reward=1.0,
        text_feedback="Use the correct answer.",
    )


def spec() -> ClaasTrainingSpec:
    """Return the tiny text model's explicit training recipe."""
    return ClaasTrainingSpec(
        scope=ClaasScope(user_id="user", application_id="app"),
        adapter_id="adapter-1",
        base_model="tiny-model",
        model_revision="base-1",
        tokenizer_id="tiny-tokenizer",
        tokenizer_revision="tokenizer-1",
        initial_policy_revision="policy-0",
        target_modules=("c_attn",),
        max_sequence_tokens=128,
        lora_rank=2,
        lora_alpha=4,
        learning_rate=0.02,
    )


def job(root: Path) -> TrainingJob:
    """Build one bounded worker job using only generated fixture evidence."""
    return TrainingJob(
        spec=spec(),
        batch=TrainingBatch(
            batch_id="batch-1", expected_policy_revision="policy-0", examples=(example(),)
        ),
        checkpoint_root=str(root),
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"exact": False}, "simulation seed"),
        ({"application": "other"}, "another user or application"),
        ({"policy": "old-policy"}, "model, tokenizer, or policy differs"),
    ],
)
def test_rejects_unbound_training_evidence(
    tmp_path: Path, changes: dict[str, str | bool], match: str
) -> None:
    """Reject provider text, cross-user data, and obsolete rollout weights."""
    item = example(
        policy=str(changes.get("policy", "policy-0")),
        application=str(changes.get("application", "app")),
        exact=bool(changes.get("exact", True)),
    )
    with pytest.raises(ValueError, match=match):
        TrainingJob(
            spec=spec(),
            batch=TrainingBatch(
                batch_id="batch-1", expected_policy_revision="policy-0", examples=(item,)
            ),
            checkpoint_root=str(tmp_path),
        )


def test_rejects_overlength_without_truncation(tmp_path: Path) -> None:
    """No saved student context or target is silently truncated to make a batch fit."""
    with pytest.raises(ValueError, match="truncation"):
        TrainingJob(
            spec=spec().model_copy(update={"max_sequence_tokens": 3}),
            batch=job(tmp_path).batch,
            checkpoint_root=str(tmp_path),
        )


def test_objective_selection_is_explicit(tmp_path: Path) -> None:
    """Scalar feedback cannot silently substitute for SDPO."""
    scalar = example().model_copy(update={"text_feedback": None})
    batch = TrainingBatch(batch_id="b", expected_policy_revision="policy-0", examples=(scalar,))
    with pytest.raises(ValueError, match="require text_feedback"):
        TrainingJob(spec=spec(), batch=batch, checkpoint_root=str(tmp_path))
    accepted = TrainingJob(
        spec=spec().model_copy(update={"objective": "reinforce"}),
        batch=batch,
        checkpoint_root=str(tmp_path),
    )
    assert accepted.batch.examples[0].scalar_reward == 1


def test_revision_binds_scope_recipe_and_exact_action(tmp_path: Path) -> None:
    """Distinct recipes cannot publish the same weight identity."""
    first = job(tmp_path)
    changed = first.model_copy(update={"spec": spec().model_copy(update={"sdpo_alpha": 1.0})})
    assert next_policy_revision(first) != next_policy_revision(changed)
    assert next_policy_revision(first) == next_policy_revision(job(tmp_path))


def test_bounded_replay_accepts_only_verified_recent_ancestors(tmp_path: Path) -> None:
    """A saved behavior policy can be reused only within the proven revision window."""
    # checkpoints_test imports job/spec fixtures from this module, so defer this edge.
    from exp.optimize.claas.backends.checkpoints_test import checkpoint

    receipt = checkpoint(tmp_path)
    replay = TrainingBatch(
        batch_id="replay", expected_policy_revision="policy-1", examples=(example(),)
    )
    validate_training_batch(spec(), replay, receipt)
    with pytest.raises(ValueError, match="model, tokenizer, or policy"):
        validate_training_batch(spec().model_copy(update={"max_policy_lag": 0}), replay, receipt)
    foreign = TrainingBatch(
        batch_id="foreign",
        expected_policy_revision="policy-1",
        examples=(example(policy="unrelated-policy"),),
    )
    with pytest.raises(ValueError, match="model, tokenizer, or policy"):
        validate_training_batch(spec(), foreign, receipt)


@pytest.mark.parametrize(
    "changes",
    [{"sampling_temperature": 0.7}, {"sampling_top_p": 0.9}, {"sampling_top_k": 10}],
)
def test_rejects_incompatible_sampling_distribution(
    tmp_path: Path, changes: dict[str, float | int]
) -> None:
    """Importance weighting requires behavior logprobs from the same unfiltered policy."""
    item = example()
    tokens = item.experience.exact_tokens
    assert tokens is not None
    changed = item.model_copy(
        update={
            "experience": item.experience.model_copy(
                update={"exact_tokens": tokens.model_copy(update=changes)}
            )
        }
    )
    with pytest.raises(ValueError, match="temperature=1"):
        TrainingJob(
            spec=spec(),
            batch=TrainingBatch(
                batch_id="filtered", expected_policy_revision="policy-0", examples=(changed,)
            ),
            checkpoint_root=str(tmp_path),
        )


@pytest.mark.parametrize("missing", ["scalar_reward", "text_feedback"])
def test_hybrid_requires_both_declared_signals(tmp_path: Path, missing: str) -> None:
    """A hybrid manifest cannot describe a one-objective update."""
    item = example().model_copy(update={missing: None})
    with pytest.raises(ValueError, match="hybrid require"):
        TrainingJob(
            spec=spec().model_copy(update={"objective": "hybrid"}),
            batch=TrainingBatch(
                batch_id="hybrid", expected_policy_revision="policy-0", examples=(item,)
            ),
            checkpoint_root=str(tmp_path),
        )


@pytest.mark.parametrize(
    "field",
    [
        "adapter_id",
        "base_model",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "initial_policy_revision",
        "target_modules",
    ],
)
def test_training_recipe_identifiers_reject_whitespace(field: str) -> None:
    """Blank model, adapter, revision, and target names cannot reach backend allocation."""
    payload = spec().model_dump()
    payload[field] = [" \t"] if field == "target_modules" else " \t"
    with pytest.raises(ValueError, match="pattern"):
        ClaasTrainingSpec.model_validate(payload)
