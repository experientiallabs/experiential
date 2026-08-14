"""Tests for sparse cross-mode simulation specification validation."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.rollouts import SimulationMode
from wmo.simulation.orchestration import SimulationModeUnsupportedError, require_implemented_mode
from wmo.simulation.specs import (
    MixedRealitySettings,
    SandboxSettings,
    SimulationSpec,
    WorldModelSettings,
    simulation_spec_digest,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_PLAN_INPUT = ArtifactInput(artifact_id="evaluation-plan", sha256="a" * 64)
_WORLD_MODEL_INPUT = ArtifactInput(artifact_id="grounded-world-model", sha256="b" * 64)


def _spec(**updates: object) -> SimulationSpec:
    values: dict[str, object] = {
        "schema_version": 1,
        "created_at": _TIME,
        "inputs": (_PLAN_INPUT, _WORLD_MODEL_INPUT),
        "code_revision": "test-revision",
        "simulation_id": "simulation-a",
        "evaluation_plan_id": "evaluation-plan",
        "cell_ids": ("cell-a", "cell-b"),
        "agent_id": "agent-a",
        "mode": SimulationMode.WORLD_MODEL,
        "world_model": WorldModelSettings(
            world_model_alias="world-model-a",
            grounded_world_model_input=_WORLD_MODEL_INPUT,
            prompt_version="text-world-model-v1",
        ),
        "seed": 7,
        "maximum_steps": 3,
    }
    values.update(updates)
    return SimulationSpec.model_validate(values)


def test_spec_preserves_only_the_selected_mode_settings_and_is_digest_stable() -> None:
    """One stable explicit cell selection produces one stable specification digest."""
    first = _spec()
    second = _spec()

    assert first.world_model is not None
    assert first.world_model.maximum_output_tokens == 16_000
    assert first.world_model.allow_tools is False
    assert first.sandbox is None
    assert first.mixed_reality is None
    assert simulation_spec_digest(first) == simulation_spec_digest(second)


def test_v1_world_model_spec_preserves_exact_identity_payload() -> None:
    """A pre-extension v1 specification retains its exact serialized fields and digest."""
    payload = _spec().model_dump(mode="json")
    parsed = SimulationSpec.model_validate(payload)

    assert parsed.model_dump(mode="json") == payload
    assert parsed.world_model is not None
    assert set(parsed.world_model.model_dump(mode="json")) == {
        "world_model_alias",
        "grounded_world_model_input",
        "prompt_version",
        "query_embedding",
        "maximum_output_tokens",
        "allow_tools",
    }
    assert simulation_spec_digest(parsed) == (
        "c6ed1992b5e18efe9f245ab93007a846b5f95fdd60ef1d0a813bf7626f67950e"
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cell_ids": ("cell-b", "cell-a")}, "must be sorted"),
        ({"cell_ids": ("cell-a", "cell-a")}, "must not repeat"),
        ({"cell_ids": ()}, "at least one"),
        ({"world_model": None}, "missing settings"),
        (
            {
                "sandbox": SandboxSettings(
                    environment_id="sandbox-a",
                    environment_sha256="b" * 64,
                )
            },
            "inactive",
        ),
    ],
)
def test_spec_rejects_ambiguous_sparse_or_inactive_settings(
    updates: dict[str, object],
    message: str,
) -> None:
    """Invalid mode cells and config combinations fail before a simulator can run."""
    with pytest.raises(ValidationError, match=message):
        _spec(**updates)


def test_mixed_reality_shape_is_persistable_but_reserved_for_a_later_simulator() -> None:
    """The shared schema holds future mode configuration without implementing that mode."""
    spec = _spec(
        mode=SimulationMode.MIXED_REALITY,
        world_model=None,
        mixed_reality=MixedRealitySettings(policy_id="policy-a"),
    )

    assert spec.mixed_reality == MixedRealitySettings(policy_id="policy-a")
    with pytest.raises(SimulationModeUnsupportedError, match="not implemented"):
        require_implemented_mode(spec, SimulationMode.WORLD_MODEL)


def test_sandbox_limits_and_run_cost_ceiling_must_be_finite() -> None:
    """Non-finite limits fail before an environment or provider can be opened."""
    with pytest.raises(ValidationError, match="maximum_time_seconds must be finite"):
        SandboxSettings(
            environment_id="sandbox-a",
            environment_sha256="b" * 64,
            maximum_time_seconds=float("inf"),
        )
    with pytest.raises(ValidationError, match="maximum_cost_usd must be finite"):
        _spec(maximum_cost_usd=float("inf"))
