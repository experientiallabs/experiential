"""Activation races, immutable revision identities, and scope-safe rollback."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exp.common.claas.contracts import ClaasScope
from exp.runtime.claas.registry import AdapterRegistry, ServingRevision, StaleRegistryError


def base_revision() -> ServingRevision:
    """Return the original model identity of a local application."""
    return ServingRevision(
        scope=ClaasScope(user_id="local", application_id="claims"),
        policy_revision="base",
        model_id="tiny",
        model_revision="commit",
        tokenizer_id="tiny",
        tokenizer_revision="commit",
    )


def candidate(base: ServingRevision, root: Path, name: str = "candidate") -> ServingRevision:
    """Make a concrete adapter reference without loading a GPU."""
    return ServingRevision.model_validate(
        {
            **base.model_dump(),
            "policy_revision": name,
            "adapter_directory": str(root / name / "student"),
            "manifest_sha256": "a" * 64,
        }
    )


def test_activation_and_rollback_preserve_generation_and_initialization(tmp_path: Path) -> None:
    """Re-running setup never resets a trained model, while rollback restores the base."""
    base = base_revision()
    registry = AdapterRegistry(tmp_path / "active.json", base.scope)
    assert registry.initialize(base).generation == 0
    adapted = candidate(base, tmp_path)
    active = registry.activate(adapted, expected_generation=0)
    assert active.previous == base
    assert registry.initialize(base) == active
    rolled_back = registry.rollback(expected_generation=1)
    assert rolled_back.active == base
    assert rolled_back.previous == adapted
    assert rolled_back.generation == 2


def test_concurrent_promotions_cannot_overwrite_the_winner(tmp_path: Path) -> None:
    """Exactly one evaluator can publish against a given active generation."""
    base = base_revision()
    registry = AdapterRegistry(tmp_path / "active.json", base.scope)
    registry.initialize(base)

    def promote(name: str) -> bool:
        """Try one independently evaluated candidate against generation zero."""
        try:
            registry.activate(candidate(base, tmp_path, name), expected_generation=0)
            return True
        except StaleRegistryError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(promote, ("first", "second"))) == 1
    assert registry.read().generation == 1


def test_scopes_and_old_revision_artifacts_cannot_be_rebound(tmp_path: Path) -> None:
    """Historical IDs remain immutable after leaving both active and previous slots."""
    base = base_revision()
    registry = AdapterRegistry(tmp_path / "active.json", base.scope)
    registry.initialize(base)
    for generation, name in enumerate(("a", "b", "c")):
        registry.activate(candidate(base, tmp_path, name), expected_generation=generation)
    rebound = candidate(base, tmp_path, "a").model_copy(update={"manifest_sha256": "b" * 64})
    with pytest.raises(ValueError, match="already bound"):
        registry.activate(rebound, expected_generation=3)
    wrong_scope = candidate(base, tmp_path).model_copy(
        update={
            "scope": ClaasScope(user_id="other", application_id="claims"),
        }
    )
    with pytest.raises(ValueError, match="scope"):
        registry.activate(wrong_scope, expected_generation=3)
    assert registry.read().generation == 3
    with pytest.raises(ValueError, match="another application"):
        AdapterRegistry(registry.path, wrong_scope.scope).read()


def test_stale_rollback_and_invalid_adapter_references_fail(tmp_path: Path) -> None:
    """Failures leave the last committed pointer intact."""
    base = base_revision()
    registry = AdapterRegistry(tmp_path / "active.json", base.scope)
    registry.initialize(base)
    with pytest.raises(ValueError, match="no previous"):
        registry.rollback(expected_generation=0)
    registry.activate(candidate(base, tmp_path), expected_generation=0)
    with pytest.raises(StaleRegistryError):
        registry.rollback(expected_generation=0)
    with pytest.raises(ValueError, match="absolute"):
        ServingRevision.model_validate(
            {
                **base.model_dump(),
                "adapter_directory": "relative",
                "manifest_sha256": "a" * 64,
            }
        )
    with pytest.raises(ValueError, match="together"):
        ServingRevision.model_validate({**base.model_dump(), "adapter_directory": str(tmp_path)})
