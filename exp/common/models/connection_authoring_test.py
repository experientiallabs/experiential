"""Tests for provider authoring without optimizer role assignment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError

from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ProviderConnection,
    ProviderConnectionAuthoringError,
    ProviderSetup,
    configure_provider_connections,
    connection_authoring,
    load_model_catalog,
    sync_provider_models,
    write_model_catalog,
)


def test_role_free_connection_authoring_creates_connections_only_catalog(tmp_path: Path) -> None:
    """Gateway setup can persist one real BYOK connection before choosing a model or role."""
    path = tmp_path / "models.toml"

    configured = configure_provider_connections(
        path,
        (
            ProviderConnection(
                name="openai",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
            ),
        ),
    )

    assert configured.models == {}
    assert configured.roles.candidates == ()
    assert load_model_catalog(path) == configured


def test_role_free_authoring_preserves_models_and_rejects_connection_rebinding(
    tmp_path: Path,
) -> None:
    """Connection updates preserve unrelated state and cannot move an existing model endpoint."""
    path = tmp_path / "models.toml"
    original = ModelCatalog(
        connections={
            "openai": ProviderConnection(
                name="openai", provider="openai", api_key_env="OPENAI_API_KEY"
            ).catalog_config()
        },
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True),
            )
        },
    )
    write_model_catalog(path, original)

    configure_provider_connections(
        path,
        (
            ProviderConnection(
                name="anthropic",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
            ),
        ),
    )
    with pytest.raises(ProviderConnectionAuthoringError, match="used by model aliases"):
        configure_provider_connections(
            path,
            (
                ProviderConnection(
                    name="openai",
                    provider="openai-compatible",
                    api_key_env="COMPATIBLE_API_KEY",
                    base_url="https://models.example.test/v1",
                ),
            ),
            replace=True,
        )

    loaded = load_model_catalog(path)
    assert loaded.models == original.models
    assert set(loaded.connections) == {"anthropic", "openai"}


def test_optimizer_provider_setup_still_requires_all_build_roles() -> None:
    """The new runtime authoring seam does not weaken build and optimize role validation."""
    with pytest.raises(ValidationError, match="world_model"):
        ProviderSetup.model_validate(
            {
                "connections": (
                    ProviderConnection(
                        name="openai",
                        provider="openai",
                        api_key_env="OPENAI_API_KEY",
                    ),
                )
            }
        )


def _hosted_models() -> tuple[ProviderConnection, dict[str, ModelRecord]]:
    """Return one synthetic hosted endpoint and its synchronized model record."""
    connection = ProviderConnection(
        name="hosted",
        provider="openai-compatible",
        api_key_env="HOSTED_API_KEY",
        base_url="https://preview.example.test/v1",
    )
    return connection, {
        "hosted-chat": ModelRecord(
            connection=connection.name,
            model="chat",
            billing_source=BillingSource.HOST_MANAGED,
            capabilities=ModelCapabilities(supports_completions=True),
        )
    }


def _write_original_catalog(path: Path) -> bytes:
    """Seed unrelated catalog state and a comment whose exact bytes must survive rollback."""
    original = ModelCatalog(
        connections={
            "openai": ProviderConnection(
                name="openai", provider="openai", api_key_env="OPENAI_API_KEY"
            ).catalog_config()
        },
        models={
            "coding": ModelRecord(
                connection="openai",
                model="coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True),
            )
        },
    )
    write_model_catalog(path, original)
    content = b"# Operator formatting must survive a failed commit.\n" + path.read_bytes()
    path.write_bytes(content)
    return content


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_failed_model_commit_restores_exact_catalog_target(
    tmp_path: Path, existing: bool, symlink: bool
) -> None:
    """Failed persistence restores exact bytes or absence without replacing a catalog link."""
    path = tmp_path / "models.toml"
    target = tmp_path / "shared.toml" if symlink else path
    original = _write_original_catalog(target) if existing else None
    if symlink:
        path.symlink_to(target)
    connection, models = _hosted_models()
    failure = OSError("synthetic commit failure")

    def fail_commit() -> None:
        """Prove the updated catalog is installed before reporting an ordinary save failure."""
        assert load_model_catalog(path).connections[connection.name] == connection.catalog_config()
        raise failure

    with pytest.raises(OSError) as raised:
        sync_provider_models(path, connection=connection, models=models, on_commit=fail_commit)

    assert raised.value is failure
    assert path.is_symlink() is symlink
    if original is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == original


def test_successful_model_commit_runs_once_after_catalog_write(tmp_path: Path) -> None:
    """The persistence callback observes the installed catalog once and retains other models."""
    path = tmp_path / "models.toml"
    _write_original_catalog(path)
    connection, models = _hosted_models()
    committed: list[ModelCatalog] = []

    def commit() -> None:
        """Observe the exact catalog made visible to the final persistence step."""
        committed.append(load_model_catalog(path))

    result = sync_provider_models(path, connection=connection, models=models, on_commit=commit)

    assert committed == [result]
    assert load_model_catalog(path) == result
    assert set(result.models) == {"coding", "hosted-chat"}


def test_failed_model_commit_preserves_a_concurrent_catalog_update(tmp_path: Path) -> None:
    """A waiting catalog writer applies its update only after failed persistence rolls back."""
    path = tmp_path / "models.toml"
    original = _write_original_catalog(path)
    connection, models = _hosted_models()
    commit_entered = Event()
    release_commit = Event()
    writer_attempted = Event()
    writer_finished = Event()

    def fail_commit() -> None:
        """Keep the synchronization lock held while another authoring operation starts."""
        commit_entered.set()
        assert release_commit.wait(timeout=5)
        raise OSError("synthetic commit failure")

    def synchronize() -> ModelCatalog:
        """Run the failing catalog update in a worker."""
        return sync_provider_models(
            path, connection=connection, models=models, on_commit=fail_commit
        )

    def add_unrelated_connection() -> ModelCatalog:
        """Attempt a normal catalog update while synchronization still owns the lock."""
        writer_attempted.set()
        result = configure_provider_connections(
            path,
            (
                ProviderConnection(
                    name="anthropic", provider="anthropic", api_key_env="ANTHROPIC_API_KEY"
                ),
            ),
        )
        writer_finished.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        syncing = executor.submit(synchronize)
        try:
            assert commit_entered.wait(timeout=5)
            writing = executor.submit(add_unrelated_connection)
            assert writer_attempted.wait(timeout=5)
            assert not writer_finished.wait(timeout=0.1)
        finally:
            release_commit.set()
        with pytest.raises(OSError, match="synthetic commit failure"):
            syncing.result(timeout=5)
        updated = writing.result(timeout=5)

    assert load_model_catalog(path) == updated
    assert set(updated.connections) == {"openai", "anthropic"}
    assert set(updated.models) == {"coding"}
    assert path.read_bytes() != original


def test_failed_model_commit_reports_failed_rollback_without_callback_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second persistence failure clearly requires recovery without exposing exception data."""
    path = tmp_path / "models.toml"
    _write_original_catalog(path)
    connection, models = _hosted_models()

    def fail_commit() -> None:
        """Simulate a downstream error with private context that must not enter recovery text."""
        raise OSError("private callback detail")

    def fail_restore(path: Path, payload: bytes) -> None:
        """Report a failed rollback while leaving the newly written catalog in place."""
        raise OSError("private filesystem detail")

    monkeypatch.setattr(connection_authoring, "write_bytes_atomic", fail_restore)
    with pytest.raises(
        RuntimeError, match="Check models.toml and credential configuration"
    ) as raised:
        sync_provider_models(path, connection=connection, models=models, on_commit=fail_commit)

    assert "private" not in str(raised.value)
    assert raised.value.__suppress_context__
    assert connection.name in load_model_catalog(path).connections
