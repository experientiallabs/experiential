"""End-to-end CLI coverage for persisted-dataset W14M Tinker SFT composition."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Never

import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    AssistantAction,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelMessage,
    ModelRecord,
    ModelRequest,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import ProjectStore
from wmo.optimize.model.sft import (
    SFTModelOptimizationConfig,
    TinkerSFTSpec,
    create_sft_model_optimization_config,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.runtime_source_test import _complete, _request
from wmo.optimize.model.sft.selection import load_latest_sft_model_optimization
from wmo.optimize.model.sft.training_test import _TIME, _FakeBackend, _persisted_dataset, _spec
from wmo.runtime.models import ModelConnectionError, RuntimeModelCatalog
from wmo.runtime.models.providers.tinker_sampling import TinkerSample
from wmo.runtime.router import RuntimeInteractionJournal


@dataclass(frozen=True)
class _ConfiguredProject:
    """One local W12 source and write-once W14M project configuration for CLI tests."""

    store: ProjectStore
    config: SFTModelOptimizationConfig


class _RestartSampler:
    """Deterministic sampling seam used to prove a registered alias resolves after restart."""

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Return a fixed response through the runtime's Tinker client adapter.

        Args:
            request: Typed request accepted by the completed trained alias.

        Returns:
            One deterministic sampling result.
        """
        del request
        return TinkerSample(
            output=AssistantAction(content="resolved after restart"),
            served_model_id="tinker://trained-handle",
        )


def _flat_cli_output(output: str) -> str:
    """Remove terminal styling, whitespace, and table borders from CLI output.

    Args:
        output: Rich-formatted text captured by the CLI test runner.

    Returns:
        Stable text suitable for cross-platform substring assertions.
    """
    return "".join(
        character
        for character in unstyle(output)
        if not character.isspace() and character not in "│┃"
    )


def _configured_project(tmp_path: Path, training: TinkerSFTSpec) -> _ConfiguredProject:
    """Persist W12 evidence, a native Tinker connection, and one selected config.

    Args:
        tmp_path: Pytest-owned project directory.
        training: Frozen settings to bind into the selected config.

    Returns:
        Project fixture with runtime evidence and a selected immutable config.
    """
    fixture = _persisted_dataset(tmp_path)
    if training.training_usd_per_million_tokens is None:
        training = training.model_copy(update={"training_usd_per_million_tokens": 100.0})
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={
                "tinker": ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY")
            },
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    config = create_sft_model_optimization_config(
        fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        model_alias="trained",
        tinker_connection="tinker",
        base_model_alias="base",
        training=training,
        created_at=_TIME,
        code_revision="w14m-test",
    )
    write_sft_model_optimization_config(fixture.store, config)
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="runtime-sft-source",
        conversation="runtime-sft-conversation",
        request=_request(ModelMessage(role="user", content="Train on this routed request")),
        output=AssistantAction(content="Completed routed response"),
        now=_TIME,
    )
    return _ConfiguredProject(store=fixture.store, config=config)


def _seed_catalog(store: ProjectStore) -> ModelCatalog:
    """Persist one unrelated provider entry as the shared setup starting point.

    Args:
        store: Project whose catalog is initialized.

    Returns:
        The exact catalog persisted for a stale setup proposal.
    """
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={"judge": ModelRecord(connection="openai", model="judge-model")},
    )
    write_model_catalog(store.model_catalog_path, catalog)
    return catalog


def test_cli_rejects_malformed_release_revision_before_project_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on invalid producer identity before creating project state.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped malformed release revision override.
    """
    root = tmp_path / ".wmo"
    monkeypatch.setenv("WMO_RELEASE_REVISION", "HEAD")

    result = CliRunner().invoke(
        app,
        ["optimize", "model", "support", "--root", str(root)],
    )

    assert result.exit_code == 2
    assert "full lowercase 40-hex" in result.output
    assert not root.exists()


def test_cli_first_run_builds_runtime_w12_and_config_without_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect explicit noninteractive Tinker settings and train from routed interactions.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped fake backend and revision replacements.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={"judge": ModelRecord(connection="openai", model="judge-model")},
        ),
    )
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="first-runtime-source",
        conversation="first-runtime-conversation",
        request=_request(ModelMessage(role="user", content="First routed training request")),
        output=AssistantAction(content="First routed training response"),
        now=_TIME,
    )
    command = importlib.import_module("wmo.cli.model_optimize")
    backend = _FakeBackend(conservative_cost_per_batch=0.10)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "automatic-cli-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            fixture.store.paths.project_id,
            "--root",
            str(fixture.store.paths.root),
            "--tinker-connection",
            "tinker-local",
            "--tinker-api-key-env",
            "TINKER_API_KEY",
            "--base-model-alias",
            "base",
            "--base-model",
            "test-base-model",
            "--maximum-cost-usd",
            "1.0",
            "--training-usd-per-million-tokens",
            "100",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert backend.open_resume_paths == [None]
    assert fixture.store.load_project().model_optimization_config is None
    latest = load_latest_sft_model_optimization(fixture.store)
    assert latest is not None
    catalog = load_model_catalog(fixture.store.model_catalog_path)
    assert catalog.connections["tinker-local"].provider == "tinker"
    assert catalog.connections["tinker-local"].api_key_env == "TINKER_API_KEY"
    assert catalog.models["base"] == ModelRecord(connection="tinker-local", model="test-base-model")
    config = command.load_sft_model_optimization_config(fixture.store, latest.config.artifact_id)
    constructed = []

    def sampler_factory(model, api_key, base_url):  # noqa: ANN001, ANN202
        """Record the restarted resolver's exact selected connection inputs.

        Args:
            model: Immutable registered trained-model snapshot.
            api_key: Credential resolved from the persisted environment-variable name.
            base_url: Optional endpoint from the selected Tinker connection.

        Returns:
            Deterministic sampling seam for the restarted runtime.
        """
        constructed.append((model.model_id, api_key, base_url))
        return _RestartSampler()

    resolver = RuntimeModelCatalog(
        load_model_catalog(fixture.store.model_catalog_path),
        environment={"TINKER_API_KEY": "runtime-secret"},
        tinker_sampler_factory=sampler_factory,
    )
    resolved = resolver.resolve(config.model_alias)

    assert constructed == [(catalog.models[config.model_alias].model, "runtime-secret", None)]
    assert (
        resolved.client.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="Hello"),))
        ).output.content
        == "resolved after restart"
    )
    drifted_connections = dict(catalog.connections)
    drifted_connections["tinker-local"] = catalog.connections["tinker-local"].model_copy(
        update={"api_key_env": "OTHER_TINKER_API_KEY"}
    )
    write_model_catalog(
        fixture.store.model_catalog_path,
        catalog.model_copy(update={"connections": drifted_connections}),
    )
    credential_reads = []

    def forbidden_credential_read(*args: object, **kwargs: object) -> Never:
        """Fail if connection drift reaches the credential boundary.

        Args:
            args: Unexpected positional credential lookup inputs.
            kwargs: Unexpected keyword credential lookup inputs.
        """
        del args, kwargs
        credential_reads.append(True)
        raise AssertionError("credential lookup must follow provenance verification")

    monkeypatch.setattr(
        "wmo.runtime.models.registry.read_connection_api_key",
        forbidden_credential_read,
    )
    restarted_after_drift = RuntimeModelCatalog(
        load_model_catalog(fixture.store.model_catalog_path),
        environment={"OTHER_TINKER_API_KEY": "drifted-secret"},
        tinker_sampler_factory=sampler_factory,
    )

    with pytest.raises(ModelConnectionError, match="connection metadata drifted"):
        restarted_after_drift.resolve(config.model_alias)
    assert credential_reads == []
    assert len(constructed) == 1


def test_cli_first_run_noninteractive_reports_all_required_tinker_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report complete setup remediation before backend composition or consent.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped replacement that fails if backend composition is reached.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={"judge": ModelRecord(connection="openai", model="judge-model")},
        ),
    )
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="missing-setup",
        conversation="missing-setup-conversation",
        request=_request(ModelMessage(role="user", content="Completed request")),
        output=AssistantAction(content="Completed response"),
        now=_TIME,
    )
    command = importlib.import_module("wmo.cli.model_optimize")
    backend_calls = []
    artifact_ids = fixture.store.artifacts.list_ids()
    catalog_bytes = fixture.store.model_catalog_path.read_bytes()

    def forbidden_backend(*args: object) -> Never:
        """Fail if incomplete first-run setup reaches backend composition.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("backend composition must not run")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            fixture.store.paths.project_id,
            "--root",
            str(fixture.store.paths.root),
        ],
    )

    flat = _flat_cli_output(result.output)
    assert result.exit_code == 2
    assert "--tinker-connection" in flat
    assert "--tinker-api-key-env" in flat
    assert "--base-model-alias" in flat
    assert "--base-model" in flat
    assert "--training-usd-per-million-tokens" in flat
    assert backend_calls == []
    assert load_latest_sft_model_optimization(fixture.store) is None
    assert fixture.store.artifacts.list_ids() == artifact_ids
    assert fixture.store.model_catalog_path.read_bytes() == catalog_bytes


def test_first_run_setup_cancellation_writes_no_catalog_or_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave catalog, artifacts, and latest selection unchanged when setup is declined.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped interactive prompt replacements.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={"judge": ModelRecord(connection="openai", model="judge-model")},
        ),
    )
    command = importlib.import_module("wmo.cli.model_optimize")
    catalog_bytes = fixture.store.model_catalog_path.read_bytes()
    artifact_ids = fixture.store.artifacts.list_ids()
    answers = iter(("tinker-local", "base", "TINKER_API_KEY", "test-base-model"))
    backend_calls = []
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="cancelled-setup",
        conversation="cancelled-setup-conversation",
        request=_request(ModelMessage(role="user", content="Completed request")),
        output=AssistantAction(content="Completed response"),
        now=_TIME,
    )

    class _TTY:
        """Minimal stdin contract used only to select interactive setup behavior."""

        def isatty(self) -> bool:
            """Report that setup may prompt interactively."""
            return True

    monkeypatch.setattr(command.sys, "stdin", _TTY())
    monkeypatch.setattr(command.Prompt, "ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(command.Confirm, "ask", lambda *args, **kwargs: False)

    def forbidden_backend(*args: object) -> Never:
        """Fail if declined setup imports or constructs the Tinker backend.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("backend composition must not run")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)

    with pytest.raises(typer.Abort):
        command.optimize_model(
            project=fixture.store.paths.project_id,
            root=fixture.store.paths.root,
            yes=False,
            tinker_connection=None,
            tinker_api_key_env=None,
            base_model_alias=None,
            base_model=None,
            maximum_cost_usd=25.0,
            training_usd_per_million_tokens=100.0,
        )

    assert fixture.store.model_catalog_path.read_bytes() == catalog_bytes
    assert fixture.store.artifacts.list_ids() == artifact_ids
    assert load_latest_sft_model_optimization(fixture.store) is None
    assert backend_calls == []


def test_cli_runs_fake_w13_then_idempotently_resumes_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The customer path trains once with --yes and later verifies a completed run without spend."""
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    first_backend = _FakeBackend(conservative_cost_per_batch=0.10)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: first_backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "w14m-test")
    attempts = []
    emitted = []
    receipts: set[str] = set()

    def capture(event, completion_id, properties, *, root):  # noqa: ANN001, ANN202
        """Record one telemetry attempt while emitting each completion identity once.

        Args:
            event: Telemetry event name.
            completion_id: Durable completion identity used for deduplication.
            properties: Aggregate completion properties.
            root: Project root receiving the local telemetry receipt.

        Returns:
            Whether this completion identity emitted for the first time.
        """
        attempt = (event, completion_id, properties, root)
        attempts.append(attempt)
        if completion_id in receipts:
            return False
        receipts.add(completion_id)
        emitted.append(attempt)
        return True

    monkeypatch.setattr(command, "capture_completion_once", capture)
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert first.exit_code == 0, first.output
    assert "registered model alias" in first.output
    assert first_backend.cost_calls > 0
    assert first_backend.open_resume_paths == [None]
    second_backend = _FakeBackend(conservative_cost_per_batch=0.10)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: second_backend)

    second = runner.invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
        ],
    )

    assert second.exit_code == 0, second.output
    assert "was already" in second.output
    assert "registered." in second.output
    assert second_backend.open_resume_paths == []
    assert len(attempts) == 2
    assert len(emitted) == 1
    assert attempts[0][0] == "wmo sft completed"
    assert len({completion_id for _event, completion_id, _properties, _root in attempts}) == 1
    assert all(
        event_root == configured.store.paths.root
        for _event, _completion_id, _properties, event_root in attempts
    )
    assert all(
        properties["training_step_count"] == 1
        for _event, _completion_id, properties, _root in attempts
    )
    assert all(
        properties["cost_usd"] == pytest.approx(0.1)
        for _event, _completion_id, properties, _root in attempts
    )


def test_cli_crash_after_sft_receipt_replays_without_duplicate_event_or_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable catalog registration and telemetry receipt survive the final CLI crash window."""
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    first_backend = _FakeBackend(conservative_cost_per_batch=0.10)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: first_backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "w14m-test")
    attempts = []
    emitted = []
    receipts: set[str] = set()

    def crash_once(event, completion_id, properties, *, root):  # noqa: ANN001, ANN202
        """Persist one telemetry receipt, then simulate a crash on its first emission.

        Args:
            event: Telemetry event name.
            completion_id: Durable completion identity used for deduplication.
            properties: Aggregate completion properties.
            root: Project root receiving the local telemetry receipt.

        Returns:
            False when the receipt already exists.

        Raises:
            RuntimeError: The first unique receipt simulates a post-write process crash.
        """
        attempt = (event, completion_id, properties, root)
        attempts.append(attempt)
        if completion_id in receipts:
            return False
        receipts.add(completion_id)
        emitted.append(attempt)
        raise RuntimeError("simulated crash after telemetry receipt")

    monkeypatch.setattr(command, "capture_completion_once", crash_once)
    runner = CliRunner()
    arguments = [
        "optimize",
        "model",
        configured.store.paths.project_id,
        "--root",
        str(configured.store.paths.root),
        "--yes",
    ]

    first = runner.invoke(app, arguments)
    assert first.exit_code == 1
    assert isinstance(first.exception, RuntimeError)
    assert "after telemetry receipt" in str(first.exception)
    assert first_backend.open_resume_paths == [None]

    replay_backend = _FakeBackend(conservative_cost_per_batch=0.10)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: replay_backend)
    replay = runner.invoke(app, arguments[:-1])

    assert replay.exit_code == 0, replay.output
    assert "was already" in replay.output
    assert "registered." in replay.output
    assert replay_backend.open_resume_paths == []
    assert len(attempts) == 2
    assert len(emitted) == 1


def test_cli_yes_does_not_bypass_the_unsupported_budget_estimate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing --yes cannot turn an unpriceable maximum-cost Tinker plan into a dispatch."""
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    backend = _FakeBackend(conservative_cost_per_batch=None)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "w14m-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "nosupportedconservativecostestimate" in _flat_cli_output(result.output)
    assert backend.open_resume_paths == []


def test_cli_empty_journal_fails_before_backend_or_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an empty runtime source before SDK composition or user spend consent.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped replacements that fail if later execution boundaries are reached.
    """
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    configured.store.paths.runtime_journal.unlink()
    command = importlib.import_module("wmo.cli.model_optimize")
    backend_calls = []
    consent_calls = []

    def forbidden_backend(*args: object) -> Never:
        """Record and fail any backend composition after empty-source validation.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("backend composition must not run")

    def forbidden_consent(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        """Record and fail any consent prompt after empty-source validation.

        Args:
            args: Unexpected positional consent inputs.
            kwargs: Unexpected keyword consent inputs.

        Raises:
            AssertionError: Always, because empty input must fail before consent.
        """
        consent_calls.append((args, kwargs))
        raise AssertionError("consent must not run")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)
    monkeypatch.setattr(command, "require_spend_consent", forbidden_consent)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "nointeractions" in _flat_cli_output(result.output)
    assert backend_calls == []
    assert consent_calls == []


def test_declined_spend_consent_does_not_resolve_credentials_or_construct_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep credential lookup, SDK import, and ServiceClient construction after consent.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped consent and backend-boundary replacements.
    """
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    backend_calls = []

    def forbidden_backend(*args: object) -> Never:
        """Fail if declined spend consent reaches credential or SDK composition.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("backend composition must follow consent")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)
    monkeypatch.setattr(command, "require_spend_consent", lambda *args, **kwargs: False)
    monkeypatch.setattr(command, "_current_revision", lambda: "declined-consent-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "was not started" in result.output
    assert backend_calls == []


def test_connection_drift_after_consent_fails_before_credential_or_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recheck frozen connection metadata after consent and before external construction.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped consent, credential, and SDK constructor replacements.
    """
    import tinker

    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    credential_reads = []
    service_calls = []

    def consent_with_connection_drift(*args: object, **kwargs: object) -> bool:
        """Mutate only the selected credential reference at the consent boundary.

        Args:
            args: Consent inputs accepted without inspection.
            kwargs: Consent inputs accepted without inspection.

        Returns:
            ``True`` after persisting the adversarial drift.
        """
        del args, kwargs
        catalog = load_model_catalog(configured.store.model_catalog_path)
        connections = dict(catalog.connections)
        connections["tinker"] = connections["tinker"].model_copy(
            update={"api_key_env": "DRIFTED_TINKER_API_KEY"}
        )
        write_model_catalog(
            configured.store.model_catalog_path,
            catalog.model_copy(update={"connections": connections}),
        )
        return True

    def forbidden_credential_read(*args: object, **kwargs: object) -> Never:
        """Fail if drift validation occurs after credential lookup.

        Args:
            args: Unexpected positional credential inputs.
            kwargs: Unexpected keyword credential inputs.
        """
        del args, kwargs
        credential_reads.append(True)
        raise AssertionError("credential lookup must follow drift validation")

    def forbidden_service_client(*args: object, **kwargs: object) -> Never:
        """Fail if drift validation occurs after ServiceClient construction.

        Args:
            args: Unexpected positional SDK inputs.
            kwargs: Unexpected keyword SDK inputs.
        """
        del args, kwargs
        service_calls.append(True)
        raise AssertionError("ServiceClient construction must follow drift validation")

    monkeypatch.setattr(command, "require_spend_consent", consent_with_connection_drift)
    monkeypatch.setattr(command, "read_connection_api_key", forbidden_credential_read)
    monkeypatch.setattr(tinker, "ServiceClient", forbidden_service_client)
    monkeypatch.setattr(command, "_current_revision", lambda: "post-consent-drift-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "metadatadriftedbeforecredentialresolution" in _flat_cli_output(result.output)
    assert credential_reads == []
    assert service_calls == []


def test_first_run_explicit_zero_training_price_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept zero only through the explicit price flag and persist it in immutable settings.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped backend and revision replacements.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={"judge": ModelRecord(connection="openai", model="judge-model")},
        ),
    )
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="free-training",
        conversation="free-training-conversation",
        request=_request(ModelMessage(role="user", content="Train this request")),
        output=AssistantAction(content="Train this response"),
        now=_TIME,
    )
    command = importlib.import_module("wmo.cli.model_optimize")
    backend = _FakeBackend(cost_per_batch=0.0, conservative_cost_per_batch=0.0)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda *_args: backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "explicit-zero-price-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            fixture.store.paths.project_id,
            "--root",
            str(fixture.store.paths.root),
            "--tinker-connection",
            "tinker-local",
            "--tinker-api-key-env",
            "TINKER_API_KEY",
            "--base-model-alias",
            "base",
            "--base-model",
            "test-base-model",
            "--training-usd-per-million-tokens",
            "0",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    latest = load_latest_sft_model_optimization(fixture.store)
    assert latest is not None
    config = command.load_sft_model_optimization_config(fixture.store, latest.config.artifact_id)
    assert config.training.training_usd_per_million_tokens == 0.0


def test_schedule_ceiling_refuses_before_tinker_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject the full local token-price reservation before importing or constructing Tinker.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped backend replacement proving the SDK boundary is not reached.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={"judge": ModelRecord(connection="openai", model="judge-model")},
        ),
    )
    _complete(
        RuntimeInteractionJournal(fixture.store.paths),
        key="over-budget",
        conversation="over-budget-conversation",
        request=_request(ModelMessage(role="user", content="Expensive request")),
        output=AssistantAction(content="Expensive response"),
        now=_TIME,
    )
    command = importlib.import_module("wmo.cli.model_optimize")
    backend_calls = []

    def forbidden_backend(*args: object) -> Never:
        """Fail if an over-budget local schedule reaches Tinker construction.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("Tinker backend must not be constructed")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "local-price-ceiling-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            fixture.store.paths.project_id,
            "--root",
            str(fixture.store.paths.root),
            "--tinker-connection",
            "tinker-local",
            "--tinker-api-key-env",
            "TINKER_API_KEY",
            "--base-model-alias",
            "base",
            "--base-model",
            "test-base-model",
            "--maximum-cost-usd",
            "1",
            "--training-usd-per-million-tokens",
            "1000000",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "exceedsmaximum_cost_usd" in _flat_cli_output(result.output)
    assert backend_calls == []


def test_selected_price_replay_rejects_explicit_drift_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an invocation price that differs from the selected immutable config.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped backend replacement proving drift fails locally.
    """
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    backend_calls = []

    def forbidden_backend(*args: object) -> Never:
        """Fail if immutable price drift reaches Tinker construction.

        Args:
            args: Unexpected project and connection arguments from backend composition.
        """
        del args
        backend_calls.append(True)
        raise AssertionError("Tinker backend must not be constructed")

    monkeypatch.setattr(command, "_compose_tinker_backend", forbidden_backend)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--training-usd-per-million-tokens",
            "101",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "differsfromtheselectedimmutable" in _flat_cli_output(result.output)
    assert backend_calls == []


def test_tinker_backend_uses_the_selected_credential_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construct training with the exact cataloged key name at the authorized boundary.

    Args:
        tmp_path: Pytest-owned project directory.
        monkeypatch: Scoped environment and SDK constructor replacements.
    """
    import tinker

    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    constructed = []

    def service_client(*, api_key: str, base_url: str | None = None) -> object:
        """Capture only whether the named credential reaches the exact SDK constructor.

        Args:
            api_key: Credential resolved from the selected environment-variable name.
            base_url: Optional selected endpoint.

        Returns:
            Opaque caller-owned service seam accepted by the backend wrapper.
        """
        constructed.append((api_key, base_url))
        return object()

    monkeypatch.setenv("TINKER_API_KEY", "training-secret")
    monkeypatch.setattr(tinker, "ServiceClient", service_client)

    backend = command._compose_tinker_backend(
        configured.store,
        "tinker",
        configured.config.connection_config_sha256,
    )

    assert isinstance(backend, command.TinkerTrainerBackend)
    assert constructed == [("training-secret", None)]


def test_catalog_setup_preserves_unrelated_concurrent_additions(tmp_path: Path) -> None:
    """Merge a confirmed Tinker selection without erasing unrelated catalog drift.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    fixture = _persisted_dataset(tmp_path)
    command = importlib.import_module("wmo.cli.model_optimize")
    observed = _seed_catalog(fixture.store)
    concurrent = observed.model_copy(
        update={
            "connections": {
                **observed.connections,
                "other": ConnectionConfig(provider="openai"),
            },
            "models": {
                **observed.models,
                "other-model": ModelRecord(connection="other", model="other-model-id"),
            },
        }
    )
    write_model_catalog(fixture.store.model_catalog_path, concurrent)

    command._persist_tinker_selection(
        fixture.store,
        observed_catalog_sha256=sha256_json(observed),
        connection_name="tinker-local",
        api_key_env="TINKER_API_KEY",
        alias="base",
        resolved_model="test-base-model",
    )

    updated = load_model_catalog(fixture.store.model_catalog_path)
    assert updated.connections["other"] == concurrent.connections["other"]
    assert updated.models["other-model"] == concurrent.models["other-model"]
    assert updated.connections["tinker-local"] == ConnectionConfig(
        provider="tinker", api_key_env="TINKER_API_KEY"
    )
    assert updated.models["base"] == ModelRecord(connection="tinker-local", model="test-base-model")


@pytest.mark.parametrize("conflict", ["connection", "alias"])
def test_catalog_setup_rejects_concurrent_target_drift(tmp_path: Path, conflict: str) -> None:
    """Reject a stale setup when a concurrent writer changes either confirmed target.

    Args:
        tmp_path: Pytest-owned project directory.
        conflict: Confirmed catalog entry changed by the simulated concurrent writer.
    """
    fixture = _persisted_dataset(tmp_path)
    command = importlib.import_module("wmo.cli.model_optimize")
    observed = _seed_catalog(fixture.store)
    connections = dict(observed.connections)
    models = dict(observed.models)
    if conflict == "connection":
        connections["tinker-local"] = ConnectionConfig(provider="openai")
    else:
        connections["tinker-local"] = ConnectionConfig(
            provider="tinker", api_key_env="TINKER_API_KEY"
        )
        models["base"] = ModelRecord(connection="tinker-local", model="different-model")
    write_model_catalog(
        fixture.store.model_catalog_path,
        observed.model_copy(update={"connections": connections, "models": models}),
    )

    with pytest.raises(typer.BadParameter, match="changed before setup commit"):
        command._persist_tinker_selection(
            fixture.store,
            observed_catalog_sha256=sha256_json(observed),
            connection_name="tinker-local",
            api_key_env="TINKER_API_KEY",
            alias="base",
            resolved_model="test-base-model",
        )


def test_catalog_setup_is_idempotent_for_same_concurrent_selection(tmp_path: Path) -> None:
    """Allow two stale first-run proposals to commit the exact same Tinker selection.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    fixture = _persisted_dataset(tmp_path)
    command = importlib.import_module("wmo.cli.model_optimize")
    observed = _seed_catalog(fixture.store)
    observed_sha256 = sha256_json(observed)
    arguments = {
        "observed_catalog_sha256": observed_sha256,
        "connection_name": "tinker-local",
        "api_key_env": "TINKER_API_KEY",
        "alias": "base",
        "resolved_model": "test-base-model",
    }

    command._persist_tinker_selection(fixture.store, **arguments)
    first_bytes = fixture.store.model_catalog_path.read_bytes()
    command._persist_tinker_selection(fixture.store, **arguments)

    assert fixture.store.model_catalog_path.read_bytes() == first_bytes


def test_catalog_setup_preserves_compatible_base_model_metadata(tmp_path: Path) -> None:
    """Retain harmless revision and capability metadata on an already matching base alias.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    fixture = _persisted_dataset(tmp_path)
    command = importlib.import_module("wmo.cli.model_optimize")
    existing_record = ModelRecord(
        connection="tinker-local",
        model="test-base-model",
        revision="provider-revision-1",
        capabilities=ModelCapabilities(
            supports_tools=True,
            context_window_tokens=32_768,
        ),
    )
    catalog = ModelCatalog(
        connections={
            "tinker-local": ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY")
        },
        models={"base": existing_record},
    )
    write_model_catalog(fixture.store.model_catalog_path, catalog)
    catalog_bytes = fixture.store.model_catalog_path.read_bytes()

    command._persist_tinker_selection(
        fixture.store,
        observed_catalog_sha256=sha256_json(catalog),
        connection_name="tinker-local",
        api_key_env="TINKER_API_KEY",
        alias="base",
        resolved_model="test-base-model",
    )

    assert fixture.store.model_catalog_path.read_bytes() == catalog_bytes
    assert load_model_catalog(fixture.store.model_catalog_path).models["base"] == existing_record
