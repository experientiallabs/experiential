"""Automatic routed-interaction W12 preparation and bounded managed W13 SFT."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import typer
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt

from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import Sha256, sha256_json
from exp.common.core.locks import file_write_lock
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCatalog,
    ModelCatalogError,
    ModelRecord,
    NumericMeasurement,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.observability.telemetry import capture_completion_once
from exp.common.project import ProjectStore, ProjectStoreError
from exp.common.release_revision import installed_release_revision
from exp.optimize.model.sft import (
    AutomaticSFTPreparationError,
    InitialSFTModelOptimizationSettings,
    SFTModelOptimizationError,
    SFTModelOptimizationPreflightError,
    TinkerSFTDependencyError,
    TinkerTrainerBackend,
    TrainerBackend,
    accept_runtime_sft_model_optimization,
    load_sft_model_optimization_config,
    preflight_sft_model_optimization,
    prepare_runtime_sft_model_optimization,
    require_completed_runtime_interactions,
    run_sft_model_optimization,
)
from exp.optimize.model.sft.selection import load_latest_sft_model_optimization
from exp.optimize.model.sft.training_contracts import (
    TinkerSFTSpec,
    conservative_training_step_cost,
)
from exp.runtime.models.credentials import ModelCredentialError, read_connection_api_key

_console = Console(theme=EXP_THEME)
_DEFAULT_LORA_RANK = 32
_DEFAULT_LEARNING_RATE = 0.0001
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_EPOCHS = 1
_DEFAULT_CHECKPOINT_EVERY_STEPS = 10
_DEFAULT_MAXIMUM_DATUM_TOKENS = 4096
_DEFAULT_MAXIMUM_COST_USD = 25.0
_USAGE_ERRORS = (
    ModelCatalogError,
    ModelCredentialError,
    SFTModelOptimizationError,
    TinkerSFTDependencyError,
    ValueError,
)


def optimize_model(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    tinker_connection: str | None = typer.Option(
        None,
        "--tinker-connection",
        help="Existing or new local name for the explicit native Tinker connection.",
    ),
    tinker_api_key_env: str | None = typer.Option(
        None,
        "--tinker-api-key-env",
        help="Environment-variable name containing the Tinker key. The key is never persisted.",
    ),
    base_model_alias: str | None = typer.Option(
        None,
        "--base-model-alias",
        help="Existing or new local alias for the exact Tinker base model.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Exact Tinker model ID, required when the base alias is not configured.",
    ),
    maximum_cost_usd: float | None = typer.Option(
        None,
        "--maximum-cost-usd",
        min=0.01,
        help="Immutable managed training cost cap. First-run default: $25.00.",
    ),
    training_usd_per_million_tokens: float | None = typer.Option(
        None,
        "--training-usd-per-million-tokens",
        min=0,
        help="Explicit Tinker training price used for conservative local reservation.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Require complete flags and never ask setup or cost questions.",
    ),
) -> None:
    """Build routed interactions into W12 and run bounded W13 SFT automatically.

    The command seals the current validated runtime journal prefix, reuses or creates its
    immutable W12 dataset and bounded config, then validates the complete local input graph before
    cost authorization. Failed and disconnected interactions never become training targets. A
    trained alias is registered only after recursively verifying W13's completed result and
    opaque sampling handle.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local ``.exp`` root containing the project and ``models.toml``.
        yes: Explicit confirmation for an in-budget estimate, never a validation or risk bypass.
        tinker_connection: Native Tinker connection name used for first-run setup.
        tinker_api_key_env: Environment-variable name used by training and later sampling.
        base_model_alias: Local alias for the exact first-run Tinker base model.
        base_model: Exact Tinker model ID when the alias is not already configured.
        maximum_cost_usd: Optional first-run or consistency-checked managed-training cost cap.
        training_usd_per_million_tokens: Explicit model-specific training price. Zero is accepted
            only when supplied by the user or entered during interactive setup.
        non_interactive: Require complete flags and refuse every prompt.

    Raises:
        typer.BadParameter: Local configuration, preflight, W13, or registration is unsafe.
    """
    started = time.monotonic()
    created_at = datetime.now(UTC)
    with usage_error(*_USAGE_ERRORS, ProjectStoreError, AutomaticSFTPreparationError):
        code_revision = installed_release_revision()
        store = ProjectStore(root, project)
        require_completed_runtime_interactions(store)
        initial_settings = _initial_settings(
            store,
            project=project,
            tinker_connection=tinker_connection,
            tinker_api_key_env=tinker_api_key_env,
            base_model_alias=base_model_alias,
            base_model=base_model,
            maximum_cost_usd=maximum_cost_usd,
            training_usd_per_million_tokens=training_usd_per_million_tokens,
            non_interactive=non_interactive,
        )
        preparation = prepare_runtime_sft_model_optimization(
            store,
            created_at=created_at,
            code_revision=code_revision,
            initial_settings=initial_settings,
        )
        config = preparation.config
        config_id = config.config_id
        local_preflight = preflight_sft_model_optimization(
            store,
            config_id,
            _LocalPreflightBackend(),
            code_revision=code_revision,
        )
        if local_preflight.completed_result is None and not preparation.accepted:
            preparation = prepare_runtime_sft_model_optimization(
                store,
                created_at=created_at,
                code_revision=code_revision,
            )
            config = preparation.config
            config_id = config.config_id
            local_preflight = preflight_sft_model_optimization(
                store,
                config_id,
                _LocalPreflightBackend(),
                code_revision=code_revision,
            )
        backend: TrainerBackend = _LocalPreflightBackend()
        preflight = local_preflight

    if preflight.completed_result is None:
        assert preflight.conservative_schedule_cost_usd is not None
        assert config.training.maximum_datum_tokens is not None
        assert config.training.training_usd_per_million_tokens is not None
        estimate = preflight.conservative_schedule_cost_usd.value
    else:
        estimate = 0.0
    if not require_spend_consent(
        _console,
        root=root,
        yes=yes,
        estimated_cost_usd=estimate,
        command=f"exp optimize model {project}",
        non_interactive=non_interactive,
        previously_confirmed=preparation.accepted and preflight.completed_result is None,
    ):
        _console.print("Managed Tinker SFT was not started.")
        return
    if preflight.completed_result is None and not preparation.accepted:
        with usage_error(AutomaticSFTPreparationError):
            accept_runtime_sft_model_optimization(
                store,
                preparation,
                created_at=created_at,
                code_revision=code_revision,
            )
    if preflight.completed_result is None:
        with usage_error(*_USAGE_ERRORS):
            backend = _compose_tinker_backend(
                store,
                config.tinker_connection,
                config.connection_config_sha256,
            )
            preflight = preflight_sft_model_optimization(
                store,
                config_id,
                backend,
                code_revision=code_revision,
            )
    with usage_error(SFTModelOptimizationError):
        completed = run_sft_model_optimization(
            store,
            config_id,
            backend,
            created_at=created_at,
            code_revision=code_revision,
            preflight=preflight,
        )
    properties: dict[str, bool | int | float] = {
        "success": True,
        "training_step_count": completed.training_result.training_step_count,
        "duration_seconds": max(time.monotonic() - started, 0.0),
    }
    if completed.training_result.total_cost_usd is not None:
        properties["cost_usd"] = completed.training_result.total_cost_usd.value
    capture_completion_once(
        "exp sft completed",
        completed.training_result.result_id,
        properties,
        root=root,
    )
    if completed.catalog_updated:
        _console.print(
            f"Verified completed W13 SFT and registered model alias {config.model_alias!r}."
        )
    else:
        _console.print(
            "Verified completed W13 SFT; model alias "
            f"{config.model_alias!r} was already registered."
        )


def _initial_settings(
    store: ProjectStore,
    *,
    project: str,
    tinker_connection: str | None,
    tinker_api_key_env: str | None,
    base_model_alias: str | None,
    base_model: str | None,
    maximum_cost_usd: float | None,
    training_usd_per_million_tokens: float | None,
    non_interactive: bool,
) -> InitialSFTModelOptimizationSettings | None:
    """Collect and persist only missing first-run Tinker catalog selections.

    Args:
        store: Project whose catalog and selection state are inspected.
        project: Project ID used for the deterministic trained-model alias prefix.
        tinker_connection: Optional explicit connection name from the CLI.
        tinker_api_key_env: Optional environment-variable name for the Tinker credential.
        base_model_alias: Optional explicit local base-model alias from the CLI.
        base_model: Optional exact Tinker model ID for a new alias.
        maximum_cost_usd: Optional finite immutable training cap shown before cost authorization.
        training_usd_per_million_tokens: Optional explicit model-specific training price.
        non_interactive: Whether setup questions are forbidden.

    Returns:
        Confirmed first-run settings, or ``None`` when an immutable selection already exists.

    Raises:
        typer.BadParameter: Required noninteractive values are missing or catalog selections
            conflict with existing provider metadata.
        typer.Abort: An interactive user declines the complete setup summary.
    """
    latest = load_latest_sft_model_optimization(store)
    bootstrap = store.load_project().model_optimization_config
    selected = latest.config if latest is not None else bootstrap
    if selected is not None:
        config = load_sft_model_optimization_config(store, selected.artifact_id)
        _require_replay_settings_match(
            config.training,
            maximum_cost_usd=maximum_cost_usd,
            training_usd_per_million_tokens=training_usd_per_million_tokens,
        )
        return None
    catalog = load_model_catalog(store.model_catalog_path)
    catalog_sha256 = sha256_json(catalog)
    interactive = not non_interactive and can_prompt(_console)
    connection_name = tinker_connection
    alias = base_model_alias
    if interactive:
        _console.print("[bold]Tinker model optimization setup[/bold]")
        if connection_name is None:
            connection_name = Prompt.ask("Tinker connection name", console=_console).strip()
        if alias is None:
            alias = Prompt.ask("Tinker base model alias", console=_console).strip()
    connection = catalog.connections.get(connection_name) if connection_name is not None else None
    api_key_env = tinker_api_key_env
    if connection is not None and connection.api_key_env is not None:
        if api_key_env is not None and api_key_env != connection.api_key_env:
            raise typer.BadParameter(
                f"Tinker connection {connection_name!r} already names a different api_key_env"
            )
        api_key_env = connection.api_key_env
    if interactive and api_key_env is None:
        api_key_env = Prompt.ask("Tinker API key environment variable", console=_console).strip()
    missing = []
    if connection_name is None:
        missing.append("--tinker-connection")
    if api_key_env is None:
        missing.append("--tinker-api-key-env")
    if alias is None:
        missing.append("--base-model-alias")
    if training_usd_per_million_tokens is None and not interactive:
        missing.append("--training-usd-per-million-tokens")
    if missing:
        raise typer.BadParameter(
            "first `exp optimize model` in noninteractive mode requires "
            + ", ".join(missing)
            + "; add those flags, plus --base-model when the alias is new; an explicit zero "
            "training price is allowed"
        )
    assert connection_name is not None
    assert api_key_env is not None
    assert alias is not None
    if connection is not None and connection.provider != "tinker":
        raise typer.BadParameter(
            f"connection {connection_name!r} uses provider {connection.provider!r}, not 'tinker'"
        )
    record = catalog.models.get(alias)
    if record is not None:
        if record.connection != connection_name:
            raise typer.BadParameter(
                f"base model alias {alias!r} uses connection {record.connection!r}, not "
                f"{connection_name!r}"
            )
        if base_model is not None and record.model != base_model:
            raise typer.BadParameter(
                f"base model alias {alias!r} already names a different exact model ID"
            )
        resolved_model = record.model
    else:
        resolved_model = base_model
        if interactive and resolved_model is None:
            resolved_model = Prompt.ask("Exact Tinker base model ID", console=_console).strip()
        if resolved_model is None:
            raise typer.BadParameter(
                f"base model alias {alias!r} is not configured; add --base-model with its exact "
                "Tinker model ID"
            )
    if interactive and training_usd_per_million_tokens is None:
        training_usd_per_million_tokens = FloatPrompt.ask(
            "Training USD per million tokens",
            console=_console,
        )
    if training_usd_per_million_tokens is None:
        raise typer.BadParameter(
            "first model optimization requires --training-usd-per-million-tokens; use an "
            "explicit 0 only when training is known to be free"
        )
    if training_usd_per_million_tokens < 0:
        raise typer.BadParameter("training price must be nonnegative")
    selected_maximum_cost_usd = (
        _DEFAULT_MAXIMUM_COST_USD if maximum_cost_usd is None else maximum_cost_usd
    )
    training = TinkerSFTSpec(
        base_model=resolved_model,
        lora_rank=_DEFAULT_LORA_RANK,
        learning_rate=_DEFAULT_LEARNING_RATE,
        batch_size=_DEFAULT_BATCH_SIZE,
        epochs=_DEFAULT_EPOCHS,
        checkpoint_every_steps=_DEFAULT_CHECKPOINT_EVERY_STEPS,
        maximum_datum_tokens=_DEFAULT_MAXIMUM_DATUM_TOKENS,
        maximum_cost_usd=selected_maximum_cost_usd,
        training_usd_per_million_tokens=training_usd_per_million_tokens,
    )
    if interactive:
        _console.print(
            "[dim]Training defaults: LoRA rank 32, learning rate 0.0001, batch size 4, "
            "1 epoch, checkpoint every 10 steps, 4096 tokens per datum, maximum cost "
            f"${selected_maximum_cost_usd:.2f}; confirmed training price "
            f"${training_usd_per_million_tokens:.6f} per million tokens.[/dim]"
        )
        if not Confirm.ask(
            f"Use Tinker connection {connection_name!r}, credential variable {api_key_env!r}, "
            f"and base model {resolved_model!r}?",
            default=True,
            console=_console,
        ):
            raise typer.Abort()
    _persist_tinker_selection(
        store,
        observed_catalog_sha256=catalog_sha256,
        connection_name=connection_name,
        api_key_env=api_key_env,
        alias=alias,
        resolved_model=resolved_model,
    )
    return InitialSFTModelOptimizationSettings(
        model_alias_prefix=f"{project}-sft",
        tinker_connection=connection_name,
        base_model_alias=alias,
        training=training,
    )


def _require_replay_settings_match(
    training: TinkerSFTSpec,
    *,
    maximum_cost_usd: float | None,
    training_usd_per_million_tokens: float | None,
) -> None:
    """Reject explicit CLI settings that drift from the selected immutable graph.

    Args:
        training: Previously selected immutable training settings.
        maximum_cost_usd: Optional cap supplied for this invocation.
        training_usd_per_million_tokens: Optional price supplied for this invocation.

    Raises:
        typer.BadParameter: Either supplied value differs from the immutable selection.
    """
    if maximum_cost_usd is not None and maximum_cost_usd != training.maximum_cost_usd:
        raise typer.BadParameter(
            "--maximum-cost-usd differs from the selected immutable model-optimization config"
        )
    if (
        training_usd_per_million_tokens is not None
        and training_usd_per_million_tokens != training.training_usd_per_million_tokens
    ):
        raise typer.BadParameter(
            "--training-usd-per-million-tokens differs from the selected immutable "
            "model-optimization config"
        )


def _persist_tinker_selection(
    store: ProjectStore,
    *,
    observed_catalog_sha256: Sha256,
    connection_name: str,
    api_key_env: str,
    alias: str,
    resolved_model: str,
) -> None:
    """Merge confirmed Tinker entries under a cross-process catalog lock.

    Unrelated concurrent catalog additions are retained. Drift in either confirmed target fails
    closed, while another process persisting the exact same selection is idempotent.

    Args:
        store: Project whose shared model catalog receives the confirmed selection.
        observed_catalog_sha256: Digest of the catalog shown and validated before confirmation.
        connection_name: Confirmed native Tinker connection name.
        api_key_env: Confirmed credential environment-variable name. No secret value is read.
        alias: Confirmed local base-model alias.
        resolved_model: Confirmed exact Tinker base-model ID.

    Raises:
        typer.BadParameter: A concurrent writer changed a confirmed connection or alias.
    """
    desired_connection = ConnectionConfig(provider="tinker", api_key_env=api_key_env)
    desired_record = ModelRecord(
        connection=connection_name,
        model=resolved_model,
        billing_source=BillingSource.CUSTOMER_MANAGED,
    )
    with file_write_lock(store.model_catalog_path, what="the local model catalog"):
        current = load_model_catalog(store.model_catalog_path)
        drifted = sha256_json(current) != observed_catalog_sha256
        existing_connection = current.connections.get(connection_name)
        connection_conflicts = existing_connection is not None and (
            existing_connection.provider != "tinker"
            or existing_connection.base_url != desired_connection.base_url
            or existing_connection.api_key_env not in {None, api_key_env}
        )
        if connection_conflicts:
            qualifier = "concurrently " if drifted else ""
            raise typer.BadParameter(
                f"Tinker connection {connection_name!r} {qualifier}changed before setup commit"
            )
        existing_record = current.models.get(alias)
        record_conflicts = existing_record is not None and (
            existing_record.connection != connection_name or existing_record.model != resolved_model
        )
        if record_conflicts:
            qualifier = "concurrently " if drifted else ""
            raise typer.BadParameter(
                f"base model alias {alias!r} {qualifier}changed before setup commit"
            )
        if (
            existing_connection == desired_connection
            and existing_record is not None
            and not record_conflicts
        ):
            return
        connections = dict(current.connections)
        models = dict(current.models)
        connections[connection_name] = desired_connection
        models.setdefault(alias, desired_record)
        write_model_catalog(
            store.model_catalog_path,
            ModelCatalog(
                schema_version=current.schema_version,
                connections=connections,
                models=models,
                roles=current.roles,
            ),
        )


class _LocalPreflightBackend:
    """Backend seam that permits local graph validation but cannot open a trainer."""

    def conservative_step_cost(
        self, spec: TinkerSFTSpec, *, batch_example_count: int
    ) -> NumericMeasurement | None:
        """Calculate the immutable caller-priced step bound without constructing an SDK.

        Args:
            spec: Frozen settings containing an explicit price and token ceiling.
            batch_example_count: Exact planned batch size.

        Returns:
            The conservative local estimate, or ``None`` when price inputs are incomplete.
        """
        return conservative_training_step_cost(spec, batch_example_count=batch_example_count)

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> Never:
        """Reject any attempt to dispatch through the local-only validation seam.

        Args:
            spec: Frozen settings that must never reach a provider through this seam.
            resume_state_path: Optional resume handle that must never be used here.

        Raises:
            SFTModelOptimizationPreflightError: Always, because this backend is non-dispatching.
        """
        del spec, resume_state_path
        raise SFTModelOptimizationPreflightError(
            "local model-optimization preflight cannot dispatch trainer work"
        )


def _compose_tinker_backend(
    store: ProjectStore,
    connection_name: str,
    expected_connection_config_sha256: Sha256,
) -> TrainerBackend:
    """Resolve the selected credential and compose the concrete Tinker SDK adapter.

    Args:
        store: Project whose secret-free catalog selects the authorized credential name.
        connection_name: Exact Tinker connection frozen into the optimization config.
        expected_connection_config_sha256: Frozen full connection metadata digest that must match
            before credential or SDK access.

    Returns:
        Concrete backend that does not call Tinker until W13 invokes ``open``.

    Raises:
        ModelCredentialError: The selected credential environment variable is absent.
        SFTModelOptimizationPreflightError: The selected connection or optional SDK is invalid.
    """
    catalog = load_model_catalog(store.model_catalog_path)
    connection = catalog.connections.get(connection_name)
    if connection is None or connection.provider != "tinker":
        raise SFTModelOptimizationPreflightError(
            f"selected connection {connection_name!r} is not a configured native Tinker connection"
        )
    current_connection_config_sha256 = sha256_json(
        {
            "provider": connection.provider,
            "base_url": connection.base_url,
            "api_key_env": connection.api_key_env,
        }
    )
    if current_connection_config_sha256 != expected_connection_config_sha256:
        raise SFTModelOptimizationPreflightError(
            "selected Tinker connection metadata drifted before credential resolution"
        )
    api_key = read_connection_api_key(connection, connection_id=connection_name)
    try:
        import tinker
    except ImportError as exc:
        raise SFTModelOptimizationPreflightError(
            "Tinker SFT requires the optional sft dependencies; run `uv sync --extra sft`"
        ) from exc
    service = (
        tinker.ServiceClient(api_key=api_key)
        if connection.base_url is None
        else tinker.ServiceClient(api_key=api_key, base_url=connection.base_url)
    )
    return TinkerTrainerBackend(service)
