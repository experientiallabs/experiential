"""Verification of the documented Fireworks and Modal openai-compatible recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_RECIPES_DOC = Path(__file__).parents[3] / "docs" / "reference" / "openai-compatible-recipes.md"


def _author_connection_and_alias(
    root: Path,
    *,
    connection: str,
    base_url: str | None,
    credential_env: str,
    model: str,
    exact_model: str,
    alias: str,
    capability_flags: tuple[str, ...] = (),
) -> None:
    """Author one openai-compatible connection, alias, identity, and grant.

    Args:
        root: Temporary EXP root receiving gateway state.
        connection: Provider connection name.
        base_url: Optional endpoint base URL exactly as the recipe documents it.
        credential_env: Credential environment variable name.
        model: Provider-side model ID for the deployment.
        exact_model: Stable operator-asserted logical model identity.
        alias: Public gateway alias to create and grant.
        capability_flags: Extra alias-create capability and price flags.
    """
    runner = CliRunner()
    base_url_arguments = [] if base_url is None else ["--base-url", base_url]
    commands = (
        ["config", "gateway", "init", "--root", str(root), "--json"],
        [
            "config",
            "gateway",
            "provider",
            "add",
            connection,
            "--provider",
            "openai-compatible",
            *base_url_arguments,
            "--credential-env",
            credential_env,
            "--root",
            str(root),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "alias",
            "create",
            alias,
            "--deployment",
            f"{connection}:{model}",
            "--exact-model",
            exact_model,
            *capability_flags,
            "--root",
            str(root),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "identity",
            "create",
            "default",
            "--root",
            str(root),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "grant",
            "add",
            "default",
            alias,
            "--root",
            str(root),
            "--non-interactive",
            "--json",
        ],
    )
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output


def test_fireworks_recipe_reaches_gateway_readiness_without_a_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented Fireworks recipe authors a servable alias as written."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-secret-canary")
    _author_connection_and_alias(
        tmp_path,
        connection="fireworks",
        base_url=FIREWORKS_BASE_URL,
        credential_env="FIREWORKS_API_KEY",
        model="accounts/fireworks/models/recipe-model",
        exact_model="recipe-model",
        alias="coding",
        capability_flags=(
            "--supports-tools",
            "--supports-structured-output",
            "--input-price",
            "900000",
            "--output-price",
            "900000",
            "--pricing-source",
            "https://fireworks.ai/pricing",
        ),
    )

    readiness = CliRunner().invoke(
        app,
        ["run", "--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )

    assert readiness.exit_code == 0, readiness.output
    assert json.loads(readiness.stdout)["status"] == "ready"
    assert "fireworks-secret-canary" not in readiness.stdout


def test_modal_recipe_fails_closed_without_its_deployment_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Modal connection is unavailable until its explicit base_url is configured."""
    monkeypatch.setenv("MODAL_API_KEY", "modal-secret-canary")
    _author_connection_and_alias(
        tmp_path,
        connection="modal-vllm",
        base_url=None,
        credential_env="MODAL_API_KEY",
        model="recipe-served-model",
        exact_model="recipe-served-model",
        alias="local-serve",
    )

    readiness = CliRunner().invoke(
        app,
        ["run", "--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )

    assert readiness.exit_code == 2
    assert "local-serve" in readiness.output


def test_modal_recipe_reaches_readiness_with_its_deployment_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented Modal recipe is servable once base_url names the deployment."""
    monkeypatch.setenv("MODAL_API_KEY", "modal-secret-canary")
    _author_connection_and_alias(
        tmp_path,
        connection="modal-vllm",
        base_url="https://workspace--app-label.modal.run/v1",
        credential_env="MODAL_API_KEY",
        model="recipe-served-model",
        exact_model="recipe-served-model",
        alias="local-serve",
        capability_flags=(
            "--input-price",
            "0",
            "--output-price",
            "0",
            "--pricing-source",
            "self-hosted Modal deployment",
        ),
    )

    readiness = CliRunner().invoke(
        app,
        ["run", "--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )

    assert readiness.exit_code == 0, readiness.output
    assert json.loads(readiness.stdout)["status"] == "ready"


def test_recipes_doc_pins_the_verified_constants_and_is_indexed() -> None:
    """The published recipe doc and this verification stay in exact agreement."""
    doc = _RECIPES_DOC.read_text(encoding="utf-8")
    index = (_RECIPES_DOC.parents[1] / "README.md").read_text(encoding="utf-8")

    assert FIREWORKS_BASE_URL in doc
    assert "FIREWORKS_API_KEY" in doc
    assert "accounts/fireworks/models/" in doc
    assert ".modal.run" in doc
    assert "exp config gateway provider add" in doc
    assert "--provider openai-compatible" in doc.replace(" \\\n  ", " ")
    assert "exp/cli/tests/openai_compatible_recipes_test.py" in doc
    assert "`reference/openai-compatible-recipes.md`" in index
