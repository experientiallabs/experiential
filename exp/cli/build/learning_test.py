"""Learning is the build's default pipeline: no opt-in flags, and an automatic recall model."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.build.learning import resolve_memory_recall
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
)
from exp.common.models.catalog_roles import ModelRoles


@pytest.mark.parametrize(
    "flags",
    [
        ["--learn"],
        ["--memory-recall", "jev"],
        ["--max-learning-attempts", "10"],
        ["--hold-out"],
    ],
)
def test_removed_learning_flags_are_rejected(tmp_path: Path, flags: list[str]) -> None:
    """The old opt-in flags no longer exist; the build learns by default."""
    root = tmp_path / ".exp"
    result = CliRunner().invoke(
        app,
        ["build", "retail", "--traces", str(tmp_path / "t.jsonl"), "--root", str(root), *flags],
    )
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert not root.exists()


def catalog(models: dict[str, tuple[str, str]], recall: str | None = None) -> ModelCatalog:
    """A catalog of ``alias -> (connection, model id)`` with one chat and one decision wire."""
    return ModelCatalog(
        connections={
            "http": ConnectionConfig(
                provider="openai-compatible", base_url="https://fixture.invalid/v1"
            ),
            "decisions": ConnectionConfig(
                provider="typesafe",
                base_url="https://fixture.invalid/v1",
                trusted_custom_origin=True,
            ),
        },
        models={
            alias: ModelRecord(
                connection=connection,
                model=model,
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(supports_completions=connection == "http"),
            )
            for alias, (connection, model) in models.items()
        },
        roles=ModelRoles(memory_recall=recall),
    )


def test_recall_prefers_clef_flash_then_clef_then_jev() -> None:
    """With no role set, the best SystemOne decision model is chosen; chat models never are."""
    models = {
        "world": ("http", "clef-flash-chat"),
        "jev": ("decisions", "jev-latest"),
        "clef": ("decisions", "cloudflare/clef"),
        "flash": ("decisions", "cloudflare/clef-flash"),
    }
    assert resolve_memory_recall(catalog(models)) == "flash"
    del models["flash"]
    assert resolve_memory_recall(catalog(models)) == "clef"
    del models["clef"]
    assert resolve_memory_recall(catalog(models)) == "jev"
    del models["jev"]
    assert resolve_memory_recall(catalog(models)) is None


def test_the_memory_recall_role_wins_and_must_be_a_decision_model() -> None:
    """An explicit role overrides the preference; a chat model is refused."""
    models = {"jev": ("decisions", "jev-latest"), "flash": ("decisions", "cloudflare/clef-flash")}
    assert resolve_memory_recall(catalog(models, recall="jev")) == "jev"
    with pytest.raises(ValueError, match="not a native decision model"):
        resolve_memory_recall(catalog({"world": ("http", "w")}, recall="world"))
