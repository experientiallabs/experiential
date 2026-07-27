"""Project-local settings stored under the selected harness root."""

from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

import tomli_w
from llm_waterfall import ChatMaxTokensField
from pydantic import BaseModel, Field, ValidationError

from wmo.config.config import ARTIFACT_DIR

SETTINGS_FILENAME = "settings.toml"


class TelemetrySettings(BaseModel):
    """Usage telemetry preferences for this harness project."""

    enabled: bool = True
    anonymous_id: str | None = None


class ModelRole(BaseModel):
    """One named model role: which provider/model handles this class of work."""

    provider: str  # a ProviderKind value ("bedrock", "azure", "openai", ...)
    model: str
    # Canonical model identity when `model` is a runtime id that does not carry it, e.g. a
    # tinker:// weights path whose base model determines the renderer and tokenizer.
    model_type: str | None = None
    region: str | None = None  # AWS Bedrock region
    endpoint: str | None = None  # Azure OpenAI / custom base URL
    deployment: str | None = None  # Azure OpenAI deployment name
    api_version: str | None = None  # Azure OpenAI API version (azure roles get a default)
    reasoning_effort: str | None = None  # structured reasoning effort, when supported
    # The output-budget parameter this backend accepts. Unset means "resolve it from the built-in
    # catalog", which is right for every named model; a custom `endpoint` outside the catalog (a
    # tinker:// student on Tinker's serving endpoint wants the classic `max_tokens`) has to say so
    # here, or every call to it 400s on the wrong parameter name.
    chat_max_tokens_field: ChatMaxTokensField | None = None


class ModelsSettings(BaseModel):
    """Role-based model defaults for this project (`.wmo/settings.toml`, `[models.<role>]`).

    Five roles keep the surface small: `worker` does quality-critical generation (scenario
    synthesis, cluster naming, agent rollouts); `judge` grades (checklist judging, inline
    validity gates) and should be a different model family from `worker` so the grader carries
    no self-preference bias toward the generator's outputs; `summary` does high-volume cheap
    extraction (trace facets/digests); `meta` is the harness-search delta proposer, which
    needs a long-context, long-output model (one proposal holds every harness surface in its
    prompt and replies with a complete replacement surface); `agent` is the agent-under-test
    whose harness `wmo optimize harness` searches. Set it to optimize a harness for a model
    distinct from the world model's serve provider (e.g. a small self-hosted agent against a
    frontier-served world model). Unset `judge`/`summary` fall back to `worker`; each command
    documents which explicit model flags and opt-in roles it uses.
    """

    worker: ModelRole | None = None
    judge: ModelRole | None = None
    summary: ModelRole | None = None
    meta: ModelRole | None = None
    agent: ModelRole | None = None

    def resolve(self, role: str) -> ModelRole | None:
        """The configured role, with unset `judge`/`summary` falling back to `worker`.

        `meta` and `agent` deliberately do NOT fall back to `worker`: each is picked for a
        need the scenario worker does not serve (the proposer's long-context/long-output
        surface; the agent-under-test's own identity), so when unset they return None and the
        caller keeps its own default (`wmo optimize harness` and closed-loop eval use the world
        model's provider for the opt-in roles when unset).
        """
        if role not in ("worker", "judge", "summary", "meta", "agent"):
            raise ValueError(
                f"unknown model role {role!r}; expected worker, judge, summary, meta, or agent"
            )
        configured: ModelRole | None = getattr(self, role)
        if role in ("meta", "agent"):
            return configured
        return configured or self.worker


class ProjectSettings(BaseModel):
    """Settings that are local to one harness project root."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)


def settings_path(root: str | Path = ARTIFACT_DIR) -> Path:
    return Path(root) / SETTINGS_FILENAME


_SETTINGS_REPAIR = "fix the file, or delete it and re-run `wmo providers set` to write a fresh one"
"""How to recover a settings file this loader refuses. Every raise below names it.

`wmo providers set` reads the file before it rewrites it, so "just re-run it" is only true once
the broken file is out of the way, hence "delete it and re-run", not "re-run".
"""


def load_settings(root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """Read `<root>/settings.toml`, defaulting when it does not exist.

    Raises:
        ValueError: The file exists but cannot be read, is not valid TOML, or does not match the
            current settings schema. The message names the path and the repair.
    """
    path = settings_path(root)
    if not path.exists():
        return ProjectSettings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        # A TOML document is UTF-8 by definition, so bytes that do not decode are the same
        # "not valid TOML" answer; `tomllib.load` decodes before it parses, and the raw
        # UnicodeDecodeError names neither the file nor the way out.
        raise ValueError(f"{path} is not valid TOML ({exc}); {_SETTINGS_REPAIR}") from exc
    except OSError as exc:
        raise ValueError(f"{path} could not be read ({exc}); {_SETTINGS_REPAIR}") from exc
    try:
        return ProjectSettings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"{path} does not match the current settings schema ({exc}); {_SETTINGS_REPAIR}"
        ) from exc


def save_settings(settings: ProjectSettings, root: str | Path = ARTIFACT_DIR) -> None:
    path = settings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude_none=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(data, fh)
    tmp.replace(path)


def set_telemetry_enabled(enabled: bool, root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    settings = load_settings(root)
    settings.telemetry.enabled = enabled
    save_settings(settings, root)
    return settings


def ensure_telemetry_anonymous_id(root: str | Path = ARTIFACT_DIR) -> str:
    settings = load_settings(root)
    if settings.telemetry.anonymous_id is None:
        settings.telemetry.anonymous_id = uuid.uuid4().hex
        save_settings(settings, root)
    return settings.telemetry.anonymous_id
