"""Reproduction manifests: everything a published number depends on, committed as data.

A manifest is a TOML file under `wmo/reproduce/manifests/`, shipped with the package. It is
deliberately data, not code: the other benchmarks this program publishes (swe-bench,
terminal-bench) get their own manifests without touching the runner, and a reader can audit
what a reproduction does without executing anything.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_PACKAGE = "wmo.reproduce.manifests"


class DataSource(BaseModel):
    """Where the benchmark's evidence lives: a public Hugging Face repo, pinned by revision."""

    model_config = ConfigDict(extra="forbid")

    hf_repo: str = Field(min_length=1)
    repo_type: Literal["dataset", "model"] = "dataset"
    revision: str | None = None  # None = default branch; pin a commit for strict runs
    files: list[str] = Field(min_length=1)


class MatrixProtocol(BaseModel):
    """The `matrix` kind: fit + tune + report on a downloaded outcome matrix, offline.

    With `pin_model` set, the protocol is the bench-defaults one: a static
    policy pinned to that pool entry (what `wmo optimize route pin` writes)
    instead of a kNN fit. A pin routes nothing, so no embedding identity or
    cache is needed at all - the replay is arithmetic over the matrix.
    """

    model_config = ConfigDict(extra="forbid")

    matrix_file: str = Field(min_length=1)  # within the downloaded snapshot
    embedding_cache_file: str | None = None  # .npy aligned to the matrix (bit-exact, offline)
    # The embedding identity the vectors have (recorded on the policy so serving rebuilds
    # the same geometry). With a cache file present, no credential for it is needed.
    embedder_kind: Literal["hashing", "azure"] = "azure"
    embedder_dim: int = 3072
    embedder_deployment: str | None = None
    embedder_endpoint: str | None = None
    fallback: str | None = None
    # Pool entry every request goes to; replaces the kNN fit with a static pin.
    pin_model: str | None = None
    # The report's one customer-facing sentence describing WHAT was measured. Without it
    # the report writer's default claims scenarios "reconstructed from your traces",
    # which is false for every real-benchmark manifest.
    scenario_label: str | None = None
    cost_quality: float = 0.25
    baselines: list[str] = Field(min_length=1)


class CommandsProtocol(BaseModel):
    """The `commands` kind: replay the cookbook's own CLI commands, verbatim and pinned.

    `{data}` in an argv expands to the downloaded snapshot directory and `{out}` to the
    run's output directory, so the pinned commands are location-independent while staying
    readable as exactly what a person would type.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[list[str]] = Field(min_length=1)
    # A pool TOML shipped beside the manifests, copied to `{out}/pool.toml` before step one,
    # so a pinned pool travels with the recipe instead of depending on the operator's config.
    pool_file: str | None = None
    report_file: str = Field(min_length=1)  # under {out}; compared against [[published]]
    estimated_spend_usd: float = Field(gt=0)  # stated before consent; a forecast, not a cap


class PublishedRow(BaseModel):
    """One published number and how close a reproduction must land to count.

    Tolerances are RELATIVE fractions (0.02 = within 2%). Zero means bit-exact, which only
    the `matrix` kind can honestly promise.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    baseline: str | None = None  # which report this row reads (matrix kind, multi-baseline)
    accuracy: float
    cost_per_run_usd: float
    latency_p50_ms: float | None = None
    tolerance_accuracy: float = 0.0
    tolerance_cost: float = 0.0
    tolerance_latency: float = 0.0


class Manifest(BaseModel):
    """One benchmark's complete reproduction recipe."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    cookbook: str = Field(min_length=1)  # repo-relative doc the numbers are narrated in
    exactness: Literal["bit-exact", "protocol-exact"]
    kind: Literal["matrix", "commands"]
    data: DataSource
    matrix: MatrixProtocol | None = None
    commands: CommandsProtocol | None = None
    published: list[PublishedRow] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)  # caveats printed with every verdict

    @model_validator(mode="after")
    def _kind_has_its_protocol(self) -> Manifest:
        if self.kind == "matrix" and self.matrix is None:
            raise ValueError(f"manifest '{self.name}' is kind=matrix but has no [matrix] table")
        if self.kind == "commands" and self.commands is None:
            raise ValueError(f"manifest '{self.name}' is kind=commands but has no [commands] table")
        if self.kind == "commands" and self.exactness == "bit-exact":
            raise ValueError(
                f"manifest '{self.name}': live-provider replays are never bit-exact; "
                "claim protocol-exact"
            )
        return self


def manifest_names() -> list[str]:
    """Every shipped manifest, by benchmark name (the `<name>.toml` stem), sorted."""
    root = resources.files(MANIFEST_PACKAGE)
    return sorted(
        entry.name.removesuffix(".toml")
        for entry in root.iterdir()
        if entry.name.endswith(".toml") and not entry.name.endswith(".pool.toml")
    )


def load_manifest(name: str) -> Manifest:
    """Load one shipped manifest by benchmark name.

    Raises:
        KeyError: no manifest of that name ships with this package (names available ones).
        ValueError: the manifest exists but does not validate (a packaging bug, not a user
            error; the tests load every shipped manifest for exactly this reason).
    """
    root = resources.files(MANIFEST_PACKAGE)
    entry = root / f"{name}.toml"
    if not entry.is_file():
        raise KeyError(
            f"no reproduction manifest named '{name}'; shipped: {', '.join(manifest_names())}"
        )
    raw = tomllib.loads(entry.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)


def load_manifest_file(path: Path) -> Manifest:
    """Load a manifest from an explicit path (for manifests under development)."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)
