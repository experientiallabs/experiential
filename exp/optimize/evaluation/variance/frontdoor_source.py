"""The front door's trace source: its exact bytes bound to the project build they produced.

A saved ``front-door-source`` record binds one trace file's digest to the completed build made
from it, so a rerun over the same file skips the project build.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input

logger = logging.getLogger(__name__)

FRONT_DOOR_PROTOCOL = "decision-variance-front-door-v1"
"""Version of the front-door sequence and its source record."""

SOURCE_TYPE = "front-door-source"
"""Artifact type binding one trace source's digest to the project build it produced."""


class TraceSource(ContractModel):
    """The trace file the front door is pointed at.

    Attributes:
        path: Local canonical trace export, or the gateway capture file with ``identity``.
        source: Canonical trace source format, as ``exp build --source`` takes it.
        identity: Gateway identity whose captures supply the build, with ``source`` gateway.
    """

    path: Path
    source: str = Field(default="otlp", min_length=1, max_length=64)
    identity: str | None = Field(default=None, min_length=1)

    def sha256(self) -> Sha256:
        """Return the digest of the trace file's exact bytes, or of a directory's files.

        A directory (such as a folder of agent session files) hashes every regular file under
        it, in sorted relative-path order, framing each path and its bytes, so a rename, an
        added or removed file, or any changed byte changes the digest.
        """
        if not self.path.is_dir():
            return sha256_bytes(self.path.read_bytes())
        digest = hashlib.sha256(b"front-door-trace-directory-v1\0")
        files = sorted(
            (item.relative_to(self.path).as_posix(), item)
            for item in self.path.rglob("*")
            if item.is_file()
        )
        for relative, item in files:
            name = relative.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(item.stat().st_size.to_bytes(8, "big"))
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        return digest.hexdigest()


class ProjectBuilder(Protocol):
    """Makes the project's completed build come from exactly one trace source."""

    def __call__(self, project: ProjectStore, source: TraceSource) -> None:
        """Build, or replay, the project from the source; raise when it cannot."""
        ...


def source_identity(source: TraceSource, build: ProjectBuildArtifacts) -> str:
    """Return the record identity binding one trace source's exact content to one build."""
    return stable_id(
        SOURCE_TYPE,
        {
            "protocol": FRONT_DOOR_PROTOCOL,
            "sha256": source.sha256(),
            "source": source.source,
            "identity": source.identity,
            "build": sha256_json(build),
        },
    )


def recorded_build(project: ProjectStore, source: TraceSource) -> ArtifactInput | None:
    """Return the source record when the project's current build came from this source."""
    build = project.load_project().build
    if build is None:
        return None
    identity = source_identity(source, build)
    if not project.artifacts.exists(identity):
        return None
    return artifact_input(project.artifacts.read(identity).manifest)


def ensure_build(
    project: ProjectStore,
    source: TraceSource,
    builder: ProjectBuilder | None,
    code_revision: str,
) -> ArtifactInput:
    """Build the project from the source unless its record is current, then save the record.

    Raises:
        ValueError: The build is pending without a builder, or the builder left no build.
    """
    recorded = recorded_build(project, source)
    if recorded is not None:
        return recorded
    if builder is None:
        raise ValueError("the project build is pending and no project builder was given")
    logger.info("front door: building project from %s", source.path)
    builder(project, source)
    build = project.load_project().build
    if build is None:
        raise ValueError("the project builder finished without a completed build")
    identity = source_identity(source, build)
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=SOURCE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted(
                        (build.trace_dataset, build.task_set, build.world_model),
                        key=lambda ref: ref.artifact_id,
                    )
                ),
            ),
            files={"build.json": build, "source.json": source},
        )
    return artifact_input(project.artifacts.read(identity).manifest)
