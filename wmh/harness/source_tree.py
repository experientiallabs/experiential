"""Portable source trees for editing and reconstructing complete harnesses."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import PurePosixPath

import tomli_w
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from wmh.core.text import validate_durable_text

# _SAFE_PATH_RE and _SLUG_RE are doc's path and surface-id grammars; sharing them keeps this
# module's file checks and Surface validation from ever drifting apart.
from wmh.harness.doc import (
    _SAFE_PATH_RE,
    _SLUG_RE,
    _STORE_METADATA_FILES,
    CODE_RUNTIME_ID,
    MAX_OUTPUT_TOKENS_ID,
    MAX_SURFACE_PATH_BYTES,
    MAX_TURNS_ID,
    RUNTIME_KIND_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
    code_surface_id,
)
from wmh.harness.skills import Skill

SYSTEM_FILE = "SYSTEM.md"
CONFIG_FILE = "config.toml"
RUNTIME_FILE = "runtime.py"
SKILLS_DIR = "skills"

_RESERVED_RENDER_FILES = frozenset({SYSTEM_FILE, CONFIG_FILE, RUNTIME_FILE})
_TREE_HASH_BYTES = 16
MAX_SOURCE_PATH_BYTES = MAX_SURFACE_PATH_BYTES


class HarnessSourceFile(BaseModel):
    """One UTF-8 text file at a canonical path inside a harness source tree."""

    model_config = ConfigDict(frozen=True)

    path: str
    content: str

    @model_validator(mode="after")
    def _validate_file(self) -> HarnessSourceFile:
        candidate = PurePosixPath(self.path)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or candidate.as_posix() != self.path
            or ".." in candidate.parts
            or not _SAFE_PATH_RE.fullmatch(self.path)
            or len(self.path.encode("utf-8")) > MAX_SOURCE_PATH_BYTES
        ):
            raise ValueError(f"source path {self.path!r} must be a canonical relative POSIX path")
        validate_durable_text(self.content, field=f"source file {self.path!r}")
        return self


class HarnessSourceTree(BaseModel):
    """A complete editable file representation that can be parsed into a HarnessDoc.

    Round-tripping through `from_doc`/`to_doc` is lossless only on the canonical subset: a
    multi-prompt document renders as one SYSTEM.md and reparses as a single `prompt:core`
    surface, and validation-only `Surface.budget` values are dropped.
    """

    model_config = ConfigDict(frozen=True)

    files: tuple[HarnessSourceFile, ...]

    @field_validator("files")
    @classmethod
    def _canonical_files(
        cls, files: tuple[HarnessSourceFile, ...]
    ) -> tuple[HarnessSourceFile, ...]:
        paths = [item.path for item in files]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"duplicate source path(s): {duplicates}")
        metadata = sorted(path for path in paths if path in _STORE_METADATA_FILES)
        if metadata:
            raise ValueError(f"source tree cannot contain store metadata file(s): {metadata}")
        # Renders land on real filesystems, so structural conflicts must fail here, at
        # validation time, not later inside a store write.
        by_fold: dict[str, str] = {}
        for path in sorted(paths):
            claimed = by_fold.setdefault(path.casefold(), path)
            if claimed != path:
                raise ValueError(
                    f"source paths {claimed!r} and {path!r} differ only by letter case and "
                    "would collide on a case-insensitive filesystem; rename one"
                )
        directory_prefixes: dict[str, str] = {}
        for path in paths:
            parts = path.split("/")
            for index in range(1, len(parts)):
                directory_prefixes.setdefault("/".join(parts[:index]), path)
        for path in sorted(paths):
            child = directory_prefixes.get(path)
            if child is not None:
                raise ValueError(
                    f"source path {path!r} is a file but is also the directory holding "
                    f"{child!r}; rename one so no file path is a directory prefix of another"
                )
        return tuple(sorted(files, key=lambda item: item.path))

    @classmethod
    def from_doc(cls, doc: HarnessDoc) -> HarnessSourceTree:
        """Render a harness into the exact source files an external editor should see."""
        files: dict[str, str] = {
            SYSTEM_FILE: doc.system_prompt(),
            CONFIG_FILE: tomli_w.dumps(
                {
                    "harness": {
                        "tools": doc.tools(),
                        "max_turns": doc.max_turns(),
                        "max_output_tokens": doc.max_output_tokens(),
                        "temperature": doc.temperature(),
                        "runtime_kind": doc.runtime_kind(),
                    }
                }
            ),
        }
        for skill in doc.skills():
            files[f"{SKILLS_DIR}/{skill.name}.md"] = skill.to_markdown()
        runtime = doc.surface(CODE_RUNTIME_ID)
        if runtime is not None:
            files[RUNTIME_FILE] = runtime.content
        for surface in doc.code_files():
            assert surface.path is not None
            if surface.path in _RESERVED_RENDER_FILES or surface.path.startswith(f"{SKILLS_DIR}/"):
                raise ValueError(
                    f"code surface path {surface.path!r} collides with a reserved file"
                )
            files[surface.path] = surface.content
        return cls(
            files=tuple(
                HarnessSourceFile(path=path, content=content) for path, content in files.items()
            )
        )

    def to_doc(self, name: str) -> HarnessDoc:
        """Parse the complete source tree into one validated harness document."""
        files = self.file_map()
        if SYSTEM_FILE not in files:
            raise ValueError(f"harness source tree is missing required {SYSTEM_FILE}")
        surfaces = [
            Surface(
                id="prompt:core",
                kind=SurfaceKind.PROMPT,
                content=files[SYSTEM_FILE],
            )
        ]
        config_text = files.get(CONFIG_FILE)
        if config_text is not None:
            parsed_config = tomllib.loads(config_text)
            unknown_tables = sorted(set(parsed_config) - {"harness"})
            if unknown_tables:
                raise ValueError(
                    f"{CONFIG_FILE} contains unknown top-level table(s): {unknown_tables}"
                )
            config = parsed_config.get("harness", {})
            if not isinstance(config, dict):
                raise ValueError(f"{CONFIG_FILE} [harness] must be a table")
            allowed_fields = {
                "tools",
                "max_turns",
                "max_output_tokens",
                "temperature",
                "runtime_kind",
            }
            unknown_fields = sorted(set(config) - allowed_fields)
            if unknown_fields:
                raise ValueError(
                    f"{CONFIG_FILE} [harness] contains unknown field(s): {unknown_fields}"
                )
            tools = config.get("tools")
            if tools is not None:
                if not isinstance(tools, list) or not all(isinstance(item, str) for item in tools):
                    raise ValueError(f"{CONFIG_FILE} harness.tools must be an array of strings")
                surfaces.append(
                    Surface(
                        id=TOOL_POLICY_ID,
                        kind=SurfaceKind.TOOL_POLICY,
                        content="\n".join(tools),
                    )
                )
            scalar_fields = (
                ("max_turns", MAX_TURNS_ID),
                ("max_output_tokens", MAX_OUTPUT_TOKENS_ID),
                ("temperature", TEMPERATURE_ID),
            )
            for field, surface_id in scalar_fields:
                if field in config:
                    surfaces.append(
                        Surface(
                            id=surface_id,
                            kind=SurfaceKind.PARAM,
                            content=str(config[field]),
                        )
                    )
            runtime_kind = config.get("runtime_kind")
            if runtime_kind is not None and not isinstance(runtime_kind, str):
                raise ValueError(f"{CONFIG_FILE} harness.runtime_kind must be a string")
            if runtime_kind and runtime_kind != "kit-python":
                surfaces.append(
                    Surface(
                        id=RUNTIME_KIND_ID,
                        kind=SurfaceKind.PARAM,
                        content=str(runtime_kind),
                    )
                )
        runtime = files.get(RUNTIME_FILE)
        if runtime is not None:
            surfaces.append(Surface(id=CODE_RUNTIME_ID, kind=SurfaceKind.CODE, content=runtime))
        for item in self.files:
            path = PurePosixPath(item.path)
            if path.parts[0] != SKILLS_DIR:
                continue
            if len(path.parts) != 2 or path.suffix != ".md":
                raise ValueError(f"skill source path {item.path!r} must be skills/<skill-name>.md")
            skill = Skill.from_markdown(item.content)
            if path.stem != skill.name:
                raise ValueError(
                    f"skill source path {item.path!r} does not match declared name {skill.name!r}"
                )
            surfaces.append(
                Surface(
                    id=f"skill:{skill.name}",
                    kind=SurfaceKind.SKILL,
                    content=skill.to_markdown(),
                )
            )
        code_paths_by_id: dict[str, str] = {}
        for item in self.files:
            if item.path in _RESERVED_RENDER_FILES or item.path.startswith(f"{SKILLS_DIR}/"):
                continue
            surface_id = _code_surface_id(item.path)
            if surface_id == CODE_RUNTIME_ID:
                raise ValueError(
                    f"code file path {item.path!r} would alias the reserved in-process runtime "
                    f"surface {CODE_RUNTIME_ID!r} (which renders as {RUNTIME_FILE}); rename "
                    "the file"
                )
            claimed = code_paths_by_id.get(surface_id)
            if claimed is not None:
                raise ValueError(
                    f"code file paths {claimed!r} and {item.path!r} both map to surface id "
                    f"{surface_id!r} ('/' and '.' both become '-'); rename one so every "
                    "path keeps a distinct id"
                )
            code_paths_by_id[surface_id] = item.path
            surfaces.append(
                Surface(
                    id=surface_id,
                    kind=SurfaceKind.CODE,
                    path=item.path,
                    content=item.content,
                )
            )
        return HarnessDoc(name=name, surfaces=surfaces)

    def file_map(self) -> dict[str, str]:
        """Return a new path-to-content mapping in canonical path order."""
        return {item.path: item.content for item in self.files}

    @property
    def total_bytes(self) -> int:
        """Return the UTF-8 payload size across all source files."""
        return sum(len(item.content.encode("utf-8")) for item in self.files)

    @property
    def tree_hash(self) -> str:
        """Return a content address covering every source path and byte."""
        digest = hashlib.blake2b(digest_size=_TREE_HASH_BYTES)
        for item in self.files:
            path = item.path.encode("utf-8")
            content = item.content.encode("utf-8")
            digest.update(len(path).to_bytes(4, "big"))
            digest.update(path)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()

    def validate_bounds(self, *, max_files: int, max_bytes: int) -> None:
        """Reject a tree that exceeds explicit file-count or UTF-8 byte bounds."""
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            raise ValueError("max_files must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if len(self.files) > max_files:
            raise ValueError(
                f"source tree has {len(self.files)} files, more than {max_files} files allowed"
            )
        if self.total_bytes > max_bytes:
            raise ValueError(
                f"source tree has {self.total_bytes} bytes, more than {max_bytes} bytes allowed"
            )


def _code_surface_id(path: str) -> str:
    """Derive a code file's surface id, failing with the path and the allowed grammar."""
    surface_id = code_surface_id(path)
    if not _SLUG_RE.fullmatch(surface_id.partition(":")[2]):
        raise ValueError(
            f"code file path {path!r} maps to surface id {surface_id!r}, which is not a valid "
            "'code:<kebab-slug>' id; use lowercase [a-z0-9] runs separated by single '/', '.', "
            "or '-' characters (for example src/agent-loop.ts)"
        )
    return surface_id
