"""Isolated Python workspaces with native tools and quiescent filesystem checkpoints."""

from __future__ import annotations

import io
import json
import os
import re
import selectors
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, Sha256, sha256_bytes
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import NativeImage, WorldAccess
from exp.runtime.environments.worlds import WorldContext

_IMAGE_LIMIT = 72 * 1024 * 1024
_OUTPUT_LIMIT = 65536
_FILE_SCRIPT = r"""
import json
import sys
from pathlib import Path

root = Path("/workspace")
data = json.load(sys.stdin)
try:
    path = Path(data["path"])
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("file path must stay inside /workspace")
    operation = data["operation"]
    if operation == "read":
        content = path.read_text()
        start = data.get("start_line", 1)
        end = data.get("end_line")
        if (type(start) is not int or start < 1
            or (end is not None and (type(end) is not int or end < start))):
            raise ValueError("line bounds must be positive and ordered")
        if "start_line" in data or end is not None:
            content = "".join(content.splitlines(keepends=True)[start - 1:end])
        result = {"content": content}
    elif operation == "write":
        content = data["content"]
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        result = {"written_bytes": len(content.encode()), "path": str(path)}
    elif operation == "edit":
        content = path.read_text()
        edits = data["edits"]
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits must be a nonempty list")
        for edit in edits:
            old = edit.get("oldText", edit.get("old_string"))
            new = edit.get("newText", edit.get("new_string"))
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError("each edit requires a nonempty oldText and a newText string")
            if content.count(old) != 1:
                raise ValueError("oldText must occur exactly once; no edits applied")
            content = content.replace(old, new, 1)
        path.write_text(content)
        result = {"written_bytes": len(content.encode()), "path": str(path)}
    else:
        raise ValueError("unknown file operation")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
except (ValueError, KeyError, TypeError, OSError) as error:
    sys.stdout.write(json.dumps({"error": str(error)}))
    sys.exit(1)
"""


@dataclass(frozen=True)
class DockerWorkspaceConfig:
    """Explicit local Docker authority and finite workspace resources.

    Attributes:
        image: Immutable local image digest containing Python and a POSIX toolchain.
        docker: Operator-selected Docker command, optionally including a privilege wrapper.
        maximum_seconds: Maximum wall time of an individual shell command.
        memory_mb: Container memory ceiling.
        workspace_mb: Writable workspace capacity, including temporary files.
    """

    image: str
    docker: tuple[str, ...] = ("docker",)
    maximum_seconds: int = 60
    memory_mb: int = 512
    workspace_mb: int = 64

    def __post_init__(self) -> None:
        """Reject mutable images and unbounded execution settings."""
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image):
            raise ValueError("workspace image must be an immutable sha256 digest")
        if not self.docker or not 1 <= self.maximum_seconds <= 120:
            raise ValueError("Docker command and a shell limit of 1 to 120 seconds are required")
        if not 128 <= self.memory_mb <= 4096 or not 1 <= self.workspace_mb <= 64:
            raise ValueError("workspace resource limits exceed supported bounds")


def _command(
    arguments: tuple[str, ...],
    *,
    payload: bytes = b"",
    seconds: float = 30,
    limit: int = _OUTPUT_LIMIT,
) -> subprocess.CompletedProcess[bytes]:
    """Run a finite local command while draining and bounding both output streams."""
    with tempfile.TemporaryFile() as input_file:
        input_file.write(payload)
        input_file.seek(0)
        process = subprocess.Popen(
            arguments, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdout is not None and process.stderr is not None
        streams = (process.stdout, process.stderr)
        buffers = {stream.fileno(): bytearray() for stream in streams}
        stdout_fd, stderr_fd = process.stdout.fileno(), process.stderr.fileno()
        deadline = time.monotonic() + seconds
        try:
            with selectors.DefaultSelector() as selector:
                for stream in streams:
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("workspace command exceeded its host deadline")
                    for key, _ in selector.select(min(remaining, 0.2)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            buffer = buffers[key.fd]
                            buffer.extend(chunk[: max(0, limit + 1 - len(buffer))])
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
        return subprocess.CompletedProcess(
            arguments,
            process.returncode,
            bytes(buffers[stdout_fd]),
            bytes(buffers[stderr_fd]),
        )


class DockerWorkspaceFactory:
    """Create isolated workspaces or restore detached native filesystem images."""

    def __init__(self, configuration: DockerWorkspaceConfig) -> None:
        """Bind the operator-selected image and Docker authority."""
        self.configuration = configuration

    @property
    def definition(self) -> JsonObject:
        """Bind checkpoint compatibility to image, resources and executable semantics."""
        config = self.configuration
        return {
            "protocol": "docker-workspace-v1",
            "image": config.image,
            "maximum_seconds": config.maximum_seconds,
            "memory_mb": config.memory_mb,
            "workspace_mb": config.workspace_mb,
            "file_script_sha256": sha256_bytes(_FILE_SCRIPT.encode()),
        }

    def create(self, task: TaskCase) -> DockerWorkspace:
        """Create an empty workspace without copying private task answers."""
        del task
        return DockerWorkspace(self.configuration)

    def restore(self, image: NativeImage) -> DockerWorkspace:
        """Restore an isolated workspace, refusing altered or unknown image parts."""
        if set(image.parts) != {"workspace.tar"}:
            raise ValueError("workspace image must contain exactly workspace.tar")
        payload = image.parts["workspace.tar"]
        if image.domains != {"workspace": sha256_bytes(payload)} or len(payload) > _IMAGE_LIMIT:
            raise ValueError("workspace image digest or size is invalid")
        workspace = DockerWorkspace(self.configuration)
        try:
            workspace._exec(("tar", "-xf", "-", "-C", "/workspace"), payload=payload)
            workspace._image = None
            if workspace.capture() != image:
                raise ValueError("workspace restoration changed native bytes or metadata")
        except BaseException:
            workspace.close()
            raise
        return workspace


class DockerWorkspace:
    """One network-disabled container with a read-only root and bounded writable filesystem."""

    def __init__(self, configuration: DockerWorkspaceConfig) -> None:
        """Start one exact owned container and initialize its private temporary directory."""
        self.configuration = configuration
        self.container = ""
        self._image: NativeImage | None = None
        config = configuration
        result = _command(
            (
                *config.docker,
                "run",
                "-d",
                "--read-only",
                "--network",
                "none",
                "--ipc",
                "none",
                "--memory",
                f"{config.memory_mb}m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "1000:1000",
                "--hostname",
                "workspace",
                "--workdir",
                "/workspace",
                "--tmpfs",
                f"/workspace:rw,size={config.workspace_mb}m,mode=0700,uid=1000,gid=1000",
                "-e",
                "HOME=/workspace",
                "-e",
                "TMPDIR=/workspace/.tmp",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                config.image,
                "/bin/sleep",
                "infinity",
            )
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        self.container = result.stdout.decode().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", self.container):
            raise RuntimeError("Docker did not return an exact container ID")
        try:
            self._exec(("mkdir", "-p", "/workspace/.tmp"))
        except BaseException:
            self.close()
            raise

    def _exec(
        self,
        arguments: tuple[str, ...],
        *,
        payload: bytes = b"",
        seconds: float = 30,
        limit: int = _OUTPUT_LIMIT,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        """Execute inside the owned container and optionally require successful completion."""
        result = _command(
            (*self.configuration.docker, "exec", "-i", self.container, *arguments),
            payload=payload,
            seconds=seconds,
            limit=limit,
        )
        if check and result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        if check and len(result.stdout) > limit:
            raise ValueError("workspace response exceeds its declared byte limit")
        return result

    def access(self, action: ToolCall) -> WorldAccess:
        """Declare native workspace access before tool execution."""
        return WorldAccess(
            reads=frozenset({"workspace"}),
            writes=frozenset() if action.name == "Read" else frozenset({"workspace"}),
        )

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Return actual filesystem operations or bounded shell stdout and exit status."""
        del context
        data = dict(action.arguments)
        if action.name == "Bash":
            command = data.get("command")
            timeout = data.get("timeout", self.configuration.maximum_seconds * 1000)
            if not isinstance(command, str) or not command.strip():
                return Observation(content="command must be a nonempty string", is_error=True)
            if (
                type(timeout) is not int
                or not 1 <= timeout <= self.configuration.maximum_seconds * 1000
            ):
                return Observation(
                    content="timeout exceeds the declared millisecond limit", is_error=True
                )
            seconds = timeout / 1000
            result = self._exec(
                (
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=1s",
                    f"{seconds}s",
                    "/bin/bash",
                    "-lc",
                    command,
                ),
                seconds=seconds + 10,
                check=False,
            )
            self._image = None
            return Observation(
                content=json.dumps(
                    {
                        "exit_code": result.returncode,
                        "stdout": result.stdout[:_OUTPUT_LIMIT].decode(errors="replace"),
                        "stderr": result.stderr[:_OUTPUT_LIMIT].decode(errors="replace"),
                        "output_truncated": max(len(result.stdout), len(result.stderr))
                        > _OUTPUT_LIMIT,
                        "timed_out": result.returncode in (124, 137),
                    }
                ),
                is_error=result.returncode != 0,
            )
        operations = {"Read": "read", "Write": "write", "Edit": "edit"}
        if action.name not in operations:
            return Observation(content="unknown workspace tool", is_error=True)
        data["operation"] = operations[action.name]
        if "path" not in data:
            data["path"] = data.get("file_path")
        if action.name == "Edit" and "edits" not in data:
            data["edits"] = [
                {"old_string": data.get("old_string"), "new_string": data.get("new_string")}
            ]
        result = self._exec(
            ("python", "-c", _FILE_SCRIPT), payload=json.dumps(data).encode(), check=False
        )
        self._image = None
        if len(result.stdout) > _OUTPUT_LIMIT:
            return Observation(
                content="File response exceeds 65536 bytes; request a smaller line range.",
                is_error=True,
                metadata={"output_truncated": True},
            )
        return Observation(
            content=result.stdout.decode(errors="replace"),
            is_error=result.returncode != 0,
        )

    def capture(self) -> NativeImage:
        """Capture a quiescent workspace exactly, refusing live background processes."""
        if self._image is None:
            processes = _command(
                (*self.configuration.docker, "top", self.container, "-eo", "pid,comm")
            )
            if processes.returncode or [
                line.split()[-1] for line in processes.stdout.decode().splitlines()[1:]
            ] != ["sleep"]:
                raise ValueError("background processes are unsupported at checkpoint boundaries")
            result = self._exec(
                ("tar", "--sort=name", "--format=gnu", "-cf", "-", "-C", "/workspace", "."),
                limit=_IMAGE_LIMIT,
            )
            self._image = NativeImage(
                parts={"workspace.tar": result.stdout},
                domains={"workspace": sha256_bytes(result.stdout)},
            )
        return self._image

    def domain_digests(self) -> dict[str, Sha256]:
        """Return the digest of the exact restorable workspace image."""
        return self.capture().domains

    def state_projection(self) -> JsonObject:
        """Expose file identities without interpreting program output as task success."""
        return workspace_projection(self.capture())

    def close(self) -> None:
        """Remove only this owned container and positively verify resource release."""
        if not self.container:
            return
        result = _command((*self.configuration.docker, "rm", "-f", self.container))
        if result.returncode or result.stdout.decode().strip() != self.container:
            raise RuntimeError("owned workspace container cleanup was not confirmed")
        self.container = ""
        self._image = None


def workspace_projection(image: NativeImage) -> JsonObject:
    """Describe saved regular files and their bytes without extracting into the host."""
    files: JsonObject = {}
    with tarfile.open(fileobj=io.BytesIO(image.parts["workspace.tar"])) as archive:
        for member in archive:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            payload = stream.read()
            files[member.name.removeprefix("./")] = {
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            }
    return {"workspace": {"files": files}}
