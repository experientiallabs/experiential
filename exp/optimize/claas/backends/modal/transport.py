"""Bounded file transport while the named sandbox owns its mounted state."""

import os
import stat
from pathlib import Path, PurePosixPath

import modal


async def upload_import(
    sandbox: modal.Sandbox, path: Path, run_id: str, *, maximum_bytes: int
) -> str:
    """Copy an exact-example file under the run identity before starting the service."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("import_path must remain a regular file during upload")
        payload = source.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("import_path grew beyond the import byte cap before upload")
    destination = f"/state/imports/{run_id}.jsonl"
    await sandbox.filesystem.make_directory.aio("/state/imports")
    existing = await sandbox.filesystem.list_files.aio("/state/imports")
    if any(entry.name == f"{run_id}.jsonl" for entry in existing):
        raise ValueError("this run_id already has an import file; choose a new run_id")
    await sandbox.filesystem.write_bytes.aio(payload, destination)
    return destination


async def download_artifact(
    volume: modal.Volume, relative_path: str, destination: Path, *, maximum_bytes: int
) -> None:
    """Stream one committed artifact with a byte cap and no local overwrite."""
    source = PurePosixPath(relative_path)
    if source.is_absolute() or ".." in source.parts or relative_path in {"", "."}:
        raise ValueError("artifact path must be a relative path inside the state Volume")
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    size = 0
    created = False
    try:
        with destination.open("xb") as target:
            created = True
            async for chunk in volume.read_file.aio(source.as_posix()):
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("artifact exceeds maximum_bytes; raise the explicit byte cap")
                target.write(chunk)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
