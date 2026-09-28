"""Bounded launch inputs and atomic local run receipts."""

import os
import stat
import tempfile
from pathlib import Path

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration
from exp.optimize.claas.training_contracts import TrainingExample

MAXIMUM_CONFIGURATION_BYTES = 131_072


def read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    """Read one bounded regular file without following its final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise ValueError(f"{path.name} must be a regular file below {maximum_bytes} bytes")
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"{path.name} grew beyond its configured byte limit")
    return payload


def load_configuration(path: Path) -> RunLaunchConfiguration:
    """Validate a launch without constructing credentials, clients, or GPU workers."""
    return RunLaunchConfiguration.model_validate_json(
        read_regular_file(path, MAXIMUM_CONFIGURATION_BYTES)
    )


def load_examples(configuration: RunLaunchConfiguration) -> tuple[TrainingExample, ...]:
    """Validate every exact JSONL record before mutating the durable buffer."""
    path = configuration.import_examples_path
    if path is None:
        return ()
    payload = read_regular_file(path, configuration.maximum_import_bytes)
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("experience import must contain nonblank JSONL records")
    if len(lines) > configuration.run.maximum_buffer_records:
        raise ValueError("experience import exceeds maximum_buffer_records")
    return tuple(TrainingExample.model_validate_json(line) for line in lines)


def write_contract(path: Path, value: ContractModel) -> None:
    """Atomically replace a private JSON receipt and sync its containing directory."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
