"""SDK-free Modal launch validation before consent, clients, or resource allocation."""

import re
from pathlib import Path

from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration

_MAXIMUM_CONFIG_BYTES = 131_072
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def validate_launch(
    resources: ModalLaunch,
    configuration: RunLaunchConfiguration,
    run_id: str,
    import_path: Path | None = None,
) -> bytes:
    """Reject unsafe mounts, transport paths, deadlines, or missing paid-compute estimates."""
    if configuration.compute_reservation_usd <= 0:
        raise ValueError("Modal requires a positive compute_reservation_usd operator estimate")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain 1–64 letters, digits, dots, underscores, or hyphens")
    if not configuration.directory.is_relative_to(Path("/state")):
        raise ValueError("Modal directory must be inside the /state Volume mount")
    if configuration.persistence != "modal-volume":
        raise ValueError("Modal runs require persistence='modal-volume' before acknowledgements")
    if configuration.run.mode == "run" and (
        configuration.host != "0.0.0.0" or not resources.authentication_secret_name
    ):
        raise ValueError("Modal run mode requires host='0.0.0.0' and authentication_secret_name")
    required = (
        configuration.run.maximum_run_seconds
        + configuration.run.cleanup_timeout_seconds
        + resources.startup_timeout_seconds
        + 30
    )
    if required >= resources.timeout_seconds:
        raise ValueError("Modal timeout must exceed startup, run, and cleanup bounds by 30 seconds")
    remote_import = configuration.import_examples_path
    if remote_import is not None and (
        not remote_import.is_relative_to(Path("/state")) or ".." in remote_import.parts
    ):
        raise ValueError("Modal import_examples_path must be inside /state without traversal")
    if remote_import is not None and import_path is None:
        raise ValueError("declared Modal import requires a local import_path for this run")
    if import_path is not None:
        validate_import_file(
            import_path,
            min(
                resources.maximum_upload_bytes,
                configuration.maximum_import_bytes,
            ),
        )
        if remote_import != Path(f"/state/imports/{run_id}.jsonl"):
            raise ValueError("import_examples_path must equal /state/imports/<run_id>.jsonl")
    payload = configuration.model_dump_json().encode()
    if len(payload) > _MAXIMUM_CONFIG_BYTES:
        raise ValueError("run configuration exceeds the 128 KiB launch limit")
    return payload


def validate_import_file(path: Path, maximum_bytes: int) -> None:
    """Reject nonregular or oversized inputs before allocating a paid container."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("import_path must be a regular JSONL file, not a symlink")
    if path.stat().st_size > maximum_bytes:
        raise ValueError("import_path exceeds maximum_upload_bytes; split the import before launch")
