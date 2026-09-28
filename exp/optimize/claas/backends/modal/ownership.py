"""App ownership survives Modal Dict expiry without a distributed SQLite lock."""

import modal

_MARKER = ".claas-modal-owner"
_OWNER_REGISTRY = "experiential-claas-volume-owners"


async def bind_owner(volume: modal.Volume, *, app_name: str, environment_name: str) -> None:
    """Check permanent ownership, then serialize first use with a conditional Dict put.

    Dict entries expire after seven idle days. Every run is bounded below one day,
    so the entry excludes competing Apps throughout startup and execution. The
    committed marker preserves the original App assignment across idle periods.
    """
    stored = bytearray()
    try:
        async for chunk in volume.read_file.aio(_MARKER):
            stored.extend(chunk)
            if len(stored) > 64:
                raise ValueError("invalid Modal Volume owner marker; inspect the Volume")
    except FileNotFoundError:
        pass
    else:
        if bytes(stored) != app_name.encode():
            raise ValueError("the state Volume belongs to another Modal App; use its original App")
    owners = modal.Dict.from_name(
        _OWNER_REGISTRY,
        environment_name=environment_name,
        create_if_missing=True,
    )
    await owners.put.aio(volume.object_id, app_name, skip_if_exists=True)
    if await owners.get.aio(volume.object_id) != app_name:
        raise ValueError("the state Volume belongs to another Modal App; use its original App")


async def commit_owner(sandbox: modal.Sandbox, app_name: str) -> None:
    """Commit the canonical App before importing data or starting the service."""
    await sandbox.filesystem.write_text.aio(app_name, f"/state/{_MARKER}")
    operation = await sandbox.exec.aio("/usr/bin/sync", "/state")
    await operation.wait.aio()
    if operation.returncode != 0:
        raise RuntimeError(
            "Modal Volume ownership commit failed; inspect the Volume before retrying"
        )
