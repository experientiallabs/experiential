"""Local resource cleanup through the native observability callback."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_observability import NativeObservabilityMixin
from exp.runtime.gateway.snapshot_file import prepare_snapshot_file


def test_callback_cleanup_evicts_shared_memo_without_disabling_active_proof(tmp_path: Path) -> None:
    """A retiring native callback releases anchors but never closes another worker's proof."""
    ledger = SQLiteAttemptLedger(tmp_path / "gateway.db")
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"{}")
    plane = NativeObservabilityMixin()
    plane._components = cast(NativeGatewayComponents, SimpleNamespace(ledger=ledger))
    with (
        prepare_snapshot_file(tmp_path, path.name, 1024) as first,
        prepare_snapshot_file(tmp_path, "missing.models.json", 1024) as second,
    ):
        ledger.classification_memo.remember(("db", path.name, 1024), (first, second))
        assert ledger.classification_memo._entries
        plane.close_thread_resources("{}")
        assert not ledger.classification_memo._entries
        assert not ledger.classification_memo._closed
        first.validate_current()
    ledger.close()
