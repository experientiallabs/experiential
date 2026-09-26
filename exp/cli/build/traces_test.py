"""Builds mine the exact saved import while source acquisition stays resumable and scoped."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.cli.build.traces import load_build_traces
from exp.common.project import ArtifactStore
from exp.common.project.paths import ProjectPaths
from exp.common.traces import load_trace_dataset
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.ingest.persistence_test import _source
from exp.common.traces.ingest.sources import load_trace_source
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.runtime.gateway.ingest import load_gateway_capture
from exp.runtime.gateway.ingest.conversion_test import _database, _experience
from exp.simulation.build import build_task_set


def test_build_preserves_credential_variable_names_in_retrieved_documentation(
    tmp_path: Path,
) -> None:
    """Public code examples survive SQLite import, immutable evidence, and scenario mining."""
    source = _source(tmp_path, count=1)
    documentation = "Create Boltz(api_key=os.environ['BOLTZ_API_KEY']); set BRAINTRUST_API_KEY."
    source.write_text(source.read_text().replace("Acme is a company.", documentation))
    root = tmp_path / "state"
    normalized = load_build_traces("powerset", root=root, path=source, source="chat-json")
    store = ArtifactStore(ProjectPaths(root=root, project_id="powerset"))

    built = build_task_set(
        normalized, store, created_at=datetime(2026, 9, 23, tzinfo=UTC), code_revision="test"
    )

    restored = load_trace_dataset(store, built.trace_dataset.dataset.dataset_id)
    assert restored.traces == normalized.traces
    assert documentation in restored.traces[0].model_dump_json()
    assert len(built.mining.tasks) == 1
    assert built.trace_dataset.dataset.invalid_trace_count == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_file_evidence_matches_saved_import_and_deduplicates(
    tmp_path: Path, dry_run: bool
) -> None:
    """Mining receives all canonical evidence and exclusions from the selected source."""
    source = _source(tmp_path)
    root = tmp_path / "state"
    expected = load_trace_source("chat-json", source)
    result = load_build_traces(
        "powerset", root=root, path=source, source="chat-json", dry_run=dry_run
    )
    assert result == expected
    assert len(result.traces) == 20 and len(result.issues) == 1
    if dry_run:
        assert not root.exists()
        return
    store = SQLiteTraceStore(trace_database_path(root))
    imports = store.list_imports("powerset")
    assert len(imports) == 1
    assert read_ingested_traces(root, imports[0]) == result
    assert load_build_traces("powerset", root=root, path=source, source="chat-json") == result
    assert store.list_imports("powerset") == imports
    source.unlink()
    assert read_ingested_traces(root, imports[0]) == expected


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_gateway_evidence_preserves_identity_and_exact_snapshot(
    tmp_path: Path, dry_run: bool
) -> None:
    """Only the requested identity's retained captures reach the build corpus."""
    source = tmp_path / "traffic.db"
    root = tmp_path / "state"
    _database(source, (_experience("developer"), _experience("other")))
    expected = load_gateway_capture(source, identity_id="developer")
    result = load_build_traces(
        "powerset", root=root, path=source, source="gateway", identity="developer", dry_run=dry_run
    )
    assert result == expected and len(result.traces) == 1
    if dry_run:
        assert not root.exists()
    else:
        imports = SQLiteTraceStore(trace_database_path(root)).list_imports("powerset")
        assert len(imports) == 1
        assert read_ingested_traces(root, imports[0]) == expected


@pytest.mark.parametrize(
    "source,identity,message",
    [
        ("gateway", None, "requires --identity ID"),
        ("chat-json", "default", "requires --source gateway"),
        ("unknown", None, "unsupported trace source"),
    ],
)
def test_invalid_build_source_creates_no_workspace(
    tmp_path: Path, source: str, identity: str | None, message: str
) -> None:
    """Invalid source selection fails before any durable import or project write."""
    root = tmp_path / "state"
    with pytest.raises(ValueError, match=message):
        load_build_traces(
            "powerset", root=root, path=tmp_path / "missing", source=source, identity=identity
        )
    assert not root.exists()


def test_empty_build_source_is_not_published(tmp_path: Path) -> None:
    """Unusable evidence cannot become a selected build corpus."""
    source = tmp_path / "empty.jsonl"
    source.write_text("{}\n")
    root = tmp_path / "state"
    with pytest.raises(ValueError, match="no valid canonical traces"):
        load_build_traces("powerset", root=root, path=source, source="chat-json")
    assert not root.exists()
