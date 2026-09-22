"""Canonical domain-owned table definitions accepted by every content database writer."""

TRACE_TABLE_SQL = {
    "trace_store_schema": """CREATE TABLE trace_store_schema
        (version INTEGER PRIMARY KEY CHECK(version=1)) STRICT""",
    "trace_records": """CREATE TABLE trace_records (
        record_sha256 TEXT PRIMARY KEY CHECK(length(record_sha256)=64),
        trace_id TEXT NOT NULL, payload TEXT NOT NULL
    ) STRICT""",
    "trace_imports": """CREATE TABLE trace_imports (
        import_id TEXT PRIMARY KEY, source_format TEXT NOT NULL,
        source TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT""",
    "trace_import_records": """CREATE TABLE trace_import_records (
        import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        record_sha256 TEXT NOT NULL REFERENCES trace_records(record_sha256),
        source TEXT NOT NULL,
        PRIMARY KEY(import_id,ordinal)
    ) STRICT""",
    "trace_project_imports": """CREATE TABLE trace_project_imports (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL, import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        UNIQUE(project_id,import_id)
    ) STRICT""",
}

PROJECT_TABLE_SQL = {
    "project_store_schema": """CREATE TABLE project_store_schema
        (version INTEGER PRIMARY KEY CHECK(version=1)) STRICT""",
    "project_config_versions": """CREATE TABLE project_config_versions (
        project_id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
        sha256 TEXT NOT NULL CHECK(length(sha256)=64), payload BLOB NOT NULL,
        PRIMARY KEY(project_id, version), UNIQUE(project_id, sha256)
    ) STRICT""",
    "project_config_heads": """CREATE TABLE project_config_heads (
        project_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
        FOREIGN KEY(project_id, version) REFERENCES project_config_versions(project_id, version)
    ) STRICT""",
    "project_artifacts": """CREATE TABLE project_artifacts (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL, artifact_type TEXT NOT NULL,
        manifest BLOB NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, artifact_id)
    ) STRICT""",
    "project_artifact_inputs": """CREATE TABLE project_artifact_inputs (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL, 
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        input_id TEXT NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, artifact_id, ordinal),
        FOREIGN KEY(project_id, artifact_id) REFERENCES project_artifacts(project_id, artifact_id)
    ) STRICT""",
    "project_artifact_files": """CREATE TABLE project_artifact_files (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL, path TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256)=64), 
        size_bytes INTEGER NOT NULL CHECK(size_bytes>=0),
        payload BLOB, blob_path TEXT,
        PRIMARY KEY(project_id, artifact_id, path),
        FOREIGN KEY(project_id, artifact_id) REFERENCES project_artifacts(project_id, artifact_id),
        CHECK((payload IS NULL) != (blob_path IS NULL))
    ) STRICT""",
    "project_state_records": """CREATE TABLE project_state_records (
        project_id TEXT NOT NULL, namespace TEXT NOT NULL, record_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision>0), payload BLOB NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, namespace, record_id)
    ) STRICT""",
    "project_state_events": """CREATE TABLE project_state_events (
        project_id TEXT NOT NULL, namespace TEXT NOT NULL, 
        sequence INTEGER NOT NULL CHECK(sequence>0),
        event_id TEXT NOT NULL, payload BLOB NOT NULL, 
        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, namespace, sequence), UNIQUE(project_id, namespace, event_id)
    ) STRICT""",
}
