//! SQLite transaction and retention mechanics, called only by native delivery.
use super::local::{CaptureConfiguration, Policy};
use rusqlite::{params, Connection};
use std::fs::OpenOptions;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Complete import definitions are also verified by Python before shared writes.
const TRACE_TABLE_SQL: &[(&str, &str)] = &[
    (
        "trace_store_schema",
        "CREATE TABLE trace_store_schema (version INTEGER PRIMARY KEY CHECK(version=1)) STRICT",
    ),
    (
        "trace_records",
        "CREATE TABLE trace_records (
        record_sha256 TEXT PRIMARY KEY CHECK(length(record_sha256)=64),
        trace_id TEXT NOT NULL, payload TEXT NOT NULL
    ) STRICT",
    ),
    (
        "trace_imports",
        "CREATE TABLE trace_imports (
        import_id TEXT PRIMARY KEY, source_format TEXT NOT NULL,
        source TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT",
    ),
    (
        "trace_import_records",
        "CREATE TABLE trace_import_records (
        import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        record_sha256 TEXT NOT NULL REFERENCES trace_records(record_sha256),
        source TEXT NOT NULL,
        PRIMARY KEY(import_id,ordinal)
    ) STRICT",
    ),
    (
        "trace_project_imports",
        "CREATE TABLE trace_project_imports (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL, import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        UNIQUE(project_id,import_id)
    ) STRICT",
    ),
];

// Project-owned metadata shares the file but remains outside capture retention.
const PROJECT_TABLE_SQL: &[(&str, &str)] = &[
    (
        "project_store_schema",
        "CREATE TABLE project_store_schema (version INTEGER PRIMARY KEY CHECK(version=1)) STRICT",
    ),
    (
        "project_config_versions",
        "CREATE TABLE project_config_versions (
        project_id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
        sha256 TEXT NOT NULL CHECK(length(sha256)=64), payload BLOB NOT NULL,
        PRIMARY KEY(project_id, version), UNIQUE(project_id, sha256)
    ) STRICT",
    ),
    (
        "project_config_heads",
        "CREATE TABLE project_config_heads (
        project_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
        FOREIGN KEY(project_id, version) REFERENCES project_config_versions(project_id, version)
    ) STRICT",
    ),
    (
        "project_artifacts",
        "CREATE TABLE project_artifacts (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL, artifact_type TEXT NOT NULL,
        manifest BLOB NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, artifact_id)
    ) STRICT",
    ),
    (
        "project_artifact_inputs",
        "CREATE TABLE project_artifact_inputs (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        input_id TEXT NOT NULL, sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, artifact_id, ordinal),
        FOREIGN KEY(project_id, artifact_id) REFERENCES project_artifacts(project_id, artifact_id)
    ) STRICT",
    ),
    (
        "project_artifact_files",
        "CREATE TABLE project_artifact_files (
        project_id TEXT NOT NULL, artifact_id TEXT NOT NULL, path TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        size_bytes INTEGER NOT NULL CHECK(size_bytes>=0),
        payload BLOB, blob_path TEXT,
        PRIMARY KEY(project_id, artifact_id, path),
        FOREIGN KEY(project_id, artifact_id) REFERENCES project_artifacts(project_id, artifact_id),
        CHECK((payload IS NULL) != (blob_path IS NULL))
    ) STRICT",
    ),
    (
        "project_state_records",
        "CREATE TABLE project_state_records (
        project_id TEXT NOT NULL, namespace TEXT NOT NULL, record_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision>0), payload BLOB NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, namespace, record_id)
    ) STRICT",
    ),
    (
        "project_state_events",
        "CREATE TABLE project_state_events (
        project_id TEXT NOT NULL, namespace TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK(sequence>0),
        event_id TEXT NOT NULL, payload BLOB NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256)=64),
        PRIMARY KEY(project_id, namespace, sequence), UNIQUE(project_id, namespace, event_id)
    ) STRICT",
    ),
];

pub(super) struct Pending {
    pub policy: Policy,
    pub payload: String,
    pub experience_id: String,
    pub response_id: String,
    pub captured_at: u64,
}

/// A committed write remains successful even when journal cleanup must retry.
pub(super) struct Persisted {
    pub maintenance_failed: bool,
}

fn safe_error(_: rusqlite::Error) -> String {
    "local gateway capture database operation failed".into()
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0)
}

pub(super) fn validate(config: &CaptureConfiguration) -> Result<(), String> {
    let invalid = || "invalid bounded local gateway capture configuration".to_string();
    if !Path::new(&config.database_path).is_absolute()
        || !(1..=4096).contains(&config.queue_capacity)
    {
        return Err(invalid());
    }
    let mut keys = std::collections::HashSet::new();
    let mut policies = std::collections::HashMap::new();
    for binding in &config.bindings {
        let policy = &binding.policy;
        if policies
            .insert(
                (&policy.scope.user_id, &policy.scope.application_id),
                policy,
            )
            .is_some_and(|previous| previous != policy)
        {
            return Err(invalid());
        }
        if [
            &binding.alias,
            &policy.scope.user_id,
            &policy.scope.application_id,
        ]
        .iter()
        .any(|value| value.trim().is_empty() || value.len() > 512)
            || !(1..=1_000_000).contains(&policy.maximum_experiences)
            || policy.maximum_experience_bytes == 0
            || policy.maximum_experience_bytes > policy.maximum_storage_bytes
            || policy.maximum_storage_bytes > i64::MAX as usize
            || policy.retention_seconds == 0
            || policy.retention_seconds > i64::MAX as u64 / 2
            || !keys.insert((&policy.scope.user_id, &binding.alias))
        {
            return Err(invalid());
        }
    }
    Ok(())
}

pub(super) fn open_database(path: &Path) -> Result<Connection, String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(file) => drop(file),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err("cannot create local gateway capture database".into()),
    }
    let connection = Connection::open(path).map_err(safe_error)?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(safe_error)?;
    let foreign_tables: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table'
             AND name NOT LIKE 'sqlite_%' AND name NOT IN (
               'gateway_captures', 'trace_store_schema', 'trace_records',
               'trace_imports', 'trace_import_records', 'trace_project_imports',
               'project_store_schema', 'project_config_versions', 'project_config_heads',
               'project_artifacts', 'project_artifact_inputs', 'project_artifact_files',
               'project_state_records', 'project_state_events'))",
            [],
            |row| row.get(0),
        )
        .map_err(safe_error)?;
    if foreign_tables {
        return Err(
            "unsupported local capture database schema; preserve this file and use a fresh traffic database"
                .into(),
        );
    }
    let trace_tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN (
               'trace_store_schema', 'trace_records', 'trace_imports',
               'trace_import_records', 'trace_project_imports')",
            [],
            |row| row.get(0),
        )
        .map_err(safe_error)?;
    if trace_tables != 0 {
        if trace_tables != 5 {
            return Err("incomplete trace import schema; preserve this traffic database".into());
        }
        for (table, expected) in TRACE_TABLE_SQL {
            let saved: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .map_err(safe_error)?;
            let normalize = |sql: &str| {
                sql.split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase()
            };
            if normalize(&saved) != normalize(expected) {
                return Err(
                    "incompatible trace table definition; preserve this traffic database".into(),
                );
            }
        }
        let supported: bool = connection
            .query_row(
                "SELECT COUNT(*)=1 AND MIN(version)=1 FROM trace_store_schema",
                [],
                |row| row.get(0),
            )
            .map_err(safe_error)?;
        if !supported {
            return Err(
                "unsupported trace import schema; use a matching Experiential release".into(),
            );
        }
    }
    let project_tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN (
               'project_store_schema', 'project_config_versions', 'project_config_heads',
               'project_artifacts', 'project_artifact_inputs', 'project_artifact_files',
               'project_state_records', 'project_state_events')",
            [],
            |row| row.get(0),
        )
        .map_err(safe_error)?;
    if project_tables != 0 {
        if project_tables != 8 {
            return Err("incomplete project schema; preserve this traffic database".into());
        }
        for (table, expected) in PROJECT_TABLE_SQL {
            let saved: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .map_err(safe_error)?;
            let normalize = |sql: &str| {
                sql.split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase()
            };
            if normalize(&saved) != normalize(expected) {
                return Err(
                    "incompatible project table definition; preserve this traffic database".into(),
                );
            }
        }
        let supported: bool = connection
            .query_row(
                "SELECT COUNT(*)=1 AND MIN(version)=1 FROM project_store_schema",
                [],
                |row| row.get(0),
            )
            .map_err(safe_error)?;
        if !supported {
            return Err("unsupported project schema; use a matching Experiential release".into());
        }
    }
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;
         CREATE TABLE IF NOT EXISTS gateway_captures (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           experience_id TEXT NOT NULL UNIQUE,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL,
           response_id TEXT NOT NULL, captured_at INTEGER NOT NULL,
           expires_at INTEGER NOT NULL, payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id, application_id, response_id));
         CREATE INDEX IF NOT EXISTS gateway_capture_scope_sequence
           ON gateway_captures(user_id, application_id, sequence);",
        )
        .map_err(safe_error)?;
    Ok(connection)
}

pub(super) fn persist(connection: &mut Connection, item: Pending) -> rusqlite::Result<Persisted> {
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO gateway_captures
         (experience_id,user_id,application_id,response_id,captured_at,expires_at,payload_bytes,payload)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT DO NOTHING",
        params![item.experience_id, item.policy.scope.user_id, item.policy.scope.application_id,
            item.response_id, item.captured_at as i64, (item.captured_at + item.policy.retention_seconds) as i64,
            item.payload.len() as i64, item.payload],
    )?;
    let removed = prune_rows(&transaction, &item.policy, now())?;
    transaction.commit()?;
    Ok(Persisted {
        maintenance_failed: removed > 0 && truncate_wal(connection).is_err(),
    })
}

pub(super) fn prune(
    connection: &Connection,
    policy: &Policy,
    timestamp: u64,
) -> rusqlite::Result<()> {
    prune_rows(connection, policy, timestamp)?;
    // Retry even when no rows were deleted: a prior checkpoint may have been
    // blocked by a reader. Idle destination maintenance calls this again.
    truncate_wal(connection)
}

fn truncate_wal(connection: &Connection) -> rusqlite::Result<()> {
    let busy: i64 =
        connection.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    if busy != 0 {
        return Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            None,
        ));
    }
    Ok(())
}

fn prune_rows(connection: &Connection, policy: &Policy, timestamp: u64) -> rusqlite::Result<usize> {
    let expired = connection.execute(
        "DELETE FROM gateway_captures WHERE user_id=?1 AND application_id=?2 AND expires_at<=?3",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            timestamp as i64
        ],
    )?;
    let evicted = connection.execute(
        "DELETE FROM gateway_captures WHERE sequence IN (
           SELECT sequence FROM (
             SELECT sequence, ROW_NUMBER() OVER (ORDER BY sequence DESC) AS rank,
               SUM(payload_bytes) OVER (ORDER BY sequence DESC) AS bytes
             FROM gateway_captures WHERE user_id=?1 AND application_id=?2
           ) WHERE rank>?3 OR bytes>?4)",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            policy.maximum_experiences as i64,
            policy.maximum_storage_bytes as i64
        ],
    )?;
    Ok(expired + evicted)
}

#[cfg(test)]
#[path = "local_store_test.rs"]
mod tests;
