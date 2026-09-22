//! Persistence, retention, and isolation checks using real local SQLite.

use super::*;
use crate::capture::local::{Binding, Scope, SqliteSink};
use serde_json::json;

fn policy() -> Policy {
    Policy {
        scope: Scope {
            user_id: "user".into(),
            application_id: "app".into(),
        },
        enabled: true,
        maximum_experiences: 2,
        maximum_storage_bytes: 10000,
        maximum_experience_bytes: 2000,
        retention_seconds: 60,
    }
}

fn pending(id: &str, policy: Policy) -> Pending {
    Pending {
        policy,
        payload: json!({"id": id}).to_string(),
        experience_id: id.into(),
        response_id: id.into(),
        captured_at: now(),
    }
}

#[test]
fn sqlite_retention_deduplication_and_scope_are_independent() {
    let path = std::env::temp_dir().join(format!(
        "capture-test-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut connection = open_database(&path).unwrap();
    let app = policy();
    persist(&mut connection, pending("one", app.clone())).unwrap();
    persist(&mut connection, pending("one", app.clone())).unwrap();
    persist(&mut connection, pending("two", app.clone())).unwrap();
    let mut other = app.clone();
    other.scope.application_id = "other".into();
    persist(&mut connection, pending("other", other)).unwrap();
    persist(&mut connection, pending("three", app.clone())).unwrap();
    let ids: Vec<String> = connection
        .prepare("SELECT experience_id FROM gateway_captures ORDER BY sequence")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(ids, ["two", "other", "three"]);
    prune(&connection, &app, now() + 61).unwrap();
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM gateway_captures", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1);
    drop(connection);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn disabled_capture_never_creates_a_database() {
    let mut disabled = policy();
    disabled.enabled = false;
    let path = std::env::temp_dir().join(format!("capture-disabled-{}.db", std::process::id()));
    let config = CaptureConfiguration {
        database_path: path.to_string_lossy().into(),
        bindings: vec![Binding {
            alias: "model".into(),
            policy: disabled,
        }],
        queue_capacity: 1,
    };
    assert!(SqliteSink::open(config).unwrap().is_none());
    assert!(!path.exists());
}

#[test]
fn another_database_schema_is_rejected_without_modifying_evidence() {
    let path = std::env::temp_dir().join(format!(
        "capture-schema-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let existing = Connection::open(&path).unwrap();
    existing
        .execute_batch(
            "CREATE TABLE other_records(payload TEXT NOT NULL);
         INSERT INTO other_records VALUES ('preserve this evidence');",
        )
        .unwrap();
    drop(existing);
    let before = std::fs::read(&path).unwrap();
    let error = open_database(&path).unwrap_err();
    assert!(error.contains("use a fresh traffic database"));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn conflicting_alias_policies_are_rejected_before_any_pruning() {
    let first = policy();
    let mut second = first.clone();
    second.retention_seconds = 1;
    let config = CaptureConfiguration {
        database_path: std::env::temp_dir()
            .join("capture-conflict.db")
            .to_string_lossy()
            .into(),
        bindings: vec![
            Binding {
                alias: "a".into(),
                policy: first,
            },
            Binding {
                alias: "b".into(),
                policy: second,
            },
        ],
        queue_capacity: 1,
    };
    assert!(validate(&config).is_err());
}

#[test]
fn expired_content_is_removed_from_database_and_wal_after_readers_release() {
    let path = std::env::temp_dir().join(format!(
        "capture-erasure-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut writer = open_database(&path).unwrap();
    let marker = "synthetic-expiring-tool-output-unique-marker";
    let mut item = pending("expired", policy());
    item.payload = marker.into();
    persist(&mut writer, item).unwrap();
    let reader = Connection::open(&path).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    let _: String = reader
        .query_row("SELECT payload FROM gateway_captures", [], |row| row.get(0))
        .unwrap();
    assert!(prune(&writer, &policy(), now() + 61).is_err());
    // The reader still holds the original pages. A new committed capture must
    // report WAL cleanup separately from the successful database insertion.
    let mut single = policy();
    single.maximum_experiences = 1;
    persist(&mut writer, pending("fresh-one", single.clone())).unwrap();
    let outcome = persist(&mut writer, pending("fresh-two", single)).unwrap();
    assert!(outcome.maintenance_failed);
    let count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM gateway_captures WHERE response_id='fresh-two'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
    reader.execute_batch("ROLLBACK").unwrap();
    prune(&writer, &policy(), now() + 61).unwrap();
    for artifact in [
        &path,
        &std::path::PathBuf::from(format!("{}-wal", path.display())),
    ] {
        let data = std::fs::read(artifact).unwrap();
        assert!(!data
            .windows(marker.len())
            .any(|bytes| bytes == marker.as_bytes()));
    }
    drop(reader);
    drop(writer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn import_namespace_survives_native_retention_and_reopen() {
    let path = std::env::temp_dir().join(format!(
        "capture-import-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let existing = Connection::open(&path).unwrap();
    for (_, sql) in TRACE_TABLE_SQL {
        existing.execute_batch(sql).unwrap();
    }
    existing
        .execute_batch(
            "INSERT INTO trace_store_schema VALUES(1);
         INSERT INTO trace_records VALUES(printf('%064d',0),'source-trace','saved import');",
        )
        .unwrap();
    drop(existing);
    let mut writer = open_database(&path).unwrap();
    persist(&mut writer, pending("expired", policy())).unwrap();
    prune(&writer, &policy(), now() + 61).unwrap();
    assert_eq!(
        writer
            .query_row("SELECT payload FROM trace_records", [], |row| row
                .get::<_, String>(0))
            .unwrap(),
        "saved import"
    );
    assert_eq!(
        writer
            .query_row("SELECT COUNT(*) FROM gateway_captures", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    drop(writer);
    let mut reopened = open_database(&path).unwrap();
    persist(&mut reopened, pending("new", policy())).unwrap();
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn incomplete_or_unsupported_import_schema_is_not_modified() {
    for complete in [false, true] {
        let path = std::env::temp_dir().join(format!(
            "capture-import-schema-{}-{}.db",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let existing = Connection::open(&path).unwrap();
        existing.execute_batch("CREATE TABLE trace_store_schema(version INTEGER); INSERT INTO trace_store_schema VALUES(2);").unwrap();
        if complete {
            existing.execute_batch("CREATE TABLE trace_records(id TEXT); CREATE TABLE trace_imports(id TEXT);
                CREATE TABLE trace_import_records(id TEXT); CREATE TABLE trace_project_imports(id TEXT);").unwrap();
        }
        drop(existing);
        let before = std::fs::read(&path).unwrap();
        assert!(open_database(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn native_writer_waits_for_a_short_import_transaction() {
    let path = std::env::temp_dir().join(format!(
        "capture-import-lock-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut writer = open_database(&path).unwrap();
    let importer = Connection::open(&path).unwrap();
    importer.execute_batch("BEGIN IMMEDIATE").unwrap();
    let (started, waiting) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        started.send(()).unwrap();
        persist(&mut writer, pending("concurrent", policy())).unwrap();
    });
    waiting.recv().unwrap();
    std::thread::sleep(Duration::from_millis(250));
    importer.execute_batch("COMMIT").unwrap();
    worker.join().unwrap();
    assert_eq!(
        importer
            .query_row("SELECT COUNT(*) FROM gateway_captures", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    drop(importer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn complete_import_names_cannot_hide_changed_columns_or_constraints() {
    for mutation in [
        "ALTER TABLE trace_records ADD COLUMN injected TEXT",
        "DROP TABLE trace_project_imports; CREATE TABLE trace_project_imports (sequence INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, import_id TEXT NOT NULL)",
        "DROP TABLE trace_records; CREATE TABLE trace_records (record_sha256 TEXT PRIMARY KEY, trace_id TEXT NOT NULL, payload TEXT NOT NULL) STRICT",
    ] {
        let path = std::env::temp_dir().join(format!(
            "capture-import-definition-{}-{}.db", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let existing = Connection::open(&path).unwrap();
        for (_, sql) in TRACE_TABLE_SQL {
            existing.execute_batch(sql).unwrap();
        }
        existing.execute_batch("INSERT INTO trace_store_schema VALUES(1)").unwrap();
        existing.execute_batch(mutation).unwrap();
        drop(existing);
        let before = std::fs::read(&path).unwrap();
        assert!(open_database(&path).unwrap_err().contains("incompatible trace table"));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn project_metadata_survives_capture_retention_and_schema_drift_is_rejected() {
    let path = std::env::temp_dir().join(format!(
        "capture-project-schema-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let existing = Connection::open(&path).unwrap();
    for (_, sql) in PROJECT_TABLE_SQL {
        existing.execute_batch(sql).unwrap();
    }
    existing
        .execute_batch(
            "INSERT INTO project_store_schema VALUES(1);
        INSERT INTO project_state_records VALUES('project', 'runs', 'saved', 1,
        X'7361766564', printf('%064d',0));",
        )
        .unwrap();
    drop(existing);
    let mut connection = open_database(&path).unwrap();
    persist(&mut connection, pending("one", policy())).unwrap();
    prune(&connection, &policy(), now() + 61).unwrap();
    let saved: Vec<u8> = connection
        .query_row("SELECT payload FROM project_state_records", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(saved, b"saved");
    connection
        .execute_batch("ALTER TABLE project_config_heads ADD COLUMN incompatible TEXT")
        .unwrap();
    drop(connection);
    let before = std::fs::read(&path).unwrap();
    assert!(open_database(&path)
        .unwrap_err()
        .contains("incompatible project table"));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
}
