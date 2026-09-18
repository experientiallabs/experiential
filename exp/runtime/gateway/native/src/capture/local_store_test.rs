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
        "claas-test-{}-{}.db",
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
        .prepare("SELECT experience_id FROM claas_experiences ORDER BY sequence")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(ids, ["two", "other", "three"]);
    prune(&connection, &app, now() + 61).unwrap();
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM claas_experiences", [], |row| {
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
    let path = std::env::temp_dir().join(format!("claas-disabled-{}.db", std::process::id()));
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
fn conflicting_alias_policies_are_rejected_before_any_pruning() {
    let first = policy();
    let mut second = first.clone();
    second.retention_seconds = 1;
    let config = CaptureConfiguration {
        database_path: std::env::temp_dir()
            .join("claas-conflict.db")
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
