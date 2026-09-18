//! SQLite transaction and retention mechanics, called only by native delivery.
use super::local::{CaptureConfiguration, Policy};
use rusqlite::{params, Connection};
use std::fs::OpenOptions;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(super) struct Pending {
    pub policy: Policy,
    pub payload: String,
    pub experience_id: String,
    pub response_id: String,
    pub captured_at: u64,
}

fn safe_error(_: rusqlite::Error) -> String {
    "CLaaS capture database operation failed".into()
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0)
}

pub(super) fn validate(config: &CaptureConfiguration) -> Result<(), String> {
    let invalid = || "invalid bounded CLaaS capture configuration".to_string();
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
        Err(_) => return Err("cannot create CLaaS capture database".into()),
    }
    let connection = Connection::open(path).map_err(safe_error)?;
    connection
        .busy_timeout(Duration::from_millis(100))
        .map_err(safe_error)?;
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;
         CREATE TABLE IF NOT EXISTS claas_experiences (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           experience_id TEXT NOT NULL UNIQUE,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL,
           response_id TEXT NOT NULL, captured_at INTEGER NOT NULL,
           expires_at INTEGER NOT NULL, payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id, application_id, response_id));
         CREATE INDEX IF NOT EXISTS claas_scope_sequence
           ON claas_experiences(user_id, application_id, sequence);",
        )
        .map_err(safe_error)?;
    Ok(connection)
}

pub(super) fn persist(connection: &mut Connection, item: Pending) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO claas_experiences
         (experience_id,user_id,application_id,response_id,captured_at,expires_at,payload_bytes,payload)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT DO NOTHING",
        params![item.experience_id, item.policy.scope.user_id, item.policy.scope.application_id,
            item.response_id, item.captured_at as i64, (item.captured_at + item.policy.retention_seconds) as i64,
            item.payload.len() as i64, item.payload],
    )?;
    prune(&transaction, &item.policy, now())?;
    transaction.commit()
}

pub(super) fn prune(
    connection: &Connection,
    policy: &Policy,
    timestamp: u64,
) -> rusqlite::Result<()> {
    connection.execute(
        "DELETE FROM claas_experiences WHERE user_id=?1 AND application_id=?2 AND expires_at<=?3",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            timestamp as i64
        ],
    )?;
    connection.execute(
        "DELETE FROM claas_experiences WHERE sequence IN (
           SELECT sequence FROM (
             SELECT sequence, ROW_NUMBER() OVER (ORDER BY sequence DESC) AS rank,
               SUM(payload_bytes) OVER (ORDER BY sequence DESC) AS bytes
             FROM claas_experiences WHERE user_id=?1 AND application_id=?2
           ) WHERE rank>?3 OR bytes>?4)",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            policy.maximum_experiences as i64,
            policy.maximum_storage_bytes as i64
        ],
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "local_store_test.rs"]
mod tests;
