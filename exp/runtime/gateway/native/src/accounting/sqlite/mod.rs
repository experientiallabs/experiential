//! Interpreter-independent local financial writes for admitted request owners.

mod attempts;
mod authority;
mod budgets;
mod cold;
mod db;
mod reconciliation;
mod requests;
mod snapshot;
#[cfg(feature = "test-support")]
mod test_authority;
mod valuation;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, SecondsFormat, Utc};
use pyo3::{exceptions::PyValueError, prelude::*};
use rusqlite::{params, Connection, OpenFlags};
use serde_json::{json, Value};

use crate::errors::PublicError;
use crate::native_callbacks::{CallbackTarget, NativeCallbacks, NativeTarget};
use authority::text;
use db::Result;

struct Store {
    database: Mutex<Connection>,
    root: PathBuf,
    maximum_snapshot_bytes: u64,
    #[cfg(feature = "test-support")]
    test_authority: bool,
}

fn now() -> DateTime<Utc> {
    DateTime::from(SystemTime::now())
}
fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Micros, false)
}
fn py_error(error: PublicError) -> PyErr {
    PyValueError::new_err(serde_json::to_string(&error).unwrap_or_default())
}

impl Store {
    fn proof(&self, db: &Connection, auth: &Value, deadline: Instant) -> Result<authority::Proof> {
        #[cfg(feature = "test-support")]
        if self.test_authority && test_authority::selected(db, auth)? {
            return authority::Proof::fixture(db, auth, deadline);
        }
        authority::Proof::prepare(db, &self.root, auth, self.maximum_snapshot_bytes, deadline)
    }

    fn stage<'a>(&self, snapshot: &'a Value, depth: usize) -> Result<&'a Value> {
        if snapshot["model_stages"]
            .as_array()
            .is_some_and(Vec::is_empty)
        {
            return Ok(snapshot);
        }
        #[cfg(feature = "test-support")]
        if self.test_authority {
            return test_authority::stage(snapshot, depth);
        }
        let _ = depth;
        Err(PublicError::internal())
    }

    fn write(
        &self,
        auth: Option<(&Value, Instant)>,
        operation: impl FnOnce(&Connection) -> Result<Value>,
    ) -> Result<Value> {
        let db = self.database.lock().map_err(|_| PublicError::internal())?;
        let proof = auth
            .map(|(auth, deadline)| self.proof(&db, auth, deadline))
            .transpose()?;
        db.execute_batch("BEGIN IMMEDIATE").map_err(db::error)?;
        let result = (|| {
            if let (Some(proof), Some((auth, _))) = (&proof, auth) {
                proof.validate(&db, auth)?;
            }
            let result = operation(&db)?;
            if let (Some(proof), Some((auth, _))) = (&proof, auth) {
                proof.validate(&db, auth)?;
            }
            db.execute_batch("COMMIT").map_err(db::error)?;
            Ok(result)
        })();
        if result.is_err() {
            let _ = db.execute_batch("ROLLBACK");
        }
        result
    }
}

struct Request {
    store: Arc<Store>,
    registration: Value,
    deadline: Instant,
    continuations: Option<Arc<super::continuation_store::Store>>,
}

impl Request {
    fn execute(&self, method: &str, input: Value) -> Result<Value> {
        let auth = &self.registration["authorization"];
        if input["request_id"] != auth["request_id"] {
            return Err(PublicError::internal());
        }
        if method == "remember" {
            let namespace = &input["namespace"];
            if ["organization_id", "identity_id", "alias_revision_id"]
                .iter()
                .any(|key| namespace[key] != auth[key])
            {
                return Err(PublicError::internal());
            }
            self.continuations
                .as_ref()
                .ok_or_else(PublicError::internal)?
                .remember(input)?;
            return Ok(json!({}));
        }
        if method == "price" {
            let db = self
                .store
                .database
                .lock()
                .map_err(|_| PublicError::internal())?;
            return attempts::price(
                &db,
                auth,
                text(&input, "attempt_id")?,
                input["terminal"].clone(),
            );
        }
        let auth_fence = (method == "start_attempt").then_some((auth, self.deadline));
        let result = self.store.write(auth_fence, |db| {
            let now = now();
            let stamp = timestamp(now);
            match method {
                "record_input_moderation" => {
                    db.execute("INSERT INTO gateway_input_moderation (request_id, inspection_id, organization_id, recorded_at, result) VALUES (?, ?, ?, ?, ?) ON CONFLICT(request_id, inspection_id) DO NOTHING",
                        params![text(auth, "request_id")?, text(&input, "inspection_id")?, text(auth, "organization_id")?, stamp, input["moderation"].to_string()]).map_err(db::error)?;
                    Ok(json!({}))
                }
                "start_attempt" => attempts::start(db, &self.store, &self.registration, &input["dispatch"], &stamp, &now.format("%Y-%m-01T00:00:00+00:00").to_string()),
                "settle" | "record_late_meter" => attempts::settle(db, auth, text(&input, "attempt_id")?, input["terminal"].clone(), &stamp),
                "abandon" => {
                    let pending = db::query(db, "SELECT attempt_id FROM gateway_attempts WHERE request_id=? AND organization_id=? AND state='dispatched'", params![text(auth, "request_id")?, text(auth, "organization_id")?])?;
                    let mut terminal = input["terminal"].clone();
                    terminal["accounting_recovery"] = json!(true);
                    terminal["finalize"] = json!(false);
                    for attempt in pending {
                        attempts::settle(db, auth, text(&attempt, "attempt_id")?, terminal.clone(), &stamp)?;
                    }
                    let has_attempt = !db::query(db, "SELECT attempt_id FROM gateway_attempts WHERE request_id=? LIMIT 1", params![text(auth, "request_id")?])?.is_empty();
                    let known = requests::finish(db, auth, &terminal, input["certify_no_effects"] == true,
                        if has_attempt { 0 } else { input["web_search_requests"].as_u64().unwrap_or(0) }, &stamp)?;
                    Ok(json!({"terminalized": true, "known_unbilled": known}))
                }
                _ => Err(PublicError::internal()),
            }
        })?;
        #[cfg(feature = "test-support")]
        if self.store.test_authority && method == "settle" && input["terminal"]["finalize"] != false
        {
            let mut result = result;
            result["capture"] = json!({"keep_prompt": true,
                "keep_response": input["terminal"]["state"] == "completed"});
            return Ok(result);
        }
        Ok(result)
    }
}

impl CallbackTarget for Request {
    fn handles(&self, method: &str) -> bool {
        matches!(
            method,
            "start_attempt"
                | "price"
                | "settle"
                | "abandon"
                | "record_late_meter"
                | "record_input_moderation"
                | "remember"
        )
    }
    fn call(&self, method: &str, argument: &str) -> Result<String> {
        let input =
            crate::lossless_json::from_str(argument).map_err(|_| PublicError::internal())?;
        Ok(self.execute(method, input)?.to_string())
    }
}

/// One durable SQLite writer shared by all native requests in this gateway.
#[pyclass(frozen)]
pub struct NativeSQLiteLedger {
    store: Arc<Store>,
}

#[pymethods]
impl NativeSQLiteLedger {
    /// Cold recovery and administrative callers share the serving writer's implementation.
    fn apply(
        &self,
        py: Python<'_>,
        operation: &str,
        input: &str,
        connection: &crate::sqlite_connection::NativeSQLiteConnection,
    ) -> PyResult<String> {
        py.detach(|| {
            let input: Value =
                crate::lossless_json::from_str(input).map_err(|_| PublicError::internal())?;
            connection
                .accounting(|db| cold::apply(db, &self.store, operation, &input))
                .map(|v| v.to_string())
        })
        .map_err(py_error)
    }
    #[new]
    #[pyo3(signature = (path, *, busy_timeout_ms=5000, maximum_snapshot_bytes=67_108_864))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        busy_timeout_ms: u64,
        maximum_snapshot_bytes: u64,
    ) -> PyResult<Self> {
        if busy_timeout_ms > 60_000
            || maximum_snapshot_bytes == 0
            || maximum_snapshot_bytes > 2 * 1024 * 1024 * 1024
        {
            return Err(PyValueError::new_err(
                "Invalid SQLite ledger resource bounds",
            ));
        }
        py.detach(|| {
            let path = path.canonicalize().map_err(|_| PublicError::internal())?;
            let root = path
                .parent()
                .ok_or_else(PublicError::internal)?
                .to_path_buf();
            let db = Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_WRITE
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX
                    | OpenFlags::SQLITE_OPEN_NOFOLLOW,
            )
            .map_err(db::error)?;
            db.busy_timeout(Duration::from_millis(busy_timeout_ms))
                .map_err(db::error)?;
            db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
                .map_err(db::error)?;
            let version: i64 = db
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .map_err(db::error)?;
            if version != 33 {
                return Err(PublicError::internal());
            }
            Ok(Self {
                store: Arc::new(Store {
                    database: Mutex::new(db),
                    root,
                    maximum_snapshot_bytes,
                    #[cfg(feature = "test-support")]
                    test_authority: false,
                }),
            })
        })
        .map_err(py_error)
    }

    #[cfg(feature = "test-support")]
    #[staticmethod]
    #[pyo3(signature = (path, *, busy_timeout_ms=5000, maximum_snapshot_bytes=67_108_864))]
    fn _test_chain_fixture(
        py: Python<'_>,
        path: PathBuf,
        busy_timeout_ms: u64,
        maximum_snapshot_bytes: u64,
    ) -> PyResult<Self> {
        let mut ledger = Self::new(py, path, busy_timeout_ms, maximum_snapshot_bytes)?;
        Arc::get_mut(&mut ledger.store)
            .ok_or_else(|| PyValueError::new_err("test fixture already shared"))?
            .test_authority = true;
        Ok(ledger)
    }

    /// Initial handoff only. No Python object is retained by the request target.
    #[pyo3(signature = (registration, *, continuations=None))]
    fn request(
        &self,
        py: Python<'_>,
        registration: &str,
        continuations: Option<&super::NativeContinuations>,
    ) -> PyResult<NativeCallbacks> {
        let continuations = continuations.map(|store| store.inner.clone());
        py.detach(|| {
            let registration: Value = crate::lossless_json::from_str(registration)
                .map_err(|_| PublicError::internal())?;
            let remaining = registration["remaining_seconds"]
                .as_f64()
                .filter(|s| s.is_finite() && (0.0..=86_400.0).contains(s))
                .ok_or_else(PublicError::internal)?;
            text(&registration["authorization"], "request_id")?;
            if registration["snapshot"]["authorization"] != registration["authorization"] {
                return Err(PublicError::internal());
            }
            Ok(NativeCallbacks {
                inner: NativeTarget::new(Arc::new(Request {
                    store: self.store.clone(),
                    registration,
                    deadline: Instant::now() + Duration::from_secs_f64(remaining),
                    continuations,
                })),
            })
        })
        .map_err(py_error)
    }

    /// Initial admission cleanup does not require a selected provider route.
    fn admission(&self, py: Python<'_>, authorization: &str) -> PyResult<NativeCallbacks> {
        py.detach(|| {
            let auth: Value = crate::lossless_json::from_str(authorization)
                .map_err(|_| PublicError::internal())?;
            text(&auth, "request_id")?;
            text(&auth, "organization_id")?;
            Ok(NativeCallbacks {
                inner: NativeTarget::new(Arc::new(Request {
                    store: self.store.clone(),
                    registration: json!({"authorization": auth}),
                    deadline: Instant::now(),
                    continuations: None,
                })),
            })
        })
        .map_err(py_error)
    }

    /// Recovery outside a serving execution uses the same persisted terms and writer.
    fn settle_existing(
        &self,
        py: Python<'_>,
        authorization: &str,
        attempt_id: &str,
        terminal: &str,
    ) -> PyResult<()> {
        py.detach(|| {
            let auth = crate::lossless_json::from_str(authorization)
                .map_err(|_| PublicError::internal())?;
            let terminal =
                crate::lossless_json::from_str(terminal).map_err(|_| PublicError::internal())?;
            self.store
                .write(None, |db| {
                    attempts::settle(db, &auth, attempt_id, terminal, &timestamp(now()))
                })
                .map(|_| ())
        })
        .map_err(py_error)
    }

    fn accept(&self, py: Python<'_>, authorization: &str, remaining_seconds: f64) -> PyResult<()> {
        py.detach(|| {
            let auth: Value = crate::lossless_json::from_str(authorization)
                .map_err(|_| PublicError::internal())?;
            if !remaining_seconds.is_finite() || !(0.0..=86_400.0).contains(&remaining_seconds) {
                return Err(PublicError::internal());
            }
            let deadline = Instant::now() + Duration::from_secs_f64(remaining_seconds);
            self.store
                .write(Some((&auth, deadline)), |db| {
                    let now = now();
                    let until =
                        now + chrono::Duration::milliseconds((remaining_seconds * 1000.0) as i64);
                    requests::accept(db, &auth, &timestamp(now), &timestamp(until))?;
                    Ok(Value::Null)
                })
                .map(|_| ())
        })
        .map_err(py_error)
    }
}

#[cfg(test)]
mod budget_tests;
