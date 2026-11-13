//! One SQLite library for local authority readers and native accounting writers.
//!
//! Cursors retain prepared statements without retaining the interpreter. The
//! connection mutex serializes stepping, finalization and close across threads.

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyString, PyTuple};
use rusqlite::{ffi, types::Value, Connection, OpenFlags};

struct Shared {
    db: Option<Connection>,
    statements: HashSet<usize>,
}

struct Error {
    code: i32,
    message: String,
}
type Result<T> = std::result::Result<T, Error>;

impl Error {
    fn closed() -> Self {
        Self {
            code: ffi::SQLITE_MISUSE,
            message: "Cannot operate on a closed database or cursor.".into(),
        }
    }
    fn lock() -> Self {
        Self {
            code: ffi::SQLITE_INTERNAL,
            message: "SQLite connection lock failed.".into(),
        }
    }
    fn sql(error: rusqlite::Error) -> Self {
        let code = match &error {
            rusqlite::Error::SqliteFailure(error, _) => error.extended_code,
            _ => ffi::SQLITE_ERROR,
        };
        Self {
            code,
            message: error.to_string(),
        }
    }
    fn python(self, py: Python<'_>) -> PyErr {
        let class = match self.code & 0xff {
            ffi::SQLITE_CONSTRAINT => "IntegrityError",
            ffi::SQLITE_MISUSE => "ProgrammingError",
            ffi::SQLITE_TOOBIG | ffi::SQLITE_RANGE | ffi::SQLITE_MISMATCH => "DataError",
            _ => "OperationalError",
        };
        let value = (|| {
            let error = py
                .import("sqlite3")?
                .getattr(class)?
                .call1((self.message,))?;
            error.setattr("sqlite_errorcode", self.code)?;
            Ok::<_, PyErr>(error)
        })();
        match value {
            Ok(value) => PyErr::from_value(value),
            Err(error) => error,
        }
    }
}

impl Shared {
    fn handle(&self) -> Result<*mut ffi::sqlite3> {
        // SAFETY: Every caller holds the connection mutex; close and finalizers
        // acquire that same mutex before invalidating the handle.
        self.db
            .as_ref()
            .map(|db| unsafe { db.handle() })
            .ok_or_else(Error::closed)
    }
    fn error(&self, code: i32) -> Error {
        let message = self
            .handle()
            .ok()
            .map(|db| unsafe {
                CStr::from_ptr(ffi::sqlite3_errmsg(db))
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| "SQLite operation failed.".into());
        Error { code, message }
    }
    fn finalize(&mut self, statement: usize) {
        if self.statements.remove(&statement) {
            // SAFETY: Only this set owns statements; removal prevents double finalization.
            unsafe {
                ffi::sqlite3_finalize(statement as *mut ffi::sqlite3_stmt);
            }
        }
    }
    fn close(&mut self) {
        for statement in self.statements.drain() {
            unsafe {
                ffi::sqlite3_finalize(statement as *mut ffi::sqlite3_stmt);
            }
        }
        self.db.take();
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        self.close();
    }
}

enum Parameters {
    Positional(Vec<Value>),
    Named(Vec<(String, Value)>),
}

fn value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if value.is_instance_of::<PyInt>() {
        return value
            .extract::<i64>()
            .map(Value::Integer)
            .map_err(|_| PyOverflowError::new_err("SQLite integers must fit signed 64 bits"));
    }
    if value.is_instance_of::<PyFloat>() {
        return value.extract().map(Value::Real);
    }
    if value.is_instance_of::<PyString>() {
        return value.extract().map(Value::Text);
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Value::Blob(bytes.as_bytes().to_vec()));
    }
    Err(PyTypeError::new_err(
        "SQLite parameters must be strings, numbers, bytes or None",
    ))
}

fn parameters(parameters: Option<&Bound<'_, PyAny>>) -> PyResult<Parameters> {
    let Some(parameters) = parameters else {
        return Ok(Parameters::Positional(Vec::new()));
    };
    if let Ok(dictionary) = parameters.cast::<PyDict>() {
        return dictionary
            .iter()
            .map(|(key, item)| Ok((key.extract()?, value(&item)?)))
            .collect::<PyResult<Vec<_>>>()
            .map(Parameters::Named);
    }
    parameters
        .try_iter()?
        .map(|item| value(&item?))
        .collect::<PyResult<Vec<_>>>()
        .map(Parameters::Positional)
}

fn bind(
    shared: &Shared,
    statement: *mut ffi::sqlite3_stmt,
    index: i32,
    value: &Value,
) -> Result<()> {
    // SAFETY: The connection lock retains the live statement and TRANSIENT
    // makes SQLite own every borrowed string/blob before this call returns.
    let code = unsafe {
        match value {
            Value::Null => ffi::sqlite3_bind_null(statement, index),
            Value::Integer(n) => ffi::sqlite3_bind_int64(statement, index, *n),
            Value::Real(n) => ffi::sqlite3_bind_double(statement, index, *n),
            Value::Text(s) => ffi::sqlite3_bind_text64(
                statement,
                index,
                s.as_ptr().cast(),
                s.len() as u64,
                ffi::SQLITE_TRANSIENT(),
                ffi::SQLITE_UTF8 as u8,
            ),
            Value::Blob(b) => ffi::sqlite3_bind_blob64(
                statement,
                index,
                b.as_ptr().cast(),
                b.len() as u64,
                ffi::SQLITE_TRANSIENT(),
            ),
        }
    };
    if code != ffi::SQLITE_OK {
        return Err(shared.error(code));
    }
    Ok(())
}

#[pyclass(frozen)]
pub(crate) struct NativeSQLiteConnection {
    inner: Arc<Mutex<Shared>>,
}

impl NativeSQLiteConnection {
    /// Borrow an existing transaction for cold administrative accounting.
    pub(crate) fn accounting<T>(
        &self,
        operation: impl FnOnce(&Connection) -> std::result::Result<T, crate::errors::PublicError>,
    ) -> std::result::Result<T, crate::errors::PublicError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| crate::errors::PublicError::internal())?;
        operation(
            guard
                .db
                .as_ref()
                .ok_or_else(crate::errors::PublicError::internal)?,
        )
    }
}

#[pyclass]
pub(crate) struct NativeSQLiteCursor {
    connection: Arc<Mutex<Shared>>,
    statement: usize,
    names: Vec<String>,
    pending: Option<Vec<Value>>,
    done: bool,
    dml: bool,
    rowcount: i64,
    lastrowid: i64,
}

impl NativeSQLiteCursor {
    fn prepare(connection: Arc<Mutex<Shared>>, sql: &str, parameters: Parameters) -> Result<Self> {
        let mut shared = connection.lock().map_err(|_| Error::lock())?;
        let db = shared.handle()?;
        let sql = CString::new(sql).map_err(|_| Error {
            code: ffi::SQLITE_MISUSE,
            message: "SQL cannot contain NUL characters".into(),
        })?;
        let mut statement = std::ptr::null_mut();
        let mut tail = std::ptr::null();
        // SAFETY: SQLite borrows the NUL-terminated SQL only while preparing.
        let code =
            unsafe { ffi::sqlite3_prepare_v2(db, sql.as_ptr(), -1, &mut statement, &mut tail) };
        if code != ffi::SQLITE_OK {
            return Err(shared.error(code));
        }
        if statement.is_null() {
            return Err(Error {
                code: ffi::SQLITE_MISUSE,
                message: "SQL statement is empty".into(),
            });
        }
        shared.statements.insert(statement as usize);
        let prepared = (|| {
            let remaining = unsafe { CStr::from_ptr(tail) }.to_string_lossy();
            if !remaining.trim().is_empty() {
                return Err(Error {
                    code: ffi::SQLITE_MISUSE,
                    message: "Execute accepts exactly one SQL statement".into(),
                });
            }
            let count = unsafe { ffi::sqlite3_bind_parameter_count(statement) };
            match parameters {
                Parameters::Positional(values) => {
                    if values.len() != count as usize {
                        return Err(Error {
                            code: ffi::SQLITE_MISUSE,
                            message: "Incorrect number of SQL parameters".into(),
                        });
                    }
                    for (index, value) in values.iter().enumerate() {
                        bind(&shared, statement, index as i32 + 1, value)?;
                    }
                }
                Parameters::Named(values) => {
                    for index in 1..=count {
                        let pointer = unsafe { ffi::sqlite3_bind_parameter_name(statement, index) };
                        if pointer.is_null() {
                            return Err(Error {
                                code: ffi::SQLITE_MISUSE,
                                message: "Positional parameter has no dictionary key".into(),
                            });
                        }
                        let name = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
                        let value = values
                            .iter()
                            .find(|(key, _)| name.get(1..) == Some(key.as_str()))
                            .ok_or_else(|| Error {
                                code: ffi::SQLITE_MISUSE,
                                message: "Missing named SQL parameter".into(),
                            })?;
                        bind(&shared, statement, index, &value.1)?;
                    }
                }
            }
            let names = (0..unsafe { ffi::sqlite3_column_count(statement) })
                .map(|column| unsafe {
                    CStr::from_ptr(ffi::sqlite3_column_name(statement, column))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            Ok(names)
        })();
        let names = match prepared {
            Ok(names) => names,
            Err(error) => {
                shared.finalize(statement as usize);
                return Err(error);
            }
        };
        let keyword = sql
            .to_string_lossy()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_uppercase();
        let mut cursor = Self {
            connection: connection.clone(),
            statement: statement as usize,
            names,
            pending: None,
            done: false,
            dml: matches!(keyword.as_str(), "INSERT" | "UPDATE" | "DELETE" | "REPLACE"),
            rowcount: -1,
            lastrowid: 0,
        };
        drop(shared);
        cursor.pending = cursor.step()?;
        Ok(cursor)
    }

    fn step(&mut self) -> Result<Option<Vec<Value>>> {
        let mut shared = self.connection.lock().map_err(|_| Error::lock())?;
        let db = shared.handle()?;
        if self.done {
            return Ok(None);
        }
        if !shared.statements.contains(&self.statement) {
            return Err(Error::closed());
        }
        let statement = self.statement as *mut ffi::sqlite3_stmt;
        let code = unsafe { ffi::sqlite3_step(statement) };
        self.lastrowid = unsafe { ffi::sqlite3_last_insert_rowid(db) };
        if code == ffi::SQLITE_DONE {
            self.done = true;
            if self.dml {
                self.rowcount = unsafe { ffi::sqlite3_changes64(db) };
            }
            shared.finalize(self.statement);
            return Ok(None);
        }
        if code != ffi::SQLITE_ROW {
            let error = shared.error(code);
            shared.finalize(self.statement);
            self.done = true;
            return Err(error);
        }
        let mut values = Vec::with_capacity(self.names.len());
        for index in 0..self.names.len() as i32 {
            let value = unsafe {
                match ffi::sqlite3_column_type(statement, index) {
                    ffi::SQLITE_NULL => Value::Null,
                    ffi::SQLITE_INTEGER => {
                        Value::Integer(ffi::sqlite3_column_int64(statement, index))
                    }
                    ffi::SQLITE_FLOAT => Value::Real(ffi::sqlite3_column_double(statement, index)),
                    kind => {
                        let bytes = ffi::sqlite3_column_bytes(statement, index) as usize;
                        let pointer = if kind == ffi::SQLITE_TEXT {
                            ffi::sqlite3_column_text(statement, index)
                        } else {
                            ffi::sqlite3_column_blob(statement, index).cast()
                        };
                        let bytes = if bytes == 0 {
                            &[]
                        } else {
                            std::slice::from_raw_parts(pointer, bytes)
                        };
                        if kind == ffi::SQLITE_TEXT {
                            Value::Text(String::from_utf8_lossy(bytes).into_owned())
                        } else {
                            Value::Blob(bytes.into())
                        }
                    }
                }
            };
            values.push(value);
        }
        Ok(Some(values))
    }
}

#[pymethods]
impl NativeSQLiteConnection {
    #[new]
    #[pyo3(signature = (path, busy_timeout_ms=5000, mode="rwc"))]
    fn new(py: Python<'_>, path: PathBuf, busy_timeout_ms: u64, mode: &str) -> PyResult<Self> {
        let flags = match mode {
            "ro" => OpenFlags::SQLITE_OPEN_READ_ONLY,
            "rw" => OpenFlags::SQLITE_OPEN_READ_WRITE,
            "rwc" => OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
            _ => return Err(PyValueError::new_err("Invalid SQLite open mode")),
        } | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        py.detach(|| {
            let db = Connection::open_with_flags(path, flags).map_err(Error::sql)?;
            // Match sqlite3.connect's raw connection contract. Application
            // connections and the accounting owner enable their own FK fence.
            db.pragma_update(None, "foreign_keys", false)
                .map_err(Error::sql)?;
            db.busy_timeout(Duration::from_millis(busy_timeout_ms))
                .map_err(Error::sql)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(Shared {
                    db: Some(db),
                    statements: HashSet::new(),
                })),
            })
        })
        .map_err(|e: Error| e.python(py))
    }
    #[pyo3(signature = (sql, arguments=None))]
    fn execute(
        &self,
        py: Python<'_>,
        sql: &str,
        arguments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NativeSQLiteCursor> {
        let arguments = parameters(arguments)?;
        py.detach(|| NativeSQLiteCursor::prepare(self.inner.clone(), sql, arguments))
            .map_err(|e| e.python(py))
    }
    fn executescript(&self, py: Python<'_>, sql: &str) -> PyResult<()> {
        py.detach(|| {
            let shared = self.inner.lock().map_err(|_| Error::lock())?;
            shared
                .db
                .as_ref()
                .ok_or_else(Error::closed)?
                .execute_batch(sql)
                .map_err(Error::sql)
        })
        .map_err(|e| e.python(py))
    }
    #[getter]
    fn in_transaction(&self) -> PyResult<bool> {
        let shared = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("SQLite connection lock failed"))?;
        Ok(shared.db.as_ref().is_some_and(|db| !db.is_autocommit()))
    }
    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            if let Ok(mut shared) = self.inner.lock() {
                shared.close();
            }
        });
    }
    fn backup(&self, py: Python<'_>, target: &Self) -> PyResult<()> {
        if Arc::ptr_eq(&self.inner, &target.inner) {
            return Err(PyValueError::new_err("Backup needs a different connection"));
        }
        py.detach(|| {
            // Ordering keeps reciprocal backups from deadlocking.
            let (first, second) = if Arc::as_ptr(&self.inner) < Arc::as_ptr(&target.inner) {
                (&self.inner, &target.inner)
            } else {
                (&target.inner, &self.inner)
            };
            let mut first = first.lock().map_err(|_| Error::lock())?;
            let mut second = second.lock().map_err(|_| Error::lock())?;
            let (source, destination) = if Arc::as_ptr(&self.inner) < Arc::as_ptr(&target.inner) {
                (&mut first, &mut second)
            } else {
                (&mut second, &mut first)
            };
            let backup = rusqlite::backup::Backup::new(
                source.db.as_ref().ok_or_else(Error::closed)?,
                destination.db.as_mut().ok_or_else(Error::closed)?,
            )
            .map_err(Error::sql)?;
            backup
                .run_to_completion(100, Duration::from_millis(10), None)
                .map_err(Error::sql)
        })
        .map_err(|e| e.python(py))
    }
    #[getter]
    fn total_changes(&self, py: Python<'_>) -> PyResult<u64> {
        let shared = self.inner.lock().map_err(|_| Error::lock().python(py))?;
        Ok(shared
            .db
            .as_ref()
            .ok_or_else(|| Error::closed().python(py))?
            .total_changes())
    }
}

#[pymethods]
impl NativeSQLiteCursor {
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.names.clone()
    }
    #[getter]
    fn rowcount(&self) -> i64 {
        self.rowcount
    }
    #[getter]
    fn lastrowid(&self) -> i64 {
        self.lastrowid
    }
    fn fetchone(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        let values = py
            .detach(|| {
                let values = if let Some(values) = self.pending.take() {
                    Some(values)
                } else {
                    self.step()?
                };
                if values.is_some() && !self.done {
                    self.pending = self.step()?;
                }
                Ok::<_, Error>(values)
            })
            .map_err(|e| e.python(py))?;
        let Some(values) = values else {
            return Ok(None);
        };
        let values: Vec<Py<PyAny>> = values
            .into_iter()
            .map(|value| match value {
                Value::Null => Ok(py.None()),
                Value::Integer(n) => n
                    .into_pyobject(py)
                    .map(|v| v.into_any().unbind())
                    .map_err(Into::into),
                Value::Real(n) => n
                    .into_pyobject(py)
                    .map(|v| v.into_any().unbind())
                    .map_err(Into::into),
                Value::Text(s) => s
                    .into_pyobject(py)
                    .map(|v| v.into_any().unbind())
                    .map_err(Into::into),
                Value::Blob(bytes) => Ok(PyBytes::new(py, &bytes).into_any().unbind()),
            })
            .collect::<PyResult<_>>()?;
        Ok(Some(PyTuple::new(py, values)?.unbind()))
    }
    fn close(&mut self, py: Python<'_>) {
        py.detach(|| {
            if let Ok(mut shared) = self.connection.lock() {
                shared.finalize(self.statement);
            }
        });
        self.pending = None;
        self.done = true;
    }
}

impl Drop for NativeSQLiteCursor {
    fn drop(&mut self) {
        if let Ok(mut shared) = self.connection.lock() {
            shared.finalize(self.statement);
        }
    }
}
