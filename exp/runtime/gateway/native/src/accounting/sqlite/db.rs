//! Parameterized SQL primitives; database diagnostics never cross the client boundary.

use rusqlite::{
    types::{Value as SqlValue, ValueRef},
    Connection, Params,
};
use serde_json::{Map, Value};

use crate::errors::PublicError;

pub(super) type Result<T> = std::result::Result<T, PublicError>;

pub(super) fn error(error: rusqlite::Error) -> PublicError {
    if matches!(error,rusqlite::Error::SqliteFailure(ref e,_) if e.code==rusqlite::ErrorCode::ConstraintViolation)
    {
        PublicError::new(
            500,
            "accounting_integrity",
            "Accounting write conflicts with durable state.",
            "api_error",
        )
    } else {
        PublicError::internal()
    }
}

pub(super) fn invariant(message: &str) -> PublicError {
    PublicError::new(500, "accounting_invariant", message, "api_error")
}

pub(super) fn query(db: &Connection, sql: &str, parameters: impl Params) -> Result<Vec<Value>> {
    let mut statement = db.prepare_cached(sql).map_err(error)?;
    let names: Vec<String> = statement
        .column_names()
        .iter()
        .map(|s| (*s).into())
        .collect();
    let mut rows = statement.query(parameters).map_err(error)?;
    let mut result = Vec::new();
    while let Some(row) = rows.next().map_err(error)? {
        let mut object = Map::new();
        for (i, name) in names.iter().enumerate() {
            let value = match row.get_ref(i).map_err(error)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(n) => n.into(),
                ValueRef::Real(n) => Value::from(n),
                ValueRef::Text(s) => std::str::from_utf8(s)
                    .map_err(|_| PublicError::internal())?
                    .into(),
                ValueRef::Blob(_) => return Err(PublicError::internal()),
            };
            object.insert(name.clone(), value);
        }
        result.push(Value::Object(object));
    }
    Ok(result)
}

fn parameter(value: &Value) -> Result<SqlValue> {
    match value {
        Value::Null => Ok(SqlValue::Null),
        Value::Bool(b) => Ok(SqlValue::Integer(i64::from(*b))),
        Value::Number(n) => n
            .as_i64()
            .map(SqlValue::Integer)
            .ok_or_else(PublicError::internal),
        Value::String(s) => Ok(SqlValue::Text(s.clone())),
        _ => Ok(SqlValue::Text(value.to_string())),
    }
}

/// Column names come only from implementation-owned static field lists.
pub(super) fn insert(db: &Connection, table: &str, fields: &[(&str, Value)]) -> Result<()> {
    let names = fields
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ");
    let slots = vec!["?"; fields.len()].join(", ");
    let parameters = fields
        .iter()
        .map(|(_, value)| parameter(value))
        .collect::<Result<Vec<_>>>()?;
    db.execute(
        &format!("INSERT INTO {table} ({names}) VALUES ({slots})"),
        rusqlite::params_from_iter(parameters),
    )
    .map_err(error)?;
    Ok(())
}

pub(super) fn update(
    db: &Connection,
    table: &str,
    id_column: &str,
    id: &str,
    fields: &[(&str, Value)],
) -> Result<()> {
    let names = fields
        .iter()
        .map(|(name, _)| format!("{name} = ?"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut parameters = fields
        .iter()
        .map(|(_, value)| parameter(value))
        .collect::<Result<Vec<_>>>()?;
    parameters.push(SqlValue::Text(id.into()));
    if db
        .execute(
            &format!("UPDATE {table} SET {names} WHERE {id_column} = ?"),
            rusqlite::params_from_iter(parameters),
        )
        .map_err(error)?
        != 1
    {
        return Err(PublicError::internal());
    }
    Ok(())
}
