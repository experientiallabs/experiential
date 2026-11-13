//! Bounded cache spending proposals and immutable provider observations.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::LazyLock;

use crate::errors::PublicError;

pub(super) type Result<T> = std::result::Result<T, PublicError>;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Project {
    pub endpoint_project: String,
    pub project_number: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Authority {
    pub tenant_scope: String,
    pub credential_scope: String,
    pub minimum_input_tokens: u64,
    pub create_input_nano_usd_per_million: u64,
    pub storage_nano_usd_per_million_token_hour: u64,
    pub maximum_prefix_bytes: u64,
    pub vertex_project: Option<Project>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Offer {
    pub operation_id: String,
    pub request_id: String,
    pub attempt_id: String,
    pub tenant_scope: String,
    pub credential_scope: String,
    pub account_key_fingerprint: String,
    pub cache_key: String,
    pub resource_prefix: String,
    pub minimum_input_tokens: u64,
    pub maximum_input_tokens: u64,
    pub create_nano_usd: u64,
    pub storage_nano_usd: u64,
    pub reservation_nano_usd: u64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub requested_at: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub expires_at: f64,
    pub renewable: bool,
    pub ttl_seconds: u16,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Claim {
    Ready {
        resource_name: String,
        #[serde(deserialize_with = "crate::lossless_json::deserialize")]
        expires_at: f64,
        token_count: u64,
        #[serde(default)]
        renewal_operation_id: Option<String>,
        #[serde(default)]
        #[serde(deserialize_with = "crate::lossless_json::deserialize")]
        renew_until: Option<f64>,
    },
    Creator {
        operation_id: String,
        #[serde(deserialize_with = "crate::lossless_json::deserialize")]
        expires_at: f64,
    },
    Unavailable {
        reason: Reason,
    },
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Reason {
    Denied,
    Pending,
    Unknown,
}

#[derive(Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Observation {
    pub operation_id: String,
    pub outcome: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub observed_at: f64,
    pub resource_name: Option<String>,
    pub total_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub expire_time: Option<f64>,
    pub http_status: Option<u16>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub create_time: Option<f64>,
}

pub(super) fn require(ok: bool) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(PublicError::internal())
    }
}

pub(super) fn scope(value: &str, maximum: usize) -> Result<()> {
    static NONPRINTING: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"[\p{C}\p{Zl}\p{Zp}]|[\p{Zs}&&[^ ]]").unwrap());
    // The regex excludes Unicode separators other than ordinary ASCII space.
    // Control and format characters are never printable binding authority.
    require(
        !value.is_empty()
            && value.chars().count() <= maximum
            && value.trim() == value
            && !NONPRINTING.is_match(value),
    )
}

pub(super) fn integer(value: u64, minimum: u64) -> Result<()> {
    require(value >= minimum && value <= i64::MAX as u64)
}

pub(super) fn timestamp(value: f64) -> Result<()> {
    require(value.is_finite() && value >= 0.0)
}

pub(super) fn digest(value: &str) -> Result<()> {
    require(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
    )
}

pub(super) fn resource(prefix: &str, name: &str) -> Result<()> {
    static PREFIX: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
        r"\A(?:cachedContents/|projects/[A-Za-z0-9_-]{1,256}/locations/[A-Za-z0-9_-]{1,256}/cachedContents/)\z").unwrap()
    });
    let id = name.strip_prefix(prefix).unwrap_or("");
    require(
        PREFIX.is_match(prefix)
            && !id.is_empty()
            && id.len() <= 256
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
    )
}

impl Authority {
    pub fn validate(&self) -> Result<()> {
        scope(&self.tenant_scope, 1024)?;
        scope(&self.credential_scope, 1024)?;
        integer(self.minimum_input_tokens, 1)?;
        integer(self.create_input_nano_usd_per_million, 0)?;
        integer(self.storage_nano_usd_per_million_token_hour, 1)?;
        integer(self.maximum_prefix_bytes, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn offer(
        &self,
        operation: String,
        request: &str,
        attempt: &str,
        account: String,
        plan: &str,
        prefix: &str,
        bytes: u64,
        now: f64,
    ) -> Result<Option<Offer>> {
        self.validate()?;
        integer(bytes, 0)?;
        if bytes == 0 || bytes > self.maximum_prefix_bytes {
            return Ok(None);
        }
        let maximum = bytes.checked_add(4096).ok_or_else(PublicError::internal)?;
        integer(maximum, 1)?;
        if maximum < self.minimum_input_tokens {
            return Ok(None);
        }
        let cost = |rate: u64, duration: u128, divisor: u128| -> Result<u64> {
            let amount = (maximum as u128)
                .checked_mul(rate as u128)
                .and_then(|n| n.checked_mul(duration))
                .ok_or_else(PublicError::internal)?;
            let amount =
                u64::try_from(amount.div_ceil(divisor)).map_err(|_| PublicError::internal())?;
            integer(amount, 0)?;
            Ok(amount)
        };
        let create = cost(self.create_input_nano_usd_per_million, 1, 1_000_000)?;
        let storage = cost(
            self.storage_nano_usd_per_million_token_hour,
            300,
            3_600_000_000,
        )?;
        let total = create
            .checked_add(storage)
            .ok_or_else(PublicError::internal)?;
        integer(storage, 1)?;
        integer(total, 1)?;
        timestamp(now)?;
        for item in [&operation, request, attempt, &account] {
            scope(item, 1024)?;
        }
        digest(plan)?;
        resource(prefix, &format!("{prefix}a"))?;
        let cache_key = hash(
            &json!([self.tenant_scope, self.credential_scope, account, plan]),
            true,
        )?;
        Ok(Some(Offer {
            operation_id: operation,
            request_id: request.into(),
            attempt_id: attempt.into(),
            tenant_scope: self.tenant_scope.clone(),
            credential_scope: self.credential_scope.clone(),
            account_key_fingerprint: account,
            cache_key,
            resource_prefix: prefix.into(),
            minimum_input_tokens: self.minimum_input_tokens,
            maximum_input_tokens: maximum,
            create_nano_usd: create,
            storage_nano_usd: storage,
            reservation_nano_usd: total,
            requested_at: now,
            expires_at: now + 300.0,
            renewable: true,
            ttl_seconds: 300,
        }))
    }
}

pub(super) fn hash(value: &Value, ascii: bool) -> Result<String> {
    let bytes = if ascii {
        super::json_text::dumps(value, true)?.into_bytes()
    } else {
        super::canonical::bytes(value, false)?
    };
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

impl Offer {
    pub fn claim(&self, claim: &Claim, now: f64, freshness: bool) -> Result<()> {
        timestamp(now)?;
        let expires = match claim {
            Claim::Unavailable { .. } => return Ok(()),
            Claim::Creator {
                operation_id,
                expires_at,
            } => {
                scope(operation_id, 1024)?;
                require(operation_id == &self.operation_id && *expires_at == self.expires_at)?;
                *expires_at
            }
            Claim::Ready {
                resource_name,
                expires_at,
                token_count,
                renewal_operation_id,
                renew_until,
            } => {
                resource(&self.resource_prefix, resource_name)?;
                self.measured(*token_count)?;
                let horizon = if self.renewable {
                    self.requested_at + 3600.0
                } else {
                    self.expires_at
                };
                require(*expires_at <= horizon)?;
                require(renewal_operation_id.is_some() == renew_until.is_some())?;
                if let (Some(id), Some(until)) = (renewal_operation_id, renew_until) {
                    scope(id, 128)?;
                    timestamp(*until)?;
                    require(self.renewable && *until > *expires_at && *until <= horizon)?;
                }
                *expires_at
            }
        };
        timestamp(expires)?;
        if freshness {
            require(expires > now + 5.0 && now >= self.requested_at)?;
        }
        Ok(())
    }

    pub fn measured(&self, tokens: u64) -> Result<()> {
        integer(tokens, 1)?;
        require(tokens >= self.minimum_input_tokens && tokens <= self.maximum_input_tokens)
    }

    pub fn observation(&self, value: &Observation) -> Result<()> {
        require(value.operation_id == self.operation_id)?;
        timestamp(value.observed_at)?;
        require(value.observed_at >= self.requested_at)?;
        if let Some(name) = &value.resource_name {
            resource(&self.resource_prefix, name)?;
        }
        if let Some(tokens) = value.total_tokens {
            self.measured(tokens)?;
        }
        if let Some(time) = value.expire_time {
            timestamp(time)?;
            require(self.requested_at < time && time <= self.expires_at)?;
        }
        if let Some(time) = value.create_time {
            timestamp(time)?;
            require(
                time > 0.0
                    && time <= value.observed_at
                    && value.expire_time.is_some_and(|e| time <= e),
            )?;
        }
        if let Some(status) = value.http_status {
            require((100..=599).contains(&status))?;
        }
        match value.outcome.as_str() {
            "ready" => require(
                value.resource_name.is_some()
                    && value.total_tokens.is_some()
                    && value.expire_time.is_some()
                    && value.http_status.is_none_or(|s| (200..300).contains(&s)),
            ),
            "not_created" => require(
                value.resource_name.is_none()
                    && value.total_tokens.is_none()
                    && value.expire_time.is_none()
                    && value.create_time.is_none()
                    && value.http_status.is_none(),
            ),
            "unknown" => Ok(()),
            "rejected" => require(
                value.resource_name.is_none()
                    && value.total_tokens.is_none()
                    && value.expire_time.is_none()
                    && value.create_time.is_none()
                    && value
                        .http_status
                        .is_some_and(|s| (400..500).contains(&s) && s != 408 && s != 409),
            ),
            _ => Err(PublicError::internal()),
        }
    }
}

pub(super) fn now() -> Result<f64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .map_err(|_| PublicError::internal())
}

pub(super) fn parse_time(value: Option<&str>) -> Result<f64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(value.ok_or_else(PublicError::internal)?)
        .map_err(|_| PublicError::internal())?;
    let time = parsed.timestamp() as f64 + parsed.timestamp_subsec_nanos() as f64 / 1e9;
    timestamp(time)?;
    Ok(time)
}

pub(super) fn iso_time(value: f64) -> Result<String> {
    let dt = chrono::DateTime::from_timestamp(value as i64, 0).ok_or_else(PublicError::internal)?;
    Ok(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, false))
}
