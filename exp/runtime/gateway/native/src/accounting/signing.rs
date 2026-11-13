//! Per-dispatch signatures over the admission-bound endpoint and exact body bytes.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use ring::hmac;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::PublicError;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub url: String,
    pub body_sha256: String,
    authority: Authority,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Authority {
    Bearer {
        token: String,
        accept: String,
    },
    Sigv4 {
        #[serde(default)]
        default_chain: bool,
        #[serde(default)]
        profile: Option<String>,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        region: String,
        service: String,
        host: String,
        path: String,
        query: String,
        accept: String,
    },
}

fn sha(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}
fn sign(key: &[u8], message: &str) -> Vec<u8> {
    hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), message.as_bytes())
        .as_ref()
        .to_vec()
}

impl Binding {
    pub fn headers(&self, input: &Value, now: DateTime<Utc>) -> Result<Value, PublicError> {
        let body = input["body"].as_str().ok_or_else(PublicError::internal)?;
        if input["url"].as_str() != Some(&self.url) || sha(body.as_bytes()) != self.body_sha256 {
            return Err(PublicError::internal());
        }
        let mut headers = BTreeMap::from([("content-type", "application/json".to_string())]);
        match &self.authority {
            Authority::Bearer { token, accept } => {
                if token.is_empty() {
                    return Err(PublicError::internal());
                }
                headers.insert("accept", accept.clone());
                headers.insert("authorization", format!("Bearer {token}"));
            }
            Authority::Sigv4 {
                default_chain,
                profile,
                access_key,
                secret_key,
                session_token,
                region,
                service,
                host,
                path,
                query,
                accept,
            } => {
                let resolved = if *default_chain {
                    Some(super::signing_credentials::resolve(
                        region,
                        profile.as_deref(),
                    )?)
                } else {
                    None
                };
                let access_key = resolved
                    .as_ref()
                    .map_or(access_key.as_str(), |c| c.access_key_id());
                let secret_key = resolved
                    .as_ref()
                    .map_or(secret_key.as_str(), |c| c.secret_access_key());
                let session_token = resolved
                    .as_ref()
                    .map_or(session_token.as_deref(), |c| c.session_token());
                if [access_key, secret_key, region, service, host]
                    .iter()
                    .any(|v| v.is_empty())
                    || !path.starts_with('/')
                    || !query.is_empty()
                {
                    return Err(PublicError::internal());
                }
                let date = now.format("%Y%m%d").to_string();
                let stamp = now.format("%Y%m%dT%H%M%SZ").to_string();
                headers.insert("accept", accept.clone());
                headers.insert("host", host.clone());
                headers.insert("x-amz-date", stamp.clone());
                if let Some(token) = session_token {
                    headers.insert("x-amz-security-token", token.to_owned());
                }
                let signed = headers.keys().copied().collect::<Vec<_>>().join(";");
                let canonical_headers: String = headers
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{k}:{}\n",
                            v.split_whitespace().collect::<Vec<_>>().join(" ")
                        )
                    })
                    .collect();
                let canonical = format!(
                    "POST\n{path}\n{query}\n{canonical_headers}\n{signed}\n{}",
                    sha(body.as_bytes())
                );
                let scope = format!("{date}/{region}/{service}/aws4_request");
                let to_sign = format!(
                    "AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{}",
                    sha(canonical.as_bytes())
                );
                let mut key = sign(format!("AWS4{secret_key}").as_bytes(), &date);
                for part in [region.as_str(), service.as_str(), "aws4_request"] {
                    key = sign(&key, part);
                }
                let signature: String = sign(&key, &to_sign)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                headers.insert("authorization", format!("AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, Signature={signature}"));
                // The transport sets Host from the same bound URL, as botocore does.
                headers.remove("host");
            }
        }
        if headers
            .values()
            .any(|v| v.bytes().any(|b| b == b'\r' || b == b'\n'))
        {
            return Err(PublicError::internal());
        }
        Ok(json!({"headers": headers}))
    }
}
