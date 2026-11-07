//! Bounded, cookie-free classifier HTTP owned by the native request path.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::Value;
use tokio::{runtime::Runtime, sync::Semaphore};

struct Transport {
    runtime: Runtime,
    public: reqwest::Client,
    configured: reqwest::Client,
    capacity: Semaphore,
}

fn transport() -> Result<&'static Transport, ()> {
    static TRANSPORT: OnceLock<Result<Transport, ()>> = OnceLock::new();
    TRANSPORT
        .get_or_init(|| {
            Ok(Transport {
                runtime: tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .max_blocking_threads(32)
                    .thread_name("gateway-inspection")
                    .enable_all()
                    .build()
                    .map_err(|_| ())?,
                public: crate::upstream::build_classifier_client(true)?,
                configured: crate::upstream::build_classifier_client(false)?,
                capacity: Semaphore::new(32),
            })
        })
        .as_ref()
        .map_err(|_| ())
}

pub(super) fn runtime() -> Result<&'static Runtime, ()> {
    Ok(&transport()?.runtime)
}

pub(super) fn inspect(adapter: &Value, payload: Value, budget: Duration) -> Result<Value, ()> {
    runtime()?.block_on(inspect_async(adapter, payload, budget))
}

pub(super) async fn inspect_async(
    adapter: &Value,
    mut payload: Value,
    budget: Duration,
) -> Result<Value, ()> {
    if budget.is_zero() || adapter["transport"]["unavailable"] == true {
        return Err(());
    }
    let t = transport()?;
    let public_only = adapter["transport"]["public_only"].as_bool().ok_or(())?;
    let url = crate::upstream::classifier_url(adapter["url"].as_str().ok_or(())?, public_only)?;
    let limit = adapter["max_response_bytes"]
        .as_u64()
        .filter(|n| (1..=1_048_576).contains(n))
        .ok_or(())? as usize;
    if let Some(criteria) = adapter["transport"]
        .get("criteria")
        .filter(|v| !v.is_null())
    {
        payload["criteria"] = criteria.clone();
    }
    let client = if public_only {
        &t.public
    } else {
        &t.configured
    };
    tokio::time::timeout(budget, async {
        let _permit = t.capacity.acquire().await.map_err(|_| ())?;
        let mut request = client
            .post(url)
            .header("accept", "application/json")
            .header("content-type", "application/json");
        if let Some(authorization) = adapter["authorization"].as_str() {
            request = request.header("authorization", authorization);
        }
        if let Some(headers) = adapter["transport"]["headers"].as_object() {
            for (name, value) in headers {
                request = request.header(name, value.as_str().ok_or(())?);
            }
        }
        let mut response = request
            .body(super::canonical::bytes(&payload, false).map_err(|_| ())?)
            .send()
            .await
            .map_err(|_| ())?;
        if (if adapter["kind"] == "decision_model" {
            response.status().as_u16() != 200
        } else {
            !response.status().is_success()
        }) || response.content_length().is_some_and(|n| n > limit as u64)
        {
            return Err(());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
            if bytes.len().saturating_add(chunk.len()) > limit {
                return Err(());
            }
            bytes.extend_from_slice(&chunk);
        }
        if adapter["kind"] == "decision_model" {
            crate::route_decisions::strict_json(&bytes).map_err(|_| ())
        } else {
            crate::lossless_json::from_slice(&bytes).map_err(|_| ())
        }
    })
    .await
    .map_err(|_| ())?
}
