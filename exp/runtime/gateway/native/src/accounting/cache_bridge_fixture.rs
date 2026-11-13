//! Native request ownership for the loopback cache creation and reuse proof.

use super::{dispatch::Dispatch, dispatch_tests, health::Policy, ownership::Owner, shared::Shared};
use crate::{
    errors::PublicError,
    native_callbacks::{CallbackTarget, NativeTarget},
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub(crate) struct Host {
    authority: Value,
    calls: Mutex<Vec<Value>>,
    recorded: Mutex<Option<Value>>,
}

impl CallbackTarget for Host {
    fn handles(&self, method: &str) -> bool {
        matches!(
            method,
            "start_attempt"
                | "settle"
                | "price"
                | "abandon"
                | "record_late_meter"
                | "cache_transaction"
        )
    }

    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        let input: Value = crate::lossless_json::from_str(argument).unwrap();
        let result = if method == "cache_transaction" {
            self.calls.lock().unwrap().push(input.clone());
            match input["action"].as_str().unwrap() {
                "authority" => self.authority.clone(),
                "claim" => match self.recorded.lock().unwrap().as_ref() {
                    Some(result) => {
                        json!({"state":"ready", "resource_name":result["resource_name"],
                        "expires_at":result["expire_time"], "token_count":result["total_tokens"]})
                    }
                    None => {
                        json!({"state":"creator", "operation_id":input["offer"]["operation_id"],
                        "expires_at":input["offer"]["expires_at"]})
                    }
                },
                "record" => {
                    let mut result = self.recorded.lock().unwrap();
                    assert!(result.is_none(), "creation is charged only once");
                    *result = Some(input["result"].clone());
                    Value::Null
                }
                action => panic!("unexpected cache operation {action}"),
            }
        } else {
            match method {
                "start_attempt" => {
                    json!({"attempt_id": format!("attempt-{}", input["request_id"].as_str().unwrap()), "promo_funded":false})
                }
                "abandon" => json!({"terminalized":true,"known_unbilled":false}),
                "price" | "settle" | "record_late_meter" => json!({}),
                _ => return Err(PublicError::internal()),
            }
        };
        Ok(result.to_string())
    }
}

pub(crate) struct Fixture {
    owner: Arc<Owner>,
}

impl Fixture {
    pub fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        self.owner
            .call(method, crate::lossless_json::from_str(argument).unwrap())
            .map(|v| v.to_string())
    }
}

pub(crate) fn host(authority: Value) -> Arc<Host> {
    Arc::new(Host {
        authority,
        calls: Mutex::default(),
        recorded: Mutex::default(),
    })
}

pub(crate) fn admit(configuration: Value, host: Arc<Host>, request: &str) -> Arc<Fixture> {
    let mut config = dispatch_tests::config();
    config.request_id = request.into();
    config.lanes.truncate(1);
    config.rules.truncate(1);
    config.redial_budgets.truncate(1);
    config.lanes[0].key.1 = configuration["bindings"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .into();
    config.features.cache = Some(crate::lossless_json::from_value(configuration).unwrap());
    let shared = Arc::new(Shared::new(Policy::default(), None, 60.0, 100).unwrap());
    let driver = Dispatch::new(config, shared, NativeTarget::new(host)).unwrap();
    let owner = Owner::new(
        request.into(),
        "tenant".into(),
        Instant::now() + Duration::from_secs(60),
        driver,
    );
    owner
        .call(
            "start_attempt",
            json!({"request_id":request, "attempt_ordinal":0}),
        )
        .unwrap();
    Arc::new(Fixture { owner })
}

pub(crate) fn assert_accounted(host: &Host, claims: usize, resource: &str) {
    let calls = host.calls.lock().unwrap();
    assert_eq!(
        calls.iter().filter(|c| c["action"] == "claim").count(),
        claims
    );
    assert_eq!(
        calls.iter().filter(|c| c["action"] == "authority").count(),
        claims
    );
    assert_eq!(calls.iter().filter(|c| c["action"] == "record").count(), 1);
    let offer = &calls.iter().find(|c| c["action"] == "claim").unwrap()["offer"];
    let record = host.recorded.lock().unwrap();
    let result = record.as_ref().unwrap();
    assert_eq!(offer["operation_id"], result["operation_id"]);
    assert_eq!(result["outcome"], "ready");
    assert_eq!(result["resource_name"], resource);
    assert_eq!(result["total_tokens"], 1536);
    assert_eq!(result["expire_time"], offer["expires_at"]);
    assert!(offer["reservation_nano_usd"].as_u64().unwrap() > 0);
    assert!(result["create_time"].as_f64().unwrap() <= result["observed_at"].as_f64().unwrap());
    assert!(
        offer["requested_at"].as_f64().unwrap() - 2.0 <= result["create_time"].as_f64().unwrap()
    );
}
