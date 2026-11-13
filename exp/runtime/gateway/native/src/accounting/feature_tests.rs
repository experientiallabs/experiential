//! Wire compatibility with the previous signing and authenticated carrier owners.

use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use ring::aead;
use serde_json::{json, Value};

use super::{carriers::Authority, signing::Binding};

#[test]
fn responses_retention_preserves_the_python_message_contract_and_byte_bounds() {
    for (index, line) in include_str!("../../tests/generated/continuations.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let context: super::continuations::Context =
            crate::lossless_json::from_value(case["context"].clone()).unwrap();
        let result = context
            .remember("request", Some(0), &case["input"])
            .unwrap()
            .unwrap();
        assert_eq!(result["state"], case["expected"], "case {index}");
        if case["has_native_items"] != true {
            assert_eq!(result["size_bytes"], case["size_bytes"], "case {index}");
        } else {
            assert!(result["size_bytes"].as_u64() >= case["size_bytes"].as_u64());
        }
        let mut refused = case["input"].clone();
        refused["refusal"] = json!(true);
        assert!(context
            .remember("request", Some(0), &refused)
            .unwrap()
            .is_none());
    }
}

#[test]
fn dispatch_signatures_match_botocore_and_reject_changed_destinations_and_bodies() {
    for (index, line) in include_str!("../../tests/generated/signing.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let binding: Binding = crate::lossless_json::from_value(case["binding"].clone()).unwrap();
        let now = case["now"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            binding.headers(&case["input"], now).unwrap(),
            case["expected"],
            "case {index}"
        );
        let mut changed = case["input"].clone();
        changed["body"] = json!("{}");
        assert!(binding.headers(&changed, now).is_err());
        changed = case["input"].clone();
        changed["url"] = json!("https://elsewhere.test");
        assert!(binding.headers(&changed, now).is_err());
    }
}

fn decrypt(carrier: &str, key: &str) -> Value {
    let (public, encoded) = carrier.rsplit_once(':').unwrap();
    let mut envelope = URL_SAFE_NO_PAD.decode(encoded).unwrap();
    let nonce = aead::Nonce::try_assume_unique_for_key(&envelope[..12]).unwrap();
    let key = aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &STANDARD.decode(key).unwrap()).unwrap(),
    );
    let plain = key
        .open_in_place(
            nonce,
            aead::Aad::from(public.as_bytes()),
            &mut envelope[12..],
        )
        .unwrap();
    crate::lossless_json::from_slice(plain).unwrap()
}

#[test]
fn native_reasoning_carriers_preserve_python_authenticated_claims_and_turn_hashes() {
    for (index, line) in include_str!("../../tests/generated/carriers.jsonl")
        .lines()
        .enumerate()
    {
        let case: Value = crate::lossless_json::from_str(line).unwrap();
        let authority: Authority =
            crate::lossless_json::from_value(case["authority"].clone()).unwrap();
        let key = case["authority"]["key"].as_str().unwrap();
        assert_eq!(
            decrypt(case["python_carrier"].as_str().unwrap(), key),
            case["expected"]
        );
        let sealed = authority.seal("request", 0, &case["input"]).unwrap();
        assert_eq!(
            decrypt(sealed["carrier"].as_str().unwrap(), key),
            case["expected"],
            "case {index}"
        );
        assert_ne!(
            sealed,
            authority.seal("request", 0, &case["input"]).unwrap()
        );
        assert!(authority.seal("request", 1, &case["input"]).is_err());
        let mut changed = case["input"].clone();
        changed["route_sha256"] = json!("wrong");
        assert!(authority.seal("request", 0, &changed).is_err());
        changed = case["input"].clone();
        changed["tool_calls"] = json!([changed["tool_calls"][0], changed["tool_calls"][0]]);
        assert!(authority.seal("request", 0, &changed).is_err());
    }
}
