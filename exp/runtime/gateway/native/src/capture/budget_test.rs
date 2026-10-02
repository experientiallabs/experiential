use super::*;
use serde_json::json;

#[test]
fn sizes_match_serde_without_an_encoded_buffer() {
    let controls: String = (0..=127).map(char::from).collect();
    for value in [
        json!(null),
        json!(false),
        json!(true),
        json!(1),
        json!(-1),
        json!(1.5),
        json!(1e30),
        json!(u64::MAX),
        json!("雪😀 café "),
        json!(controls),
        json!([]),
        json!({}),
        json!({"nested\0": [1, true, "a\n\"b", {"x":"\\"}]}),
    ] {
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(json_bytes(&value), encoded.len());
        assert_eq!(
            encode(&value, encoded.len()).as_deref(),
            Some(encoded.as_str())
        );
        assert!(encode(&value, encoded.len() - 1).is_none());
    }
}

#[test]
fn string_sizes_match_encoding_for_every_control_and_unicode_boundary() {
    // Exercise dense escaping as well as short tails and large ordinary text.
    let controls: String = (0..=127).map(char::from).collect();
    for text in ["", "a", "\\\"\n\t", "雪😀 café", controls.as_str()] {
        for repeat in [1, 7, 64, 4096] {
            let value = text.repeat(repeat);
            assert_eq!(
                string_bytes(&value),
                serde_json::to_string(&value).unwrap().len()
            );
        }
    }
}

#[test]
fn node_heavy_trees_and_spare_vectors_are_charged() {
    let mut values = Vec::with_capacity(4096);
    values.push(Value::Null);
    let value = Value::Array(values);
    assert!(heap_bytes(&value) >= 4096 * std::mem::size_of::<Value>());
    let nodes = Value::Array(vec![Value::Null; 1000]);
    assert!(heap_bytes(&nodes) > json_bytes(&nodes));
}

#[test]
fn small_ordered_maps_include_minimum_entry_and_index_allocations() {
    let value: Value = serde_json::from_str(r#"{"x":null}"#).unwrap();
    let entry =
        std::mem::size_of::<String>() + std::mem::size_of::<Value>() + std::mem::size_of::<usize>();
    let minimum_allocation = 3 * entry + 4 * std::mem::size_of::<usize>() + 16;
    assert!(heap_bytes(&value) >= minimum_allocation);
}

#[test]
fn final_encoding_capacity_stays_inside_its_payload_reservation() {
    for size in [31, 257, 1025, 8193] {
        let value = json!({"content": "x".repeat(size), "last": "🌍"});
        let limit = json_bytes(&value);
        let encoded = encode(&value, limit).unwrap();
        assert_eq!(encoded.len(), limit);
        assert!(encoded.capacity() <= limit);
        assert_eq!(serde_json::from_str::<Value>(&encoded).unwrap(), value);
    }
}
