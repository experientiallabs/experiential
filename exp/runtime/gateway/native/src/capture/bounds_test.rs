use super::*;

fn encoded_size(value: &Value) -> usize {
    serde_json::to_string(value).unwrap().len()
}

fn trim(value: &mut Value, maximum: usize) -> usize {
    let original = json_bytes(value);
    let (count, remaining) = trim_strings(value, maximum, original);
    assert_eq!(remaining, encoded_size(value));
    count
}

/// Independent encoded-size oracle for the specified prefix/marker policy.
fn reference_trim(value: &mut Value, maximum: usize) -> usize {
    if encoded_size(value) <= maximum {
        return 0;
    }
    let mut leaves = Vec::new();
    string_leaves(value, "", &mut leaves);
    leaves.sort_by_key(|leaf| std::cmp::Reverse(leaf.0));
    let mut count = 0;
    for (_, path) in leaves {
        let excess = encoded_size(value).saturating_sub(maximum);
        if excess == 0 {
            break;
        }
        let Some(Value::String(text)) = value.pointer_mut(&path) else {
            continue;
        };
        let mut end = text
            .len()
            .saturating_sub(excess + MARKER_ALLOWANCE)
            .max(HEAD_BYTES);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let removed = text.len() - end;
        text.truncate(end);
        text.push_str(&format!(" [truncated for capture: {removed} bytes]"));
        count += 1;
    }
    count
}

#[test]
fn incremental_sizes_preserve_exact_truncation_policy() {
    for unit in ["a", "雪😀", "\n\t\\\"", "\u{0001}"] {
        for count in [1, 8, 32] {
            let messages: Vec<_> = (0..count)
                .map(|index| json!({"role":"tool", "content":unit.repeat(8192 + index)}))
                .collect();
            let input = json!({"messages": messages, "a/b~c": {"key": unit.repeat(5000)}});
            for maximum in [0, HEAD_BYTES, 24000, encoded_size(&input)] {
                let mut actual = input.clone();
                let mut expected = input.clone();
                assert_eq!(
                    trim(&mut actual, maximum),
                    reference_trim(&mut expected, maximum)
                );
                assert_eq!(actual, expected);
                assert_eq!(json_bytes(&actual), encoded_size(&actual));
            }
        }
    }
}

#[test]
fn near_prefix_markers_can_increase_size_without_invalidating_accounting() {
    let input = json!(["x".repeat(HEAD_BYTES + 1), "y".repeat(HEAD_BYTES + 2)]);
    for maximum in [encoded_size(&input) - 1, 0] {
        let mut actual = input.clone();
        let mut expected = input.clone();
        assert_eq!(trim(&mut actual, maximum), 2);
        assert_eq!(reference_trim(&mut expected, maximum), 2);
        assert_eq!(actual, expected);
        assert!(encoded_size(&actual) > encoded_size(&input));
    }
}

#[test]
fn oversized_prompt_keeps_the_largest_prefix_that_fits_not_only_four_kib() {
    for unit in ["x", "雪😀"] {
        let text = unit.repeat(5 * 1024 * 1024 / unit.len());
        let mut value = json!([{"role": "user", "content": text}]);
        assert_eq!(trim(&mut value, COLUMN_BUDGET), 1);
        let retained = value[0]["content"].as_str().unwrap();
        assert!(retained.len() > COLUMN_BUDGET - 128);
        assert!(retained.contains("[truncated for capture:"));
        assert!(encoded_size(&value) <= COLUMN_BUDGET);
        assert_eq!(value[0]["role"], "user");
    }
}

#[test]
fn equally_sized_messages_have_deterministic_minimum_loss_ordering() {
    let mut value = json!(["a".repeat(10000), "b".repeat(10000), "c".repeat(10000)]);
    assert_eq!(trim(&mut value, 29500), 1);
    assert!(value[0].as_str().unwrap().starts_with(&"a".repeat(9400)));
    assert_eq!(value[1], "b".repeat(10000));
    assert_eq!(value[2], "c".repeat(10000));
    assert!(encoded_size(&value) <= 29500);
}
