//! What a served decision answer carries beyond the validated answers: the
//! gateway's routing headers and the decoder's disclosed request rewrites.

use serde_json::{json, Value};

use super::DecisionsAdmission;

/// Attach the decoder's rewrites as `x-experiential-ignored-parameters`, the
/// body key every serving surface discloses through; none adds nothing.
pub(super) fn disclose(mut body: Value, ignored_parameters: &[String]) -> Value {
    if ignored_parameters.is_empty() {
        return body;
    }
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "x-experiential-ignored-parameters".to_string(),
            json!(ignored_parameters),
        );
    }
    body
}

/// The gateway's routing headers on an answer served at `depth`.
pub(super) fn served_headers(
    admission: &DecisionsAdmission,
    depth: usize,
    client_request_id: Option<String>,
) -> Vec<(String, String)> {
    let wire = &admission.route[depth];
    let mut headers = vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
        (
            "x-gateway-canonical-model".to_string(),
            admission.exact_model_id.clone(),
        ),
        ("x-gateway-provider".to_string(), wire.provider.clone()),
        (
            "x-gateway-deployment".to_string(),
            wire.deployment_id.clone(),
        ),
        ("x-gateway-route-depth".to_string(), depth.to_string()),
        (
            "x-gateway-route-reason".to_string(),
            admission.route_reason.clone(),
        ),
    ];
    if let Some(value) = client_request_id {
        headers.push(("x-client-request-id".to_string(), value));
    }
    headers
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::disclose;

    #[test]
    fn disclosure_is_added_only_when_the_decoder_rewrote_something() {
        let body = json!({"id": "decision-1", "answers": {}});
        assert_eq!(disclose(body.clone(), &[]), body);
        let disclosed = disclose(body, &["state->replaced(lone_surrogate)".to_string()]);
        assert_eq!(
            disclosed["x-experiential-ignored-parameters"],
            json!(["state->replaced(lone_surrogate)"])
        );
        assert_eq!(disclosed["id"], "decision-1");
    }
}
