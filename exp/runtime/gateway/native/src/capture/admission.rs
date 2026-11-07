//! Register an admitted capture through the same Rust-only validation boundary.
use super::{collector::Collector, record::Request};
use serde::Deserialize;
use serde_json::{value::RawValue, Value};
use std::sync::Arc;

#[derive(Deserialize)]
struct ContextSource<'a> {
    #[serde(borrow)]
    context: &'a RawValue,
}

pub(crate) fn begin(collector: &Collector, request_json: &[u8]) -> bool {
    if !collector.config.truncate_request
        && request_json.len() > collector.config.maximum_request_bytes
    {
        return collector.skip();
    }
    let Ok(mut request) = crate::lossless_json::from_slice::<Request>(request_json) else {
        return collector.skip();
    };
    // Keep the original source for the capture consumer's numeric projection.
    if request
        .context
        .get("source_json")
        .is_none_or(Value::is_null)
        && super::response::contains_wide_number(&request.context)
    {
        let Ok(source) = serde_json::from_slice::<ContextSource>(request_json) else {
            return collector.skip();
        };
        let Some(context) = Arc::make_mut(&mut request.context).as_object_mut() else {
            return collector.skip();
        };
        context.insert(
            "source_json".into(),
            Value::String(source.context.get().to_owned()),
        );
    }
    collector.begin(request)
}
