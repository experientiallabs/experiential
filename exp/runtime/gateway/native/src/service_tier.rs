//! Bounded processing-tier evidence, independent of requested tier and prices.

use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ServiceTierObservation {
    served: Option<&'static str>,
    resolution: Resolution,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Resolution {
    #[default]
    Missing,
    Confirmed,
    Unknown,
    Conflicting,
}

impl ServiceTierObservation {
    /// Missing fields say nothing; malformed or contradictory evidence stays unresolved.
    pub(crate) fn observe(&mut self, value: Option<&Value>) {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            return;
        };
        let served = match value.as_str() {
            Some("default" | "standard") => Some("default"),
            Some("priority" | "fast") => Some("priority"),
            Some("flex") => Some("flex"),
            _ => None,
        };
        if self.resolution == Resolution::Conflicting || self.resolution == Resolution::Unknown {
            return;
        }
        if served.is_none() {
            self.served = None;
            self.resolution = Resolution::Unknown;
        } else if self.served.is_some() && self.served != served {
            self.served = None;
            self.resolution = Resolution::Conflicting;
        } else {
            self.served = served;
            self.resolution = Resolution::Confirmed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn absence_is_not_standard_and_conflicts_never_pick_a_card() {
        let mut tier = ServiceTierObservation::default();
        tier.observe(None);
        tier.observe(Some(&Value::Null));
        assert_eq!(tier.resolution, Resolution::Missing);
        tier.observe(Some(&json!("priority")));
        tier.observe(None);
        assert_eq!(tier.served, Some("priority"));
        tier.observe(Some(&json!("default")));
        tier.observe(Some(&json!("priority")));
        assert_eq!(tier.resolution, Resolution::Conflicting);
        assert_eq!(tier.served, None);
    }

    #[test]
    fn unknown_and_malformed_labels_stay_unknown() {
        for value in [json!("auto"), json!("unpriced"), json!(42)] {
            let mut tier = ServiceTierObservation::default();
            tier.observe(Some(&value));
            tier.observe(Some(&json!("default")));
            assert_eq!(tier.resolution, Resolution::Unknown);
            assert_eq!(tier.served, None);
        }
    }
}
