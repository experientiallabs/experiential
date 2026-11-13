//! Process-owned state shared by admission and native request callbacks.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::errors::PublicError;

use super::health::{Health, Policy};
use super::loads::Loads;
use super::sticky::Sticky;

pub(super) struct Policies {
    pub health: Health,
    pub loads: Loads,
    pub sticky: Sticky,
}

pub(super) struct Shared {
    epoch: Instant,
    pub counters: super::counters::Counters,
    pub policies: Mutex<Policies>,
    pub recovery: Arc<Mutex<super::session_recovery::Registry>>,
}

impl Shared {
    pub fn new(
        policy: Policy,
        default_bound: Option<u64>,
        activity_window: f64,
        sticky_maximum: usize,
    ) -> Result<Self, PublicError> {
        Ok(Self {
            epoch: Instant::now(),
            counters: super::counters::Counters::default(),
            recovery: Arc::new(Mutex::new(
                super::session_recovery::Registry::new(65536, 3600.0, 60.0, 5.0)
                    .ok_or_else(PublicError::internal)?,
            )),
            policies: Mutex::new(Policies {
                health: Health::new(policy).ok_or_else(PublicError::internal)?,
                loads: Loads::new(default_bound, activity_window)
                    .ok_or_else(PublicError::internal)?,
                sticky: Sticky::new(sticky_maximum).ok_or_else(PublicError::internal)?,
            }),
        })
    }
    pub fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }
}
