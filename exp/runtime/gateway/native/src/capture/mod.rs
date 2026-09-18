//! Shared content capture, independent of destination and hosted tenancy policy.

pub(crate) mod collector;
pub(crate) mod delivery;
mod local;
mod local_store;
mod projection;
pub(crate) mod python;
pub(crate) mod record;
pub(crate) mod response;
