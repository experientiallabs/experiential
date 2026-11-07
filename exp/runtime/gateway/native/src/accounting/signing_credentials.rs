//! Native, single-flight refresh for ambient AWS credential providers.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_credential_types::{provider::ProvideCredentials, Credentials};
use tokio::sync::{Mutex as AsyncMutex, OnceCell};

use crate::errors::PublicError;

type Key = (String, Option<String>);
type Providers = Mutex<VecDeque<(Key, Arc<Provider>)>>;
#[derive(Default)]
struct Provider {
    chain: OnceCell<DefaultCredentialsChain>,
    credentials: AsyncMutex<Option<Credentials>>,
}
impl Provider {
    async fn get(&self, key: &Key) -> Result<Credentials, PublicError> {
        let mut credentials = self.credentials.lock().await;
        if let Some(current) = credentials.as_ref().filter(|c| fresh(c, SystemTime::now())) {
            return Ok(current.clone());
        }
        let chain = self
            .chain
            .get_or_init(|| async {
                let mut builder = DefaultCredentialsChain::builder()
                    .region(aws_types::region::Region::new(key.0.clone()));
                if let Some(profile) = &key.1 {
                    builder = builder.profile_name(profile);
                }
                builder.build().await
            })
            .await;
        // A refresh failure never falls back to an expired credential.
        let next = chain
            .provide_credentials()
            .await
            .map_err(|_| PublicError::internal())?;
        if next
            .expiry()
            .is_some_and(|expiry| expiry <= SystemTime::now())
        {
            return Err(PublicError::internal());
        }
        *credentials = Some(next.clone());
        Ok(next)
    }
}
fn fresh(credentials: &Credentials, now: SystemTime) -> bool {
    credentials.expiry().is_none_or(|expiry| {
        expiry
            .duration_since(now)
            .is_ok_and(|left| left > Duration::from_secs(60))
    })
}

pub(super) fn resolve(region: &str, profile: Option<&str>) -> Result<Credentials, PublicError> {
    static PROVIDERS: OnceLock<Providers> = OnceLock::new();
    let key = (region.to_owned(), profile.map(str::to_owned));
    let provider = {
        let mut providers = PROVIDERS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| PublicError::internal())?;
        if let Some(index) = providers.iter().position(|(k, _)| k == &key) {
            let (_, provider) = providers.remove(index).ok_or_else(PublicError::internal)?;
            providers.push_back((key.clone(), provider.clone()));
            provider
        } else {
            let provider = Arc::new(Provider::default());
            if providers.len() == 64 {
                providers.pop_front();
            }
            providers.push_back((key.clone(), provider.clone()));
            provider
        }
    };
    super::inspection_http::runtime()
        .map_err(|_| PublicError::internal())?
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(10), provider.get(&key))
                .await
                .map_err(|_| PublicError::internal())?
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporary_credentials_refresh_before_expiration_without_reading_ambient_credentials() {
        let now = SystemTime::now();
        for (seconds, expected) in [(0, false), (30, false), (60, false), (61, true)] {
            let credential = Credentials::new(
                "fixture",
                "fixture",
                None,
                Some(now + Duration::from_secs(seconds)),
                "fixture",
            );
            assert_eq!(fresh(&credential, now), expected);
        }
        assert!(fresh(
            &Credentials::new("fixture", "fixture", None, None, "fixture"),
            now
        ));
    }
}
