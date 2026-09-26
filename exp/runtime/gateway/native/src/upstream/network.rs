//! Socket destination policy for hosted, customer-configurable provider URLs.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::errors::{Failure, FailureClass};

/// A pooled client whose provider request construction always checks its policy.
/// The inner client stays private so a new route cannot bypass the URL gate.
#[derive(Clone)]
pub(crate) struct UpstreamClient {
    inner: reqwest::Client,
    public_only: bool,
}

impl UpstreamClient {
    pub(super) fn new(inner: reqwest::Client, public_only: bool) -> Self {
        Self { inner, public_only }
    }

    /// Validate the canonical URL before attaching credentials or building a request.
    pub(crate) fn post(&self, raw: &str) -> Result<reqwest::RequestBuilder, Failure> {
        // Preserve reqwest's builder errors and the existing transport detail
        // for standalone callers, which may legitimately use local HTTP.
        if !self.public_only {
            return Ok(self.inner.post(raw));
        }
        if raw
            .bytes()
            .any(|byte| byte <= 32 || byte == 127 || byte == b'\\')
        {
            return Err(refused());
        }
        let url = reqwest::Url::parse(raw).map_err(|_| refused())?;
        validate_url(&url)?;
        Ok(self.inner.post(url))
    }
}

fn refused() -> Failure {
    Failure::new(
        FailureClass::Transport,
        "provider endpoint is not allowed; configure a public HTTPS endpoint",
    )
    .with_retry(false, true)
}

fn validate_url(url: &reqwest::Url) -> Result<(), Failure> {
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.port() == Some(0)
    {
        return Err(refused());
    }
    let host = url.host_str().ok_or_else(refused)?;
    // reqwest's WHATWG parser canonicalizes integer/octal/hex IPv4 spellings.
    // Literal addresses bypass reqwest's DNS resolver, so check them here too.
    if let Ok(address) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        if !public_address(address) {
            return Err(refused());
        }
    }
    Ok(())
}

/// Install the same restriction at connection time. No ambient proxy may move
/// DNS resolution outside this resolver, and redirects remain disabled by the
/// shared client builder. Pooled sockets retain their already-checked peer.
pub(super) fn restrict(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .https_only(true)
        .no_proxy()
        .dns_resolver(Arc::new(PublicResolver(SystemResolver)))
}

struct SystemResolver;

impl Resolve for SystemResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((name.as_str(), 0))
                .await?
                .collect::<Vec<_>>();
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

struct PublicResolver<R>(R);

impl<R: Resolve> Resolve for PublicResolver<R> {
    fn resolve(&self, name: Name) -> Resolving {
        let resolving = self.0.resolve(name);
        Box::pin(async move {
            let addresses: Vec<SocketAddr> = resolving.await?.collect();
            if addresses.is_empty()
                || addresses
                    .iter()
                    .any(|address| !public_address(address.ip()))
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "provider DNS must resolve only to public addresses",
                )
                .into());
            }
            // Hand these exact vetted sockets to reqwest. A preflight lookup
            // followed by a separate connection lookup would permit rebinding.
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

/// Conservative global-unicast policy from the IANA special-purpose registries.
/// IPv6 transition mechanisms are excluded so embedded private IPv4 destinations
/// cannot cross the boundary via mapped, translated, NAT64, Teredo or 6to4 forms.
fn public_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => public_v4(address),
        IpAddr::V6(address) => public_v6(address),
    }
}

fn public_v4(address: Ipv4Addr) -> bool {
    let value = u32::from(address);
    let blocked = [
        ([0, 0, 0, 0], 8),
        ([10, 0, 0, 0], 8),
        ([100, 64, 0, 0], 10),
        ([127, 0, 0, 0], 8),
        ([169, 254, 0, 0], 16),
        ([172, 16, 0, 0], 12),
        ([192, 0, 0, 0], 24),
        ([192, 0, 2, 0], 24),
        ([192, 88, 99, 0], 24),
        ([192, 168, 0, 0], 16),
        ([198, 18, 0, 0], 15),
        ([198, 51, 100, 0], 24),
        ([203, 0, 113, 0], 24),
        ([224, 0, 0, 0], 4),
        ([240, 0, 0, 0], 4),
    ];
    !blocked.iter().any(|(network, bits)| {
        let mask = u32::MAX << (32 - bits);
        value & mask == u32::from_be_bytes(*network) & mask
    })
}

fn public_v6(address: Ipv6Addr) -> bool {
    let value = u128::from(address);
    let in_prefix = |network: u128, bits: u32| {
        let mask = u128::MAX << (128 - bits);
        value & mask == network & mask
    };
    in_prefix(0x2000_u128 << 112, 3)
        && !in_prefix(0x2001_u128 << 112, 23)
        && !in_prefix(0x2001_0db8_u128 << 96, 32)
        && !in_prefix(0x2002_u128 << 112, 16)
        && !in_prefix(0x3fff_u128 << 112, 20)
}

#[cfg(test)]
#[path = "network_tests.rs"]
mod tests;
