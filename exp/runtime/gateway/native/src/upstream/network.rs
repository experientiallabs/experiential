//! Socket destination policy for hosted, customer-configurable provider URLs.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::errors::{Failure, FailureClass};

/// Ports a tenant origin may use: HTTPS plus the usual self-hosted serving
/// ports, the same set the platform admits when a customer saves an endpoint.
/// Holding the socket to it means a hostname re-pointed after admission still
/// cannot reach any other service port.
const TENANT_PORTS: [u16; 4] = [443, 8000, 8080, 8443];

/// Operator-managed lists of the platform's own public infrastructure, which a
/// tenant origin may never reach. Setting a variable (comma-separated) replaces
/// its default, matching the platform's admission-time environment contract.
/// The address list ships EMPTY: this crate is published, so the hosting
/// platform supplies its own addresses through the environment rather than the
/// engine naming them.
const TENANT_DENY_ADDRESSES_ENV: &str = "EXPLABS_TENANT_ENDPOINT_DENY_ADDRESSES";
const TENANT_DENY_DOMAINS_ENV: &str = "EXPLABS_TENANT_ENDPOINT_DENY_DOMAINS";
const DEFAULT_TENANT_DENY_ADDRESSES: [&str; 0] = [];
const DEFAULT_TENANT_DENY_DOMAINS: [&str; 2] = ["xplabs.ai", "experientiallabs.ai"];

/// The connection-time refusal of a tenant origin whose DNS answer names the
/// platform's own infrastructure. It extends the public-address refusal's
/// sentence, so the hosted endpoint-policy hold counts both the same way.
const TENANT_DNS_REFUSAL: &str =
    "provider DNS must resolve only to public addresses outside the platform's own infrastructure";

/// The destinations a tenant origin may never reach, beyond the public-only rule.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TenantPolicy {
    addresses: Vec<IpAddr>,
    domains: Vec<String>,
}

impl TenantPolicy {
    /// Read the operator-managed lists from the worker environment.
    pub(crate) fn from_env() -> Self {
        Self::from_lists(
            std::env::var(TENANT_DENY_ADDRESSES_ENV).ok().as_deref(),
            std::env::var(TENANT_DENY_DOMAINS_ENV).ok().as_deref(),
        )
    }

    /// Build the policy from each list's raw value; `None` keeps its default.
    /// An address entry that does not parse is skipped, never fatal.
    fn from_lists(addresses: Option<&str>, domains: Option<&str>) -> Self {
        Self {
            addresses: listed(addresses, &DEFAULT_TENANT_DENY_ADDRESSES)
                .iter()
                .filter_map(|item| item.parse().ok())
                .collect(),
            domains: listed(domains, &DEFAULT_TENANT_DENY_DOMAINS),
        }
    }

    fn denies_address(&self, address: IpAddr) -> bool {
        self.addresses.contains(&address)
    }

    /// Whether `host` is one of the listed domains or a name beneath one.
    fn denies_host(&self, host: &str) -> bool {
        let name = host.trim_end_matches('.').to_ascii_lowercase();
        self.domains
            .iter()
            .any(|domain| name == *domain || name.ends_with(&format!(".{domain}")))
    }
}

/// One operator list: the variable's trimmed, lowercased comma-separated
/// entries when it is set (an empty value disables the list), else the default.
fn listed(raw: Option<&str>, default: &[&str]) -> Vec<String> {
    match raw {
        None => default.iter().map(|item| (*item).to_string()).collect(),
        Some(raw) => raw
            .split(',')
            .map(|item| item.trim().trim_end_matches('.').to_ascii_lowercase())
            .filter(|item| !item.is_empty())
            .collect(),
    }
}

/// A pooled client whose provider request construction always checks its policy.
/// The inner client stays private so a new route cannot bypass the URL gate.
#[derive(Clone)]
pub(crate) struct UpstreamClient {
    inner: reqwest::Client,
    public_only: bool,
    /// Set on the tenant-origin client: the port and infrastructure rules its
    /// URLs are held to before any request is built.
    policy: Option<Arc<TenantPolicy>>,
    /// The sibling a tenant origin dials through, present beside a hosted
    /// (public-only) client.
    tenant: Option<Box<UpstreamClient>>,
}

impl UpstreamClient {
    pub(super) fn new(inner: reqwest::Client, public_only: bool) -> Self {
        Self {
            inner,
            public_only,
            policy: None,
            tenant: None,
        }
    }

    /// Attach the tenant-origin sibling, built over the tenant resolver.
    pub(super) fn with_tenant(mut self, inner: reqwest::Client, policy: Arc<TenantPolicy>) -> Self {
        self.tenant = Some(Box::new(Self {
            inner,
            public_only: true,
            policy: Some(policy),
            tenant: None,
        }));
        self
    }

    /// The client one rung dials through: its tenant sibling for a
    /// customer-addressed origin on a hosted plane, otherwise this client. A
    /// standalone (non-public) plane has no sibling and keeps its local reach.
    pub(crate) fn for_origin(&self, tenant_origin: bool) -> &UpstreamClient {
        match &self.tenant {
            Some(tenant) if tenant_origin => tenant,
            _ => self,
        }
    }

    /// PATCH counterpart of [`Self::post`], under the same URL validation.
    pub(crate) fn patch(&self, raw: &str) -> Result<reqwest::RequestBuilder, Failure> {
        if !self.public_only {
            return Ok(self.inner.patch(raw));
        }
        if raw
            .bytes()
            .any(|byte| byte <= 32 || byte == 127 || byte == b'\\')
        {
            return Err(refused());
        }
        let url = reqwest::Url::parse(raw).map_err(|_| refused())?;
        validate_url(&url, self.policy.as_deref())?;
        Ok(self.inner.patch(url))
    }

    /// GET counterpart of [`Self::post`], under the same URL validation: the
    /// video content relay downloads provider assets through it.
    pub(crate) fn get(&self, raw: &str) -> Result<reqwest::RequestBuilder, Failure> {
        if !self.public_only {
            return Ok(self.inner.get(raw));
        }
        if raw
            .bytes()
            .any(|byte| byte <= 32 || byte == 127 || byte == b'\\')
        {
            return Err(refused());
        }
        let url = reqwest::Url::parse(raw).map_err(|_| refused())?;
        validate_url(&url, self.policy.as_deref())?;
        Ok(self.inner.get(url))
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
        validate_url(&url, self.policy.as_deref())?;
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

fn tenant_refused() -> Failure {
    Failure::new(
        FailureClass::Transport,
        "provider endpoint is not allowed for a customer endpoint; serve it from your own \
         host on port 443, 8000, 8080 or 8443",
    )
    .with_retry(false, true)
}

fn validate_url(url: &reqwest::Url, policy: Option<&TenantPolicy>) -> Result<(), Failure> {
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
        if policy.is_some_and(|policy| policy.denies_address(address)) {
            return Err(tenant_refused());
        }
    }
    if let Some(policy) = policy {
        let port_allowed = url
            .port_or_known_default()
            .is_some_and(|port| TENANT_PORTS.contains(&port));
        if !port_allowed || policy.denies_host(host) {
            return Err(tenant_refused());
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

/// The tenant-origin counterpart of [`restrict`]: the same public-only socket
/// rule, plus a refusal of any DNS answer naming the platform's own
/// infrastructure, applied to the exact sockets reqwest then dials.
pub(super) fn restrict_tenant(
    builder: reqwest::ClientBuilder,
    policy: Arc<TenantPolicy>,
) -> reqwest::ClientBuilder {
    builder
        .https_only(true)
        .no_proxy()
        .dns_resolver(Arc::new(TenantResolver {
            inner: PublicResolver(SystemResolver),
            policy,
        }))
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

struct TenantResolver<R> {
    inner: PublicResolver<R>,
    policy: Arc<TenantPolicy>,
}

impl<R: Resolve> Resolve for TenantResolver<R> {
    fn resolve(&self, name: Name) -> Resolving {
        let resolving = self.inner.resolve(name);
        let policy = Arc::clone(&self.policy);
        Box::pin(async move {
            let addresses: Vec<SocketAddr> = resolving.await?.collect();
            if addresses
                .iter()
                .any(|address| policy.denies_address(address.ip()))
            {
                return Err(
                    io::Error::new(io::ErrorKind::PermissionDenied, TENANT_DNS_REFUSAL).into(),
                );
            }
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
