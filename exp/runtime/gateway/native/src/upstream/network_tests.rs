//! Regression evidence for literal, DNS, rebinding, and proxy destination checks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::*;

#[test]
fn only_public_unicast_addresses_pass() {
    for value in [
        "8.8.8.8",
        "1.1.1.1",
        "100.63.255.255",
        "100.128.0.0",
        "172.15.255.255",
        "172.32.0.0",
        "2001:4860:4860::8888",
        "2606:4700:4700::1111",
    ] {
        assert!(public_address(value.parse().unwrap()), "{value}");
    }
    for value in [
        "0.0.0.0",
        "0.1.2.3",
        "10.0.0.1",
        "100.64.0.1",
        "100.127.255.255",
        "127.0.0.1",
        "169.254.169.254",
        "169.254.170.2",
        "172.16.0.1",
        "172.31.255.255",
        "192.0.0.9",
        "192.0.2.1",
        "192.88.99.1",
        "192.168.0.1",
        "198.18.0.1",
        "198.19.255.255",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "239.255.255.255",
        "240.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:8.8.8.8",
        "::ffff:127.0.0.1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b:1::1",
        "100::1",
        "2001::1",
        "2001:20::1",
        "2001:db8::1",
        "2002:a9fe:a9fe::1",
        "3fff::1",
        "3fff:fff::1",
        "fc00::1",
        "fd00:ec2::254",
        "fe80::1",
        "ff02::1",
    ] {
        assert!(!public_address(value.parse().unwrap()), "{value}");
    }
}

#[test]
fn canonical_url_gate_covers_literal_parser_bypasses() {
    let client = crate::upstream::build_client(Duration::from_secs(1), true).unwrap();
    for url in [
        "http://public.example/v1",
        "https://user:password@public.example/v1",
        "https://public.example:0/v1",
        "https://public.example/v1#fragment",
        "https://127.0.0.1/v1",
        "https://127.1/v1",
        "https://2130706433/v1",
        "https://0177.0.0.1/v1",
        "https://0x7f000001/v1",
        "https://%31%32%37.0.0.1/v1",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/v1",
        "https://[::ffff:169.254.169.254]/v1",
        "https://[64:ff9b::a9fe:a9fe]/v1",
        "https://public.example\\@127.0.0.1/v1",
        "not a URL",
    ] {
        let failure = client.post(url).expect_err(url);
        assert!(!failure.retryable_same_deployment);
        assert!(failure.failover_eligible);
    }
    for url in [
        "https://provider.example/v1",
        "https://8.8.8.8/v1",
        "https://[2001:4860:4860::8888]:443/v1",
        "https://provider.example/generate?key=value",
    ] {
        assert!(client.post(url).is_ok(), "{url}");
    }
    let local = crate::upstream::build_client(Duration::from_secs(1), false).unwrap();
    assert!(local.post("http://127.0.0.1:8080/v1").is_ok());
    assert!(local
        .post("not a URL")
        .unwrap()
        .build()
        .unwrap_err()
        .is_builder());
}

/// A real TLS connection exercises SystemResolver's port-zero answers all the
/// way through reqwest. Opt in because ordinary unit tests need no Internet.
#[tokio::test]
#[ignore = "requires outbound public DNS and HTTPS; run explicitly for egress-policy changes"]
async fn restricted_system_resolver_completes_public_https() {
    let client = crate::upstream::build_client(Duration::from_secs(10), true).unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(20),
        client
            .post("https://example.com:443/")
            .unwrap()
            .body("")
            .send(),
    )
    .await
    .expect("bounded public HTTPS request")
    .expect("public DNS, TCP and verified TLS");
    // The site's method policy may return 405. Any HTTP response proves the
    // public address was dialed at port 443 and the TLS hostname was verified.
    assert_eq!(response.url().host_str(), Some("example.com"));
    assert_eq!(response.url().port_or_known_default(), Some(443));
}

#[derive(Clone)]
struct Answers {
    addresses: Vec<SocketAddr>,
    calls: Arc<AtomicUsize>,
}

impl Answers {
    fn new(values: &[&str]) -> Self {
        Self {
            addresses: values.iter().map(|value| value.parse().unwrap()).collect(),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Resolve for Answers {
    fn resolve(&self, _name: Name) -> Resolving {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let addresses = self.addresses.clone();
        Box::pin(async move { Ok(Box::new(addresses.into_iter()) as Addrs) })
    }
}

#[tokio::test]
async fn every_dns_answer_is_checked_and_the_vetted_socket_set_is_reused() {
    for values in [
        vec![],
        vec!["127.0.0.1:443"],
        vec!["8.8.8.8:443", "[fd00::1]:443"],
        vec!["10.0.0.1:443", "[2001:4860:4860::8888]:443"],
    ] {
        let resolver = PublicResolver(Answers::new(&values));
        assert!(resolver
            .resolve("provider.example".parse().unwrap())
            .await
            .is_err());
    }
    let answers = Answers::new(&["8.8.8.8:443", "[2001:4860:4860::8888]:443"]);
    let resolver = PublicResolver(answers.clone());
    let sockets: Vec<_> = resolver
        .resolve("provider.example".parse().unwrap())
        .await
        .unwrap()
        .collect();
    assert_eq!(sockets, answers.addresses);
    assert_eq!(answers.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn connection_time_rebinding_is_rejected_without_contacting_target_or_proxy() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let answers = Answers::new(&[&address.to_string()]);
    let proxy = reqwest::Proxy::all(format!("http://{address}")).unwrap();
    let builder = crate::upstream::client_builder(Duration::from_secs(1)).proxy(proxy);
    let inner = restrict(builder)
        .dns_resolver(Arc::new(PublicResolver(answers.clone())))
        .build()
        .unwrap();
    let client = UpstreamClient::new(inner, true);
    // Admission could have accepted this domain when it was public. At dial
    // time it now points at loopback. The actual resolver must refuse it.
    let request = client.post("https://provider.example/v1").unwrap();
    let result = tokio::time::timeout(Duration::from_secs(2), request.body("secret").send())
        .await
        .unwrap();
    assert!(result.is_err());
    assert_eq!(answers.calls.load(Ordering::SeqCst), 1);
    assert!(
        tokio::time::timeout(Duration::from_millis(30), listener.accept())
            .await
            .is_err()
    );
}

#[test]
fn serve_configuration_requires_an_explicit_hosted_opt_in() {
    let local: crate::server::ServeConfig =
        serde_json::from_str(r#"{"host":"127.0.0.1","port":8080}"#).unwrap();
    assert!(!local.public_upstreams_only);
    let hosted: crate::server::ServeConfig =
        serde_json::from_str(r#"{"host":"127.0.0.1","port":8080,"public_upstreams_only":true}"#)
            .unwrap();
    assert!(hosted.public_upstreams_only);
}
