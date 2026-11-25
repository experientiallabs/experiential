# Provider destination policy

`serve_native_gateway(..., public_upstreams_only=True)` restricts the shared
native provider transport to public HTTPS destinations. Hosted embedders that
accept customer endpoints must enable it. The default permits local/private
servers for the standalone gateway.

The restricted client validates canonical IP literals before constructing a
request and checks every A/AAAA address returned by its connection resolver.
Any non-public answer rejects the entire resolution. Reqwest connects using
that same checked address set, preserving the original hostname for TLS and
HTTP. Existing pooled sockets retain their checked peer. Environment proxies
are disabled and redirects remain disabled. IPv6 transition address ranges are
excluded along with private, loopback, link-local, reserved, documentation, and
multicast ranges.

The policy covers native generation, embedding, image, decision, and Google
cache creation requests, and the video content download.

The asynchronous video job submit and poll calls run in Python, so they hold
the same policy there (`exp/runtime/gateway/public_egress.py`). The setting is
one process-wide flag that can only be turned on:
`serve_native_gateway(public_upstreams_only=True)` turns it on, and a hosted
embedder should call `require_public_upstreams()` before constructing its
control plane, because the video poller starts at construction. It is never
turned back off, since pollers run until `close()` and serves may overlap, so
every control plane in the process (execution-bound ones included) keeps the
policy until the process exits. A process that never turns it on keeps the
standalone default that permits local and private servers. With
the policy on, that client sends only `https` URLs without credentials,
resolves the host once, rejects the whole resolution if any answer is outside
the ranges above, and connects to a checked address while TLS and the `Host`
header keep the original hostname. Address literals are checked without
resolution. Environment proxies are ignored and redirects stay disabled on the
video client whether or not the policy is on; `SSL_CERT_FILE` /
`SSL_CERT_DIR` still select the trust store for a private CA.

The policy does not cover other separate HTTP clients in Python control-plane
callbacks, custom tools, batch jobs, guardrail endpoints, or endpoint probes.
Those paths still need their own destination checks and network egress
restrictions. URL admission checks alone cannot prevent a later DNS change.
