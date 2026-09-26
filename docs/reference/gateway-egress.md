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
cache creation requests. It does not cover separate HTTP clients in Python
control-plane callbacks, custom tools, batch jobs, or endpoint probes. Those
paths still need their own destination checks and network egress restrictions.
URL admission checks alone cannot prevent a later DNS change.
