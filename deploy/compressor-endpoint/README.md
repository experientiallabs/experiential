# Compressor endpoint

The team's shared context compressor: LLMLingua-2 (177M multilingual BERT token classifier)
served fp32 on one H100, behind TLS and a bearer token. Anyone on the team can point a client
at it instead of running a 700MB model locally, which on a laptop CPU costs 4.5 s per 10k
tokens against 0.2 s here.

- URL: `https://40.80.93.150:8443`
- Host: `h100-dev-box-6` (Standard_NC80adis_H100_v5, centralindia), reachable with plain
  `ssh h100-dev-box-6`
- Service root on the box: `/nvme/work/wmo-compressor` (the root filesystem sits at ~97% full,
  so everything lives on `/nvme`)
- Client: `wmo.optimize.compression_endpoint.LLMLingua2EndpointCompressor`

## Using it

```bash
cp .env.example .env      # then fill in the two variables below
```

```
WMO_COMPRESSOR_URL=https://40.80.93.150:8443
WMO_COMPRESSOR_API_KEY=<ask whoever administers the box>
```

Register it into the D-COMPRESS seam once, then policies may name `llmlingua2-endpoint` like
any other compressor:

```python
from wmo.optimize.compression import CompressionConfig
from wmo.optimize.compression_endpoint import register_endpoint_compressor

register_endpoint_compressor()
result = ...  # anything that resolves a compressor by id now finds it
```

Registration is explicit rather than on import, because building the client needs credentials
and the seam's rule is that a misconfiguration fails at mount. It also contacts `/healthz` once
to confirm the box really is running absolute-threshold selection before attesting
`append_stable`, so the serving admission ticket is verified against the running server rather
than asserted by the client. To use it directly without the registry:

```python
from wmo.optimize.compression import CompressionConfig
from wmo.optimize.compression_endpoint import LLMLingua2EndpointCompressor

client = LLMLingua2EndpointCompressor.from_env()
result = client.compress(
    ["the quarterly revenue report shows ...", "tool output: {...}"],
    CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=0.5),
)
result.segments, result.tokens_in_compressed, result.cost_usd
```

The endpoint serves a self-signed certificate, pinned rather than CA-signed (there is no domain
to buy a certificate for, and a pinned certificate is strictly stronger than trusting every
public CA). Clients verify against `compressor-cert.pem` in this directory, which the client
picks up automatically; `WMO_COMPRESSOR_CERT` overrides the path. From the shell:

```bash
curl --cacert deploy/compressor-endpoint/compressor-cert.pem https://40.80.93.150:8443/healthz
```

`/healthz` needs no auth and reports the running version, the weight fingerprint, uptime, and
the startup self-test verdict. `POST /v1/compress` needs `Authorization: Bearer <token>` and
takes `{"segments": [...], "threshold": 0.5}`.

## What `threshold` means

It is an absolute cutoff on each word's keep probability, NOT a target compression ratio, and
that distinction is the whole reason this endpoint is usable in serving. Stock LLMLingua-2 keeps
the top-k fraction per input, so every keep/drop decision depends on the whole input's score
distribution: appending one turn rewrites 45-81% of the already-emitted compressed prefix
(C1 round 0) and forfeits the provider's prompt cache. An absolute threshold decides each word
locally, so an unchanged segment always compresses to the same bytes and the cache survives.

Compression ratio is therefore an outcome, not a setting. At threshold 0.5 the measured keep
ratio on the C1 audit corpus is 0.64 (36% of tokens removed). Threshold 0.0 is a strict no-op.

The threshold is a per-request VALUE; the selection RULE is not reachable from the API. There
is no percentile, top-k, or quantile code path in the server at all, the comparison is a
literal `p >= threshold`, and `/healthz` publishes `selection_rule` so a client can verify it.
That matters because the seam admits a compressor to serving only if it attests
`append_stable`, and the attestation is true only for absolute-threshold selection.

## Measured baseline

Through the endpoint, from a laptop in the US, over the 120-transcript C1 audit corpus
(135,859 GPT-2 tokens), threshold 0.5. Reproduce with
`uv run python .agents/scripts/verify_compressor_endpoint.py`.

Ranges are across three runs, because the network leg varies while the model does not: the keep
ratio came out byte-identical (0.641614) every time, so treat the spread as network and GPU
scheduling, not as compressor variance.

| | through the endpoint | C1 on-box bench |
| --- | --- | --- |
| $/10k tokens | $0.00080 to $0.00100 | $0.000534 |
| s/10k tokens (wall, incl. network) | 0.68 to 0.74 | 0.197 |
| s/10k tokens (server compute only) | 0.29 to 0.37 | 0.197 |
| p50 per request, ~9k-token batch | 0.46 to 0.52 s | n/a |
| p50 per request, ~1.1k-token single transcript | 0.25 s | n/a |
| keep ratio | 0.642 | 0.596 (GPT-2 basis) |

An 800-scenario routing bank fit (the seam's `CompressingEmbedder` compresses every fit
scenario in one `compress` call) goes through as a SINGLE round trip in 1.7 to 2.0 s.
The request caps are sized for that: 1024 segments and 8M chars, both published on `/healthz`
so a client can check them rather than trust a constant. The client splits anything larger on
fixed boundaries, which preserves append stability because boundaries depend only on the
segment list and the server is batch-invariant.

Read those two rows together before optimizing anything: for small requests the round trip to
centralindia dominates, not the GPU. A 1.1k-token call spends ~0.25 s almost entirely on the
network, while a 9k-token call spends 0.46 s. Batch segments into one call where you can.

Cost is honest per call: the server times its own compute and prices it as GPU-seconds at half
the box's retail rate ($19.544/hr, 2 GPUs, so $9.772/GPU-hr). It does not amortize idle time,
so the box's real cost per useful token is higher whenever the endpoint is quiet.

Break-even against the tokens it removes (3.6k per 10k) still holds through the network:
3.4x at the cheapest pool tier (gpt-5.4-mini, $0.75/M in), 13x at sonnet-5, 22x at gpt-5.5.

## Guarantees, and how they are enforced

The service refuses to start unless three properties hold, checked against the real loaded
weights on every boot (`run_self_test`, reported in `/healthz`):

1. **Determinism.** The same input compressed three times is byte-identical.
2. **Batch-composition invariance.** A segment compresses identically alone and alongside
   others. This is the property fp16 breaks (C1 measured keep/drop flips with batch mates), and
   it is why the server is fp32-only and refuses any other dtype. Requests are additionally
   serialized by a lock, so a response never depends on concurrent traffic.
3. **Losslessness at threshold 0.0.** Nothing is dropped when nothing may be.

A failing self-test exits nonzero, which under `Restart=always` becomes a visible restart loop
in journald rather than a service quietly emitting non-deterministic output.

One more rule in the same spirit: a word the model did not score (a chunk truncated at BERT's
512-position limit) is KEPT. A compressor may never drop content it did not look at.

## Operating it

```bash
./deploy.sh                # rsync + bootstrap + restart + health check, idempotent
./deploy.sh --cert-only    # refetch the pinned public cert after a rotation

ssh h100-dev-box-6 sudo systemctl status wmo-compressor
ssh h100-dev-box-6 sudo journalctl -u wmo-compressor -f
```

Redeploying does NOT rotate the certificate: `deploy.sh` copies four files and leaves `tls/`,
`venv/`, and `hf/` on the box alone, so the pinned cert every client verifies against survives.
(It did rotate them briefly: an `rsync --delete-excluded` was deleting everything it was not
explicitly told to copy, which regenerated TLS and rebuilt the venv on every deploy. Fixed, and
verified by deploying twice and diffing the fingerprint.)

`deploy.sh` pushes this directory to the box and runs `bootstrap.sh` there, which builds the
venv (from the pip cache C1's bench already populated, so torch is a local unpack), copies the
weights, generates TLS material and the token if absent, installs the systemd unit, and waits
for health. Re-running it is safe: nothing is regenerated if it already exists.

Rate limit: 60 requests/min sustained, burst 120, per token. Raise it by setting
`WMO_COMPRESSOR_RATE_PER_MIN` / `WMO_COMPRESSOR_BURST` in the box's environment file and
restarting.

### Rotating the token

The token was generated on the box with `openssl rand -hex 32` and has never been in git, in a
transcript, or on any other machine except each holder's gitignored `.env`. To rotate:

```bash
ssh h100-dev-box-6
printf 'WMO_COMPRESSOR_TOKEN=%s\n' "$(openssl rand -hex 32)" | sudo tee /etc/wmo-compressor/env >/dev/null
sudo chmod 600 /etc/wmo-compressor/env && sudo chown root:root /etc/wmo-compressor/env
sudo systemctl restart wmo-compressor
sudo cat /etc/wmo-compressor/env    # read it once, hand it out over a private channel
```

Then every client updates `WMO_COMPRESSOR_API_KEY` in their own `.env`. Rotating the TLS
certificate is the same shape: delete `/nvme/work/wmo-compressor/tls/*`, re-run `./deploy.sh`,
and commit the refreshed `compressor-cert.pem`.

### Security posture

Bearer token (constant-time compared) over pinned TLS, per-token rate limiting, request size
caps, and an unauthenticated health endpoint that exposes only version and uptime. Inbound TCP
8443 is open to the internet through the Azure NSG rule `allow-wmo-compressor-8443` on
`h100-dev-box-6NSG`, because teammates connect from arbitrary networks. The exposure that
remains is the token itself: anyone holding it can spend the box's GPU within the rate limit.
Treat it as a shared credential, rotate it when someone leaves, and never paste it anywhere it
could be logged.

If the box is ever repurposed, delete the NSG rule as well as the service:

```bash
az network nsg rule delete -g H100-DEV-BOX-6 --nsg-name h100-dev-box-6NSG -n allow-wmo-compressor-8443
ssh h100-dev-box-6 'sudo systemctl disable --now wmo-compressor && sudo rm -rf /etc/wmo-compressor'
```
