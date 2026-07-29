# Local models as routing candidates

Any OpenAI-compatible server you run yourself (Ollama, vLLM, llama.cpp) can join the
routing pool. On the wire it is a plain `openai` pool entry with an explicit `endpoint`:
no new provider kind, no API key (WMO sends `WMO_ENDPOINT_API_KEY` or a placeholder
against an explicit endpoint, never a real OpenAI key).

## Register

Interactive: run `wmo providers set`, pick `openai`, and answer the endpoint prompt with
your server's URL. The picker then lists what that server actually serves (its own
`GET /v1/models`), and prices default to $0 per Mtok, stamped explicitly. The endpoint
prompt covers ROUTING CANDIDATES; to also point the local WORKER agent at your server,
pass `--endpoint` on the command line (the scripted example below does both at once).

Scripted:

```bash
ollama pull qwen3:4b   # serves OpenAI-compatible at http://localhost:11434/v1
wmo providers set --provider openai --model qwen3:4b \
  --endpoint http://localhost:11434/v1 --pool-model "qwen3:4b" --tier open
```

Both paths live-ping the server before writing the entry. The written entry:

```toml
[[model]]
name = "qwen3-4b"
kind = "openai"
model = "qwen3:4b"
endpoint = "http://localhost:11434/v1"
tier = "open"
input_per_mtok = 0.0
output_per_mtok = 0.0
```

Prices are always explicit for endpoint-backed entries, even when the model id shadows a
built-in one: the price a candidate bills at is a property of the server. The $0 default
applies only to LOCALLY hosted servers (localhost, 127.0.0.1, `host.docker.internal`,
`*.local`); a remote OpenAI-compatible endpoint (Together, Groq, a LiteLLM proxy) bills
real money, so registration demands `--input-per-mtok`/`--output-per-mtok` for it.
Servers started with an API key (vLLM `--api-key`) are probed and served with
`WMO_ENDPOINT_API_KEY` as the bearer token.

## Serve it

```bash
wmo optimize route pin <world-model> --model qwen3-4b
wmo serve --name <world-model>
```

`pin` installs an honest static policy (every request to the local model, nothing
measured, nothing saved); replace it with `wmo optimize route fit` on a real outcome
matrix to let the router choose per request. The request log records the routed pool
entry, the provider runtime id, and the $0 effective cost per call.

## Turn candidates off without deleting them

Every pool entry participates in model selection by default. `enabled = false` on an
entry keeps it (handle, prices, comments) but removes it from everything that chooses
models: `wmo optimize route sweep`, `wmo optimize model`, and `wmo optimize route pin`.
The entry still validates at load, and policies fitted while it was on keep serving it.

## Zero-cost semantics

A $0 candidate is maximally price-favored by the cost knob, which is arithmetically
correct: it is free. What protects quality is the decision guard, exactly as for any
other candidate: a free pick serves only when the paired evidence says it is not worse
than the baseline. A pool where every candidate is free has no price to trade against,
so the savings half of the cost/quality dial refuses loudly; the coverage half works.
Savings against a priced fallback read as ~100% on requests the free arm served, priced
on the same tokens the request actually used.

## Docker note (hosted platform stacks)

Store the URL you typed. A stack that serves from inside a Docker container translates
loopback hostnames (`localhost`, `127.0.0.1`, `::1`) to `host.docker.internal` at the
provider boundary only; nothing on disk changes, so the same config serves identically
outside a container.
