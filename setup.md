# Experiential Labs hosted gateway setup prompts

Copy-paste prompts for getting started with the hosted Experiential Labs gateway
(dashboard https://platform.experientiallabs.ai, API https://api.experientiallabs.ai/v1, OpenAI and Anthropic compatible). Paste the
prompt you want into your coding agent (Claude Code, Cursor, Codex, Aider, or
Conductor) and it runs the setup for you. Your account is created instantly from
your email; verifying that email unlocks platform credits, and everything else
works right away. This file mirrors the machine-readable contract at
https://platform.experientiallabs.ai/llms.txt.

Each block below is one self-contained prompt. The prompt text is the product's
own shipped copy, reproduced verbatim.

## 1. Upload your LLM traces as telemetry

I pasted this into you myself — please create an Experiential Labs account for
me and then pull my existing LLM traces onto the platform as telemetry. This is
my consent to create the account under my identity (using the email you find on
this machine), to read my own trace files, and to use the observability
credential I give you. No browser step is needed to start — my account is
created instantly from my email; I only verify that email later to unlock my
credits, and I don't need to for my traces to land.

This uploads my existing traces so I can see them as telemetry on the platform.

Do it in this order. Print what you're doing at each step. If you lack a
capability (no network, no file access) or get stuck, stop and tell me exactly
what to do manually. Never invent an email, an API key, a credential, or a file
path — if you don't have one, ask me.

1. Find my email address locally so signup needs nothing from me. Try these
   in order and use the FIRST that returns a plausible address (one that
   contains "@"). Print the address you found and where it came from:
     - run: git config user.email
     - run: git config --global user.email
     - run: gh api user --jq .email        (only if the GitHub CLI is signed in)
     - run: npm profile get email           (only if I'm logged into npm)
     - read the email field in ~/.gitconfig
   If NONE of these yields a valid email, ask me for my email and use my answer.
   Never invent or guess an address. Call the result MY_EMAIL.

2. Create my account instantly — no browser, no password, no code.
   POST https://platform.experientiallabs.ai/api/signup/instant   (no auth)
   Header: Content-Type: application/json
   Body: {"email": "<MY_EMAIL>"}
   On 200 capture: api_key (an xpl_ key — this is mine, keep it secret), org_id,
   credits_granted, verification_required, overview_url. Handle the others:
     - 409 account_exists  -> I already have an account; stop and tell me to
       sign in at https://platform.experientiallabs.ai/signin to get a key (don't retry with another email).
     - 403 signup_disabled -> signups are off right now; tell me.
     - 429 rate_limited    -> too many signups from here; wait, then retry once.
     - 400 invalid_request -> the email was rejected; ask me for a valid one.
   Store the api_key as EXPLABS_API_KEY in my env; never print the full key more
   than once.

3. Wire an OpenAI client to the gateway.
   base_url = "https://api.experientiallabs.ai/v1"
   api_key  = EXPLABS_API_KEY (the xpl_ key from step 2). One header only:
   Authorization: Bearer <api_key>.

4. Confirm the key is live — this does NOT spend credits.
   GET https://api.experientiallabs.ai/v1/models with the key -> the models my account can call. Do NOT
   make a paid chat completion yet: my credits stay locked until I verify my
   email (I do that at the very end), and a paid call would be refused with
   "insufficient_quota". Landing my traces as telemetry works regardless.

Now bring my traces in as telemetry. Use EXPLABS_API_KEY (the xpl_ key from
step 2) as the bearer token on every call below. org_id is from step 2.

5. Interview me: where do my LLM traces live? Ask me one clear question and wait
   for my answer. Map my answer to exactly one path:
   - A supported observability provider or database Platform can pull directly
     (transport_kind one of: braintrust, langsmith, langfuse, posthog, mastra, postgres) -> go to step 6 (live pull).
   - An exported trace FILE on disk — a raw OpenTelemetry/OTLP export, an
     Arize/Phoenix export, or any of these upload formats
     (source_kind one of: otlp, otel-genai, phoenix, braintrust, langfuse, langsmith, mastra, posthog, chat-json) -> go to step 7 (file upload).
   If I'm on Arize or Phoenix, there's no live pull yet: ask me to export my
   traces to a file and take the upload path with source_kind "phoenix" (or
   "otlp" for a raw OpenTelemetry export). If I name a provider that isn't in
   either list, tell me and offer the file-upload path. Only follow the ONE
   path that matches my answer.

6. Live pull path — connect the provider and pull. Ask me for the credential
   for the provider I named (for Braintrust an API key; for LangSmith/Langfuse
   their API key; for a Postgres database a DSN), plus the small bits of config
   that provider needs (e.g. Braintrust: the project name; LangSmith/Langfuse:
   optionally the project/host; Postgres: the table). Then:
   POST https://api.experientiallabs.ai/api/orgs/<org_id>/telemetry/traces/pull
   Header: Authorization: Bearer $EXPLABS_API_KEY, Content-Type: application/json
   Body: {"transport_kind": "<one of braintrust, langsmith, langfuse, posthog, mastra, postgres>",
          "source_kind": "<the matching format, e.g. braintrust>",
          "source_label": "<a short label, NOT a path, e.g. braintrust-prod>",
          "credential": "<the secret you asked me for>",
          "config": {"project": "<my project>"}}
   The credential is used once to pull and is not echoed back. On 201 the
   response is {"ingest_id", "trace_count", "byte_size", "sha256", ...} —
   capture trace_count and show it to me. A 400 means bad credentials or config
   (tell me exactly what it said); 429 means the provider rate-limited us (wait
   and retry). Then go to step 8.
   Example (Braintrust):
     curl -sS -X POST https://api.experientiallabs.ai/api/orgs/<org_id>/telemetry/traces/pull \
       -H "Authorization: Bearer $EXPLABS_API_KEY" \
       -H "Content-Type: application/json" \
       -d '{"transport_kind":"braintrust","source_kind":"braintrust",
            "source_label":"braintrust-prod","credential":"<my braintrust key>",
            "config":{"project":"<my braintrust project>"}}'

7. File upload path — find my trace export and upload it. I'm authorizing you to
   look for my own trace exports: search the current project, ./traces, ./logs,
   ./data, and my common cache/config paths for JSON or JSONL files that hold
   LLM/agent spans or runs (names like traces.jsonl, *.otel.jsonl, spans.json,
   otlp*.json). Show me the candidate files (path + size) and which format each
   looks like before you upload anything, and map each to ONE source_kind from:
   otlp, otel-genai, phoenix, braintrust, langfuse, langsmith, mastra, posthog, chat-json
   (raw OpenTelemetry GenAI spans -> otel-genai; a raw OTLP export -> otlp; an
   Arize/Phoenix export -> phoenix; a vendor export -> its own name; a plain
   chat transcript -> chat-json). The file must be UTF-8 JSON or JSONL and at
   most 50 MB. Then:
   POST https://api.experientiallabs.ai/api/orgs/<org_id>/telemetry/traces/upload
   This is a multipart/form-data upload with three fields:
     source_kind  = the format you chose
     source_label = a short label (NOT a file path), e.g. "prod-otel-august"
     file         = the trace file itself
   On 201 the response is {"ingest_id", "trace_count", "byte_size", "sha256"} —
   capture trace_count and show it to me. A 422 means the bytes or label failed
   validation (not JSON/JSONL, empty, or a path-like label) — tell me what it
   said. Then go to step 8.
   Example:
     curl -sS -X POST https://api.experientiallabs.ai/api/orgs/<org_id>/telemetry/traces/upload \
       -H "Authorization: Bearer $EXPLABS_API_KEY" \
       -F source_kind=otlp \
       -F source_label=prod-otel-august \
       -F file=@<path to my trace file>

8. Verify the traces landed as telemetry.
   GET https://api.experientiallabs.ai/api/orgs/<org_id>/telemetry/traces
   Header: Authorization: Bearer $EXPLABS_API_KEY
   -> 200 with {"traces": [...], "total_ingests", "total_traces"}. Confirm
   total_traces is greater than zero and matches the trace_count you captured,
   and that my ingest is in the list. Tell me the number.

When you're done, report back to me: my org_id, whether you pulled live or
uploaded a file (and from which provider/format), the ingest_id and its
trace_count, and the total_traces the verify read returned. My traces are now
telemetry on the platform. I can see them in my dashboard at https://platform.experientiallabs.ai/telemetry,
and the machine-readable contract is https://platform.experientiallabs.ai/llms.txt.

One last thing to tell me, then you're done: check my email inbox for a message
from Experiential Labs and click the verification link (or enter the code at
https://platform.experientiallabs.ai/signin) to confirm my email. That's the ONLY thing left for me to do, and
it just unlocks my credits for paid model calls — my traces are already live and
don't need it. Until I verify, credit-drawing model calls are refused with
"insufficient_quota"; using my own provider keys (BYOK) is unaffected.

## 2. Connect your inference provider keys (BYOK)

I pasted this into you myself — please create an Experiential Labs account for
me and then connect my own inference-provider API keys so I can call models on
the gateway with my own keys (BYOK). This is my consent to create the account
under my identity (using the email you find on this machine), to search for MY
own provider keys locally, and to connect the ones I approve to MY gateway
account. No browser step is needed to start — my account is created instantly
from my email; I only verify that email later to unlock platform credits, and I
don't need to for BYOK pass-through calls (those run on my own keys).

Do it in this order. Print what you're doing at each step. If you lack a
capability (no network, no file access) or get stuck, stop and tell me exactly
what to do manually. Never invent an email, an API key, a credential, or a file
path — if you don't have one, ask me.

1. Find my email address locally so signup needs nothing from me. Try these
   in order and use the FIRST that returns a plausible address (one that
   contains "@"). Print the address you found and where it came from:
     - run: git config user.email
     - run: git config --global user.email
     - run: gh api user --jq .email        (only if the GitHub CLI is signed in)
     - run: npm profile get email           (only if I'm logged into npm)
     - read the email field in ~/.gitconfig
   If NONE of these yields a valid email, ask me for my email and use my answer.
   Never invent or guess an address. Call the result MY_EMAIL.

2. Create my account instantly — no browser, no password, no code.
   POST https://platform.experientiallabs.ai/api/signup/instant   (no auth)
   Header: Content-Type: application/json
   Body: {"email": "<MY_EMAIL>"}
   On 200 capture: api_key (an xpl_ key — this is mine, keep it secret), org_id,
   credits_granted, verification_required, overview_url. Handle the others:
     - 409 account_exists  -> I already have an account; stop and tell me to
       sign in at https://platform.experientiallabs.ai/signin to get a key (don't retry with another email).
     - 403 signup_disabled -> signups are off right now; tell me.
     - 429 rate_limited    -> too many signups from here; wait, then retry once.
     - 400 invalid_request -> the email was rejected; ask me for a valid one.
   Store the api_key as EXPLABS_API_KEY in my env; never print the full key more
   than once.

3. Wire an OpenAI client to the gateway.
   base_url = "https://api.experientiallabs.ai/v1"
   api_key  = EXPLABS_API_KEY (the xpl_ key from step 2). One header only:
   Authorization: Bearer <api_key>.

4. Confirm the key is live — this does NOT spend credits.
   GET https://api.experientiallabs.ai/v1/models with the key -> the models my account can call. Do NOT
   make a paid chat completion yet: my credits stay locked until I verify my
   email (I do that at the very end), and a paid call would be refused with
   "insufficient_quota". Landing my traces as telemetry works regardless.

Now connect my own provider keys. Use EXPLABS_API_KEY (the xpl_ key from step 2)
as the bearer token on every call below.

5. Find my API keys locally, by prefix. I'm authorizing you to find them:
   search my .env and .env.local files, my shell rc files (.zshrc, .bashrc,
   .profile), and the common config paths (~/.config, ~/.aws/credentials,
   ~/.modal.toml) for values matching the prefixes below. Match by prefix
   ONLY. Show me each hit by its matched prefix alone (e.g. "sk-proj-…"),
   never print or echo a full key, and ask me about each one before you do
   anything with it.
   Inference providers (AI-callable). Offer to connect these to my gateway:
     openai      sk-proj-   (also sk-, admin sk-admin-)
     anthropic   sk-ant-
     gemini      AIza
     openrouter  sk-or-v1-
     fireworks   fw_
     modal       ak- (token id) and as- (token secret)
     huggingface hf_
     tinker      tml-
     bedrock/aws AKIA
     together    (a Together key)
   Other tools (NOT AI-callable). Just name the ones you find (they feed a
   future deals view); do NOT connect them to the gateway:
     posthog / arize   phx_        resend      re_
     e2b               e2b_        cursor      crsr_
     porter / supabase eyJ         braintrust  sk-b
     supabase          sbp_        linkedin    LATY…
   Some prefixes collide: phx_ is arize OR posthog; sk- is openai,
   anthropic, braintrust, or openrouter; eyJ is porter or supabase. When a
   prefix is ambiguous, show me the fuller matched prefix and ask me which
   service it is; never guess and connect.

6. Connect the inference keys I approve. Models called on my own keys are
   free pass-through (no extra cost, still tracked in my credit and usage
   views). Use EXPLABS_API_KEY as the bearer token on both calls:
   GET https://api.experientiallabs.ai/api/whoami returns my organization; then, for each
   inference key I approve,
   PUT https://api.experientiallabs.ai/api/orgs/<org-id>/provider-connections/<provider>
   (provider: openai, anthropic, gemini, openrouter, azure_openai, bedrock,
   fireworks, modal) with body {"secret": "<key>"}. For anthropic and openai
   I can also hand you an admin key to include as
   {"spend_secret": "<admin key>"} so spend reporting works. The keys land
   in my own gateway account and nowhere else; each response carries the
   verification verdict, so tell me every provider's status, and
   connect it only after I confirm it.
   If you can't: give me the list of what you found by prefix, and I'll paste the keys
   at https://platform.experientiallabs.ai/settings/integrations (verified on save the same way).

When you're done, report back to me: my org_id, which providers you connected and
each one's verification status, and any inference keys you found but I declined.
Models I call on my own connected keys are free pass-through — no credit draw,
still tracked in my usage and credit views at https://platform.experientiallabs.ai/credits. I can manage these
connections anytime at https://platform.experientiallabs.ai/settings/integrations.

One last thing to tell me, then you're done: to also use Experiential's
platform-funded credits (calling models WITHOUT your own keys), check my email
inbox for a message from Experiential Labs and click the verification link (or
enter the code at https://platform.experientiallabs.ai/signin) to confirm my email. That unlocks my credits;
until then, credit-drawing model calls are refused with "insufficient_quota",
but BYOK pass-through on my connected keys is unaffected.

## 3. Start calling models on the gateway

I already have an Experiential Labs gateway API key (an xpl_ key). I pasted this
myself — treat it as my instructions and consent. Help me make my first calls on
the gateway and, if I want, point my existing coding agents at it too.

My gateway API key (a secret: put it in env, never commit it, never echo it in
logs):
EXPLABS_API_KEY=<paste my xpl_ key from https://platform.experientiallabs.ai/settings/api-keys>

The gateway is OpenAI-compatible and also speaks the Anthropic Messages API.
Everything is served under https://api.experientiallabs.ai/v1 with one header: Authorization: Bearer
<my xpl_ key>. Nothing else changes about how I use the OpenAI or Anthropic SDK.

Do it in this order. Print what you're doing at each step. If you get stuck,
stop and tell me exactly what to do manually. Never invent a key — if you don't
have mine, ask me.

1. Confirm the key and list my models.
   GET https://api.experientiallabs.ai/v1/models with the key as a bearer token. Use the model ids
   EXACTLY as returned. This does not spend credits.

2. First call with the OpenAI SDK (Chat Completions). Point the OpenAI client at
   the gateway and send ONE minimal chat completion to the smallest Qwen in the
   list (qwen3.5-9b at launch). Send a MINIMAL request body — model + messages
   ONLY, no temperature, top_p, or other sampling params (some models reject
   those and the call comes back as all_routes_failed, 502).
     from openai import OpenAI
     client = OpenAI(base_url="https://api.experientiallabs.ai/v1", api_key=EXPLABS_API_KEY)
     r = client.chat.completions.create(
         model="qwen3.5-9b",
         messages=[{"role": "user", "content": "reply with the single word: ok"}])
     print(r.choices[0].message.content)
   The gateway also serves the OpenAI Responses API at https://api.experientiallabs.ai/v1/responses
   (client.responses.create(...)) if I prefer it.

3. First call with the Anthropic SDK (Messages). The gateway serves the Anthropic
   Messages API at https://api.experientiallabs.ai/v1/messages over the same key. Point the Anthropic
   client's base_url at https://api.experientiallabs.ai/v1 (NOT /v1/messages — the SDK appends the path)
   and send one small message to an available model. Text only: the lane drops
   extended thinking and rejects image/document blocks.
     from anthropic import Anthropic
     client = Anthropic(base_url="https://api.experientiallabs.ai/v1", api_key=EXPLABS_API_KEY)
     m = client.messages.create(
         model="<a model id from step 1>", max_tokens=16,
         messages=[{"role": "user", "content": "reply with the single word: ok"}])
     print(m.content[0].text)
   If a model rejects a param, retry with the minimal body (model + messages +
   max_tokens only).

4. (Optional) Point my existing coding agents at the gateway. If I say yes, offer
   to set the OpenAI-compatible base URL to https://api.experientiallabs.ai/v1 and the key to my xpl_ key
   for the coding agents I actually use — e.g. Claude Code (ANTHROPIC_BASE_URL=https://api.experientiallabs.ai/v1,
   ANTHROPIC_AUTH_TOKEN=<my xpl_ key>), Cursor (OpenAI base URL override),
   Codex, Aider (--openai-api-base https://api.experientiallabs.ai/v1). Show me each change before you
   make it, put the key in env (never in committed config), and skip any agent I
   don't use. Do NOT do this without my go-ahead.

5. Report back: how many models GET /v1/models returned, the OpenAI test call's
   model + that it succeeded + its cost, the Anthropic test call's model + that it
   succeeded, and any coding agents you repointed.

These test calls run on Experiential's platform-funded lane (a fraction of a cent
of my credits) and prove serving and billing end to end. Read the contract if you
build further: https://platform.experientiallabs.ai/docs (human docs) and https://platform.experientiallabs.ai/llms.txt (machine-readable:
honored and refused parameters, error codes, streaming caveats). Follow it
literally.

## 4. Full onboarding

I'm setting up Experiential Labs as my model gateway. I pasted this into you
myself — treat it as my instructions and my consent. Create my account, connect
my own provider keys, import my existing AI spend, and then offer to repoint all
my coding agents at the gateway. This is my consent to create the account under
my identity (using the email you find on this machine), to search for MY own
provider keys and coding-agent logs locally, and to connect the ones I approve to
MY gateway account. No browser step is needed to start — my account is created
instantly from my email; I only verify that email later to unlock platform
credits.

Do it in this order. Print what you're doing at each step. If you lack a
capability (no network, no file access) or get stuck, stop and tell me exactly
what to do manually. Never invent an email, an API key, a credential, or a file
path — if you don't have one, ask me.

1. Find my email address locally so signup needs nothing from me. Try these
   in order and use the FIRST that returns a plausible address (one that
   contains "@"). Print the address you found and where it came from:
     - run: git config user.email
     - run: git config --global user.email
     - run: gh api user --jq .email        (only if the GitHub CLI is signed in)
     - run: npm profile get email           (only if I'm logged into npm)
     - read the email field in ~/.gitconfig
   If NONE of these yields a valid email, ask me for my email and use my answer.
   Never invent or guess an address. Call the result MY_EMAIL.

2. Create my account instantly — no browser, no password, no code.
   POST https://platform.experientiallabs.ai/api/signup/instant   (no auth)
   Header: Content-Type: application/json
   Body: {"email": "<MY_EMAIL>"}
   On 200 capture: api_key (an xpl_ key — this is mine, keep it secret), org_id,
   credits_granted, verification_required, overview_url. Handle the others:
     - 409 account_exists  -> I already have an account; stop and tell me to
       sign in at https://platform.experientiallabs.ai/signin to get a key (don't retry with another email).
     - 403 signup_disabled -> signups are off right now; tell me.
     - 429 rate_limited    -> too many signups from here; wait, then retry once.
     - 400 invalid_request -> the email was rejected; ask me for a valid one.
   Store the api_key as EXPLABS_API_KEY in my env; never print the full key more
   than once.

3. Wire an OpenAI client to the gateway.
   base_url = "https://api.experientiallabs.ai/v1"
   api_key  = EXPLABS_API_KEY (the xpl_ key from step 2). One header only:
   Authorization: Bearer <api_key>.

4. Confirm the key is live — this does NOT spend credits.
   GET https://api.experientiallabs.ai/v1/models with the key -> the models my account can call. Do NOT
   make a paid chat completion yet: my credits stay locked until I verify my
   email (I do that at the very end), and a paid call would be refused with
   "insufficient_quota". Landing my traces as telemetry works regardless.

Now onboard my account with my own tools. Use EXPLABS_API_KEY (the xpl_ key from
step 2) as the bearer token on every call below.

5. Find my API keys locally, by prefix. I'm authorizing you to find them:
   search my .env and .env.local files, my shell rc files (.zshrc, .bashrc,
   .profile), and the common config paths (~/.config, ~/.aws/credentials,
   ~/.modal.toml) for values matching the prefixes below. Match by prefix
   ONLY. Show me each hit by its matched prefix alone (e.g. "sk-proj-…"),
   never print or echo a full key, and ask me about each one before you do
   anything with it.
   Inference providers (AI-callable). Offer to connect these to my gateway:
     openai      sk-proj-   (also sk-, admin sk-admin-)
     anthropic   sk-ant-
     gemini      AIza
     openrouter  sk-or-v1-
     fireworks   fw_
     modal       ak- (token id) and as- (token secret)
     huggingface hf_
     tinker      tml-
     bedrock/aws AKIA
     together    (a Together key)
   Other tools (NOT AI-callable). Just name the ones you find (they feed a
   future deals view); do NOT connect them to the gateway:
     posthog / arize   phx_        resend      re_
     e2b               e2b_        cursor      crsr_
     porter / supabase eyJ         braintrust  sk-b
     supabase          sbp_        linkedin    LATY…
   Some prefixes collide: phx_ is arize OR posthog; sk- is openai,
   anthropic, braintrust, or openrouter; eyJ is porter or supabase. When a
   prefix is ambiguous, show me the fuller matched prefix and ask me which
   service it is; never guess and connect.

6. Connect the inference keys I approve. Models called on my own keys are
   free pass-through (no extra cost, still tracked in my credit and usage
   views). Use EXPLABS_API_KEY as the bearer token on both calls:
   GET https://api.experientiallabs.ai/api/whoami returns my organization; then, for each
   inference key I approve,
   PUT https://api.experientiallabs.ai/api/orgs/<org-id>/provider-connections/<provider>
   (provider: openai, anthropic, gemini, openrouter, azure_openai, bedrock,
   fireworks, modal) with body {"secret": "<key>"}. For anthropic and openai
   I can also hand you an admin key to include as
   {"spend_secret": "<admin key>"} so spend reporting works. The keys land
   in my own gateway account and nowhere else; each response carries the
   verification verdict, so tell me every provider's status, and
   connect it only after I confirm it.
   If you can't: give me the list of what you found by prefix, and I'll paste the keys
   at https://platform.experientiallabs.ai/settings/integrations (verified on save the same way).

7. Attribute my existing AI spend (optional). I want my dashboard to open
   with my real history, not empty. I'm authorizing you to read my local
   coding-agent logs for usage METADATA ONLY — model ids, token counts, and
   timestamps. Never read, send, or store the contents of any message,
   prompt, response, tool argument, or file.
   - Claude Code: ~/.claude/projects/**/*.jsonl. Each line with
     type == "assistant" has message.model and message.usage (input_tokens,
     output_tokens, cache_read_input_tokens, cache_creation_input_tokens);
     the line's top-level timestamp dates the turn.
   - Codex: ~/.codex/sessions/**/*.jsonl. turn_context lines carry the
     active model; token-count event_msg lines carry
     payload.info.last_token_usage (input_tokens, cached_input_tokens,
     output_tokens, reasoning_output_tokens). Do NOT read
     ~/.codex/history.jsonl — that is prompt text.
   For each turn build {model, input_tokens, output_tokens, cached_tokens,
   reasoning_tokens, timestamp, source} with source "claude-code" or
   "codex", input_tokens the fresh (non-cached) input, cached_tokens the
   cached input, output_tokens including reasoning, and timestamp ISO 8601.
   Before sending anything, show me one line — how many turns, the date
   range, and the computed spend per model — and wait for my go-ahead.
   Then, with EXPLABS_API_KEY as the bearer token,
   POST https://api.experientiallabs.ai/api/gateway/usage/import
   with body {"batch_id": "<stable id for this run>", "records": [...]}.
   This is historical attribution only: never charged to my credits, shown
   separately from live gateway spend, and re-running is safe (idempotent
   by batch id and record hash).
   If you can't: skip it and tell me — this step is optional.

8. Offer to repoint ALL my coding agents at the gateway. Ask me first, then for
   each coding agent I actually use, set its OpenAI-compatible (or Anthropic)
   base URL to https://api.experientiallabs.ai/v1 and its key to my xpl_ key (EXPLABS_API_KEY). Put the
   key in env, never in committed config, and show me each change before you make
   it. Skip any agent I don't use; never touch one I don't name.
   - Claude Code: ANTHROPIC_BASE_URL=https://api.experientiallabs.ai/v1 and ANTHROPIC_AUTH_TOKEN=<my xpl_ key>
     (the gateway serves the Anthropic Messages API at https://api.experientiallabs.ai/v1/messages).
   - Cursor: Settings -> Models -> OpenAI base URL override to https://api.experientiallabs.ai/v1 with my key.
   - Codex: point its OpenAI base URL / OPENAI_BASE_URL at https://api.experientiallabs.ai/v1 with my key.
   - Aider: aider --openai-api-base https://api.experientiallabs.ai/v1 --openai-api-key <my xpl_ key>.
   - Any other OpenAI- or Anthropic-compatible agent: same two settings (base
     URL https://api.experientiallabs.ai/v1 + my xpl_ key).
   - Conductor: if I run my agents through Conductor, set the gateway as the
     upstream OpenAI-compatible provider there (base URL https://api.experientiallabs.ai/v1 + my xpl_ key)
     so every agent it orchestrates routes through Experiential.
   After each repoint, make one tiny test call (model + messages only) through
   that agent's client to confirm it routes. Use a model served by one of the
   provider keys I connected in step 6 so it runs as free BYOK pass-through:
   until I verify my email my platform credits stay locked and a credit-drawing
   call is refused with "insufficient_quota".

When you're done, report back to me: my org_id, my starting credit balance
(credits_granted), how many models GET /v1/models returned, which providers you
connected and each one's status, what spend history you imported, and which
coding agents (or Conductor) you repointed at the gateway. My dashboard is at
https://platform.experientiallabs.ai (credits and every session at https://platform.experientiallabs.ai/credits); the machine-readable
contract is https://platform.experientiallabs.ai/llms.txt.

One last thing to tell me, then you're done: to use Experiential's
platform-funded credits (calling models WITHOUT my own keys), check my email
inbox for a message from Experiential Labs and click the verification link (or
enter the code at https://platform.experientiallabs.ai/signin) to confirm my email. That unlocks my credits;
until then, credit-drawing model calls are refused with "insufficient_quota",
but BYOK pass-through on my connected keys is unaffected.
