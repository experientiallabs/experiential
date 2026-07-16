# Pulling context from the tools you already use

A world model is only as grounded as the facts it can see. **Context connectors** pull the
documents, issues, messages, mail, and events your team already keeps in GitHub, Google, Slack,
and Notion, plus live web search results from Brave, into normalized, replayable **context
bundles**, and attach them to a model's knowledge base so the environment stops guessing at
entities and rules it could simply know.

Everything plugs into **one interface** (`ContextConnector`: `connect`, `verify`, `pull`) and
**one normalized shape** (`ContextItem`), so adding a service is a thin connector, never a
rewrite. Credentials live once per user in `~/.wmh/connectors.toml`; bundles live per project
under `.wmh/context/<name>/` (a `manifest.json` plus one item per `items.jsonl` line).

## Quickstart

```bash
wmh connect github                                   # authorize once (device flow)
wmh context pull github --target owner/repo          # pull issues/PRs/README into a bundle
wmh context attach github-20260715-120000 --model m  # render it into the model's knowledge base
```

`wmh connect` with no service lists every connector and its credential status. `wmh context
list` and `wmh context show <bundle>` inspect what was pulled (`show` renders the first 20
items; raise `--limit` for more); `--overwrite` replaces a bundle, and `--remove` on
`wmh connect <service>` deletes a stored credential. Every `wmh context` command operates on
the current directory's `.wmh/context/`; pass `--dir <project>` to work against another
project directory. Every connector also accepts a plain bearer token via `WMH_<NAME>_TOKEN`
(name upper-cased, hyphens to underscores), which skips the file entirely for CI and headless
runs; `brave` additionally honors the `BRAVE_SEARCH_API_KEY` deployments already carry for the
grounding engine.

## Services

| Connector | What it pulls | Item kinds | Auth |
|---|---|---|---|
| `github` | one repo's issues, PRs, README | `issue`, `pull_request`, `document` | device-flow OAuth or PAT |
| `google-calendar` | calendar events | `event` | browser OAuth (PKCE) |
| `google-drive` | files, Google docs as text | `document`, `file` | browser OAuth (PKCE) |
| `gmail` | mail matching a search | `email` | browser OAuth (PKCE) |
| `slack` | one channel's history, threads folded | `message`, `thread` | pasted user token (or BYO OAuth) |
| `notion` | pages flattened to markdown | `page` | hosted MCP OAuth or integration secret |
| `brave` | web search results, pages fetched as text | `page` | API key |

## The connect / pull / attach loop

1. **connect**: `wmh connect <service>` runs the service's interactive auth flow (browser
   OAuth, device code, or a pasted token), verifies the credential with a cheap identity call,
   and saves it with your identity stamped as the account. A token injected via
   `WMH_<NAME>_TOKEN` is verified and used as-is; nothing is written to disk.
2. **pull**: `wmh context pull <service> [--target ...] [--query ...] [--since ...]
   [--until ...] [--limit N] [--name <bundle>]` fetches normalized items (capped at `--limit`)
   and writes a named bundle; `--name` defaults to `<service>-<YYYYMMDD-HHMMSS>` (UTC), like
   the quickstart's `github-20260715-120000`. The manifest records the exact query, the pull
   time, and the account, so a bundle is replayable provenance, not a mystery blob.
3. **attach**: `wmh context attach <bundle> --model <name>` renders the bundle to one
   deterministic markdown document and writes it as `knowledge/context-<bundle>.md` in the
   model's artifact; `--root` picks the artifact directory holding the world models (default
   `.wmh`). From the next serve it renders into the env prompt's KNOWLEDGE BASE
   section, subject to the knowledge base's 24,000-char render budget; the command reports
   whether the base still fits, and `--max-chars` trims the bundle by dropping whole items with
   a visible "items omitted" note. Models built without `--knowledge` (and holding no
   `knowledge/` dir) are refused with instructions to opt in.

## GitHub

Pulls one repository's issues, pull requests (a single combined listing, newest first), and its
README into a context bundle.

### Setup

None: the experientiallabs "World Model Harness" GitHub App ships embedded (device flow, no
client secret involved), so `wmh connect github` works out of the box. Access is granted per
repository: after connecting, pick the repositories the connection can reach at
<https://github.com/apps/world-model-harness/installations/new>. The app's permissions are
read and write on contents, issues, and pull requests (write powers upcoming agent features
such as opening PRs; today's `context pull` only reads) plus read on metadata. To use your own
GitHub App instead (Settings > Developer settings > GitHub Apps, with "Enable Device Flow"
checked), override it:

```bash
export WMH_GITHUB_CLIENT_ID=<your GitHub App client id>
```

Alternatively, skip the app entirely with a personal access token:

```bash
export WMH_GITHUB_TOKEN=ghp_...   # classic PAT with repo scope, or a fine-grained token with repo read access
```

### Connect

```bash
wmh connect github
```

Runs the RFC 8628 device flow (permissions are fixed in the GitHub App, so no scopes are
requested): open the printed URL, enter the code, and the credential lands in
`~/.wmh/connectors.toml` with your GitHub login stamped as the account. The command then points
at the app installation page: a token only reaches repositories where the app is installed, so
pick those once and pulls work from then on (tokens expire after eight hours and refresh
automatically).

### Pull

```bash
# The 50 most recently updated issues/PRs, plus the README:
wmh context pull github --target owner/repo --limit 50

# Only items updated since a date:
wmh context pull github --target owner/repo --since 2026-01-01

# Search instead of listing (GitHub search syntax works):
wmh context pull github --target owner/repo --query "label:bug crash"
```

- `--target owner/repo` is required.
- Issues and PRs come from `GET /repos/{owner}/{repo}/issues` (`state=all`, sorted by update
  time, descending, `--since` passed through); rows carrying a `pull_request` key become
  `pull_request` items, the rest `issue` items. Titles read `#<number> <title>`; metadata
  carries the state, label names, author login, and comment count.
- The README is appended as one `document` item when the limit leaves room for it (skipped
  silently when the repo has none).
- `--query` routes through `GET /search/issues` with `repo:owner/repo` prefixed; `--since` /
  `--until` become `updated:>=` / `updated:<=` qualifiers there.

### Caveats

- `--until` only takes effect in search mode; the plain listing API has no upper time bound.
- Hitting the rate limit surfaces a 403 whose message includes the reset time from
  `X-RateLimit-Reset`.
- GitHub answers 404 for private repos the credential cannot see; the error names the repo and
  says to reconnect with an account that has access.
- Issue and PR comment threads are not pulled, only the opening body.

## Google: Calendar, Drive, Gmail

Three read-only connectors share one Google OAuth app: `google-calendar`, `google-drive`, and
`gmail`. Each connector requests exactly one scope and stores its own credential, so v1 asks for
a separate browser consent per service.

### Setup

1. In Google Cloud Console, create an OAuth client and enable the APIs you need (Google
   Calendar API, Google Drive API, Gmail API).
2. Point the harness at your OAuth client (Google requires the client secret even for installed
   apps; it is not treated as confidential in that setup):

```bash
export WMH_GOOGLE_CLIENT_ID="...apps.googleusercontent.com"
export WMH_GOOGLE_CLIENT_SECRET="GOCSPX-..."
```

### Connect

```bash
wmh connect google-calendar   # scope: calendar.readonly
wmh connect google-drive      # scope: drive.readonly
wmh connect gmail             # scope: gmail.readonly
```

Each command opens a browser consent (loopback OAuth with PKCE) and verifies the credential with
a cheap identity call (primary calendar summary, Drive user, Gmail address). Tokens land in
`~/.wmh/connectors.toml`, one table per connector, and access tokens auto-refresh before every
pull.

### Pull

```bash
# Calendar: target is a calendar id (default "primary"); window defaults to now-30d .. now+60d
wmh context pull google-calendar --since 2026-07-01 --until 2026-09-01 --limit 50
wmh context pull google-calendar --target team@group.calendar.google.com

# Drive: free text becomes a fullText search; target is a folder id; trashed files are excluded
wmh context pull google-drive --query "quarterly roadmap"
wmh context pull google-drive --target 1AbCdEfGhIjK --since 2026-06-01

# Gmail: query passes through as Gmail search syntax; since/until map to after:/before:
wmh context pull gmail --query "from:alerts@example.com is:unread" --since 2026-06-01
```

What each pull produces:

- `google-calendar`: `event` items. Title is the event summary; the body is a readable block
  (When/Location/Attendees, then the description); `url` is the event's htmlLink.
- `google-drive`: `document` items for readable files, `file` items for other binaries (empty
  body, mimeType in metadata). Google Docs and Slides are exported as plain text, Sheets as
  CSV (the first sheet only); `text/*` and JSON blobs are downloaded. Fetched content is
  capped at 200000 characters per file with a visible truncation marker. `--since`/`--until`
  become `modifiedTime` bounds in the Drive query.
- `gmail`: `email` items. Title is the Subject; the body is a From/To/Date header block plus
  the text/plain part (recursive multipart walk, base64url decoded), falling back to
  tag-stripped HTML; `created_at` comes from internalDate; `url` deep-links into the mailbox.

### Caveats

- Separate consent per service is a v1 decision: each connector keeps its own token, so
  revoking or reconnecting one never disturbs the others.
- If Google revokes the refresh token (`invalid_grant`), the pull fails with an error telling
  you to re-run `wmh connect <name>`.
- Per-file Drive content fetches that fail (for example export size limits) degrade to an empty
  body with `fetch_error` recorded in the item metadata instead of aborting the pull.
- `WMH_GOOGLE_CALENDAR_TOKEN` / `WMH_GOOGLE_DRIVE_TOKEN` / `WMH_GMAIL_TOKEN` inject a plain
  bearer token (no file, no refresh), useful for CI and headless runs.

## Slack

Pulls one channel's history. Each thread (parent plus all replies) becomes a single `thread`
item; standalone messages become one `message` item each. The connector uses a **user token**
(`xoxp-...`), so it sees exactly what the installing user can see.

### Setup (default: workspace app + pasted token)

1. Open https://api.slack.com/apps, click **Create New App**, choose **From an app manifest**,
   pick your workspace, and paste:

   ```yaml
   display_information:
     name: wmh context puller
     description: Read-only channel history for wmh context bundles
   oauth_config:
     scopes:
       user:
         - channels:history
         - channels:read
         - groups:history
         - groups:read
         - users:read
   settings:
     org_deploy_enabled: false
     socket_mode_enabled: false
   ```

2. Click **Install to Workspace** and approve.
3. Copy the **User OAuth Token** (`xoxp-...`) from the app's *OAuth & Permissions* page.

### Connect

```bash
wmh connect slack            # prompts for the pasted xoxp- token, verifies via auth.test
export WMH_SLACK_TOKEN=xoxp-...   # or inject the token via env; nothing is written to disk
```

Verification calls `auth.test` and reports the identity as `user @ team`.

### Pull

```bash
wmh context pull slack --target "#general" --limit 50
wmh context pull slack --target C0123456789 --since 2026-07-01
```

- `target`: channel name (with or without `#`) or channel id. Unknown names fail with a sample
  of the channels the token can actually see.
- `since` / `until`: ISO-8601 date or datetime, mapped to the history `oldest`/`latest` epoch
  bounds.
- `limit`: caps the item count; a thread with all its replies counts as one item.

### Caveats

- **Rate limits and tier caps.** On HTTP 429 the pull fails immediately with the `Retry-After`
  wait (it never sleep-retries). Slack caps commercially distributed non-Marketplace apps at
  roughly 1 request/min on the conversation-history endpoints; a workspace-internal app
  installed from the manifest above (the pasted-token path) is not subject to that cap.
- **Private channels** require the `groups:*` user scopes and the installing user to be a
  member of the channel.
- **BYO OAuth app (advanced).** Setting `WMH_SLACK_CLIENT_ID` and `WMH_SLACK_CLIENT_SECRET`
  switches `wmh connect slack` to the browser flow. It sends the user scopes via Slack's
  `user_scope` parameter and unwraps the token Slack nests at `authed_user.access_token`. Slack
  only accepts **HTTPS** redirect URLs while the built-in flow serves a plain
  `http://127.0.0.1` callback, so this path needs an app/relay configured for that; prefer the
  pasted token.
- Permalinks (`https://<team>.slack.com/archives/...`) and the `account` identity come from the
  team domain captured at connect time. Tokens injected purely via `WMH_SLACK_TOKEN` (never run
  through `wmh connect slack`) produce items with `url = null`.
- Bodies render as `[@name at HH:MM] text` lines (UTC), with `<@U...>` mention encodings
  replaced by display names from one cached `users.list` pass.

## Notion

Pulls Notion pages as `page` context items: title, page body flattened to markdown (headings,
lists, to-dos, code, quotes, callouts; one level of nested blocks), canonical URL, and
created/last-edited timestamps.

### Setup

The default flow talks to Notion's hosted MCP server and needs the `connectors` extra:

```bash
uv sync --extra connectors   # or: pip install 'world-model-harness[connectors]'
```

No OAuth app registration is required: the connector uses MCP dynamic client registration
against `https://mcp.notion.com/mcp`.

### Connect

```bash
wmh connect notion
```

Precedence at connect time:

1. `$WMH_NOTION_TOKEN` set: the env token is verified against the REST API and used as-is
   (nothing is written to disk; ideal for CI and headless runs).
2. A pasted internal-integration secret (`ntn_...` / `secret_...`) at the prompt: verified via
   `GET /v1/users/me` and stored as a token credential. Create one under
   notion.so/my-integrations and share the target pages with it.
3. Press Enter at the prompt: browser OAuth via the Notion MCP server. Approve access in the
   opened tab; tokens land in `~/.wmh/connectors.toml` and refresh automatically on later runs.

`wmh connect` verification returns `Notion MCP (<n> tools)` for the OAuth path, or
`<bot name> (<workspace>)` for the token path.

### Pull

```bash
wmh context pull notion --query "quarterly roadmap" --limit 25
wmh context pull notion --query retro --since 2026-06-01
```

- OAuth credential: pulls go through the MCP server's search and fetch tools.
- Token credential: pulls use the REST API (`POST /v1/search` sorted by `last_edited_time`
  descending, then `GET /v1/blocks/{id}/children` per page).

### Caveats

- `--since`/`--until` are applied client-side on `last_edited_time` (Notion's search API has no
  time filter); the descending sort keeps this cheap.
- `--target` is ignored: Notion search is workspace-wide. Scope with `--query` instead.
- An internal integration only sees pages explicitly shared with it; the MCP OAuth grant sees
  whatever the authorizing user approved.
- Block flattening recurses one level into nested blocks; deeper nesting is omitted.
- Databases are skipped: only page objects are pulled.
- A 401/403 means the credential is stale: re-run `wmh connect notion` (or refresh
  `$WMH_NOTION_TOKEN`).

## Brave Search

Pulls live web search results as `page` context items: one item per result, with the page
content fetched and stripped to readable text as the body (degrading to the search snippet when
the fetch fails), the result's rank and snippet in metadata, and `created_at` from the page age
when Brave reports one.

### Setup

Get a free API key at https://api-dashboard.search.brave.com/ (the same key the engine's
`--grounder brave` option uses), then either export it or paste it at connect time:

```bash
export BRAVE_SEARCH_API_KEY=BSA...   # already deployed in our environments
```

The generic `WMH_BRAVE_TOKEN` override also works and takes precedence over
`BRAVE_SEARCH_API_KEY` when both are set. Either env var skips the credential file entirely.

### Connect

```bash
wmh connect brave    # uses the env key when set (nothing written to disk), else prompts
```

Verification runs one minimal search (`q=wmh`, `count=1`) and reports `Brave Search (key
valid)`.

### Pull

```bash
wmh context pull brave --query "world model harness"
wmh context pull brave --query "release notes" --target example.com --limit 20
wmh context pull brave --query wmh --since 2026-06-01 --until 2026-07-01
```

- `--query` is required: it is the web search itself.
- `--target <domain>` scopes results to one site (prepended as a `site:` filter).
- `--since`/`--until` map to Brave's `freshness` range (`YYYY-MM-DDtoYYYY-MM-DD`); `--since`
  alone ranges to today, and `--until` alone is dropped (Brave has no until-only form).
- Results are capped at 50 per pull regardless of `--limit` (offset pagination in pages of
  20): Brave's result relevance degrades sharply at deep offsets, so more is noise.

### Caveats

- **Rate limits.** The free tier allows 1 request/second; on HTTP 429 the pull fails
  immediately quoting the `Retry-After` wait (it never sleep-retries).
- Each result's page is fetched through the SSRF-guarded fetcher shared with the grounding
  engine (http(s) only, public addresses only, redirects re-checked); a failed or refused
  fetch degrades that item's body to the search snippet with `fetch_error` recorded in
  metadata, never aborting the pull. Fetched text is capped at 200000 characters with a
  visible truncation marker.
- A 401/403 means the key is bad or the subscription lapsed: check the key at
  https://api-dashboard.search.brave.com/ and re-run `wmh connect brave`.

## Shared OAuth apps

OAuth endpoint and scope configuration per provider is embedded in `wmh/connect/apps.py`
(`EMBEDDED_APPS`), together with the registered client ids where they exist: GitHub ships
embedded today; Google and Slack are pending registration, so those OAuth paths need a
bring-your-own client via environment overrides meanwhile. A client id is a public identifier
for a native app (RFC 8252): flows are secured by PKCE or the device grant, and no client secret
ever ships in the repo. Env overrides always win over whatever is embedded:

```bash
export WMH_<NAME>_CLIENT_ID=...       # e.g. WMH_GITHUB_CLIENT_ID, WMH_GOOGLE_CLIENT_ID
export WMH_<NAME>_CLIENT_SECRET=...   # only where the provider requires one
```

`<NAME>` is the app name upper-cased with hyphens as underscores. Set-but-empty values are
treated as unset. The github/google/slack connectors resolve their app through this layering on
every call; Notion never needs one (MCP dynamic client registration), and Brave uses a plain
API key, no OAuth app at all.

### What the maintainer registers per provider

- **GitHub**: one GitHub App with **Enable Device Flow** checked, installable by **any
  account**, webhook inactive, repository permissions contents/issues/pull requests read and
  write plus metadata read (write powers the upcoming agent PR features; installation is
  repo-scoped, so users control the blast radius). Only the client id is embedded; the device
  flow needs no client secret, so nothing confidential ships.
- **Google**: a Google Cloud project with the Calendar, Drive, and Gmail APIs enabled, plus an
  OAuth client. The consent screen must request exactly the three read-only scopes
  (`calendar.readonly`, `drive.readonly`, `gmail.readonly`). Note `gmail.readonly` and
  `drive.readonly` are **restricted** scopes: shipping the app to the public requires Google's
  OAuth verification and, for Gmail, a CASA security assessment; until that clears, the
  embedded client stays in testing mode and users bring their own client. Google requires the
  client secret even for installed apps (it is not treated as confidential there), so embedding
  it is expected.
- **Slack**: one distributed Slack app created from a manifest, installed once, requesting the
  user scopes below. Distribution must be enabled (Manage Distribution) so any workspace can
  install it; note the non-Marketplace history rate cap in the Slack caveats above, and that
  Slack requires HTTPS redirect URLs, so the embedded app needs a hosted redirect relay before
  the browser flow works out of the box.

  ```json
  {
    "display_information": {
      "name": "wmh context puller",
      "description": "Read-only channel history for wmh context bundles"
    },
    "oauth_config": {
      "redirect_urls": ["https://<hosted-relay>/slack/callback"],
      "scopes": {
        "user": [
          "channels:history",
          "channels:read",
          "groups:history",
          "groups:read",
          "users:read"
        ]
      }
    },
    "settings": {
      "org_deploy_enabled": false,
      "socket_mode_enabled": false
    }
  }
  ```

- **Notion**: nothing. The hosted MCP server registers clients dynamically (RFC 7591) and
  discovers endpoints via protected-resource metadata.

## The context item contract (what a connector produces)

A pull returns `list[ContextItem]`; the bundle store persists them verbatim, one JSON object per
`items.jsonl` line:

- `id`: stable identifier within the service (issue number, page id, message ts).
- `source`: the connector name.
- `kind`: `document | page | issue | pull_request | message | thread | email | event | file`.
- `title` / `body`: what `render_markdown` turns into a `## title` section with a fact line.
- `url`, `created_at`, `updated_at`: provenance (ISO-8601 timestamps) when the service has them.
- `metadata`: connector-specific extras (labels, channel, mimeType) as arbitrary JSON.

## How the pieces fit

```
  wmh connect <service> ──▶ ContextConnector.connect ──▶ ConnectorAuth ──▶ ~/.wmh/connectors.toml
  wmh context pull      ──▶ ContextConnector.pull    ──▶ list[ContextItem] ──▶ .wmh/context/<bundle>/
  wmh context attach    ──▶ render_markdown          ──▶ models/<m>/knowledge/context-<bundle>.md
```

- `wmh/connect/connector.py`: the `ContextConnector` protocol, the `ConnectUI` callback bundle,
  and the registry (`register_connector`, `get_connector`, `list_connectors`).
- `wmh/connect/oauth.py`: provider-agnostic OAuth building blocks (PKCE, loopback flow, device
  flow, refresh); connectors compose these instead of reimplementing OAuth.
- `wmh/connect/apps.py`: the shared OAuth app registry plus the env overrides above.
- `wmh/connect/credentials.py`: the user-global credential store and `WMH_<NAME>_TOKEN`.
- `wmh/connect/store.py`: bundle persistence (`ContextStore`) and `render_markdown`.
- `wmh/cli/connect_cmds.py`: the `wmh connect` / `wmh context` commands.

## Add a connector in ~60 lines

A connector owns one service end to end. Token-auth services need no OAuth at all:

```python
# wmh/connect/myservice.py
"""MyService context connector: pasted-token auth, notes pulled as documents."""

from __future__ import annotations

import httpx

from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.types import (
    ConnectError, ConnectorAuth, ContextItem, ItemKind, PullQuery, opt_str,
)


class MyServiceConnector:
    name = "myservice"
    label = "MyService"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport  # tests inject httpx.MockTransport; None = real network

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        token = ui.prompt_secret("MyService API token").strip()
        if not token:
            raise ConnectError("no token entered; create one at myservice.test/settings")
        auth = ConnectorAuth(kind="token", access_token=token)
        return auth.model_copy(update={"account": self.verify(auth)})

    def verify(self, auth: ConnectorAuth) -> str:
        with self._client(auth) as client:
            response = client.get("/me")
        if response.status_code != 200:
            raise ConnectError(
                f"myservice rejected the credential (HTTP {response.status_code}); "
                "run `wmh connect myservice` to re-connect"
            )
        return opt_str(response.json().get("login")) or "MyService user"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        with self._client(auth) as client:
            response = client.get("/notes", params={"limit": query.limit})
        return [
            ContextItem(
                id=str(note["id"]), source=self.name, kind=ItemKind.DOCUMENT,
                title=note["title"], body=note["text"], url=opt_str(note.get("url")),
            )
            for note in response.json()[: query.limit]
        ]

    def _client(self, auth: ConnectorAuth) -> httpx.Client:
        return httpx.Client(
            base_url="https://api.myservice.test",
            headers={"Authorization": f"Bearer {auth.access_token}"},
            transport=self._transport,
        )


register_connector(MyServiceConnector())
```

Then import it in `wmh/connect/__init__.py` (for registration on package import), add an inline
`myservice_test.py` driving it through `httpx.MockTransport` (no network), and `wmh connect
myservice` / `wmh context pull myservice` pick it up. OAuth services build `connect` from
`run_loopback_flow` / `run_device_flow` plus an `EMBEDDED_APPS` entry; mirror the four bundled
connectors for reference.

## Conventions

Connectors live in `wmh/connect/`, are typed (no `Any`/bare `dict`; use `wmh.core.types`
`JsonValue`/`JsonObject` for vendor JSON), never print (the CLI's `ConnectUI` carries all
presentation), and are tested inline with `httpx.MockTransport` fixtures, never the network.
Anything that talks HTTP accepts `transport: httpx.BaseTransport | None = None`. Heavy SDKs are
optional extras imported lazily (the Notion connector's `mcp` SDK is the `connectors` extra).
Every error says what went wrong and what to do next. Knowledge attachments are written as
`context-<bundle>.md`, so they can never collide with the reserved knowledge file names
(`rules.md`, `entities.md`, `schemas.md`, `learned.md`, `grounded.md`).
