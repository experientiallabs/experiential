# Connector fetch library (`wmh.connect`)

`wmh.connect` fetches content from the tools a team already uses (GitHub, Google, Slack, Notion,
plus Brave web search) and normalizes every result into one vendor-agnostic shape,
`ContextItem`. It is a library, not a CLI: a host supplies a per-service access token and calls a
connector's `pull`. Token and OAuth acquisition are the caller's responsibility; the platform
supplies tokens.

## Call site

```python
from wmh.connect import ConnectorAuth, PullQuery, get_connector

connector = get_connector("github")
items = connector.pull(
    ConnectorAuth(kind="token", access_token=token),
    PullQuery(target="owner/repo", query="is:open", limit=20),
)
for item in items:
    print(item.kind, item.title, item.url)
```

`get_connector(name)` looks a connector up by name; `list_connectors()` returns every registered
name, sorted. Both come from `wmh.connect`. Registration happens on import, so importing the
package is enough:

```python
import wmh.connect

wmh.connect.list_connectors()
# ['brave', 'github', 'gmail', 'google-calendar', 'google-drive', 'notion', 'slack']
```

## `ConnectorAuth`: the caller-supplied credential

`pull` takes a `ConnectorAuth`. The host holds the tokens (from its own OAuth flows or a stored
integration secret) and passes them in. The library never acquires tokens on its own during a
`pull`.

- `kind`: `"token"` for a pasted or host-injected bearer token, `"oauth"` for a browser or
  device OAuth grant.
- `access_token`: the bearer credential API calls send (required).
- `refresh_token`, `expires_at`, `scopes`, `account`, `extra`: optional OAuth metadata; `extra`
  carries connector-specific values (for example a Slack team id) as JSON.

For most host integrations `ConnectorAuth(kind="token", access_token=...)` is all that is needed.

## `PullQuery`: what to pull

Every connector's `pull` accepts the same `PullQuery`:

- `target`: the service-specific container (a repo `"owner/name"`, a Slack channel name, a
  calendar id, a Drive folder id).
- `query`: a free-text or service search-syntax filter.
- `since` / `until`: ISO-8601 lower and upper bounds on item time.
- `limit`: the maximum number of items to fetch (default 100; connectors cap at this).

## `ContextItem`: the normalized result

`pull` returns `list[ContextItem]`. Every connector maps its raw vendor payload into this shape,
so downstream code never touches vendor JSON:

- `id`: stable identifier within the source service (issue number, page id, message timestamp).
- `source`: the connector name that produced the item (for example `"github"`).
- `kind`: one of the `ItemKind` values: `document`, `page`, `issue`, `pull_request`, `message`,
  `thread`, `email`, `event`, `file`.
- `title`: a short human title.
- `body`: the content itself, plain text or markdown.
- `url`: a canonical link back to the item, when the service has one.
- `created_at` / `updated_at`: ISO-8601 timestamps, when known.
- `metadata`: connector-specific extras (labels, authors, channel ids) as JSON.

## Connectors and their targeting

| Connector | What `pull` returns | Item kinds | `PullQuery` fields it reads |
|---|---|---|---|
| `github` | one repo's issues, PRs, README | `issue`, `pull_request`, `document` | `target` (`owner/repo`), `query`, `limit` |
| `google-calendar` | calendar events | `event` | `target` (calendar id), `since`, `until`, `limit` |
| `google-drive` | files, Google docs as text | `document`, `file` | `target` (folder id), `query`, `limit` |
| `gmail` | mail matching a search | `email` | `query`, `limit` |
| `slack` | one channel's history, threads folded | `message`, `thread` | `target` (channel), `since`, `until`, `limit` |
| `notion` | pages flattened to markdown | `page` | `query`, `limit` |
| `brave` | web search results, pages fetched as text | `page` | `query`, `limit` |

## Errors

A failed pull raises `ConnectError` (bad or expired auth, unreachable host, a rejected request)
with a message that says what went wrong. Import it from `wmh.connect`.

## Optional dependency: `connectors` extra

The Notion connector's remote-MCP path uses the `mcp` client SDK, packaged as the optional
`connectors` extra (`pip install "world-model-harness[connectors]"`). The SDK is imported lazily
inside that path only, so `import wmh.connect` and every other connector's `pull` work without
the extra installed. Notion also accepts a pasted integration secret, which uses the REST API and
needs no extra.
