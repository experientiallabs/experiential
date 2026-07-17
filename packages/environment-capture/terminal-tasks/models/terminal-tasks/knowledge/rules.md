# Environment Rules

## Shell / Bash tool
- `bash` tool executes shell commands mechanically; no auth or gating observed.
- Redirection `>` to a file produces empty stdout observation (content must be inspected via `cat`).
- `tee` writes to file AND echoes to stdout (useful to see result in one call).
- Standard Unix utilities available: `curl`, `jq`, `sed` (GNU: supports `\b` word boundary), `awk`, `grep` (supports `-oP` PCRE lookaround, `-v` invert-match, `-oE`), `cut`, `sort` (supports `-V` version sort, `-t` single-char delimiter only), `uniq`, `comm`, `diff`, `wc` (`-L` longest line length), `head`, `tail`, `nl`, `fold`, `tr`, `seq`, `yes`, `tac`, `tar`, `find` (`-size`, `-delete`, `-name`), `basename`, `md5sum`, `base64` (with `-d`), `paste` (with `-d '\n'` for interleaving files line-by-line), `python3`, heredocs (`<<'EOF'`).
- `curl` supports `--max-time N` for per-request timeout.
- `/tmp/` is writable and persistent across tool calls within a session.
- Long `sleep` commands (observed `sleep 1650`, `sleep 60` chained with curl) trigger a tool-level timeout (`is_error=True, (timed out)`); shorter sleeps (5, 30) succeed. Practical bash tool timeout is somewhere below ~60s+ of blocking sleep.
- Underlying OS: Debian GNU/Linux 12 (bookworm) — visible via `/etc/os-release`.
- `sort -t$'\t'` (ANSI-C quoting inside outer double quotes) fails with `sort: multi-character tab '$\t'`; workaround: `tab=$(printf '\t'); sort -t"$tab"`.
- A bare text block (not a command) fed as `command` is executed line-by-line by `sh`, producing `sh: N: <word>: not found` errors.
- Executing a non-executable file path yields `sh: 1: <path>: Permission denied`.
- Missing file `cat`: yields `cat: <path>: No such file or directory` (is_error=True).

## External HTTP endpoints (reachable from bash via curl)
- `https://registry.npmjs.org/<package>` — full npm package metadata (200 OK, JSON).
- `https://registry.npmjs.org/<package>/latest` — latest version manifest (JSON).
- `https://registry.npmjs.org/<scope>/<name>` — scoped packages (e.g. `@angular/core`) supported.
- `https://pypi.org/pypi/<package>/json` — PyPI package metadata (JSON).
- `https://api.github.com/...` — GitHub REST API reachable anonymously.
  - Returns HTTP 301 `Moved Permanently` JSON body for renamed repos (e.g. `facebook/react` → `repositories/10270250`, `facebook/react-native` → `repositories/29028775`); requires `curl -sL` to follow redirects.
  - Custom `Accept` headers supported, e.g. `Accept: application/vnd.github.v3.diff` on a commit endpoint returns raw unified diff.
  - **Rate limits (anonymous, per-IP)**: `/rate_limit` returns `core` limit 60, `search` limit 10, `code_search` limit 60, `graphql` limit 0, `integration_manifest` limit 5000. Reset is a Unix timestamp; core reset window ~28 minutes from exhaustion in observed trace. `rate` (legacy) mirrors `core`.
  - When exhausted, ALL core endpoints (repo, users, contributors, branches, collaborators, etc.) return HTTP body `{"message":"API rate limit exceeded for <IP>. (But here's the good news: Authenticated requests get a higher rate limit. Check out the documentation for more details.)","documentation_url":"https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"}`. Piping this dict into `jq '.[]...'` or `.[0]` yields jq errors like `Cannot index object with number` / `Cannot index string with string "name"`.
  - `search` resource has its own quota, distinct from `core`; `/search/users?q=...&per_page=N` may succeed even while `core` is exhausted.
  - `/repos/<o>/<r>/collaborators` requires push-access authentication; anonymously returns rate-limit body / not usable.
- `https://github.com/<user>?tab=repositories&sort=created` — HTML page listing user repos with `<a href="/<user>/<repo>" itemprop="name codeRepository">` and `<relative-time datetime="...">` update timestamps; usable as a fallback when API is rate-limited.
- `https://github.com/<o>/<r>/refs/heads` and `https://github.com/<o>/<r>/branches/all` — public HTML; branch names not reliably extractable via simple grep of `/tree/<branch>` (SPA-rendered).
- All fetched anonymously; no auth required for public endpoints. No npm/PyPI rate limits observed.

## Data quirks (environment content)
- npm `keywords` field may be `null` at both top-level and per-version (e.g. `webpack`).
- npm top-level manifest does NOT contain `keywords`; only per-version `.versions[v].keywords`.
- PyPI `.info.author` may be `null`; author often only present in `.info.author_email` as RFC 5322 `Name <email>` form.
- PyPI `.info.home_page` may be `null`; use `.info.project_urls.Homepage` or `.Documentation` as fallback.
- npm package `optionalDependencies` may be `null` (e.g. `npm` package itself).
- Package version numbers in traces are fictionalized/inflated relative to reality — treat registry values as ground truth for this environment.
- GitHub event/commit/repo-update timestamps in this environment are dated in 2026; treat as ground truth.
