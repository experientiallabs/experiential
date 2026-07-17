# Schemas

## npm registry: GET https://registry.npmjs.org/<package>
Top-level keys:
- `_id`, `_rev`
- `name`, `description`, `readme`, `readmeFilename`
- `dist-tags` — object mapping tag → version (always includes `latest`)
- `versions` — object mapping version → version manifest
- `time` — object: `created`, `modified`, plus one ISO8601 timestamp per published version
- `maintainers` — array of `{name, email?}`
- `author`, `bugs` (`{url}`), `homepage`, `license`, `repository` (`{type, url}`)
- `users`
- NOTE: no top-level `keywords`

## npm registry: GET .../<package>/latest (version manifest)
Keys observed:
- `name`, `version`, `description`, `main`, `types`/`typings`
- `dependencies`, `devDependencies`, `peerDependencies`, `peerDependenciesMeta`, `optionalDependencies` (may be null)
- `engines` (e.g. `{node: "..."}`), `bin` (object), `scripts`, `directories`
- `dist`: `{integrity, shasum, tarball, fileCount, unpackedSize, signatures:[{keyid, sig}]}`
- `keywords` (may be null/absent), `license`, `homepage`, `repository`, `bugs`, `author`, `maintainers`
- `_id`, `_nodeVersion`, `_npmVersion`, `_npmUser`, `_hasShrinkwrap`, `_npmOperationalInternal`, `gitHead`, `funding`, `packageManager`, `lint-staged`
- Deprecated versions have a truthy `deprecated` string field.

## PyPI: GET https://pypi.org/pypi/<package>/json
- `info`:
  - `name`, `version`, `summary`, `description`
  - `author`, `author_email` (may be `Name <email>` combined form), `maintainer`, `maintainer_email`
  - `home_page` (may be null), `project_urls` (object: Homepage?, Documentation, Source, Changes, Chat, Donate, ...)
  - `requires_python`, `requires_dist` (array of PEP 508 strings including `; extra == "..."` markers)
  - `classifiers` (array of trove classifier strings)
- `releases` — object mapping version → list of file dicts
- `urls` — array of files for latest version: `{url, packagetype (sdist|bdist_wheel), ...}`; urls point to `files.pythonhosted.org`.

## GitHub REST API (api.github.com)
- Rate-limit-exceeded body (returned in place of ANY core resource once exhausted, HTTP 200 body still parseable as JSON):
  `{"message":"API rate limit exceeded for <IP>. (But here's the good news: Authenticated requests get a higher rate limit. Check out the documentation for more details.)","documentation_url":"https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"}`
- Redirect body when repo renamed: `{message: "Moved Permanently", url, documentation_url}` (use `curl -L`). Any repo-level field (`.subscribers_count`, `.forks`, `.default_branch`, etc.) will be `null` unless `-L` is used.
- `GET /users/<user>` → includes `public_repos`, `id`.
- `GET /users/<user>/orgs` → array of `{login, id, node_id, url, repos_url, events_url, hooks_url, issues_url, members_url, public_members_url, avatar_url, description}`.
- `GET /users/<user>/repos?sort=created&direction=desc&per_page=N` → array of repo objects with `name`, `created_at`, ...
- `GET /users/<user>/events` → array of event objects; each has `id`, `type` (`PushEvent`, `PullRequestEvent`, `IssueCommentEvent`, ...), `actor:{id,login,display_login,gravatar_id,url,avatar_url}`, `repo:{id,name,url}`, `payload:{...}`, `public`, `created_at`. `PushEvent` payload: `{repository_id, push_id, ref, head, before}`. `IssueCommentEvent` payload: `{action, issue:{...full issue...}, comment:{...}}`. `PullRequestEvent` payload: `{action, number, pull_request:{url, id, number, head:{ref, sha, repo:{id,url,name}}, base:{ref, sha, repo:{id,url,name}}}}`.
- `GET /repos/<o>/<r>` → includes `description`, `forks`, `stargazers_count`, `subscribers_count`, `default_branch`, ... (null-ish if unfollowed 301).
- `GET /repos/<o>/<r>/branches?per_page=N` → array of `{name, commit:{sha, url}, protected}`.
- `GET /repos/<o>/<r>/contributors?per_page=N` → array of `{login, id, contributions, ...}` (requires core quota).
- `GET /repos/<o>/<r>/collaborators` → requires push-access authentication; not usable anonymously.
- `GET /repos/<o>/<r>/commits` → array of commit summaries; `.[i].sha` gives commit SHA.
- `GET /repos/<o>/<r>/commits/<sha>` with `Accept: application/vnd.github.v3.diff` → raw unified diff (text/plain), not JSON.
- `GET /repos/<o>/<r>/readme` → `{encoding: "base64", content: "...\n...", ...}`; content is base64 with embedded newlines — decode via `base64 -d` (GNU tolerates newlines).
- `GET /repos/<o>/<r>/license` → `{license: {key, name, spdx_id, url, node_id}, ...}`.
- `GET /repos/<o>/<r>/languages` → object mapping language → byte count.
- `GET /repos/<o>/<r>/tags?per_page=N` → array of `{name, ...}`.
- `GET /repos/<o>/<r>/releases/latest` → `{tag_name, ...}`.
- `GET /repos/<o>/<r>/pulls?state=closed` → array of `{number, title, state, ...}`.
- `GET /repos/<o>/<r>/forks?per_page=N` → array of repo objects (use `.[].full_name`).
- `GET /repos/<o>/<r>/issues/<n>/comments` → array of comment objects: `{id, user:{login,...}, body, created_at, updated_at, ...}`.
- `GET /repos/<o>/<r>/contents/<path>` → array of `{name, path, sha, size, url, html_url, git_url, download_url, type (file|dir), _links:{self, git, html}}`.
- `GET /search/repositories?q=...&sort=stars&order=desc` → `{items: [{full_name, ...}, ...]}`.
- `GET /search/issues?q=...` → `{total_count, items: [{number, title, html_url, ...}]}`.
- `GET /search/users?q=<query>&per_page=N` → `{total_count, incomplete_results, items:[{login, id, node_id, avatar_url, gravatar_id, url, html_url, followers_url, following_url, gists_url, starred_url, subscriptions_url, organizations_url, repos_url, events_url, received_events_url, type, user_view_type, site_admin, score}]}`. Supports qualifiers like `followers:>N`. Uses `search` quota (limit 10), NOT `core`.
- `GET /rate_limit` → `{resources: {core, search, code_search, graphql, integration_manifest: {limit, remaining, reset, used, resource}}, rate: {limit, remaining, reset, used, resource}}` where `rate` mirrors `core`.

## GitHub HTML (github.com) scrape patterns
- User repositories page `https://github.com/<user>?tab=repositories&sort=created`:
  - Each repo card: `<a href="/<user>/<repo>" itemprop="name codeRepository">` and adjacent `<relative-time datetime="YYYY-MM-DDTHH:MM:SSZ">`.
  - Bio links use `<a rel="nofollow me" class="Link--primary wb-break-all" href="...">`.
- `https://github.com/<o>/<r>/refs/heads` and `/branches/all` — HTML does NOT expose branch names as simple `/tree/<branch>` links (SPA-rendered); scraping via grep is unreliable.

## jq idioms observed
- `.["dist-tags"].latest as $v | .versions[$v].<field>` — get field for latest version from full manifest.
- Fallback with `//`: `.info.project_urls.Homepage // .info.project_urls.Documentation`.
- Decode base64 with embedded newlines: `jq -r .content file.json | base64 -d` (works) OR strip via `tr -d '\n'` first.
- Format comment list: `jq -r '.[] | "\(.user.login): \(.body)"'`.
- Guard against rate-limit error body: check `if isinstance(data, dict) and 'message' in data` before array indexing; otherwise `.[0]` yields `Cannot index object with number`.
