# GitLab CI

The repository lives at `gitlab.com/experiential-labs-group/experiential`.
`.gitlab-ci.yml` holds only the pipeline-wide settings and includes every file
in `.gitlab/ci/`. Each file there owns one pipeline area and is named after it;
the areas are `python-package` and `gate`. `.gitlab/ci/_templates.yml` holds the
shared hidden keys: pinned tool versions and install snippets. The gate and
gateway-latency workflows under `.github/workflows/` have no runner; the `gate`
area runs the former's Linux jobs on demand.

## Pipelines and triggers

`workflow:rules` runs merge request pipelines for branches with an open merge
request and branch pipelines otherwise, and admits tag pushes and manual
(`web`, `api`) pipelines. Because one pipeline can come from any of these
sources, **every job has `rules:`** that name the sources it runs for. A job
without rules would run in every pipeline.

`python-package` runs for two sources only:

| Runs on | Rule |
| --- | --- |
| Protected release tags | `$CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+(rc\d+)?$/ && $CI_COMMIT_REF_PROTECTED == "true" && $CI_PIPELINE_SOURCE == "push"` |
| Manual runs | `$CI_PIPELINE_SOURCE =~ /^(web\|api)$/ && $RUN_WORKFLOW == "python-package"` |

`gate` runs for manual runs only:

| Runs on | Rule |
| --- | --- |
| Manual runs | `$CI_PIPELINE_SOURCE =~ /^(web\|api)$/ && $RUN_WORKFLOW == "gate"` |

It mirrors the Linux jobs of `.github/workflows/gate.yml`: `gate-rust` (cargo
fmt, clippy with `-D warnings`, cargo test, the real-bridge cache test, and the
native parity tests), `gate-lint` (ruff check, ruff format, ty, and the Tinker
adapter tests), and five `gate-tests` shards of the non-live pytest suite, one
xdist worker per runner CPU. Two
known-red tests are deselected by node id in the `rest` shard, each with a
comment naming why; every other test is strict. Run it on a branch before
handing off a change that has no other test pipeline.

Merge requests and pushes to `main` run nothing. Every job uses a GitLab-hosted
runner, which bills the group's compute minutes, and macOS minutes cost six
times the small Linux rate.

A manual run sets `RUN_WORKFLOW` to the area (`python-package` or `gate`) and,
for `python-package`, the area's inputs in the **Run pipeline** form:

| Variable | Choices | Use |
| --- | --- | --- |
| `PYTHON_PACKAGE_SOURCE` | `latest-release` (default), `selected-ref` | Rebuild the latest release's tag, or the ref the pipeline runs on |
| `PYTHON_PACKAGE_PUBLISH` | `false` (default), `true` | Run the release checks and publish to PyPI |

## Jobs

- **Images.** Linux and macOS jobs set `image:`. The hosted Windows runner
  cannot: it runs a fixed Windows Server 2022 VM with PowerShell.
- **Tools.** uv, maturin and the Rust toolchain come from `UV_VERSION`,
  `MATURIN_VERSION` and `RUST_TOOLCHAIN` in `.tools`. Container images are
  pinned by digest, and every install reads `uv.lock` and `Cargo.lock` as
  committed (`--locked`).
- **Dependency audit.** `gate-dependency-audit` runs OSV-Scanner
  (`OSV_SCANNER_VERSION`, checksum-verified) over `uv.lock` and the native
  crate's `Cargo.lock`, and fails on any known advisory. An accepted advisory
  is one `IgnoredVulns` entry in `.gitlab/ci/osv-scanner.toml`, by ID, with a
  reason and an `ignoreUntil` date; new advisories always fail.
- **Ordering.** Jobs order with `needs:`; a job without dependencies sets
  `needs: []` so it starts at once. Values pass between jobs through
  `artifacts:reports:dotenv`, and files through `artifacts:`.
- **Caches.** The windows-x86_64 wheel, the slowest job, keeps the Rust
  toolchain, crate downloads, and compiled dependencies in the job cache under
  `.ci-cache/`, which `.gitignore` lists. uv's Python installs stay out of it,
  because the runner cannot archive the directory links uv creates there.
  Pipelines for protected refs, or started by a Maintainer or Owner, use the
  protected cache, and every other pipeline a separate one, so a Developer's
  merge request cannot seed what a release builds with. The first run after
  the cache is cleared rebuilds everything.

## Runners

| Runner tag | Used by | GitLab plan |
| --- | --- | --- |
| `saas-linux-small-amd64` (untagged default) | The linux-x86_64 wheel, the `experiential` build, the Capture smoke, the publish job, and every `gate` job | Free |
| `saas-linux-small-arm64` | The linux-aarch64 wheel | Free |
| `saas-windows-medium-amd64` (beta) | The windows-x86_64 wheel | Free |
| `saas-macos-medium-m1` (beta) | The macos-arm64 and macos-x86_64 wheels | Premium or Ultimate |

A job whose runner the plan does not include stays pending, and the publish
job waits for every wheel. A self-hosted runner registered with the same tag
serves it instead.

## Releases

A release is a tag `v<version>`, where `<version>` is the `experiential`
version in `pyproject.toml` (for example `v0.7.179` or `v0.7.94rc1`). `v*` tags
must be protected (**Settings > Repository > Protected tags**, Maintainers
only): a tag that is not protected starts no jobs. Creating a GitLab release
with a new tag, or pushing the tag, starts a tag pipeline that builds the
`experiential` sdist and wheel and the `exp-gateway-native` sdist and abi3
wheels for linux-x86_64, linux-aarch64, macos-x86_64, macos-arm64, and
windows-x86_64, smoke-tests them, and publishes both distributions to PyPI.

A manual `python-package` run on `main` with `PYTHON_PACKAGE_SOURCE=latest-release`
and `PYTHON_PACKAGE_PUBLISH=true` rebuilds and publishes the tag of the latest
GitLab release. Files whose exact names are already on PyPI are skipped.

While `.github/deferred-native-release.toml` exists, release tags and
publishing runs fail before they publish.

Publishing uploads with a PyPI API token. GitLab issues this project no ID
tokens, because its path was used by an earlier project, and PyPI's GitLab
trusted publisher accepts only a path-based subject. The
`python-package-publish` job runs in the `pypi` environment and uploads each
distribution with its own project-scoped token: `exp-gateway-native` first with
`PYPI_TOKEN_EXP_GATEWAY_NATIVE`, then `experiential` with
`PYPI_TOKEN_EXPERIENTIAL`, so `experiential` never reaches PyPI ahead of the
native version it requires. Both variables have environment scope `pypi`, so the
build and smoke-test jobs, which install third-party packages, never receive
them, and no other job may declare an environment.

## Variables and settings

Jobs authenticate with `CI_JOB_TOKEN`, and the publish job with the PyPI tokens.

| Setting | Use |
| --- | --- |
| **Settings > CI/CD > Variables**: `PYPI_TOKEN_EXP_GATEWAY_NATIVE` and `PYPI_TOKEN_EXPERIENTIAL`, protected, masked, not expanded, environment scope `pypi` | The project-scoped PyPI API tokens `python-package-publish` uploads `exp-gateway-native` and `experiential` with; only that job, on protected refs, receives them |
| **Settings > Repository > Protected tags**: `v*`, Maintainers | Release tag pipelines run and publish only for protected tags |
| **Settings > CI/CD > Variables > Minimum role to use pipeline variables**: Maintainer | Manual runs set `RUN_WORKFLOW` and the `PYTHON_PACKAGE_*` inputs |
| Environment `pypi` | `python-package-publish` deploys to it |
