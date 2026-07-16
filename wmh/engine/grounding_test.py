"""Tests for the web-grounding seam."""

from __future__ import annotations

import json

import pytest

from wmh.core.types import Action, ActionKind
from wmh.engine.grounding import (
    BraveGrounder,
    FetchGrounder,
    FileRead,
    GroundingResult,
    NullGrounder,
    SourceResolver,
    extract_file_read,
    extract_get_url,
    extract_package_query,
    get_grounder,
    registry_grounded_knowledge,
    render_grounding,
    source_grounded_knowledge,
)


def test_null_grounder_never_searches() -> None:
    assert NullGrounder().ground("anything") == []


def test_get_grounder_maps_kinds() -> None:
    assert isinstance(get_grounder("none"), NullGrounder)
    with pytest.raises(ValueError, match="grounder"):
        get_grounder("bing")


def test_get_grounder_brave_without_key_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_SEARCH_API_KEY"):
        get_grounder("brave")


def test_brave_grounder_parses_results_and_sends_key() -> None:
    seen: dict[str, str] = {}

    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        seen["url"] = url
        seen.update(headers)
        return json.dumps(
            {
                "web": {
                    "results": [
                        {
                            "title": "tomli-w · PyPI",
                            "url": "https://pypi.org/project/tomli-w/",
                            "description": "A lil' TOML writer.",
                        }
                    ]
                }
            }
        )

    grounder = BraveGrounder(api_key="k123", fetch=fake_fetch)
    results = grounder.ground("tomli_w python package")
    assert results == [
        GroundingResult(
            title="tomli-w · PyPI",
            url="https://pypi.org/project/tomli-w/",
            snippet="A lil' TOML writer.",
        )
    ]
    assert "tomli_w+python+package" in seen["url"] or "tomli_w%20python%20package" in seen["url"]
    assert seen["X-Subscription-Token"] == "k123"


def test_brave_grounder_tolerates_missing_fields() -> None:
    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        return json.dumps({"web": {"results": [{"title": "only title"}]}})

    results = BraveGrounder(api_key="k", fetch=fake_fetch).ground("q")
    assert results[0].title == "only title"
    assert results[0].url == "" and results[0].snippet == ""


def test_render_grounding_is_compact_markdown() -> None:
    text = render_grounding(
        [
            GroundingResult(title="t1", url="https://a", snippet="s1"),
            GroundingResult(title="t2", url="https://b", snippet="s2"),
        ]
    )
    assert "t1" in text and "https://b" in text and "s2" in text


def test_render_grounding_empty_says_so() -> None:
    assert "no results" in render_grounding([]).lower()


def test_extract_get_url_finds_curl_get_targets() -> None:
    a = _bash('curl -s "https://api.github.com/repos/octocat/Hello-World" | jq .id')
    assert extract_get_url(a) == "https://api.github.com/repos/octocat/Hello-World"
    # plain flag soup still yields the url
    a2 = _bash("curl -sL -H 'Accept: application/json' https://pypi.org/pypi/flask/json")
    assert extract_get_url(a2) == "https://pypi.org/pypi/flask/json"


def test_extract_get_url_rejects_mutating_or_non_curl_commands() -> None:
    assert extract_get_url(_bash('curl -X POST -d "x=1" https://api.example.com/things')) is None
    assert extract_get_url(_bash("curl --data foo https://api.example.com/things")) is None
    assert extract_get_url(_bash("curl -T file.txt https://api.example.com/up")) is None
    assert extract_get_url(_bash("wget https://example.com/file")) is None  # curl only, for now
    assert extract_get_url(_bash("echo hello")) is None
    assert extract_get_url(Action(kind=ActionKind.MESSAGE, content="curl https://x.dev")) is None


def test_fetch_grounder_gets_url_and_memoizes() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        calls.append(url)
        return '{"info": {"home_page": null}}'

    grounder = FetchGrounder(fetch=fake_fetch)
    results = grounder.ground("https://pypi.org/pypi/flask/json")
    assert results[0].url == "https://pypi.org/pypi/flask/json"
    assert '"home_page": null' in results[0].snippet
    grounder.ground("https://pypi.org/pypi/flask/json")  # second ask
    assert calls == ["https://pypi.org/pypi/flask/json"]  # memoized: one real fetch


def test_fetch_grounder_caps_body_and_swallows_fetch_errors() -> None:
    def big(url: str, headers: dict[str, str]) -> str:
        return "x" * 100_000

    capped = FetchGrounder(fetch=big, max_chars=500).ground("https://a.dev")
    assert len(capped[0].snippet) <= 520  # body cap + truncation marker

    def boom(url: str, headers: dict[str, str]) -> str:
        raise OSError("connection refused")

    assert FetchGrounder(fetch=boom).ground("https://b.dev") == []  # fail-safe: no results


def test_fetch_grounder_ignores_non_url_queries() -> None:
    def fail(url: str, headers: dict[str, str]) -> str:
        raise AssertionError("must not fetch")

    assert FetchGrounder(fetch=fail).ground("tomli_w python package") == []


def test_extract_file_read_parses_read_verbs() -> None:
    r = extract_file_read(_bash("sed -n '660,690p' /testbed/django/db/models/query.py"))
    assert r == FileRead(path="django/db/models/query.py", start=660, end=690)
    r2 = extract_file_read(_bash("cd /testbed && cat astropy/modeling/separable.py"))
    assert r2 == FileRead(path="astropy/modeling/separable.py", start=None, end=None)
    r3 = extract_file_read(_bash("head -50 /testbed/setup.py"))
    assert r3 == FileRead(path="setup.py", start=1, end=50)


def test_extract_file_read_rejects_writes_pipes_and_non_reads() -> None:
    assert extract_file_read(_bash("cat > /testbed/x.py << EOF")) is None
    assert extract_file_read(_bash("sed -i 's/a/b/' /testbed/x.py")) is None
    assert extract_file_read(_bash("cat /testbed/x.py | grep foo")) is None  # output transformed
    assert extract_file_read(_bash("python /testbed/x.py")) is None
    assert extract_file_read(Action(kind=ActionKind.MESSAGE, content="cat x.py")) is None


def test_source_resolver_fetches_pinned_slice_and_memoizes() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        calls.append(url)
        return "\n".join(f"line{i}" for i in range(1, 101))

    resolver = SourceResolver(
        {"astropy__astropy-1": {"repo": "astropy/astropy", "base_commit": "abc123"}},
        fetch=fake_fetch,
    )
    read = FileRead(path="astropy/io/core.py", start=5, end=7)
    text = resolver.resolve("astropy__astropy-1", read)
    assert text == "line5\nline6\nline7"
    assert calls == ["https://raw.githubusercontent.com/astropy/astropy/abc123/astropy/io/core.py"]
    resolver.resolve("astropy__astropy-1", FileRead(path="astropy/io/core.py", start=1, end=2))
    assert len(calls) == 1  # same file: memoized
    assert resolver.resolve("unknown-instance", read) is None  # unpinned instance: no-op


def test_source_annotate_stale_serves_labeled_base_for_edited_files() -> None:
    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        return "\n".join(f"base{i}" for i in range(1, 10))

    resolver = SourceResolver({"i-1": {"repo": "o/r", "base_commit": "c"}}, fetch=fake_fetch)
    read_action = _bash("sed -n '2,3p' /testbed/pkg/edited.py")
    prior = [_bash("sed -i 's/a/b/' /testbed/pkg/edited.py")]
    # strict gate refuses...
    assert source_grounded_knowledge(None, read_action, "i-1", prior, resolver) is None
    # ...annotate_stale serves the base WITH the pre-edit label
    text = source_grounded_knowledge(None, read_action, "i-1", prior, resolver, annotate_stale=True)
    assert text is not None and "BASE-COMMIT version" in text and "base2" in text
    # untouched files keep the plain ground-truth header
    fresh = source_grounded_knowledge(
        None,
        _bash("sed -n '2,3p' /testbed/pkg/fresh.py"),
        "i-1",
        prior,
        resolver,
        annotate_stale=True,
    )
    assert fresh is not None and "ground truth at the base commit" in fresh


def test_extract_package_query_parses_pip_and_npm() -> None:
    assert extract_package_query(_bash("pip show flask")) == ("pypi", "flask")
    assert extract_package_query(_bash("pip install requests==2.31.0")) == ("pypi", "requests")
    assert extract_package_query(_bash("pip3 install 'tomli-w>=1.0'")) == ("pypi", "tomli-w")
    assert extract_package_query(_bash("npm view lodash")) == ("npm", "lodash")
    assert extract_package_query(_bash("pip install -r requirements.txt")) is None
    assert extract_package_query(_bash("pip freeze")) is None


def test_extract_package_query_handles_real_corpus_shapes() -> None:
    # piped output, flags before the name, python -m pip, compound commands — all measured shapes
    assert extract_package_query(_bash("pip install pycryptodome 2>&1 | tail -1")) == (
        "pypi",
        "pycryptodome",
    )
    assert extract_package_query(
        _bash("python3 -m pip install --break-system-packages pycryptodome 2>&1 | tail -3")
    ) == ("pypi", "pycryptodome")
    assert extract_package_query(
        _bash("cd /testbed && pip show django 2>/dev/null | head -5 && which python")
    ) == ("pypi", "django")
    assert extract_package_query(_bash("pip install pyyaml --quiet 2>/dev/null")) == (
        "pypi",
        "pyyaml",
    )
    # local/editable installs name a path, not a registry package: refuse
    assert extract_package_query(_bash("cd /testbed && pip install -e . -q 2>&1")) is None
    assert extract_package_query(_bash("pip install .")) is None


def test_registry_grounded_knowledge_polls_pypi() -> None:
    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        assert url == "https://pypi.org/pypi/flask/json"
        return (
            '{"info": {"name": "Flask", "version": "3.0.2", "summary": "A web framework.",'
            ' "requires_dist": ["Werkzeug>=3.0", "Jinja2>=3.1"], "requires_python": ">=3.8"}}'
        )

    text = registry_grounded_knowledge(None, _bash("pip show flask"), fetch=fake_fetch)
    assert text is not None
    assert "Flask" in text and "3.0.2" in text and "Werkzeug>=3.0" in text
    # non-package actions: untouched knowledge
    assert registry_grounded_knowledge("kb", _bash("ls /tmp"), fetch=fake_fetch) == "kb"


def _bash(command: str) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command})


def test_http_get_guard_rejects_non_http_schemes_and_private_hosts() -> None:
    from wmh.engine.grounding import _assert_public_http_url

    with pytest.raises(ValueError, match="http"):
        _assert_public_http_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="http"):
        _assert_public_http_url("ftp://mirror.example.com/pkg")
    with pytest.raises(ValueError, match="no host"):
        _assert_public_http_url("https://")
    # loopback + cloud-metadata link-local: the classic SSRF pivots
    with pytest.raises(ValueError, match="non-public"):
        _assert_public_http_url("http://127.0.0.1:8000/admin")
    with pytest.raises(ValueError, match="non-public"):
        _assert_public_http_url("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError, match="non-public"):
        _assert_public_http_url("http://10.0.0.5/internal")
    with pytest.raises(ValueError, match="non-public"):
        _assert_public_http_url("http://localhost/x")


def test_extract_file_read_refuses_lookalike_binaries() -> None:
    # `zcat`/`tomcat` end in 'cat' but are not reads of the named file's text.
    assert extract_file_read(_bash("zcat /testbed/data.gz")) is None
    assert extract_file_read(_bash("tomcat /testbed/conf.xml")) is None


def test_append_learned_keeps_distinct_prefix_facts(tmp_path) -> None:  # noqa: ANN001
    from wmh.engine.knowledge import KnowledgeBase

    kb = KnowledgeBase(tmp_path / "knowledge")
    assert kb.append_learned(
        "users must re-authenticate after 30 minutes of idle time", provenance="s1"
    )
    # A distinct, more GENERAL fact that happens to prefix the recorded one must not be dropped.
    assert kb.append_learned("users must re-authenticate", provenance="s2")
    learned = (tmp_path / "knowledge" / "learned.md").read_text(encoding="utf-8")
    assert learned.count("- users must re-authenticate") == 2
