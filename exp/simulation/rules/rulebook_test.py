"""Tests for rendering and writing the Markdown rule store."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.simulation.rules.merge import apply_weighings, attach_witnesses, merge_rules
from exp.simulation.rules.model import (
    Operand,
    Predicate,
    ResearchQuestion,
    RuleBook,
    RuleProposal,
    RuleWeighing,
    ToolContract,
    TrajectoryRules,
    Witness,
)
from exp.simulation.rules.rulebook import (
    RULES_JSON,
    render_rulebook,
    rulebook_json,
    tool_file_paths,
    write_rulebook,
)

_PAID = Predicate(
    op="eq", left=Operand(scope="before", path="/order/status"), right=Operand(value="paid")
)


def _book() -> RuleBook:
    """A small book with a contract, tool rules, an invariant, a contradiction and questions."""
    contract = ToolContract(
        name="refund",
        declared_schema={"type": "object", "properties": {"amount": {}, "order_id": {}}},
        argument_types={"amount": ("number",), "order_id": ("string",)},
        required_arguments=("order_id",),
        result_shapes=("object{ok}",),
        error_forms=("not_found",),
        calls=5,
        failures=1,
        examples=("trace:t1:span:1",),
    )

    def proposal(trace: str, **fields: str | Predicate | tuple[str, ...] | None) -> RuleProposal:
        """A refund precondition proposal from one trace, with overrides."""
        values = {
            "tool": "refund",
            "kind": "precondition",
            "statement": "Order must be paid",
            "evidence": (f"trace:{trace}:span:1",),
            "confidence": 0.5,
            **fields,
        }
        return RuleProposal.model_validate(values)

    analyses = [
        TrajectoryRules(
            trace_id="t1",
            rules=(
                proposal("t1", predicate=_PAID),
                proposal("t1", tool=None, kind="invariant", statement="Balances never go negative"),
            ),
            questions=(
                ResearchQuestion(tool="refund", question="Can amount be zero?", why_it_matters="x"),
                ResearchQuestion(question="Is time zone fixed?", why_it_matters="y"),
            ),
            limitations=("none",),
        ),
        TrajectoryRules(
            trace_id="t2",
            rules=(
                proposal("t2", predicate=_PAID),
                proposal("t2", kind="effect", statement="Sets status", counter_evidence=("c:9",)),
            ),
            limitations=("none",),
        ),
    ]
    merged = merge_rules({"refund": contract}, analyses)
    paid = next(rule for rule in merged.rules if rule.predicate == _PAID)
    weighing = RuleWeighing(
        rule_id=paid.rule_id,
        status="supported",
        rationale="Both refusals turn on payment.",
        supporting=paid.evidence,
        independence="Two separate tasks.",
    )
    book = apply_weighings(merged, [weighing])
    witness = Witness(kind="native", reference="run:7", supports=True)
    return attach_witnesses(book, {paid.rule_id: [witness]})


def test_render_lays_out_index_tool_and_environment_files() -> None:
    """The store has an index, one file per tool and an environment file."""
    files = render_rulebook(_book())
    assert list(files) == ["README.md", "environment.md", "tools/refund.md"]
    readme = files["README.md"]
    assert "- Sources: 2 traces" in readme
    assert "- supported: 1" in readme
    assert "- contradicted: 1" in readme
    assert "- unknown: 1" in readme
    assert "[tools/refund.md](tools/refund.md)" in readme
    assert "Can amount be zero?" in readme
    assert "Balances never go negative" in files["environment.md"]
    assert "Is time zone fixed?" in files["environment.md"]
    assert "Can amount be zero?" not in files["environment.md"]


def test_tool_file_has_contract_rules_and_questions() -> None:
    """A tool file summarizes its contract and lists every rule with evidence on both sides."""
    text = render_rulebook(_book())["tools/refund.md"]
    assert "- Declared schema keys: `amount`, `order_id`" in text
    assert "- Required arguments: `order_id`" in text
    assert "  - `amount`: number" in text
    assert "`object{ok}`" in text
    assert "`not_found`" in text
    assert "- Calls: 5" in text
    assert text.index("### Precondition") < text.index("### Effect")
    assert '{"args":[],"left":{"path":"/order/status"' in text
    assert "```json" in text
    assert "- Status: contradicted" in text
    assert "  - `c:9`" in text
    assert "  - native `run:7`: supports" in text
    assert "- Support: 2 trajectories" in text
    assert "- Weighing: Both refusals turn on payment." in text
    assert "  - Independence: Two separate tasks." in text
    assert "Can amount be zero?" in text
    assert "Balances never go negative" not in text


def test_render_is_deterministic() -> None:
    """The same book renders byte-identical files."""
    assert render_rulebook(_book()) == render_rulebook(_book())
    assert rulebook_json(_book()) == rulebook_json(_book())


def test_tool_paths_are_sanitized_and_collisions_resolved() -> None:
    """Unsafe names become safe file names, and colliding names all get digest suffixes."""
    paths = tool_file_paths(["../../etc/passwd", "Search", "search", "a b/c", ""])
    assert paths["../../etc/passwd"] == "tools/etc_passwd.md"
    assert paths["a b/c"] == "tools/a_b_c.md"
    assert paths[""] == "tools/tool.md"
    assert paths["Search"] != paths["search"]
    assert paths["Search"].startswith("tools/search-")
    assert paths == tool_file_paths(["search", "", "a b/c", "Search", "../../etc/passwd"])
    for path in paths.values():
        assert path.count("/") == 1


def test_write_creates_a_new_directory_with_json_copy(tmp_path: Path) -> None:
    """All files land in the new directory, and rules.json reads back as the same book."""
    book = _book()
    target = tmp_path / "store" / "rules"
    written = write_rulebook(book, target)
    assert written == tuple(sorted(written))
    names = sorted(path.relative_to(target).as_posix() for path in written)
    assert names == ["README.md", "environment.md", RULES_JSON, "tools/refund.md"]
    assert RuleBook.model_validate_json((target / RULES_JSON).read_text()) == book
    assert (target / "README.md").read_text() == render_rulebook(book)["README.md"]
    assert [path.name for path in tmp_path.joinpath("store").iterdir()] == ["rules"]


def test_write_refuses_an_existing_directory(tmp_path: Path) -> None:
    """An existing directory, even an empty one, is never written into."""
    target = tmp_path / "rules"
    target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        write_rulebook(_book(), target)
    assert list(target.iterdir()) == []
    assert [path.name for path in tmp_path.iterdir()] == ["rules"]
