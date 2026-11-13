//! Pin insensitive matching to the same Unicode simple-fold data as RE2.

use std::sync::OnceLock;

use regex_syntax::ast::{Ast, ClassBracketed, ClassSet, ClassSetItem, Flag, GroupKind};

pub(super) fn apply(ast: &mut Ast, mut insensitive: bool) -> Result<bool, &'static str> {
    walk(ast, &mut insensitive)
}

fn walk(ast: &mut Ast, insensitive: &mut bool) -> Result<bool, &'static str> {
    let replacement = match ast {
        Ast::Literal(literal) if *insensitive => {
            Some(format!("[{}]", range(literal.c as u32, literal.c as u32)))
        }
        Ast::ClassBracketed(class) if *insensitive => Some(bracket(class)?),
        Ast::Flags(flags) => {
            if let Some(value) = flags.flags.flag_state(Flag::CaseInsensitive) {
                *insensitive = value;
            }
            None
        }
        Ast::Group(group) => {
            let mut local = *insensitive;
            if let GroupKind::NonCapturing(flags) = &group.kind {
                if let Some(value) = flags.flag_state(Flag::CaseInsensitive) {
                    local = value;
                }
            }
            return walk(&mut group.ast, &mut local);
        }
        Ast::Repetition(repetition) => return walk(&mut repetition.ast, insensitive),
        Ast::Concat(concat) => return walk_all(&mut concat.asts, insensitive),
        Ast::Alternation(alternation) => return walk_all(&mut alternation.asts, insensitive),
        _ => None,
    };
    if let Some(replacement) = replacement {
        *ast = regex_syntax::ast::parse::Parser::new()
            .parse(&format!("(?-i:{replacement})"))
            .map_err(|_| "invalid folded RE2 class")?;
        return Ok(true);
    }
    Ok(false)
}

fn walk_all(nodes: &mut [Ast], insensitive: &mut bool) -> Result<bool, &'static str> {
    let mut changed = false;
    for node in nodes {
        changed |= walk(node, insensitive)?;
    }
    Ok(changed)
}

fn bracket(class: &ClassBracketed) -> Result<String, &'static str> {
    Ok(format!(
        "[{}{}]",
        if class.negated { "^" } else { "" },
        set(&class.kind)?
    ))
}

fn set(value: &ClassSet) -> Result<String, &'static str> {
    match value {
        ClassSet::Item(item) => member(item),
        ClassSet::BinaryOp(op) => {
            use regex_syntax::ast::ClassSetBinaryOpKind as Kind;
            let operator = match op.kind {
                Kind::Intersection => "&&",
                Kind::Difference => "--",
                Kind::SymmetricDifference => "~~",
            };
            Ok(format!("[{}]{operator}[{}]", set(&op.lhs)?, set(&op.rhs)?))
        }
    }
}

fn member(value: &ClassSetItem) -> Result<String, &'static str> {
    match value {
        ClassSetItem::Empty(_) => Ok(String::new()),
        ClassSetItem::Literal(literal) => Ok(range(literal.c as u32, literal.c as u32)),
        ClassSetItem::Range(r) => Ok(range(r.start.c as u32, r.end.c as u32)),
        ClassSetItem::Bracketed(class) => bracket(class),
        ClassSetItem::Union(union) => union.items.iter().map(member).collect(),
        ClassSetItem::Ascii(ascii) => {
            let name = format!("{:?}", ascii.kind).to_lowercase();
            let hir = regex_syntax::Parser::new()
                .parse(&format!("[[:{name}:]]"))
                .map_err(|_| "invalid ASCII class")?;
            let regex_syntax::hir::HirKind::Class(regex_syntax::hir::Class::Unicode(class)) =
                hir.kind()
            else {
                return Err("invalid ASCII class kind");
            };
            let content: String = class
                .ranges()
                .iter()
                .map(|r| range(r.start() as u32, r.end() as u32))
                .collect();
            Ok(format!(
                "[{}{}]",
                if ascii.negated { "^" } else { "" },
                content
            ))
        }
        // Unicode and Perl classes have already become explicit RE2 ranges.
        _ => Err("untranslated RE2 character class"),
    }
}

fn range(first: u32, last: u32) -> String {
    static FOLDS: OnceLock<Vec<(u32, Vec<u32>)>> = OnceLock::new();
    let folds = FOLDS.get_or_init(|| {
        crate::lossless_json::from_str(include_str!("../generated/re2_casefold.json")).unwrap()
    });
    let mut out = if first == last {
        format!("\\x{{{first:x}}}")
    } else {
        format!("\\x{{{first:x}}}-\\x{{{last:x}}}")
    };
    let start = folds.partition_point(|(code, _)| *code < first);
    let end = folds.partition_point(|(code, _)| *code <= last);
    for (_, targets) in &folds[start..end] {
        for code in targets {
            if *code < first || *code > last {
                out.push_str(&format!("\\x{{{code:x}}}"));
            }
        }
    }
    out
}
