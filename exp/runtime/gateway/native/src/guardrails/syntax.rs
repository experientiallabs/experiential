//! Translate an authored RE2 expression into the equivalent Rust `regex`
//! expression.
//!
//! Authored patterns are validated by the python RE2 engine at load, so this
//! translation never widens or narrows what an operator may write. It only
//! rewrites the atoms whose meaning differs between the two engines, so the
//! native detector matches exactly what RE2 matches:
//!
//! * RE2 perl classes (`\d`, `\w`, `\s` and their negations) are ASCII only,
//!   while the Rust `regex` crate reads them as Unicode classes.
//! * RE2 word boundaries (`\b`, `\B`) are ASCII only, while the Rust crate
//!   reads them as Unicode boundaries.
//!
//! Byte atoms are emitted for the byte matcher. The UTF-8 detector rejects
//! those atoms unless the caller uses the byte matcher explicitly.

/// ASCII expansion of one RE2 perl class, outside and inside a class.
struct PerlClass {
    /// Standalone form, already wrapped as its own character class.
    outer: &'static str,
    /// Class-member form, or `None` when the class cannot be expanded as a
    /// member of an enclosing character class.
    inner: Option<&'static str>,
}

/// Return the ASCII expansion of one RE2 perl class escape.
fn perl_class(letter: char) -> Option<PerlClass> {
    match letter {
        'd' => Some(PerlClass {
            outer: "[0-9]",
            inner: Some("[0-9]"),
        }),
        'D' => Some(PerlClass {
            outer: "[^0-9]",
            inner: Some("[^0-9]"),
        }),
        'w' => Some(PerlClass {
            outer: "[0-9A-Za-z_]",
            inner: Some("[0-9A-Za-z_]"),
        }),
        'W' => Some(PerlClass {
            outer: "[^0-9A-Za-z_]",
            inner: Some("[^0-9A-Za-z_]"),
        }),
        's' => Some(PerlClass {
            outer: "[\\t\\n\\x0C\\r ]",
            inner: Some("[\\t\\n\\x0C\\r ]"),
        }),
        'S' => Some(PerlClass {
            outer: "[^\\t\\n\\x0C\\r ]",
            inner: Some("[^\\t\\n\\x0C\\r ]"),
        }),
        _ => None,
    }
}

/// Rewrite one authored RE2 expression for the Rust `regex` crate.
///
/// Returns the translated expression, or an error naming the untranslatable
/// construct. An error is not an authoring failure: the caller declines the
/// native fast path and keeps the python adapter for that rule.
#[cfg(test)]
fn to_rust_syntax(pattern: &str) -> Result<String, &'static str> {
    with_flags(pattern, false)
}

pub(crate) fn with_flags(pattern: &str, insensitive: bool) -> Result<String, &'static str> {
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    // A class opens with an optional `^` and may then carry a literal `]`.
    let mut class_start = 0usize;
    let mut class_len = 0usize;
    while let Some(current) = chars.next() {
        if in_class {
            class_len += 1;
        }
        match current {
            '\\' => {
                let escaped = chars.next().ok_or("trailing escape")?;
                if ('0'..='7').contains(&escaped) {
                    let mut octal = String::from(escaped);
                    while octal.len() < 3 && chars.peek().is_some_and(|c| ('0'..='7').contains(c)) {
                        octal.push(chars.next().unwrap());
                    }
                    if octal.len() == 1 && escaped != '0' {
                        return Err("backreferences are not supported");
                    }
                    let value =
                        u32::from_str_radix(&octal, 8).map_err(|_| "invalid octal escape")?;
                    out.push_str(&format!("\\x{{{value:x}}}"));
                } else if escaped.is_ascii_digit() {
                    return Err("invalid octal escape");
                } else if escaped == 'Q' && !in_class {
                    let mut literal = String::new();
                    while let Some(next) = chars.next() {
                        if next == '\\' && chars.peek() == Some(&'E') {
                            chars.next();
                            break;
                        }
                        literal.push(next);
                    }
                    out.push_str(&regex::escape(&literal));
                } else if escaped == 'C' && !in_class {
                    out.push_str("(?s-u:.)");
                } else if escaped == 'p' || escaped == 'P' {
                    let first = chars.next().ok_or("missing Unicode property")?;
                    let mut name = String::new();
                    if first == '{' {
                        loop {
                            let next = chars.next().ok_or("unterminated Unicode property")?;
                            if next == '}' {
                                break;
                            }
                            name.push(next);
                        }
                    } else {
                        name.push(first);
                    }
                    let negate = (escaped == 'P') ^ name.starts_with('^');
                    let name = name.strip_prefix('^').unwrap_or(&name);
                    let class = unicode_class(name).ok_or("unknown RE2 Unicode property")?;
                    if negate {
                        out.push_str("[^");
                        out.push_str(class);
                        out.push(']');
                    } else {
                        out.push_str(class);
                    }
                } else if let Some(expansion) = perl_class(escaped) {
                    if in_class {
                        let inner = expansion.inner.ok_or("negated perl class inside a class")?;
                        out.push_str(inner);
                    } else {
                        out.push_str(expansion.outer);
                    }
                } else if escaped == 'b' || escaped == 'B' {
                    if in_class {
                        return Err("word boundary inside a class");
                    }
                    out.push_str(if escaped == 'b' {
                        "(?-u:\\b)"
                    } else {
                        "(?-u:\\B)"
                    });
                } else {
                    if escaped.is_ascii_alphanumeric()
                        && !matches!(escaped, 'a' | 'f' | 't' | 'n' | 'r' | 'v' | 'x' | 'A' | 'z')
                    {
                        return Err("invalid RE2 escape");
                    }
                    out.push('\\');
                    out.push(escaped);
                }
                if in_class {
                    class_len += 1;
                }
            }
            '[' if !in_class => {
                in_class = true;
                class_start = out.len();
                class_len = 0;
                out.push('[');
            }
            '[' if in_class && chars.peek() == Some(&':') => {
                // A POSIX class name is ASCII only in both engines; copy it
                // through verbatim up to its closing `:]`.
                out.push('[');
                let mut closed = false;
                while let Some(inner) = chars.next() {
                    out.push(inner);
                    class_len += 1;
                    if inner == ':' && chars.peek() == Some(&']') {
                        out.push(']');
                        chars.next();
                        class_len += 1;
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err("unterminated posix class");
                }
            }
            '[' if in_class => {
                // RE2 treats a nested opening bracket as a literal member.
                // Rust instead opens a nested class with different membership.
                out.push_str(r"\[");
            }
            '&' | '~' if in_class => {
                // Rust reserves doubled members for intersection and symmetric
                // difference. RE2 treats each member literally.
                out.push('\\');
                out.push(current);
            }
            '-' if in_class => {
                let leading = class_len == 1
                    || (class_len == 2 && out.as_bytes().get(class_start + 1) == Some(&b'^'));
                if leading || chars.peek() == Some(&']') {
                    out.push_str(r"\-");
                } else {
                    if chars.peek() == Some(&'-') && !out.ends_with(r"\-") {
                        return Err("invalid RE2 character range");
                    }
                    out.push('-');
                }
            }
            '(' if !in_class && chars.peek() == Some(&'?') => {
                let mut suffix = chars.clone();
                suffix.next();
                // RE2 flags are exactly i,m,s,U. Rust additionally accepts
                // x,u,R and their negations, which must not widen the language.
                if !matches!(suffix.peek(), Some(':' | '<' | 'P')) {
                    for flag in suffix {
                        if flag == ':' || flag == ')' {
                            break;
                        }
                        if !matches!(flag, 'i' | 'm' | 's' | 'U' | '-') {
                            return Err("invalid RE2 flag");
                        }
                    }
                }
                out.push('(');
            }
            '{' if !in_class => {
                let suffix: String = chars
                    .clone()
                    .take_while(|c| c.is_ascii_digit() || *c == ',')
                    .collect();
                let following = chars.clone().nth(suffix.len());
                let valid = !suffix.is_empty()
                    && !suffix.starts_with(',')
                    && following == Some('}')
                    && suffix.matches(',').count() <= 1
                    && !suffix.split(',').any(|n| n.len() > 1 && n.starts_with('0'));
                if !valid {
                    out.push('\\');
                }
                out.push('{');
            }
            ']' if in_class => {
                // A `]` in the first content position is a literal member in
                // RE2; the Rust parser requires it escaped.
                let leading = class_len == 1
                    || (class_len == 2 && out.as_bytes().get(class_start + 1) == Some(&b'^'));
                if leading {
                    out.push_str("\\]");
                } else {
                    in_class = false;
                    out.push(']');
                }
            }
            other => out.push(other),
        }
    }
    if in_class {
        return Err("unterminated character class");
    }
    let mut ast = regex_syntax::ast::parse::Parser::new()
        .parse(&out)
        .map_err(|_| "invalid regular expression")?;
    repetition_size(&ast)?;
    if super::folding::apply(&mut ast, insensitive)? {
        Ok(ast.to_string())
    } else {
        Ok(out)
    }
}

fn unicode_class(name: &str) -> Option<&'static str> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    // RE2 parses `Any` itself rather than from its Unicode group table.
    if name == "Any" {
        return Some(r"[\x{0}-\x{10ffff}]");
    }
    static GROUPS: OnceLock<HashMap<String, String>> = OnceLock::new();
    GROUPS
        .get_or_init(|| {
            crate::lossless_json::from_str(include_str!("../generated/re2_groups.json")).unwrap()
        })
        .get(name)
        .map(String::as_str)
}

/// RE2 caps counted repeats, including the product of nested counted repeats.
fn repetition_size(ast: &regex_syntax::ast::Ast) -> Result<u32, &'static str> {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    let size = match ast {
        Ast::Repetition(r) => {
            let inner = repetition_size(&r.ast)?;
            let count = match r.op.kind {
                RepetitionKind::Range(
                    RepetitionRange::Exactly(n) | RepetitionRange::AtLeast(n),
                ) => n,
                RepetitionKind::Range(RepetitionRange::Bounded(_, n)) => n,
                _ => 1,
            };
            inner.saturating_mul(count.max(1))
        }
        Ast::Group(g) => repetition_size(&g.ast)?,
        Ast::Concat(c) => c
            .asts
            .iter()
            .map(repetition_size)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(1),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .map(repetition_size)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(1),
        _ => 1,
    };
    (size <= 1000)
        .then_some(size)
        .ok_or("RE2 repetition limit exceeded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perl_classes_become_ascii_classes() {
        assert_eq!(to_rust_syntax(r"\d+").unwrap(), "[0-9]+");
        assert_eq!(to_rust_syntax(r"\w").unwrap(), "[0-9A-Za-z_]");
        assert_eq!(to_rust_syntax(r"\D").unwrap(), "[^0-9]");
    }

    #[test]
    fn whitespace_excludes_the_vertical_tab_re2_omits() {
        let translated = to_rust_syntax(r"a\sb").unwrap();
        let expression = regex::Regex::new(&translated).unwrap();
        assert!(expression.is_match("a b"));
        assert!(expression.is_match("a\tb"));
        assert!(!expression.is_match("a\u{000b}b"));
    }

    #[test]
    fn perl_classes_inside_a_class_expand_to_members() {
        assert_eq!(to_rust_syntax(r"[\d-]").unwrap(), r"[[0-9]\-]");
        assert_eq!(to_rust_syntax(r"[a\w]").unwrap(), "[a[0-9A-Za-z_]]");
    }

    #[test]
    fn the_any_property_re2_special_cases_compiles() {
        let expression = regex::Regex::new(&to_rust_syntax(r"secret=\p{Any}+").unwrap()).unwrap();
        assert!(expression.is_match("secret=abc"));
        assert!(expression.is_match("secret=\n"));
        let negated = regex::Regex::new(&to_rust_syntax(r"x\P{Any}").unwrap()).unwrap();
        assert!(!negated.is_match("xa"));
    }

    #[test]
    fn word_boundaries_become_ascii_boundaries() {
        assert_eq!(to_rust_syntax(r"\bx\B").unwrap(), "(?-u:\\b)x(?-u:\\B)");
    }

    #[test]
    fn escapes_and_posix_classes_pass_through() {
        assert_eq!(to_rust_syntax(r"a\.b").unwrap(), r"a\.b");
        assert_eq!(to_rust_syntax(r"[[:alpha:]]+").unwrap(), "[[:alpha:]]+");
        assert_eq!(to_rust_syntax(r"[\]]").unwrap(), r"[\]]");
    }

    #[test]
    fn untranslatable_constructs_are_declined() {
        assert!(to_rust_syntax(r"(a)\1").is_err());
        assert_eq!(to_rust_syntax(r"[\D]").unwrap(), "[[^0-9]]");
        assert!(to_rust_syntax(r"[abc").is_err());
    }

    #[test]
    fn re2_octal_and_quoted_literals_keep_their_meaning() {
        let pattern = to_rust_syntax(r"\123\12\0\Q[a].\E").unwrap();
        assert!(regex::Regex::new(&pattern).unwrap().is_match("S\n\0[a]."));
        let bytes = regex::bytes::Regex::new(&to_rust_syntax(r"\C\C").unwrap()).unwrap();
        assert!(bytes.is_match("é".as_bytes()));
    }

    #[test]
    fn a_class_bracket_is_not_closed_by_its_own_escape() {
        assert_eq!(to_rust_syntax(r"[a\[b]c").unwrap(), r"[a\[b]c");
    }
}
