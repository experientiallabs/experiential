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
//! A pattern this module cannot translate with certainty is rejected. A
//! rejected pattern keeps running through the python adapter, so behavior is
//! preserved and only the native fast path is declined.

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
            inner: Some("0-9"),
        }),
        'D' => Some(PerlClass {
            outer: "[^0-9]",
            inner: None,
        }),
        'w' => Some(PerlClass {
            outer: "[0-9A-Za-z_]",
            inner: Some("0-9A-Za-z_"),
        }),
        'W' => Some(PerlClass {
            outer: "[^0-9A-Za-z_]",
            inner: None,
        }),
        's' => Some(PerlClass {
            outer: "[\\t\\n\\x0C\\r ]",
            inner: Some("\\t\\n\\x0C\\r "),
        }),
        'S' => Some(PerlClass {
            outer: "[^\\t\\n\\x0C\\r ]",
            inner: None,
        }),
        _ => None,
    }
}

/// Rewrite one authored RE2 expression for the Rust `regex` crate.
///
/// Returns the translated expression, or an error naming the untranslatable
/// construct. An error is not an authoring failure: the caller declines the
/// native fast path and keeps the python adapter for that rule.
pub fn to_rust_syntax(pattern: &str) -> Result<String, &'static str> {
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
                if escaped.is_ascii_digit() && escaped != '0' {
                    return Err("backreferences are not supported");
                }
                if let Some(expansion) = perl_class(escaped) {
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
            '-' if in_class && chars.peek() == Some(&'-') => {
                return Err("ambiguous class subtraction syntax");
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
    Ok(out)
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
        assert_eq!(to_rust_syntax(r"[\d-]").unwrap(), "[0-9-]");
        assert_eq!(to_rust_syntax(r"[a\w]").unwrap(), "[a0-9A-Za-z_]");
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
        assert!(to_rust_syntax(r"[\D]").is_err());
        assert!(to_rust_syntax(r"[abc").is_err());
    }

    #[test]
    fn a_class_bracket_is_not_closed_by_its_own_escape() {
        assert_eq!(to_rust_syntax(r"[a\[b]c").unwrap(), r"[a\[b]c");
    }
}
