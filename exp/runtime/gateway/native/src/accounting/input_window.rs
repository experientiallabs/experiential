//! The bounded decision-model window ends at the latest nonblank user message.

use serde_json::Value;

use super::search_subject::{folded, items};

fn instruction(message: &Value) -> bool {
    matches!(message["role"].as_str(), Some("system" | "developer"))
}

fn block_text(block: &Value) -> &str {
    if block["type"] == "text" {
        block["text"].as_str().unwrap_or("")
    } else {
        ""
    }
}

fn text(message: &Value) -> String {
    if !message["provider_native_item"].is_null() {
        return String::new();
    }
    if !message["content"].is_null() {
        return folded(message);
    }
    if !message["provider_anthropic_block"].is_null() {
        return block_text(&message["provider_anthropic_block"]).to_owned();
    }
    items(&message["provider_anthropic_blocks"])
        .iter()
        .map(block_text)
        .collect()
}

fn cut(text: &str, limit: usize, tail: bool) -> String {
    let chars: Vec<_> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_owned();
    }
    let back = if tail { limit / 2 } else { 0 };
    let mut result: String = chars[..limit - back].iter().collect();
    result.push_str(if tail { "\n[...]\n" } else { "\n[...]" });
    result.extend(chars[chars.len() - back..].iter());
    result
}

pub(super) fn window(request: &Value, bounds: &Value) -> Result<String, ()> {
    let limit = |key: &str| {
        bounds[key]
            .as_u64()
            .filter(|n| *n <= 30_000)
            .map(|n| n as usize)
            .ok_or(())
    };
    let system = limit("system_chars")?;
    let latest_user = limit("latest_user_chars")?;
    let prior_count = limit("prior_messages")?;
    let prior_chars = limit("prior_chars")?;
    if prior_count > 8
        || latest_user == 0
        || prior_chars == 0
        || system + latest_user + prior_count * prior_chars > 30_000
    {
        return Err(());
    }
    let messages = items(&request["messages"]);
    let texts: Vec<_> = messages.iter().map(text).collect();
    // CPython additionally treats the four information separators as whitespace.
    let latest = messages.iter().zip(&texts).rposition(|(m, t)| {
        m["role"] == "user"
            && t.chars()
                .any(|c| !(c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)))
    });
    let end = latest.unwrap_or(messages.len());
    let instructions = messages[..end]
        .iter()
        .zip(&texts[..end])
        .filter(|(m, t)| instruction(m) && !t.is_empty())
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut sections = Vec::new();
    if system != 0 && !instructions.is_empty() {
        sections.push(format!("[system]\n{}", cut(&instructions, system, false)));
    }
    let mut prior = Vec::new();
    for (message, text) in messages[..end].iter().zip(&texts[..end]).rev() {
        if prior.len() == prior_count {
            break;
        }
        if !instruction(message) && !text.is_empty() {
            prior.push(format!(
                "[{}]\n{}",
                message["role"].as_str().ok_or(())?,
                cut(text, prior_chars, true)
            ));
        }
    }
    sections.extend(prior.into_iter().rev());
    if let Some(index) = latest {
        sections.push(format!(
            "[latest user]\n{}",
            cut(&texts[index], latest_user, true)
        ));
    }
    Ok(sections.join("\n\n"))
}
