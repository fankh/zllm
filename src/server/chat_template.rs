//! GGUF-embedded chat templates (`tokenizer.chat_template`), rendered
//! with minijinja — the same approach as llama.cpp's minja. The model's
//! own template is always more faithful than family heuristics; the
//! vocab-probed `ChatFamily` fallback in rest.rs covers template-less
//! GGUFs and render failures.

use candle_core::quantized::gguf_file;
use std::path::Path;

/// Chat-relevant GGUF metadata, read once at model load/swap.
#[derive(Debug, Default, Clone)]
pub struct GgufChatMeta {
    /// The model's own Jinja chat template, if embedded.
    pub template: Option<String>,
    /// Stop ids the GGUF DECLARES (`tokenizer.ggml.eos_token_id` /
    /// `eot_token_id` / `eom_token_id`) — the ground truth the vocab
    /// probe only approximates. Unioned with the probe at request time.
    pub stop_ids: Vec<u32>,
    pub bos_id: Option<u32>,
}

/// Best-effort read — a missing/corrupt file yields the empty default
/// (heuristics take over), never an error.
pub fn read_gguf_chat_meta(path: &Path) -> GgufChatMeta {
    let mut meta = GgufChatMeta::default();
    let Ok(mut f) = std::fs::File::open(path) else {
        return meta;
    };
    let Ok(content) = gguf_file::Content::read(&mut f) else {
        return meta;
    };
    let md = &content.metadata;
    meta.template = md
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()));
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
    ] {
        if let Some(id) = md.get(key).and_then(|v| v.to_u32().ok()) {
            if !meta.stop_ids.contains(&id) {
                meta.stop_ids.push(id);
            }
        }
    }
    meta.bos_id = md
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok());
    meta
}

/// Render a chat template with the context every mainstream template
/// expects: `messages`, `bos_token`, `eos_token`,
/// `add_generation_prompt = true`, plus the `raise_exception` and
/// `strftime_now` callables (Llama-3.x templates date-stamp themselves).
pub fn render(
    template: &str,
    messages: &[(String, String)],
    bos_token: &str,
    eos_token: &str,
) -> Result<String, String> {
    let mut env = minijinja::Environment::new();
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<minijinja::value::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    let ctx = minijinja::context! {
        messages => msgs,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => true,
    };
    env.render_str(template, ctx)
        .map_err(|e| format!("chat template render failed: {e}"))
}

/// Minimal strftime over the current UTC date, covering the specifiers
/// chat templates actually use (Llama-3.x: "%d %b %Y"). Unknown
/// specifiers pass through verbatim.
fn strftime_now(fmt: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    const MON_S: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MON_L: [&str; 12] = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ];
    fmt.replace("%d", &format!("{d:02}"))
        .replace("%m", &format!("{m:02}"))
        .replace("%B", MON_L[(m - 1) as usize])
        .replace("%b", MON_S[(m - 1) as usize])
        .replace("%Y", &y.to_string())
        .replace("%y", &format!("{:02}", y.rem_euclid(100)))
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's
/// `civil_from_days`, the standard branch-free civil-calendar algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn renders_chatml() {
        let msgs = vec![
            ("system".to_string(), "Be brief.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let out = render(CHATML, &msgs, "", "<|im_end|>").unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let tpl = "{{ raise_exception('nope') }}";
        let err = render(tpl, &[], "", "").unwrap_err();
        assert!(err.contains("nope"), "{err}");
    }

    #[test]
    fn strftime_covers_llama3_pattern() {
        let s = strftime_now("%d %b %Y");
        // e.g. "13 Jul 2026" — two-digit day, short month, 4-digit year.
        let parts: Vec<&str> = s.split(' ').collect();
        assert_eq!(parts.len(), 3, "{s}");
        assert_eq!(parts[0].len(), 2);
        assert!(parts[2].parse::<i32>().unwrap() >= 2026);
    }

    #[test]
    fn civil_epoch_and_known_date() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2026-07-13 = 20647 days after epoch.
        assert_eq!(civil_from_days(20_647), (2026, 7, 13));
    }
}
