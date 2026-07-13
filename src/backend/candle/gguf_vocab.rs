//! Build a HuggingFace `tokenizers::Tokenizer` from the vocab EMBEDDED
//! in a GGUF (`tokenizer.ggml.*`) — single-file model loading, no
//! sibling `tokenizer.json` required.
//!
//! Scope: byte-level BPE vocabs (`tokenizer.ggml.model = "gpt2"` —
//! Llama 3, Qwen2, and most modern models). SentencePiece vocabs
//! (`"llama"` — Llama 2, Mistral v0.3) still need the sibling
//! `tokenizer.json`; faithfully reconstructing SPM normalization is a
//! deeper project than the payoff justifies while every SPM model ships
//! one. The per-family split regex comes from `tokenizer.ggml.pre`,
//! mirroring llama.cpp's pre-tokenizer registry.
//!
//! Fidelity is enforced by a model-gated oracle test comparing this
//! construction against the sibling tokenizer.json token-for-token.

use candle_core::quantized::gguf_file::{Content, Value};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// GPT-2's original split regex — the `default`/unknown-pre fallback.
const PRE_GPT2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
/// Llama-3 family (`tokenizer.ggml.pre = "llama-bpe"`).
const PRE_LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Qwen2 family — like llama-bpe but single-digit number splitting.
const PRE_QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn split_regex_for(pre: &str) -> &'static str {
    match pre {
        "llama-bpe" | "llama3" | "smaug-bpe" => PRE_LLAMA3,
        "qwen2" | "deepseek-r1-qwen" => PRE_QWEN2,
        _ => PRE_GPT2,
    }
}

fn md_str(c: &Content, key: &str) -> Option<String> {
    c.metadata.get(key).and_then(|v| v.to_string().ok().map(|s| s.to_string()))
}

fn md_bool(c: &Content, key: &str) -> Option<bool> {
    c.metadata.get(key).and_then(|v| v.to_bool().ok())
}

fn md_str_array(c: &Content, key: &str) -> Option<Vec<String>> {
    match c.metadata.get(key)? {
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| v.to_string().ok().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    }
}

fn md_i32_array(c: &Content, key: &str) -> Option<Vec<i32>> {
    match c.metadata.get(key)? {
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| v.to_i32().ok().or_else(|| v.to_u32().ok().map(|u| u as i32)))
                .collect(),
        ),
        _ => None,
    }
}

/// Construct a tokenizer from GGUF-embedded vocab. `Err` carries a
/// user-facing reason ("SPM vocab — provide tokenizer.json", …).
pub fn tokenizer_from_gguf(content: &Content) -> Result<Tokenizer, String> {
    let model_kind = md_str(content, "tokenizer.ggml.model").unwrap_or_default();
    if model_kind != "gpt2" {
        return Err(format!(
            "GGUF-embedded vocab kind {model_kind:?} is not supported (only byte-level BPE \"gpt2\"); \
             place the model's tokenizer.json next to the GGUF"
        ));
    }
    let tokens = md_str_array(content, "tokenizer.ggml.tokens")
        .ok_or("GGUF missing tokenizer.ggml.tokens")?;
    let merges_raw = md_str_array(content, "tokenizer.ggml.merges")
        .ok_or("GGUF missing tokenizer.ggml.merges")?;
    let token_type = md_i32_array(content, "tokenizer.ggml.token_type").unwrap_or_default();
    let pre = md_str(content, "tokenizer.ggml.pre").unwrap_or_default();

    let vocab: tokenizers::models::bpe::Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .filter_map(|m| {
            let (a, b) = m.split_once(' ')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(true) // llama-3/qwen tokenizer.json semantics: vocab hits skip merge walk
        .build()
        .map_err(|e| format!("BPE build: {e}"))?;
    let mut tok = Tokenizer::new(bpe);

    let split = Split::new(
        SplitPattern::Regex(split_regex_for(&pre).to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| format!("split regex: {e}"))?;
    let byte_level = ByteLevel::new(false, true, false); // no prefix space, trim offsets, regex handled by Split
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ])));
    tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(
        false, true, false,
    )));

    // Control tokens (type 3) are special: never split, stripped on
    // decode(skip_special). USER_DEFINED (4) are added but not special.
    let mut specials: Vec<AddedToken> = Vec::new();
    let mut user_defined: Vec<AddedToken> = Vec::new();
    for (i, ty) in token_type.iter().enumerate() {
        match ty {
            3 => specials.push(AddedToken::from(tokens[i].clone(), true)),
            4 => user_defined.push(AddedToken::from(tokens[i].clone(), false)),
            _ => {}
        }
    }
    if !specials.is_empty() {
        tok.add_special_tokens(&specials);
    }
    if !user_defined.is_empty() {
        tok.add_tokens(&user_defined);
    }

    // BOS handling: llama-3 prepends BOS on encode; qwen2 does not.
    // Some quants omit add_bos_token entirely (bartowski Llama-3.2 does)
    // — default by family, matching what each family's HF tokenizer.json
    // post-processor does.
    let add_bos = md_bool(content, "tokenizer.ggml.add_bos_token")
        .unwrap_or(matches!(pre.as_str(), "llama-bpe" | "llama3" | "smaug-bpe"));
    if add_bos {
        let bos_id = content
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok())
            .ok_or("add_bos_token set but no bos_token_id")?;
        let bos = tokens
            .get(bos_id as usize)
            .cloned()
            .ok_or("bos_token_id out of range")?;
        let post = TemplateProcessing::builder()
            .try_single(format!("{bos}:0 $A:0"))
            .map_err(|e| format!("post-processor single: {e}"))?
            .try_pair(format!("{bos}:0 $A:0 {bos}:1 $B:1"))
            .map_err(|e| format!("post-processor pair: {e}"))?
            .special_tokens(vec![(bos, bos_id)])
            .build()
            .map_err(|e| format!("post-processor: {e}"))?;
        tok.with_post_processor(Some(post));
    }

    Ok(tok)
}
