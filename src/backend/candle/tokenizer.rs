use crate::error::{Result, ZllmError};

pub struct LlamaTokenizer {
    inner: tokenizers::Tokenizer,
}

impl LlamaTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ZllmError::Model(format!("failed to load tokenizer: {e}")))?;
        Ok(Self { inner })
    }

    /// Build from the vocab EMBEDDED in a GGUF (single-file loading).
    /// BPE ("gpt2") vocabs only — see `gguf_vocab` for scope and the
    /// oracle test enforcing parity with sibling tokenizer.json files.
    pub fn from_gguf_content(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        let inner = crate::backend::candle::gguf_vocab::tokenizer_from_gguf(content)
            .map_err(ZllmError::Model)?;
        Ok(Self { inner })
    }

    /// Convenience: open a GGUF file and build from its embedded vocab.
    pub fn from_gguf_file(path: &std::path::Path) -> Result<Self> {
        let mut f = std::fs::File::open(path)
            .map_err(|e| ZllmError::Model(format!("cannot open {}: {e}", path.display())))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut f)
            .map_err(|e| ZllmError::Model(format!("invalid GGUF: {e}")))?;
        Self::from_gguf_content(&content)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| ZllmError::Model(format!("encode error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text = self
            .inner
            .decode(tokens, true)
            .map_err(|e| ZllmError::Model(format!("decode error: {e}")))?;
        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("</s>")
            .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
            .or_else(|| self.inner.token_to_id("<|eot_id|>"))
    }

    /// Vocab lookup for a literal token string. Used for chat-template
    /// family detection and stop-set construction.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Reverse lookup: token id → its literal surface string (specials
    /// included — unlike `decode`, which strips them). Used to feed
    /// `bos_token`/`eos_token` into GGUF chat templates.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Every token id that should terminate generation for this model:
    /// the EOS token plus whichever end-of-turn specials exist in this
    /// vocab (Llama-3 `<|eot_id|>`/`<|eom_id|>`, ChatML `<|im_end|>`,
    /// Phi `<|end|>`, GPT-style `<|endoftext|>`). Derived from the loaded
    /// tokenizer rather than hardcoded ids, so the stop set is correct
    /// across model families — Llama-3's 128009 was previously baked in
    /// and wrong for everything else.
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(eos) = self.eos_token_id() {
            ids.push(eos);
        }
        for tok in ["<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>", "<|endoftext|>"] {
            if let Some(id) = self.inner.token_to_id(tok) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        // Last-resort fallback so an exotic tokenizer with none of the
        // known stop tokens still terminates (128001 = Llama-3
        // <|end_of_text|>, the previous hardcoded default).
        if ids.is_empty() {
            ids.push(128001);
        }
        ids
    }

    /// id → decoded surface bytes for every vocab entry, for grammar-constrained
    /// decoding (the FSM walks these bytes through its DFA). `None` for special
    /// tokens (decode to "" with skip-special) and tokens that aren't valid UTF-8
    /// on their own (partial byte-level BPE pieces decode with U+FFFD) — those
    /// are simply disallowed while a grammar is active. Byte-level BPE (Llama-3)
    /// round-trips single tokens faithfully; SentencePiece decoders may strip a
    /// leading space on isolated tokens, so grammar fidelity there is best-effort.
    /// Cost: one decode per vocab entry (~128k) — build once and cache per model.
    pub fn token_bytes_table(&self) -> Vec<Option<Vec<u8>>> {
        let n = self.inner.get_vocab_size(true);
        (0..n as u32)
            .map(|id| match self.inner.decode(&[id], true) {
                Ok(s) if !s.is_empty() && !s.contains('\u{FFFD}') => Some(s.into_bytes()),
                _ => None,
            })
            .collect()
    }
}
