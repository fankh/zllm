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

    pub fn from_hf(model_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| ZllmError::Model(format!("HF API error: {e}")))?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| ZllmError::Model(format!("failed to download tokenizer: {e}")))?;
        Self::from_file(
            tokenizer_path
                .to_str()
                .ok_or_else(|| ZllmError::Model("invalid path".into()))?,
        )
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
}
