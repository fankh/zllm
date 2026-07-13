//! Architecture registry — the ONE place that knows GGUF metadata keys and
//! per-family hyperparameters.
//!
//! Today exactly one family is supported (dense llama); adding a family
//! means adding an `ArchSpec` entry here and, only when the block math
//! actually differs, flags on the spec consumed by the forward pass
//! (qkv bias, activation, qk-norm, …). Before this module the `llama.*`
//! parsing was duplicated four ways (candle fork, candle backend, gpu,
//! vulkan) with divergent missing-key behavior — the candle backend
//! silently defaulted to 32/4096/128256 while every other loader errored.
//! Missing mandatory keys are a hard error everywhere now.

use candle_core::quantized::gguf_file;
use std::collections::HashMap;

/// One supported GGUF model family. The `prefix` is the arch name GGUF
/// uses both in `general.architecture` and as the metadata key prefix
/// (`llama.block_count`, `qwen2.block_count`, …).
pub struct ArchSpec {
    pub prefix: &'static str,
}

pub const LLAMA: ArchSpec = ArchSpec { prefix: "llama" };

/// Look up the spec for a `general.architecture` value. `None` = the
/// family is not supported; callers surface that as a load/swap error.
pub fn spec_for(arch: &str) -> Option<&'static ArchSpec> {
    if arch.eq_ignore_ascii_case(LLAMA.prefix) {
        Some(&LLAMA)
    } else {
        None
    }
}

/// Hyperparameters read from GGUF metadata under an arch prefix.
///
/// Mandatory (missing ⇒ `Err`): layer count, hidden width, head counts —
/// nothing downstream can run without them, and guessing produced
/// garbage-dimension bugs historically. Optional fields keep per-caller
/// policy: the candle fork requires `rope_dim`/`rms_eps` (it errors),
/// the GPU/Vulkan engines default `rms_eps` to 1e-5.
#[derive(Debug)]
pub struct HParams {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub rope_dim: Option<usize>,
    /// Explicit per-head width (`{p}.attention.key_length`). Most models
    /// omit it (head_dim = n_embd / n_head), but Mistral-Nemo and
    /// Mistral-Small-3 ship head_dim = 128 with n_embd/n_head = 160 —
    /// loaders MUST prefer this over the division when present.
    pub head_dim: Option<usize>,
    pub rms_eps: Option<f32>,
    /// Defaults to 10000.0 when absent, matching every prior loader.
    pub rope_freq_base: f32,
    /// 0 = dense. MoE (`> 1`) is rejected by the candle fork at load.
    pub n_expert: usize,
    pub vocab_size: Option<usize>,
}

impl HParams {
    pub fn read(
        md: &HashMap<String, gguf_file::Value>,
        spec: &ArchSpec,
    ) -> Result<Self, String> {
        let p = spec.prefix;
        let req_u = |key: String| -> Result<usize, String> {
            md.get(&key)
                .ok_or_else(|| format!("GGUF metadata missing {key}"))?
                .to_u32()
                .map(|v| v as usize)
                .map_err(|e| format!("GGUF metadata {key}: {e}"))
        };
        let opt_u = |key: String| md.get(&key).and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let opt_f = |key: String| md.get(&key).and_then(|v| v.to_f32().ok());

        let n_head = req_u(format!("{p}.attention.head_count"))?;
        Ok(Self {
            n_layers: req_u(format!("{p}.block_count"))?,
            n_embd: req_u(format!("{p}.embedding_length"))?,
            n_head,
            // Per the GGUF spec, absent head_count_kv means MHA (== head_count).
            n_head_kv: opt_u(format!("{p}.attention.head_count_kv")).unwrap_or(n_head),
            rope_dim: opt_u(format!("{p}.rope.dimension_count")),
            head_dim: opt_u(format!("{p}.attention.key_length")),
            rms_eps: opt_f(format!("{p}.attention.layer_norm_rms_epsilon")),
            rope_freq_base: opt_f(format!("{p}.rope.freq_base")).unwrap_or(10000.0),
            n_expert: opt_u(format!("{p}.expert_count")).unwrap_or(0),
            vocab_size: opt_u(format!("{p}.vocab_size")),
        })
    }

    /// Per-head width: the explicit `attention.key_length` when the GGUF
    /// carries one, else the classic `n_embd / n_head`.
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.n_embd / self.n_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf_file::Value;

    fn md(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    fn base_llama() -> HashMap<String, Value> {
        md(&[
            ("llama.block_count", Value::U32(16)),
            ("llama.embedding_length", Value::U32(2048)),
            ("llama.attention.head_count", Value::U32(32)),
            ("llama.attention.head_count_kv", Value::U32(8)),
        ])
    }

    #[test]
    fn mandatory_keys_error_when_missing() {
        // No silent 32/4096 defaults — that bug class is dead.
        let mut m = base_llama();
        m.remove("llama.block_count");
        let err = HParams::read(&m, &LLAMA).unwrap_err();
        assert!(err.contains("llama.block_count"), "{err}");
    }

    #[test]
    fn head_dim_falls_back_to_division() {
        let hp = HParams::read(&base_llama(), &LLAMA).unwrap();
        assert_eq!(hp.head_dim(), 2048 / 32);
    }

    #[test]
    fn key_length_overrides_head_dim() {
        // Mistral-Nemo / Mistral-Small-3 shape: n_embd/n_head would give
        // 160; the GGUF says 128 and 128 must win.
        let mut m = md(&[
            ("llama.block_count", Value::U32(40)),
            ("llama.embedding_length", Value::U32(5120)),
            ("llama.attention.head_count", Value::U32(32)),
            ("llama.attention.head_count_kv", Value::U32(8)),
        ]);
        m.insert("llama.attention.key_length".into(), Value::U32(128));
        let hp = HParams::read(&m, &LLAMA).unwrap();
        assert_eq!(hp.head_dim(), 128);
    }

    #[test]
    fn head_count_kv_defaults_to_mha() {
        let mut m = base_llama();
        m.remove("llama.attention.head_count_kv");
        let hp = HParams::read(&m, &LLAMA).unwrap();
        assert_eq!(hp.n_head_kv, hp.n_head);
    }

    #[test]
    fn unknown_arch_is_unsupported() {
        assert!(spec_for("llama").is_some());
        assert!(spec_for("LLaMA").is_some());
        assert!(spec_for("mamba").is_none());
    }
}
