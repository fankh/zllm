//! Quantized llama model — fork of `candle_transformers::models::quantized_llama`.
//!
//! ## Why this fork exists (v0.7)
//!
//! Upstream `ModelWeights::forward` runs the whole transformer pass —
//! embed → 32 layers → norm → lm-head → logits — as one opaque op. Its
//! fields and `LayerWeights` are private, so there's no way from
//! outside `candle-transformers` to observe or mutate the residual
//! stream mid-forward. zllm's hook architecture needs that — hooks fire
//! per-layer.
//!
//! This file is a verbatim copy of `candle_transformers::models::quantized_llama`
//! at version 0.10.2, with three additions:
//!
//! 1. `pub struct ModelWeights` keeps a small new method
//!    `forward_with_callback` that runs the same pass but invokes a
//!    closure `(layer_idx, &Tensor)` after each transformer block.
//! 2. The existing `pub fn forward` delegates to it with a no-op
//!    closure, preserving call-site compatibility.
//! 3. The upstream `#[cfg(test)] mod tests` block (~150 lines tied to
//!    candle-transformers' internal test helpers) is dropped.
//!
//! Upstream license: MIT OR Apache-2.0. Copyright holders are the
//! `candle` maintainers (Hugging Face). See
//! <https://github.com/huggingface/candle/blob/main/LICENSE-MIT>.
//!
//! When upgrading Candle: diff this file against the matching
//! upstream version and re-apply the three modifications.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

use crate::backend::candle::q4k_repack::{
    BlockQ4K, BlockQ4Kx8, BlockQ8K, QK_K,
    gemm_q4_k_8x8_q8_k, gemv_q4_k_8x8_q8_k, matvec_q4k_8x8_par,
    quantize_q8_k, repack_q4_k_to_q4_kx8,
};
use crate::backend::candle::q4k_avx512::vec_dot_q4k_q8k;

pub const MAX_SEQ_LEN: usize = 4096;

/// Repacked Q4_K_M weights laid out as `(n_rows/8) * (n_cols/QK_K)` blocks
/// of `BlockQ4Kx8`. Row-group-major: all super-blocks of row-group `g`
/// are contiguous at `blocks[g*nb_per_row .. (g+1)*nb_per_row]`, so the
/// gemv inner loop walks them with cache locality.
#[derive(Debug)]
struct RepackedQ4K {
    blocks: Vec<BlockQ4Kx8>,
    n_rows: usize,
    n_cols: usize,
    nb_per_row: usize,
}

/// Try to repack a `QTensor` from Q4_K row-major into our 8-row
/// interleaved layout. Returns `None` (without erroring) for any
/// QTensor we don't yet handle (wrong dtype, non-2D, indivisible shape).
fn try_repack_q4k(qtensor: &QTensor) -> Result<Option<RepackedQ4K>> {
    if qtensor.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    let dims = qtensor.shape().dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let n_rows = dims[0];
    let n_cols = dims[1];
    if n_rows % 8 != 0 || n_cols % QK_K != 0 {
        return Ok(None);
    }
    let nb_per_row = n_cols / QK_K;

    let bytes = qtensor.data()?;
    let block_sz = std::mem::size_of::<BlockQ4K>();
    if bytes.len() != n_rows * nb_per_row * block_sz {
        return Ok(None);
    }
    // Alignment check — BlockQ4K only needs u16 alignment (2 bytes).
    let ptr = bytes.as_ptr();
    if (ptr as usize) % std::mem::align_of::<BlockQ4K>() != 0 {
        return Ok(None);
    }
    // SAFETY: BlockQ4K is `repr(C)`; layout matches ggml's `block_q4_K`.
    // Size and alignment verified above.
    let raw: &[BlockQ4K] = unsafe {
        std::slice::from_raw_parts(
            ptr as *const BlockQ4K,
            n_rows * nb_per_row,
        )
    };

    let n_groups = n_rows / 8;
    let mut blocks = Vec::with_capacity(n_groups * nb_per_row);
    for g in 0..n_groups {
        for c in 0..nb_per_row {
            let mut eight = [BlockQ4K {
                d: 0, dmin: 0, scales: [0; 12], qs: [0; QK_K / 2],
            }; 8];
            for i in 0..8 {
                eight[i] = raw[(g * 8 + i) * nb_per_row + c];
            }
            blocks.push(repack_q4_k_to_q4_kx8(&eight));
        }
    }
    Ok(Some(RepackedQ4K { blocks, n_rows, n_cols, nb_per_row }))
}

/// Raw Q4_K weight rows kept alongside Candle's QTensor. Enables a
/// decode-only dispatch path that calls our AVX-512 `vec_dot_q4k_q8k`
/// per output row, bypassing Candle's AVX2 vec_dot.
///
/// Memory cost: ~doubles weight footprint for Q4_K layers (we keep
/// both Candle's copy and ours). For Llama-3.2-1B Q4_K_M that's
/// ~400 MB extra. Acceptable for benchmarking; can be made conditional.
#[derive(Debug)]
struct RawQ4K {
    blocks: Vec<BlockQ4K>,
    n_rows: usize,
    n_cols: usize,
    nb_per_row: usize,
}

fn try_extract_raw_q4k(qtensor: &QTensor) -> Result<Option<RawQ4K>> {
    if qtensor.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    let dims = qtensor.shape().dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let n_rows = dims[0];
    let n_cols = dims[1];
    if n_cols % QK_K != 0 {
        return Ok(None);
    }
    let nb_per_row = n_cols / QK_K;
    let bytes = qtensor.data()?;
    let block_sz = std::mem::size_of::<BlockQ4K>();
    if bytes.len() != n_rows * nb_per_row * block_sz {
        return Ok(None);
    }
    let ptr = bytes.as_ptr();
    if (ptr as usize) % std::mem::align_of::<BlockQ4K>() != 0 {
        return Ok(None);
    }
    // Copy blocks out of Candle's storage so we own them and can
    // outlive any borrow on the QTensor.
    let total = n_rows * nb_per_row;
    let raw: &[BlockQ4K] = unsafe {
        std::slice::from_raw_parts(ptr as *const BlockQ4K, total)
    };
    let blocks = raw.to_vec();
    Ok(Some(RawQ4K { blocks, n_rows, n_cols, nb_per_row }))
}

/// Runtime gate. `ZLLM_Q4K_SIMD=1` enables the prefill 8×8 SIMD path
/// (currently a no-op against Candle baseline). Default off.
fn simd_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ZLLM_Q4K_SIMD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Runtime gate for the AVX-512 vec_dot decode path. Default OFF —
/// the kernel is bit-exact correct against Candle (proven by tests +
/// live diff after fixing f16_to_f32 subnormal handling on 2026-06-10)
/// but is empirically ~40% SLOWER than Candle's AVX2 vec_dot in the
/// real model context. Reason: Q4_K matmul is memory-bandwidth bound
/// at decode time; AVX-512 doesn't help because we can't pull data
/// faster, and the 2-super-block-per-ZMM iteration adds overhead.
/// The microbench shows near-peak GFLOPS in cache-resident scenario;
/// real workloads are bandwidth-limited. Code retained for future
/// experimentation (AVX-512 VNNI, prefetch tuning, etc.). Enable with
/// `ZLLM_Q4K_AVX512=1`.
fn avx512_decode_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ZLLM_Q4K_AVX512")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Runtime gate for the threaded repacked-layout (`BlockQ4Kx8`) decode
/// matmul: the llama.cpp-style streaming kernel (8 interleaved rows per
/// gemv, parallel over row-groups). Default **OFF** — measured ~3% slower
/// than Candle's per-row AVX2 path on this box because decode is
/// memory-bandwidth bound (the interleaved layout reads the same bytes,
/// so it can't beat Candle's already-sequential access). Kept behind
/// `ZLLM_Q4K_REPACK=1` as a correct reference kernel (validated bit-exact
/// vs Candle in tests) — also the CPU oracle for the future Vulkan GEMM.
fn repack_decode_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ZLLM_Q4K_REPACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// QMatMul wrapper adding some tracing + an optional Q4_K SIMD fast path.
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
    repacked: Option<Arc<RepackedQ4K>>,
    raw_q4k: Option<Arc<RawQ4K>>,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let repacked = try_repack_q4k(&qtensor)?.map(Arc::new);
        let raw_q4k = try_extract_raw_q4k(&qtensor)?.map(Arc::new);
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span, repacked, raw_q4k })
    }

    /// Decode hot path: input shape `(1, 1, n_cols)` with Q4_K weights.
    /// Quantize the single activation row to `BlockQ8K`, then loop over
    /// output rows calling our AVX-512 `vec_dot_q4k_q8k`. Bypasses
    /// Candle's AVX2 vec_dot entirely.
    fn try_forward_avx512(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        if !avx512_decode_enabled() { return Ok(None); }
        let Some(raw) = self.raw_q4k.as_ref() else { return Ok(None); };
        if xs.dtype() != DType::F32 { return Ok(None); }
        let dims = xs.dims();
        match dims {
            [1, 1, h] | [1, h] if *h == raw.n_cols => {},
            _ => return Ok(None),
        }
        // Pull the input row into contiguous f32. flatten_all is fine
        // here — this is a single-row tensor of ~8KB.
        let input: Vec<f32> = xs.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(input.len(), raw.n_cols);

        // Quantize the activation to BlockQ8K (one row).
        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; raw.nb_per_row];
        quantize_q8_k(&input, &mut act);

        // For each output row, call vec_dot.
        let mut out = vec![0.0_f32; raw.n_rows];
        for row in 0..raw.n_rows {
            let w = &raw.blocks[row * raw.nb_per_row .. (row + 1) * raw.nb_per_row];
            out[row] = vec_dot_q4k_q8k(w, &act);
        }

        // Match input rank: (1, 1, n_rows) or (1, n_rows).
        let mut shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        shape.push(raw.n_rows);
        let t = Tensor::from_vec(out, shape, xs.device())?;
        Ok(Some(t))
    }

    /// Decode hot path over the **interleaved** `BlockQ4Kx8` layout.
    /// Input `(1, 1, n_cols)` / `(1, n_cols)` with Q4_K weights: quantize
    /// the single activation row to `BlockQ8K` once, then run the
    /// threaded 8-rows-per-call gemv (`matvec_q4k_8x8_par`) over all
    /// row-groups. This streams weights sequentially (8 rows interleaved
    /// per super-block) the way llama.cpp does, instead of Candle's
    /// strided per-row vec_dot. Falls through (returns `None`) for any
    /// shape/dtype we don't own.
    fn try_forward_repacked_decode(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        if !repack_decode_enabled() { return Ok(None); }
        let Some(rep) = self.repacked.as_ref() else { return Ok(None); };
        if xs.dtype() != DType::F32 { return Ok(None); }
        let dims = xs.dims();
        match dims {
            [1, 1, h] | [1, h] if *h == rep.n_cols => {}
            _ => return Ok(None),
        }
        debug_assert_eq!(rep.n_rows % 8, 0);

        // Pull the single activation row contiguous and quantize once.
        let input: Vec<f32> = xs.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(input.len(), rep.n_cols);
        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; rep.nb_per_row];
        quantize_q8_k(&input, &mut act);

        let n_groups = rep.n_rows / 8;
        let mut out = vec![0.0f32; rep.n_rows];
        matvec_q4k_8x8_par(&rep.blocks, &act, &mut out, n_groups, rep.nb_per_row);

        let mut shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        shape.push(rep.n_rows);
        let t = Tensor::from_vec(out, shape, xs.device())?;
        Ok(Some(t))
    }

    /// Dispatch to the 8×8 repacked kernel for **prefill** only
    /// (seq_len ≥ PREFILL_MIN_SEQ). Decode (seq_len=1) intentionally
    /// falls through to Candle's row-at-a-time AVX2 path — see
    /// memory `zllm-q4k-repack-layout` for the empirical reason.
    fn try_forward_simd(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        const PREFILL_MIN_SEQ: usize = 4;
        if !simd_enabled() { return Ok(None); }
        let Some(rep) = self.repacked.as_ref() else { return Ok(None); };
        if xs.dtype() != DType::F32 { return Ok(None); }
        let dims = xs.dims();
        // Accept (1, seq, hidden) or (seq, hidden) with seq ≥ threshold.
        let (seq, hidden) = match dims {
            [1, s, h] if *h == rep.n_cols && *s >= PREFILL_MIN_SEQ => (*s, *h),
            [s, h]    if *h == rep.n_cols && *s >= PREFILL_MIN_SEQ => (*s, *h),
            _ => return Ok(None),
        };

        // Flatten contiguous f32 input — shape is (seq, hidden) row-major.
        let input: Vec<f32> = xs.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(input.len(), seq * hidden);

        // Quantize each of `seq` rows independently to nb_per_row BlockQ8K.
        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; seq * rep.nb_per_row];
        for r in 0..seq {
            let row_in = &input[r * hidden .. (r + 1) * hidden];
            let row_out = &mut act[r * rep.nb_per_row .. (r + 1) * rep.nb_per_row];
            quantize_q8_k(row_in, row_out);
        }

        // gemm output: row-major (seq, n_rows). gemm fills column-blocks
        // in 8-row tiles, but the row layout matches what Candle expects.
        let mut out = vec![0.0f32; seq * rep.n_rows];
        gemm_q4_k_8x8_q8_k(
            &rep.blocks, &act, &mut out,
            seq, rep.n_rows, rep.nb_per_row,
        );

        // Restore original rank: (1, seq, n_rows) or (seq, n_rows).
        let mut shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        shape.push(rep.n_rows);
        let t = Tensor::from_vec(out, shape, xs.device())?;
        Ok(Some(t))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        #[cfg(feature = "profile")]
        let t = std::time::Instant::now();
        // Note: try_forward_avx512 (vec_dot Q4_K) is gated by env var
        // ZLLM_Q4K_AVX512 — default OFF because the scalar+AVX-512
        // kernel has a subtle bug on real model data that synthetic
        // test cases miss. The algorithm is close (matches dequant+dot
        // reference within 1e-3 on synthetic) but produces visible
        // logit drift on real Q4_K_M weights. Future work.
        let r = if let Some(t) = self.try_forward_avx512(xs)? {
            Ok(t)
        } else if let Some(t) = self.try_forward_repacked_decode(xs)? {
            Ok(t)
        } else if let Some(t) = self.try_forward_simd(xs)? {
            Ok(t)
        } else {
            self.inner.forward(xs)
        };
        #[cfg(feature = "profile")]
        TIMING.add_qmm(t.elapsed().as_micros() as u64);
        r
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    /// Pre-allocated KV cache. Decode path reads strided views from
    /// this buffer directly via the custom CPU SDPA in `sdpa_cpu`.
    kv_cache: candle_nn::kv_cache::KvCache,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // This call to contiguous ensures that the fast kernel can be called below. It's
            // actually a no-op except when processing the initial prompt so has no significant
            // impact on performance.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Append new K/V into the pre-allocated cache. Returns strided
        // views into the buffer (no copy of history). KvCache requires
        // contiguous source tensors for slice_set.
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let current_len = self.kv_cache.current_seq_len();

        let y = if seq_len == 1 && q.device().is_cpu() {
            // Custom CPU SDPA: walks the strided KV view directly,
            // skips repeat_kv (GQA done by index), no .contiguous() on
            // history, and writes output in (1, 1, n_embd) concat-heads
            // layout so the caller skips transpose+reshape.
            crate::backend::candle::sdpa_cpu::sdpa_gqa_decode(
                &q, &k, &v,
                self.n_head, self.n_kv_head, self.head_dim, current_len,
            )?
        } else if q.device().is_metal() && seq_len == 1 {
            // Metal's SDPA returns (1, n_head, 1, head_dim) — needs the
            // head-merge transpose+reshape below.
            let y = candle_nn::ops::sdpa(
                &q, &k, &v, None, false,
                1. / (self.head_dim as f32).sqrt(), 1.,
            )?;
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?
        } else {
            // Prefill / non-CPU-non-Metal: original matmul path,
            // returns (1, n_head, seq_len, head_dim).
            let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let y = att.matmul(&v.contiguous()?)?;
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?
        };
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// Mask cache keyed by (seq_len, kv_len).
    /// kv_len = index_pos + seq_len, so the mask is rectangular when prefix
    /// KV cache entries exist (index_pos > 0).
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000., &ct.device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: candle_nn::kv_cache::KvCache::new(2, MAX_SEQ_LEN),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: candle_nn::kv_cache::KvCache::new(2, MAX_SEQ_LEN),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    /// Build a causal attention mask of shape `(seq_len, kv_len)` where
    /// `kv_len = index_pos + seq_len`.
    ///
    /// When `index_pos == 0` the mask is square `(seq_len, seq_len)` — the
    /// classic case with an empty KV cache.
    ///
    /// When `index_pos > 0` the KV cache already holds `index_pos` entries from
    /// a previously fed prefix.  The mask becomes rectangular: the first
    /// `index_pos` columns are all 0 (every query attends to every prefix key)
    /// and the remaining `seq_len` columns form the standard causal triangle
    /// (query at global position `index_pos + i` cannot attend to keys at global
    /// positions `> index_pos + i`).
    ///
    /// # Shape example  (index_pos=65, seq_len=4)
    /// ```text
    ///              kv 0..64 (prefix)   kv 65  kv 66  kv 67  kv 68
    /// query 65:       0  0 … 0           0      1      1      1
    /// query 66:       0  0 … 0           0      0      1      1
    /// query 67:       0  0 … 0           0      0      0      1
    /// query 68:       0  0 … 0           0      0      0      0
    /// ```
    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // Delegate to the callback variant with a no-op so we don't duplicate
        // the loop body. Generic over the closure, so monomorphization
        // collapses the empty closure to nothing — no runtime cost.
        self.forward_with_callback(x, index_pos, |_, _| {})
    }

    /// Same as `forward`, but invokes `on_layer(layer_idx, &hidden_state)`
    /// after every transformer block (post attn + post mlp + residual).
    /// This is the v0.7 hook surface: callers (the chat handler, the
    /// inference runner) wrap their `HookRegistry` in a closure here and
    /// observe the residual stream at every depth.
    ///
    /// `hidden_state` is the candle `Tensor` of shape `(1, seq_len, n_embd)`
    /// for the current forward; the callback gets a borrow, so reading
    /// it for memory capture is cheap, and writing back (e.g., steering)
    /// requires building a new tensor and overwriting via a follow-up
    /// API — kept out of v0.7 scope.
    /// Clear every layer's KV cache. Call between independent chat
    /// requests — without this the cache keeps growing across requests
    /// and the position counter eventually exceeds the model's
    /// effective range, producing immediate-EOS responses.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache.reset();
        }
    }

    /// Keep only the first `n` token positions in every layer's KV
    /// cache. Used by prompt-prefix caching: when a new request shares
    /// the first N tokens with the cached request, we truncate to N
    /// and prefill only the new suffix instead of redoing the prefix.
    /// `n == 0` is equivalent to `clear_kv_cache`; `n >= current len`
    /// is a no-op for that layer.
    pub fn truncate_kv_cache(&mut self, n: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            let len = layer.kv_cache.current_seq_len();
            if n >= len { continue; }
            if n == 0 {
                layer.kv_cache.reset();
                continue;
            }
            // Snapshot the prefix to keep, reset, then re-append. The
            // append walks `slice_set` which is a fast in-buffer copy.
            let (k_keep, v_keep) = {
                let k = layer.kv_cache.k()?.unwrap().narrow(2, 0, n)?.contiguous()?;
                let v = layer.kv_cache.v()?.unwrap().narrow(2, 0, n)?.contiguous()?;
                (k, v)
            };
            layer.kv_cache.reset();
            layer.kv_cache.append(&k_keep, &v_keep)?;
        }
        Ok(())
    }

    pub fn forward_with_callback<F: FnMut(usize, &Tensor)>(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        mut on_layer: F,
    ) -> Result<Tensor> {
        #[cfg(feature = "profile")]
        let t_total = std::time::Instant::now();
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let x = layer_in;
            let residual = &x;
            #[cfg(feature = "profile")]
            let t_attn = std::time::Instant::now();
            #[cfg(feature = "profile")]
            timing::QMM_BUCKET.with(|b| b.set(timing::QmmBucket::Attn));
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            #[cfg(feature = "profile")]
            TIMING.add_attn(t_attn.elapsed().as_micros() as u64);

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            #[cfg(feature = "profile")]
            let t_ffn = std::time::Instant::now();
            #[cfg(feature = "profile")]
            timing::QMM_BUCKET.with(|b| b.set(timing::QmmBucket::Ffn));
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            #[cfg(feature = "profile")]
            TIMING.add_ffn(t_ffn.elapsed().as_micros() as u64);
            layer_in = x;

            on_layer(layer_idx, &layer_in);
        }
        #[cfg(feature = "profile")]
        let t_norm = std::time::Instant::now();
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        #[cfg(feature = "profile")]
        TIMING.add_norm(t_norm.elapsed().as_micros() as u64);
        let _enter = self.span_output.enter();
        #[cfg(feature = "profile")]
        let t_lm = std::time::Instant::now();
        #[cfg(feature = "profile")]
        timing::QMM_BUCKET.with(|b| b.set(timing::QmmBucket::LmHead));
        let out = self.output.forward(&x);
        #[cfg(feature = "profile")]
        {
            TIMING.add_lm_head(t_lm.elapsed().as_micros() as u64);
            TIMING.add_forward(t_total.elapsed().as_micros() as u64);
        }
        out
    }

}

#[cfg(feature = "profile")]
pub use timing::TIMING;

#[cfg(feature = "profile")]
mod timing {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub struct Timing {
        forward_us: AtomicU64,
        attn_us: AtomicU64,
        ffn_us: AtomicU64,
        norm_us: AtomicU64,
        lm_us: AtomicU64,
        n_forwards: AtomicU64,
        // Finer matmul splits — sum of these across attn+ffn+lm should
        // approach forward time minus norms/rope/softmax.
        qmm_attn_us: AtomicU64,
        qmm_ffn_us: AtomicU64,
        qmm_lm_us: AtomicU64,
        qmm_n: AtomicU64,
    }
    pub static TIMING: Timing = Timing {
        forward_us: AtomicU64::new(0),
        attn_us: AtomicU64::new(0),
        ffn_us: AtomicU64::new(0),
        norm_us: AtomicU64::new(0),
        lm_us: AtomicU64::new(0),
        n_forwards: AtomicU64::new(0),
        qmm_attn_us: AtomicU64::new(0),
        qmm_ffn_us: AtomicU64::new(0),
        qmm_lm_us: AtomicU64::new(0),
        qmm_n: AtomicU64::new(0),
    };
    /// Which call-site bucket a QMatMul invocation belongs to. Set via
    /// a thread-local before entering the attn / ffn / lm sections so
    /// the wrapper can attribute its time.
    #[derive(Copy, Clone)]
    pub enum QmmBucket { None, Attn, Ffn, LmHead }
    thread_local! {
        pub static QMM_BUCKET: std::cell::Cell<QmmBucket> =
            const { std::cell::Cell::new(QmmBucket::None) };
    }
    pub struct Snapshot {
        pub n_forwards: u64,
        pub total_ms: u64,
        pub attention_ms: u64,
        pub ffn_ms: u64,
        pub norm_ms: u64,
        pub lm_head_ms: u64,
        pub qmm_attn_ms: u64,
        pub qmm_ffn_ms: u64,
        pub qmm_lm_ms: u64,
        pub qmm_calls: u64,
    }
    impl Timing {
        pub fn add_forward(&self, us: u64) {
            self.forward_us.fetch_add(us, Ordering::Relaxed);
            self.n_forwards.fetch_add(1, Ordering::Relaxed);
        }
        pub fn add_attn(&self, us: u64) { self.attn_us.fetch_add(us, Ordering::Relaxed); }
        pub fn add_ffn(&self, us: u64) { self.ffn_us.fetch_add(us, Ordering::Relaxed); }
        pub fn add_norm(&self, us: u64) { self.norm_us.fetch_add(us, Ordering::Relaxed); }
        pub fn add_lm_head(&self, us: u64) { self.lm_us.fetch_add(us, Ordering::Relaxed); }
        pub fn add_qmm(&self, us: u64) {
            self.qmm_n.fetch_add(1, Ordering::Relaxed);
            QMM_BUCKET.with(|b| match b.get() {
                QmmBucket::Attn => { self.qmm_attn_us.fetch_add(us, Ordering::Relaxed); }
                QmmBucket::Ffn => { self.qmm_ffn_us.fetch_add(us, Ordering::Relaxed); }
                QmmBucket::LmHead => { self.qmm_lm_us.fetch_add(us, Ordering::Relaxed); }
                QmmBucket::None => {}
            });
        }
        pub fn snapshot(&self) -> Snapshot {
            Snapshot {
                n_forwards: self.n_forwards.load(Ordering::Relaxed),
                total_ms: self.forward_us.load(Ordering::Relaxed) / 1000,
                attention_ms: self.attn_us.load(Ordering::Relaxed) / 1000,
                ffn_ms: self.ffn_us.load(Ordering::Relaxed) / 1000,
                norm_ms: self.norm_us.load(Ordering::Relaxed) / 1000,
                lm_head_ms: self.lm_us.load(Ordering::Relaxed) / 1000,
                qmm_attn_ms: self.qmm_attn_us.load(Ordering::Relaxed) / 1000,
                qmm_ffn_ms: self.qmm_ffn_us.load(Ordering::Relaxed) / 1000,
                qmm_lm_ms: self.qmm_lm_us.load(Ordering::Relaxed) / 1000,
                qmm_calls: self.qmm_n.load(Ordering::Relaxed),
            }
        }
    }
}

impl ModelWeights {
    /// Forward with a per-layer "should we exit now?" callback. When
    /// the callback returns `true` after a layer, the remaining layers
    /// are skipped, final RMSNorm + LM head run against the current
    /// hidden state, and logits are returned. The callback gets
    /// `(layer_idx, &hidden_at_this_layer)` — same arguments as
    /// `forward_with_callback`'s `on_layer`.
    ///
    /// Returns `(logits, exit_layer)`. `exit_layer` is `n_layers - 1`
    /// when no early exit fired (i.e. full forward ran).
    ///
    /// **KV staleness caveat**: when we exit at layer K, layers
    /// K+1..n have NOT processed this token's K/V. The next forward
    /// will use stale KV for those layers. Acceptable for greedy
    /// chat where the model converges to the same token anyway; not
    /// acceptable for high-precision tasks. Caller's responsibility
    /// to decide.
    pub fn forward_with_early_exit<F: FnMut(usize, &Tensor) -> bool>(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        mut should_exit: F,
    ) -> Result<(Tensor, usize)> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let last_layer = self.layers.len().saturating_sub(1);
        let mut exit_at = last_layer;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
            if layer_idx < last_layer && should_exit(layer_idx, &layer_in) {
                exit_at = layer_idx;
                break;
            }
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        let logits = self.output.forward(&x)?;
        Ok((logits, exit_at))
    }

    /// Diagnostic forward: project the hidden state at EVERY layer
    /// through the final RMSNorm + LM head, returning the top-1 token
    /// at each layer (plus the actual full-forward logits). Costs one
    /// LM head matmul per layer — only use for analysis, not steady
    /// inference. Returns `(final_logits, top1_per_layer)` where the
    /// vec has length `n_layers` and entry `i` is the argmax of
    /// projecting layer `i`'s residual through the head.
    pub fn forward_per_layer_argmax(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let mut per_layer_top1: Vec<u32> = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
            // Project this layer's hidden through final norm + LM head.
            let normed = self.norm.forward(&layer_in)?;
            let last_pos = normed.i((.., seq_len - 1, ..))?;
            let logits = self.output.forward(&last_pos)?;
            let v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > best_v { best_v = x; best = i; }
            }
            per_layer_top1.push(best as u32);
        }
        let normed = self.norm.forward(&layer_in)?;
        let last_pos = normed.i((.., seq_len - 1, ..))?;
        let final_logits = self.output.forward(&last_pos)?;
        Ok((final_logits, per_layer_top1))
    }

    /// Same as `forward_with_callback` but returns logits at **every**
    /// input position, shape `(1, seq_len, vocab)`. Used by spec-decode.
    pub fn forward_all_positions(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

