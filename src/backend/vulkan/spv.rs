//! Compiled SPIR-V shader binaries for the raw-Vulkan backend,
//! extracted from `mod.rs` (V1_PLAN split). Pure `include_bytes!`
//! byte arrays - no logic. Glob-imported by the parent as
//! `use spv::*` so every `*_SPV` reference in `mod.rs` is unchanged.
#![allow(dead_code)]

pub(super) const COOPMAT_MATMUL_SPV: &[u8] = include_bytes!("shaders/coopmat_matmul.spv");
pub(super) const COOPMAT_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_gemm.spv");
pub(super) const COOPMAT_Q4K_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_q4k_gemm.spv");
pub(super) const COOPMAT_Q4K_GEMM_M16_SPV: &[u8] = include_bytes!("shaders/coopmat_q4k_gemm_m16.spv");
pub(super) const DECODE_MATVEC_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q4k.spv");
pub(super) const DECODE_MATVEC_Q3K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q3k.spv"); // gate+up→Q3 (naga-gen)
pub(super) const DECODE_MATVEC_Q4K_ROPE_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q4k_rope.spv");
pub(super) const RMSNORM_SPV: &[u8] = include_bytes!("shaders/rmsnorm.spv");
pub(super) const ROPE_SPV: &[u8] = include_bytes!("shaders/rope.spv");
pub(super) const SDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/sdpa_decode.spv");
pub(super) const SDPA_FLASH_PARTIAL_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial.spv");
pub(super) const SDPA_FLASH_COMBINE_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_combine.spv");
pub(super) const SDPA_FLASH_COMBINE_H_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_combine_h.spv");
pub(super) const SDPA_FLASH_PARTIAL2_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial2.spv");
pub(super) const SDPA_FLASH_BLOCK: usize = 32; // must match BLOCK in the flash shaders
// Use the barrier-lean single-pass decode SDPA up to this depth (must stay < sdpa_decode's
// MAXSEQ=512); only beyond it does the flash partial/combine path's extra parallelism pay
// for its 2 dispatches + combine. The single-pass tree-reduces once, not per position.
pub(super) const FLASH_MIN_SEQ: usize = 256;
// Hierarchical flash combine: each level-1 workgroup merges this many block-partials into one
// super-partial (grid n_head × ceil(nblk/SUPER) => many workgroups, fixing the flat combine's
// occupancy starvation); level 2 merges the few super-partials. ~4-6x faster combine at long ctx.
pub(super) const SDPA_SUPER: usize = 8;
pub(super) const SILU_MUL_SPV: &[u8] = include_bytes!("shaders/silu_mul.spv");
pub(super) const DECODE_MATVEC_Q6K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k.spv");
#[cfg(test)]
pub(super) const DECODE_MATVEC_Q6K_V2_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k_v2.spv");
#[cfg(test)]
pub(super) const GRID_BARRIER_PROBE_SPV: &[u8] = include_bytes!("shaders/grid_barrier_probe.spv");
#[cfg(test)]
pub(super) const DECODE_MATVEC_Q6K_PERSIST_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k_persist.spv");
#[cfg(test)]
pub(super) const Q6K_MEGAKERNEL_PROBE_SPV: &[u8] = include_bytes!("shaders/q6k_megakernel_probe.spv");
#[cfg(test)]
pub(super) const FFN_MEGAKERNEL_SPV: &[u8] = include_bytes!("shaders/ffn_megakernel.spv");
pub(super) const KV_WRITE_SPV: &[u8] = include_bytes!("shaders/kv_write.spv");
// Test-only shaders (ignored GPU tests: slot-indirection, rp-skinny bench, barrier probes).
#[cfg(test)]
pub(super) const DECODE_MATVEC_DOWN_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_down_q4k.spv");
#[cfg(test)]
pub(super) const BSDPA_SLOT_SPV: &[u8] = include_bytes!("shaders/bsdpa_slot.spv");
#[cfg(test)]
pub(super) const KVWRITE_SLOT_SPV: &[u8] = include_bytes!("shaders/kvwrite_slot.spv");
#[cfg(test)]
pub(super) const SKINNY_GEMM_Q4K_RP_SPV: &[u8] = include_bytes!("shaders/skinny_gemm_q4k_rp.spv");
#[cfg(test)]
pub(super) const INC_SPV: &[u8] = include_bytes!("shaders/inc.spv");
#[cfg(test)]
pub(super) const INC_COH_SPV: &[u8] = include_bytes!("shaders/inc_coh.spv");
pub(super) const KV_WRITE_HM_SPV: &[u8] = include_bytes!("shaders/kv_write_hm.spv"); // head-major write (ZLLM_HEADMAJOR_KV)
pub(super) const SDPA_FLASH_PARTIAL_HM_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial_hm.spv"); // head-major partial (naga-gen)
pub(super) const KV_TRANSPOSE_HM_SPV: &[u8] = include_bytes!("shaders/kv_transpose_hm.spv"); // pos→head-major cache transpose (naga-gen)
pub(super) const RESIDUAL_ADD_SPV: &[u8] = include_bytes!("shaders/residual_add.spv");
pub(super) const ARGMAX_SPV: &[u8] = include_bytes!("shaders/argmax.spv");
// Batched prefill kernels.
pub(super) const BNORM_SPV: &[u8] = include_bytes!("shaders/bnorm.spv");
pub(super) const BROPE_SPV: &[u8] = include_bytes!("shaders/brope.spv");
pub(super) const BSDPA_SPV: &[u8] = include_bytes!("shaders/bsdpa.spv");
pub(super) const BSDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/bsdpa_decode.spv");
pub(super) const BSDPA_OFFSET_SPV: &[u8] = include_bytes!("shaders/bsdpa_offset.spv"); // chunked-prefill offset SDPA (naga-gen)
pub(super) const COOPMAT_ATTN_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_attn_gemm.spv"); // prefill attention QK^T on WMMA
pub(super) const COOPMAT_ATTN_GEMM_N64_SPV: &[u8] = include_bytes!("shaders/coopmat_attn_gemm_n64.spv"); // BN=64 variant (PV, N=hd)
pub(super) const CAUSAL_SOFTMAX_SPV: &[u8] = include_bytes!("shaders/causal_softmax.spv");
pub(super) const COOPMAT_FLASH_ATTN_SPV: &[u8] = include_bytes!("shaders/coopmat_flash_attn.spv"); // fused FA (S never global)
pub(super) const BSILU_SPV: &[u8] = include_bytes!("shaders/bsilu.spv");
pub(super) const TO_F16_SPV: &[u8] = include_bytes!("shaders/to_f16.spv");
pub(super) const BMV_Q4K_SPV: &[u8] = include_bytes!("shaders/batched_matvec_q4k.spv");
pub(super) const SKINNY_GEMM_Q4K_SPV: &[u8] = include_bytes!("shaders/skinny_gemm_q4k.spv");
pub(super) const BMV_F16_SPV: &[u8] = include_bytes!("shaders/batched_matvec_f16.spv");
pub(super) const BMV_Q6K_SPV: &[u8] = include_bytes!("shaders/batched_matvec_q6k.spv");
