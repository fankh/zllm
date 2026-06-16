//! AVX-512 BW vec_dot_q4k_q8k — single-row Q4_K × Q8_K dot product.
//!
//! Port of Candle's AVX2 `vec_dot_q4k_q8k` with 2× wider vectors.
//! Strategy: process **two adjacent super-blocks per outer iteration**
//! by combining their data into 512-bit registers.
//!
//! Per outer iteration we consume `xs[i] + xs[i+1]` and `ys[i] + ys[i+1]`
//! (288 + 584 bytes = 872 bytes), producing two f32 accumulator
//! contributions in one pass. When `nb` is odd we tail-handle the last
//! block with the scalar path so the implementation stays simple.
//!
//! This is decode-only (called per row by QMatMul forward for
//! seq_len = 1). For prefill the existing Candle path is faster
//! because it amortizes across activation rows.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::backend::candle::q4k_repack::{BlockQ4K, BlockQ8K, QK_K};

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

/// Decode the 8 sub-scales from the 12-byte `scales` array into a
/// 16-byte buffer holding `(scale0, scale1, scale2, scale3, scale4,
/// scale5, scale6, scale7, min0, min1, ..., min7)` as u8 each.
/// Mirrors the `utmp` packing in Candle/ggml.
#[inline(always)]
fn unpack_scales(scales: &[u8; 12]) -> [u32; 4] {
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    utmp[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    utmp[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    utmp
}

/// f16 → f32 — use the tested implementation from q4k_repack.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    crate::backend::candle::q4k_repack::f16_to_f32_pub(bits)
}

/// Phase 2: batched Q4_K × Q8_K matmul, designed to amortize weight
/// reads across a batch of M activation rows. Loop order is
/// **(output_col, batch_m, super_block)** — weights for one output row
/// are loaded once per col and reused M times.
///
/// Shape contract:
/// - `weights`: `n_out_rows * nb_per_row` BlockQ4K, row-major
///   (each output row = nb_per_row contiguous super-blocks).
/// - `activations`: `m * nb_per_row` BlockQ8K, row-major
///   (each batch row = nb_per_row contiguous super-blocks).
/// - `output`: `m * n_out_rows` f32, row-major (M, n_out_rows).
///
/// The win: at M=8, FFN w1/w3 (8192 output rows, 8 super-blocks each)
/// reads weights ~9.4 MB ONCE per matmul instead of 8× per matmul.
/// L1/L2 hot weight reuse across the inner M loop.
///
/// Inner kernel: dispatches to AVX-512BW `vec_dot_q4k_q8k` when
/// available; scalar fallback otherwise.
pub fn matmul_q4k_q8k(
    weights: &[BlockQ4K],
    activations: &[BlockQ8K],
    output: &mut [f32],
    n_out_rows: usize,
    nb_per_row: usize,
    m: usize,
) {
    debug_assert_eq!(weights.len(), n_out_rows * nb_per_row,
        "weights len {} != n_out_rows {} * nb_per_row {}",
        weights.len(), n_out_rows, nb_per_row);
    debug_assert_eq!(activations.len(), m * nb_per_row,
        "activations len {} != m {} * nb_per_row {}",
        activations.len(), m, nb_per_row);
    debug_assert_eq!(output.len(), m * n_out_rows);

    // Single-thread MVP. Parallelism added once correctness + amortization
    // confirmed (Phase 2.1).
    for col in 0..n_out_rows {
        let weight = &weights[col * nb_per_row .. (col + 1) * nb_per_row];
        for batch_m in 0..m {
            let act = &activations[batch_m * nb_per_row .. (batch_m + 1) * nb_per_row];
            // SAFETY: each (batch_m, col) writes a distinct cell.
            output[batch_m * n_out_rows + col] = vec_dot_q4k_q8k(weight, act);
        }
    }
}

/// Parallel variant — rayon's persistent pool over disjoint col ranges.
/// Each rayon task holds a chunk of cols; for each col it does M
/// sequential dot products against the M batch activations. Writes to
/// disjoint output cells, no synchronization needed.
pub fn matmul_q4k_q8k_par(
    weights: &[BlockQ4K],
    activations: &[BlockQ8K],
    output: &mut [f32],
    n_out_rows: usize,
    nb_per_row: usize,
    m: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(weights.len(), n_out_rows * nb_per_row);
    debug_assert_eq!(activations.len(), m * nb_per_row);
    debug_assert_eq!(output.len(), m * n_out_rows);

    // Pass pointer as usize to bypass Send check; each thread writes to
    // disjoint output cells (proved by the col-range partitioning).
    let out_addr = output.as_mut_ptr() as usize;

    (0..n_out_rows).into_par_iter()
        .with_min_len(64)
        .with_max_len(256)
        .for_each(|col| {
            let weight = &weights[col * nb_per_row .. (col + 1) * nb_per_row];
            let out_ptr = out_addr as *mut f32;
            for batch_m in 0..m {
                let act = &activations[batch_m * nb_per_row .. (batch_m + 1) * nb_per_row];
                let val = vec_dot_q4k_q8k(weight, act);
                // SAFETY: each (batch_m, col) cell is written exactly once.
                unsafe {
                    *out_ptr.add(batch_m * n_out_rows + col) = val;
                }
            }
        });
}

/// Public dispatcher. Use AVX-512BW when available; otherwise fall back
/// to the scalar reference. The scalar version exists for testing and
/// for non-x86_64 builds.
pub fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
        {
            return unsafe { vec_dot_q4k_q8k_avx512(xs, ys) };
        }
    }
    vec_dot_q4k_q8k_scalar(xs, ys)
}

/// Scalar reference. Slow but obvious. Used to validate the AVX-512
/// kernel and as a portable fallback.
pub fn vec_dot_q4k_q8k_scalar(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    let mut acc = 0.0_f32;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let d_f32 = f16_to_f32(x.d);
        let dmin_f32 = f16_to_f32(x.dmin);
        let yd = y.d;
        let d_total = yd * d_f32;
        let dmin_total = -yd * dmin_f32;

        let utmp = unpack_scales(&x.scales);
        // The unpacked utmp has 8 u8 scales at utmp[0..2] and 8 u8 mins
        // at utmp[2..4], in little-endian byte order.
        let scale_bytes: [u8; 8] = [
            (utmp[0] & 0xff) as u8,
            ((utmp[0] >> 8) & 0xff) as u8,
            ((utmp[0] >> 16) & 0xff) as u8,
            ((utmp[0] >> 24) & 0xff) as u8,
            (utmp[1] & 0xff) as u8,
            ((utmp[1] >> 8) & 0xff) as u8,
            ((utmp[1] >> 16) & 0xff) as u8,
            ((utmp[1] >> 24) & 0xff) as u8,
        ];
        let min_bytes: [u8; 8] = [
            (utmp[2] & 0xff) as u8,
            ((utmp[2] >> 8) & 0xff) as u8,
            ((utmp[2] >> 16) & 0xff) as u8,
            ((utmp[2] >> 24) & 0xff) as u8,
            (utmp[3] & 0xff) as u8,
            ((utmp[3] >> 8) & 0xff) as u8,
            ((utmp[3] >> 16) & 0xff) as u8,
            ((utmp[3] >> 24) & 0xff) as u8,
        ];

        // Sum (paired_bsum × min) — each 32-quant sub-block covers TWO
        // adjacent bsum groups of 16. Candle's AVX2 path does this via
        // _mm_hadd_epi16; we just add pairs explicitly.
        let mut min_sum = 0i32;
        for sub in 0..8 {
            let bs = y.bsums[2 * sub] as i32 + y.bsums[2 * sub + 1] as i32;
            min_sum += bs * min_bytes[sub] as i32;
        }
        acc += dmin_total * (min_sum as f32);

        // Inner: for each 32-quant sub-block, dot(q4_quants, q8_quants) * scale.
        let mut sumi = 0i32;
        for sub in 0..8 {
            let scale = scale_bytes[sub] as i32;
            let q8_base = sub * 32;
            let mut local = 0i32;
            // 32 quants per sub-block. Q4 packing:
            //   sub 0: q[0..32]   = low nibble of qs[0..32]
            //   sub 1: q[32..64]  = high nibble of qs[0..32]
            //   sub 2: q[64..96]  = low nibble of qs[32..64]
            //   sub 3: q[96..128] = high nibble of qs[32..64]
            //   ... pattern: pair (sub, sub+1) uses qs[16*sub .. 16*sub + 32]
            //   Equivalently: outer step of 64 quants = 32 bytes of qs;
            //   low nibbles → sub 2j, high nibbles → sub 2j+1.
            let pair = sub / 2;
            let nibble = sub & 1;
            let qs_base = pair * 32;
            for k in 0..32 {
                let raw = x.qs[qs_base + k];
                let q4 = if nibble == 0 { raw & 0x0F } else { raw >> 4 };
                local += q4 as i32 * y.qs[q8_base + k] as i32;
            }
            sumi += scale * local;
        }
        acc += d_total * (sumi as f32);
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────
// AVX-512 BW path
// ─────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn vec_dot_q4k_q8k_avx512(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    // Accumulator: f32 sums, kept in a ZMM register.
    let mut acc = _mm512_setzero_ps();
    let mut acc_min = 0.0_f32;
    let m4 = _mm512_set1_epi8(0x0F);

    for (x, y) in xs.iter().zip(ys.iter()) {
        let d_f32 = f16_to_f32(x.d);
        let dmin_f32 = f16_to_f32(x.dmin);
        let yd = y.d;
        let d_total = yd * d_f32;
        let dmin_total = -yd * dmin_f32;

        let utmp = unpack_scales(&x.scales);
        let scales_u64 = (utmp[0] as u64) | ((utmp[1] as u64) << 32);
        let mins_u64 = (utmp[2] as u64) | ((utmp[3] as u64) << 32);

        // Mins contribution: each 32-quant sub-block covers 2 bsum groups.
        let mut min_sum = 0i32;
        let mins_bytes = mins_u64.to_le_bytes();
        for sub in 0..8 {
            let bs = y.bsums[2 * sub] as i32 + y.bsums[2 * sub + 1] as i32;
            min_sum += bs * mins_bytes[sub] as i32;
        }
        acc_min += dmin_total * (min_sum as f32);

        // Main: produce a 512-bit i32 sum across 16 lanes, then
        // multiply by `d_total` and accumulate into `acc`.
        //
        // Strategy: process 4 sub-blocks per ZMM iteration. Two outer
        // iterations covers all 8 sub-blocks.
        let scales_bytes = scales_u64.to_le_bytes();

        let q4_ptr = x.qs.as_ptr();
        let q8_ptr = y.qs.as_ptr();

        let mut sumi = _mm512_setzero_si512();

        for jj in 0..2 {
            // Sub-blocks covered: (4*jj, 4*jj+1, 4*jj+2, 4*jj+3).
            //   low  nibbles: subs (4*jj, 4*jj+2)
            //   high nibbles: subs (4*jj+1, 4*jj+3)
            //
            // Q4 data: 64 packed bytes starting at qs[64*jj]. We split:
            //   low_zmm  = (qs[64*jj..64*jj+32] & 0x0F,
            //              qs[64*jj+32..64*jj+64] & 0x0F)
            //   high_zmm = (qs[64*jj..64*jj+32] >> 4,
            //              qs[64*jj+32..64*jj+64] >> 4)
            //
            // The two halves of low_zmm correspond to two different
            // sub-blocks and thus two different scales — we encode that
            // by building a per-lane scale vector.

            let q4_bytes = _mm512_loadu_si512(q4_ptr.add(64 * jj) as *const __m512i);
            let q4_low = _mm512_and_si512(q4_bytes, m4);
            let q4_high = _mm512_and_si512(
                _mm512_srli_epi16::<4>(q4_bytes),
                m4,
            );

            // Q8 data: 4 sub-blocks × 32 bytes = 128 bytes per zmm pair.
            // We need 2 ZMMs for q8 low (sub 4*jj, 4*jj+2)
            // and 2 ZMMs for q8 high (sub 4*jj+1, 4*jj+3).
            //
            // Layout reminder: q8.qs is sub-block contiguous (sub i at
            // qs[32*i..32*(i+1)]).
            let s0 = 4 * jj;
            let q8_low = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(
                    q8_ptr.add(32 * s0) as *const __m256i,
                )),
                _mm256_loadu_si256(q8_ptr.add(32 * (s0 + 2)) as *const __m256i),
            );
            let q8_high = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(
                    q8_ptr.add(32 * (s0 + 1)) as *const __m256i,
                )),
                _mm256_loadu_si256(q8_ptr.add(32 * (s0 + 3)) as *const __m256i),
            );

            // Build per-lane scale vectors: low scales (sub s0, s0+2)
            // broadcast across each 32-lane half.
            let sc_low_a = scales_bytes[s0] as i16;
            let sc_low_b = scales_bytes[s0 + 2] as i16;
            let sc_high_a = scales_bytes[s0 + 1] as i16;
            let sc_high_b = scales_bytes[s0 + 3] as i16;
            // 32 i16 lanes per ZMM; first 16 use _a, next 16 use _b.
            let scale_low_zmm = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set1_epi16(sc_low_a)),
                _mm256_set1_epi16(sc_low_b),
            );
            let scale_high_zmm = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set1_epi16(sc_high_a)),
                _mm256_set1_epi16(sc_high_b),
            );

            // p16 = q4l * q8l → 32 i16 lanes per ZMM
            let p16_low = _mm512_maddubs_epi16(q4_low, q8_low);
            let p16_high = _mm512_maddubs_epi16(q4_high, q8_high);

            // multiply each i16 pair by its scale and accumulate into i32
            let acc_low = _mm512_madd_epi16(scale_low_zmm, p16_low);
            let acc_high = _mm512_madd_epi16(scale_high_zmm, p16_high);

            sumi = _mm512_add_epi32(sumi, acc_low);
            sumi = _mm512_add_epi32(sumi, acc_high);
        }

        let vd = _mm512_set1_ps(d_total);
        let sumi_f = _mm512_cvtepi32_ps(sumi);
        acc = _mm512_fmadd_ps(vd, sumi_f, acc);
    }

    let final_main = _mm512_reduce_add_ps(acc);
    final_main + acc_min
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::candle::q4k_repack::{K_SCALE_SIZE};

    /// Build a synthetic Q4_K block from deterministic seed data so we
    /// can compare the AVX-512 kernel against scalar (and indirectly
    /// against Candle's AVX2 path via end-to-end model test).
    fn mk_q4k(seed: u64) -> BlockQ4K {
        let mut b = BlockQ4K {
            d: 0,
            dmin: 0,
            scales: [0; K_SCALE_SIZE],
            qs: [0; QK_K / 2],
        };
        // d = ~0.05, dmin = ~0.01 (chosen so values stay in range)
        b.d = 0x2A3F;     // ~0.0526 in f16
        b.dmin = 0x1F00;  // ~0.0070 in f16
        for i in 0..K_SCALE_SIZE {
            let v = ((i as u64).wrapping_mul(seed).wrapping_add(13)) & 0xff;
            b.scales[i] = v as u8;
        }
        for i in 0..QK_K / 2 {
            let v = ((i as u64).wrapping_mul(seed).wrapping_add(101)) & 0xff;
            b.qs[i] = v as u8;
        }
        b
    }

    fn mk_q8k(seed: u64) -> BlockQ8K {
        let mut b = BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        b.d = 0.013_f32 + (seed as f32).rem_euclid(7.0) * 0.001;
        for i in 0..QK_K {
            let v = ((i as i64).wrapping_mul(seed as i64).wrapping_add(17)) & 0xff;
            b.qs[i] = (v as i32 - 128) as i8;
        }
        for g in 0..QK_K / 16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += b.qs[g * 16 + k] as i32;
            }
            b.bsums[g] = s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        b
    }

    /// Verbatim port of Candle's k_quants.rs Q4_K × Q8_K scalar vec_dot.
    fn candle_scalar_vec_dot(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];
        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];
        let mut sumf = 0.0_f32;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a_off = 0usize;
            let mut q4_off = 0usize;
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8;
                }
                a_off += 32;
                for l in 0..32 {
                    aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8;
                }
                a_off += 32;
                q4_off += 32;
            }

            utmp[0] = u32::from_le_bytes([x.scales[0], x.scales[1], x.scales[2], x.scales[3]]);
            utmp[1] = u32::from_le_bytes([x.scales[4], x.scales[5], x.scales[6], x.scales[7]]);
            utmp[2] = u32::from_le_bytes([x.scales[8], x.scales[9], x.scales[10], x.scales[11]]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            scales[0..4].copy_from_slice(&utmp[0].to_le_bytes());
            scales[4..8].copy_from_slice(&utmp[1].to_le_bytes());
            mins[0..4].copy_from_slice(&utmp[2].to_le_bytes());
            mins[4..8].copy_from_slice(&utmp[3].to_le_bytes());

            let mut sumi = 0i32;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a_off = 0usize;
            let mut q8_off = 0usize;
            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8_off += 8;
                    a_off += 8;
                }
            }
            let d = f16_to_f32(x.d) * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = f16_to_f32(x.dmin) * y.d;
            sumf -= dmin * sumi as f32;
        }
        sumf + sums.iter().sum::<f32>()
    }

    /// Read production dump and reproduce the bug — same exact bytes,
    /// side-by-side comparison.
    #[test]
    #[ignore]
    fn repro_from_dump() {
        use std::fs;
        let path = "C:/llm-test/bench_serving/vd_dump.bin";
        let bytes = fs::read(path).unwrap();
        let mut off = 0;
        let at = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize; off += 4;
        let n_cols = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize; off += 4;
        let nb_per_row = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize; off += 4;
        let candle_ref = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap()); off += 4;
        println!("dumped row {} n_cols={} nb_per_row={} candle={:.4}",
                 at, n_cols, nb_per_row, candle_ref);

        // Input: n_cols f32.
        let mut input = vec![0.0_f32; n_cols];
        for i in 0..n_cols {
            input[i] = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
            off += 4;
        }
        // Weights: nb_per_row BlockQ4K's.
        let block_sz = std::mem::size_of::<BlockQ4K>();
        let row_blocks: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(
                bytes[off..].as_ptr() as *const BlockQ4K, nb_per_row,
            )
        };
        println!("input[0..5] = {:?}", &input[0..5]);
        println!("blk0.d = 0x{:04x}, blk0.dmin = 0x{:04x}, blk0.scales = {:?}",
                 row_blocks[0].d, row_blocks[0].dmin, &row_blocks[0].scales);

        // Path 1: dequantize + f32 dot
        let mut row_f32 = [0.0_f32; QK_K];
        let mut deq_dot = 0.0_f64;
        for (b, blk) in row_blocks.iter().enumerate() {
            crate::backend::candle::q4k_repack::dequantize_q4k_block(blk, &mut row_f32);
            for k in 0..QK_K {
                deq_dot += (row_f32[k] as f64) * (input[b * QK_K + k] as f64);
            }
        }
        println!("deq_dot = {:.4}", deq_dot as f32);

        // Path 2: vec_dot with quantize_q8_k.
        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; nb_per_row];
        crate::backend::candle::q4k_repack::quantize_q8_k(&input, &mut act);
        println!("act[0].d = {:.6}, act[0].qs[0..4] = {:?}, act[0].bsums[0..4] = {:?}",
                 act[0].d, &act[0].qs[0..4], &act[0].bsums[0..4]);
        let vd_avx = vec_dot_q4k_q8k(row_blocks, &act);
        let vd_scalar = vec_dot_q4k_q8k_scalar(row_blocks, &act);
        println!("vd_avx = {:.4}, vd_scalar = {:.4}", vd_avx, vd_scalar);

        // Path 3: dequantize Q4_K to f32, dequantize Q8_K to f32, dot.
        // This factors out Q8_K quantization noise. Should match vec_dot
        // EXACTLY (modulo float associativity).
        let mut deq_q8_dot = 0.0_f64;
        for (b, blk) in row_blocks.iter().enumerate() {
            crate::backend::candle::q4k_repack::dequantize_q4k_block(blk, &mut row_f32);
            let a = &act[b];
            for k in 0..QK_K {
                deq_q8_dot += (row_f32[k] as f64) * (a.d as f64 * a.qs[k] as f64);
            }
        }
        println!("deq_q8_dot (f32×Q8K) = {:.4}", deq_q8_dot as f32);

        // Path 4: Candle's exact scalar vec_dot implementation, copied
        // verbatim from k_quants.rs:1378-1452. If this returns
        // -0.2084 then OUR vec_dot has a bug. If it returns 0.0051
        // then we agree with Candle scalar but disagree with Candle AVX2.
        let candle_scalar = candle_scalar_vec_dot(row_blocks, &act);
        println!("candle_scalar (verbatim port) = {:.4}", candle_scalar);

        // Path 5: Quantize the same f32 input via Candle's QTensor and
        // diff against our act bytes.
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let dev = Device::Cpu;
        let xt = Tensor::from_vec(input.clone(), (1, n_cols), &dev).unwrap();
        let cqt = QTensor::quantize(&xt, GgmlDType::Q8K).unwrap();
        let cbytes = cqt.data().unwrap();
        let candle_acts: &[BlockQ8K] = unsafe {
            std::slice::from_raw_parts(cbytes.as_ptr() as *const BlockQ8K, nb_per_row)
        };
        let mut max_qs_diff = 0i32;
        let mut act_d_diff = 0.0f32;
        for b in 0..nb_per_row {
            act_d_diff = act_d_diff.max((act[b].d - candle_acts[b].d).abs());
            for k in 0..QK_K {
                let d = (act[b].qs[k] as i32 - candle_acts[b].qs[k] as i32).abs();
                if d > max_qs_diff { max_qs_diff = d; }
            }
        }
        println!("Q8K diff vs Candle: max_qs_diff={max_qs_diff}, max_d_diff={act_d_diff}");
        println!("candle_act[0].d = {:.6} vs ours = {:.6}",
                 candle_acts[0].d, act[0].d);
        println!("candle_act[0].qs[0..4] = {:?} vs ours = {:?}",
                 &candle_acts[0].qs[0..4], &act[0].qs[0..4]);

        // Path 6: our vec_dot on Candle's BlockQ8K input.
        let vd_with_candle_act = vec_dot_q4k_q8k(row_blocks, candle_acts);
        println!("vec_dot(row, Candle Q8K) = {:.4}", vd_with_candle_act);

        // PATH 7: per-sub debug for super-block 0 — compute sumi and
        // min_sum BOTH ways and compare.
        let blk0 = &row_blocks[0];
        let a0 = &act[0];
        let d_f32 = crate::backend::candle::q4k_repack::f16_to_f32_pub(blk0.d);
        let dmin_f32 = crate::backend::candle::q4k_repack::f16_to_f32_pub(blk0.dmin);
        // Get scales/mins via decode_q4k_scale_min (the dequantize path).
        // And via unpack_scales (our path).
        let utmp = unpack_scales(&blk0.scales);
        let scale_bytes = (utmp[0] as u64 | ((utmp[1] as u64) << 32)).to_le_bytes();
        let min_bytes   = (utmp[2] as u64 | ((utmp[3] as u64) << 32)).to_le_bytes();
        println!("\nSuper-block 0 debug:");
        println!("  d={:.6e}, dmin={:.6e}", d_f32, dmin_f32);
        println!("  scales={:?}", &scale_bytes);
        println!("  mins  ={:?}", &min_bytes);

        // Dequantize this super-block and compute per-sub contributions.
        let mut blk0_f32 = [0.0_f32; QK_K];
        crate::backend::candle::q4k_repack::dequantize_q4k_block(blk0, &mut blk0_f32);
        let mut deq_per_sub = [0.0_f64; 8];
        for sub in 0..8 {
            for k in 0..32 {
                let i = sub * 32 + k;
                deq_per_sub[sub] += (blk0_f32[i] as f64) * (a0.d as f64 * a0.qs[i] as f64);
            }
        }
        println!("  deq per-sub: {:?}", deq_per_sub.iter().map(|v| *v as f32).collect::<Vec<_>>());

        // Our vec_dot per-sub: contribution[sub] = a.d*d*scale[sub]*local - a.d*dmin*min[sub]*sub_bsum
        let mut our_per_sub = [0.0_f64; 8];
        for sub in 0..8 {
            let mut local = 0i32;
            let pair = sub / 2;
            let nibble = sub & 1;
            let qs_base = pair * 32;
            for k in 0..32 {
                let raw = blk0.qs[qs_base + k];
                let q4 = if nibble == 0 { raw & 0x0F } else { raw >> 4 };
                local += q4 as i32 * a0.qs[sub * 32 + k] as i32;
            }
            let sub_bsum = a0.bsums[2 * sub] as i32 + a0.bsums[2 * sub + 1] as i32;
            let main = a0.d as f64 * d_f32 as f64 * scale_bytes[sub] as f64 * local as f64;
            let mins = a0.d as f64 * dmin_f32 as f64 * min_bytes[sub] as f64 * sub_bsum as f64;
            our_per_sub[sub] = main - mins;
        }
        println!("  our per-sub: {:?}", our_per_sub.iter().map(|v| *v as f32).collect::<Vec<_>>());

        // Diff
        for sub in 0..8 {
            let diff = our_per_sub[sub] - deq_per_sub[sub];
            if diff.abs() > 1e-5 {
                println!("  SUB {}: DIFF {:.6e} (ours={:.6e} deq={:.6e})",
                         sub, diff, our_per_sub[sub], deq_per_sub[sub]);
            }
        }
        // FULL-ROW VEC_DOT using the per-sub formula (in f64). If this
        // matches Candle but ours doesn't, the vec_dot implementation
        // itself is buggy.
        let mut my_vd = 0.0_f64;
        for (xb, yb) in row_blocks.iter().zip(act.iter()) {
            let d_f = crate::backend::candle::q4k_repack::f16_to_f32_pub(xb.d);
            let dm_f = crate::backend::candle::q4k_repack::f16_to_f32_pub(xb.dmin);
            let utmp = unpack_scales(&xb.scales);
            let sb = (utmp[0] as u64 | ((utmp[1] as u64) << 32)).to_le_bytes();
            let mb = (utmp[2] as u64 | ((utmp[3] as u64) << 32)).to_le_bytes();
            for sub in 0..8 {
                let pair = sub / 2;
                let nibble = sub & 1;
                let qs_base = pair * 32;
                let mut local = 0i32;
                for k in 0..32 {
                    let raw = xb.qs[qs_base + k];
                    let q4 = if nibble == 0 { raw & 0x0F } else { raw >> 4 };
                    local += q4 as i32 * yb.qs[sub * 32 + k] as i32;
                }
                let sub_bsum = yb.bsums[2*sub] as i32 + yb.bsums[2*sub+1] as i32;
                let main = yb.d as f64 * d_f as f64 * sb[sub] as f64 * local as f64;
                let mins = yb.d as f64 * dm_f as f64 * mb[sub] as f64 * sub_bsum as f64;
                my_vd += main - mins;
            }
        }
        println!("my_vd (f64 per-sub-summed) = {:.4}", my_vd as f32);

        // For b=1, do per-sub breakdown to find where exactly we diverge.
        let b = 1;
        let bx = &row_blocks[b];
        let by = &act[b];
        let d_f = crate::backend::candle::q4k_repack::f16_to_f32_pub(bx.d);
        let dm_f = crate::backend::candle::q4k_repack::f16_to_f32_pub(bx.dmin);
        let utmp_b = unpack_scales(&bx.scales);
        let sb_b = (utmp_b[0] as u64 | ((utmp_b[1] as u64) << 32)).to_le_bytes();
        let mb_b = (utmp_b[2] as u64 | ((utmp_b[3] as u64) << 32)).to_le_bytes();
        let mut blk_f32 = [0.0_f32; QK_K];
        crate::backend::candle::q4k_repack::dequantize_q4k_block(bx, &mut blk_f32);
        let mut deq_per_sub_1 = [0.0_f64; 8];
        let mut our_per_sub_1 = [0.0_f64; 8];
        for sub in 0..8 {
            for k in 0..32 {
                let i = sub * 32 + k;
                deq_per_sub_1[sub] += (blk_f32[i] as f64) * (by.d as f64 * by.qs[i] as f64);
            }
            let mut local = 0i32;
            let pair = sub / 2;
            let nibble = sub & 1;
            let qs_base = pair * 32;
            for k in 0..32 {
                let raw = bx.qs[qs_base + k];
                let q4 = if nibble == 0 { raw & 0x0F } else { raw >> 4 };
                local += q4 as i32 * by.qs[sub * 32 + k] as i32;
            }
            let sub_bsum = by.bsums[2 * sub] as i32 + by.bsums[2 * sub + 1] as i32;
            let main = by.d as f64 * d_f as f64 * sb_b[sub] as f64 * local as f64;
            let mins = by.d as f64 * dm_f as f64 * mb_b[sub] as f64 * sub_bsum as f64;
            our_per_sub_1[sub] = main - mins;
        }
        println!("\nSuper-block b={} (y.d={:.6}, d={:.4e}, dmin={:.4e}):", b, by.d, d_f, dm_f);
        println!("  scales={:?} mins={:?}", &sb_b, &mb_b);
        println!("  deq per-sub: {:?}", deq_per_sub_1.iter().map(|v| *v as f32).collect::<Vec<_>>());
        println!("  our per-sub: {:?}", our_per_sub_1.iter().map(|v| *v as f32).collect::<Vec<_>>());
        for sub in 0..8 {
            let d = our_per_sub_1[sub] - deq_per_sub_1[sub];
            if d.abs() > 1e-5 {
                println!("  SUB {}: DIFF {:.6e}", sub, d);
            }
        }

        // The bug: deq_dot ≈ candle, vd_avx != candle.
        println!("\nExpected: candle={:.4}, got vd_avx={:.4}, diff={:.4}",
                 candle_ref, vd_avx, vd_avx - candle_ref);
        println!("Q8K-noise reference: {:.4}, vec_dot diff from it: {:.4}",
                 deq_q8_dot as f32, vd_avx - deq_q8_dot as f32);
    }

    /// REAL ground-truth test: quantize random f32 weights via Candle's
    /// own Q4_K quantizer, then compare our vec_dot against Candle's
    /// QMatMul.forward on the same input. This catches algorithmic
    /// bugs that synthetic byte-fill data misses.
    #[test]
    fn matches_candle_qmatmul_real_q4k() {
        use candle_core::{Device, Tensor, DType};
        use candle_core::quantized::{QTensor, GgmlDType, QMatMul};

        let dev = Device::Cpu;
        // Match a REAL Llama-3.2-1B attention shape: Q projection
        // is 2048×2048. This exercises both n_cols=2048 (nb_per_row=8
        // super-blocks) AND n_rows=2048 (the integration's full
        // output-row loop). The "1448 rows in 2048" diagnostic we saw
        // means we need to exercise the full row range.
        let n_rows = 2048;
        let n_cols = 8 * QK_K; // 2048

        // Build deterministic f32 weights, quantize via Candle.
        let mut w = vec![0.0_f32; n_rows * n_cols];
        for i in 0..(n_rows * n_cols) {
            // mild values to stay in Q4_K's expressible range
            let s = ((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0;
            w[i] = s * 0.5;
        }
        let w_t = Tensor::from_vec(w, (n_rows, n_cols), &dev).unwrap();
        // Build TWO quantizations so we can keep one for byte extraction
        // and feed the other to Candle's QMatMul.
        let qt_bytes = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
        let qt_for_mm = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
        let qmm = QMatMul::from_qtensor(qt_for_mm).unwrap();

        // Build a deterministic f32 activation row (1, 1, n_cols) with
        // magnitudes that match real Llama activations after RMSNorm
        // (max abs around 1.3, not 0.3).
        let mut x = vec![0.0_f32; n_cols];
        for i in 0..n_cols {
            let s = ((i as i64).wrapping_mul(11400714819323198485u64 as i64) & 0xFFFF)
                as f32 / 32768.0 - 1.0;
            x[i] = s * 1.3;
        }
        let x_t = Tensor::from_vec(x.clone(), (1, 1, n_cols), &dev).unwrap();

        // Ground truth: Candle's QMatMul.forward.
        use candle_nn::Module;
        let ref_out = qmm.forward(&x_t).unwrap();
        let ref_vec: Vec<f32> = ref_out
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Our path: extract bytes, cast to &[BlockQ4K], quantize x to Q8_K,
        // call vec_dot per row.
        let bytes = qt_bytes.data().unwrap();
        let block_sz = std::mem::size_of::<BlockQ4K>();
        let nb_per_row = n_cols / QK_K;
        assert_eq!(bytes.len(), n_rows * nb_per_row * block_sz);
        let raw: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const BlockQ4K,
                n_rows * nb_per_row,
            )
        };

        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; nb_per_row];
        crate::backend::candle::q4k_repack::quantize_q8_k(&x, &mut act);

        let mut ours = vec![0.0_f32; n_rows];
        let mut ours_scalar = vec![0.0_f32; n_rows];
        for r in 0..n_rows {
            let w_row = &raw[r * nb_per_row .. (r + 1) * nb_per_row];
            ours[r] = vec_dot_q4k_q8k(w_row, &act);
            ours_scalar[r] = vec_dot_q4k_q8k_scalar(w_row, &act);
        }

        // Compare each row — focus on absolute error since rel error
        // blows up for near-zero values (legitimate noise from int8
        // quant). We tolerate small abs differences.
        let mut max_abs_err = 0.0_f32;
        let mut max_abs_err_scalar = 0.0_f32;
        let mut max_at = 0usize;
        for r in 0..n_rows {
            let e = (ours[r] - ref_vec[r]).abs();
            let e_s = (ours_scalar[r] - ref_vec[r]).abs();
            if e > max_abs_err { max_abs_err = e; max_at = r; }
            if e_s > max_abs_err_scalar { max_abs_err_scalar = e_s; }
        }
        eprintln!(
            "n_rows={n_rows} n_cols={n_cols}: max_abs_err_avx512={:.6} at row {} (candle={:.4} ours={:.4}), max_abs_err_scalar={:.6}",
            max_abs_err, max_at, ref_vec[max_at], ours[max_at], max_abs_err_scalar,
        );
        // Absolute error budget: this is the bit-exact integration
        // case, should be tiny (numerical noise only).
        assert!(max_abs_err < 0.01, "avx-512 abs err {} too high", max_abs_err);
        assert!(max_abs_err_scalar < 0.01, "scalar abs err {} too high", max_abs_err_scalar);
    }

    /// Ground-truth test: compare our vec_dot against a fully
    /// dequantized f32 dot product. This catches bugs that "scalar ==
    /// avx-512" can't (both could be wrong in the same way).
    #[test]
    fn matches_dequantize_then_f32_dot() {
        use crate::backend::candle::q4k_repack::dequantize_q4k_block;
        let xs: Vec<_> = (0..8).map(|i| mk_q4k(3 + i as u64 * 7)).collect();
        let ys: Vec<_> = (0..8).map(|i| mk_q8k(7 + i as u64 * 11)).collect();

        // Ground truth: dequantize both, dot product in f32.
        let mut truth = 0.0_f64;
        let mut buf = [0.0_f32; QK_K];
        for (x, y) in xs.iter().zip(ys.iter()) {
            dequantize_q4k_block(x, &mut buf);
            for k in 0..QK_K {
                let y_f32 = y.qs[k] as f32 * y.d;
                truth += (buf[k] as f64) * (y_f32 as f64);
            }
        }
        let truth = truth as f32;

        let s = vec_dot_q4k_q8k_scalar(&xs, &ys);
        let a = vec_dot_q4k_q8k(&xs, &ys);
        let scale = truth.abs().max(1e-3);
        let err_s = (s - truth).abs() / scale;
        let err_a = (a - truth).abs() / scale;
        assert!(err_s < 1e-3,
            "scalar vs truth: scalar={} truth={} rel_err={}", s, truth, err_s);
        assert!(err_a < 1e-3,
            "avx512 vs truth: avx512={} truth={} rel_err={}", a, truth, err_a);
    }

    #[test]
    fn avx512_matches_scalar_one_block() {
        let xs = vec![mk_q4k(3)];
        let ys = vec![mk_q8k(7)];
        let s = vec_dot_q4k_q8k_scalar(&xs, &ys);
        let a = vec_dot_q4k_q8k(&xs, &ys);
        let err = (s - a).abs() / (s.abs() + 1e-6);
        assert!(err < 1e-4, "one-block: scalar {} avx512 {}", s, a);
    }

    #[test]
    fn avx512_matches_scalar_eight_blocks() {
        // Llama-3.2-1B: hidden=2048, QK_K=256 → 8 blocks per row.
        let xs: Vec<_> = (0..8).map(|i| mk_q4k(3 + i as u64 * 7)).collect();
        let ys: Vec<_> = (0..8).map(|i| mk_q8k(7 + i as u64 * 11)).collect();
        let s = vec_dot_q4k_q8k_scalar(&xs, &ys);
        let a = vec_dot_q4k_q8k(&xs, &ys);
        let err = (s - a).abs() / (s.abs() + 1e-6);
        assert!(err < 1e-4, "eight-block: scalar {} avx512 {}", s, a);
    }

    /// Throughput microbench (ignored by default — run with
    /// `cargo test --release --lib q4k_avx512 -- --ignored --nocapture`).
    /// Measures vec_dot_q4k_q8k calls per second with cache-resident
    /// inputs so memory bandwidth isn't the bottleneck.
    #[test]
    #[ignore]
    fn microbench_throughput() {
        use std::time::Instant;
        let n_blocks = 8;
        let xs: Vec<_> = (0..n_blocks).map(|i| mk_q4k(3 + i as u64 * 7)).collect();
        let ys: Vec<_> = (0..n_blocks).map(|i| mk_q8k(7 + i as u64 * 11)).collect();
        let iters = 200_000;

        // Warm caches.
        let mut sink = 0.0_f32;
        for _ in 0..1000 {
            sink += vec_dot_q4k_q8k(&xs, &ys);
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            sink += vec_dot_q4k_q8k(&xs, &ys);
        }
        let dt = t0.elapsed();
        let calls_per_s = iters as f64 / dt.as_secs_f64();

        // Ops per call: 8 sub-blocks × 32 i8×i8 muladds × 8 super-blocks
        //             = 8 × 32 × 8 = 2048 muladds per super-block,
        //             × 8 super-blocks = 16384 muladds per call.
        let muladds_per_call = 8 * 32 * 8 * n_blocks as u64;
        let gops = (calls_per_s as f64 * muladds_per_call as f64) / 1e9;

        println!(
            "[bench] avx512 vec_dot_q4k_q8k: {:.1}M calls/s, {:.1} GMUL/s (1 muladd = 1 op), sink={}",
            calls_per_s / 1e6, gops, sink,
        );

        // Same for scalar.
        let t0 = Instant::now();
        for _ in 0..iters {
            sink += vec_dot_q4k_q8k_scalar(&xs, &ys);
        }
        let dt = t0.elapsed();
        let scalar_calls_per_s = iters as f64 / dt.as_secs_f64();
        let scalar_gops = (scalar_calls_per_s as f64 * muladds_per_call as f64) / 1e9;
        println!(
            "[bench] scalar  vec_dot_q4k_q8k: {:.1}M calls/s, {:.1} GMUL/s, sink={}",
            scalar_calls_per_s / 1e6, scalar_gops, sink,
        );
        println!(
            "[bench] avx512 speedup over scalar: {:.1}×",
            calls_per_s / scalar_calls_per_s,
        );
    }
}
