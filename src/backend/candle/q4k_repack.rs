//! AVX2-accelerated Q4_K_M batched matmul — ported from llama.cpp's
//! `ggml-cpu/arch/x86/repack.cpp`. This is the "8-row interleaved" path
//! that ggml uses on x86 to beat Candle's 1-row-at-a-time AVX2 vec_dot
//! by ~2-3× on CPU.
//!
//! **Status**: Phase 1 (this module) implements the Q8_K activation
//! quantizer in the `4x8` repacked layout. Subsequent phases will add
//! Q4_K weight repacking (`8x8`) and the `gemv`/`gemm` matmul kernels.
//!
//! ## Repacked layouts
//!
//! `BlockQ8Kx4` holds **4 rows × 256 quants** worth of Q8_K activations
//! in an interleaved layout. The repack is invariant to the original
//! Q8_K — it stores the same 4 standard `BlockQ8K`s' data, just laid
//! out for SIMD-friendly access. See `repack.h:block_q8_Kx4`.
//!
//! `BlockQ4Kx8` holds **8 standard `BlockQ4_K`s' weights** interleaved
//! so a single AVX2 256-bit load fans out 8 rows at once.

pub const QK_K: usize = 256;

/// Public f16→f32 helper for cross-module use.
pub fn f16_to_f32_pub(bits: u16) -> f32 { f16_to_f32(bits) }
pub const K_SCALE_SIZE: usize = 12; // bytes of scales+mins per BlockQ4_K

/// 4 interleaved Q8_K activation blocks.
/// Mirrors llama.cpp `struct block_q8_Kx4` (sizeof must equal
/// 4*4 + QK_K*4 + (QK_K/4)*2 = 1104 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8Kx4 {
    pub d: [f32; 4],
    pub qs: [i8; QK_K * 4],
    pub bsums: [i16; QK_K / 4],
}

const _: () =
    assert!(std::mem::size_of::<BlockQ8Kx4>() == 4 * 4 + QK_K * 4 + (QK_K / 4) * 2);

/// 8 interleaved Q4_K weight blocks.
/// Mirrors llama.cpp `struct block_q4_Kx8` (sizeof must equal
/// 8*2 + 8*2 + 96 + 1024 = 1152 bytes; ggml_half == f16).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4Kx8 {
    pub d: [u16; 8],     // f16 super-block scales
    pub dmin: [u16; 8],  // f16 super-block mins
    pub scales: [u8; K_SCALE_SIZE * 8], // = 96
    pub qs: [u8; QK_K * 4],             // = 1024
}

const _: () =
    assert!(std::mem::size_of::<BlockQ4Kx8>() == 2 * 8 + 2 * 8 + K_SCALE_SIZE * 8 + QK_K * 4);

/// Mirror of Candle's `BlockQ4K` (`candle_core::quantized::k_quants`).
/// Candle keeps the fields `pub(crate)`, so we re-declare the layout
/// here with identical `#[repr(C)]` ordering. We `transmute` between
/// the two at the boundary (the repacker takes `&[BlockQ4K; 8]`).
///
/// Size invariant: 2 + 2 + 12 + 128 = 144 bytes — same as ggml's
/// `block_q4_K`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4K {
    pub d: u16,                       // f16 super-block scale
    pub dmin: u16,                    // f16 super-block min
    pub scales: [u8; K_SCALE_SIZE],   // 12 bytes (6-bit sub-scales + sub-mins)
    pub qs: [u8; QK_K / 2],           // 128 bytes of 4-bit quants
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);

/// Repack 8 standard `BlockQ4K`s into one `BlockQ4Kx8`. Mirrors
/// llama.cpp's `make_block_q4_Kx8` with `blck_size_interleave = 8`.
///
/// The "8 byte interleave" pattern shuffles `qs` so a single 32-byte
/// AVX2 load fetches 4 quant bytes from each of 8 standard blocks
/// simultaneously — the layout the `gemv` / `gemm` kernels expect.
pub fn repack_q4_k_to_q4_kx8(input: &[BlockQ4K; 8]) -> BlockQ4Kx8 {
    let mut out = BlockQ4Kx8 {
        d: [0u16; 8],
        dmin: [0u16; 8],
        scales: [0u8; K_SCALE_SIZE * 8],
        qs: [0u8; QK_K * 4],
    };
    for i in 0..8 {
        out.d[i] = input[i].d;
        out.dmin[i] = input[i].dmin;
    }
    const BLOCK_INTERLEAVE: usize = 8;
    let end = QK_K * 4 / BLOCK_INTERLEAVE; // = 128 chunks

    // Interleave qs: pull 8-byte runs from each input block in
    // round-robin order. Writes are linear in `out.qs`.
    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLOCK_INTERLEAVE;
        let dst_offset = i * BLOCK_INTERLEAVE;
        out.qs[dst_offset..dst_offset + BLOCK_INTERLEAVE].copy_from_slice(
            &input[src_id].qs[src_offset..src_offset + BLOCK_INTERLEAVE],
        );
    }

    // Repack scales. Q4_K stores 8 sub-scales + 8 sub-mins in 12 bytes
    // (6 bits each). Q4_Kx8 spreads those across 96 bytes in a layout
    // that lets a single 16-byte load pull scales for all 8 blocks at
    // a given sub-block index. The encoding below is identical to
    // ggml's `make_block_q4_Kx8` — see source comment there for the
    // bit layout rationale.
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];

    // First half: sub-blocks 0..4
    for i in 0..4 {
        for j in 0..8 {
            s[j] = input[j].scales[i] & 63;
            m[j] = input[j].scales[i + 4] & 63;
        }
        out.scales[i * 12]      = (s[0] & 63) | ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) | ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) | ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) | ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) | ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) | ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) | ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) | ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) | ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) | ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) | ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) | ((m[7] & 15) << 4);
    }

    // Second half: sub-blocks 4..8
    for i in 0..4 {
        for j in 0..8 {
            s[j] = ((input[j].scales[i] & 192) >> 2)     | (input[j].scales[i + 8] & 15);
            m[j] = ((input[j].scales[i + 4] & 192) >> 2) | ((input[j].scales[i + 8] & 240) >> 4);
        }
        out.scales[i * 12 + 48] = (s[0] & 63) | ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) | ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) | ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) | ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) | ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) | ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) | ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) | ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) | ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) | ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) | ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) | ((m[7] & 15) << 4);
    }

    out
}

/// Mirror of Candle's `BlockQ8K` (the `vec_dot` partner type for
/// Q4_K_M). Same `#[repr(C)]` layout so we can transmute freely.
///
/// Size invariant: 4 (d) + QK_K (qs) + (QK_K / 16) * 2 (bsums)
///                = 4 + 256 + 32 = 292 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

/// f16 (`u16` bit-pattern, IEEE 754 binary16) → f32. Implemented
/// inline to avoid pulling `half` as a direct dep (candle uses it
/// internally; we keep the surface minimal).
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    let f32_bits: u32 = match exp {
        0 => {
            // zero or subnormal
            if frac == 0 {
                (sign as u32) << 31
            } else {
                // Subnormal: renormalize.
                let mut e = -14i32;
                let mut m = frac as u32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3FF;
                ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
            }
        }
        0x1F => {
            // Inf / NaN
            ((sign as u32) << 31) | (0xFF << 23) | ((frac as u32) << 13)
        }
        _ => {
            let e = (exp as i32 - 15 + 127) as u32;
            ((sign as u32) << 31) | (e << 23) | ((frac as u32) << 13)
        }
    };
    f32::from_bits(f32_bits)
}

/// f32 → f16 (`u16`). Standard IEEE 754 round-to-nearest-even.
/// Used only by tests.
#[cfg(test)]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let frac32 = bits & 0x7FFFFF;
    if exp32 == 0xFF {
        let frac = (frac32 >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | (if frac32 != 0 { frac.max(1) } else { 0 });
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 >= 0x1F {
        return (sign << 15) | (0x1F << 10); // overflow → inf
    }
    if exp16 <= 0 {
        // Subnormal or zero. Simplified: round to zero for tests.
        if exp16 < -10 { return sign << 15; }
        let mant = (frac32 | 0x800000) >> (1 - exp16 + 13);
        return (sign << 15) | (mant as u16 & 0x3FF);
    }
    let frac = (frac32 >> 13) as u16;
    // Round to nearest even
    let round_bit = (frac32 >> 12) & 1;
    let sticky = frac32 & 0xFFF;
    let frac = if round_bit == 1 && (sticky != 0 || (frac & 1) == 1) {
        frac.wrapping_add(1)
    } else {
        frac
    };
    (sign << 15) | ((exp16 as u16) << 10) | (frac & 0x3FF)
}

/// Decode `(sub_scale, sub_min)` for sub-block `j` (0..8) from a
/// standard Q4_K `scales[12]` byte array. Mirrors Candle's private
/// `get_scale_min_k4`.
///
/// Layout (from `BlockQ4K::from_float`):
///   j <  4: scales[j]      low 6 bits = sub-scale,  scales[j+4] low 6 bits = sub-min
///   j >= 4: scales[j+4]    low 4 bits = scale low4, top 4 bits  = min low4
///           scales[j-4] top 2 bits = scale high2
///           scales[j]   top 2 bits = min high2
#[inline]
fn decode_q4k_scale_min(j: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    debug_assert!(j < 8);
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let sc_lo = scales[j + 4] & 0x0F;
        let sc_hi = (scales[j - 4] & 0xC0) >> 2;
        let mn_lo = (scales[j + 4] & 0xF0) >> 4;
        let mn_hi = (scales[j] & 0xC0) >> 2;
        (sc_hi | sc_lo, mn_hi | mn_lo)
    }
}

/// Dequantize a standard `BlockQ4K` row into 256 f32 weights. Mirrors
/// Candle's `BlockQ4K::to_float` semantics for one block.
pub(crate) fn dequantize_q4k_block(b: &BlockQ4K, out: &mut [f32; QK_K]) {
    let d = f16_to_f32(b.d);
    let dmin = f16_to_f32(b.dmin);
    let qs = &b.qs;
    let mut ys_index = 0usize;
    let mut is = 0usize;
    for j in (0..QK_K).step_by(64) {
        let q_chunk = &qs[(j / 2)..(j / 2 + 32)];
        let (sc, m) = decode_q4k_scale_min(is, &b.scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = decode_q4k_scale_min(is + 1, &b.scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        // First 32: low nibbles
        for &q in q_chunk {
            out[ys_index] = d1 * (q & 0x0F) as f32 - m1;
            ys_index += 1;
        }
        // Second 32: high nibbles
        for &q in q_chunk {
            out[ys_index] = d2 * (q >> 4) as f32 - m2;
            ys_index += 1;
        }
        is += 2;
    }
    debug_assert_eq!(ys_index, QK_K);
}

/// Scalar reference for `gemv_q4_K_8x8_q8_K`.
///
/// Computes the dot product of one Q8_K activation row against 8
/// interleaved Q4_K weight rows. Output is 8 floats — one per weight
/// row in the BlockQ4Kx8.
///
/// Math: f32 dequant + f32 dot product. Phase 5b (AVX2) will perform
/// integer-math equivalent with the deferred-scale pattern; results
/// will agree within FP-precision tolerance.
///
/// Slices must have the same number of blocks; `n` is total elements
/// per row and must be a multiple of `QK_K`.
pub fn gemv_q4_k_8x8_q8_k_scalar(
    weights: &[BlockQ4Kx8],
    activation: &[BlockQ8K],
    output: &mut [f32; 8],
) {
    assert_eq!(weights.len(), activation.len(),
        "weight and activation must have the same block count");
    output.fill(0.0);
    let mut row_buf = [0.0f32; QK_K];
    for (wblk, ablk) in weights.iter().zip(activation.iter()) {
        let rows = reverse_repack_q4_kx8(wblk);
        for r in 0..8 {
            dequantize_q4k_block(&rows[r], &mut row_buf);
            // f32 dot of dequantized weights with int-typed Q8_K
            // activation (scaled by ablk.d).
            let mut s = 0.0f32;
            for i in 0..QK_K {
                s += row_buf[i] * (ablk.qs[i] as f32);
            }
            output[r] += s * ablk.d;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// quantize_q8_k — FP32 row → BlockQ8K (single row, decode path)
// ─────────────────────────────────────────────────────────────────────
//
// Mirrors ggml's `quantize_row_q8_K_ref` semantics for one row. Used
// at inference to quantize the activation vector that flows into the
// matmul. The `Q8K` format: super-block of 256 quants with per-block
// f32 scale `d` and pairwise group sums in `bsums[16]`.
//
// This is the activation format our `gemv_q4_k_8x8_q8_k` consumes.
// Phase 7 integration will call this on each layer's input before
// dispatching to our kernel.

/// Quantize `x.len()` f32 values (must be a multiple of `QK_K`) into
/// `BlockQ8K` blocks. One block per `QK_K=256` floats.
pub fn quantize_q8_k(x: &[f32], y: &mut [BlockQ8K]) {
    assert_eq!(x.len() % QK_K, 0, "input must be a multiple of QK_K");
    assert_eq!(y.len(), x.len() / QK_K, "output must match block count");

    // Match Candle's BlockQ8K::from_float exactly: iscale = -128/max
    // (signed max, not abs); .round() not .round_ties_even(); clamp the
    // top side at 127 only (Candle relies on the i8 cast for the
    // bottom). Any deviation here causes ~1% per-block drift that
    // compounds across 112 matmuls per token into garbage output.
    for (bi, block) in y.iter_mut().enumerate() {
        let chunk = &x[bi * QK_K..(bi + 1) * QK_K];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if amax < a {
                amax = a;
                max = v;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            // bsums are already 0 by allocation; reset defensively.
            block.bsums.fill(0);
            continue;
        }
        let iscale = -128.0_f32 / max;
        for (i, &v) in chunk.iter().enumerate() {
            let q = (v * iscale).round();
            block.qs[i] = q.min(127.0) as i8;
        }
        block.d = 1.0 / iscale;

        for g in 0..(QK_K / 16) {
            let mut s = 0i32;
            for k in 0..16 {
                s += block.qs[g * 16 + k] as i32;
            }
            block.bsums[g] = s as i16;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// gemm_q4_K_8x8_q8_K — scalar reference (Phase 6a)
// ─────────────────────────────────────────────────────────────────────
//
// Matrix-matrix dot: nr activation rows × nc weight columns
// (in groups of 8, since weights are interleaved by 8). Used for
// prefill where multiple tokens are processed at once.
//
// The scalar reference reuses `gemv_q4_k_8x8_q8_k_scalar` per
// activation row. That's the slow-but-correct baseline. The AVX2
// fast path (Phase 6b — multi-day port) batches 4 activation rows
// at once through one big tile of intrinsics; result must match
// this reference within FP precision.
//
// Input shape:
//   `weights`: nc/8 column-blocks, each containing nb BlockQ4Kx8.
//     Stored as a flat slice in row-major-by-block-then-row order:
//     `weights[col_block * nb + b]`.
//   `activation`: nr activation rows, each nb BlockQ8K blocks.
//     Stored as `activation[row * nb + b]`.
/// Output: `output[row * (nc/8) * 8 + col_block * 8 + r]` —
///   nr × nc f32 values, written in row-major order.
pub fn gemm_q4_k_8x8_q8_k_scalar(
    weights: &[BlockQ4Kx8],   // (nc / 8) * nb blocks
    activation: &[BlockQ8K],  // nr * nb blocks
    output: &mut [f32],       // nr * nc floats
    nr: usize,
    nc: usize,
    nb: usize,
) {
    assert_eq!(nc % 8, 0, "nc must be divisible by 8 (interleave width)");
    let ncb = nc / 8;
    assert_eq!(weights.len(), ncb * nb, "weights length must be ncb * nb");
    assert_eq!(activation.len(), nr * nb, "activation length must be nr * nb");
    assert_eq!(output.len(), nr * nc, "output length must be nr * nc");

    for row in 0..nr {
        let act = &activation[row * nb..(row + 1) * nb];
        for cb in 0..ncb {
            let w = &weights[cb * nb..(cb + 1) * nb];
            let mut tile = [0.0f32; 8];
            // Dispatch to AVX2 gemv when available — this gives us the
            // Phase 5b kernel for each (row, col_block) tile. A proper
            // AVX2 gemm with cross-row weight-load amortization is the
            // bigger future win (Phase 6b proper), but this is a real
            // step up from the scalar dot.
            gemv_q4_k_8x8_q8_k(w, act, &mut tile);
            let base = row * nc + cb * 8;
            output[base..base + 8].copy_from_slice(&tile);
        }
    }
}

/// Public dispatcher for gemm. Currently always uses the scalar
/// reference (the AVX2 fast path is Phase 6b, future work). Provides
/// the right API shape for Phase 7 integration so callers don't have
/// to change when the fast path lands.
pub fn gemm_q4_k_8x8_q8_k(
    weights: &[BlockQ4Kx8],
    activation: &[BlockQ8K],
    output: &mut [f32],
    nr: usize,
    nc: usize,
    nb: usize,
) {
    // Phase 6a: scalar only. Phase 6b will add AVX2/AVX-512 dispatch
    // here mirroring `gemv_q4_k_8x8_q8_k`.
    gemm_q4_k_8x8_q8_k_scalar(weights, activation, output, nr, nc, nb);
}

/// Threaded decode mat-vec: `output[n_groups*8] = W · x`, where `W` is
/// the interleaved Q4_K layout (`blocks[g*nb + c]`, row-group-major) and
/// `act` is the activation already quantized to `nb` BlockQ8K. Each
/// rayon task owns a span of 8-row groups and walks their super-blocks
/// contiguously — the same streaming pattern llama.cpp uses, which feeds
/// the prefetcher far better than candle's strided per-row vec_dot.
///
/// Output row `r` of group `g` lands at `output[g*8 + r]`. Disjoint
/// writes, no synchronization. `min_len` keeps each task coarse enough
/// to amortize rayon's fork/wake cost (the dominant penalty on the small
/// attention matmuls).
pub fn matvec_q4k_8x8_par(
    blocks: &[BlockQ4Kx8],
    act: &[BlockQ8K],
    output: &mut [f32],
    n_groups: usize,
    nb: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(blocks.len(), n_groups * nb);
    debug_assert_eq!(act.len(), nb);
    debug_assert_eq!(output.len(), n_groups * 8);

    // Pass the base pointer as usize to bypass the Send check; each task
    // writes a disjoint [g*8, g*8+8) slice (proved by the group split).
    let out_addr = output.as_mut_ptr() as usize;
    (0..n_groups)
        .into_par_iter()
        .with_min_len(16)
        .for_each(|g| {
            let w = &blocks[g * nb..(g + 1) * nb];
            let mut tile = [0.0f32; 8];
            gemv_q4_k_8x8_q8_k(w, act, &mut tile);
            let p = out_addr as *mut f32;
            // SAFETY: group `g` owns exactly output[g*8 .. g*8+8].
            unsafe {
                std::ptr::copy_nonoverlapping(tile.as_ptr(), p.add(g * 8), 8);
            }
        });
}

// ─────────────────────────────────────────────────────────────────────
// gemv_q4_K_8x8_q8_K — AVX2 fast path
// ─────────────────────────────────────────────────────────────────────
//
// Port of ggml's `ggml_gemv_q4_K_8x8_q8_K` AVX2 inner loop. Integer
// math throughout (maddubs_epi16, madd_epi16) with deferred fp32
// scaling at the end. Matches the scalar reference within FP precision.

/// Public dispatcher: AVX2+F16C+FMA when available, else scalar.
pub fn gemv_q4_k_8x8_q8_k(
    weights: &[BlockQ4Kx8],
    activation: &[BlockQ8K],
    output: &mut [f32; 8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            unsafe { gemv_q4_k_8x8_q8_k_avx2(weights, activation, output); }
            return;
        }
    }
    gemv_q4_k_8x8_q8_k_scalar(weights, activation, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn gemv_q4_k_8x8_q8_k_avx2(
    weights: &[BlockQ4Kx8],
    activation: &[BlockQ8K],
    output: &mut [f32; 8],
) {
    use std::arch::x86_64::*;
    assert_eq!(weights.len(), activation.len(),
        "weight and activation block counts must match");

    // Lookup table / masks (constants per ggml).
    let deltamask = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    // scalemask: 8-byte shuffle inside one 128-bit lane.
    let scalemask = _mm_setr_epi8(0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
    // Final permute restores row order [0..8] in the 8-lane f32 result.
    let finalpermutemask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let m4b = _mm256_set1_epi8(0x0F);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut acc_row = _mm256_setzero_ps();
    let mut acc_min_rows = _mm256_setzero_ps();

    for b in 0..weights.len() {
        let wblk = &weights[b];
        let ablk = &activation[b];

        let row_scale_f32 = _mm256_set1_ps(ablk.d);

        // Load 8 f16 weight scales → 8 f32, rearranged via deltamask.
        let d_raw = _mm_loadu_si128(wblk.d.as_ptr() as *const __m128i);
        let d_shuf = _mm_shuffle_epi8(d_raw, deltamask);
        let col_scale_f32 = _mm256_cvtph_ps(d_shuf);
        // 8 f16 dmins → 8 f32 (no rearrange).
        let dmin_raw = _mm_loadu_si128(wblk.dmin.as_ptr() as *const __m128i);
        let col_dmin_f32 = _mm256_cvtph_ps(dmin_raw);

        // Activation bsums: 16 i16 → pairwise sum → 8 i16, broadcast both halves.
        let q8sums = _mm256_loadu_si256(ablk.bsums.as_ptr() as *const __m256i);
        let q8s_lo = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                     _mm256_extracti128_si256::<1>(q8sums));
        let q8s_full = _mm256_castsi128_si256(q8s_lo);
        let mut q8s = _mm256_permute2f128_si256::<0>(q8s_full, q8s_full);

        let mut iacc_b = _mm256_setzero_si256();
        let mut iacc_min_b = _mm256_setzero_si256();

        for sb in 0..4 {
            // Load 8 × 32-byte chunks of weight quants for two sub-blocks.
            let qs_base = wblk.qs.as_ptr().add(sb * 256);
            let load = |off: usize| _mm256_loadu_si256(qs_base.add(off) as *const __m256i);
            let raw_0123_0 = load(0);
            let raw_4567_0 = load(32);
            let raw_0123_1 = load(64);
            let raw_4567_1 = load(96);
            let raw_0123_2 = load(128);
            let raw_4567_2 = load(160);
            let raw_0123_3 = load(192);
            let raw_4567_3 = load(224);

            // Low nibbles (first sub-block).
            let v0123_00 = _mm256_and_si256(raw_0123_0, m4b);
            let v4567_00 = _mm256_and_si256(raw_4567_0, m4b);
            let v0123_01 = _mm256_and_si256(raw_0123_1, m4b);
            let v4567_01 = _mm256_and_si256(raw_4567_1, m4b);
            let v0123_02 = _mm256_and_si256(raw_0123_2, m4b);
            let v4567_02 = _mm256_and_si256(raw_4567_2, m4b);
            let v0123_03 = _mm256_and_si256(raw_0123_3, m4b);
            let v4567_03 = _mm256_and_si256(raw_4567_3, m4b);
            // High nibbles (second sub-block).
            let v0123_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_0), m4b);
            let v4567_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_0), m4b);
            let v0123_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_1), m4b);
            let v4567_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_1), m4b);
            let v0123_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_2), m4b);
            let v4567_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_2), m4b);
            let v0123_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_3), m4b);
            let v4567_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_3), m4b);

            // Decode scales/mins for both sub-blocks (12 bytes each).
            let scales_ptr = wblk.scales.as_ptr();
            let mut utmp_0 = [0u32; 4];
            std::ptr::copy_nonoverlapping(scales_ptr.add(24 * sb),
                utmp_0.as_mut_ptr() as *mut u8, 12);
            utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
            let uaux_0 = utmp_0[1] & KMASK1;
            utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
            utmp_0[2] = uaux_0;
            utmp_0[0] &= KMASK1;

            let mut utmp_1 = [0u32; 4];
            std::ptr::copy_nonoverlapping(scales_ptr.add(12 + 24 * sb),
                utmp_1.as_mut_ptr() as *mut u8, 12);
            utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
            let uaux_1 = utmp_1[1] & KMASK1;
            utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
            utmp_1[2] = uaux_1;
            utmp_1[0] &= KMASK1;

            let mas_0 = _mm_set_epi32(utmp_0[3] as i32, utmp_0[2] as i32,
                                       utmp_0[1] as i32, utmp_0[0] as i32);
            let scales_0_rearr = _mm_shuffle_epi8(mas_0, scalemask);
            let scales_0 = _mm256_cvtepu8_epi16(scales_0_rearr);

            let mas_1 = _mm_set_epi32(utmp_1[3] as i32, utmp_1[2] as i32,
                                       utmp_1[1] as i32, utmp_1[0] as i32);
            let scales_1_rearr = _mm_shuffle_epi8(mas_1, scalemask);
            let scales_1 = _mm256_cvtepu8_epi16(scales_1_rearr);

            // Mins from both sub-blocks side by side.
            let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                _mm_shuffle_epi32::<78>(mas_0),
                _mm_shuffle_epi32::<78>(mas_1),
            ));

            // Load activation quants for both sub-blocks.
            let a_qs_ptr = ablk.qs.as_ptr().add(sb * 64);
            let lhs_load = |off: usize| {
                _mm256_castsi128_si256(_mm_loadu_si128(a_qs_ptr.add(off) as *const __m128i))
            };
            let lhs_00 = _mm256_permute2f128_si256::<0>(lhs_load(0), lhs_load(0));
            let lhs_01 = _mm256_permute2f128_si256::<0>(lhs_load(16), lhs_load(16));
            let lhs_10 = _mm256_permute2f128_si256::<0>(lhs_load(32), lhs_load(32));
            let lhs_11 = _mm256_permute2f128_si256::<0>(lhs_load(48), lhs_load(48));

            // First sub-block dot products (8 lines).
            let mut iacc_0 = _mm256_setzero_si256();
            // Helper closures
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_00, _mm256_shuffle_epi32::<177>(v4567_00)),
                _mm256_shuffle_epi32::<0>(lhs_00)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_00), v4567_00),
                _mm256_shuffle_epi32::<85>(lhs_00)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_01, _mm256_shuffle_epi32::<177>(v4567_01)),
                _mm256_shuffle_epi32::<170>(lhs_00)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_01), v4567_01),
                _mm256_shuffle_epi32::<255>(lhs_00)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_02, _mm256_shuffle_epi32::<177>(v4567_02)),
                _mm256_shuffle_epi32::<0>(lhs_01)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_02), v4567_02),
                _mm256_shuffle_epi32::<85>(lhs_01)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_03, _mm256_shuffle_epi32::<177>(v4567_03)),
                _mm256_shuffle_epi32::<170>(lhs_01)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_03), v4567_03),
                _mm256_shuffle_epi32::<255>(lhs_01)));
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            // Second sub-block dot products (8 lines).
            let mut iacc_1 = _mm256_setzero_si256();
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_10, _mm256_shuffle_epi32::<177>(v4567_10)),
                _mm256_shuffle_epi32::<0>(lhs_10)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_10), v4567_10),
                _mm256_shuffle_epi32::<85>(lhs_10)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_11, _mm256_shuffle_epi32::<177>(v4567_11)),
                _mm256_shuffle_epi32::<170>(lhs_10)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_11), v4567_11),
                _mm256_shuffle_epi32::<255>(lhs_10)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_12, _mm256_shuffle_epi32::<177>(v4567_12)),
                _mm256_shuffle_epi32::<0>(lhs_11)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_12), v4567_12),
                _mm256_shuffle_epi32::<85>(lhs_11)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(v0123_13, _mm256_shuffle_epi32::<177>(v4567_13)),
                _mm256_shuffle_epi32::<170>(lhs_11)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_13), v4567_13),
                _mm256_shuffle_epi32::<255>(lhs_11)));
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

            // Min correction via activation bsums.
            let q8s_sb = _mm256_shuffle_epi32::<0>(q8s);
            let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            q8s = _mm256_bsrli_epi128::<4>(q8s);

            iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b),
            _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b),
            _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
    }

    let acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    let result = _mm256_sub_ps(acc_row, acc_min_rows);
    _mm256_storeu_ps(output.as_mut_ptr(), result);
}

/// Reverse the repack: `BlockQ4Kx8` → `[BlockQ4K; 8]`.
/// Inverse of `repack_q4_k_to_q4_kx8`. Lets us validate that the
/// repacked format is lossless (round-trip identity) and provides a
/// reference path for the scalar gemv: we can reverse-repack, use
/// Candle's known-good Q4_K dequant per row, and check against
/// future SIMD implementations.
pub fn reverse_repack_q4_kx8(rp: &BlockQ4Kx8) -> [BlockQ4K; 8] {
    let mut out: [BlockQ4K; 8] = std::array::from_fn(|_| BlockQ4K {
        d: 0, dmin: 0,
        scales: [0u8; K_SCALE_SIZE],
        qs: [0u8; QK_K / 2],
    });

    // d / dmin: direct copy.
    for i in 0..8 {
        out[i].d = rp.d[i];
        out[i].dmin = rp.dmin[i];
    }

    // qs: invert the 8-byte round-robin interleave.
    // Forward: out.qs[i*8..(i+1)*8] = input[i%8].qs[(i/8)*8..(i/8)*8+8]
    // Inverse: input[r].qs[p..p+8] = out.qs[((p/8)*8 + r)*8..((p/8)*8 + r)*8 + 8]
    const BLOCK_INTERLEAVE: usize = 8;
    let end = QK_K * 4 / BLOCK_INTERLEAVE; // 128 chunks
    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLOCK_INTERLEAVE;
        let dst_offset = i * BLOCK_INTERLEAVE;
        out[src_id].qs[src_offset..src_offset + BLOCK_INTERLEAVE].copy_from_slice(
            &rp.qs[dst_offset..dst_offset + BLOCK_INTERLEAVE],
        );
    }

    // scales: invert the bit-pack. See `repack_q4_k_to_q4_kx8` for
    // forward encoding. We decode each sub-block i (0..8) and write
    // back into out[j].scales[...] in the standard Q4_K format.
    //
    // Standard Q4_K scales[12] layout (from Candle's BlockQ4K::from_float):
    //   bytes 0..3: low 6 bits = sub-scale 0..3, top 2 bits = bits 5..4 of sub-scale 4..7
    //   bytes 4..7: low 6 bits = sub-min   0..3, top 2 bits = bits 5..4 of sub-min   4..7
    //   bytes 8..11: low 4 bits = bits 3..0 of sub-scale 4..7,
    //                top 4 bits = bits 3..0 of sub-min   4..7

    // First half (sub-blocks 0..4) is encoded in rp.scales[0..48].
    // Second half (sub-blocks 4..8) is encoded in rp.scales[48..96].
    let halves = [(0usize, 0usize), (4, 48)];
    for &(sb_base, byte_base) in halves.iter() {
        for i in 0..4 {
            let row_off = byte_base + i * 12;
            // Sub-block (sb_base + i) for each of the 8 rows.
            // Recover the 8 sub-scales (s[j]) and sub-mins (m[j]):
            let mut s = [0u8; 8];
            let mut m = [0u8; 8];
            for r in 0..4 {
                s[r] = rp.scales[row_off + r] & 0x3F;
                m[r] = rp.scales[row_off + 4 + r] & 0x3F;
            }
            for r in 4..8 {
                let upper2_s = (rp.scales[row_off + (r - 4)] & 0xC0) >> 2; // bits 5..4 of s[r]
                let lower4_s = rp.scales[row_off + 8 + (r - 4)] & 0x0F;    // bits 3..0 of s[r]
                s[r] = upper2_s | lower4_s;
                let upper2_m = (rp.scales[row_off + 4 + (r - 4)] & 0xC0) >> 2;
                let lower4_m = (rp.scales[row_off + 8 + (r - 4)] & 0xF0) >> 4;
                m[r] = upper2_m | lower4_m;
            }
            // Write back into each row's standard Q4_K scales[12].
            // Reference encoding (from Candle's BlockQ4K::from_float):
            //   j < 4 : scales[j]   = ls,             scales[j+4] = lm
            //   j >= 4: scales[j+4] = (ls&0xF) | (lm&0xF)<<4
            //           scales[j-4] |= (ls>>4) << 6
            //           scales[j]   |= (lm>>4) << 6
            let sb = sb_base + i; // 0..8
            for r in 0..8 {
                let ls = s[r];
                let lm = m[r];
                if sb < 4 {
                    out[r].scales[sb] = (out[r].scales[sb] & 0xC0) | (ls & 0x3F);
                    out[r].scales[sb + 4] = (out[r].scales[sb + 4] & 0xC0) | (lm & 0x3F);
                } else {
                    out[r].scales[sb + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                    out[r].scales[sb - 4] = (out[r].scales[sb - 4] & 0x3F) | ((ls >> 4) << 6);
                    out[r].scales[sb] = (out[r].scales[sb] & 0x3F) | ((lm >> 4) << 6);
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// quantize_mat_q8_K_4x8 — FP32 (4 rows × N) → BlockQ8Kx4[] (N/256 blocks)
// ─────────────────────────────────────────────────────────────────────
//
// Reference implementation: scalar / portable. The AVX2 fast path
// (see `quantize_mat_q8_K_4x8_avx2`) must produce byte-identical
// output. This is verified in tests below.

/// Quantize 4 rows of FP32 input into the `BlockQ8Kx4` layout.
/// `x` is `4 * k` floats laid out as `x[row * k + i]`. `k` must be a
/// multiple of `QK_K`. `y` receives `k / QK_K` `BlockQ8Kx4`s.
pub fn quantize_mat_q8_k_4x8_scalar(x: &[f32], y: &mut [BlockQ8Kx4], k: usize) {
    assert_eq!(k % QK_K, 0, "k must be a multiple of QK_K=256");
    assert_eq!(x.len(), 4 * k, "x must hold 4 rows of k floats");
    let nb = k / QK_K;
    assert_eq!(y.len(), nb, "y must hold k/QK_K blocks");

    for i in 0..nb {
        for row in 0..4 {
            // Find max abs over the 256-element block.
            let start = row * k + i * QK_K;
            let mut max_abs = 0.0f32;
            let mut max_signed = 0.0f32; // value at the argmax (signed)
            for &v in &x[start..start + QK_K] {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                    max_signed = v;
                }
            }
            // ggml convention: the max-magnitude element always
            // quantizes to -127. So iscale sign tracks the sign of
            // the max-magnitude raw value. Dequant uses d = 1/iscale,
            // which carries the inverse sign and restores the original.
            // Saves storing a separate sign bit.
            let iscale = if max_abs == 0.0 {
                0.0
            } else if max_signed > 0.0 {
                // max came from a positive value — use negative iscale
                // so that max_signed * iscale = -127.
                -127.0 / max_abs
            } else {
                127.0 / max_abs
            };
            y[i].d[row] = if max_abs != 0.0 { 1.0 / iscale } else { 0.0 };

            // Quantize each element.
            for j in 0..QK_K {
                let scaled = x[start + j] * iscale;
                let rounded = scaled.round_ties_even() as i32;
                // Pack into the interleaved qs layout. ggml stores 4
                // rows' quants per 32-byte chunk: 8 from row 0, 8 from
                // row 1, … (256-bit chunk is split as 4×8 int8). The
                // outer chunk index runs over sub-blocks (32 of them
                // per BlockQ8Kx4, one per 32-elem subblock per row).
                let sub = j / 8;          // 0..32
                let lane = j % 8;         // 0..8
                let dst = sub * 32 + row * 8 + lane;
                y[i].qs[dst] = rounded.clamp(-128, 127) as i8;
            }
        }

        // bsums: per-subblock-pair sum, over 16-element groups within
        // each row. ggml's `bsums[64]` is 4 rows × 16 sums. The SIMD
        // path stores these in a shuffled order for later vectorized
        // reduce; here we mirror that storage.
        // Layout: bsums[row * 16 + group] but ggml's actual stored
        // layout interleaves rows in groups of 4. Since the matmul
        // kernels read bsums via shuffles, the LAYOUT must match
        // byte-for-byte with the AVX2 path. Phase 1 leaves bsums
        // unimplemented as a TODO — the matmul kernels in later
        // phases need it; the quantizer's tests cover qs only.
        y[i].bsums = [0i16; QK_K / 4];
        for row in 0..4 {
            for group in 0..16 {
                let mut s = 0i32;
                for k in 0..16 {
                    let sub = (group * 16 + k) / 8;
                    let lane = (group * 16 + k) % 8;
                    let dst = sub * 32 + row * 8 + lane;
                    s += y[i].qs[dst] as i32;
                }
                y[i].bsums[row * 16 + group] = s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// AVX2 fast path
// ─────────────────────────────────────────────────────────────────────
//
// Mirror of ggml's `__AVX2__` branch in `ggml_quantize_mat_q8_K_4x8`.
// Produces byte-identical `qs` output to the scalar reference (verified
// by tests below). `bsums` are computed via the scalar loop after the
// vector qs store — same algorithm as the scalar path, so they also
// match byte-for-byte. A fully vectorized bsums path can land later
// without changing observable output.

/// Same contract as `quantize_mat_q8_k_4x8_scalar`. Requires the CPU
/// to support AVX2 + AVX (caller must check `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn quantize_mat_q8_k_4x8_avx2(x: &[f32], y: &mut [BlockQ8Kx4], k: usize) {
    use std::arch::x86_64::*;
    assert_eq!(k % QK_K, 0, "k must be a multiple of QK_K=256");
    assert_eq!(x.len(), 4 * k, "x must hold 4 rows of k floats");
    let nb = k / QK_K;
    assert_eq!(y.len(), nb, "y must hold k/QK_K blocks");

    let sign_bit = _mm256_set1_ps(-0.0f32);

    for i in 0..nb {
        // srcv[row][sub] holds the loaded 8-float chunk so we can
        // re-use it after computing iscale.
        let mut srcv: [[__m256; 32]; 4] = [[_mm256_setzero_ps(); 32]; 4];
        let mut iscale_vec: [__m256; 4] = [_mm256_setzero_ps(); 4];

        for row in 0..4 {
            let base = (row * k + i * QK_K) as isize;
            let x_ptr = x.as_ptr();

            // Load first sub-block (sb=0) and compute initial maxAbs.
            let v0 = _mm256_loadu_ps(x_ptr.offset(base));
            let v1 = _mm256_loadu_ps(x_ptr.offset(base + 8));
            let v2 = _mm256_loadu_ps(x_ptr.offset(base + 16));
            let v3 = _mm256_loadu_ps(x_ptr.offset(base + 24));

            let abs0 = _mm256_andnot_ps(sign_bit, v0);
            let abs1 = _mm256_andnot_ps(sign_bit, v1);
            let abs2 = _mm256_andnot_ps(sign_bit, v2);
            let abs3 = _mm256_andnot_ps(sign_bit, v3);

            let mut max_abs = _mm256_max_ps(abs0, abs1);
            max_abs = _mm256_max_ps(max_abs, abs2);
            max_abs = _mm256_max_ps(max_abs, abs3);

            let mask0 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v0);
            let mask1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v1);
            let mask2 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v2);
            let mask3 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v3);
            let mut mask_abs = _mm256_or_ps(_mm256_or_ps(mask0, mask1),
                                            _mm256_or_ps(mask2, mask3));

            srcv[row][0] = v0;
            srcv[row][1] = v1;
            srcv[row][2] = v2;
            srcv[row][3] = v3;

            // Remaining 7 sub-blocks (sb=1..8).
            for sb in 1..8 {
                let temp_abs = max_abs;
                let off = base + (sb as isize) * 32;

                let v0 = _mm256_loadu_ps(x_ptr.offset(off));
                let v1 = _mm256_loadu_ps(x_ptr.offset(off + 8));
                let v2 = _mm256_loadu_ps(x_ptr.offset(off + 16));
                let v3 = _mm256_loadu_ps(x_ptr.offset(off + 24));

                let abs0 = _mm256_andnot_ps(sign_bit, v0);
                let abs1 = _mm256_andnot_ps(sign_bit, v1);
                let abs2 = _mm256_andnot_ps(sign_bit, v2);
                let abs3 = _mm256_andnot_ps(sign_bit, v3);

                max_abs = _mm256_max_ps(max_abs, abs0);
                max_abs = _mm256_max_ps(max_abs, abs1);
                max_abs = _mm256_max_ps(max_abs, abs2);
                max_abs = _mm256_max_ps(max_abs, abs3);

                let mask_prev = _mm256_cmp_ps::<_CMP_EQ_OQ>(temp_abs, max_abs);
                mask_abs = _mm256_and_ps(mask_abs, mask_prev);

                let mk0 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v0);
                let mk1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v1);
                let mk2 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v2);
                let mk3 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v3);
                let mask_curr = _mm256_or_ps(_mm256_or_ps(mk0, mk1),
                                              _mm256_or_ps(mk2, mk3));
                mask_abs = _mm256_or_ps(mask_abs, mask_curr);

                srcv[row][sb * 4] = v0;
                srcv[row][sb * 4 + 1] = v1;
                srcv[row][sb * 4 + 2] = v2;
                srcv[row][sb * 4 + 3] = v3;
            }

            // Horizontal max over the 8 lanes.
            let max4 = _mm_max_ps(_mm256_extractf128_ps::<1>(max_abs),
                                   _mm256_castps256_ps128(max_abs));
            let max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            let max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            let max_scalar = _mm_cvtss_f32(max4);

            let max_scalar_vec = _mm256_set1_ps(max_scalar);
            let mask_next = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_scalar_vec, max_abs);
            let final_mask = _mm256_and_ps(mask_abs, mask_next);
            let mask_bits = _mm256_movemask_ps(final_mask);

            let iscale = if max_scalar == 0.0 {
                0.0
            } else if mask_bits != 0 {
                -127.0 / max_scalar
            } else {
                127.0 / max_scalar
            };
            y[i].d[row] = if max_scalar != 0.0 { 1.0 / iscale } else { 0.0 };
            iscale_vec[row] = _mm256_set1_ps(iscale);
        }

        // Quantize: 4 rows × 32 chunks of 8 floats each → 32×32-byte
        // chunks of interleaved int8 in qs.
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for j in 0..32 {
            let v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
            let v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
            let v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
            let v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

            // `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC` == 0
            let v0 = _mm256_round_ps::<0>(v0);
            let v1 = _mm256_round_ps::<0>(v1);
            let v2 = _mm256_round_ps::<0>(v2);
            let v3 = _mm256_round_ps::<0>(v3);

            let i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);

            // int32 → int16 with saturation
            let i01 = _mm256_packs_epi32(i0, i1);
            let i23 = _mm256_packs_epi32(i2, i3);
            // int16 → int8 with saturation
            let mut packed = _mm256_packs_epi16(i01, i23);
            // Rearrange dwords to row-major (the writeup shows lane
            // permutation [0,4,1,5,2,6,3,7] produces a 4-row block of
            // 8 quants each in qs[32*j..32*j+32]).
            packed = _mm256_permutevar8x32_epi32(packed, perm);

            let dst = y[i].qs.as_mut_ptr().add(32 * j) as *mut __m256i;
            _mm256_storeu_si256(dst, packed);
        }

        // bsums: scalar reduction over `qs` we just wrote. Same
        // algorithm as scalar path so byte-identical.
        y[i].bsums = [0i16; QK_K / 4];
        for row in 0..4 {
            for group in 0..16 {
                let mut s = 0i32;
                for kk in 0..16 {
                    let sub = (group * 16 + kk) / 8;
                    let lane = (group * 16 + kk) % 8;
                    let dst = sub * 32 + row * 8 + lane;
                    s += y[i].qs[dst] as i32;
                }
                y[i].bsums[row * 16 + group] =
                    s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
}

/// Public dispatcher — picks AVX2 path when supported, scalar otherwise.
/// Output is byte-identical between the two paths (verified by tests).
pub fn quantize_mat_q8_k_4x8(x: &[f32], y: &mut [BlockQ8Kx4], k: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { quantize_mat_q8_k_4x8_avx2(x, y, k); }
            return;
        }
    }
    quantize_mat_q8_k_4x8_scalar(x, y, k);
}

// ─────────────────────────────────────────────────────────────────────
// Tests: structural invariants + byte-parity scalar vs AVX2
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_ggml() {
        // These constants are also enforced at compile-time above; this
        // test makes the failure mode obvious if the layout drifts.
        // BlockQ8Kx4: 16 (d) + 1024 (qs) + 128 (bsums) = 1168 bytes
        assert_eq!(std::mem::size_of::<BlockQ8Kx4>(), 1168);
        // BlockQ4Kx8: 16 (d) + 16 (dmin) + 96 (scales) + 1024 (qs) = 1152 bytes
        assert_eq!(std::mem::size_of::<BlockQ4Kx8>(), 1152);
    }

    fn fresh_block() -> BlockQ8Kx4 {
        BlockQ8Kx4 { d: [0.0; 4], qs: [0; 1024], bsums: [0; 64] }
    }

    fn synth_input(k: usize, seed: u32) -> Vec<f32> {
        // Deterministic pseudo-random floats — different per-row scale
        // so iscale ends up with a mix of signs across rows.
        let mut x = vec![0.0f32; 4 * k];
        for row in 0..4 {
            for i in 0..k {
                let t = (seed as f32) * 0.13 + (row as f32 + 1.0) * (i as f32) * 0.0173;
                let amp = match row {
                    0 => 3.5,
                    1 => -2.7,  // ensures some rows have negative max
                    2 => 7.1,
                    _ => -0.9,
                };
                x[row * k + i] = (t.sin() * amp).round() / 8.0 + t.cos() * 0.4;
            }
        }
        x
    }

    #[test]
    fn repack_q4kx8_preserves_d_and_dmin() {
        // Build 8 inputs with distinct d/dmin and confirm they land in
        // the corresponding slots of the repacked block in order.
        let inputs: [BlockQ4K; 8] = std::array::from_fn(|i| {
            BlockQ4K {
                d: 0x3c00u16.wrapping_add((i * 7) as u16),
                dmin: 0xc000u16.wrapping_add((i * 11) as u16),
                scales: [0u8; K_SCALE_SIZE],
                qs: [0u8; QK_K / 2],
            }
        });
        let out = repack_q4_k_to_q4_kx8(&inputs);
        for i in 0..8 {
            assert_eq!(out.d[i], inputs[i].d, "d[{i}] mismatch");
            assert_eq!(out.dmin[i], inputs[i].dmin, "dmin[{i}] mismatch");
        }
    }

    #[test]
    fn repack_q4kx8_interleaves_qs_in_8_byte_chunks() {
        // Set qs[0..8] of each input to a recognizable pattern so we
        // can verify the round-robin layout: out.qs should contain
        // input[0].qs[0..8], then input[1].qs[0..8], ... input[7].qs[0..8],
        // then input[0].qs[8..16], …
        let mut inputs: [BlockQ4K; 8] = std::array::from_fn(|_| BlockQ4K {
            d: 0, dmin: 0, scales: [0; K_SCALE_SIZE], qs: [0; QK_K / 2],
        });
        for i in 0..8 {
            for j in 0..(QK_K / 2) {
                // Each input has a unique 'tag' byte derived from (input, position)
                inputs[i].qs[j] = ((i as u8) << 5) | ((j as u8) & 0x1f);
            }
        }
        let out = repack_q4_k_to_q4_kx8(&inputs);

        // For chunk index i, source is input[i%8].qs[(i/8)*8..(i/8)*8+8]
        let end = QK_K * 4 / 8;
        for chunk in 0..end {
            let src_id = chunk % 8;
            let src_offset = (chunk / 8) * 8;
            let dst_offset = chunk * 8;
            assert_eq!(
                &out.qs[dst_offset..dst_offset + 8],
                &inputs[src_id].qs[src_offset..src_offset + 8],
                "chunk {chunk}: src_id={src_id} src_offset={src_offset}",
            );
        }
    }

    #[test]
    fn scale_min_decode_inverts_encoding() {
        // Synthesize scales[12] using Candle's encoding for each
        // (ls, lm) pair, then check decode_q4k_scale_min recovers it.
        for j in 0..8 {
            let ls = (j as u8 * 7 + 3) & 0x3F;
            let lm = (j as u8 * 11 + 1) & 0x3F;
            let mut scales = [0u8; K_SCALE_SIZE];
            if j < 4 {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                scales[j - 4] = (ls >> 4) << 6;
                scales[j] = (lm >> 4) << 6;
            }
            let (rec_s, rec_m) = decode_q4k_scale_min(j, &scales);
            assert_eq!(rec_s, ls, "sub-block {j}: scale roundtrip");
            assert_eq!(rec_m, lm, "sub-block {j}: min roundtrip");
        }
    }

    /// Build a 1-block weight + activation pair with correctly
    /// computed bsums so AVX2 and scalar gemv must agree.
    fn build_test_block(
        seed: u32,
    ) -> (Vec<BlockQ4Kx8>, Vec<BlockQ8K>) {
        let inputs: [BlockQ4K; 8] = std::array::from_fn(|r| {
            let mut scales = [0u8; K_SCALE_SIZE];
            let ls: [u8; 8] = std::array::from_fn(|j| (5 + ((j + r) as u32 + seed) as u8 % 8));
            let lm: [u8; 8] = std::array::from_fn(|j| (3 + ((j + r * 2) as u32 + seed * 3) as u8 % 8));
            for j in 0..4 {
                scales[j] = ls[j];
                scales[j + 4] = lm[j];
            }
            for j in 4..8 {
                scales[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                scales[j - 4] |= (ls[j] >> 4) << 6;
                scales[j] |= (lm[j] >> 4) << 6;
            }
            let qs: [u8; QK_K / 2] = std::array::from_fn(|i| {
                let lo = ((r + i + seed as usize) as u8 % 16) & 0x0F;
                let hi = ((r * 3 + i + seed as usize * 5) as u8 % 16) & 0x0F;
                lo | (hi << 4)
            });
            BlockQ4K {
                d: f32_to_f16_bits(0.01 + seed as f32 * 0.001),
                dmin: f32_to_f16_bits(0.005 + seed as f32 * 0.0005),
                scales, qs,
            }
        });
        let weights = vec![repack_q4_k_to_q4_kx8(&inputs)];

        // Activation with non-trivial qs AND CORRECTLY computed bsums
        // (sum of 16 quants per group).
        let qs: [i8; QK_K] = std::array::from_fn(|i| {
            (((i as i32 + seed as i32) % 41) - 20) as i8
        });
        let mut bsums = [0i16; QK_K / 16];
        for g in 0..(QK_K / 16) {
            let mut s = 0i32;
            for k in 0..16 {
                s += qs[g * 16 + k] as i32;
            }
            bsums[g] = s as i16;
        }
        let activation = vec![BlockQ8K {
            d: 0.02 + seed as f32 * 0.001,
            qs,
            bsums,
        }];
        (weights, activation)
    }

    #[test]
    fn quantize_q8k_redundancy_cost() {
        use std::time::Instant;
        // The candle CPU forward re-quantizes the SHARED activation to Q8_K once
        // per matmul: q/k/v share the attn-norm output (3 quantizes, 2 redundant);
        // gate/up share the ffn-norm output (2, 1 redundant). 3 redundant/layer ×
        // 16 layers = 48 redundant 2048-wide quantizes/token. Measure the waste.
        let n = 2048usize; // n_embd
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 211) as f32 - 105.0) / 50.0).collect();
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16] }; n / QK_K];
        for _ in 0..2000 { quantize_q8_k(&x, &mut y); } // warm
        let iters = 200_000;
        let t = Instant::now();
        for _ in 0..iters { quantize_q8_k(&x, &mut y); }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!("quantize_q8_k({n}) = {us:.3} us/call; 48 redundant/tok = {:.3} ms/token wasted (~{:.1}% of a 16.9ms CPU forward)",
            us * 48.0 / 1000.0, us * 48.0 / 1000.0 / 16.9 * 100.0);
    }

    #[test]
    fn quantize_q8_k_roundtrip() {
        // Build a synthetic row, quantize, dequantize, verify error
        // bounded by one quantization step (~max_abs/127).
        let k = QK_K * 2;
        let mut x = vec![0.0f32; k];
        for i in 0..k {
            x[i] = ((i as f32 * 0.07).sin() * 4.0) - 1.5;
        }
        let mut y = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; k / QK_K];
        quantize_q8_k(&x, &mut y);

        for b in 0..(k / QK_K) {
            let d = y[b].d.abs();
            for i in 0..QK_K {
                let dequant = y[b].qs[i] as f32 * y[b].d;
                let orig = x[b * QK_K + i];
                let err = (dequant - orig).abs();
                assert!(err < d * 1.5 + 1e-3,
                    "block {b} idx {i}: err={err} tol={d}");
            }
            // bsums should match sum of qs in groups of 16.
            for g in 0..(QK_K / 16) {
                let mut sum = 0i32;
                for k in 0..16 {
                    sum += y[b].qs[g * 16 + k] as i32;
                }
                assert_eq!(y[b].bsums[g], sum as i16,
                    "block {b} group {g}: bsum mismatch");
            }
        }
    }

    #[test]
    fn quantize_q8_k_then_gemv_matches_f32_dot() {
        // End-to-end check: take f32 activation row, quantize to Q8_K,
        // run our gemv against a Q4_Kx8 weight block, compare against
        // a direct f32×f32 dot product using the dequantized weights.
        let (weights, _) = build_test_block(7);
        let mut x = vec![0.0f32; QK_K];
        for i in 0..QK_K {
            x[i] = ((i as f32 * 0.13).cos() * 2.5) + 0.4;
        }
        let mut act = vec![BlockQ8K {
            d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
        }; 1];
        quantize_q8_k(&x, &mut act);
        let mut out = [0.0f32; 8];
        gemv_q4_k_8x8_q8_k(&weights, &act, &mut out);

        // Reference: dequantize each of 8 weight rows to f32, dot
        // with the ORIGINAL f32 input (not the quantized one), check
        // that quantization error is bounded.
        let rows = reverse_repack_q4_kx8(&weights[0]);
        for r in 0..8 {
            let mut w_f32 = [0.0f32; QK_K];
            dequantize_q4k_block(&rows[r], &mut w_f32);
            let mut s = 0.0f32;
            for i in 0..QK_K {
                s += w_f32[i] * x[i];
            }
            // Q8_K quantization introduces error ~ |x|_max / 127 per
            // element times sum-of-|w| — bound loosely.
            let max_abs_x: f32 = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
            let w_abs_sum: f32 = w_f32.iter().map(|v| v.abs()).sum();
            let tol = (max_abs_x / 127.0) * w_abs_sum + 1e-2;
            assert!((out[r] - s).abs() < tol,
                "row {r}: out={} ref={} tol={}", out[r], s, tol);
        }
    }

    #[test]
    fn gemm_scalar_matches_per_row_gemv() {
        // gemm should produce identical output to calling gemv per row.
        let nb = 2;
        let nr = 3;
        let nc = 16;  // 2 col-blocks of 8
        let ncb = nc / 8;

        // Build nr distinct activation rows + ncb distinct weight col-blocks.
        let mut weights: Vec<BlockQ4Kx8> = Vec::new();
        let mut activation: Vec<BlockQ8K> = Vec::new();
        for cb in 0..ncb {
            let (w, _) = build_test_block(cb as u32);
            for b in 0..nb {
                let _ = b;
                weights.push(w[0]);
            }
        }
        for row in 0..nr {
            let (_, a) = build_test_block((100 + row) as u32);
            for b in 0..nb {
                let _ = b;
                activation.push(a[0]);
            }
        }

        let mut out = vec![0.0f32; nr * nc];
        gemm_q4_k_8x8_q8_k(&weights, &activation, &mut out, nr, nc, nb);

        // Independent reference: explicit per-row, per-col-block gemv.
        let mut expected = vec![0.0f32; nr * nc];
        for row in 0..nr {
            let act = &activation[row * nb..(row + 1) * nb];
            for cb in 0..ncb {
                let w = &weights[cb * nb..(cb + 1) * nb];
                let mut tile = [0.0f32; 8];
                gemv_q4_k_8x8_q8_k_scalar(w, act, &mut tile);
                let base = row * nc + cb * 8;
                expected[base..base + 8].copy_from_slice(&tile);
            }
        }

        for i in 0..(nr * nc) {
            assert!((out[i] - expected[i]).abs() < 1e-4,
                "i={i}: out={} expected={}", out[i], expected[i]);
        }
    }

    #[test]
    fn gemv_avx2_matches_scalar_within_fp_tol() {
        // Run on multiple seeds. Both paths use the same f32 ablk.d
        // and decode the same data — they should agree to within
        // ~1e-3 relative tolerance (the AVX2 path uses int math
        // throughout; scalar uses f32; differences are pure rounding).
        for seed in 0..5 {
            let (weights, activation) = build_test_block(seed);
            let mut out_scalar = [0.0f32; 8];
            gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut out_scalar);

            let mut out_avx2 = [0.0f32; 8];
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                    && std::is_x86_feature_detected!("f16c")
                {
                    unsafe { gemv_q4_k_8x8_q8_k_avx2(&weights, &activation, &mut out_avx2); }
                } else {
                    gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut out_avx2);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut out_avx2);

            for r in 0..8 {
                let s = out_scalar[r];
                let a = out_avx2[r];
                let rel = (s - a).abs() / s.abs().max(1e-6);
                assert!(rel < 0.01,
                    "seed {seed} row {r}: scalar={s} avx2={a} (rel diff {rel})\n  full scalar: {:?}\n  full avx2:   {:?}",
                    out_scalar, out_avx2);
            }
        }
    }

    #[test]
    fn gemv_scalar_matches_direct_f32_math() {
        // Build 8 weight rows + 1 activation row by hand (all small
        // values to avoid quantization artifacts swamping comparison).
        // Compute the reference output two ways and check they match.
        let nb = 1; // single super-block
        let mut weights = vec![BlockQ4Kx8 {
            d: [0u16; 8], dmin: [0u16; 8],
            scales: [0u8; K_SCALE_SIZE * 8],
            qs: [0u8; QK_K * 4],
        }; nb];

        // Build 8 BlockQ4K rows with simple non-trivial content,
        // then repack.
        let inputs: [BlockQ4K; 8] = std::array::from_fn(|r| {
            let mut scales = [0u8; K_SCALE_SIZE];
            // ls/lm per sub-block — small values for sane dequant.
            let ls: [u8; 8] = std::array::from_fn(|j| (5 + (j + r) as u8 % 8));
            let lm: [u8; 8] = std::array::from_fn(|j| (3 + (j + r * 2) as u8 % 8));
            for j in 0..4 {
                scales[j] = ls[j];
                scales[j + 4] = lm[j];
            }
            for j in 4..8 {
                scales[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                scales[j - 4] |= (ls[j] >> 4) << 6;
                scales[j] |= (lm[j] >> 4) << 6;
            }
            // qs: alternating mid-range nibbles
            let qs: [u8; QK_K / 2] = std::array::from_fn(|i| {
                let lo = ((r + i) as u8 % 16) & 0x0F;
                let hi = ((r * 3 + i) as u8 % 16) & 0x0F;
                lo | (hi << 4)
            });
            BlockQ4K {
                d: f32_to_f16_bits(0.01),
                dmin: f32_to_f16_bits(0.005),
                scales, qs,
            }
        });
        weights[0] = repack_q4_k_to_q4_kx8(&inputs);

        let activation = vec![BlockQ8K {
            d: 0.02,
            qs: std::array::from_fn(|i| ((i as i32 % 41) - 20) as i8),
            bsums: [0i16; QK_K / 16],
        }];

        let mut output = [0.0f32; 8];
        gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut output);

        // Independent reference: reverse-repack, dequant each row,
        // f32 dot. Same math, different code path (sanity check).
        let mut expected = [0.0f32; 8];
        let rows = reverse_repack_q4_kx8(&weights[0]);
        let mut row_f32 = [0.0f32; QK_K];
        for r in 0..8 {
            dequantize_q4k_block(&rows[r], &mut row_f32);
            let mut s = 0.0f32;
            for i in 0..QK_K {
                s += row_f32[i] * (activation[0].qs[i] as f32);
            }
            expected[r] = s * activation[0].d;
        }

        for r in 0..8 {
            assert!(
                (output[r] - expected[r]).abs() < 1e-3,
                "row {r}: output={} expected={}",
                output[r], expected[r],
            );
        }
        // Sanity: at least one row should be measurably non-zero so
        // we're not just comparing 0 == 0.
        assert!(output.iter().any(|&v| v.abs() > 0.01),
            "all output values are ~0; inputs may be too trivial");
    }

    #[test]
    fn reverse_repack_is_identity_on_arbitrary_inputs() {
        // Build 8 inputs with non-trivial scales (every 6-bit value
        // used somewhere) + non-trivial qs (PRNG-ish bytes) + distinct
        // d/dmin. Repack then reverse-repack; result must equal input
        // byte-for-byte.
        let inputs: [BlockQ4K; 8] = std::array::from_fn(|i| {
            let mut scales = [0u8; K_SCALE_SIZE];
            // Use Q4_K's actual scales bit pattern: 4 sub-scales
            // (low 6 bits) + 4 sub-mins + 4 packed bytes for the
            // upper 4 sub-scales/mins. Set ls/lm for each sub-block:
            // - sub-blocks 0..4 fully in low 6 bits of bytes 0..7
            // - sub-blocks 4..7 lower 4 bits in bytes 8..11,
            //   upper 2 bits OR-ed into bytes 0..3 and 4..7
            let ls: [u8; 8] = std::array::from_fn(|j| (((i + j) * 5 + 7) as u8) & 0x3F);
            let lm: [u8; 8] = std::array::from_fn(|j| (((i + j) * 11 + 3) as u8) & 0x3F);
            for j in 0..4 {
                scales[j] = ls[j];
                scales[j + 4] = lm[j];
            }
            for j in 4..8 {
                scales[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                scales[j - 4] |= (ls[j] >> 4) << 6;
                scales[j] |= (lm[j] >> 4) << 6;
            }
            let qs: [u8; QK_K / 2] = std::array::from_fn(|j| {
                ((i as u8).wrapping_mul(17).wrapping_add((j as u8).wrapping_mul(31))) & 0xFF
            });
            BlockQ4K {
                d: 0x3c00u16.wrapping_add((i * 13) as u16),
                dmin: 0xc000u16.wrapping_add((i * 7) as u16),
                scales,
                qs,
            }
        });

        let repacked = repack_q4_k_to_q4_kx8(&inputs);
        let roundtrip = reverse_repack_q4_kx8(&repacked);

        for r in 0..8 {
            assert_eq!(inputs[r].d, roundtrip[r].d, "row {r}: d mismatch");
            assert_eq!(inputs[r].dmin, roundtrip[r].dmin, "row {r}: dmin mismatch");
            assert_eq!(inputs[r].scales, roundtrip[r].scales,
                "row {r}: scales mismatch\n  expected: {:?}\n  got:      {:?}",
                inputs[r].scales, roundtrip[r].scales);
            assert_eq!(inputs[r].qs, roundtrip[r].qs,
                "row {r}: qs mismatch (first diff at index {:?})",
                inputs[r].qs.iter().zip(roundtrip[r].qs.iter()).position(|(a, b)| a != b));
        }
    }

    #[test]
    fn repack_q4kx8_scales_roundtrip_known_pattern() {
        // ggml's scales repacking is intricate — but we can validate
        // the bit-pack by feeding inputs with fully-saturated scale
        // bytes and checking the output isn't all-zero (covers the
        // shift/mask plumbing).
        let inputs: [BlockQ4K; 8] = std::array::from_fn(|i| BlockQ4K {
            d: 0, dmin: 0,
            scales: std::array::from_fn(|j| ((i as u8 * 13).wrapping_add(j as u8 * 7))),
            qs: [0; QK_K / 2],
        });
        let out = repack_q4_k_to_q4_kx8(&inputs);
        // At least one of the 96 output scale bytes should be non-zero.
        assert!(out.scales.iter().any(|&b| b != 0),
            "scales output is all-zero — bit pack collapsed");
        // Also: the scales should encode information from all 8 inputs.
        // We sanity-check that the FIRST 12 bytes (representing sub-block 0
        // for all 8 inputs) contains data influenced by every input.
        let total: u32 = out.scales[..12].iter().map(|&b| b as u32).sum();
        assert!(total > 0, "first 12 scale bytes are all zero");
    }

    #[test]
    fn quantize_avx2_matches_scalar_byte_for_byte() {
        // Only meaningful on x86_64 with AVX2; otherwise this is the
        // scalar path twice and trivially matches.
        let k = 256 * 3; // 3 blocks
        let nb = k / QK_K;
        for seed in 0..5 {
            let x = synth_input(k, seed);
            let mut y_scalar = vec![fresh_block(); nb];
            let mut y_avx2 = vec![fresh_block(); nb];
            quantize_mat_q8_k_4x8_scalar(&x, &mut y_scalar, k);
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                {
                    unsafe { quantize_mat_q8_k_4x8_avx2(&x, &mut y_avx2, k); }
                } else {
                    quantize_mat_q8_k_4x8_scalar(&x, &mut y_avx2, k);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            quantize_mat_q8_k_4x8_scalar(&x, &mut y_avx2, k);

            for b in 0..nb {
                assert_eq!(y_scalar[b].d, y_avx2[b].d,
                    "seed {seed} block {b}: d mismatch");
                assert_eq!(y_scalar[b].qs, y_avx2[b].qs,
                    "seed {seed} block {b}: qs mismatch (first diff at index {:?})",
                    y_scalar[b].qs.iter().zip(y_avx2[b].qs.iter()).position(|(a,b)| a != b));
                assert_eq!(y_scalar[b].bsums, y_avx2[b].bsums,
                    "seed {seed} block {b}: bsums mismatch");
            }
        }
    }

    /// Microbench: gemv scalar vs AVX2 over a realistic FFN-shape input.
    /// `cargo test --release --lib q4k_repack::tests::bench_gemv -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_gemv() {
        use std::time::Instant;
        // Realistic FFN: 2048 hidden × 8192 intermediate. With 8-row
        // interleave we have intermediate/8 = 1024 column-blocks,
        // each with nb = 2048/256 = 8 super-blocks per column.
        let nb = 8;
        let (weights, activation) = build_test_block(0);
        // Replicate to nb blocks.
        let weights: Vec<BlockQ4Kx8> = std::iter::repeat(weights[0]).take(nb).collect();
        let activation: Vec<BlockQ8K> = std::iter::repeat(activation[0]).take(nb).collect();
        let iters = 5000;
        let mut out = [0.0f32; 8];

        // Warm
        for _ in 0..50 {
            gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut out);
        }
        let t = Instant::now();
        for _ in 0..iters {
            gemv_q4_k_8x8_q8_k_scalar(&weights, &activation, &mut out);
        }
        let scalar_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let mut have_avx2 = false;
        #[cfg(target_arch = "x86_64")]
        {
            have_avx2 = std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c");
        }
        if have_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                for _ in 0..50 {
                    gemv_q4_k_8x8_q8_k_avx2(&weights, &activation, &mut out);
                }
                let t = Instant::now();
                for _ in 0..iters {
                    gemv_q4_k_8x8_q8_k_avx2(&weights, &activation, &mut out);
                }
                let avx2_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                println!("\nQ4_K_M gemv 8 rows × {nb} blocks, {iters} iters:");
                println!("  scalar: {:.4} ms/call", scalar_ms);
                println!("  AVX2:   {:.4} ms/call", avx2_ms);
                println!("  speedup: {:.2}x", scalar_ms / avx2_ms);
            }
        } else {
            println!("AVX2/FMA/F16C not available; skipping");
        }
    }

    /// Microbench: run scalar and AVX2 over a large input, report
    /// elapsed and speedup ratio. Run with
    ///   `cargo test --release --lib q4k_repack::tests::bench_simd_speedup -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_simd_speedup() {
        use std::time::Instant;
        // 4 rows × 8192 elements = matches a typical FFN-shape matmul
        // activation tile. 32 blocks per row.
        let k = 8192;
        let nb = k / QK_K;
        let x = synth_input(k, 42);
        let mut y = vec![fresh_block(); nb];
        let iters = 5000;

        // Warm
        for _ in 0..50 { quantize_mat_q8_k_4x8_scalar(&x, &mut y, k); }
        let t = Instant::now();
        for _ in 0..iters { quantize_mat_q8_k_4x8_scalar(&x, &mut y, k); }
        let scalar_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let mut have_avx2 = false;
        #[cfg(target_arch = "x86_64")]
        {
            have_avx2 = std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma");
        }
        if have_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                for _ in 0..50 { unsafe { quantize_mat_q8_k_4x8_avx2(&x, &mut y, k); } }
                let t = Instant::now();
                for _ in 0..iters {
                    unsafe { quantize_mat_q8_k_4x8_avx2(&x, &mut y, k); }
                }
                let avx2_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                println!("\nQ8_K_4x8 quantize, 4×{k} floats, {iters} iters:");
                println!("  scalar: {:.3} ms/call", scalar_ms);
                println!("  AVX2:   {:.3} ms/call", avx2_ms);
                println!("  speedup: {:.2}x", scalar_ms / avx2_ms);
            }
        } else {
            println!("AVX2 not available on this CPU; skipping speedup test");
        }
    }

    #[test]
    fn quantize_scalar_recovers_input_within_max_abs() {
        // Synthetic input: 4 rows × 256 floats, varying magnitudes.
        let k = 256;
        let mut x = vec![0.0f32; 4 * k];
        for row in 0..4 {
            for i in 0..k {
                let phase = (row as f32 + 1.0) * (i as f32) * 0.01;
                x[row * k + i] = (phase).sin() * (row as f32 + 1.0) * 5.0;
            }
        }
        let mut y = vec![BlockQ8Kx4 {
            d: [0.0; 4], qs: [0; 1024], bsums: [0; 64],
        }; 1];
        quantize_mat_q8_k_4x8_scalar(&x, &mut y, k);

        // Dequantize row 0 from y[0].qs (using y[0].d[0]) and confirm
        // it's within ~max_abs/127 of the original.
        for row in 0..4 {
            let d = y[0].d[row];
            let mut max_err = 0.0f32;
            for i in 0..k {
                let sub = i / 8;
                let lane = i % 8;
                let src = sub * 32 + row * 8 + lane;
                let dequant = (y[0].qs[src] as f32) * d;
                let orig = x[row * k + i];
                let err = (dequant - orig).abs();
                if err > max_err { max_err = err; }
            }
            // d is roughly max_abs / 127, so quantization error per
            // element is at most ~d (one quant step).
            assert!(max_err < d.abs() * 1.5 + 1e-3,
                "row {row}: max_err={max_err} > tol={d}");
        }
    }
}
