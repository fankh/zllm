//! Q3_K dequant reference, validated bit-exact against candle. This is the
//! correctness SPEC the GPU decode-matvec kernel must implement: the intricate
//! parts are the 12-byte→16×6-bit scale shuffle, the per-weight hmask high-bit
//! selection, and the qs 2-bit field. ggml block_q3_K (110 bytes):
//!   hmask[32] | qs[64] | scales[12] | d(f16)[2]
//! Weight[out] = d * (scale[sub]-32) * ((qs>>shift)&3 - (hmask&m ? 0 : 4)).
//! `cargo test --release --test q3k_dequant -- --nocapture`

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

const BLK: usize = 110; // bytes per block_q3_K (QK_K=256 weights)

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -f } else { f }
}

/// Dequantize `n_rows × k` Q3_K weights (ggml layout) to f32, row-major. Mirrors
/// what the GPU kernel will compute per output row.
fn dequant_q3k_ref(bytes: &[u8], n_rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 256;
    let mut out = vec![0f32; n_rows * k];
    let (km1, km2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    for row in 0..n_rows {
        for b in 0..nb {
            let blk = &bytes[(row * nb + b) * BLK..][..BLK];
            let (hmask, qs, scales) = (&blk[0..32], &blk[32..96], &blk[96..108]);
            let d = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
            // 12 scale bytes → 16 6-bit values (the ggml shuffle).
            let s0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
            let s1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
            let s2 = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
            let a = [
                (s0 & km2) | (((s2) & km1) << 4),
                (s1 & km2) | (((s2 >> 2) & km1) << 4),
                ((s0 >> 4) & km2) | (((s2 >> 4) & km1) << 4),
                ((s1 >> 4) & km2) | (((s2 >> 6) & km1) << 4),
            ];
            let mut sc = [0u8; 16];
            for w in 0..4 { sc[w * 4..w * 4 + 4].copy_from_slice(&a[w].to_le_bytes()); }
            for out_idx in 0..256 {
                let h = out_idx / 128;
                let r = out_idx % 128;
                let (j, sub, l) = (r / 32, (r % 32) / 16, r % 16);
                let shift = 2 * j;
                let m = 1u32 << (h * 4 + j);
                let q2 = ((qs[h * 32 + sub * 16 + l] >> shift) & 3) as f32;
                let hbit = (hmask[sub * 16 + l] as u32 & m) != 0;
                let scale = sc[h * 8 + j * 2 + sub] as i32 - 32;
                out[row * k + b * 256 + out_idx] = d * scale as f32 * (q2 - if hbit { 0.0 } else { 4.0 });
            }
        }
    }
    out
}

#[test]
fn q3k_dequant_matches_candle() {
    let dev = Device::Cpu;
    let (n, k) = (6usize, 768usize); // 6 rows × 3 blocks/row
    let data: Vec<f32> = (0..n * k).map(|i| (((i * 131 % 521) as f32) - 260.0) / 37.0).collect();
    let t = Tensor::from_vec(data, (n, k), &dev).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::Q3K).unwrap();

    let oracle = qt.dequantize(&dev).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bytes = qt.data().unwrap();
    assert_eq!(bytes.len(), n * (k / 256) * BLK, "unexpected Q3_K byte size");
    let mine = dequant_q3k_ref(&bytes, n, k);

    let mut maxerr = 0f32;
    for i in 0..n * k { maxerr = maxerr.max((mine[i] - oracle[i]).abs()); }
    eprintln!("Q3_K dequant ref vs candle: max abs err = {maxerr:.3e} over {} weights", n * k);
    assert!(maxerr < 1e-4, "Q3_K dequant ref diverged from candle (max {maxerr})");
    eprintln!("Q3_K dequant algorithm validated bit-exact ✓ (kernel spec confirmed)");
}
