//! Phase 2 validation: our `matmul_q4k_q8k_par` should
//! (a) match Candle's QMatMul.forward bit-tolerance on real Q4_K data,
//! (b) amortize weights across batch (near-linear M scaling),
//! (c) beat Candle's per-row matmul at M >= 4 on FFN-shaped weights.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use std::time::Instant;

use zllm::backend::candle::q4k_avx512::matmul_q4k_q8k_par;
use zllm::backend::candle::q4k_repack::{quantize_q8_k, BlockQ4K, BlockQ8K, QK_K};

fn deterministic_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let s = ((i as u64).wrapping_mul(seed).wrapping_add(0x9E37_79B9_7F4A_7C15)
            & 0xFFFF) as f32;
        out[i] = (s / 32768.0 - 1.0) * 0.5;
    }
    out
}

/// Extract raw &[BlockQ4K] from a Candle Q4_K QTensor. Same trick as
/// in QMatMul::from_qtensor (try_extract_raw_q4k).
fn qtensor_to_blocks<'a>(qt: &'a QTensor) -> Vec<BlockQ4K> {
    let bytes = qt.data().unwrap();
    let block_sz = std::mem::size_of::<BlockQ4K>();
    let n_blocks = bytes.len() / block_sz;
    let raw: &[BlockQ4K] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4K, n_blocks)
    };
    raw.to_vec()
}

/// Quantize a batched activation (M rows × hidden floats) into Q8_K
/// row-major blocks (M × nb_per_row).
fn quantize_batch(input: &[f32], m: usize, hidden: usize) -> Vec<BlockQ8K> {
    let nb_per_row = hidden / QK_K;
    let mut out = vec![BlockQ8K {
        d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16],
    }; m * nb_per_row];
    for r in 0..m {
        let row_in = &input[r * hidden .. (r + 1) * hidden];
        let row_out = &mut out[r * nb_per_row .. (r + 1) * nb_per_row];
        quantize_q8_k(row_in, row_out);
    }
    out
}

#[test]
fn matmul_q4k_q8k_matches_candle() {
    let dev = Device::Cpu;
    let n_out = 64;
    let hidden = 8 * QK_K; // 2048
    let nb_per_row = hidden / QK_K;
    let m = 4;

    // Build weights + quantize via Candle (twice — one for bytes, one for QMatMul).
    let w = deterministic_f32(n_out * hidden, 0xDEADBEEF);
    let w_t = Tensor::from_vec(w.clone(), (n_out, hidden), &dev).unwrap();
    let qt_bytes = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
    let qt_for_mm = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
    let qmm = QMatMul::from_qtensor(qt_for_mm).unwrap();

    let blocks = qtensor_to_blocks(&qt_bytes);

    // Batched activation.
    let x = deterministic_f32(m * hidden, 0xCAFEF00D);
    let x_t = Tensor::from_vec(x.clone(), (m, hidden), &dev).unwrap();

    // Candle reference.
    let ref_out = qmm.forward(&x_t).unwrap();
    let ref_vec: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(ref_vec.len(), m * n_out);

    // Our path.
    let act = quantize_batch(&x, m, hidden);
    let mut out = vec![0.0_f32; m * n_out];
    matmul_q4k_q8k_par(&blocks, &act, &mut out, n_out, nb_per_row, m);

    let mut max_abs = 0.0_f32;
    for (i, (a, b)) in out.iter().zip(ref_vec.iter()).enumerate() {
        let e = (a - b).abs();
        if e > max_abs {
            max_abs = e;
        }
        let _ = i;
    }
    println!("Phase 2 correctness: max_abs_err = {:.6}", max_abs);
    assert!(max_abs < 0.01, "Phase 2 matmul abs err too high: {}", max_abs);
}

fn bench_shape(name: &str, n_out: usize, hidden: usize) {
    let dev = Device::Cpu;
    let nb_per_row = hidden / QK_K;

    // Build weights once.
    let w = deterministic_f32(n_out * hidden, 0xDEADBEEF);
    let w_t = Tensor::from_vec(w, (n_out, hidden), &dev).unwrap();
    let qt_bytes = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
    let qt_for_mm = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
    let qmm = QMatMul::from_qtensor(qt_for_mm).unwrap();
    let blocks = qtensor_to_blocks(&qt_bytes);

    println!("\n=== shape: {} weight=({}, {}) ===", name, n_out, hidden);
    println!(
        " {:>4} | {:>11} | {:>11} | {:>11} | {:>11} | {:>10}",
        "M", "Candle ms", "Ours ms", "Candle tok/s", "Ours tok/s", "Speedup"
    );

    let warmup = 3;
    let measure = 10;

    for &m in &[1_usize, 2, 4, 8, 16] {
        let x = deterministic_f32(m * hidden, 0xCAFEF00D + m as u64);
        let x_t = Tensor::from_vec(x.clone(), (m, hidden), &dev).unwrap();
        let act = quantize_batch(&x, m, hidden);

        // Candle
        for _ in 0..warmup {
            let _ = qmm.forward(&x_t).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..measure {
            let _ = qmm.forward(&x_t).unwrap();
        }
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0 / measure as f64;

        // Ours
        let mut out = vec![0.0_f32; m * n_out];
        for _ in 0..warmup {
            matmul_q4k_q8k_par(&blocks, &act, &mut out, n_out, nb_per_row, m);
        }
        let t0 = Instant::now();
        for _ in 0..measure {
            matmul_q4k_q8k_par(&blocks, &act, &mut out, n_out, nb_per_row, m);
        }
        let ours_ms = t0.elapsed().as_secs_f64() * 1000.0 / measure as f64;

        let candle_tps = (m as f64 * 1000.0) / candle_ms;
        let ours_tps = (m as f64 * 1000.0) / ours_ms;
        let speedup = candle_ms / ours_ms;

        println!(
            " {:>4} | {:>11.3} | {:>11.3} | {:>11.1} | {:>11.1} | {:>9.2}x",
            m, candle_ms, ours_ms, candle_tps, ours_tps, speedup
        );
    }
}

#[test]
fn phase_2_amortization_bench() {
    println!("Phase 2 bench: ours (col-outer, batch-inner, amortized) vs Candle (batch-outer, no amortize)");
    bench_shape("attention Q/O (2048x2048)", 2048, 2048);
    bench_shape("attention K/V (512x2048)", 512, 2048);
    bench_shape("FFN w1/w3 (8192x2048)", 8192, 2048);
    bench_shape("FFN w2 (2048x8192)", 2048, 8192);
    bench_shape("LM head (128256x2048)", 128256, 2048);

    println!("\nDecision rule from survey:");
    println!(" - Per-token throughput should improve near-linearly with M for FFN.");
    println!(" - If speedup at M=8 < 2x for FFN w1/w3, Phase 2 needs more tuning (prefetch/tiling).");
}
