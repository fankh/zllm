//! Phase 0 — Does Candle's QMatMul amortize weight reads across batch?
//!
//! Survey of vLLM/SGLang/llama.cpp says: continuous batching's throughput
//! win comes from treating N concurrent decode tokens as the M dimension
//! of a GEMM. If `QMatMul.forward(xs)` with `xs.shape = (M, hidden)` runs
//! ~M× as fast as M independent (1, hidden) calls, Candle already does
//! it right and we just need a scheduler. If it runs ~1× as fast (no
//! amortization), we need a custom batched GEMM for Q4_K_M.
//!
//! Shapes tested are real Llama-3.2-1B: hidden=2048, FFN intermediate=8192,
//! vocab=128256.
//!
//! Run: `cargo test --release --test matmul_scaling -- --nocapture`

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use std::time::Instant;

fn deterministic_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let s = ((i as u64).wrapping_mul(seed).wrapping_add(0x9E37_79B9_7F4A_7C15) & 0xFFFF) as f32;
        out[i] = (s / 32768.0 - 1.0) * 0.5;
    }
    out
}

fn bench_shape(name: &str, n_rows: usize, n_cols: usize) {
    let dev = Device::Cpu;

    // Build f32 weights, quantize to Q4_K, wrap in QMatMul.
    let w = deterministic_f32(n_rows * n_cols, 1234567);
    let w_t = Tensor::from_vec(w, (n_rows, n_cols), &dev).unwrap();
    let qt = QTensor::quantize(&w_t, GgmlDType::Q4K).unwrap();
    let qmm = QMatMul::from_qtensor(qt).unwrap();

    println!("\n=== shape: {} weight=({}, {}) ===", name, n_rows, n_cols);
    println!(
        " {:>5} | {:>11} | {:>11} | {:>11} | {:>6}",
        "M", "ms total", "µs/M", "tok/s @ M", "speedup"
    );

    let warmup = 3;
    let measure = 20;
    let mut baseline_us_per_m: Option<f64> = None;

    for &m in &[1_usize, 2, 4, 8, 16, 32] {
        let x_data = deterministic_f32(m * n_cols, 0xBADCAFE + m as u64);
        let xs = Tensor::from_vec(x_data, (m, n_cols), &dev).unwrap();

        // Warmup.
        for _ in 0..warmup {
            let _ = qmm.forward(&xs).unwrap();
        }

        // Measure.
        let t0 = Instant::now();
        for _ in 0..measure {
            let out = qmm.forward(&xs).unwrap();
            // Force consumption so it can't be optimized away.
            let _ = out.flatten_all().unwrap().mean(0).unwrap();
        }
        let elapsed = t0.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0 / measure as f64;
        let us_per_m = (total_ms * 1000.0) / m as f64;
        let tok_per_s = (m as f64 * 1000.0) / total_ms;

        if baseline_us_per_m.is_none() {
            baseline_us_per_m = Some(us_per_m);
        }
        let speedup = baseline_us_per_m.unwrap() / us_per_m;

        println!(
            " {:>5} | {:>11.3} | {:>11.2} | {:>11.1} | {:>5.2}x",
            m, total_ms, us_per_m, tok_per_s, speedup
        );
    }
}

#[test]
fn candle_matmul_amortization() {
    println!("Phase 0: Does Candle Q4_K QMatMul amortize across batch?");
    println!("If µs/M decreases as M grows, weights are being reused.");
    println!("If µs/M stays flat, Candle does M independent GEMVs (no amortization).");
    println!();

    // Real Llama-3.2-1B Q projection: 2048 -> 2048
    bench_shape("attention Q/O proj", 2048, 2048);
    // Llama-3.2-1B K/V projection: 2048 -> 512 (GQA)
    bench_shape("attention K/V proj", 512, 2048);
    // Llama-3.2-1B FFN w1/w3: 2048 -> 8192
    bench_shape("FFN w1/w3", 8192, 2048);
    // Llama-3.2-1B FFN w2: 8192 -> 2048
    bench_shape("FFN w2", 2048, 8192);
    // Llama-3.2-1B LM head: 2048 -> 128256
    bench_shape("LM head", 128256, 2048);

    println!("\nInterpretation:");
    println!(" - speedup ~M  → matmul fully amortizes (ideal: continuous batching is plumbing only)");
    println!(" - speedup ~√M → partial amortization (cache reuse, no GEMM)");
    println!(" - speedup ~1  → no amortization (must write custom batched GEMM)");
}
