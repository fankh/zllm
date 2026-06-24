//! Heterogeneous PREFILL sizing: prefill is compute-bound (constant tok/s with P,
//! ~4 GB/s of the 256 bus → 98% idle), so unlike DECODE (bandwidth-bound, CPU+iGPU
//! zero-sum) the CPU's AVX-512 cores could add prefill GEMM throughput in parallel
//! with the iGPU's coopmat GEMMs WITHOUT bus contention. This measures: CPU prefill
//! tok/s, iGPU prefill tok/s, and whether running both concurrently is ADDITIVE
//! (iGPU not slowed) or zero-sum (iGPU slowed = they contend).
//! `cargo test --release --features vulkan --test hetero_prefill -- --ignored --nocapture`
#![cfg(feature = "vulkan")]

use candle_core::Device;
use std::time::Instant;
use zllm::backend::candle::backend::CandleCpuBackend;
use zllm::backend::vulkan::{VkContext, VkModel};

const MODEL: &str = "C:/models/llama-3.2-1b/Llama-3.2-1B-Q4pure.gguf";

#[test]
#[ignore]
fn hetero_prefill_additivity() {
    if !std::path::Path::new(MODEL).exists() { eprintln!("model not found; skipping"); return; }
    let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
    let vmodel = VkModel::load(MODEL, ctx).expect("vk load");
    let p = 384usize;
    let prompt: Vec<u32> = (0..p as u32).map(|i| (100 + i * 13) % 28000).collect();

    // iGPU prefill alone (warm, then min-of-3).
    let _ = vmodel.prefill_forward(&prompt);
    let gpu_ms = (0..3).map(|_| { let t = Instant::now(); let _ = vmodel.prefill_forward(&prompt); t.elapsed().as_secs_f64() * 1e3 })
        .fold(f64::MAX, f64::min);

    // CPU prefill alone (candle batched forward over the same prompt).
    let mut cpu = CandleCpuBackend::with_device(Device::Cpu);
    cpu.load_model_requant(std::path::Path::new(MODEL), &|_| None).expect("cpu load");
    cpu.reset_position(); let _ = cpu.forward_all_logits(&prompt).unwrap(); // warm
    cpu.reset_position();
    let t = Instant::now(); let _ = cpu.forward_all_logits(&prompt).unwrap(); let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    let gpu_tps = p as f64 / gpu_ms * 1e3;
    let cpu_tps = p as f64 / cpu_ms * 1e3;
    eprintln!("iGPU prefill {p} tok: {gpu_ms:6.1} ms = {gpu_tps:5.0} tok/s");
    eprintln!("CPU  prefill {p} tok: {cpu_ms:6.1} ms = {cpu_tps:5.0} tok/s");

    // Concurrent: CPU prefill on a worker thread, iGPU prefill on this thread.
    let prompt2 = prompt.clone();
    let h = std::thread::spawn(move || {
        cpu.reset_position();
        let t = Instant::now(); let _ = cpu.forward_all_logits(&prompt2).unwrap(); t.elapsed().as_secs_f64() * 1e3
    });
    let t = Instant::now(); let _ = vmodel.prefill_forward(&prompt); let gpu_conc = t.elapsed().as_secs_f64() * 1e3;
    let cpu_conc = h.join().unwrap();

    eprintln!("\nCONCURRENT: iGPU {gpu_conc:.1} ms (alone {gpu_ms:.1}, slowdown {:.2}x), CPU {cpu_conc:.1} ms (alone {cpu_ms:.1}, slowdown {:.2}x)",
        gpu_conc / gpu_ms, cpu_conc / cpu_ms);
    let agg = (2 * p) as f64 / gpu_conc.max(cpu_conc) * 1e3;
    eprintln!("  aggregate {agg:.0} tok/s vs iGPU-alone {gpu_tps:.0} = {:.2}x", agg / gpu_tps);
    eprintln!("  → iGPU slowdown ~1.0 + aggregate > iGPU-alone ⇒ prefill is ADDITIVE (heterogeneous win = +{:.0}%, the CPU's {cpu_tps:.0} tok/s on top)",
        cpu_tps / gpu_tps * 100.0);
}
