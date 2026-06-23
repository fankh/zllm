//! Sub-Q4 quantization sensitivity sweep (the disciplined first step before any
//! Q3 kernel work). For each weight-tensor CLASS we round-trip it to Q3_K / Q2_K
//! (dequantize → re-quantize, run on candle's path — no new kernel) and measure
//! how much the output distribution shifts (mean KL divergence vs the all-Q4
//! baseline over a real eval corpus) against the bytes/token it saves in decode.
//!
//! Output: a Pareto table — sorted by KL-per-MB-saved — that says which tensors
//! can drop below Q4 cheaply. Decode is memory-bound (~663 MB/token streamed), so
//! MB saved ≈ proportional decode speedup; KL is the quality cost.
//!
//! `cargo test --release --test quant_sensitivity -- --ignored --nocapture`

use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::Device;
use zllm::backend::candle::backend::CandleCpuBackend;

const MODEL: &str = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

/// bits/weight for each k-quant (block_size 256): Q4_K 144B, Q3_K 110B, Q2_K 84B.
fn bits(dt: GgmlDType) -> f64 {
    match dt {
        GgmlDType::Q4K => 4.5,
        GgmlDType::Q3K => 3.4375,
        GgmlDType::Q2K => 2.625,
        _ => 4.5,
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
}

/// KL(P || Q) for one position; P from `base` logits, Q from `cfg` logits. Stable
/// via log-softmax. Measures how far the degraded model's distribution moved.
fn kl(base: &[f32], cfg: &[f32]) -> f64 {
    let lse = |l: &[f32]| -> (f64, f32) {
        let m = l.iter().cloned().fold(f32::MIN, f32::max);
        let s: f64 = l.iter().map(|&x| ((x - m) as f64).exp()).sum();
        (s.ln() + m as f64, m)
    };
    let (lse_b, _) = lse(base);
    let (lse_c, _) = lse(cfg);
    let mut acc = 0.0;
    for i in 0..base.len() {
        let logp = base[i] as f64 - lse_b;
        if logp < -30.0 { continue; } // p ~ 0 contributes ~nothing
        let logq = cfg[i] as f64 - lse_c;
        acc += logp.exp() * (logp - logq);
    }
    acc
}

#[test]
#[ignore]
fn quant_sensitivity_sweep() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("model not found at {MODEL}; skipping");
        return;
    }
    // Per-class element counts (for bytes/token accounting) from the GGUF infos.
    let classes = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "token_embd"];
    let mut elems = std::collections::HashMap::<&str, usize>::new();
    {
        let mut f = std::fs::File::open(MODEL).unwrap();
        let ct = gguf_file::Content::read(&mut f).unwrap();
        for (name, info) in &ct.tensor_infos {
            for c in &classes {
                if name.contains(c) {
                    *elems.entry(c).or_default() += info.shape.elem_count();
                }
            }
        }
    }
    let total_q4_bytes: f64 = classes.iter().map(|c| elems[c] as f64 * 4.5 / 8.0).sum();
    eprintln!("decode weight stream @ all-Q4: {:.0} MB/token (sum of swept classes)\n", total_q4_bytes / 1e6);

    let mut be = CandleCpuBackend::with_device(Device::Cpu);

    // --- eval corpus: the baseline model's own greedy continuation (in-dist) ---
    be.load_model_requant(std::path::Path::new(MODEL), &|_| None).expect("baseline load");
    let seed: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374]; // "The capital of France is"
    let n_gen = 96usize;
    let mut eval = seed.clone();
    be.reset_position();
    let mut next = argmax(&be.forward_logits(&seed).unwrap());
    for _ in 0..n_gen {
        eval.push(next);
        next = argmax(&be.forward_logits(&[next]).unwrap());
    }
    eprintln!("eval corpus: {} tokens (seed + {n_gen} greedy)\n", eval.len());

    // --- baseline per-position logits (teacher-forced over the eval corpus) ---
    be.reset_position();
    let base_logits = be.forward_all_logits(&eval).unwrap();

    // --- sweep: each class × {Q3, Q2} ---
    let levels = [("Q3", GgmlDType::Q3K), ("Q2", GgmlDType::Q2K)];
    let mut rows: Vec<(String, f64, f64)> = Vec::new(); // (label, mean_kl, mb_saved)
    for (lname, dt) in levels {
        for c in &classes {
            let cc = *c;
            be.load_model_requant(std::path::Path::new(MODEL), &move |n: &str| n.contains(cc).then_some(dt)).unwrap();
            be.reset_position();
            let cfg_logits = be.forward_all_logits(&eval).unwrap();
            // average KL over the GENERATED positions (skip the seed prefix).
            let start = seed.len();
            let mut sum = 0.0;
            for p in start..eval.len() {
                sum += kl(&base_logits[p], &cfg_logits[p]);
            }
            let mean_kl = sum / (eval.len() - start) as f64;
            let mb_saved = elems[cc] as f64 * (4.5 - bits(dt)) / 8.0 / 1e6;
            rows.push((format!("{cc:>12} → {lname}"), mean_kl, mb_saved));
        }
    }

    // --- report: Pareto-sorted by KL per MB saved (cheapest quality cost first) ---
    rows.sort_by(|a, b| (a.1 / a.2.max(1e-9)).partial_cmp(&(b.1 / b.2.max(1e-9))).unwrap());
    eprintln!("{:>20}  {:>10}  {:>10}  {:>12}  {:>7}", "class → dtype", "mean KL", "MB saved", "KL per MB", "% stream");
    eprintln!("{}", "-".repeat(70));
    for (label, mkl, mb) in &rows {
        eprintln!("{label:>20}  {mkl:>10.5}  {mb:>9.1}M  {:>12.5}  {:>6.1}%", mkl / mb.max(1e-9), mb / (total_q4_bytes / 1e6) * 100.0);
    }
    eprintln!("\nLower KL = less quality loss; higher MB saved = more decode speedup.");
    eprintln!("Pick a bytes/token target and read off the low-KL classes to drop.");
}
