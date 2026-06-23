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
use std::path::Path;
use zllm::backend::candle::backend::CandleCpuBackend;
use zllm::backend::candle::tokenizer::LlamaTokenizer;

const MODEL: &str = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const TOKENIZER: &str = "C:/models/llama-3.2-1b/tokenizer.json";

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

/// Per-class weight element counts (for bytes/token accounting).
fn class_elems(classes: &[&'static str]) -> std::collections::HashMap<&'static str, usize> {
    let mut f = std::fs::File::open(MODEL).unwrap();
    let ct = gguf_file::Content::read(&mut f).unwrap();
    let mut e = std::collections::HashMap::new();
    for (name, info) in &ct.tensor_infos {
        for c in classes {
            if name.contains(c) { *e.entry(*c).or_default() += info.shape.elem_count(); }
        }
    }
    e
}

/// Negative log-likelihood of `target` under `logits` (nats).
fn nll(logits: &[f32], target: u32) -> f64 {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let lse = logits.iter().map(|&x| ((x - m) as f64).exp()).sum::<f64>().ln() + m as f64;
    lse - logits[target as usize] as f64
}

/// Load `path` with the requant `rules`, then over `corpus` (teacher-forced)
/// return (mean KL vs `base_logits`, perplexity) on the predicted positions.
fn eval_config(
    be: &mut CandleCpuBackend, path: &Path, corpus: &[Vec<u32>],
    base_logits: &[Vec<Vec<f32>>], rules: &[(&'static str, GgmlDType)],
) -> (f64, f64) {
    be.load_model_requant(path, &|n: &str| rules.iter().find(|(c, _)| n.contains(c)).map(|(_, d)| *d)).unwrap();
    let (mut kl_sum, mut nll_sum, mut n) = (0.0, 0.0, 0usize);
    for (pi, toks) in corpus.iter().enumerate() {
        be.reset_position();
        let lg = be.forward_all_logits(toks).unwrap();
        for i in 0..toks.len() - 1 {
            kl_sum += kl(&base_logits[pi][i], &lg[i]);
            nll_sum += nll(&lg[i], toks[i + 1]);
            n += 1;
        }
    }
    (kl_sum / n as f64, (nll_sum / n as f64).exp())
}

/// GATE 2: validate the mixed-Q3 RECIPE end to end — diverse HELD-OUT text (not
/// model-generated), real perplexity (not just self-KL), the COMBINED recipe
/// measured directly (KL isn't additive), and a greedy coherence check.
/// `cargo test --release --test quant_sensitivity quant_recipe_gate -- --ignored --nocapture`
#[test]
#[ignore]
fn quant_recipe_gate() {
    if !std::path::Path::new(MODEL).exists() || !std::path::Path::new(TOKENIZER).exists() {
        eprintln!("model/tokenizer not found; skipping");
        return;
    }
    let tok = LlamaTokenizer::from_file(TOKENIZER).expect("tokenizer");
    // Held-out, original text — descriptive / technical / reflective.
    let passages = [
        "The harbor at dawn was crowded with fishing boats returning from the night's work. Gulls circled overhead while the tired crews unloaded crates of silver fish onto the wet stone quay, and a cold wind carried the smell of salt and diesel across the gray water.",
        "A hash map stores key-value pairs and supports average constant-time lookup. When the number of entries grows beyond a load factor, the table is resized and every key is rehashed into a larger array of buckets. Collisions are resolved by chaining or by open addressing.",
        "She had always believed that small habits matter more than grand intentions. Every morning she wrote a single honest sentence in a worn notebook, and after three years the pages held a quiet record of an ordinary life slowly becoming something deliberate and her own.",
    ];
    let corpus: Vec<Vec<u32>> = passages.iter().map(|p| tok.encode(p).unwrap()).collect();
    let n_tok: usize = corpus.iter().map(|c| c.len()).sum();
    eprintln!("held-out corpus: {} passages, {n_tok} tokens\n", corpus.len());

    let classes = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "token_embd"];
    let elems = class_elems(&classes);
    let total_q4: f64 = classes.iter().map(|c| elems[c] as f64 * 4.5 / 8.0).sum();
    let mb_saved = |rules: &[(&str, GgmlDType)]| -> f64 {
        rules.iter().map(|(c, dt)| elems.get(c).copied().unwrap_or(0) as f64 * (4.5 - bits(*dt)) / 8.0 / 1e6).sum()
    };

    let mut be = CandleCpuBackend::with_device(Device::Cpu);
    // baseline (all-Q4): per-passage logits + perplexity reference.
    be.load_model_requant(Path::new(MODEL), &|_| None).unwrap();
    let mut base_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    let (mut nll_sum, mut n) = (0.0, 0usize);
    for toks in &corpus {
        be.reset_position();
        let lg = be.forward_all_logits(toks).unwrap();
        for i in 0..toks.len() - 1 { nll_sum += nll(&lg[i], toks[i + 1]); n += 1; }
        base_logits.push(lg);
    }
    let base_ppl = (nll_sum / n as f64).exp();
    eprintln!("baseline all-Q4 perplexity (held-out): {base_ppl:.3}\n");

    // Recipes (KL isn't additive, so each is measured directly).
    let conservative: Vec<(&str, GgmlDType)> = vec![("ffn_gate", GgmlDType::Q3K), ("ffn_up", GgmlDType::Q3K)];
    // moderate: add the cheap-KL attn projections but NOT ffn_down/token_embd (the
    // heaviest per-class KL) — to locate the knee between conservative and balanced.
    let moderate: Vec<(&str, GgmlDType)> = ["ffn_gate", "ffn_up", "attn_q", "attn_output"]
        .iter().map(|c| (*c, GgmlDType::Q3K)).collect();
    let balanced: Vec<(&str, GgmlDType)> = ["ffn_gate", "ffn_up", "attn_q", "attn_output", "ffn_down", "token_embd"]
        .iter().map(|c| (*c, GgmlDType::Q3K)).collect();
    let mut aggressive = balanced.clone();
    aggressive.retain(|(c, _)| *c != "ffn_down");
    aggressive.push(("ffn_down", GgmlDType::Q2K)); // the known cliff — should show in ppl

    eprintln!("{:>32}  {:>8}  {:>9}  {:>8}  {:>10}  {:>10}", "recipe", "mean KL", "perplex", "Δppl%", "MB saved", "~tok/s*");
    eprintln!("{}", "-".repeat(86));
    eprintln!("{:>32}  {:>8}  {:>9.3}  {:>8}  {:>10}  {:>10.0}", "baseline (all-Q4)", "-", base_ppl, "-", "-", 209.7);
    for (label, rules) in [("conservative: gate+up→Q3", &conservative), ("moderate: +attn_q/o→Q3", &moderate), ("balanced: +down+lm→Q3", &balanced), ("aggressive: balanced, down→Q2", &aggressive)] {
        let (kl_m, ppl) = eval_config(&mut be, Path::new(MODEL), &corpus, &base_logits, rules);
        let mb = mb_saved(rules);
        let tps = 209.7 * total_q4 / (total_q4 - mb * 1e6); // bandwidth-proportional estimate
        eprintln!("{label:>32}  {kl_m:>8.4}  {ppl:>9.3}  {:>7.1}%  {:>9.1}M  {tps:>10.0}", (ppl / base_ppl - 1.0) * 100.0, mb);
    }
    eprintln!("\n* projected tok/s ASSUMES a Q3 decode kernel matches Q4 bandwidth efficiency (decode is bw-bound).");

    // Coherence: greedy 40 tokens, baseline vs balanced — eyeball that it stays sane.
    let prompt = tok.encode("The three primary colors are").unwrap();
    let greedy_gen = |be: &mut CandleCpuBackend| -> String {
        be.reset_position();
        let mut next = argmax(&be.forward_logits(&prompt).unwrap());
        let mut out = vec![next];
        for _ in 0..39 { next = argmax(&be.forward_logits(&[next]).unwrap()); out.push(next); }
        tok.decode(&out).unwrap()
    };
    be.load_model_requant(Path::new(MODEL), &|_| None).unwrap();
    let base_txt = greedy_gen(&mut be);
    be.load_model_requant(Path::new(MODEL), &|n: &str| balanced.iter().find(|(c, _)| n.contains(c)).map(|(_, d)| *d)).unwrap();
    let bal_txt = greedy_gen(&mut be);
    eprintln!("\ncoherence (greedy 40 tok from \"The three primary colors are\"):");
    eprintln!("  baseline: {base_txt}");
    eprintln!("  balanced: {bal_txt}");
}
