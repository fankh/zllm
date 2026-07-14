//! GPU-path sampling helpers (temperature/top-k/top-p over the
//! kernel's top-K candidate pool) + the per-request seeding, extracted
//! from `mod.rs` (V1_PLAN monolith split). Pure math, no GPU state.
//! `SamplingParams` is re-exported by the parent so
//! `backend::gpu::SamplingParams` is unchanged.

/// Per-token sampling seed: decorrelate by generation index (the kernel further
/// hashes (seed, token_id), so a cheap mix here is enough).
pub(super) fn step_seed(base: u32, gen_idx: u32) -> u32 { base.wrapping_add(gen_idx.wrapping_mul(0x9E3779B1)) }

/// Per-request sampling knobs. `temp ≤ 0` = greedy; `top_k = 0` = no top-k cap;
/// `top_p ≥ 1` (or 0) = no nucleus cap. top-k/top-p sample within the GPU's
/// `TOPK_K` candidate pool, so effective top_k is capped at `TOPK_K`.
#[derive(Clone, Copy)]
pub struct SamplingParams { pub temp: f32, pub top_k: u32, pub top_p: f32 }
impl SamplingParams {
    pub fn greedy() -> Self { Self { temp: 0.0, top_k: 0, top_p: 1.0 } }
    pub fn temperature(temp: f32) -> Self { Self { temp, top_k: 0, top_p: 1.0 } }
    /// Whether this needs the top-K candidate path (vs greedy / full-dist temp).
    pub(super) fn needs_topk(&self) -> bool { self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0) }
}

/// Deterministic uniform in (0,1) from a seed (splitmix64).
fn rng01(seed: u32) -> f32 {
    let mut z = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 40) as f32 + 0.5) / 16777216.0
}

/// Sample a token from a stream's descending top-K `(vals, idxs)` under
/// `params`, seeded by `seed`. Applies temperature, top-k, then top-p (nucleus),
/// renormalizes, and draws by inverse-CDF. Greedy (`temp ≤ 0`) returns the top-1.
pub(super) fn sample_topk(vals: &[f32], idxs: &[u32], params: SamplingParams, seed: u32) -> u32 {
    // Drop kernel sentinels (fewer than K distinct logits — never for real vocab).
    let n = vals.iter().take_while(|&&v| v > -3.0e37).count().min(idxs.len());
    if n == 0 { return idxs[0]; }
    if params.temp <= 0.0 { return idxs[0]; } // greedy
    let kk = if params.top_k == 0 { n } else { (params.top_k as usize).min(n) };
    let maxv = vals[0]; // descending → max is first
    let mut probs: Vec<f32> = (0..kk).map(|i| ((vals[i] - maxv) / params.temp).exp()).collect();
    let z: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= z; }
    // Nucleus: smallest prefix whose cumulative prob ≥ top_p.
    let mut cut = kk;
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() { cum += p; if cum >= params.top_p { cut = i + 1; break; } }
    }
    let zc: f32 = probs[..cut].iter().sum();
    let mut acc = 0.0;
    let target = rng01(seed) * zc;
    for i in 0..cut { acc += probs[i]; if acc >= target { return idxs[i]; } }
    idxs[cut - 1]
}
