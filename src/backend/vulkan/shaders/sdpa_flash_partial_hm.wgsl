// Head-major flash-attention partial (ZLLM_HEADMAJOR_KV). One workgroup per
// (query head, KV block of 32 positions); 64 threads. Reads K/V from the
// head-major cache [kv_head, pos, hd] so each block's 32 positions are
// CONTIGUOUS in memory (8 KB run) instead of strided by n_kv*hd — the whole
// point of the layout. Outputs the SAME (av, m, l) triple per block as
// sdpa_flash_partial2 so the combine shader is unchanged: per (head, block)
// it writes hd un-normalized weighted-V accumulators, then block-max m and
// block sum-exp l. Compute uses a shared-mem reduction (no subgroup ops, so
// naga compiles it). Model dim MAX_SEQ is a const. SPV: gen_headmajor_spv.
struct P { n_head: u32, n_kv: u32, hd: u32, seq_len: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // [n_head*hd]
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;  // head-major [n_kv, MAX_SEQ, hd]
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;  // [(h*nblk+blk)*(hd+2)]
@group(0) @binding(4) var<uniform>             p:    P;
const BLOCK: u32 = 32u;
const MAX_SEQ: u32 = 4096u;
var<workgroup> sc:  array<f32, 32>;   // block scores → probs
var<workgroup> qsh: array<f32, 64>;   // this head's query
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let blk = wid.y;
    let t = lid.x;
    let hd = p.hd;
    let n_blocks = (p.seq_len + BLOCK - 1u) / BLOCK;
    let kvh = h / (p.n_head / p.n_kv);
    let scale = 1.0 / sqrt(f32(hd));
    if (t < hd) { qsh[t] = q[h * hd + t]; }
    workgroupBarrier();
    let start = blk * BLOCK;
    let end = min(start + BLOCK, p.seq_len);
    let bk = end - start;
    // Phase 1: score for position start+t (thread t<bk), contiguous head-major K read.
    if (t < bk) {
        let kb = kvh * MAX_SEQ * hd + (start + t) * hd;
        var s = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) { s = s + qsh[d] * kc[kb + d]; }
        sc[t] = s * scale;
    }
    workgroupBarrier();
    // Phase 2: block max (all threads read the bk valid scores).
    var m = -1e30;
    for (var i = 0u; i < bk; i = i + 1u) { m = max(m, sc[i]); }
    workgroupBarrier();                          // all reads done before the exp overwrite
    // Phase 2b: exp in place.
    if (t < bk) { sc[t] = exp(sc[t] - m); }
    workgroupBarrier();
    // Phase 3: un-normalized weighted V (thread t owns head-dim t), contiguous head-major V read.
    let base = (h * n_blocks + blk) * (hd + 2u);
    if (t < hd) {
        var av = 0.0;
        for (var i = 0u; i < bk; i = i + 1u) {
            av = av + sc[i] * vc[kvh * MAX_SEQ * hd + (start + i) * hd + t];
        }
        part[base + t] = av;
    }
    if (t == 0u) {
        var l = 0.0;
        for (var i = 0u; i < bk; i = i + 1u) { l = l + sc[i]; }
        part[base + hd] = m;
        part[base + hd + 1u] = l;
    }
}
