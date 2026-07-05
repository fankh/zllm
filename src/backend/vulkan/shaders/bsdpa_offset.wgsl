// Batched OFFSET causal SDPA for CHUNKED prefill: m query rows at positions
// base_pos+row, each attending the resident pos-major KV cache [0..=base_pos+row].
// Replaces bsdpa_decode (one THREAD per query·head, serial cache walk — measured
// 387 tok/s chunked prefill) with one WORKGROUP (64 threads) per (query, head),
// using sdpa_decode.comp v2's barrier-lean structure: phase 1 strided full dots
// in registers (no per-position barrier), one max-reduce, one exp+sum, phase 4
// thread-per-dim V walk. LDS: sc[MAX_SEQ] = 16 KB (fits RDNA3's 32 KB, ~2 wg/CU
// — fine at m*n_head-sized grids). hd <= 64. SPV: gen_headmajor_spv.
struct P { n_head: u32, n_kv: u32, hd: u32, m_rows: u32, base_pos: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // [m, n_head*hd] roped
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;  // pos-major cache [seq, n_kv*hd]
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;  // [m, n_head*hd]
@group(0) @binding(4) var<uniform>             p:    P;
const MAX_SEQ: u32 = 4096u;
var<workgroup> sc:  array<f32, MAX_SEQ>; // scores, then probs
var<workgroup> qsh: array<f32, 64>;      // this (row, head)'s query
var<workgroup> red: array<f32, 64>;      // reduction scratch
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x / p.n_head;
    let h = wid.x % p.n_head;
    let t = lid.x;
    let hd = p.hd;
    let kvh = h / (p.n_head / p.n_kv);
    let seq_len = p.base_pos + row + 1u;   // causal: attend 0..=base_pos+row
    let scale = 1.0 / sqrt(f32(hd));
    let qb = row * (p.n_head * hd) + h * hd;
    if (t < hd) { qsh[t] = q[qb + t]; }
    workgroupBarrier();
    // Phase 1: scores — each thread owns positions {t, t+64, ...}, full dot in registers.
    var pos = t;
    while (pos < seq_len) {
        let kb = (pos * p.n_kv + kvh) * hd;
        var s = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) { s = s + qsh[d] * kc[kb + d]; }
        sc[pos] = s * scale;
        pos = pos + 64u;
    }
    workgroupBarrier();
    // Phase 2: max over scores.
    var mx = -1e30;
    pos = t;
    while (pos < seq_len) { mx = max(mx, sc[pos]); pos = pos + 64u; }
    red[t] = mx; workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) { if (t < st) { red[t] = max(red[t], red[t + st]); } workgroupBarrier(); }
    let m = red[0]; workgroupBarrier();
    // Phase 3: exp + running sum (overwrite scores with probs).
    var lsum = 0.0;
    pos = t;
    while (pos < seq_len) { let e = exp(sc[pos] - m); sc[pos] = e; lsum = lsum + e; pos = pos + 64u; }
    red[t] = lsum; workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) { if (t < st) { red[t] = red[t] + red[t + st]; } workgroupBarrier(); }
    let l = red[0]; workgroupBarrier();
    // Phase 4: output — thread t owns head-dim t, walks cached V.
    if (t < hd) {
        var av = 0.0;
        for (var i = 0u; i < seq_len; i = i + 1u) {
            av = av + sc[i] * vc[(i * p.n_kv + kvh) * hd + t];
        }
        outp[qb + t] = av / l;
    }
}
