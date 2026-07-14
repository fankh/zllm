//! WGSL compute-shader source for the wgpu backend, extracted from
//! `mod.rs` (V1_PLAN M4 monolith split). Pure `const &str` kernel
//! source — no logic. Glob-imported by the parent as `use shaders::*`
//! so every `*_WGSL` reference in `mod.rs` is unchanged.

/// WGSL Q4_K mat-vec: `out[row] = sum_k dequant(W)[row,k] * x[k]`.
///
/// One invocation per output row. Reads raw Q4_K block bytes (144 B/block,
/// `repr(C)` identical to ggml `block_q4_K`) as `array<u32>` and unpacks
/// the f16 super-scales (`unpack2x16float`), the 6-bit sub-scales/mins,
/// and the 4-bit quants entirely in-shader — the same math as the CPU
/// `dequantize_q4k_block`. f32 activation, f32 accumulation.
// SKINNY Q4_K GEMM (decode batching, M <= 8). The tiled GEMM above parallelizes
// its dot phase over M — at M=8 only 8 of 64 threads work (12.5% utilization);
// that thread starvation, not LDS bandwidth, was the measured "skinny wall"
// (~104 tok/s aggregate vs llama's 710 at M=8). Here threads parallelize over K
// like the matvec: one workgroup per output row; thread t owns (sub-block t/8,
// qs word t%8) of each 256-col block, dequantizes IN REGISTERS (no weight LDS,
// weights stream from global exactly once), multiplies its 4 nibbles into M
// per-thread accumulators against an LDS-staged x tile, and a final LDS tree
// folds the 64 partial sums per m. Weight traffic = matvec-optimal; x is tiny
// (M*K) and L2-resident. Same uniform/bind layout as the tiled GEMM.
// WGSL port of the vulkan backend's proven skinny_gemm_q4k.comp design:
//   * one THREAD = one output column (row of W), fully reducing its K — no
//     cross-thread reduction, acc[M] is the only register array;
//   * A[M, 256-chunk] staged in LDS once per chunk, REUSED by all 64 columns
//     (A streams N/64 times, not N times — kills the x-reread);
//   * scalar-fold dequant (d*sc*q - dmin*mn) — no multi-array register blocking
//     (that miscompiled under glslang; kept scalar here too).
pub(super) const Q4K_SKINNY_WGSL: &str = r#"
struct GP { n_rows: u32, nb: u32, n_cols: u32, m_rows: u32, gx: u32, acc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read>       wq:   array<u32>;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform>             p:    GP;
const BLOCK_U32: u32 = 36u;
const MMAX: u32 = 8u;
const TILE: u32 = 256u;                // K-chunk = one Q4_K super-block
var<workgroup> xs: array<f32, 2048>;   // A[M, chunk] = 8 KB, shared across 64 columns
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let col = (wid.x + wid.y * p.gx) * 64u + t;   // output column owned by this thread
    let mm = min(p.m_rows, MMAX);
    var acc: array<f32, 8>;
    for (var m: u32 = 0u; m < MMAX; m = m + 1u) { acc[m] = 0.0; }
    for (var b: u32 = 0u; b < p.nb; b = b + 1u) {
        // Stage A[0..mm, b*256 .. +256] into LDS (64 threads, coalesced).
        var idx = t;
        while (idx < mm * TILE) {
            let m = idx / TILE; let kk = idx % TILE;
            xs[idx] = x[m * p.n_cols + b * TILE + kk];
            idx = idx + 64u;
        }
        workgroupBarrier();
        if (col < p.n_rows) {
            let blk = (col * p.nb + b) * BLOCK_U32;
            let dd = unpack2x16float(wq[blk]);
            let d = dd.x; let dmin = dd.y;
            var u0 = wq[blk + 1u]; var u1 = wq[blk + 2u]; var u2 = wq[blk + 3u];
            let u3 = ((u2 >> 4u) & 0x0f0f0f0fu) | (((u1 >> 6u) & 0x03030303u) << 4u);
            let uaux = u1 & 0x3f3f3f3fu;
            u1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6u) & 0x03030303u) << 4u);
            u2 = uaux; u0 = u0 & 0x3f3f3f3fu;
            for (var sub: u32 = 0u; sub < 8u; sub = sub + 1u) {
                var sc: f32; var mn: f32;
                if (sub < 4u) { sc = f32((u0 >> (sub*8u)) & 0xffu); mn = f32((u2 >> (sub*8u)) & 0xffu); }
                else          { sc = f32((u1 >> ((sub-4u)*8u)) & 0xffu); mn = f32((u3 >> ((sub-4u)*8u)) & 0xffu); }
                let coef = d * sc; let coefmn = dmin * mn;
                let qs0 = blk + 4u + (sub / 2u) * 8u;
                let hi = (sub & 1u) == 1u;
                for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                    let word = wq[qs0 + w];
                    for (var bsel: u32 = 0u; bsel < 4u; bsel = bsel + 1u) {
                        let byte = (word >> (bsel * 8u)) & 0xffu;
                        var q: u32; if (hi) { q = byte >> 4u; } else { q = byte & 0x0fu; }
                        let wval = coef * f32(q) - coefmn;
                        let kpos = sub * 32u + w * 4u + bsel;
                        for (var m: u32 = 0u; m < mm; m = m + 1u) {
                            acc[m] = acc[m] + wval * xs[m * TILE + kpos];
                        }
                    }
                }
            }
        }
        workgroupBarrier();
    }
    if (col < p.n_rows) {
        for (var m: u32 = 0u; m < mm; m = m + 1u) {
            let oi = m * p.n_rows + col;
            if (p.acc == 1u) { outp[oi] = outp[oi] + acc[m]; } else { outp[oi] = acc[m]; }
        }
    }
}
"#;

pub(super) const Q4K_MATVEC_WGSL: &str = r#"
struct Params { n_rows: u32, nb_per_row: u32, gx: u32, acc: u32, out_base: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       wq: array<u32>;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  outp: array<f32>;
@group(0) @binding(3) var<uniform>              p: Params;

const BLOCK_U32: u32 = 36u; // 144 bytes / 4
var<workgroup> partial: array<f32, 64>;

// Coalesced: ONE workgroup (64 threads) per output row. Thread `t` owns
// sub-blocks g = t, t+64, … (8 per Q4_K block), so adjacent threads read
// adjacent block data. Reduction over the 64 partials. dequant per sub-block
// = d*scale*sum(q4*x) - dmin*min*sum(x).
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * p.gx;
    if (row >= p.n_rows) { return; }
    let t = lid.x;
    let total_sub = p.nb_per_row * 8u;
    var acc: f32 = 0.0;
    var g = t;
    loop {
        if (g >= total_sub) { break; }
        let b = g / 8u;
        let sub = g % 8u;
        let blk = (row * p.nb_per_row + b) * BLOCK_U32;
        let dd = unpack2x16float(wq[blk]);
        let d = dd.x; let dmin = dd.y;
        var u0 = wq[blk + 1u]; var u1 = wq[blk + 2u]; var u2 = wq[blk + 3u];
        let u3 = ((u2 >> 4u) & 0x0f0f0f0fu) | (((u1 >> 6u) & 0x03030303u) << 4u);
        let uaux = u1 & 0x3f3f3f3fu;
        u1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6u) & 0x03030303u) << 4u);
        u2 = uaux;
        u0 = u0 & 0x3f3f3f3fu;
        var sc: f32; var mn: f32;
        if (sub < 4u) { sc = f32((u0 >> (sub*8u)) & 0xffu); mn = f32((u2 >> (sub*8u)) & 0xffu); }
        else          { sc = f32((u1 >> ((sub-4u)*8u)) & 0xffu); mn = f32((u3 >> ((sub-4u)*8u)) & 0xffu); }
        let pair = sub / 2u;
        let hi = (sub & 1u) == 1u;
        let qs0 = blk + 4u + pair * 8u;   // 8 u32 = 32 bytes
        let xb = b * 256u + sub * 32u;
        var dot: f32 = 0.0; var xsum: f32 = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = wq[qs0 + w];
            for (var bsel: u32 = 0u; bsel < 4u; bsel = bsel + 1u) {
                let byte = (word >> (bsel * 8u)) & 0xffu;
                var q: u32; if (hi) { q = byte >> 4u; } else { q = byte & 0x0fu; }
                let xv = x[xb + w * 4u + bsel];
                dot = dot + f32(q) * xv;
                xsum = xsum + xv;
            }
        }
        acc = acc + d * sc * dot - dmin * mn * xsum;
        g = g + 64u;
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let oi = p.out_base + row;
        if (p.acc == 1u) { outp[oi] = outp[oi] + partial[0]; } else { outp[oi] = partial[0]; }
    }
}
"#;

/// Q6_K mat-vec, struct-of-arrays. Per 256-value block: `ql` (128 bytes,
/// low 4 bits), `qh` (64 bytes, high 2 bits), 16 i8 `scales` (as f32), one
/// f16 `d` (as f32). Dequant mirrors ggml `dequantize_row_q6_K`:
/// `q = ((ql&0xF)|(qh_bits<<4)) - 32`, value = `d * scale * q`. One thread
/// per output row.
pub(super) const Q6K_MATVEC_WGSL: &str = r#"
struct P6 { n_rows: u32, nb: u32, gx: u32, acc: u32, out_base: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       ql:     array<u32>;  // nb_total*32 u32 (128B/blk)
@group(0) @binding(1) var<storage, read>       qh:     array<u32>;  // nb_total*16 u32 (64B/blk)
@group(0) @binding(2) var<storage, read>       scl:    array<f32>;  // nb_total*16
@group(0) @binding(3) var<storage, read>       dd:     array<f32>;  // nb_total
@group(0) @binding(4) var<storage, read>       x:      array<f32>;
@group(0) @binding(5) var<storage, read_write> outp:   array<f32>;
@group(0) @binding(6) var<uniform>             p:      P6;
var<workgroup> partial: array<f32, 64>;

// Coalesced, per element: ONE workgroup per row; thread `t` handles
// elements e = t, t+64, … so adjacent threads read adjacent ql/qh/x.
// Q6_K value = d * scale * (6-bit signed q). No min term.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * p.gx;
    if (row >= p.n_rows) { return; }
    let t = lid.x;
    let ncols = p.nb * 256u;
    var acc: f32 = 0.0;
    var e = t;
    loop {
        if (e >= ncols) { break; }
        let b = e / 256u;
        let pp = e % 256u;
        let half = pp / 128u;
        let pq = pp % 128u;
        let sub = pq / 32u;     // 0..3
        let l = pq % 32u;       // 0..31
        let blk = row * p.nb + b;
        // ql byte: half plane, l (sub 0,2) or l+32 (sub 1,3).
        let ql_l = l + (sub & 1u) * 32u;
        let ql_bi = blk * 128u + half * 64u + ql_l;
        let ql_byte = (ql[ql_bi >> 2u] >> ((ql_bi & 3u) * 8u)) & 0xffu;
        let nib = select(ql_byte >> 4u, ql_byte & 0xfu, sub < 2u);
        let qh_bi = blk * 64u + half * 32u + l;
        let qh_byte = (qh[qh_bi >> 2u] >> ((qh_bi & 3u) * 8u)) & 0xffu;
        let qbits = (qh_byte >> (sub * 2u)) & 3u;
        let q = i32(nib | (qbits << 4u)) - 32;
        let scale = scl[blk * 16u + half * 8u + (l / 16u) + 2u * sub];
        acc = acc + dd[blk] * scale * f32(q) * x[e];
        e = e + 64u;
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let oi = p.out_base + row;
        if (p.acc == 1u) { outp[oi] = outp[oi] + partial[0]; } else { outp[oi] = partial[0]; }
    }
}
"#;

/// Fused FFN down-projection (Q4_K): identical to the Q4_K matvec but the
/// activation element is `silu(gate[k]) * up[k]`, computed inline — removes
/// the separate silu_mul dispatch. `gate`/`up` replace `x`.
pub(super) const Q4K_DOWN_WGSL: &str = r#"
struct Params { n_rows: u32, nb_per_row: u32, gx: u32, acc: u32, out_base: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       wq:   array<u32>;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read>       up:   array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform>             p:    Params;
const BLOCK_U32: u32 = 36u;
var<workgroup> partial: array<f32, 64>;
fn act(idx: u32) -> f32 { let g = gate[idx]; return (g / (1.0 + exp(-g))) * up[idx]; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * p.gx;
    if (row >= p.n_rows) { return; }
    let t = lid.x;
    let total_sub = p.nb_per_row * 8u;
    var acc: f32 = 0.0;
    var g = t;
    loop {
        if (g >= total_sub) { break; }
        let b = g / 8u;
        let sub = g % 8u;
        let blk = (row * p.nb_per_row + b) * BLOCK_U32;
        let dd = unpack2x16float(wq[blk]);
        let d = dd.x; let dmin = dd.y;
        var u0 = wq[blk + 1u]; var u1 = wq[blk + 2u]; var u2 = wq[blk + 3u];
        let u3 = ((u2 >> 4u) & 0x0f0f0f0fu) | (((u1 >> 6u) & 0x03030303u) << 4u);
        let uaux = u1 & 0x3f3f3f3fu;
        u1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6u) & 0x03030303u) << 4u);
        u2 = uaux;
        u0 = u0 & 0x3f3f3f3fu;
        var sc: f32; var mn: f32;
        if (sub < 4u) { sc = f32((u0 >> (sub*8u)) & 0xffu); mn = f32((u2 >> (sub*8u)) & 0xffu); }
        else          { sc = f32((u1 >> ((sub-4u)*8u)) & 0xffu); mn = f32((u3 >> ((sub-4u)*8u)) & 0xffu); }
        let pair = sub / 2u;
        let hi = (sub & 1u) == 1u;
        let qs0 = blk + 4u + pair * 8u;
        let xb = b * 256u + sub * 32u;
        var dot: f32 = 0.0; var xsum: f32 = 0.0;
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
            let word = wq[qs0 + w];
            for (var bsel: u32 = 0u; bsel < 4u; bsel = bsel + 1u) {
                let byte = (word >> (bsel * 8u)) & 0xffu;
                var q: u32; if (hi) { q = byte >> 4u; } else { q = byte & 0x0fu; }
                let xv = act(xb + w * 4u + bsel);
                dot = dot + f32(q) * xv;
                xsum = xsum + xv;
            }
        }
        acc = acc + d * sc * dot - dmin * mn * xsum;
        g = g + 64u;
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let oi = p.out_base + row;
        if (p.acc == 1u) { outp[oi] = outp[oi] + partial[0]; } else { outp[oi] = partial[0]; }
    }
}
"#;

/// Fused FFN down-projection (Q6_K): Q6_K matvec with `silu(gate)*up` activation.
pub(super) const Q6K_DOWN_WGSL: &str = r#"
struct P6 { n_rows: u32, nb: u32, gx: u32, acc: u32, out_base: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       ql:   array<u32>;
@group(0) @binding(1) var<storage, read>       qh:   array<u32>;
@group(0) @binding(2) var<storage, read>       scl:  array<f32>;
@group(0) @binding(3) var<storage, read>       dd:   array<f32>;
@group(0) @binding(4) var<storage, read>       gate: array<f32>;
@group(0) @binding(5) var<storage, read>       up:   array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
@group(0) @binding(7) var<uniform>             p:    P6;
var<workgroup> partial: array<f32, 64>;
fn act(idx: u32) -> f32 { let g = gate[idx]; return (g / (1.0 + exp(-g))) * up[idx]; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * p.gx;
    if (row >= p.n_rows) { return; }
    let t = lid.x;
    let ncols = p.nb * 256u;
    var acc: f32 = 0.0;
    var e = t;
    loop {
        if (e >= ncols) { break; }
        let b = e / 256u;
        let pp = e % 256u;
        let half = pp / 128u;
        let pq = pp % 128u;
        let sub = pq / 32u;
        let l = pq % 32u;
        let blk = row * p.nb + b;
        let ql_l = l + (sub & 1u) * 32u;
        let ql_bi = blk * 128u + half * 64u + ql_l;
        let ql_byte = (ql[ql_bi >> 2u] >> ((ql_bi & 3u) * 8u)) & 0xffu;
        let nib = select(ql_byte >> 4u, ql_byte & 0xfu, sub < 2u);
        let qh_bi = blk * 64u + half * 32u + l;
        let qh_byte = (qh[qh_bi >> 2u] >> ((qh_bi & 3u) * 8u)) & 0xffu;
        let qbits = (qh_byte >> (sub * 2u)) & 3u;
        let q = i32(nib | (qbits << 4u)) - 32;
        let scale = scl[blk * 16u + half * 8u + (l / 16u) + 2u * sub];
        acc = acc + dd[blk] * scale * f32(q) * act(e);
        e = e + 64u;
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let oi = p.out_base + row;
        if (p.acc == 1u) { outp[oi] = outp[oi] + partial[0]; } else { outp[oi] = partial[0]; }
    }
}
"#;

/// Batched Q4_K GEMM for PREFILL: `out[m,n] = sum_k dequant(W)[n,k] * x[m,k]`
/// for m in 0..M prompt rows. One workgroup per output row n: the 64 threads
/// cooperatively dequantize weight row n into shared memory ONCE, then each
/// thread computes that row's dot against its strided set of prompt rows —
/// so each weight is read once and reused across all M rows (the compute-
/// bound amortization that makes prefill fast). x is [M, n_cols] row-major,
/// out is [M, n_rows] row-major. n_cols ≤ 2048 (shared-mem row).
pub(super) const Q4K_GEMM_WGSL: &str = r#"
struct GP { n_rows: u32, nb: u32, n_cols: u32, m_rows: u32, gx: u32, acc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read>       wq:   array<u32>;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform>             p:    GP;
const BLOCK_U32: u32 = 36u;
// 256-wide tile (1 KB LDS) — the sweet spot on RDNA3.5: small enough that LDS
// no longer caps workgroup occupancy (8 KB tile → only ~8 workgroups/WGP, so
// weight-read latency stalled), big enough that the per-chunk barrier count and
// dequant parallelism stay healthy. Measured ~2x throughput + 2.8x lower TTFT
// vs a 2048 tile. (Sweep: 2048→243, 512→313, 256→492, 128→408 tok/s at M=256.)
const TILE: u32 = 256u;
var<workgroup> wrow: array<f32, 256>;

// K-tiled batched GEMM: a workgroup owns output row n. The weight row is
// dequantized into shared mem one TILE-wide chunk at a time and reused across
// all M prompt rows; per-thread accumulators carry across chunks. acc[8] caps
// M at 512 (longer prompts are processed in M-chunks by the caller).
// workgroup_size 64 (2 wave32s) is the sweet spot — 128 dropped occupancy and
// was ~4x slower at M=256, despite halving per-thread dot rows.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * p.gx;
    if (n >= p.n_rows) { return; }
    let t = lid.x;
    var acc: array<f32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc[i] = 0.0; }
    let n_chunks = (p.n_cols + TILE - 1u) / TILE;
    for (var chunk: u32 = 0u; chunk < n_chunks; chunk = chunk + 1u) {
        let col0 = chunk * TILE;
        let cn = min(TILE, p.n_cols - col0);   // cols in this chunk
        // Cooperatively dequantize this chunk of weight row n into wrow, one
        // sub-block per thread (the scale unpack is done once per sub-block;
        // a per-element variant that uses all 64 threads was tried and lost —
        // the redundant per-element scale unpacking cost more than idle threads).
        let sub_start = col0 / 32u;            // global sub-block index in the row
        let sub_count = cn / 32u;
        var sg = t;
        loop {
            if (sg >= sub_count) { break; }
            let gsub = sub_start + sg;
            let b = gsub / 8u;
            let sub = gsub % 8u;
            let blk = (n * p.nb + b) * BLOCK_U32;
            let dd = unpack2x16float(wq[blk]);
            let d = dd.x; let dmin = dd.y;
            var u0 = wq[blk + 1u]; var u1 = wq[blk + 2u]; var u2 = wq[blk + 3u];
            let u3 = ((u2 >> 4u) & 0x0f0f0f0fu) | (((u1 >> 6u) & 0x03030303u) << 4u);
            let uaux = u1 & 0x3f3f3f3fu;
            u1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6u) & 0x03030303u) << 4u);
            u2 = uaux;
            u0 = u0 & 0x3f3f3f3fu;
            var sc: f32; var mn: f32;
            if (sub < 4u) { sc = f32((u0 >> (sub*8u)) & 0xffu); mn = f32((u2 >> (sub*8u)) & 0xffu); }
            else          { sc = f32((u1 >> ((sub-4u)*8u)) & 0xffu); mn = f32((u3 >> ((sub-4u)*8u)) & 0xffu); }
            let pair = sub / 2u;
            let hi = (sub & 1u) == 1u;
            let qs0 = blk + 4u + pair * 8u;
            let dst = sg * 32u;
            let dsc = d * sc; let dmn = dmin * mn;
            for (var w: u32 = 0u; w < 8u; w = w + 1u) {
                let word = wq[qs0 + w];
                for (var bsel: u32 = 0u; bsel < 4u; bsel = bsel + 1u) {
                    let byte = (word >> (bsel * 8u)) & 0xffu;
                    var q: u32; if (hi) { q = byte >> 4u; } else { q = byte & 0x0fu; }
                    wrow[dst + w * 4u + bsel] = dsc * f32(q) - dmn;
                }
            }
            sg = sg + 64u;
        }
        workgroupBarrier();
        var mi: u32 = 0u; var m = t;
        loop {
            if (m >= p.m_rows) { break; }
            let xb = m * p.n_cols + col0;
            var dot: f32 = 0.0;
            // cn is always a multiple of 32 — unroll the inner dot 8-wide so the
            // driver vectorizes the shared-mem weight reads + x loads (this loop
            // is ALU/latency-bound, not bandwidth-bound: the unroll is ~2.4x).
            for (var k: u32 = 0u; k < cn; k = k + 8u) {
                dot = dot + wrow[k] * x[xb + k] + wrow[k + 1u] * x[xb + k + 1u]
                          + wrow[k + 2u] * x[xb + k + 2u] + wrow[k + 3u] * x[xb + k + 3u]
                          + wrow[k + 4u] * x[xb + k + 4u] + wrow[k + 5u] * x[xb + k + 5u]
                          + wrow[k + 6u] * x[xb + k + 6u] + wrow[k + 7u] * x[xb + k + 7u];
            }
            acc[mi] = acc[mi] + dot;
            mi = mi + 1u; m = m + 64u;
        }
        workgroupBarrier();
    }
    var mi: u32 = 0u; var m = t;
    loop {
        if (m >= p.m_rows) { break; }
        let oi = m * p.n_rows + n;
        if (p.acc == 1u) { outp[oi] = outp[oi] + acc[mi]; } else { outp[oi] = acc[mi]; }
        mi = mi + 1u; m = m + 64u;
    }
}
"#;

/// Batched Q6_K GEMM for prefill (K-tiled), mirroring Q4K_GEMM but with the
/// Q6_K per-element dequant into the shared weight-row tile.
pub(super) const Q6K_GEMM_WGSL: &str = r#"
struct GP { n_rows: u32, nb: u32, n_cols: u32, m_rows: u32, gx: u32, acc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read>       ql:   array<u32>;
@group(0) @binding(1) var<storage, read>       qh:   array<u32>;
@group(0) @binding(2) var<storage, read>       scl:  array<f32>;
@group(0) @binding(3) var<storage, read>       dd:   array<f32>;
@group(0) @binding(4) var<storage, read>       x:    array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(6) var<uniform>             p:    GP;
const TILE: u32 = 256u;                   // 1 KB LDS — see Q4K_GEMM note on occupancy
var<workgroup> wrow: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * p.gx;
    if (n >= p.n_rows) { return; }
    let t = lid.x;
    var acc: array<f32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc[i] = 0.0; }
    let n_chunks = (p.n_cols + TILE - 1u) / TILE;
    for (var chunk: u32 = 0u; chunk < n_chunks; chunk = chunk + 1u) {
        let col0 = chunk * TILE;
        let cn = min(TILE, p.n_cols - col0);
        var e = t;
        loop {
            if (e >= cn) { break; }
            let col = col0 + e;
            let b = col / 256u;
            let pp = col % 256u;
            let half = pp / 128u;
            let pq = pp % 128u;
            let sub = pq / 32u;
            let l = pq % 32u;
            let blk = n * p.nb + b;
            let ql_l = l + (sub & 1u) * 32u;
            let ql_bi = blk * 128u + half * 64u + ql_l;
            let ql_byte = (ql[ql_bi >> 2u] >> ((ql_bi & 3u) * 8u)) & 0xffu;
            let nib = select(ql_byte >> 4u, ql_byte & 0xfu, sub < 2u);
            let qh_bi = blk * 64u + half * 32u + l;
            let qh_byte = (qh[qh_bi >> 2u] >> ((qh_bi & 3u) * 8u)) & 0xffu;
            let qbits = (qh_byte >> (sub * 2u)) & 3u;
            let q = i32(nib | (qbits << 4u)) - 32;
            let scale = scl[blk * 16u + half * 8u + (l / 16u) + 2u * sub];
            wrow[e] = dd[blk] * scale * f32(q);
            e = e + 64u;
        }
        workgroupBarrier();
        var mi: u32 = 0u; var m = t;
        loop {
            if (m >= p.m_rows) { break; }
            let xb = m * p.n_cols + col0;
            var dot: f32 = 0.0;
            for (var k: u32 = 0u; k < cn; k = k + 8u) {
                dot = dot + wrow[k] * x[xb + k] + wrow[k + 1u] * x[xb + k + 1u]
                          + wrow[k + 2u] * x[xb + k + 2u] + wrow[k + 3u] * x[xb + k + 3u]
                          + wrow[k + 4u] * x[xb + k + 4u] + wrow[k + 5u] * x[xb + k + 5u]
                          + wrow[k + 6u] * x[xb + k + 6u] + wrow[k + 7u] * x[xb + k + 7u];
            }
            acc[mi] = acc[mi] + dot;
            mi = mi + 1u; m = m + 64u;
        }
        workgroupBarrier();
    }
    var mi: u32 = 0u; var m = t;
    loop {
        if (m >= p.m_rows) { break; }
        let oi = m * p.n_rows + n;
        if (p.acc == 1u) { outp[oi] = outp[oi] + acc[mi]; } else { outp[oi] = acc[mi]; }
        mi = mi + 1u; m = m + 64u;
    }
}
"#;

/// Batched RMSNorm: one workgroup per prompt row m of `x[M, n]` → `y[M, n]`.
pub(super) const BNORM_WGSL: &str = r#"
struct NP { n: u32, eps: u32 };
@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       wgt: array<f32>;
@group(0) @binding(2) var<storage, read_write> y:   array<f32>;
@group(0) @binding(3) var<uniform>             np:  NP;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = wid.x; let t = lid.x; let n = np.n; let base = m * n;
    var s: f32 = 0.0; var i = t;
    loop { if (i >= n) { break; } let v = x[base + i]; s = s + v * v; i = i + 256u; }
    partial[t] = s; workgroupBarrier();
    var stride = 128u;
    loop { if (stride == 0u) { break; } if (t < stride) { partial[t] = partial[t] + partial[t + stride]; } workgroupBarrier(); stride = stride / 2u; }
    let inv = 1.0 / sqrt(partial[0] / f32(n) + bitcast<f32>(np.eps));
    i = t; loop { if (i >= n) { break; } y[base + i] = x[base + i] * inv * wgt[i]; i = i + 256u; }
}
"#;

/// Batched interleaved RoPE over M tokens; cos/sin are `[M, head_dim/2]`
/// (precomputed for positions pos..pos+M). One thread per (token, head, pair).
pub(super) const BROPE_WGSL: &str = r#"
struct RP { n_head: u32, head_dim: u32, m_rows: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> x:    array<f32>;
@group(0) @binding(1) var<storage, read>       cosb: array<f32>;
@group(0) @binding(2) var<storage, read>       sinb: array<f32>;
@group(0) @binding(3) var<uniform>             p:    RP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.head_dim / 2u;
    let per = p.n_head * half;
    let idx = gid.x;
    if (idx >= p.m_rows * per) { return; }
    let m = idx / per; let r = idx % per;
    let h = r / half; let j = r % half;
    let c = cosb[m * half + j]; let s = sinb[m * half + j];
    let base = m * (p.n_head * p.head_dim) + h * p.head_dim + 2u * j;
    let a = x[base]; let b = x[base + 1u];
    x[base] = a * c - b * s; x[base + 1u] = a * s + b * c;
}
"#;

/// Batched causal GQA SDPA for prefill: thread per (token m, query head h);
/// query at position pos+m attends causally to cache positions 0..=pos+m.
pub(super) const BSDPA_WGSL: &str = r#"
struct SP { n_head: u32, n_kv_head: u32, head_dim: u32, m_rows: u32, pos: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform>             p:    SP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.m_rows * p.n_head) { return; }
    let m = idx / p.n_head; let h = idx % p.n_head;
    let hd = p.head_dim;
    let kvh = h / (p.n_head / p.n_kv_head);
    let scale = 1.0 / sqrt(f32(hd));
    let seq_len = p.pos + m + 1u;
    let q_base = m * (p.n_head * hd) + h * hd;
    var av: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { av[d] = 0.0; }
    var mx: f32 = -1e30; var l: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let kv_base = (t * p.n_kv_head + kvh) * hd;
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { s = s + q[q_base + d] * kc[kv_base + d]; }
        s = s * scale;
        let m_new = max(mx, s); let corr = exp(mx - m_new); let pe = exp(s - m_new);
        l = l * corr + pe;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { av[d] = av[d] * corr + pe * vc[kv_base + d]; }
        mx = m_new;
    }
    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { outp[q_base + d] = av[d] * inv; }
}
"#;

/// Batched DECODE SDPA: M independent concurrent streams, each a single query
/// attending its OWN KV cache at its OWN position. Thread per (stream s, head
/// h); stream s's cache occupies [s*max_seq*kv_dim ..], `posb[s]` is its last
/// filled position (attends 0..=posb[s]). This is the coalesced-serving kernel:
/// the matmuls around it run once for all M streams (weights amortized), only
/// the attention is per-stream.
pub(super) const BDSDPA_WGSL: &str = r#"
struct BP { n_head: u32, n_kv_head: u32, head_dim: u32, m_streams: u32, max_seq: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read>       posb: array<u32>;
@group(0) @binding(5) var<storage, read>       slots: array<u32>;  // cache slot for each batch position
@group(0) @binding(6) var<uniform>             p:    BP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.m_streams * p.n_head) { return; }
    let s = idx / p.n_head; let h = idx % p.n_head;
    let hd = p.head_dim;
    let kvh = h / (p.n_head / p.n_kv_head);
    let scale = 1.0 / sqrt(f32(hd));
    let seq_len = posb[s] + 1u;
    let q_base = s * (p.n_head * hd) + h * hd;
    let stream_kv = slots[s] * p.max_seq * p.n_kv_head * hd;   // base of this seq's cache slot
    var av: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { av[d] = 0.0; }
    var mx: f32 = -1e30; var l: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let kv_base = stream_kv + (t * p.n_kv_head + kvh) * hd;
        var sdot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { sdot = sdot + q[q_base + d] * kc[kv_base + d]; }
        sdot = sdot * scale;
        let m_new = max(mx, sdot); let corr = exp(mx - m_new); let pe = exp(sdot - m_new);
        l = l * corr + pe;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { av[d] = av[d] * corr + pe * vc[kv_base + d]; }
        mx = m_new;
    }
    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { outp[q_base + d] = av[d] * inv; }
}
"#;

/// PagedAttention decode SDPA. Identical math to BDSDPA, but the KV is a shared
/// block pool and each key position `t` is gathered through a per-slot block
/// table: physical position = block_table[slot*max_blocks + t/block_size] *
/// block_size + t%block_size. Lets sequences hold non-contiguous KV and share a
/// pool sized for *actual* usage rather than m_max × max_seq.
pub(super) const BDSDPA_PAGED_WGSL: &str = r#"
// Block pool is PACKED F16 (two dims per u32, pack2x16float) — halves KV bytes
// per token and doubles resident streams per GB. f16 KV is llama.cpp's default
// precision; unpacked pairs feed the same fp32 online-softmax accumulation.
struct BP { n_head: u32, n_kv_head: u32, head_dim: u32, m_streams: u32, block_size: u32, max_blocks: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       kc:   array<u32>;   // packed-f16 pool: n_blocks*block_size*kv_dim/2
@group(0) @binding(2) var<storage, read>       vc:   array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read>       posb: array<u32>;
@group(0) @binding(5) var<storage, read>       slots: array<u32>;
@group(0) @binding(6) var<storage, read>       block_table: array<u32>;  // [n_slots * max_blocks]
@group(0) @binding(7) var<uniform>             p:    BP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.m_streams * p.n_head) { return; }
    let s = idx / p.n_head; let h = idx % p.n_head;
    let hd = p.head_dim;
    let hd2 = hd / 2u;
    let kvh = h / (p.n_head / p.n_kv_head);
    let scale = 1.0 / sqrt(f32(hd));
    let seq_len = posb[s] + 1u;
    let q_base = s * (p.n_head * hd) + h * hd;
    let bt_base = slots[s] * p.max_blocks;   // this slot's block-table row
    var av: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { av[d] = 0.0; }
    var mx: f32 = -1e30; var l: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let phys_block = block_table[bt_base + t / p.block_size];
        let phys_pos = phys_block * p.block_size + (t % p.block_size);
        let kv_base = (phys_pos * p.n_kv_head + kvh) * hd2;
        var sdot: f32 = 0.0;
        for (var j: u32 = 0u; j < hd2; j = j + 1u) {
            let k2 = unpack2x16float(kc[kv_base + j]);
            sdot = sdot + q[q_base + 2u * j] * k2.x + q[q_base + 2u * j + 1u] * k2.y;
        }
        sdot = sdot * scale;
        let m_new = max(mx, sdot); let corr = exp(mx - m_new); let pe = exp(sdot - m_new);
        l = l * corr + pe;
        for (var j: u32 = 0u; j < hd2; j = j + 1u) {
            let v2 = unpack2x16float(vc[kv_base + j]);
            av[2u * j] = av[2u * j] * corr + pe * v2.x;
            av[2u * j + 1u] = av[2u * j + 1u] * corr + pe * v2.y;
        }
        mx = m_new;
    }
    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { outp[q_base + d] = av[d] * inv; }
}
"#;

/// Pack-and-scatter this step's new K/V rows into the packed-f16 block pool.
/// Replaces the per-row `copy_buffer_to_buffer` scatter (m×2 copy commands with
/// an f32 pool): one dispatch converts the f32 staging rows → packed f16 at each
/// row's physical pool position, computed in-shader from (slots, posb,
/// block_table) — the same indexing the paged SDPA uses for the current token.
pub(super) const BKV_PACK_WGSL: &str = r#"
struct KP { kv_dim: u32, m_rows: u32, block_size: u32, max_blocks: u32 };
@group(0) @binding(0) var<storage, read>       k_src: array<f32>;   // [m, kv_dim] roped K
@group(0) @binding(1) var<storage, read>       v_src: array<f32>;   // [m, kv_dim]
@group(0) @binding(2) var<storage, read_write> k_pool: array<u32>;  // packed f16
@group(0) @binding(3) var<storage, read_write> v_pool: array<u32>;
@group(0) @binding(4) var<storage, read>       posb: array<u32>;
@group(0) @binding(5) var<storage, read>       slots: array<u32>;
@group(0) @binding(6) var<storage, read>       block_table: array<u32>;
@group(0) @binding(7) var<uniform>             p: KP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kd2 = p.kv_dim / 2u;
    let idx = gid.x;
    if (idx >= p.m_rows * kd2) { return; }
    let s = idx / kd2; let j = idx % kd2;
    let pos = posb[s];
    let phys_block = block_table[slots[s] * p.max_blocks + pos / p.block_size];
    let phys_pos = phys_block * p.block_size + (pos % p.block_size);
    let src = s * p.kv_dim + 2u * j;
    let dst = phys_pos * kd2 + j;
    k_pool[dst] = pack2x16float(vec2<f32>(k_src[src], k_src[src + 1u]));
    v_pool[dst] = pack2x16float(vec2<f32>(v_src[src], v_src[src + 1u]));
}
"#;

/// Batched argmax: one workgroup per stream reduces that stream's `vocab`-wide
/// logit row of `logits[M, vocab]` to its argmax → `out_idx[s]`. Lets batched
/// decode read back M u32s instead of M*128k logits.
pub(super) const BARGMAX_WGSL: &str = r#"
struct BA { vocab: u32, m_streams: u32 };
@group(0) @binding(0) var<storage, read>       logits:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<uniform>             p:       BA;
var<workgroup> vmax: array<f32, 256>;
var<workgroup> imax: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wid.x; let t = lid.x;
    if (s >= p.m_streams) { return; }
    let base = s * p.vocab;
    var bv: f32 = -1e30; var bi: u32 = 0u;
    var k = t;
    loop { if (k >= p.vocab) { break; } let v = logits[base + k]; if (v > bv) { bv = v; bi = k; } k = k + 256u; }
    vmax[t] = bv; imax[t] = bi;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            if (vmax[t + stride] > vmax[t]) { vmax[t] = vmax[t + stride]; imax[t] = imax[t + stride]; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { out_idx[s] = imax[0]; }
}
"#;

/// Batched temperature sampling via the Gumbel-max trick: a categorical draw
/// from softmax(logits/temp) equals argmax_i(logits[i]/temp + g_i) with
/// g_i = -log(-log(u_i)), u_i ~ Uniform(0,1). So this is BARGMAX over a perturbed
/// score — same workgroup-per-stream reduction, no full-logit readback. Per
/// stream: `temp[s]` (≤0 → greedy, no noise) and `seed[s]` (advance each step).
pub(super) const BSAMPLE_WGSL: &str = r#"
struct BS { vocab: u32, m_streams: u32 };
@group(0) @binding(0) var<storage, read>       logits:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<storage, read>       temp:    array<f32>;
@group(0) @binding(3) var<storage, read>       seed:    array<u32>;
@group(0) @binding(4) var<uniform>             p:       BS;
var<workgroup> vmax: array<f32, 256>;
var<workgroup> imax: array<u32, 256>;
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u); h = h * 0x7feb352du;
    h = h ^ (h >> 15u); h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}
fn rand01(s: u32, i: u32) -> f32 {
    let r = hash_u32(s ^ hash_u32(i));
    return (f32(r >> 8u) + 0.5) / 16777216.0;   // strictly in (0,1)
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wid.x; let t = lid.x;
    if (s >= p.m_streams) { return; }
    let base = s * p.vocab;
    let temp_s = temp[s];
    let greedy = temp_s <= 0.0;
    let inv_t = select(1.0 / temp_s, 1.0, greedy);
    let sd = seed[s];
    var bv: f32 = -1e30; var bi: u32 = 0u;
    var k = t;
    loop {
        if (k >= p.vocab) { break; }
        var v = logits[base + k];
        if (!greedy) {
            let u = rand01(sd, k);
            v = v * inv_t + (-log(-log(u)));     // logit/temp + Gumbel noise
        }
        if (v > bv) { bv = v; bi = k; }
        k = k + 256u;
    }
    vmax[t] = bv; imax[t] = bi;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            if (vmax[t + stride] > vmax[t]) { vmax[t] = vmax[t + stride]; imax[t] = imax[t + stride]; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { out_idx[s] = imax[0]; }
}
"#;

/// Batched top-K extraction: one workgroup per stream writes that stream's K
/// largest logits (value + index) in descending order to `vals[s*K..]` /
/// `idxs[s*K..]`. Found by K rounds of "largest (value,index) strictly below the
/// previous winner" (lexicographic: value desc, index asc → deterministic ties).
/// Only M*K pairs are read back (vs M*vocab), so the CPU can apply
/// top-k/top-p/temperature flexibly without the full-logit cliff. `K` is the
/// dispatch's `p.k` (≤ KMAX).
pub(super) const BTOPK_WGSL: &str = r#"
struct BT { vocab: u32, m_streams: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> vals:   array<f32>;
@group(0) @binding(2) var<storage, read_write> idxs:   array<u32>;
@group(0) @binding(3) var<uniform>             p:      BT;
var<workgroup> wv: array<f32, 256>;
var<workgroup> wi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wid.x; let t = lid.x;
    if (s >= p.m_streams) { return; }
    let base = s * p.vocab;
    // prev = the previous round's winner (value, index); seed above any logit.
    var prev_v: f32 = 3.0e38; var prev_i: u32 = 0xffffffffu;
    for (var r: u32 = 0u; r < p.k; r = r + 1u) {
        // Each thread's best candidate strictly below `prev` in (val desc, idx asc).
        var bv: f32 = -3.0e38; var bi: u32 = 0xffffffffu;
        var c = t;
        loop {
            if (c >= p.vocab) { break; }
            let v = logits[base + c];
            // below prev?  v < prev_v  OR  (v == prev_v AND c > prev_i)
            let below = (v < prev_v) || (v == prev_v && c > prev_i);
            // better than current best?  v > bv  OR  (v == bv AND c < bi)
            let better = (v > bv) || (v == bv && c < bi);
            if (below && better) { bv = v; bi = c; }
            c = c + 256u;
        }
        wv[t] = bv; wi[t] = bi;
        workgroupBarrier();
        var stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) {
                let av = wv[t]; let ai = wi[t]; let bv2 = wv[t + stride]; let bi2 = wi[t + stride];
                if ((bv2 > av) || (bv2 == av && bi2 < ai)) { wv[t] = bv2; wi[t] = bi2; }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) { vals[s * p.k + r] = wv[0]; idxs[s * p.k + r] = wi[0]; }
        prev_v = wv[0]; prev_i = wi[0];
        workgroupBarrier();
    }
}
"#;

/// Argmax over `logits` (one workgroup, 256 threads, strided scan + reduce).
/// Writes the winning index to `out_idx[0]`. Strict `>` keeps the lowest
/// index on ties — matching a first-max CPU argmax.
pub(super) const ARGMAX_WGSL: &str = r#"
struct PA { n: u32 };
@group(0) @binding(0) var<storage, read>       logits:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<uniform>             p:       PA;
var<workgroup> vmax: array<f32, 256>;
var<workgroup> imax: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var bv: f32 = -1e30;
    var bi: u32 = 0u;
    var k = t;
    loop { if (k >= p.n) { break; } let v = logits[k]; if (v > bv) { bv = v; bi = k; } k = k + 256u; }
    vmax[t] = bv; imax[t] = bi;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            if (vmax[t + stride] > vmax[t]) { vmax[t] = vmax[t + stride]; imax[t] = imax[t + stride]; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { out_idx[0] = imax[0]; }
}
"#;

/// Dense f32 mat-vec, **coalesced**: ONE workgroup (64 threads) per output
/// row; thread `t` reads w[row,t], w[row,t+64], … so adjacent threads read
/// adjacent memory (coalesced — the key to hitting memory bandwidth), then a
/// shared-memory reduction. 2D workgroup grid (row = wg.x + wg.y*gx) because
/// n_rows (128256) exceeds the 65535 per-dimension dispatch limit.
pub(super) const F32_MATVEC_WGSL: &str = r#"
struct PF { n_rows: u32, n_cols: u32, gx: u32, acc: u32, out_base: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       w:    array<f32>;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform>             p:    PF;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * p.gx;
    if (row >= p.n_rows) { return; }
    let t = lid.x;
    let base = row * p.n_cols;
    var s: f32 = 0.0;
    var k = t;
    loop { if (k >= p.n_cols) { break; } s = s + w[base + k] * x[k]; k = k + 64u; }
    partial[t] = s;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let oi = p.out_base + row;
        if (p.acc == 1u) { outp[oi] = outp[oi] + partial[0]; } else { outp[oi] = partial[0]; }
    }
}
"#;

/// Fused FFN activation: `h[i] = silu(gate[i]) * up[i]`, with
/// `silu(x) = x * sigmoid(x)`. Elementwise; keeps `h` GPU-resident.
pub(super) const SILU_MUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up:   array<f32>;
@group(0) @binding(2) var<storage, read_write>  h:    array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&gate)) {
        let g = gate[i];
        h[i] = (g / (1.0 + exp(-g))) * up[i];
    }
}
"#;

/// Interleaved RoPE (`rope_i`), applied in-place. The last dim is treated
/// as adjacent pairs (x[2j], x[2j+1]) rotated by angle from cos/sin[j];
/// cos/sin (length head_dim/2) are the precomputed tables for the current
/// position, shared across all heads. One thread per (head, pair).
pub(super) const ROPE_WGSL: &str = r#"
struct RopeP { n_head: u32, head_dim: u32, base: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> x:    array<f32>;
@group(0) @binding(1) var<storage, read>       cosb: array<f32>;
@group(0) @binding(2) var<storage, read>       sinb: array<f32>;
@group(0) @binding(3) var<uniform>             rp:   RopeP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rp.head_dim / 2u;
    let total = rp.n_head * half;
    let t = gid.x;
    if (t >= total) { return; }
    let h = t / half;
    let j = t % half;
    let base = rp.base + h * rp.head_dim + 2u * j;
    let a = x[base];
    let b = x[base + 1u];
    let c = cosb[j];
    let s = sinb[j];
    x[base]      = a * c - b * s;
    x[base + 1u] = a * s + b * c;
}
"#;

/// GQA decode self-attention with online (flash-style) softmax. One thread
/// per query head: streams the cached positions once, tracking running max
/// `m`, denominator `l`, and weighted-V accumulator — no separate score
/// buffer, numerically stable. Decode needs no causal mask (the new token
/// attends to all `seq_len` cached positions, itself included). GQA: query
/// head `h` reads kv head `h / (n_head/n_kv_head)`. head_dim ≤ 128.
pub(super) const SDPA_DECODE_WGSL: &str = r#"
struct SdpaP { n_head: u32, n_kv_head: u32, head_dim: u32, seq_len: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // n_head*head_dim
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;  // seq_len*n_kv_head*head_dim
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;  // same
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;  // n_head*head_dim
@group(0) @binding(4) var<uniform>             p:    SdpaP;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.n_head) { return; }
    let hd = p.head_dim;
    let kvh = h / (p.n_head / p.n_kv_head);
    let scale = 1.0 / sqrt(f32(hd));
    let q_base = h * hd;
    var acc: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = 0.0; }
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    for (var t: u32 = 0u; t < p.seq_len; t = t + 1u) {
        let kv_base = (t * p.n_kv_head + kvh) * hd;
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { s = s + q[q_base + d] * kc[kv_base + d]; }
        s = s * scale;
        let m_new = max(m, s);
        let corr = exp(m - m_new);
        let pe = exp(s - m_new);
        l = l * corr + pe;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = acc[d] * corr + pe * vc[kv_base + d]; }
        m = m_new;
    }
    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { outp[q_base + d] = acc[d] * inv; }
}
"#;

/// Llama RMSNorm: `y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]`.
/// One workgroup of 256 threads; sum-of-squares via shared-memory
/// reduction. Out-of-place (x preserved for the residual).
pub(super) const RMSNORM_WGSL: &str = r#"
struct NormP { n: u32, eps: f32 };
@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       wgt: array<f32>;
@group(0) @binding(2) var<storage, read_write> y:   array<f32>;
@group(0) @binding(3) var<uniform>             np:  NormP;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = np.n;
    var s: f32 = 0.0;
    var i = tid;
    loop { if (i >= n) { break; } let v = x[i]; s = s + v * v; i = i + 256u; }
    partial[tid] = s;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(partial[0] / f32(n) + np.eps);
    i = tid;
    loop { if (i >= n) { break; } y[i] = x[i] * inv * wgt[i]; i = i + 256u; }
}
"#;

/// In-place residual add: `a[i] += b[i]`.
pub(super) const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) { a[i] = a[i] + b[i]; }
}
"#;

