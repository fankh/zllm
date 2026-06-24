// Decode matvec (Q3_K): out[N] = dequant(W)[N,K] · x[K], M=1. One workgroup (64
// threads) per output row, like decode_matvec_q4k. Reads a PACKED Q3_K weight
// buffer (29 u32 / 256-weight block, 4-aligned), produced by vk_up_q3k:
//   [0]      d (f32 bits)
//   [1..5]   16 pre-shuffled 6-bit scale bytes (the ggml 12B→16 shuffle done on CPU)
//   [5..13]  hmask[32]  (quants high bit, inverted)
//   [13..29] qs[64]     (quants low 2 bits)
// Weight[out] = d * (scale[sub]-32) * ((qs>>shift)&3 - (hmask&m ? 0 : 4)).
// Validated bit-exact vs candle (q3k_dequant) + on-GPU (vk_q3k_matvec).
// SPIR-V generated offline by naga (gen_q3k_spv test) → decode_matvec_q3k.spv.
struct P { n: u32, k: u32, nb: u32, gx: u32 };
@group(0) @binding(0) var<storage, read>       w: array<u32>;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;
var<workgroup> partial: array<f32, 64>;
// Bandwidth layout: 64 threads == the 64 qs positions of a block (qs_idx = t).
// Thread t owns one qs byte + one hmask byte (adjacent threads share a u32 →
// coalesced) and produces the 4 weights at shifts j=0..3 from them. Iterates
// blocks in the outer loop. Far fewer / coalesced loads than per-weight indexing.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * (p.gx & 0xffffu);
    let t = lid.x;                  // 0..63 == qs_idx
    let on = row < p.n;
    let h = t >> 5u;                // qs_idx / 32
    let lo = t & 31u;               // hmask_idx = qs_idx % 32
    let sub = lo >> 4u;             // (qs_idx % 32) / 16
    var acc = 0.0;
    var b = 0u;
    loop {
        if (b >= p.nb || !on) { break; }
        let base = (row * p.nb + b) * 29u;
        let d = bitcast<f32>(w[base]);
        let qbyte = (w[base + 13u + (t >> 2u)] >> ((t & 3u) * 8u)) & 0xffu;
        let hbyte = (w[base + 5u + (lo >> 2u)] >> ((lo & 3u) * 8u)) & 0xffu;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let scale_idx = h * 8u + j * 2u + sub;
            let scale = f32(i32((w[base + 1u + (scale_idx >> 2u)] >> ((scale_idx & 3u) * 8u)) & 0xffu) - 32);
            let q2 = f32((qbyte >> (2u * j)) & 3u);
            let hbit = (hbyte & (1u << (h * 4u + j))) != 0u;
            let out_idx = h * 128u + j * 32u + lo;
            acc = acc + d * scale * (q2 - select(4.0, 0.0, hbit)) * x[b * 256u + out_idx];
        }
        b = b + 1u;
    }
    partial[t] = acc;
    workgroupBarrier();
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (t < s) { partial[t] = partial[t] + partial[t + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (t == 0u && on) { o[row] = partial[0]; }
}
