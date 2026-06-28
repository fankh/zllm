// One-time KV-cache transpose pos-major [seq, kv_dim] → head-major
// [kv_head, pos, hd], run at the prefill→decode boundary when ZLLM_HEADMAJOR_KV
// is on. The batched prefill writes the prompt's K/V pos-major (its own prompt
// self-attention reads pos-major); decode then reads head-major, so the cache is
// converted once here. `src` is a pos-major COPY of the cache (in-place transpose
// would alias); `dst` is the cache itself. One thread per element. Model dims are
// consts (Llama-3.2-1B). SPV: gen_headmajor_spv.
struct P { n: u32, p0: u32, p1: u32, p2: u32 };  // n = seq_len * kv_dim
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;  // head-major cache
@group(0) @binding(1) var<storage, read>       src: array<f32>;  // pos-major copy [seq, kv_dim]
@group(0) @binding(2) var<uniform>             p: P;
const HD: u32 = 64u;
const KV_DIM: u32 = 512u;
const MAX_SEQ: u32 = 4096u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= p.n) { return; }
    let pos = e / KV_DIM;
    let rem = e % KV_DIM;
    let kvh = rem / HD;
    let d = rem % HD;
    dst[kvh * MAX_SEQ * HD + pos * HD + d] = src[e];
}
