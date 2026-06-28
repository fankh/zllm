// Head-major KV-cache write (ZLLM_HEADMAJOR_KV). The default cache is
// [pos, kv_head, hd] (pos-major); each SDPA workgroup then reads its kv-head
// STRIDED (gap = n_kv*hd between positions), which cold-streams at ~100 GB/s vs
// the ~187 a contiguous read hits. Head-major lays it out [kv_head, pos, hd] so a
// kv-head's positions are CONTIGUOUS. This writes the new token's K (or V) into
// that layout. Model dims are consts (Llama-3.2-1B) so the uniform {n, base}
// is unchanged from kv_write.comp (base = pos*kv_dim). SPV: gen_headmajor_spv.
struct P { n: u32, base: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;  // head-major cache
@group(0) @binding(1) var<storage, read>       src: array<f32>;  // [n = kv_dim] roped K/V
@group(0) @binding(2) var<uniform>             p: P;
const HD: u32 = 64u;
const KV_DIM: u32 = 512u;
const MAX_SEQ: u32 = 4096u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let pos = p.base / KV_DIM;
    let kvh = i / HD;
    let d = i % HD;
    dst[kvh * MAX_SEQ * HD + pos * HD + d] = src[i];
}
