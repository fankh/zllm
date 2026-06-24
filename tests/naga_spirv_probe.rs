//! Feasibility probe: can naga (already in-tree via wgpu) compile a compute
//! shader to SPIR-V? The env has no glslang/glslc/SDK, and the ash VkModel path
//! needs committed `.spv`. If naga can emit valid SPIR-V for a workgroup compute
//! shader with storage buffers + a uniform, we can generate new kernels' `.spv`
//! offline here instead of needing the Vulkan SDK.

#[test]
fn naga_compiles_compute_to_spirv() {
    // A minimal compute shader shaped like a decode matvec: storage in/out + uniform.
    let wgsl = r#"
struct P { n: u32, k: u32 };
@group(0) @binding(0) var<storage, read>       w: array<u32>;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    var acc = 0.0;
    var t = lid.x;
    while (t < p.k) { acc = acc + f32(w[t] & 0xfu) * x[t]; t = t + 64u; }
    partial[lid.x] = acc;
    workgroupBarrier();
    var s = 32u;
    while (s > 0u) { if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; } workgroupBarrier(); s = s >> 1u; }
    if (lid.x == 0u && row < p.n) { o[row] = partial[0]; }
}
"#;
    let module = naga::front::wgsl::parse_str(wgsl).expect("naga wgsl parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("naga validate");
    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )
    .expect("naga spv write");
    // SPIR-V is a stream of 32-bit words; first word is the magic number.
    assert_eq!(spv[0], 0x0723_0203, "not a SPIR-V magic header");
    eprintln!("naga emitted {} SPIR-V words ({} bytes), magic OK ✓", spv.len(), spv.len() * 4);
}
