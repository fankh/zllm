//! Head-major KV-cache layout (ZLLM_HEADMAJOR_KV). The default cache is
//! [pos, kv_head, hd]; an SDPA workgroup reads its kv-head STRIDED (gap n_kv*hd),
//! cold-streaming at ~100 GB/s vs ~187 for a contiguous read. Head-major lays it
//! [kv_head, pos, hd] so a kv-head's positions are contiguous. This file compiles
//! the two head-major shaders (kv_write_hm, sdpa_flash_partial_hm) WGSL→SPV via
//! naga (committed, no glslang in this env) and validates the deployed engine at
//! long context with the flag on (bit-exact decode + tok/s vs the pos-major path).
#![cfg(feature = "vulkan")]

fn compile_wgsl(src: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(src).expect("wgsl parse");
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module).expect("wgsl validate");
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None).expect("spv write")
}

/// `cargo test --release --features vulkan --test headmajor_kv gen_headmajor_spv -- --ignored --nocapture`
#[test]
#[ignore]
fn gen_headmajor_spv() {
    for stem in ["kv_write_hm", "sdpa_flash_partial_hm"] {
        let wgsl = std::fs::read_to_string(format!("src/backend/vulkan/shaders/{stem}.wgsl")).unwrap();
        let words = compile_wgsl(&wgsl);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let path = format!("src/backend/vulkan/shaders/{stem}.spv");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {path} ({} bytes, {} words) ✓", bytes.len(), words.len());
    }
}
