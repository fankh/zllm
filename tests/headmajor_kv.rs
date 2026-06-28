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

/// The deployed server prefills via the batched prefill_forward (pos-major), so
/// head-major needs the prefill→decode transpose. This loads the model twice (flag
/// off, then on), prefills the SAME prompt + decodes, and asserts the two token
/// streams are identical — i.e. the transpose + head-major prefill path is correct.
/// `cargo test --release --features vulkan --test headmajor_kv prefill_headmajor_matches -- --ignored --nocapture`
#[test]
#[ignore]
fn prefill_headmajor_matches() {
    use zllm::backend::vulkan::{VkContext, VkModel};
    let path = std::env::var("ZLLM_MODEL").unwrap_or_else(|_| "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string());
    if !std::path::Path::new(&path).exists() { eprintln!("model not found; skipping"); return; }
    let prompt: Vec<u32> = (0..96u32).map(|i| 100 + (i * 37) % 40000).collect(); // deterministic ~96-tok prompt (in-vocab)
    let n_gen = 40usize;
    let run = |hm: bool| -> Vec<u32> {
        if hm { unsafe { std::env::set_var("ZLLM_HEADMAJOR_KV", "1"); } } else { unsafe { std::env::remove_var("ZLLM_HEADMAJOR_KV"); } }
        let ctx = VkContext::new().expect("vk ctx");
        let model = VkModel::load(&path, ctx).expect("load");
        let argmax = |v: &[f32]| { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let mut next = argmax(&model.prefill_forward(&prompt));
        let mut toks = vec![next];
        let mut pos = prompt.len();
        for _ in 1..n_gen { next = model.forward_argmax(next, pos); toks.push(next); pos += 1; }
        toks
    };
    let pos_major = run(false);
    let head_major = run(true);
    let agree = pos_major.iter().zip(&head_major).take_while(|(a, b)| a == b).count();
    eprintln!("pos-major  gen: {pos_major:?}");
    eprintln!("head-major gen: {head_major:?}");
    eprintln!("prefill+decode agree on {agree}/{n_gen} tokens (head-major transpose path)");
    assert_eq!(pos_major, head_major, "head-major prefill path diverged from pos-major");
}

/// `cargo test --release --features vulkan --test headmajor_kv gen_headmajor_spv -- --ignored --nocapture`
#[test]
#[ignore]
fn gen_headmajor_spv() {
    for stem in ["kv_write_hm", "sdpa_flash_partial_hm", "kv_transpose_hm"] {
        let wgsl = std::fs::read_to_string(format!("src/backend/vulkan/shaders/{stem}.wgsl")).unwrap();
        let words = compile_wgsl(&wgsl);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let path = format!("src/backend/vulkan/shaders/{stem}.spv");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {path} ({} bytes, {} words) ✓", bytes.len(), words.len());
    }
}
