//! Q3_K decode matvec kernel — validated on the real Vulkan device against
//! candle. The kernel (WGSL, compiled to SPIR-V by naga in-process) reads a
//! packed Q3_K weight buffer (29 u32/block: d_f32 | 16 pre-shuffled scale bytes |
//! hmask[32] | qs[64]) and computes out[row] = dequant(W)[row,:]·x — one workgroup
//! per output row, like the Q4_K decode matvec. This is the gate+up→Q3 kernel
//! (validated recipe: +2.7% ppl, ~+11% decode on the deployed model).
//! `cargo test --release --features vulkan --test vk_q3k_matvec -- --ignored --nocapture`
#![cfg(feature = "vulkan")]

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use zllm::backend::vulkan::VkContext;

const BLK: usize = 110;

fn f16_to_f32(bits: u16) -> f32 {
    let (sign, exp, mant) = ((bits >> 15) & 1, (bits >> 10) & 0x1f, bits & 0x3ff);
    let f = if exp == 0 { (mant as f32) * 2f32.powi(-24) }
        else if exp == 0x1f { if mant == 0 { f32::INFINITY } else { f32::NAN } }
        else { (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15) };
    if sign == 1 { -f } else { f }
}

/// Repack candle Q3_K blocks (110 B) into the GPU's 29-u32/block layout.
fn pack_q3k(bytes: &[u8], n: usize, nb: usize) -> Vec<u32> {
    let (km1, km2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let mut out = vec![0u32; n * nb * 29];
    let u = |s: &[u8], i: usize| u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]);
    for row in 0..n {
        for b in 0..nb {
            let blk = &bytes[(row * nb + b) * BLK..][..BLK];
            let base = (row * nb + b) * 29;
            out[base] = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]])).to_bits();
            let (s0, s1, s2) = (u(blk, 96), u(blk, 100), u(blk, 104));
            let a = [
                (s0 & km2) | ((s2 & km1) << 4),
                (s1 & km2) | (((s2 >> 2) & km1) << 4),
                ((s0 >> 4) & km2) | (((s2 >> 4) & km1) << 4),
                ((s1 >> 4) & km2) | (((s2 >> 6) & km1) << 4),
            ];
            for w in 0..4 { out[base + 1 + w] = a[w]; }            // 16 scale bytes
            for w in 0..8 { out[base + 5 + w] = u(blk, w * 4); }   // hmask[32]
            for w in 0..16 { out[base + 13 + w] = u(blk, 32 + w * 4); } // qs[64]
        }
    }
    out
}

const WGSL: &str = include_str!("../src/backend/vulkan/shaders/decode_matvec_q3k.wgsl");

fn compile_wgsl(src: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(src).expect("wgsl parse");
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module).expect("wgsl validate");
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None).expect("spv write")
}

#[test]
#[ignore]
fn vk_q3k_matvec_vs_candle() {
    let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
    let dev = Device::Cpu;
    let (n, k) = (320usize, 2048usize); // a realistic FFN gate/up shape slice
    let nb = k / 256;
    let data: Vec<f32> = (0..n * k).map(|i| (((i * 2654435761u64 as usize % 1021) as f32) - 510.0) / 80.0).collect();
    let t = Tensor::from_vec(data, (n, k), &dev).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::Q3K).unwrap();
    let wq = qt.dequantize(&dev).unwrap().to_vec2::<f32>().unwrap(); // oracle weights [n][k]
    let bytes = qt.data().unwrap();

    let xs: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 5.0).collect();
    // CPU oracle: out[row] = W[row,:]·x
    let oracle: Vec<f32> = (0..n).map(|r| (0..k).map(|c| wq[r][c] * xs[c]).sum()).collect();

    let packed = pack_q3k(&bytes, n, nb);
    let wbytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    let spv = compile_wgsl(WGSL);
    let (gpu, ms) = ctx.decode_matvec_spv(&spv, &wbytes, n, nb, &xs, 1).expect("q3 matvec");

    let mut maxabs = 0f32;
    let mut maxrel = 0f32;
    for r in 0..n {
        let e = (gpu[r] - oracle[r]).abs();
        maxabs = maxabs.max(e);
        maxrel = maxrel.max(e / (oracle[r].abs() + 1e-3));
    }
    eprintln!("Q3_K matvec GPU vs candle: max abs {maxabs:.3e}, max rel {maxrel:.3e}, n={n} k={k}, {ms:.2}ms");
    assert!(maxrel < 1e-3, "Q3_K GPU matvec diverged from candle (max rel {maxrel})");
    eprintln!("Q3_K decode matvec kernel VALIDATED on GPU ✓");
}

/// Isolate Q3 vs Q4 STREAMING efficiency at the real w13 (gate+up) shape.
/// CAVEAT: this runs through `decode_matvec_spv`→`decode_matvec_q4k_inner`, which
/// forces REQUIRE_FULL_SUBGROUPS — under that flag the naga-compiled Q3 SPV is
/// ~2.3× slow (106 vs 246 GB/s), a MISLEADING ARTIFACT. The DEPLOYED path
/// (`make_pipeline_raw`, no subgroup flag) runs Q3 fast: in-forward VK_MVONLY
/// measures Q3 +3.9% vs Q4 (242 vs 233 tok/s). Trust the in-forward number.
/// `cargo test --release --features vulkan --test vk_q3k_matvec vk_q3_vs_q4_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn vk_q3_vs_q4_bench() {
    let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
    let dev = Device::Cpu;
    let (n, k) = (16384usize, 2048usize); // w13 = gate+up concat: [2*n_inter, n_embd]
    let nb = k / 256;
    let iters = 60u32;
    let data: Vec<f32> = (0..n * k).map(|i| (((i * 2654435761u64 as usize % 1031) as f32) - 515.0) / 80.0).collect();
    let t = Tensor::from_vec(data, (n, k), &dev).unwrap();
    let xs: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 5.0).collect();

    // Q4: candle Q4_K bytes (144 B/block) through the deployed decode_matvec_q4k.
    let q4 = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
    let q4b = q4.data().unwrap();
    let (_o4, ms4) = ctx.decode_matvec_q4k(&q4b, n, nb, &xs, iters).expect("q4");
    let gb4 = q4b.len() as f64 * iters as f64 / (ms4 / 1e3) / 1e9;

    // Q3: packed 29-u32/block through the new Q3 kernel.
    let q3 = QTensor::quantize(&t, GgmlDType::Q3K).unwrap();
    let packed = pack_q3k(&q3.data().unwrap(), n, nb);
    let q3b: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    let spv = compile_wgsl(WGSL);
    let (_o3, ms3) = ctx.decode_matvec_spv(&spv, &q3b, n, nb, &xs, iters).expect("q3");
    let gb3 = q3b.len() as f64 * iters as f64 / (ms3 / 1e3) / 1e9;

    let (mb4, mb3) = (q4b.len() as f64 / 1e6, q3b.len() as f64 / 1e6);
    eprintln!("w13 {n}x{k} matvec, {iters} iters:");
    eprintln!("  Q4: {mb4:.1} MB, {:.3} ms/dispatch, {gb4:.0} GB/s", ms4 / iters as f64);
    eprintln!("  Q3: {mb3:.1} MB, {:.3} ms/dispatch, {gb3:.0} GB/s", ms3 / iters as f64);
    eprintln!("  bytes ratio Q3/Q4 = {:.3}; per-dispatch time ratio Q3/Q4 = {:.3}; Q3 GB/s / Q4 GB/s = {:.3}",
        mb3 / mb4, ms3 / ms4, gb3 / gb4);
    eprintln!("  → if time ratio < 1: Q3 IS faster (win); if ~1: ALU offsets the fewer bytes (parity confirmed).");
}

/// Generate the committed SPIR-V from the WGSL (naga is dev-only; the VkModel
/// lib `include_bytes!`s the .spv). Run once after editing the .wgsl:
/// `cargo test --features vulkan --test vk_q3k_matvec gen_q3k_spv -- --ignored --nocapture`
#[test]
#[ignore]
fn gen_q3k_spv() {
    let words = compile_wgsl(WGSL);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let path = "src/backend/vulkan/shaders/decode_matvec_q3k.spv";
    std::fs::write(path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes, {} words) ✓", path, bytes.len(), words.len());
}
