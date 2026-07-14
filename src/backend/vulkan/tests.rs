//! Raw-Vulkan backend tests, extracted verbatim from `mod.rs`
//! (V1_PLAN split). Declared there as `#[cfg(test)] mod tests;`.
//! Content is byte-identical to the former inline module body.

    use super::*;

    /// Cross-dim generalization: the head-major shaders now read hd/kv_dim from
    /// the uniform, so one SPV serves any hd<=64 model. This validates them at dims
    /// DIFFERENT from the 1B (hd=32, n_kv=2, n_head=4 vs hd=64, n_kv=8, n_head=32):
    /// spot-checks kv_write_hm's layout at hd=32, then runs the partial on a
    /// head-major cache and compares (after a CPU log-sum-exp combine) to a CPU
    /// softmax-attention reference. `--lib vk_headmajor_crossdim -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_headmajor_crossdim() {
        use ash::vk;
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let (n_head, n_kv, hd, seq) = (4usize, 2usize, 32usize, 70usize);
        let max_seq = MAX_SEQ; // must equal MAX_SEQ in the shaders (engine cache cap)
        let kv_dim = n_kv * hd;  // 64 ≠ the 1B's 512
        let scale = 1.0f32 / (hd as f32).sqrt();
        let f = |a: usize, b: usize| (((a * 7 + b * 13) % 17) as f32) * 0.1 - 0.8; // deterministic synthetic
        let q: Vec<f32> = (0..n_head * hd).map(|i| f(i, 1)).collect();
        let ksrc: Vec<Vec<f32>> = (0..seq).map(|p| (0..kv_dim).map(|e| f(p, e + 2)).collect()).collect();
        let vsrc: Vec<Vec<f32>> = (0..seq).map(|p| (0..kv_dim).map(|e| f(p, e + 5)).collect()).collect();
        let nblk = seq.div_ceil(32);
        unsafe {
            let dev = &ctx.device;
            let hm = |kvh: usize, p: usize, d: usize| kvh * max_seq * hd + p * hd + d; // head-major index
            // Build the head-major K/V cache directly (the layout the partial reads).
            let (kc, _kcm, kcp) = ctx.uma_buffer((max_seq * kv_dim * 4) as u64).unwrap();
            let (vc, _vcm, vcp) = ctx.uma_buffer((max_seq * kv_dim * 4) as u64).unwrap();
            let kcs = std::slice::from_raw_parts_mut(kcp as *mut f32, max_seq * kv_dim);
            let vcs = std::slice::from_raw_parts_mut(vcp as *mut f32, max_seq * kv_dim);
            for p in 0..seq { for kvh in 0..n_kv { for d in 0..hd {
                kcs[hm(kvh, p, d)] = ksrc[p][kvh * hd + d];
                vcs[hm(kvh, p, d)] = vsrc[p][kvh * hd + d];
            }}}
            let submit = |cmd: vk::CommandBuffer| {
                let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
                dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
                dev.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
                dev.destroy_fence(fence, None);
            };
            // (1) Spot-check kv_write_hm at hd=32: write position 3's K via the shader, expect the head-major layout.
            let (kvw_p, kvw_l, kvw_sl, _) = ctx.make_pipeline_raw(KV_WRITE_HM_SPV, 2);
            let (chk, _chm, chp) = ctx.uma_buffer((max_seq * kv_dim * 4) as u64).unwrap();
            std::ptr::write_bytes(chp, 0, max_seq * kv_dim * 4);
            let (wsrc, _wsm, wsp) = ctx.uma_buffer((kv_dim * 4) as u64).unwrap();
            std::ptr::copy_nonoverlapping(ksrc[3].as_ptr() as *const u8, wsp, kv_dim * 4);
            let (wub, _wum, wup) = ctx.uma_buffer(16).unwrap();
            std::ptr::copy_nonoverlapping([kv_dim as u32, (3 * kv_dim) as u32, hd as u32, 0u32].as_ptr() as *const u8, wup, 16);
            let wset = vk_alloc_set(dev, {
                let p = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2),
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).unwrap(); p
            }, kvw_sl, &[chk, wsrc], wub);
            let cpool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
            let wcmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(wcmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(wcmd, vk::PipelineBindPoint::COMPUTE, kvw_p);
            dev.cmd_bind_descriptor_sets(wcmd, vk::PipelineBindPoint::COMPUTE, kvw_l, 0, &[wset], &[]);
            dev.cmd_dispatch(wcmd, (kv_dim as u32).div_ceil(64), 1, 1);
            dev.end_command_buffer(wcmd).unwrap(); submit(wcmd);
            let chks = std::slice::from_raw_parts(chp as *const f32, max_seq * kv_dim);
            let mut wmax = 0f32;
            for kvh in 0..n_kv { for d in 0..hd { wmax = wmax.max((chks[hm(kvh, 3, d)] - ksrc[3][kvh * hd + d]).abs()); } }
            eprintln!("kv_write_hm @hd=32 layout max err: {wmax:.2e}");
            assert!(wmax < 1e-6, "kv_write_hm wrong layout at hd=32");

            // (2) Run the head-major partial at hd=32 over the cache.
            let (fp_p, fp_l, fp_sl, _) = ctx.make_pipeline_raw(SDPA_FLASH_PARTIAL_HM_SPV, 4);
            let (qb, _qm, qp) = ctx.uma_buffer((n_head * hd * 4) as u64).unwrap();
            std::ptr::copy_nonoverlapping(q.as_ptr() as *const u8, qp, n_head * hd * 4);
            let (part, _pm, pp) = ctx.uma_buffer((n_head * nblk * (hd + 2) * 4) as u64).unwrap();
            let (sub, _sm, sup) = ctx.uma_buffer(16).unwrap();
            std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, seq as u32].as_ptr() as *const u8, sup, 16);
            let pool2 = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).unwrap();
            let fset = vk_alloc_set(dev, pool2, fp_sl, &[qb, kc, vc, part], sub);
            let fcmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(fcmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(fcmd, vk::PipelineBindPoint::COMPUTE, fp_p);
            dev.cmd_bind_descriptor_sets(fcmd, vk::PipelineBindPoint::COMPUTE, fp_l, 0, &[fset], &[]);
            dev.cmd_dispatch(fcmd, n_head as u32, nblk as u32, 1);
            dev.end_command_buffer(fcmd).unwrap(); submit(fcmd);
            let ps = std::slice::from_raw_parts(pp as *const f32, n_head * nblk * (hd + 2));

            // (3) CPU log-sum-exp combine of the partials, vs a CPU softmax-attention reference.
            let mut maxerr = 0f32;
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                // GPU: combine blocks
                let mut gm = f32::MIN;
                for b in 0..nblk { gm = gm.max(ps[(h * nblk + b) * (hd + 2) + hd]); }
                let mut denom = 0f32; let mut acc = vec![0f32; hd];
                for b in 0..nblk {
                    let base = (h * nblk + b) * (hd + 2);
                    let w = (ps[base + hd] - gm).exp();
                    denom += ps[base + hd + 1] * w;
                    for d in 0..hd { acc[d] += ps[base + d] * w; }
                }
                // CPU reference
                let mut sc = vec![0f32; seq];
                for p in 0..seq { let mut s = 0f32; for d in 0..hd { s += q[h * hd + d] * ksrc[p][kvh * hd + d]; } sc[p] = s * scale; }
                let rm = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut rl = 0f32; for p in 0..seq { sc[p] = (sc[p] - rm).exp(); rl += sc[p]; }
                for d in 0..hd {
                    let mut rv = 0f32; for p in 0..seq { rv += sc[p] * vsrc[p][kvh * hd + d]; }
                    let refv = rv / rl; let gpuv = acc[d] / denom;
                    maxerr = maxerr.max((refv - gpuv).abs());
                }
            }
            eprintln!("head-major partial @hd=32,n_kv=2,seq=70 vs CPU attention: max err {maxerr:.2e}");
            assert!(maxerr < 1e-4, "head-major partial wrong at non-1B dims");
            eprintln!("✓ head-major shaders generalize to hd=32/n_kv=2 (different from the 1B's hd=64/n_kv=8)");
        }
    }

    /// Phase 2.0 spike: bring up the ash device, confirm the coopmat shapes,
    /// run a 16x16x16 cooperative-matrix multiply on the iGPU, and validate it
    /// against a CPU matmul. Proves the whole raw-Vulkan + WMMA path works.
    #[test]
    fn vk_coopmat_matmul_16_matches_cpu() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        eprintln!("Vulkan adapter: {}", ctx.adapter_name);
        eprintln!("subgroup size: {}", ctx.subgroup_size);
        eprintln!("fp16 coopmat shapes (M,N,K): {:?}", ctx.coopmat_shapes);
        assert!(
            ctx.coopmat_shapes.contains(&(16, 16, 16)),
            "device does not report a 16x16x16 fp16 coopmat config"
        );

        // Deterministic small test matrices.
        let a: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let b: Vec<f32> = (0..256).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect();
        let gpu = ctx.coopmat_matmul_16(&a, &b).expect("coopmat matmul");

        // CPU reference: C[r,c] = sum_k A[r,k]*B[k,c], row-major 16x16.
        let mut max_abs = 0f32;
        for r in 0..16 {
            for c in 0..16 {
                let mut acc = 0f32;
                for k in 0..16 {
                    acc += a[r * 16 + k] * b[k * 16 + c];
                }
                max_abs = max_abs.max((gpu[r * 16 + c] - acc).abs());
            }
        }
        eprintln!("coopmat 16x16x16 vs CPU: max_abs_err = {max_abs:.5}");
        // fp16 inputs → some rounding; fp32 accumulate keeps it small.
        assert!(max_abs < 0.05, "coopmat matmul error too high: {max_abs}");
    }

    /// PERFORMANCE COMPARISON: the LDS-blocked coopmat GEMM (the Phase 2.1
    /// compute core) — validate vs CPU, then measure throughput (GFLOP/s) at a
    /// prefill-scale shape, against the current wgpu f32 path and the iGPU peak.
    /// `cargo test --release --features vulkan --lib vk_coopmat_gemm -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_coopmat_gemm_throughput() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        // Correctness at a small shape (fp16 in, fp32 accumulate).
        let (m, n, k) = (128usize, 128usize, 128usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let (c, _) = ctx.coopmat_gemm_f16(m, n, k, &a, &b, 1).expect("gemm");
        let mut max_abs = 0f32;
        for r in 0..8 { for cc in 0..8 {
            let mut acc = 0f32; for kk in 0..k { acc += a[r * k + kk] * b[kk * n + cc]; }
            max_abs = max_abs.max((c[r * n + cc] - acc).abs());
        } }
        eprintln!("coopmat GEMM {m}x{n}x{k} vs CPU: max_abs_err = {max_abs:.4}");
        assert!(max_abs < 0.5, "coopmat GEMM wrong: {max_abs}");

        // Throughput at a prefill-scale shape (M prompt rows, N=K=2048).
        for &(m, n, k) in &[(512usize, 2048usize, 2048usize), (256, 2048, 2048), (2048, 2048, 2048)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 29) as f32 - 14.0) * 0.01).collect();
            let _ = ctx.coopmat_gemm_f16(m, n, k, &a, &b, 2).expect("warm"); // warm
            let iters = 30u32;
            let (_, ms) = ctx.coopmat_gemm_f16(m, n, k, &a, &b, iters).expect("bench");
            let per = ms / iters as f64;
            let gflops = 2.0 * (m * n * k) as f64 / (per / 1e3) / 1e9;
            eprintln!("coopmat GEMM M={m:>4} N={n} K={k}: {per:6.3} ms/iter, {gflops:6.0} GFLOP/s", );
        }
        eprintln!("  reference: wgpu f32 Q4_K GEMM ~1000 GFLOP/s (M=256, 2.09ms); 8060S fp16 coopmat peak ~59000 GFLOP/s");
    }

    /// PHASE 2.1: the real Q4_K coopmat prefill GEMM (dequant folded into the
    /// LDS staging). Validate vs the candle Q4_K dequant oracle, then measure
    /// throughput — the number that directly replaces the wgpu f32 Q4_K GEMM.
    /// `cargo test --release --features vulkan --lib vk_coopmat_q4k -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_coopmat_q4k_gemm_throughput() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        // Correctness: small shape vs candle dequant oracle (f16 path → cosine).
        let (n, nb, m) = (128usize, 2usize, 128usize);
        let k = nb * 256;
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let qt = QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap();
        let bytes = qt.data().unwrap();
        let deq: Vec<f32> = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
        let (gpu, _) = ctx.coopmat_q4k_gemm(&bytes, n, nb, &x, m, 1).expect("q4k gemm");
        let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
        for mm in 0..8 { for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k { acc += (x[mm * k + kk] as f64) * (deq[nn * k + kk] as f64); }
            let g = gpu[mm * n + nn] as f64;
            dot += g * acc; ng += g * g; nc += acc * acc;
        } }
        let cos = dot / (ng.sqrt() * nc.sqrt());
        eprintln!("coopmat Q4_K GEMM vs candle oracle: cosine = {cos:.5}");
        assert!(cos > 0.99, "coopmat Q4_K GEMM wrong: cosine {cos}");

        // Throughput at prefill scale (N=K=2048), the wgpu-GEMM replacement.
        let (n, nb) = (2048usize, 8usize);
        let k = nb * 256;
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let bytes = QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec();
        for &m in &[128usize, 256, 512, 2048] {
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
            let _ = ctx.coopmat_q4k_gemm(&bytes, n, nb, &x, m, 2).expect("warm");
            let iters = 30u32;
            let (_, ms) = ctx.coopmat_q4k_gemm(&bytes, n, nb, &x, m, iters).expect("bench");
            let per = ms / iters as f64;
            let gflops = 2.0 * (m * n * k) as f64 / (per / 1e3) / 1e9;
            eprintln!("coopmat Q4_K GEMM M={m:>4} N={n} K={k}: {per:6.3} ms/iter, {gflops:6.0} GFLOP/s");
        }
        eprintln!("  vs wgpu f32 Q4_K GEMM: M=256 2.09ms (~1030 GFLOP/s), M=512 3.74ms (~1150)");
    }

    /// BM=16 small-M coopmat Q4_K GEMM (spec-decode verify): correctness vs candle +
    /// bandwidth at M=8. The weight-stationary matvec hit ~27 GB/s (x-reread); this
    /// tiles so x is read once per 16x128 tile — should approach the ~200 GB/s wall.
    /// `cargo test --release --features vulkan --lib vk_coopmat_m16 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_coopmat_q4k_gemm_m16() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let (n, nb, m) = (2048usize, 8usize, 8usize); // a wq-shaped layer matvec, verify window M=8
        let k = nb * 256;
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let qt = QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap();
        let bytes = qt.data().unwrap();
        let deq: Vec<f32> = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
        let (gpu, _) = ctx.coopmat_q4k_gemm_m16(&bytes, n, nb, &x, m, 1).expect("m16 gemm");
        let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
        for mm in 0..m { for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k { acc += (x[mm * k + kk] as f64) * (deq[nn * k + kk] as f64); }
            let g = gpu[mm * n + nn] as f64;
            dot += g * acc; ng += g * g; nc += acc * acc;
        } }
        let cos = dot / (ng.sqrt() * nc.sqrt());
        eprintln!("BM=16 coopmat Q4_K GEMM (M={m}) vs candle: cosine = {cos:.5}");
        assert!(cos > 0.99, "BM=16 GEMM wrong: cosine {cos}");
        // Bandwidth sweep over N (utilization scales with N/128 workgroups).
        for &nn in &[2048usize, 16384, 65536] {
            let mut w = vec![0f32; nn * k];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
            let b = QTensor::quantize(&Tensor::from_vec(w, (nn, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec();
            let _ = ctx.coopmat_q4k_gemm_m16(&b, nn, nb, &x, m, 4).expect("warm");
            let iters = 100u32;
            let (_, ms) = ctx.coopmat_q4k_gemm_m16(&b, nn, nb, &x, m, iters).expect("bench");
            let per = ms / iters as f64;
            let gbs = (nn * nb * 144) as f64 / (per / 1e3) / 1e9;
            eprintln!("  N={nn:>6} ({} workgroups): {per:.3} ms/iter, {gbs:.1} GB/s", nn / 128);
        }
        eprintln!("  (weight-stationary matvec was ~27 GB/s; M=1 decode kernel ~208)");
    }

    /// END-TO-END PROJECTION: measure the coopmat Q4_K GEMM time for a full
    /// Llama-3.2-1B forward's worth of GEMMs (the dominant prefill cost) and
    /// project prefill tok/s — the number that replaces the current ~492.
    /// (GEMM-only; norm/rope/sdpa would add some overhead, so this is an upper
    /// bound on a coopmat prefill.)
    /// `cargo test --release --features vulkan --lib vk_coopmat_prefill -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_coopmat_prefill_projection() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        // Llama-3.2-1B per-layer GEMM shapes: (N_out, K_in, count/layer).
        let shapes: [(usize, usize, usize); 4] = [
            (2048, 2048, 2), // wq, wo
            (512, 2048, 2),  // wk, wv
            (8192, 2048, 2), // w1 (gate), w3 (up)
            (2048, 8192, 1), // w2 (down)
        ];
        let n_layers = 16usize;
        let mk_weight = |n: usize, k: usize| -> Vec<u8> {
            let mut w = vec![0f32; n * k];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
            QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec()
        };
        for &m in &[256usize, 512] {
            let mut forward_ms = 0f64;
            for &(n, k, count) in &shapes {
                let nb = k / 256;
                let bytes = mk_weight(n, k);
                let x: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
                let _ = ctx.coopmat_q4k_gemm(&bytes, n, nb, &x, m, 2).expect("warm");
                let iters = 30u32;
                let (_, ms) = ctx.coopmat_q4k_gemm(&bytes, n, nb, &x, m, iters).expect("bench");
                forward_ms += (ms / iters as f64) * count as f64;
            }
            forward_ms *= n_layers as f64;
            let tok_s = m as f64 / (forward_ms / 1e3);
            eprintln!("coopmat prefill GEMM projection: M={m:>4} -> {forward_ms:6.1} ms/forward (GEMMs only) => {tok_s:5.0} tok/s");
        }
        eprintln!("  vs current wgpu prefill ~492 tok/s (incl norm/rope/sdpa); llama.cpp iGPU 5747");
    }

    /// DECODE matvec: validate vs the candle Q4_K oracle, then measure the
    /// streaming bandwidth (GB/s) — the metric that bounds decode tok/s. The
    /// achievable bus is ~215 GB/s; llama.cpp gets ~153 effective (201 tok/s),
    /// so >153 here projects a decode lead.
    /// `cargo test --release --features vulkan --lib vk_decode_matvec -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_decode_matvec_bandwidth() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        // Correctness vs candle dequant oracle (f32 → tight).
        let (n, nb) = (256usize, 2usize);
        let k = nb * 256;
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let qt = QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap();
        let bytes = qt.data().unwrap();
        let deq: Vec<f32> = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        let (gpu, _) = ctx.decode_matvec_q4k(&bytes, n, nb, &x, 1).expect("matvec");
        let mut max_abs = 0f32;
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k { acc += (deq[nn * k + kk] as f64) * (x[kk] as f64); }
            max_abs = max_abs.max((gpu[nn] - acc as f32).abs());
        }
        eprintln!("decode matvec vs candle oracle: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "decode matvec wrong: {max_abs}");

        // Streaming bandwidth on representative decode weights.
        for &(n, k) in &[(8192usize, 2048usize), (128256, 2048)] {
            let nb = k / 256;
            let mut w = vec![0f32; n * k];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
            let bytes = QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec();
            let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
            let _ = ctx.decode_matvec_q4k(&bytes, n, nb, &x, 4).expect("warm");
            let iters = 50u32;
            let (_, ms) = ctx.decode_matvec_q4k(&bytes, n, nb, &x, iters).expect("bench");
            let gbps = bytes.len() as f64 * iters as f64 / (ms / 1e3) / 1e9;
            eprintln!("decode matvec N={n:>6} K={k}: {:.3} ms/iter, {gbps:5.0} GB/s ({:.1} MB weight)",
                ms / iters as f64, bytes.len() as f64 / 1e6);
        }
        eprintln!("  llama.cpp decode ~153 GB/s effective (201 tok/s); achievable bus ~215 GB/s (~280 tok/s wall)");
    }

    /// DECODE projection: sum the matvec time for a full Llama-3.2-1B forward's
    /// worth of weights (the bandwidth-bound decode cost) and project tok/s —
    /// the number that would beat llama's 201. Matvec-only (norm/rope/sdpa are
    /// tiny at M=1); a fused minimal-barrier forward approaches it.
    /// `cargo test --release --features vulkan --lib vk_decode_projection -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_decode_projection() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        let mk = |n: usize, k: usize| -> Vec<u8> {
            let mut w = vec![0f32; n * k];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
            QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec()
        };
        let time = |bytes: &[u8], n: usize, nb: usize| -> f64 {
            let x: Vec<f32> = (0..nb * 256).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
            let _ = ctx.decode_matvec_q4k(bytes, n, nb, &x, 8).expect("warm");
            let iters = 60u32;
            let (_, ms) = ctx.decode_matvec_q4k(bytes, n, nb, &x, iters).expect("bench");
            ms / iters as f64
        };
        // Per-layer matvec shapes (N_out, K_in, count) + LM head.
        let shapes: [(usize, usize, usize); 4] = [
            (2048, 2048, 2), (512, 2048, 2), (8192, 2048, 2), (2048, 8192, 1),
        ];
        let mut forward_ms = 0f64;
        for &(n, k, count) in &shapes {
            forward_ms += time(&mk(n, k), n, k / 256) * count as f64;
        }
        forward_ms *= 16.0; // layers
        forward_ms += time(&mk(128256, 2048), 128256, 8); // LM head
        let tok_s = 1000.0 / forward_ms;
        eprintln!("decode matvec projection: {forward_ms:.3} ms/token (matvecs only) => {tok_s:.0} tok/s");
        eprintln!("  vs current wgpu decode ~80 tok/s; llama.cpp iGPU 201; bandwidth wall ~280");
    }

    /// FUSED DECODE FORWARD: the whole token forward (16 layers of
    /// rmsnorm→QKV→RoPE→SDPA→O→rmsnorm→gate/up→silu→down, then final norm + LM
    /// head) recorded in ONE command buffer with minimal per-dependency
    /// barriers (the hand-managed barriers wgpu can't do). Dummy resident
    /// weights — this measures whether the fused forward holds the ~237 tok/s
    /// matvec throughput once the small ops + real barriers are included.
    /// `cargo test --release --features vulkan --lib vk_fused_decode -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_fused_decode_throughput() {
        
        
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        unsafe { fused_decode_inner(&ctx); }
    }

    /// END-TO-END: load the real Llama-3.2-1B GGUF into the VkModel and check
    /// verify_forward (batched-decode forward for spec-decode verification) must
    /// produce the same per-position argmax as feeding the tokens one at a time.
    /// `cargo test --release --features vulkan --lib vk_verify_forward -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_verify_forward() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        let prompt = [128000u32, 791, 6864, 315, 9822, 374];
        // Fill the resident KV cache 0..P-1, ending with the token to feed at P.
        let mut next = 0u32;
        for (i, &t) in prompt.iter().enumerate() { next = model.forward_argmax(t, i); }
        let p = prompt.len();
        // Sequential reference: feed [next, 500, 600] at P, P+1, P+2 one at a time.
        let s = [next, 500u32, 600u32];
        let seq: Vec<u32> = s.iter().enumerate().map(|(i, &t)| model.forward_argmax(t, p + i)).collect();
        // Batched verify over the same tokens at the same positions (overwrites P..P+2).
        let ver = model.verify_forward(&s, p);
        assert_eq!(ver, seq, "verify_forward per-position argmax diverged from sequential");
        eprintln!("verify_forward: {} positions, per-position argmax == sequential {seq:?} ✓", s.len());
    }

    /// VkModel coopmat-path Prompt-Lookup speculative decode: bit-identical to
    /// greedy single-token decode, but >1 token/forward on echo-heavy text.
    /// `cargo test --release --features vulkan --lib vk_pld -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_pld() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        let (k, eos) = (48usize, u32::MAX);
        let mut prompt = vec![128000u32];
        for _ in 0..10 { prompt.extend_from_slice(&[264, 265, 266, 267, 268]); }

        // Greedy single-token reference (timed).
        let tg = Instant::now();
        let mut next = 0u32;
        for (i, &t) in prompt.iter().enumerate() { next = model.forward_argmax(t, i); }
        let (mut greedy, mut pos) = (vec![next], prompt.len());
        while greedy.len() < k && next != eos {
            let t = model.forward_argmax(next, pos); greedy.push(t); next = t; pos += 1;
        }
        let g_tps = greedy.len() as f64 / tg.elapsed().as_secs_f64();

        // PLD (timed) — fresh prefill overwrites the cache.
        let tp = Instant::now();
        let (produced, forwards) = model.generate_pld(&prompt, k, eos, 3, 7);
        let p_tps = produced.len() as f64 / tp.elapsed().as_secs_f64();

        assert_eq!(produced, greedy, "VkModel PLD diverged from greedy");
        eprintln!("VkModel PLD: {} tokens in {forwards} forwards = {:.2} tok/forward, bit-identical ✓", produced.len(), produced.len() as f64 / forwards.max(1) as f64);
        eprintln!("  wall-clock: greedy {g_tps:.0} -> PLD {p_tps:.0} tok/s = {:.2}x (coopmat path)", p_tps / g_tps);

        // Cost probe — coopmat-PLD status. Small-M weight-stationary matvecs replaced
        // the 128-row coopmat tile (verify_inner now runs at real_m, no padding), which
        // is why PLD rose 0.58x -> ~0.8x. But the M-scaling below shows the remaining
        // wall is a ~38ms M-INDEPENDENT fixed cost: verify(1) ~= 6x a greedy token even
        // though greedy runs the SAME model in ~6.6ms. The gap is structural — greedy is
        // record-once + fused megakernels, verify is ~110 unfused dispatches re-recorded
        // every call. PLD wins only when verify_cost < (accepted+1)*greedy_cost (~4
        // tokens); the fixed ~38ms alone exceeds that, so the next lever is fusing the
        // verify forward + record-once (match the decode path), not more matvec tuning.
        let vinp = [next, 264, 265, 266, 267, 268, 264, 265];
        let n = 20;
        let tg1 = Instant::now(); for i in 0..n { let _ = model.forward_argmax(next, prompt.len() + i % 4); } let g_ms = tg1.elapsed().as_secs_f64() * 1e3 / n as f64;
        for mm in [1usize, 2, 4, 8] {
            let _ = model.verify_forward(&vinp[..mm], prompt.len()); // warm up (builds for this M)
            let tv = Instant::now(); for _ in 0..n { let _ = model.verify_forward(&vinp[..mm], prompt.len()); } let v_ms = tv.elapsed().as_secs_f64() * 1e3 / n as f64;
            eprintln!("  probe: verify({mm})={v_ms:.2}ms = {:.1} greedy tokens (greedy_tok={g_ms:.2}ms; win at M needs < accepted+1)", v_ms / g_ms);
        }
        // Decompose where verify's time goes vs a 6ms greedy decode of the same M=1
        // work: matvec-only floor, +sdpa/kvcopy, +norms/residual = full. Localizes
        // the recoverable overhead (stale SDPA + transfer-copy K/V vs the deployed
        // fused kvwrite + barrier-lean SDPA in forward_inner).
        let timed = |m: usize| { let _ = model.verify_forward(&vinp[..m], prompt.len());
            let t = Instant::now(); for _ in 0..n { let _ = model.verify_forward(&vinp[..m], prompt.len()); } t.elapsed().as_secs_f64() * 1e3 / n as f64 };
        for (label, var) in [("matvec-only", "VK_V_MVONLY"), ("no-sdpa+kvcopy", "VK_V_NOSDPA"), ("full", "")] {
            if !var.is_empty() { unsafe { std::env::set_var(var, "1"); } }
            let (v1, v4) = (timed(1), timed(4));
            if !var.is_empty() { unsafe { std::env::remove_var(var); } }
            eprintln!("  decomp[{label:>14}]: verify(1)={v1:.2}ms  verify(4)={v4:.2}ms");
        }
    }

    /// TREE DRAFTING acceptance prototype: generate real greedy text with VkModel,
    /// then SIMULATE the spec-decode loop with linear n-gram drafts vs a draft TREE
    /// at MATCHED verify-row budgets, measuring tokens/forward (acceptance). No GPU
    /// tree-mask kernel yet — this validates whether the acceptance gain is worth
    /// building one. Higher tok/forward shifts the spec-decode win threshold above
    /// the verify matvec floor (~28ms @ M=4 → break-even at 4.75 tok/forward).
    /// `cargo test --release --features vulkan --lib vk_tree_drafting -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_tree_drafting() {
        use crate::engine::spec_decode::{lookup_draft_best, lookup_tree, tree_accept};
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        let max_len = 3; // n-gram lookup length
        // Simulate the greedy spec loop: at each forward, commit accepted+1 tokens.
        let sim_lin = |full: &[u32], start: usize, k: usize| -> (usize, usize) {
            let (mut p, mut fwd) = (start, 0usize);
            while p < full.len() {
                let d = lookup_draft_best(&full[..p], &full[..p], max_len, k).unwrap_or_default();
                let acc = d.iter().zip(&full[p..]).take_while(|(a, b)| a == b).count();
                p += (acc + 1).min(full.len() - p); fwd += 1;
            }
            (full.len() - start, fwd)
        };
        let sim_tree = |full: &[u32], start: usize, branch: usize, nodes: usize| -> (usize, usize) {
            let (mut p, mut fwd) = (start, 0usize);
            while p < full.len() {
                let t = lookup_tree(&full[..p], &full[..p], max_len, branch, nodes);
                let acc = tree_accept(&t, &full[p..]);
                p += (acc + 1).min(full.len() - p); fwd += 1;
            }
            (full.len() - start, fwd)
        };
        // Iterated LINEAR lookup: re-look-up after each drafted token to extend the
        // single spine (isolates iteration from the tree's branching).
        let sim_lin_iter = |full: &[u32], start: usize, k: usize| -> (usize, usize) {
            let (mut p, mut fwd) = (start, 0usize);
            while p < full.len() {
                let mut ctx: Vec<u32> = full[..p].to_vec();
                let mut draft: Vec<u32> = Vec::new();
                for _ in 0..k {
                    match lookup_draft_best(&full[..p], &ctx, max_len, 1) {
                        Some(d) if !d.is_empty() => { ctx.push(d[0]); draft.push(d[0]); }
                        _ => break,
                    }
                }
                let acc = draft.iter().zip(&full[p..]).take_while(|(a, b)| a == b).count();
                p += (acc + 1).min(full.len() - p); fwd += 1;
            }
            (full.len() - start, fwd)
        };
        // Generate greedy continuations for a few prompts (real model output).
        let prompts: Vec<(&str, Vec<u32>)> = vec![
            ("general", vec![128000, 791, 6864, 315, 9822, 374]),       // "The capital of France is"
            ("list",    vec![128000, 7896, 220, 16, 311, 220, 605, 25]), // "List 1 to 10:"
            ("echo",    { let mut p = vec![128000u32]; for _ in 0..8 { p.extend_from_slice(&[791, 4062, 14198, 39935]); } p }), // repetitive
        ];
        let n_gen = 240usize;
        for (label, prompt) in &prompts {
            let mut full = prompt.clone();
            let mut next = 0u32;
            for (i, &t) in prompt.iter().enumerate() { next = model.forward_argmax(t, i); }
            full.push(next); let mut pos = prompt.len();
            while full.len() < prompt.len() + n_gen { next = model.forward_argmax(next, pos); full.push(next); pos += 1; }
            eprintln!("[{label}] {} generated tokens (matched verify-row budget B: linear k=B-1 vs tree B nodes, branch 2):", n_gen);
            for &b in &[4usize, 8, 16] {
                let (lt, lf) = sim_lin(&full, prompt.len(), b - 1);
                let (it, if_) = sim_lin_iter(&full, prompt.len(), b - 1);
                let (tt, tf) = sim_tree(&full, prompt.len(), 2, b);
                let (ltpf, itpf, ttpf) = (lt as f64 / lf as f64, it as f64 / if_ as f64, tt as f64 / tf as f64);
                eprintln!("   B={b:>2}: linear {ltpf:.2}  |  iter-linear {itpf:.2}  |  tree {ttpf:.2} tok/fwd  (tree vs linear {:+.0}%, tree vs iter {:+.0}%)",
                    (ttpf / ltpf - 1.0) * 100.0, (ttpf / itpf - 1.0) * 100.0);
            }
        }
    }

    /// greedy decode matches candle CPU token-for-token (the engine the server
    /// will use). Also prints decode tok/s on real weights.
    /// `cargo test --release --features vulkan --lib vk_model_vs_candle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_model_vs_candle() {
        use std::time::Instant;
        let path = std::env::var("ZLLM_MODEL").unwrap_or_else(|_| "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string());
        let path = path.as_str();
        if !std::path::Path::new(path).exists() { eprintln!("model not found at {path}; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; } };
        let t = Instant::now();
        let model = VkModel::load(path, ctx).expect("load");
        eprintln!("VkModel loaded in {:.2}s (vocab {})", t.elapsed().as_secs_f64(), model.vocab);
        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let prompt: Vec<u32> = vec![128000]; // BOS
        let n_gen: usize = std::env::var("ZLLM_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
        let _ = &argmax;
        let n_time: usize = std::env::var("ZLLM_NTIME").ok().and_then(|s| s.parse().ok()).unwrap_or(128); // match llama-bench tg128
        // ZLLM_REPS>1 re-runs the 128-tok decode back-to-back (ctx reset to ~0 each rep) so the
        // iGPU heats up like `llama-bench -r N` — fair sustained-throughput comparison, not a burst.
        let reps: usize = std::env::var("ZLLM_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut next = 0u32;
        for (i, &tk) in prompt.iter().enumerate() { next = model.forward_argmax(tk, i); }
        let first_next = next;
        let mut vk_gen = vec![next];
        // ZLLM_CTX=D: isolate the decode rate AT depth D (no avg-ctx confound). Warm the KV
        // to D untimed, then time REPS windows of `window` steps starting at pos D (each window
        // re-attends KV[0..D], overwriting D..D+window). Shows how SDPA scales with context.
        if let Ok(cs) = std::env::var("ZLLM_CTX") {
            let depth: usize = cs.parse().unwrap_or(512);
            let window = 32usize;
            let mut n = first_next; let mut pos = prompt.len();
            while pos < depth { n = model.forward_argmax(n, pos); pos += 1; }
            let nseed = n;
            let rd: usize = std::env::var("ZLLM_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
            let t = Instant::now();
            for _ in 0..rd { let mut nn = nseed; let mut p2 = depth; for _ in 0..window { nn = model.forward_argmax(nn, p2); p2 += 1; } }
            let dt = t.elapsed();
            eprintln!("VkModel decode @ctx~{}: {:.1} tok/s ({} reps x {} steps)", depth, (rd * window) as f64 / dt.as_secs_f64(), rd, window);
            return;
        }
        let t0 = Instant::now();
        let mut total = 0usize;
        for rep in 0..reps {
            next = first_next; let mut pos = prompt.len();
            for _ in 1..n_time { next = model.forward_argmax(next, pos); if rep == 0 { vk_gen.push(next); } pos += 1; total += 1; }
        }
        let dt = t0.elapsed();
        eprintln!("VkModel decode: {:.1} tok/s over {} tokens ({} reps, avg ctx ~{})", total as f64 / dt.as_secs_f64(), total, reps, n_time / 2);
        eprintln!("VkModel gen: {vk_gen:?}");

        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::Backend;
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path)).expect("candle load");
        let mut clog = cb.forward_logits(&prompt).unwrap();
        let mut cnext = argmax(&clog);
        let mut cand_gen = vec![cnext];
        for _ in 1..n_gen { clog = cb.forward_logits(&[cnext]).unwrap(); cnext = argmax(&clog); cand_gen.push(cnext); }
        eprintln!("candle  gen: {cand_gen:?}");
        let agree = vk_gen.iter().zip(&cand_gen).take_while(|(a, b)| a == b).count();
        eprintln!("VkModel/candle agree on first {agree}/{n_gen} tokens");
        assert!(agree >= 8, "VkModel diverges from candle too early ({agree}); kernel/wiring bug");
    }

    /// BATCHED THROUGHPUT SCALING: time the batched forward (verify_forward, M rows in ONE
    /// forward via batched_matvec_q4k) at M=1..8 to see how aggregate tok/s scales with the
    /// batch — does weight-amortization push the BW-bound single-stream into compute-bound
    /// throughput that beats llama-server's batched decode?
    /// `ZLLM_CTX=64 cargo test --release --features vulkan --lib vk_batch_scaling -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_batch_scaling() {
        use std::time::Instant;
        let path = std::env::var("ZLLM_MODEL").unwrap_or_else(|_| "C:/models/llama-3.2-1b/Llama-3.2-1B-Q4pure.gguf".to_string());
        if !std::path::Path::new(&path).exists() { eprintln!("model not found at {path}; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let model = VkModel::load(&path, ctx).expect("load");
        let depth: usize = std::env::var("ZLLM_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let mut next = 128000u32;
        for i in 0..depth { next = model.forward_argmax(if i == 0 { 128000 } else { next }, i); }
        eprintln!("=== batched VkModel decode scaling @ctx~{} (vs llama: 1=229 8=1008) ===", depth);
        for m in [1usize, 2, 4, 8] {
            let toks: Vec<u32> = (0..m).map(|i| (100 + i) as u32).collect();
            let reps = 60usize;
            for _ in 0..3 { let _ = model.verify_forward(&toks, depth); } // warm
            let t0 = Instant::now();
            for _ in 0..reps { let _ = model.verify_forward(&toks, depth); }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("  M={}: {:.0} tok/s aggregate  ({:.2} ms/forward, {:.0} tok/s/stream)",
                m, (m * reps) as f64 / dt, dt * 1e3 / reps as f64, reps as f64 / dt);
        }
    }

    /// SLOT KV-WRITE scatter (step 3a): each stream writes its roped K/V into its own
    /// slot at its own position. Validated by reading the cache back.
    /// `cargo test --release --features vulkan --lib vk_kvwrite_slot -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_kvwrite_slot_correctness() {
        use ash::vk;
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let dev = &ctx.device;
        let (kv_dim, max_seq, m_rows, n_slots) = (512usize, 48usize, 4usize, 4usize);
        let src: Vec<f32> = (0..m_rows * kv_dim).map(|i| (i as f32) * 0.001 + 1.0).collect();
        let sp: Vec<u32> = vec![0, 5, 1, 12, 2, 20, 3, 30];
        unsafe {
            let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
            let (dst, dm, dp) = ctx.uma_buffer((n_slots * max_seq * kv_dim * 4) as u64).unwrap(); std::ptr::write_bytes(dp, 0, n_slots * max_seq * kv_dim * 4); bufs.push((dst, dm));
            let (sb, sm, srp) = ctx.uma_buffer((src.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, srp, src.len() * 4); bufs.push((sb, sm));
            let (spb, spm, spp) = ctx.uma_buffer((sp.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(sp.as_ptr() as *const u8, spp, sp.len() * 4); bufs.push((spb, spm));
            let (ub, um, up) = ctx.uma_buffer(16).unwrap(); std::ptr::copy_nonoverlapping([kv_dim as u32, max_seq as u32, m_rows as u32, 0u32].as_ptr() as *const u8, up, 16); bufs.push((ub, um));
            let (p, l, sl, module) = ctx.make_pipeline_raw(KVWRITE_SLOT_SPV, 3);
            let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).unwrap();
            let set = vk_alloc_set(dev, pool, sl, &[dst, sb, spb], ub);
            let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
            let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
            dev.cmd_dispatch(cmd, ((m_rows * kv_dim) as u32).div_ceil(64), 1, 1);
            dev.end_command_buffer(cmd).unwrap();
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            let cache = std::slice::from_raw_parts(dp as *const f32, n_slots * max_seq * kv_dim);
            let mut ok = true;
            for r in 0..m_rows {
                let (slot, pos) = (sp[r * 2] as usize, sp[r * 2 + 1] as usize);
                for i in 0..kv_dim { if (cache[(slot * max_seq + pos) * kv_dim + i] - src[r * kv_dim + i]).abs() > 1e-9 { ok = false; } }
            }
            eprintln!("kvwrite_slot: {} streams scattered to their (slot,pos) {}", m_rows, if ok { "✓ exact" } else { "WRONG" });
            assert!(ok, "kvwrite_slot scatter wrong");
            dev.destroy_fence(fence, None); dev.destroy_command_pool(cmd_pool, None); dev.destroy_descriptor_pool(pool, None);
            dev.destroy_pipeline(p, None); dev.destroy_pipeline_layout(l, None); dev.destroy_descriptor_set_layout(sl, None); dev.destroy_shader_module(module, None);
            for (b, mm) in bufs { dev.unmap_memory(mm); dev.destroy_buffer(b, None); dev.free_memory(mm, None); }
        }
    }

    /// SLOT-INDIRECTED DECODE SDPA (step 2 of the coopmat batched decoder): each of
    /// the m query rows attends its OWN KV slot at its OWN position. Validated against
    /// a CPU softmax-attention reference.
    /// `cargo test --release --features vulkan --lib vk_bsdpa_slot -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_bsdpa_slot_correctness() {
        use ash::vk;
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let dev = &ctx.device;
        let (n_head, n_kv, hd) = (32usize, 8usize, 64usize);
        let (m_rows, n_slots, max_seq) = (4usize, 4usize, 48usize);
        let attn_dim = n_head * hd; // 2048
        let kv_dim = n_kv * hd;     // 512
        let q: Vec<f32> = (0..m_rows * attn_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let kc: Vec<f32> = (0..n_slots * max_seq * kv_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
        let vc: Vec<f32> = (0..n_slots * max_seq * kv_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.03).collect();
        let sp: Vec<u32> = vec![0, 5, 1, 12, 2, 20, 3, 30]; // (slot, pos) per row
        // CPU reference.
        let mut cpu = vec![0f32; m_rows * attn_dim];
        let scale = 1.0 / (hd as f32).sqrt();
        for m in 0..m_rows {
            let (slot, pos) = (sp[m * 2] as usize, sp[m * 2 + 1] as usize);
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                let qb = m * attn_dim + h * hd;
                let mut sc = vec![0f32; pos + 1];
                for t in 0..=pos { let kb = (slot * max_seq + t) * kv_dim + kvh * hd;
                    let mut s = 0.0; for d in 0..hd { s += q[qb + d] * kc[kb + d]; } sc[t] = s * scale; }
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for d in 0..hd { let mut a = 0.0; for t in 0..=pos { let kb = (slot * max_seq + t) * kv_dim + kvh * hd; a += sc[t] * vc[kb + d]; } cpu[qb + d] = a / sum; }
            }
        }
        // GPU.
        unsafe {
            let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
            let upf = |bufs: &mut Vec<_>, d: &[f32]| { let (b, mm, p) = ctx.uma_buffer((d.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, d.len() * 4); bufs.push((b, mm)); b };
            let qb_ = upf(&mut bufs, &q); let kcb = upf(&mut bufs, &kc); let vcb = upf(&mut bufs, &vc);
            let (ob, om, op) = ctx.uma_buffer((m_rows * attn_dim * 4) as u64).unwrap(); bufs.push((ob, om));
            let (spb, sm, spp) = ctx.uma_buffer((sp.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(sp.as_ptr() as *const u8, spp, sp.len() * 4); bufs.push((spb, sm));
            let (ub, um, up) = ctx.uma_buffer(20).unwrap(); std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, m_rows as u32, max_seq as u32].as_ptr() as *const u8, up, 20); bufs.push((ub, um));
            let (p, l, sl, module) = ctx.make_pipeline_raw(BSDPA_SLOT_SPV, 5);
            let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(5),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).unwrap();
            let set = vk_alloc_set(dev, pool, sl, &[qb_, kcb, vcb, ob, spb], ub);
            let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
            let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
            dev.cmd_dispatch(cmd, ((m_rows * n_head) as u32).div_ceil(64), 1, 1);
            dev.end_command_buffer(cmd).unwrap();
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            let gpu = std::slice::from_raw_parts(op as *const f32, m_rows * attn_dim);
            let max_err = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("bsdpa_slot vs CPU softmax-attention: max_abs_err = {max_err:.2e} over {} streams (slots {:?})", m_rows, sp);
            assert!(max_err < 1e-4, "bsdpa_slot wrong: max_err {max_err}");
            dev.destroy_fence(fence, None); dev.destroy_command_pool(cmd_pool, None); dev.destroy_descriptor_pool(pool, None);
            dev.destroy_pipeline(p, None); dev.destroy_pipeline_layout(l, None); dev.destroy_descriptor_set_layout(sl, None); dev.destroy_shader_module(module, None);
            for (b, mm) in bufs { dev.unmap_memory(mm); dev.destroy_buffer(b, None); dev.free_memory(mm, None); }
        }
    }

    /// COOPMAT BATCHED-FORWARD THROUGHPUT PROBE: the batched coopmat forward is
    /// fixed at 128 rows (PREFILL_MAX_M), so aggregate tok/s for M concurrent decode
    /// streams ≈ M / time(128-row forward). Tells us whether a coopmat batched
    /// decoder beats llama (M=32: 1458 tok/s) before building the slot-indirected stack.
    /// `cargo test --release --features vulkan --lib vk_coopmat_batched_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_coopmat_batched_probe() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        // ZLLM_PP-token prompt → one coopmat prefill forward (default = PREFILL_MAX_M, full chunk).
        let pp: usize = std::env::var("ZLLM_PP").ok().and_then(|s| s.parse().ok()).unwrap_or(PREFILL_MAX_M);
        let prompt: Vec<u32> = (0..pp).map(|i| ((i * 37 + 1) % 30000) as u32).collect();
        let _ = model.prefill_forward(&prompt); // warm
        let iters = 10;
        let t = Instant::now();
        for _ in 0..iters { let _ = model.prefill_forward(&prompt); }
        let t128 = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!("coopmat batched forward (128 rows): {:.1} ms", t128 * 1e3);
        eprintln!("  => aggregate decode tok/s if M streams share this forward (GEMM proxy; SDPA extra):");
        for m in [8usize, 16, 32, 64, 128] {
            let tps = m as f64 / t128;
            let vs = if m == 32 { format!("  vs llama M=32: 1458 ({})", if tps > 1458.0 { "WIN" } else { "below" }) } else { String::new() };
            eprintln!("    M={m:>3}: {tps:>5.0} tok/s{vs}");
        }
    }

    /// HETEROGENEOUS SERVING PROBE: decode on the iGPU (VkModel) and the CPU
    /// (candle) ALONE, then BOTH concurrently, to test whether the shared memory
    /// bus has headroom (iGPU decode ~136 GB/s of 256 → ~53%). If aggregate-
    /// concurrent > iGPU-alone, routing some streams to the idle CPU adds real
    /// serving throughput. If they contend badly, the strategy is dead.
    /// `cargo test --release --features vulkan --lib vk_cpu_concurrent -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_cpu_concurrent_serving() {
        use std::time::{Duration, Instant};
        use std::thread;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let vk = VkModel::load(path, ctx).expect("vk load");
        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::Backend;
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path)).expect("candle load");
        let am = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };

        // Prime both engines (short prompt; we only measure decode rate).
        let prompt: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374];
        let mut vk_next = 0u32; for (i, &t) in prompt.iter().enumerate() { vk_next = vk.forward_argmax(t, i); }
        let mut vk_pos = prompt.len();
        let mut cb_next = am(&cb.forward_logits(&prompt).unwrap());

        // Decode loops run for a fixed wall-clock window so concurrency overlaps fully.
        fn dec_vk(m: &VkModel, mut next: u32, mut pos: usize, dur: Duration) -> (usize, u32, usize) {
            let t = Instant::now(); let mut n = 0usize;
            while t.elapsed() < dur { next = m.forward_argmax(next, pos); pos += 1; n += 1; }
            (n, next, pos)
        }
        fn dec_cb(m: &mut CandleCpuBackend, mut next: u32, dur: Duration) -> (usize, u32) {
            
            let am = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
            let t = Instant::now(); let mut n = 0usize;
            while t.elapsed() < dur { next = am(&m.forward_logits(&[next]).unwrap()); n += 1; }
            (n, next)
        }
        let dur = Duration::from_secs(3);

        // 1. iGPU alone.
        let (a_vk, vn, vp) = dec_vk(&vk, vk_next, vk_pos, dur); vk_next = vn; vk_pos = vp;
        let vk_alone = a_vk as f64 / dur.as_secs_f64();
        // 2. CPU alone.
        let (a_cb, cn) = dec_cb(&mut cb, cb_next, dur); cb_next = cn;
        let cpu_alone = a_cb as f64 / dur.as_secs_f64();
        // 3. Both concurrently (separate threads → real shared-bus contention).
        let h_vk = thread::spawn(move || dec_vk(&vk, vk_next, vk_pos, dur));
        let h_cb = thread::spawn(move || dec_cb(&mut cb, cb_next, dur));
        let (c_vk, _, _) = h_vk.join().unwrap();
        let (c_cb, _) = h_cb.join().unwrap();
        let vk_conc = c_vk as f64 / dur.as_secs_f64();
        let cpu_conc = c_cb as f64 / dur.as_secs_f64();

        eprintln!("ALONE:       iGPU {vk_alone:.0} tok/s | CPU {cpu_alone:.0} tok/s");
        eprintln!("CONCURRENT:  iGPU {vk_conc:.0} tok/s | CPU {cpu_conc:.0} tok/s  (iGPU kept {:.0}%, CPU kept {:.0}%)",
            vk_conc / vk_alone * 100.0, cpu_conc / cpu_alone * 100.0);
        let agg = vk_conc + cpu_conc;
        eprintln!("VERDICT: heterogeneous aggregate {agg:.0} tok/s  vs iGPU-alone {vk_alone:.0}  =  {:.2}x  ({})",
            agg / vk_alone, if agg > vk_alone * 1.05 { "WIN — bus has headroom, CPU adds throughput" } else { "no win — bus contention cancels it out" });
    }

    /// BATCHED PREFILL vs candle: prefill a multi-token prompt through the GEMM
    /// path, check the last-token logits (cosine + argmax) match candle, then a
    /// few decode steps continue correctly from pos=M. Validates the whole
    /// batched forward (norm/QKV-GEMM/rope/causal-sdpa/wo/FFN/KV-fill).
    /// `cargo test --release --features vulkan --lib vk_prefill_vs_candle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_prefill_vs_candle() {
        use std::time::Instant;
        let path = std::env::var("ZLLM_MODEL").unwrap_or_else(|_| "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string());
        let path = path.as_str();
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let plen: usize = std::env::var("ZLLM_PLEN").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let prompt: Vec<u32> = if plen <= 6 { vec![128000, 791, 6864, 315, 9822, 374] }
            else { let mut p = vec![128000u32]; for i in 1..plen { p.push(((100 + i * 13) % 28000) as u32); } p };
        let n_gen = 12usize;

        // VkModel: batched prefill + decode.
        let t0 = Instant::now();
        let pf_logits = model.prefill_forward(&prompt);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut next = argmax(&pf_logits);
        let mut vk_gen = vec![next];
        let mut pos = prompt.len();
        for _ in 1..n_gen { next = model.forward_argmax(next, pos); vk_gen.push(next); pos += 1; }
        eprintln!("prefill {} tok in {prefill_ms:.1} ms ({:.0} tok/s)", prompt.len(), prompt.len() as f64 / (prefill_ms / 1e3));
        eprintln!("VkModel gen: {vk_gen:?}");

        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::Backend;
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path)).expect("candle load");
        let clog = cb.forward_logits(&prompt).unwrap();
        // cosine of prefill logits vs candle last-token logits
        let (mut a, mut b, mut ab) = (0f64, 0f64, 0f64);
        for i in 0..model.vocab { let (x, y) = (pf_logits[i] as f64, clog[i] as f64); a += x * x; b += y * y; ab += x * y; }
        let cos = ab / (a.sqrt() * b.sqrt());
        let mut cnext = argmax(&clog);
        let mut cand_gen = vec![cnext];
        for _ in 1..n_gen { let cl = cb.forward_logits(&[cnext]).unwrap(); cnext = argmax(&cl); cand_gen.push(cnext); }
        eprintln!("candle  gen: {cand_gen:?}");
        let agree = vk_gen.iter().zip(&cand_gen).take_while(|(a, b)| a == b).count();
        // Cross-check vs the candle-exact decode path: sequential prefill of the
        // same prompt should match batched prefill modulo f16 (isolates f16 from
        // the candle comparison).
        for (i, &tk) in prompt.iter().enumerate() { model.prefill_step(tk, i); } // re-fill cache sequentially
        let seq_logits = model.forward(*prompt.last().unwrap(), prompt.len() - 1);
        let (mut sa, mut sb, mut sab) = (0f64, 0f64, 0f64);
        for i in 0..model.vocab { let (x, y) = (pf_logits[i] as f64, seq_logits[i] as f64); sa += x * x; sb += y * y; sab += x * y; }
        let cos_seq = sab / (sa.sqrt() * sb.sqrt());
        eprintln!("prefill cosine vs candle={cos:.5}, vs decode-path={cos_seq:.5}; first token {}; greedy agree {agree}/{n_gen} (f16-limited)",
            if vk_gen[0] == cand_gen[0] { "MATCH" } else { "DIFFER" });
        assert!(cos > 0.999, "prefill logits diverge from candle: cos={cos}");
        assert_eq!(vk_gen[0], cand_gen[0], "prefill's first generated token differs from candle");
    }

    /// BARRIER-COHERENCE PROBE: chains N `b[i]+=1` dispatches with a barrier
    /// between each, then checks every element == N. Run under each barrier mode
    /// to learn whether the cheap (execution-only) barrier is correctness-safe on
    /// this GPU — if so, the ~55us cache-flush of the full barrier is avoidable.
    /// `cargo test --release --features vulkan --lib vk_barrier_coherence -- --ignored --nocapture`
    /// then re-run with VK_EXECBAR=1 and VK_NOBAR=1 and compare.
    #[test]
    #[ignore]
    fn vk_barrier_coherence_probe() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        unsafe { barrier_coherence_inner(&ctx); }
    }

    /// Q6_K MATVEC BANDWIDTH: measures the decode Q6_K matvec in isolation
    /// (GB/s on the streamed SoA bytes) + validates vs candle. Lets the kernel
    /// be tuned in ms instead of full-model runs. `VK_Q6K_SPV=v2` swaps variants.
    /// `cargo test --release --features vulkan --lib vk_q6k_bandwidth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_q6k_bandwidth() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; }
        };
        unsafe { q6k_bandwidth_inner(&ctx); }
    }

    /// FFN MEGAKERNEL A/B: runs one FFN block (norm->W13->silu->W2->residual)
    /// both as today's 5 separate dispatches and as one persistent-workgroup
    /// megakernel with grid barriers; checks they produce the same output and
    /// compares timing. `VK_GRID_G=480 cargo test --release --features vulkan --lib vk_ffn_megakernel_ab -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_ffn_megakernel_ab() {
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        unsafe { ffn_megakernel_ab_inner(&ctx); }
    }

    /// MEGAKERNEL VIABILITY: persistent Q6_K matvec WITH grid barriers between
    /// passes (one dispatch, G workgroups). Reports bandwidth + deadlock. If it
    /// sustains ~210 GB/s at high G (above the in-forward 155), a megakernel can
    /// beat llama; if it deadlocks or starves at low G, it can't.
    /// `VK_GRID_G=320 cargo test --release --features vulkan --lib vk_megakernel_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_megakernel_probe() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; }
        };
        unsafe { megakernel_probe_inner(&ctx); }
    }

    /// MEGAKERNEL FEASIBILITY: tests a grid-wide barrier (cross-workgroup sync
    /// in one dispatch). `VK_GRID_G=N` sets the workgroup count. Reports whether
    /// it deadlocks (G > resident capacity) and whether cross-wg memory is
    /// visible after the barrier. A megakernel is only possible if this passes.
    /// `VK_GRID_G=40 cargo test --release --features vulkan --lib vk_grid_barrier -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_grid_barrier() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; }
        };
        unsafe { grid_barrier_inner(&ctx); }
    }

    /// SDPA CORRECTNESS: runs the parallel decode-SDPA kernel on pseudo-random
    /// GQA q/K/V and compares to a CPU softmax-attention reference. Guards the
    /// workgroup-per-head rewrite (which gave the decode forward its lead).
    /// `cargo test --release --features vulkan --lib vk_sdpa_correctness -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_sdpa_correctness() {
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        unsafe { sdpa_correctness_inner(&ctx); }
    }

    /// SDPA TIMING: isolates the flash partial vs combine GPU time at ZLLM_CTX depth, so
    /// kernel changes are compared without the thermal/forward noise of end-to-end decode.
    /// `ZLLM_CTX=2048 cargo test --release --features vulkan --lib vk_sdpa_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_sdpa_bench() {
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        unsafe { sdpa_bench_inner(&ctx); }
    }

    /// BATCHED-MATVEC SCALING: weight-stationary Q4_K matvec (batched_matvec_q4k) at M=1..8 on
    /// an uncached shape — does holding the weight stream constant while growing M give ~M×
    /// aggregate throughput (the batching ceiling that decides if VkModel can beat llama)?
    /// `cargo test --release --features vulkan --lib vk_bmv_scaling -- --ignored --nocapture`
    /// MEGAKERNEL FOUNDATION: does an atomic grid-wide barrier work coherently on
    /// this GPU? Persistent 2-matvec kernel (y1=W1·x → grid-sync → y2=W2·y1) in ONE
    /// dispatch; if y2 is bit-exact vs CPU, the whole-forward megakernel is buildable.
    /// `cargo test --release --features vulkan --lib vk_megakernel_poc -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_megakernel_poc() {
        const WGSL: &str = r#"
struct P { n1: u32, k: u32, n2: u32, n_wg: u32 };
@group(0) @binding(0) var<storage, read>       w1: array<f32>;
@group(0) @binding(1) var<storage, read>       w2: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> y1: array<f32>;
@group(0) @binding(4) var<storage, read_write> y2: array<f32>;
@group(0) @binding(5) var<storage, read_write> sync: array<atomic<u32>, 4>;
@group(0) @binding(6) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let gid = wid.x * 64u + lid.x;
    let nthreads = p.n_wg * 64u;
    var r = gid;
    loop { if (r >= p.n1) { break; } var a = 0.0; for (var j = 0u; j < p.k; j = j + 1u) { a = a + w1[r * p.k + j] * x[j]; } y1[r] = a; r = r + nthreads; }
    workgroupBarrier(); storageBarrier();
    if (lid.x == 0u) {
        atomicAdd(&sync[0], 1u);
        var spin = 0u;
        loop { if (atomicLoad(&sync[0]) >= p.n_wg) { break; } spin = spin + 1u; if (spin > 10000000u) { break; } }
    }
    workgroupBarrier(); storageBarrier();
    r = gid;
    loop { if (r >= p.n2) { break; } var a = 0.0; for (var j = 0u; j < p.n1; j = j + 1u) { a = a + w2[r * p.n1 + j] * y1[j]; } y2[r] = a; r = r + nthreads; }
}
"#;
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let m = naga::front::wgsl::parse_str(WGSL).expect("wgsl parse");
        let i = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m).expect("validate");
        let spv = naga::back::spv::write_vec(&m, &i, &naga::back::spv::Options::default(), None).expect("spv");
        let (n1, k, n2, n_wg) = (2048usize, 2048usize, 2048usize, 128u32); // 128 wg = definitely resident
        let w1: Vec<f32> = (0..n1 * k).map(|i| ((i % 17) as f32 - 8.0) / 20.0).collect();
        let w2: Vec<f32> = (0..n2 * n1).map(|i| ((i % 13) as f32 - 6.0) / 20.0).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) / 5.0).collect();
        let mut y1 = vec![0f32; n1]; for r in 0..n1 { let mut a = 0.0; for j in 0..k { a += w1[r * k + j] * x[j]; } y1[r] = a; }
        let mut y2 = vec![0f32; n2]; for r in 0..n2 { let mut a = 0.0; for j in 0..n1 { a += w2[r * n1 + j] * y1[j]; } y2[r] = a; }
        let (gpu, ms) = ctx.megakernel_poc(&spv, &w1, &w2, &x, n1, k, n2, n_wg).expect("poc");
        let mut maxrel = 0f32; for r in 0..n2 { maxrel = maxrel.max((gpu[r] - y2[r]).abs() / (y2[r].abs() + 1e-3)); }
        eprintln!("megakernel PoC (2 matvec + atomic grid-sync, {n_wg} wg): max rel {maxrel:.3e}, {ms:.2}ms");
        assert!(maxrel < 1e-3, "GRID-SYNC RACE: y2 wrong (max rel {maxrel}) → device-scope sync NOT coherent in WGSL on this GPU; megakernel not buildable this way");
        eprintln!("✓ atomic grid-sync is COHERENT on this GPU → megakernel foundation WORKS");
    }

    /// Per-grid-sync cost (decides megakernel viability: ~194 syncs/token).
    /// `cargo test --release --features vulkan --lib vk_grid_sync_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_grid_sync_cost() {
        const WGSL: &str = r#"
struct P { n_wg: u32, n_sync: u32 };
@group(0) @binding(0) var<storage, read_write> sync: array<atomic<u32>, 256>;
@group(0) @binding(1) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    for (var s = 0u; s < p.n_sync; s = s + 1u) {
        workgroupBarrier(); storageBarrier();
        if (lid.x == 0u) {
            atomicAdd(&sync[s], 1u);
            var spin = 0u;
            loop { if (atomicLoad(&sync[s]) >= p.n_wg) { break; } spin = spin + 1u; if (spin > 10000000u) { break; } }
        }
        workgroupBarrier(); storageBarrier();
    }
}
"#;
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let m = naga::front::wgsl::parse_str(WGSL).expect("parse");
        let i = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m).expect("validate");
        let spv = naga::back::spv::write_vec(&m, &i, &naga::back::spv::Options::default(), None).expect("spv");
        for n_wg in [128u32, 256, 512, 1024] {
            let c1 = ctx.grid_sync_cost(&spv, n_wg, 1).unwrap();
            let cn = ctx.grid_sync_cost(&spv, n_wg, 200).unwrap();
            let per = (cn - c1) / 199.0 * 1000.0; // µs/sync
            eprintln!("n_wg={n_wg:>4}: {per:6.2} µs/sync → ~194 syncs/token = {:.3} ms ({:.0}% of a 4.76ms decode)", per * 194.0 / 1000.0, per * 194.0 / 1000.0 / 4.76 * 100.0);
        }
    }

    #[test]
    #[ignore]
    fn vk_bmv_scaling() {
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        unsafe { bmv_scaling_inner(&ctx); }
    }

    /// MATVEC TIMING: per-shape decode-matvec GB/s (Q4_K). CAVEAT: the iter-loop reuses one
    /// weight buffer, so shapes that fit in cache (<~16MB: wq/wv/w2) report CACHE bandwidth
    /// (can exceed the 256 GB/s DRAM peak) — only lm_head (148MB) reflects the true cold-DRAM
    /// rate (~215 GB/s = 84% of peak). In the real decode every weight streams cold (663MB model
    /// > cache), so the whole forward is DRAM-bound at the lm_head rate — matvecs are near-optimal.
    /// `cargo test --release --features vulkan --lib vk_matvec_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_matvec_bench() {
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        unsafe { matvec_bench_inner(&ctx); }
    }

// ---------------------------------------------------------------
// Standalone test helpers (were module-level #[cfg(test)] fns in
// mod.rs; moved here so mod.rs holds only implementation).
// ---------------------------------------------------------------

#[cfg(test)]
unsafe fn bmv_scaling_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    let dev = &ctx.device;
    let (n, k) = (32768usize, 2048usize); // 37MB Q4 weight — uncached (> L2), so real DRAM streaming
    let wbytes = {
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = ((((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32) / 32768.0 - 1.0) * 0.3; }
        QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec()
    };
    let (mv_p, mv_l, mv_sl, _) = ctx.make_pipeline_raw(BMV_Q4K_SPV, 3);
    let (sk_p, sk_l, sk_sl, _) = ctx.make_pipeline_raw(SKINNY_GEMM_Q4K_SPV, 3);
    let (rp_p, rp_l, rp_sl, _) = ctx.make_pipeline_raw(SKINNY_GEMM_Q4K_RP_SPV, 3);
    let wb = { let (b, _m, p) = ctx.uma_buffer(wbytes.len() as u64).unwrap(); std::ptr::copy_nonoverlapping(wbytes.as_ptr(), p, wbytes.len()); b };
    // Block-major-transposed repack for the coalesced-W variant: [b][col-tile][j][col-in-tile].
    let n_tiles = (n + 63) / 64;
    let nb_sb = k / 256;
    let wsrc = std::slice::from_raw_parts(wbytes.as_ptr() as *const u32, wbytes.len() / 4);
    let mut rsrc = vec![0u32; nb_sb * n_tiles * 36 * 64];
    for c in 0..n {
        let (ctt, cin) = (c / 64, c % 64);
        for b in 0..nb_sb { for j in 0..36 { rsrc[((b * n_tiles + ctt) * 36 + j) * 64 + cin] = wsrc[(c * nb_sb + b) * 36 + j]; } }
    }
    let rwb = { let bytes = rsrc.len() * 4; let (b, _m, p) = ctx.uma_buffer(bytes as u64).unwrap(); std::ptr::copy_nonoverlapping(rsrc.as_ptr() as *const u8, p, bytes); b };
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(24).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(80),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(24)]), None).unwrap();
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let gxof = (n as u32).min(65535);
    eprintln!("=== batched_matvec_q4k scaling, N={} K={} (37MB, uncached) ===", n, k);
    let mut t1 = 0.0f64;
    for m in [1usize, 2, 4, 8] {
        let (xb, _mx, xp) = ctx.uma_buffer((m * k * 4) as u64).unwrap();
        for i in 0..m * k { *((xp as *mut f32).add(i)) = ((i % 17) as f32 - 8.0) * 0.1; }
        let (ob, _mo, obp) = ctx.uma_buffer((m * n * 4) as u64).unwrap();
        let (obs, _mos, obsp) = ctx.uma_buffer((m * n * 4) as u64).unwrap();
        let (obr, _mor, obrp) = ctx.uma_buffer((m * n * 4) as u64).unwrap();
        let (ub, _mu, up) = ctx.uma_buffer(20).unwrap();
        std::ptr::copy_nonoverlapping([m as u32, n as u32, k as u32, (k / 256) as u32, gxof].as_ptr() as *const u8, up, 20);
        let (ub_rp, _mur, upr) = ctx.uma_buffer(20).unwrap();
        std::ptr::copy_nonoverlapping([m as u32, n as u32, k as u32, (k / 256) as u32, n_tiles as u32].as_ptr() as *const u8, upr, 20);
        let mkset3 = |sl: vk::DescriptorSetLayout, wbuf: vk::Buffer, out_b: vk::Buffer, ub_: vk::Buffer| -> vk::DescriptorSet {
            let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[sl])).unwrap()[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [wbuf, xb, out_b].iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
            let uiv = [vk::DescriptorBufferInfo::default().buffer(ub_).range(vk::WHOLE_SIZE)];
            let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uiv));
            dev.update_descriptor_sets(&w, &[]); set
        };
        let set_bmv = mkset3(mv_sl, wb, ob, ub);
        let set_sk = mkset3(sk_sl, wb, obs, ub);
        let set_rp = mkset3(rp_sl, rwb, obr, ub_rp);
        let iters = 200u32;
        // (pipeline, layout, set, grid_x, grid_y)
        let time_k = |p: vk::Pipeline, l: vk::PipelineLayout, s: vk::DescriptorSet, gx: u32, gy: u32| -> f64 {
            let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            for _ in 0..iters {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
                dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[s], &[]);
                dev.cmd_dispatch(cmd, gx, gy, 1);
                dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
            }
            dev.end_command_buffer(cmd).unwrap();
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap(); let cmds = [cmd];
            for _ in 0..2 { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
            let t0 = std::time::Instant::now();
            dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64; dev.destroy_fence(fence, None); us
        };
        let us = time_k(mv_p, mv_l, set_bmv, gxof, (n as u32).div_ceil(gxof));
        let us_sk = time_k(sk_p, sk_l, set_sk, (n as u32).div_ceil(64), 1);
        let us_rp = time_k(rp_p, rp_l, set_rp, n_tiles as u32, 1);
        let a = std::slice::from_raw_parts(obp as *const f32, m * n);
        let bb = std::slice::from_raw_parts(obsp as *const f32, m * n);
        let cc = std::slice::from_raw_parts(obrp as *const f32, m * n);
        let maxd = a.iter().zip(bb).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let maxd_rp = a.iter().zip(cc).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        if m == 1 { t1 = us; }
        let gbps = |us: f64| n as f64 * k as f64 * 0.5625 / us / 1e3;
        eprintln!("  M={}: bmv {:>6.1}us ({:>4.0}GB/s) | skinny {:>6.1}us ({:>4.0}GB/s, {:.1}x) | skinny+repack {:>6.1}us ({:>4.0}GB/s, {:.1}x vs M=1)  maxdiff sk={:.1e} rp={:.1e}",
            m, us, gbps(us), us_sk, gbps(us_sk), us / us_sk, us_rp, gbps(us_rp), (m as f64) * t1 / us_rp, maxd, maxd_rp);
    }
}

#[cfg(test)]
unsafe fn matvec_bench_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    let dev = &ctx.device;
    let qbytes = |n: usize, k: usize| -> Vec<u8> {
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = ((((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32) / 32768.0 - 1.0) * 0.3; }
        QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec()
    };
    let (mv_p, mv_l, mv_sl, _) = ctx.make_pipeline_raw(DECODE_MATVEC_Q4K_SPV, 3);
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(8).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(24),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(8)]), None).unwrap();
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let gxof = |n: usize| (n as u32).min(65535);
    // (label, rows N, reduction K)
    let shapes = [("wq/wo", 2048usize, 2048usize), ("wv/wk", 512, 2048), ("w13", 16384, 2048), ("w2", 2048, 8192), ("lm_head", 128256, 2048)];
    for (label, n, k) in shapes {
        let wb = { let bytes = qbytes(n, k); let (b, _m, p) = ctx.uma_buffer(bytes.len() as u64).unwrap(); std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()); b };
        let (xb, _mx, xp) = ctx.uma_buffer((k * 4) as u64).unwrap();
        for i in 0..k { *((xp as *mut f32).add(i)) = ((i % 17) as f32 - 8.0) * 0.1; }
        let (ob, _mo, _) = ctx.uma_buffer((n * 4) as u64).unwrap();
        let (ub, _mu, up) = ctx.uma_buffer(16).unwrap();
        std::ptr::copy_nonoverlapping([n as u32, k as u32, (k / 256) as u32, gxof(n)].as_ptr() as *const u8, up, 16);
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[mv_sl])).unwrap()[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [wb, xb, ob].iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let ui = [vk::DescriptorBufferInfo::default().buffer(ub).range(vk::WHOLE_SIZE)];
        let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui));
        dev.update_descriptor_sets(&w, &[]);
        let iters = 300u32;
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        for _ in 0..iters {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mv_p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, mv_l, 0, &[set], &[]);
            dev.cmd_dispatch(cmd, gxof(n), (n as u32).div_ceil(gxof(n)), 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        }
        dev.end_command_buffer(cmd).unwrap();
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap(); let cmds = [cmd];
        for _ in 0..2 { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
        let t0 = std::time::Instant::now();
        dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes = n as f64 * k as f64 * 0.5625; // Q4_K ~4.5 bpw
        eprintln!("matvec {:>8} N={:>6} K={:>5}: {:>6.1}us  {:>5.0} GB/s", label, n, k, us, bytes / us / 1e3);
        dev.destroy_fence(fence, None);
    }
}

#[cfg(test)]
unsafe fn sdpa_bench_inner(ctx: &VkContext) {
    use ash::vk;
    let dev = &ctx.device;
    let seq: usize = std::env::var("ZLLM_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let (n_head, n_kv, hd) = (32usize, 8usize, 64usize);
    let nblk = seq.div_ceil(SDPA_FLASH_BLOCK);
    let rnd = |i: usize, s: usize| ((i.wrapping_mul(2654435761).wrapping_add(s.wrapping_mul(40503))) & 0xFFFF) as f32 / 32768.0 - 1.0;
    let mk = |data: Vec<f32>| { let (b, _m, p) = ctx.uma_buffer((data.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p, data.len() * 4); b };
    let q = mk((0..n_head * hd).map(|i| rnd(i, 1)).collect());
    let kc = mk((0..seq * n_kv * hd).map(|i| rnd(i, 2)).collect());
    let vc = mk((0..seq * n_kv * hd).map(|i| rnd(i, 3)).collect());
    let (out, _mo, out_ptr) = ctx.uma_buffer((n_head * hd * 4) as u64).unwrap();
    let (part, _mp, _) = ctx.uma_buffer((n_head * nblk * (hd + 2) * 4) as u64).unwrap();
    let (ub, _mu, up) = ctx.uma_buffer(16).unwrap();
    let pv = [n_head as u32, n_kv as u32, hd as u32, seq as u32];
    std::ptr::copy_nonoverlapping(pv.as_ptr() as *const u8, up, 16);
    let ui = [vk::DescriptorBufferInfo::default().buffer(ub).range(vk::WHOLE_SIZE)];
    let (fp_p, fp_l, fp_sl, _) = ctx.make_pipeline_raw(SDPA_FLASH_PARTIAL_SPV, 4);
    let (f2_p, f2_l, f2_sl, _) = ctx.make_pipeline_raw(SDPA_FLASH_PARTIAL2_SPV, 4); // 2-pass partial (A/B)
    let (fc_p, fc_l, fc_sl, _) = ctx.make_pipeline_raw(SDPA_FLASH_COMBINE_SPV, 2);
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(6).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(16),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(6)]), None).unwrap();
    let mkset = |sl: vk::DescriptorSetLayout, bufs: &[vk::Buffer]| -> vk::DescriptorSet {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(bufs.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui));
        dev.update_descriptor_sets(&w, &[]); set
    };
    let s_fp = mkset(fp_sl, &[q, kc, vc, part]);
    let s_f2 = mkset(f2_sl, &[q, kc, vc, part]);
    let s_fc = mkset(fc_sl, &[part, out]);
    // Hierarchical (two-level) combine setup.
    let sup: usize = std::env::var("ZLLM_SUPER").ok().and_then(|s| s.parse().ok()).unwrap_or(8); // blocks per super-partial
    let n_super = nblk.div_ceil(sup);
    let (superp, _msp, _) = ctx.uma_buffer((n_head * n_super * (hd + 2) * 4) as u64).unwrap();
    let (ch_p, ch_l, ch_sl, _) = ctx.make_pipeline_raw(SDPA_FLASH_COMBINE_H_SPV, 2);
    let mk_u6 = |d: [u32; 6]| -> vk::Buffer { let (b, _m, p) = ctx.uma_buffer(24).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 24); b };
    let u_l1 = mk_u6([n_head as u32, n_kv as u32, hd as u32, nblk as u32, sup as u32, 0]);          // L1: merge `sup` blocks -> super-partial
    let u_l2 = mk_u6([n_head as u32, n_kv as u32, hd as u32, n_super as u32, n_super as u32, 1]);   // L2: merge all super-partials -> final
    let mkset_h = |in_b: vk::Buffer, out_b: vk::Buffer, u: vk::Buffer| -> vk::DescriptorSet {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&ch_sl))).unwrap()[0];
        let bi: Vec<[vk::DescriptorBufferInfo; 1]> = [in_b, out_b].iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let uiv = [vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)];
        let mut w: Vec<vk::WriteDescriptorSet> = bi.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uiv));
        dev.update_descriptor_sets(&w, &[]); set
    };
    let s_l1 = mkset_h(part, superp, u_l1);
    let s_l2 = mkset_h(superp, out, u_l2);
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let iters = 300u32;
    let rec = |with_combine: bool| -> vk::CommandBuffer {
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        for _ in 0..iters {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fp_p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fp_l, 0, &[s_fp], &[]);
            dev.cmd_dispatch(cmd, n_head as u32, nblk as u32, 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
            if with_combine {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fc_p);
                dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fc_l, 0, &[s_fc], &[]);
                dev.cmd_dispatch(cmd, n_head as u32, 1, 1);
                dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
            }
        }
        dev.end_command_buffer(cmd).unwrap(); cmd
    };
    let rec2 = || -> vk::CommandBuffer {  // partial + hierarchical (L1 + L2) combine
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        for _ in 0..iters {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fp_p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fp_l, 0, &[s_fp], &[]);
            dev.cmd_dispatch(cmd, n_head as u32, nblk as u32, 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, ch_p);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, ch_l, 0, &[s_l1], &[]);
            dev.cmd_dispatch(cmd, n_head as u32, n_super as u32, 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, ch_l, 0, &[s_l2], &[]);
            dev.cmd_dispatch(cmd, n_head as u32, 1, 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        }
        dev.end_command_buffer(cmd).unwrap(); cmd
    };
    let time = |cmd: vk::CommandBuffer| -> f64 {
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap(); let cmds = [cmd];
        for _ in 0..2 { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
        let t0 = std::time::Instant::now();
        dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64; dev.destroy_fence(fence, None); us
    };
    // A/B the partial kernels (online vs 2-pass), partial-only timing.
    let rec_p = |pp: vk::Pipeline, pl: vk::PipelineLayout, ps: vk::DescriptorSet| -> vk::CommandBuffer {
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        for _ in 0..iters {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pp);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pl, 0, &[ps], &[]);
            dev.cmd_dispatch(cmd, n_head as u32, nblk as u32, 1);
            dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        }
        dev.end_command_buffer(cmd).unwrap(); cmd
    };
    let p_us = time(rec(false));
    let p2_us = time(rec_p(f2_p, f2_l, s_f2));
    let f_us = time(rec(true));
    // Correctness: flat combine output vs hierarchical, max abs diff.
    let _ = time(rec(true)); let flat_out = std::slice::from_raw_parts(out_ptr as *const f32, n_head * hd).to_vec();
    let _ = time(rec2()); let hier_out = std::slice::from_raw_parts(out_ptr as *const f32, n_head * hd).to_vec();
    // 2-pass partial + flat combine, vs online partial + flat combine (validates the 2-pass partial).
    let rec_2pc = {
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, f2_p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, f2_l, 0, &[s_f2], &[]);
        dev.cmd_dispatch(cmd, n_head as u32, nblk as u32, 1);
        dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fc_p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fc_l, 0, &[s_fc], &[]);
        dev.cmd_dispatch(cmd, n_head as u32, 1, 1);
        dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        dev.end_command_buffer(cmd).unwrap(); cmd
    };
    let _ = time(rec_2pc); let p2_out = std::slice::from_raw_parts(out_ptr as *const f32, n_head * hd).to_vec();
    let maxd_h = flat_out.iter().zip(&hier_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let maxd = flat_out.iter().zip(&p2_out).map(|(a, b)| (a - b).abs()).fold(maxd_h, f32::max);
    let h_us = time(rec2());
    eprintln!("SDPA flash @ctx{} (nblk={}): partial online={:.1}us  2pass={:.1}us | flat combine={:.1}us (total {:.1}) | hier combine={:.1}us (total {:.1}, sup={} nsuper={}) | maxdiff={:.2e}",
        seq, nblk, p_us, p2_us, f_us - p_us, f_us, h_us - p_us, h_us, sup, n_super, maxd);
}

#[cfg(test)]
unsafe fn sdpa_correctness_inner(ctx: &VkContext) {
    // Exercise both paths: single-pass (short ctx) and flash 2-pass (long ctx,
    // multiple KV blocks), each vs a CPU softmax-attention reference.
    sdpa_case(ctx, 32, false);
    sdpa_case(ctx, 200, false); // deep single-pass (barrier-lean kernel), within MAXSEQ
    sdpa_case(ctx, 520, true); // > SDPA_FLASH_BLOCK and not a block multiple
}

#[cfg(test)]
unsafe fn ffn_megakernel_ab_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    let dev = &ctx.device;
    let (n_embd, n_inter) = (2048usize, 8192usize);
    let g: u32 = std::env::var("VK_GRID_G").ok().and_then(|s| s.parse().ok()).unwrap_or(480);
    let eps = 1e-5f32;
    let q = |n: usize, k: usize, dt: GgmlDType, s: i64| -> Vec<u8> {
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = ((((i as i64).wrapping_mul(2654435761).wrapping_add(s) & 0xFFFF) as f32) / 32768.0 - 1.0) * 0.4; }
        QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), dt).unwrap().data().unwrap().to_vec()
    };
    let x0: Vec<f32> = (0..n_embd).map(|i| ((i % 31) as f32 - 15.0) * 0.05).collect();
    let fnw: Vec<f32> = (0..n_embd).map(|i| 0.7 + (i % 13) as f32 * 0.02).collect();
    let w13_bytes = q(2 * n_inter, n_embd, GgmlDType::Q4K, 1);
    let w2_bytes = q(n_embd, n_inter, GgmlDType::Q6K, 7);

    let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
    let w13b = vk_up_bytes(ctx, &mut bufs, &w13_bytes);
    let (w2ql, w2qh, w2scl, w2dd) = vk_up_q6k(ctx, &mut bufs, &w2_bytes, n_embd * (n_inter / 256));
    let fnwb = vk_up_f32(ctx, &mut bufs, &fnw);
    let gxof = |n: usize| (n as u32).min(65535);
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(12).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(40),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(12),
    ]), None).unwrap();
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let fbar = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);

    let uni = |bufs: &mut Vec<(vk::Buffer, vk::DeviceMemory)>, d: [u32; 4]| -> vk::Buffer { let (b, m, p) = ctx.uma_buffer(16).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16); bufs.push((b, m)); b };
    let iters = 300u32;
    let time_cmd = |cmd: vk::CommandBuffer| -> f64 {
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap(); let cmds = [cmd];
        for _ in 0..2 { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
        let t0 = std::time::Instant::now();
        dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64; dev.destroy_fence(fence, None); us
    };
    let alloc_set = |sl: vk::DescriptorSetLayout, sb: &[(vk::Buffer, u64, u64)], u: vk::Buffer| -> vk::DescriptorSet {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = sb.iter().map(|&(b, o, r)| [vk::DescriptorBufferInfo::default().buffer(b).offset(o).range(if r == 0 { vk::WHOLE_SIZE } else { r })]).collect();
        let ui = [vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)];
        let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(sb.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui));
        dev.update_descriptor_sets(&w, &[]); set
    };

    // ---- Separate path: rmsnorm -> W13 (Q4) -> silu -> W2 (Q6) -> residual ----
    let (rms_p, rms_l, rms_sl, _) = ctx.make_pipeline_raw(RMSNORM_SPV, 3);
    let (mv_p, mv_l, mv_sl, _) = ctx.make_pipeline_raw(DECODE_MATVEC_Q4K_SPV, 3);
    let (si_p, si_l, si_sl, _) = ctx.make_pipeline_raw(SILU_MUL_SPV, 3);
    let (q6_p, q6_l, q6_sl, _) = ctx.make_pipeline_raw(DECODE_MATVEC_Q6K_SPV, 6);
    let (ad_p, ad_l, ad_sl, _) = ctx.make_pipeline_raw(RESIDUAL_ADD_SPV, 2);
    let (xs, _xsm, xs_ptr) = ctx.uma_buffer((n_embd * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(x0.as_ptr() as *const u8, xs_ptr, n_embd * 4);
    let (normed, _) = vk_zeros(ctx, &mut bufs, n_embd);
    let (gu, _) = vk_zeros(ctx, &mut bufs, 2 * n_inter);
    let (hb, _) = vk_zeros(ctx, &mut bufs, n_inter);
    let (ffnb, _) = vk_zeros(ctx, &mut bufs, n_embd);
    let w = |b: vk::Buffer| (b, 0u64, 0u64);
    let s_rms = alloc_set(rms_sl, &[w(xs), w(fnwb), w(normed)], uni(&mut bufs, [n_embd as u32, eps.to_bits(), 0, 0]));
    let s_w13 = alloc_set(mv_sl, &[w(w13b), w(normed), w(gu)], uni(&mut bufs, [(2 * n_inter) as u32, n_embd as u32, (n_embd / 256) as u32, gxof(2 * n_inter)]));
    let s_si = alloc_set(si_sl, &[(gu, 0, (n_inter * 4) as u64), (gu, (n_inter * 4) as u64, (n_inter * 4) as u64), w(hb)], uni(&mut bufs, [n_inter as u32, 0, 0, 0]));
    let s_w2 = alloc_set(q6_sl, &[w(w2ql), w(w2qh), w(w2scl), w(w2dd), w(hb), w(ffnb)], uni(&mut bufs, [n_embd as u32, (n_inter / 256) as u32, gxof(n_embd), 0]));
    let s_add = alloc_set(ad_sl, &[w(xs), w(ffnb)], uni(&mut bufs, [n_embd as u32, 0, 0, 0]));
    let cmd_sep = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    dev.begin_command_buffer(cmd_sep, &vk::CommandBufferBeginInfo::default()).unwrap();
    let bar_s = || dev.cmd_pipeline_barrier(cmd_sep, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
    let d = |p: vk::Pipeline, l: vk::PipelineLayout, set: vk::DescriptorSet, gx: u32| { dev.cmd_bind_pipeline(cmd_sep, vk::PipelineBindPoint::COMPUTE, p); dev.cmd_bind_descriptor_sets(cmd_sep, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]); dev.cmd_dispatch(cmd_sep, gx, 1, 1); };
    for _ in 0..iters {
        d(rms_p, rms_l, s_rms, 1); bar_s();
        d(mv_p, mv_l, s_w13, gxof(2 * n_inter)); bar_s();
        d(si_p, si_l, s_si, ((n_inter as u32) + 63) / 64); bar_s();
        d(q6_p, q6_l, s_w2, gxof(n_embd)); bar_s();
        d(ad_p, ad_l, s_add, ((n_embd as u32) + 63) / 64); bar_s();
    }
    dev.end_command_buffer(cmd_sep).unwrap();
    let sep_us = time_cmd(cmd_sep);
    let x_sep = std::slice::from_raw_parts(xs_ptr as *const f32, n_embd).to_vec();

    // ---- Megakernel: one persistent dispatch, grid barriers ----
    let (mg_p, mg_l, mg_sl, _) = ctx.make_pipeline_raw(FFN_MEGAKERNEL_SPV, 11);
    let (xm, _xmm, xm_ptr) = ctx.uma_buffer((n_embd * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(x0.as_ptr() as *const u8, xm_ptr, n_embd * 4);
    let (nm, _) = vk_zeros(ctx, &mut bufs, n_embd);
    let (gum, _) = vk_zeros(ctx, &mut bufs, 2 * n_inter);
    let (hm, _) = vk_zeros(ctx, &mut bufs, n_inter);
    let (ctr, _ctrm, ctr_ptr) = ctx.uma_buffer(8).unwrap(); std::ptr::write_bytes(ctr_ptr, 0, 8);
    let (umega, _umm, ump) = ctx.uma_buffer(32).unwrap();
    std::ptr::copy_nonoverlapping([n_embd as u32, n_inter as u32, (n_embd / 256) as u32, (n_inter / 256) as u32, g, eps.to_bits(), 0u32, 0u32].as_ptr() as *const u8, ump, 32);
    let s_mega = {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&mg_sl))).unwrap()[0];
        let sb = [xm, fnwb, nm, w13b, gum, hm, w2ql, w2qh, w2scl, w2dd, ctr];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = sb.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let ui = [vk::DescriptorBufferInfo::default().buffer(umega).range(vk::WHOLE_SIZE)];
        let mut wr: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        wr.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(11).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui));
        dev.update_descriptor_sets(&wr, &[]); set
    };
    let cmd_mg = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    dev.begin_command_buffer(cmd_mg, &vk::CommandBufferBeginInfo::default()).unwrap();
    dev.cmd_bind_pipeline(cmd_mg, vk::PipelineBindPoint::COMPUTE, mg_p);
    dev.cmd_bind_descriptor_sets(cmd_mg, vk::PipelineBindPoint::COMPUTE, mg_l, 0, &[s_mega], &[]);
    for _ in 0..iters {
        dev.cmd_fill_buffer(cmd_mg, ctr, 0, 8, 0);
        dev.cmd_pipeline_barrier(cmd_mg, vk::PipelineStageFlags::TRANSFER | cs, cs, vk::DependencyFlags::empty(), &[fbar], &[], &[]);
        dev.cmd_dispatch(cmd_mg, g, 1, 1);
        dev.cmd_pipeline_barrier(cmd_mg, cs, vk::PipelineStageFlags::TRANSFER | cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
    }
    dev.end_command_buffer(cmd_mg).unwrap();
    let mega_us = time_cmd(cmd_mg);
    let x_mega = std::slice::from_raw_parts(xm_ptr as *const f32, n_embd).to_vec();

    let (mut a, mut b, mut ab, mut me) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n_embd { let (xs, xm) = (x_sep[i] as f64, x_mega[i] as f64); a += xs * xs; b += xm * xm; ab += xs * xm; me = me.max((xs - xm).abs()); }
    let cos = ab / (a.sqrt() * b.sqrt());
    eprintln!("FFN A/B [G={g}]: separate {sep_us:.1} us, megakernel {mega_us:.1} us => {:.2}x | cos={cos:.6} max_err={me:.3e}",
        sep_us / mega_us);
    assert!(cos > 0.9999, "megakernel output differs from separate path: cos={cos}");

    let _ = (mg_p, mg_l);
    dev.destroy_command_pool(cmd_pool, None); dev.destroy_descriptor_pool(pool, None);
    for (b, m) in bufs { dev.unmap_memory(m); dev.destroy_buffer(b, None); dev.free_memory(m, None); }
}

#[cfg(test)]
unsafe fn megakernel_probe_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    let dev = &ctx.device;
    let (n, nb) = (131072usize, 8usize);
    let k = nb * 256;
    let g: u32 = std::env::var("VK_GRID_G").ok().and_then(|s| s.parse().ok()).unwrap_or(320);
    let passes: u32 = 20;
    let mut wv = vec![0f32; n * k];
    for i in 0..wv.len() { wv[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
    let qt = QTensor::quantize(&Tensor::from_vec(wv, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q6K).unwrap();
    let bytes = qt.data().unwrap();
    let deq: Vec<f32> = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();

    let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
    let (ql, qh, scl, dd) = vk_up_q6k(ctx, &mut bufs, &bytes, n * nb);
    let xb = vk_up_f32(ctx, &mut bufs, &x);
    let (ob, ob_ptr) = vk_zeros(ctx, &mut bufs, n);
    let (ct, _ctm, ct_ptr) = ctx.uma_buffer(8).unwrap(); std::ptr::write_bytes(ct_ptr, 0, 8);
    let (u, _) = vk_uni(ctx, &mut bufs, [n as u32, nb as u32, g, passes]);

    let (p, l, sl, m) = ctx.make_pipeline_raw(Q6K_MEGAKERNEL_PROBE_SPV, 7);
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(7),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
    ]), None).unwrap();
    let set = vk_alloc_set(dev, pool, sl, &[ql, qh, scl, dd, xb, ob, ct], u);
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
    dev.cmd_dispatch(cmd, g, 1, 1);
    dev.end_command_buffer(cmd).unwrap();
    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd];
    let t0 = std::time::Instant::now();
    dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
    let wait = dev.wait_for_fences(&[fence], true, 5_000_000_000);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    match wait {
        Err(vk::Result::TIMEOUT) => eprintln!("megakernel probe [G={g}]: DEADLOCK after 5s (G exceeds resident capacity for this kernel)"),
        Err(e) => eprintln!("megakernel probe [G={g}]: wait error {e:?}"),
        Ok(()) => {
            let gbs = (n * nb * 212) as f64 * passes as f64 / (ms / 1e3) / 1e9;
            let op = ob_ptr as *const f32;
            let (mut ca, mut cb, mut cab) = (0f64, 0f64, 0f64);
            for nn in 0..n { let mut acc = 0f64; for kk in 0..k { acc += (deq[nn*k+kk] as f64) * (x[kk] as f64); } let gg = *op.add(nn) as f64; ca += gg*gg; cb += acc*acc; cab += gg*acc; }
            let cos = cab / (ca.sqrt() * cb.sqrt());
            eprintln!("megakernel probe [G={g}]: {ms:.2} ms / {passes} passes => {gbs:.1} GB/s (vs normal 226, in-forward 155) | cos={cos:.6}");
        }
    }
    let _ = (m, l, sl, p, pool, cmd_pool, fence, ct);
}

#[cfg(test)]
unsafe fn grid_barrier_inner(ctx: &VkContext) {
    use ash::vk;
    let dev = &ctx.device;
    let n: usize = 8192;
    let g: u32 = std::env::var("VK_GRID_G").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let per = (n as u32).div_ceil(g);
    let (b_buf, _m1, _b_ptr) = ctx.uma_buffer((n * 4) as u64).unwrap();
    let (c_buf, _m2, c_ptr) = ctx.uma_buffer(((1 + g as usize) * 4) as u64).unwrap();
    std::ptr::write_bytes(c_ptr, 0, (1 + g as usize) * 4);
    let (u_buf, _m3, u_ptr) = ctx.uma_buffer(16).unwrap();
    std::ptr::copy_nonoverlapping([n as u32, g, per, 0u32].as_ptr() as *const u8, u_ptr, 16);

    let (p, l, sl, m) = ctx.make_pipeline_raw(GRID_BARRIER_PROBE_SPV, 2);
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
    ]), None).unwrap();
    let set = vk_alloc_set(dev, pool, sl, &[b_buf, c_buf], u_buf);
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
    dev.cmd_dispatch(cmd, g, 1, 1);
    dev.end_command_buffer(cmd).unwrap();
    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd];
    let t0 = std::time::Instant::now();
    dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
    let wait = dev.wait_for_fences(&[fence], true, 3_000_000_000); // 3s timeout → catches deadlock
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    match wait {
        Err(vk::Result::TIMEOUT) => eprintln!("grid barrier [G={g}]: DEADLOCK (G exceeds resident capacity) after 3s"),
        Err(e) => eprintln!("grid barrier [G={g}]: wait error {e:?}"),
        Ok(()) => {
            let cp = c_ptr as *const u32;
            let mut wrong = 0u32; let mut sample = 0u32;
            for w in 0..g as usize { let v = *cp.add(1 + w); if w == 0 { sample = v; } if v != n as u32 { wrong += 1; } }
            eprintln!("grid barrier [G={g}]: {ms:.3} ms, each wg's sum-of-all = {sample} (want {n}), wrong={wrong}/{g} => {}",
                if wrong == 0 { "WORKS (cross-wg visible)" } else { "STALE (barrier broken)" });
        }
    }
    let _ = (b_buf, c_buf, u_buf); // leak (test exits)
    let _ = (p, l, sl, m, pool, cmd_pool, fence);
}

#[cfg(test)]
unsafe fn q6k_bandwidth_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    let dev = &ctx.device;
    let (n, nb) = (131072usize, 8usize); // > MALL cache (~217 MB SoA) → real DRAM bandwidth
    let k = nb * 256;
    let variant = std::env::var("VK_Q6K_SPV").unwrap_or_default();
    let persist_g: u32 = std::env::var("VK_GRID_G").ok().and_then(|s| s.parse().ok()).unwrap_or(160);
    let (spv, soa_bytes_per_block): (&[u8], usize) = match variant.as_str() {
        "v2" => (DECODE_MATVEC_Q6K_V2_SPV, 212),
        "persist" => (DECODE_MATVEC_Q6K_PERSIST_SPV, 212),
        _ => (DECODE_MATVEC_Q6K_SPV, 212),
    };
    let persist = variant == "persist";

    // Quantize a tensor to Q6_K → raw bytes → SoA.
    let mut wv = vec![0f32; n * k];
    for i in 0..wv.len() { wv[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
    let qt = QTensor::quantize(&Tensor::from_vec(wv, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q6K).unwrap();
    let bytes = qt.data().unwrap();
    let deq: Vec<f32> = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();

    let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
    let (ql, qh, scl, dd) = vk_up_q6k(ctx, &mut bufs, &bytes, n * nb);
    let xb = vk_up_f32(ctx, &mut bufs, &x);
    let (ob, ob_ptr) = vk_zeros(ctx, &mut bufs, n);
    let (u, _) = vk_uni(ctx, &mut bufs, [n as u32, nb as u32, if persist { persist_g } else { (n as u32).min(65535) }, 0]);

    let (p, l, sl, m) = ctx.make_pipeline_raw(spv, 6);
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(6),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
    ]), None).unwrap();
    let set = vk_alloc_set(dev, pool, sl, &[ql, qh, scl, dd, xb, ob], u);

    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    let (gx, gy) = if persist { (persist_g, 1u32) } else { ((n as u32).min(65535), (n as u32).div_ceil((n as u32).min(65535))) };
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    let iters = 100u32;
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
    for _ in 0..iters { dev.cmd_dispatch(cmd, gx, gy, 1); dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]); }
    dev.end_command_buffer(cmd).unwrap();
    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd];
    // warm
    dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
    dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    dev.reset_fences(&[fence]).unwrap();
    let t0 = std::time::Instant::now();
    dev.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
    dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let bytes_streamed = (n * nb * soa_bytes_per_block) as f64 * iters as f64;
    let gbs = bytes_streamed / (ms / 1e3) / 1e9;

    // correctness
    let op = ob_ptr as *const f32;
    let (mut max_err, mut cos_a, mut cos_b, mut cos_ab) = (0f64, 0f64, 0f64, 0f64);
    for nn in 0..n {
        let mut acc = 0f64;
        for kk in 0..k { acc += (deq[nn * k + kk] as f64) * (x[kk] as f64); }
        let g = *op.add(nn) as f64;
        max_err = max_err.max((g - acc).abs());
        cos_a += g * g; cos_b += acc * acc; cos_ab += g * acc;
    }
    let cos = cos_ab / (cos_a.sqrt() * cos_b.sqrt());
    let label = if variant.is_empty() { "current" } else { &variant };
    eprintln!("Q6_K matvec [{label}]: {ms:.3} ms/{iters} => {:.2} ms/iter, {gbs:.1} GB/s | cos={cos:.6} max_err={max_err:.3e}", ms / iters as f64);
    assert!(cos > 0.9999, "Q6_K matvec wrong: cos={cos}");

    dev.destroy_fence(fence, None); dev.destroy_command_pool(cmd_pool, None); dev.destroy_descriptor_pool(pool, None);
    dev.destroy_pipeline(p, None); dev.destroy_pipeline_layout(l, None); dev.destroy_descriptor_set_layout(sl, None); dev.destroy_shader_module(m, None);
    for (b, mm) in bufs { dev.unmap_memory(mm); dev.destroy_buffer(b, None); dev.free_memory(mm, None); }
}

#[cfg(test)]
unsafe fn sdpa_case(ctx: &VkContext, seq_len: usize, flash: bool) {
    use ash::vk;
    let dev = &ctx.device;
    let (n_head, n_kv, hd) = (32usize, 8usize, 64usize);
    let qn = n_head * hd; let kvn = seq_len * n_kv * hd;
    let rnd = |i: usize, s: usize| ((i.wrapping_mul(2654435761).wrapping_add(s.wrapping_mul(40503))) & 0xFFFF) as f32 / 32768.0 - 1.0;
    let qd: Vec<f32> = (0..qn).map(|i| rnd(i, 1)).collect();
    let kd: Vec<f32> = (0..kvn).map(|i| rnd(i, 2)).collect();
    let vd: Vec<f32> = (0..kvn).map(|i| rnd(i, 3)).collect();

    let mkbuf = |data: &[f32]| { let (b, _m, p) = ctx.uma_buffer((data.len() * 4) as u64).unwrap(); std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p, data.len() * 4); b };
    let q = mkbuf(&qd); let kc = mkbuf(&kd); let vc = mkbuf(&vd);
    let (out, _mo, out_ptr) = ctx.uma_buffer((qn * 4) as u64).unwrap();
    let (ub, _mu, up) = ctx.uma_buffer(16).unwrap();
    let pv = [n_head as u32, n_kv as u32, hd as u32, seq_len as u32];
    std::ptr::copy_nonoverlapping(pv.as_ptr() as *const u8, up, 16);
    let ui = [vk::DescriptorBufferInfo::default().buffer(ub).range(vk::WHOLE_SIZE)];

    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(6),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(2),
    ]), None).unwrap();
    let mkset = |sl: vk::DescriptorSetLayout, bufs: &[vk::Buffer]| -> vk::DescriptorSet {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)|
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(bufs.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui));
        dev.update_descriptor_sets(&writes, &[]);
        set
    };
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    if flash {
        let nblk = seq_len.div_ceil(SDPA_FLASH_BLOCK);
        let part = { let (b, _m, _p) = ctx.uma_buffer((n_head * nblk * (hd + 2) * 4) as u64).unwrap(); b };
        let (fp_p, fp_l, fp_sl, _a) = ctx.make_pipeline_raw(SDPA_FLASH_PARTIAL_SPV, 4);
        let (fc_p, fc_l, fc_sl, _b) = ctx.make_pipeline_raw(SDPA_FLASH_COMBINE_SPV, 2);
        let s_fp = mkset(fp_sl, &[q, kc, vc, part]);
        let s_fc = mkset(fc_sl, &[part, out]);
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fp_p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fp_l, 0, &[s_fp], &[]);
        dev.cmd_dispatch(cmd, n_head as u32, nblk as u32, 1);
        dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fc_p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, fc_l, 0, &[s_fc], &[]);
        dev.cmd_dispatch(cmd, n_head as u32, 1, 1);
    } else {
        let (sd_p, sd_l, sd_sl, _a) = ctx.make_pipeline_raw(SDPA_DECODE_SPV, 4);
        let set = mkset(sd_sl, &[q, kc, vc, out]);
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sd_p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, sd_l, 0, &[set], &[]);
        dev.cmd_dispatch(cmd, n_head as u32, 1, 1);
    }
    dev.end_command_buffer(cmd).unwrap();
    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd]; let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    dev.queue_submit(ctx.queue, &[submit], fence).unwrap();
    dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();

    // CPU reference: standard softmax attention per head with GQA.
    let scale = 1.0 / (hd as f32).sqrt();
    let mut refo = vec![0f32; qn];
    for h in 0..n_head {
        let kvh = h / (n_head / n_kv);
        let mut sc = vec![0f32; seq_len];
        let mut mx = f32::MIN;
        for t in 0..seq_len {
            let mut s = 0f32;
            for d in 0..hd { s += qd[h * hd + d] * kd[(t * n_kv + kvh) * hd + d]; }
            sc[t] = s * scale; mx = mx.max(sc[t]);
        }
        let mut den = 0f32;
        for t in 0..seq_len { sc[t] = (sc[t] - mx).exp(); den += sc[t]; }
        for d in 0..hd {
            let mut a = 0f32;
            for t in 0..seq_len { a += sc[t] * vd[(t * n_kv + kvh) * hd + d]; }
            refo[h * hd + d] = a / den;
        }
    }
    let op = out_ptr as *const f32;
    let (mut max_err, mut argmax) = (0f32, 0usize);
    for i in 0..qn { let e = (*op.add(i) - refo[i]).abs(); if e > max_err { max_err = e; argmax = i; } }
    let kind = if flash { "flash" } else { "single" };
    eprintln!("SDPA correctness [{kind}, seq={seq_len}]: max_abs_err={max_err:.3e} at {argmax} => {}",
        if max_err < 1e-3 { "PASS" } else { "FAIL" });
    assert!(max_err < 1e-3, "SDPA {kind} mismatch: {max_err}");
}

#[cfg(test)]
unsafe fn barrier_coherence_inner(ctx: &VkContext) {
    use ash::vk;
    let dev = &ctx.device;
    let n: usize = 8192;
    let steps: u32 = 64;
    let (buf, _m, ptr) = ctx.uma_buffer((n * 4) as u64).unwrap();
    std::ptr::write_bytes(ptr, 0, n * 4); // b[i] = 0
    let (b, _m2, up) = ctx.uma_buffer(16).unwrap();
    let pv = [n as u32, 0u32, 0u32, 0u32];
    std::ptr::copy_nonoverlapping(pv.as_ptr() as *const u8, up, 16);
    let coh = std::env::var("VK_COH").is_ok(); // VK_COH=1 uses the `coherent` buffer shader
    let (p_p, p_l, p_sl, _m) = ctx.make_pipeline_raw(if coh { INC_COH_SPV } else { INC_SPV }, 1);

    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(2),
    ]), None).unwrap();
    let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&p_sl))).unwrap()[0];
    let bi = [vk::DescriptorBufferInfo::default().buffer(buf).range(vk::WHOLE_SIZE)];
    let ui = [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)];
    dev.update_descriptor_sets(&[
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui),
    ], &[]);

    let no_barriers = std::env::var("VK_NOBAR").is_ok();
    let exec_bar = std::env::var("VK_EXECBAR").is_ok();
    let buf_bar = std::env::var("VK_BUFBAR").is_ok(); // BufferMemoryBarrier scoped to the buffer
    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let bbar = [vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).buffer(buf).offset(0).size(vk::WHOLE_SIZE)];
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p_p);
    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p_l, 0, &[set], &[]);
    let no_barriers2 = no_barriers; // capture for the closure
    for s in 0..steps {
        dev.cmd_dispatch(cmd, (n as u32).div_ceil(64), 1, 1);
        if s + 1 < steps && !no_barriers2 {
            if exec_bar      { dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[], &[], &[]); }
            else if buf_bar  { dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[], &bbar, &[]); }
            else             { dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]); }
        }
    }
    dev.end_command_buffer(cmd).unwrap();
    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    // Correctness from one submit (buffer started at 0).
    dev.queue_submit(ctx.queue, &[submit], fence).unwrap();
    dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    let fp = ptr as *const f32;
    let (mut mn, mut mx, mut wrong) = (f32::MAX, f32::MIN, 0u32);
    for i in 0..n { let v = *fp.add(i); mn = mn.min(v); mx = mx.max(v); if (v - steps as f32).abs() > 0.5 { wrong += 1; } }
    // Timing: resubmit the same cmd buffer (values keep accumulating; we only time).
    let iters = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..iters { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[submit], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
    let us_per_bar = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64 * (steps - 1) as f64);
    let mode = format!("{}{}", if no_barriers { "NOBAR" } else if exec_bar { "EXEC-only" } else if buf_bar { "BUFBAR" } else { "FULL" }, if coh { "+coherent" } else { "" });
    eprintln!("barrier [{mode}]: got min={mn} max={mx} wrong={wrong}/{n} => {} | ~{us_per_bar:.1} us/barrier",
        if wrong == 0 { "CORRECT" } else { "STALE" });
}

#[cfg(test)]
unsafe fn fused_decode_inner(ctx: &VkContext) {
    use ash::vk;
    use candle_core::{Device, Tensor};
    use candle_core::quantized::{QTensor, GgmlDType};
    use std::time::Instant;
    let dev = &ctx.device;
    // Llama-3.2-1B config.
    let (n_embd, n_head, n_kv, hd, n_inter, vocab, n_layers, max_seq) =
        (2048usize, 32usize, 8usize, 64usize, 8192usize, 128256usize, 16usize, 64usize);
    let kv_dim = n_kv * hd;
    // VK_SEQ sets the KV-cache depth (context length) to stress SDPA scaling.
    let seq_len: u32 = std::env::var("VK_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq = (seq_len as usize).max(max_seq);
    let eps = 1e-5f32;

    let q4k = |n: usize, k: usize| -> Vec<u8> {
        let mut w = vec![0f32; n * k];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec()
    };
    let buf_bytes = |bytes: &[u8]| { let (b, _m, p) = ctx.uma_buffer(bytes.len() as u64).unwrap(); std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()); b };
    let buf_f32 = |len: usize| { let (b, _m, _p) = ctx.uma_buffer((len * 4) as u64).unwrap(); b };
    let uni = |d: [u32; 4]| { let (b, _m, p) = ctx.uma_buffer(16).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16); b };

    // Resident dummy weights (shared across layers) + activations + KV cache.
    let wq = buf_bytes(&q4k(n_embd, n_embd)); let wk = buf_bytes(&q4k(kv_dim, n_embd)); let wv = buf_bytes(&q4k(kv_dim, n_embd));
    let wo = buf_bytes(&q4k(n_embd, n_embd)); let w1 = buf_bytes(&q4k(n_inter, n_embd)); let w3 = buf_bytes(&q4k(n_inter, n_embd));
    let w2 = buf_bytes(&q4k(n_embd, n_inter)); let lm = buf_bytes(&q4k(vocab, n_embd));
    let normw = buf_f32(n_embd); let cosb = buf_f32(hd / 2); let sinb = buf_f32(hd / 2);
    let x = buf_f32(n_embd); let normed = buf_f32(n_embd); let q = buf_f32(n_embd); let k = buf_f32(kv_dim); let v = buf_f32(kv_dim);
    let attn = buf_f32(n_embd); let gate = buf_f32(n_inter); let up = buf_f32(n_inter); let h = buf_f32(n_inter);
    let (logits, _lmm, logits_ptr) = ctx.uma_buffer((vocab * 4) as u64).unwrap(); // keep ptr for correctness check
    let kc = buf_f32(max_seq * kv_dim); let vc = buf_f32(max_seq * kv_dim);
    let n_blocks_max = max_seq.div_ceil(SDPA_FLASH_BLOCK);
    let part = buf_f32(n_head * n_blocks_max * (hd + 2)); // flash-attn per-block partials

    // Pipelines (storage-buffer count per kernel).
    let (mv_p, mv_l, mv_sl, _m0) = ctx.make_pipeline_raw(DECODE_MATVEC_Q4K_SPV, 3);
    let (rn_p, rn_l, rn_sl, _m1) = ctx.make_pipeline_raw(RMSNORM_SPV, 3);
    let (ro_p, ro_l, ro_sl, _m2) = ctx.make_pipeline_raw(ROPE_SPV, 3);
    let (sd_p, sd_l, sd_sl, _m3) = ctx.make_pipeline_raw(SDPA_DECODE_SPV, 4);
    let (fp_p, fp_l, fp_sl, _mf0) = ctx.make_pipeline_raw(SDPA_FLASH_PARTIAL_SPV, 4); // flash pass 1
    let (fc_p, fc_l, fc_sl, _mf1) = ctx.make_pipeline_raw(SDPA_FLASH_COMBINE_SPV, 2); // flash pass 2
    let (si_p, si_l, si_sl, _m4) = ctx.make_pipeline_raw(SILU_MUL_SPV, 3);
    let (dn_p, dn_l, dn_sl, _m5) = ctx.make_pipeline_raw(DECODE_MATVEC_DOWN_Q4K_SPV, 4); // fused silu·mul + down

    let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(20).pool_sizes(&[
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(80),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(20),
    ]), None).unwrap();
    let mkset = |sl: vk::DescriptorSetLayout, bufs: &[vk::Buffer], u: vk::Buffer| -> vk::DescriptorSet {
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
        let mut infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        infos.push([vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)]);
        let writes: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| {
            let ty = if i + 1 == infos.len() { vk::DescriptorType::UNIFORM_BUFFER } else { vk::DescriptorType::STORAGE_BUFFER };
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(ty).buffer_info(info)
        }).collect();
        dev.update_descriptor_sets(&writes, &[]);
        set
    };
    let gxof = |n: usize| (n as u32).min(65535);
    let mvuni = |n: usize, kk: usize| uni([n as u32, kk as u32, (kk / 256) as u32, gxof(n)]);
    // Descriptor sets (built once, reused every layer).
    let s_rn = mkset(rn_sl, &[x, normw, normed], uni([n_embd as u32, eps.to_bits(), 0, 0]));
    let s_wq = mkset(mv_sl, &[wq, normed, q], mvuni(n_embd, n_embd));
    let s_wk = mkset(mv_sl, &[wk, normed, k], mvuni(kv_dim, n_embd));
    let s_wv = mkset(mv_sl, &[wv, normed, v], mvuni(kv_dim, n_embd));
    let s_rq = mkset(ro_sl, &[q, cosb, sinb], uni([n_head as u32, hd as u32, 0, 0]));
    let s_rk = mkset(ro_sl, &[k, cosb, sinb], uni([n_kv as u32, hd as u32, 0, 0]));
    let s_sd = mkset(sd_sl, &[q, kc, vc, attn], uni([n_head as u32, n_kv as u32, hd as u32, seq_len]));
    let s_fp = mkset(fp_sl, &[q, kc, vc, part], uni([n_head as u32, n_kv as u32, hd as u32, seq_len]));
    let s_fc = mkset(fc_sl, &[part, attn], uni([n_head as u32, n_kv as u32, hd as u32, seq_len]));
    let s_wo = mkset(mv_sl, &[wo, attn, x], mvuni(n_embd, n_embd));
    let s_w1 = mkset(mv_sl, &[w1, normed, gate], mvuni(n_inter, n_embd));
    let s_w3 = mkset(mv_sl, &[w3, normed, up], mvuni(n_inter, n_embd));
    let s_si = mkset(si_sl, &[gate, up, h], uni([n_inter as u32, 0, 0, 0]));
    let s_w2 = mkset(mv_sl, &[w2, h, x], mvuni(n_embd, n_inter));
    let s_w2d = mkset(dn_sl, &[w2, gate, up, x], mvuni(n_embd, n_inter)); // fused: reads gate+up, silu·mul inline
    let s_lm = mkset(mv_sl, &[lm, normed, logits], mvuni(vocab, n_embd));

    let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family), None).unwrap();
    let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
    let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
    let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
    let disp = |p: vk::Pipeline, l: vk::PipelineLayout, set: vk::DescriptorSet, gx: u32, gy: u32| {
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, l, 0, &[set], &[]);
        dev.cmd_dispatch(cmd, gx, gy, 1);
    };
    // The decode wall was NOT the barriers. VK_NOBAR=1 (drop all barriers) and
    // VK_EXECBAR=1 (execution-only) both hit ~370 tok/s — but the coherence probe
    // proves both are STALE: this driver elides a memory-barrier-less pipeline
    // barrier, so those numbers are racing layers, not a real floor. A *correct*
    // full barrier is only ~2us (see vk_barrier_coherence_probe).
    // The real wall was the SDPA kernel: one thread per head (1 of 40 CUs) with a
    // float[128] accumulator that spilled to scratch — ~410us x 16 layers = ~6.6ms.
    // VK_SKIP=sdpa isolated it (107 -> 358 tok/s). Rewritten as one workgroup per
    // head (parallel over head-dim, no spill), the full forward is ~290 tok/s,
    // beating llama's 201. Fusion experiments that backfired (kept for the record):
    // rmsnorm-into-matvec and silu-into-down (VK_FUSE=1) both recompute a per-
    // element transform once per output row — always a net loss.
    let no_barriers = std::env::var("VK_NOBAR").is_ok();
    let exec_bar = std::env::var("VK_EXECBAR").is_ok(); // execution-only barrier (no mem flush) — isolates flush cost
    let fuse_ffn = std::env::var("VK_FUSE").is_ok();     // VK_FUSE=1 folds silu into down (backfires: redundant exp/row)
    // Per-category skip flags (VK_SKIP=norm,rope,sdpa,silu) to attribute the
    // gap between the ~4.2ms serial-matvec floor and the ~10ms full forward.
    let skip = std::env::var("VK_SKIP").unwrap_or_default();
    let (skip_norm, skip_rope, skip_sdpa, skip_silu) =
        (skip.contains("norm"), skip.contains("rope"), skip.contains("sdpa"), skip.contains("silu"));
    let bar = || { if !no_barriers {
        if exec_bar { dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[], &[], &[]); }
        else        { dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]); }
    } };
    let mv = |set, n: usize| disp(mv_p, mv_l, set, gxof(n), (n as u32).div_ceil(gxof(n)));
    for _ in 0..n_layers {
        if !skip_norm { disp(rn_p, rn_l, s_rn, 1, 1); bar(); }                         // attn norm
        mv(s_wq, n_embd); mv(s_wk, kv_dim); mv(s_wv, kv_dim); bar();                   // QKV
        if !skip_rope {
            disp(ro_p, ro_l, s_rq, ((n_head * hd / 2) as u32).div_ceil(64), 1);
            disp(ro_p, ro_l, s_rk, ((n_kv * hd / 2) as u32).div_ceil(64), 1); bar();   // RoPE q,k
        }
        if !skip_sdpa {
            if (seq_len as usize) > SDPA_FLASH_BLOCK {
                // Flash: partial blocks (grid n_head × n_blocks) then combine.
                let nblk = (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK) as u32;
                disp(fp_p, fp_l, s_fp, n_head as u32, nblk); bar();
                disp(fc_p, fc_l, s_fc, n_head as u32, 1); bar();
            } else {
                disp(sd_p, sd_l, s_sd, n_head as u32, 1); bar();                       // single-pass (short ctx)
            }
        }
        mv(s_wo, n_embd); bar();                                                       // O proj
        if !skip_norm { disp(rn_p, rn_l, s_rn, 1, 1); bar(); }                         // ffn norm
        mv(s_w1, n_inter); mv(s_w3, n_inter); bar();                                   // gate, up
        if fuse_ffn {
            // Fused: down-proj reads gate+up and computes silu(gate)*up inline. This
            // BACKFIRES (88 vs 104) — silu/exp is recomputed once per output row.
            disp(dn_p, dn_l, s_w2d, gxof(n_embd), (n_embd as u32).div_ceil(gxof(n_embd))); bar();
        } else {
            if !skip_silu { disp(si_p, si_l, s_si, (n_inter as u32).div_ceil(64), 1); bar(); } // silu·mul
            mv(s_w2, n_embd); bar();                                                   // down proj
        }
    }
    disp(rn_p, rn_l, s_rn, 1, 1); bar();                                              // final norm
    mv(s_lm, vocab);                                                                   // LM head
    dev.end_command_buffer(cmd).unwrap();

    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    let cmds = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    for _ in 0..3 { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[submit], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); } // warm
    let toks = 30;
    let t0 = Instant::now();
    for _ in 0..toks { dev.reset_fences(&[fence]).unwrap(); dev.queue_submit(ctx.queue, &[submit], fence).unwrap(); dev.wait_for_fences(&[fence], true, u64::MAX).unwrap(); }
    let per = t0.elapsed().as_secs_f64() * 1e3 / toks as f64;
    // Correctness probe: checksum the logits. Full-barrier is the reference; if
    // exec-only/coherent modes match it, the lighter barrier is safe on this GPU.
    let lp = logits_ptr as *const f32;
    let (mut checksum, mut nan) = (0f64, 0u32);
    let mut first = [0f32; 4];
    for i in 0..vocab { let v = *lp.add(i); if v.is_nan() { nan += 1; } checksum += v as f64; if i < 4 { first[i] = v; } }
    eprintln!("FUSED decode forward: {per:.3} ms/token => {:.0} tok/s", 1000.0 / per);
    eprintln!("  logits checksum={checksum:.4} nan={nan} first={first:?}");
    eprintln!("  vs wgpu decode ~80; llama.cpp iGPU 201 (BEATEN); matvec-only ceiling ~355");
}
