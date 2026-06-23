//! AMD iGPU compute path via `wgpu` (→ Vulkan on this box).
//!
//! Why wgpu and not raw Vulkan (`ash`): this machine has the Vulkan
//! *runtime* but no SDK / glslang / cmake, so GLSL→SPIR-V toolchains
//! (shaderc, vulkano-shaders) won't build. wgpu compiles WGSL with the
//! pure-Rust `naga` crate — no external toolchain. The tradeoff is no
//! `VK_KHR_cooperative_matrix` from WGSL, but the iGPU's decode advantage
//! over the CPU is *memory bandwidth* (it reaches far more of the shared
//! ~256 GB/s than the CPU cores' ~55 GB/s), which a plain compute shader
//! captures. Cooperative-matrix (compute-bound prefill) can move to
//! hand-authored SPIR-V later if needed.
//!
//! This module starts as a feasibility spike (device bring-up + a trivial
//! validated compute dispatch). The Q4_K matmul kernel and the on-GPU
//! forward path build on top of it.

use pollster::FutureExt as _;

/// A live GPU device + queue, plus identifying info. One per process.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    /// Shared Q4_K mat-vec pipeline (created once, reused for every
    /// matmul — the inference path records many dispatches against it).
    q4k_pipeline: wgpu::ComputePipeline,
    /// Fused SiLU(gate)·up elementwise pipeline for the FFN.
    silu_mul_pipeline: wgpu::ComputePipeline,
    /// Interleaved RoPE (rope_i) pipeline, applied in-place to Q and K.
    rope_pipeline: wgpu::ComputePipeline,
    /// GQA decode self-attention (online softmax), one thread per query head.
    sdpa_pipeline: wgpu::ComputePipeline,
    /// RMSNorm (single-workgroup sum-of-squares reduction), x→y out-of-place.
    norm_pipeline: wgpu::ComputePipeline,
    /// In-place residual add: `a[i] += b[i]`.
    add_pipeline: wgpu::ComputePipeline,
    /// Q6_K mat-vec (struct-of-arrays layout). Needed for the Q6_K-quantized
    /// attn_v/ffn_down weights in Q4_K_M files.
    q6k_pipeline: wgpu::ComputePipeline,
    /// Plain f32 mat-vec (bandwidth-bound). Used for the LM head from the
    /// already-dequantized embedding — exact + ~6x faster than the Q6_K path.
    f32_pipeline: wgpu::ComputePipeline,
    /// Argmax over the logits (greedy decode) — returns 1 u32 so we read
    /// back 4 bytes instead of the 128k-wide logit vector.
    argmax_pipeline: wgpu::ComputePipeline,
    /// FFN down-projection with SiLU·mul fused into the activation read,
    /// removing the separate silu_mul dispatch. Q4_K / Q6_K variants.
    q4k_down_pipeline: wgpu::ComputePipeline,
    q6k_down_pipeline: wgpu::ComputePipeline,
    /// Batched Q4_K GEMM for prefill (weight row reused across M prompt rows).
    q4k_gemm_pipeline: wgpu::ComputePipeline,
    q6k_gemm_pipeline: wgpu::ComputePipeline,
    bnorm_pipeline: wgpu::ComputePipeline,
    brope_pipeline: wgpu::ComputePipeline,
    bsdpa_pipeline: wgpu::ComputePipeline,
    /// Batched decode: per-stream SDPA (each of M concurrent streams attends
    /// its OWN KV cache at its OWN position) and per-stream argmax.
    bdsdpa_pipeline: wgpu::ComputePipeline,
    /// Paged variant of bdsdpa: the KV is a shared block pool and each stream's
    /// key positions are gathered through a per-slot block table (PagedAttention).
    bdsdpa_paged_pipeline: wgpu::ComputePipeline,
    bargmax_pipeline: wgpu::ComputePipeline,
    /// Batched temperature sampling (Gumbel-max argmax over perturbed logits).
    bsample_pipeline: wgpu::ComputePipeline,
    /// Batched top-K extraction (for CPU-side top-k/top-p sampling).
    btopk_pipeline: wgpu::ComputePipeline,
}

/// A dense f32 weight matrix resident on the GPU (row-major).
pub struct ResidentF32 {
    w: wgpu::Buffer,
    params: wgpu::Buffer,
    pub n_rows: usize,
}

/// A Q6_K weight matrix resident on the GPU, repacked to struct-of-arrays
/// (the native 210-byte block isn't u32-aligned). `ql`/`qh` are packed
/// nibble/2-bit planes; `scales` (i8→f32) and `d` (f16→f32) are expanded.
pub struct ResidentQ6K {
    ql: wgpu::Buffer,
    qh: wgpu::Buffer,
    scales: wgpu::Buffer,
    d: wgpu::Buffer,
    params: wgpu::Buffer,
    pub n_rows: usize,
    pub nb_per_row: usize,
}

/// A Q4_K weight matrix resident on the GPU: uploaded once, reused across
/// every forward. The whole point of the GPU path is that weights never
/// re-transfer and intermediate activations stay in GPU buffers between
/// matmuls (no per-op CPU↔GPU round-trip, which is what would kill decode).
pub struct ResidentQ4K {
    w_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    pub n_rows: usize,
    pub nb_per_row: usize,
}

impl GpuContext {
    /// Bring up the high-performance adapter (the discrete/integrated GPU,
    /// not a software fallback). Prefers Vulkan. Returns an error string
    /// if no suitable adapter or device can be acquired.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .ok_or_else(|| "no suitable GPU adapter found".to_string())?;

        let info = adapter.get_info();
        // Use the hardware's real limits (the iGPU supports multi-GB
        // storage buffers); the wgpu defaults cap a storage binding at
        // 128 MB, too small for an LM-head weight matrix.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("zllm-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .map_err(|e| format!("request_device failed: {e}"))?;

        let q4k_pipeline = Self::make_pipeline(&device, "q4k-matvec", Q4K_MATVEC_WGSL);
        let silu_mul_pipeline = Self::make_pipeline(&device, "silu-mul", SILU_MUL_WGSL);
        let rope_pipeline = Self::make_pipeline(&device, "rope-i", ROPE_WGSL);
        let sdpa_pipeline = Self::make_pipeline(&device, "sdpa-decode", SDPA_DECODE_WGSL);
        let norm_pipeline = Self::make_pipeline(&device, "rmsnorm", RMSNORM_WGSL);
        let add_pipeline = Self::make_pipeline(&device, "residual-add", ADD_WGSL);
        let q6k_pipeline = Self::make_pipeline(&device, "q6k-matvec", Q6K_MATVEC_WGSL);
        let f32_pipeline = Self::make_pipeline(&device, "f32-matvec", F32_MATVEC_WGSL);
        let argmax_pipeline = Self::make_pipeline(&device, "argmax", ARGMAX_WGSL);
        let q4k_down_pipeline = Self::make_pipeline(&device, "q4k-down", Q4K_DOWN_WGSL);
        let q6k_down_pipeline = Self::make_pipeline(&device, "q6k-down", Q6K_DOWN_WGSL);
        let q4k_gemm_pipeline = Self::make_pipeline(&device, "q4k-gemm", Q4K_GEMM_WGSL);
        let q6k_gemm_pipeline = Self::make_pipeline(&device, "q6k-gemm", Q6K_GEMM_WGSL);
        let bnorm_pipeline = Self::make_pipeline(&device, "bnorm", BNORM_WGSL);
        let brope_pipeline = Self::make_pipeline(&device, "brope", BROPE_WGSL);
        let bsdpa_pipeline = Self::make_pipeline(&device, "bsdpa", BSDPA_WGSL);
        let bdsdpa_pipeline = Self::make_pipeline(&device, "bdsdpa", BDSDPA_WGSL);
        let bdsdpa_paged_pipeline = Self::make_pipeline(&device, "bdsdpa-paged", BDSDPA_PAGED_WGSL);
        let bargmax_pipeline = Self::make_pipeline(&device, "bargmax", BARGMAX_WGSL);
        let bsample_pipeline = Self::make_pipeline(&device, "bsample", BSAMPLE_WGSL);
        let btopk_pipeline = Self::make_pipeline(&device, "btopk", BTOPK_WGSL);

        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend: info.backend,
            q4k_pipeline,
            silu_mul_pipeline,
            rope_pipeline,
            sdpa_pipeline,
            norm_pipeline,
            add_pipeline,
            q6k_pipeline,
            f32_pipeline,
            argmax_pipeline,
            q4k_down_pipeline,
            q6k_down_pipeline,
            q4k_gemm_pipeline,
            q6k_gemm_pipeline,
            bnorm_pipeline,
            brope_pipeline,
            bsdpa_pipeline,
            bdsdpa_pipeline,
            bdsdpa_paged_pipeline,
            bargmax_pipeline,
            bsample_pipeline,
            btopk_pipeline,
        })
    }

    // ---- prefill (batched) op recorders; build params+bind group per call ----
    fn gemm_params(&self, n_rows: usize, n_cols: usize, m_rows: usize, acc: u32) -> (wgpu::Buffer, u32) {
        use wgpu::util::DeviceExt;
        let gx = (n_rows as u32).min(65535);
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n_rows as u32, (n_cols / 256) as u32, n_cols as u32, m_rows as u32, gx, acc, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        (buf, gx)
    }
    fn record_gemm(&self, enc: &mut wgpu::CommandEncoder, w: &ResidentWeight, x: &wgpu::Buffer, out: &wgpu::Buffer, n_cols: usize, m_rows: usize, acc: u32) {
        let n_rows = w.n_rows();
        let (pbuf, gx) = self.gemm_params(n_rows, n_cols, m_rows, acc);
        let (pipe, bg) = match w {
            ResidentWeight::Q4(w) => (&self.q4k_gemm_pipeline, self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q4k_gemm_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.w_buf), bge(1, x), bge(2, out), bge(3, &pbuf)] })),
            ResidentWeight::Q6(w) => (&self.q6k_gemm_pipeline, self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q6k_gemm_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.ql), bge(1, &w.qh), bge(2, &w.scales), bge(3, &w.d), bge(4, x), bge(5, out), bge(6, &pbuf)] })),
            ResidentWeight::F32(_) => unreachable!(),
        };
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(pipe);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(gx, (n_rows as u32).div_ceil(gx), 1);
    }
    fn record_bnorm(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, wgt: &wgpu::Buffer, y: &wgpu::Buffer, n: usize, eps: f32, m_rows: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n as u32, eps.to_bits()]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bnorm_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, x), bge(1, wgt), bge(2, y), bge(3, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bnorm_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(m_rows as u32, 1, 1);
    }
    fn record_brope(&self, enc: &mut wgpu::CommandEncoder, buf: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer, n_head: usize, head_dim: usize, m_rows: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_head as u32, head_dim as u32, m_rows as u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.brope_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, buf), bge(1, cos), bge(2, sin), bge(3, &pbuf)] });
        let total = (m_rows * n_head * (head_dim / 2)) as u32;
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.brope_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(total.div_ceil(64), 1, 1);
    }
    #[allow(clippy::too_many_arguments)]
    fn record_bsdpa(&self, enc: &mut wgpu::CommandEncoder, q: &wgpu::Buffer, kc: &wgpu::Buffer, vc: &wgpu::Buffer, out: &wgpu::Buffer, n_head: usize, n_kv_head: usize, head_dim: usize, m_rows: usize, pos: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, m_rows as u32, pos as u32, 0u32, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bsdpa_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, q), bge(1, kc), bge(2, vc), bge(3, out), bge(4, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bsdpa_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(((m_rows * n_head) as u32).div_ceil(64), 1, 1);
    }
    /// Batched DECODE SDPA: each of `m` streams attends its own KV cache.
    #[allow(clippy::too_many_arguments)]
    fn record_bdsdpa(&self, enc: &mut wgpu::CommandEncoder, q: &wgpu::Buffer, kc: &wgpu::Buffer, vc: &wgpu::Buffer, out: &wgpu::Buffer, posb: &wgpu::Buffer, slots: &wgpu::Buffer, n_head: usize, n_kv_head: usize, head_dim: usize, m: usize, max_seq: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, m as u32, max_seq as u32, 0u32, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bdsdpa_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, q), bge(1, kc), bge(2, vc), bge(3, out), bge(4, posb), bge(5, slots), bge(6, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bdsdpa_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(((m * n_head) as u32).div_ceil(64), 1, 1);
    }
    /// Paged decode SDPA: like `record_bdsdpa` but the KV is a shared block pool
    /// (`kc`/`vc`) gathered per key position through `block_table` (per slot,
    /// `max_blocks` entries). `block_size` positions per physical block.
    #[allow(clippy::too_many_arguments)]
    fn record_bdsdpa_paged(&self, enc: &mut wgpu::CommandEncoder, q: &wgpu::Buffer, kc: &wgpu::Buffer, vc: &wgpu::Buffer, out: &wgpu::Buffer, posb: &wgpu::Buffer, slots: &wgpu::Buffer, block_table: &wgpu::Buffer, n_head: usize, n_kv_head: usize, head_dim: usize, m: usize, block_size: usize, max_blocks: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, m as u32, block_size as u32, max_blocks as u32, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bdsdpa_paged_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, q), bge(1, kc), bge(2, vc), bge(3, out), bge(4, posb), bge(5, slots), bge(6, block_table), bge(7, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bdsdpa_paged_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(((m * n_head) as u32).div_ceil(64), 1, 1);
    }
    /// Batched argmax: one workgroup per stream → `out_idx[s]` (m u32 readback).
    fn record_bargmax(&self, enc: &mut wgpu::CommandEncoder, logits: &wgpu::Buffer, out_idx: &wgpu::Buffer, vocab: usize, m: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[vocab as u32, m as u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bargmax_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, logits), bge(1, out_idx), bge(2, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bargmax_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(m as u32, 1, 1);
    }
    /// Batched temperature sampling: one workgroup per stream draws from
    /// softmax(logits[s]/temp[s]) via Gumbel-max → `out_idx[s]`. `temp[s] ≤ 0`
    /// is greedy. `seed[s]` should advance each step for fresh draws.
    fn record_bsample(&self, enc: &mut wgpu::CommandEncoder, logits: &wgpu::Buffer, out_idx: &wgpu::Buffer, temp: &wgpu::Buffer, seed: &wgpu::Buffer, vocab: usize, m: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[vocab as u32, m as u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bsample_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, logits), bge(1, out_idx), bge(2, temp), bge(3, seed), bge(4, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.bsample_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(m as u32, 1, 1);
    }
    /// Batched top-K: one workgroup per stream writes its `k` largest
    /// (value, index) pairs (descending) to `vals`/`idxs` (`m*k` each).
    fn record_btopk(&self, enc: &mut wgpu::CommandEncoder, logits: &wgpu::Buffer, vals: &wgpu::Buffer, idxs: &wgpu::Buffer, vocab: usize, m: usize, k: usize) {
        use wgpu::util::DeviceExt;
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[vocab as u32, m as u32, k as u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.btopk_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, logits), bge(1, vals), bge(2, idxs), bge(3, &pbuf)] });
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&self.btopk_pipeline); p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(m as u32, 1, 1);
    }

    /// Standalone batched Q4_K GEMM (upload + dispatch + readback), for
    /// validation/benchmark. `x` is `m_rows * n_cols` (row-major); returns
    /// `m_rows * n_rows`. `n_cols` must be ≤ 2048.
    pub fn gemm_q4k_f32(&self, weight_bytes: &[u8], n_rows: usize, nb: usize, x: &[f32], m_rows: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n_cols = nb * 256;
        assert_eq!(weight_bytes.len(), n_rows * nb * 144);
        assert_eq!(x.len(), m_rows * n_cols);
        assert!(n_cols <= 2048, "Q4_K GEMM shared-mem row is 2048");
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm-w"), contents: weight_bytes, usage: wgpu::BufferUsages::STORAGE });
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm-x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE });
        let out_buf = self.alloc_activation(m_rows * n_rows, true);
        let gx = (n_rows as u32).min(65535);
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm-p"),
            contents: bytemuck::cast_slice(&[n_rows as u32, nb as u32, n_cols as u32, m_rows as u32, gx, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.q4k_gemm_pipeline.get_bind_group_layout(0), entries: &[
                bge(0, &w_buf), bge(1, &x_buf), bge(2, &out_buf), bge(3, &p_buf) ] });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.q4k_gemm_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, (n_rows as u32).div_ceil(gx), 1);
        }
        self.queue.submit([enc.finish()]);
        self.read_buffer(&out_buf, m_rows * n_rows)
    }

    fn make_pipeline(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Upload a Q4_K weight matrix to the GPU once. `weight_bytes` is
    /// row-major raw Q4_K (`n_rows * nb_per_row * 144` bytes).
    pub fn upload_q4k(&self, weight_bytes: &[u8], n_rows: usize, nb_per_row: usize) -> ResidentQ4K {
        use wgpu::util::DeviceExt;
        assert_eq!(weight_bytes.len(), n_rows * nb_per_row * 144);
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident-q4k"),
            contents: weight_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4k-params"),
            contents: bytemuck::cast_slice(&[n_rows as u32, nb_per_row as u32, (n_rows as u32).min(65535), 0u32, 0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        ResidentQ4K { w_buf, params_buf, n_rows, nb_per_row }
    }

    /// Upload Q6_K weights, repacking the 210-byte blocks to struct-of-
    /// arrays (ql/qh byte planes + f32 scales + f32 d) for aligned,
    /// straightforward shader indexing. `bytes` is `n_rows*nb_per_row*210`.
    pub fn upload_q6k(&self, bytes: &[u8], n_rows: usize, nb_per_row: usize) -> ResidentQ6K {
        use wgpu::util::DeviceExt;
        let nbk = n_rows * nb_per_row;
        assert_eq!(bytes.len(), nbk * 210, "Q6_K byte length mismatch");
        let mut ql = vec![0u8; nbk * 128];
        let mut qh = vec![0u8; nbk * 64];
        let mut scales = vec![0f32; nbk * 16];
        let mut d = vec![0f32; nbk];
        for b in 0..nbk {
            let base = b * 210;
            ql[b * 128..b * 128 + 128].copy_from_slice(&bytes[base..base + 128]);
            qh[b * 64..b * 64 + 64].copy_from_slice(&bytes[base + 128..base + 192]);
            for s in 0..16 { scales[b * 16 + s] = (bytes[base + 192 + s] as i8) as f32; }
            let dbits = u16::from_le_bytes([bytes[base + 208], bytes[base + 209]]);
            d[b] = crate::backend::candle::q4k_repack::f16_to_f32_pub(dbits);
        }
        let sto = wgpu::BufferUsages::STORAGE;
        ResidentQ6K {
            ql: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q6k-ql"), contents: &ql, usage: sto }),
            qh: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q6k-qh"), contents: &qh, usage: sto }),
            scales: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q6k-scl"), contents: bytemuck::cast_slice(&scales), usage: sto }),
            d: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q6k-d"), contents: bytemuck::cast_slice(&d), usage: sto }),
            params: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q6k-p"), contents: bytemuck::cast_slice(&[n_rows as u32, nb_per_row as u32, (n_rows as u32).min(65535), 0u32, 0u32, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST }),
            n_rows, nb_per_row,
        }
    }

    /// Record a resident Q6_K matvec: `out_buf[n_rows] = W · x_buf`.
    pub fn record_matvec_q6k(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &ResidentQ6K,
        x_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.q6k_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.ql.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.qh.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w.scales.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: w.d.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: w.params.as_entire_binding() },
            ] });
        let mut pass = enc.begin_compute_pass(&Default::default());
        let gx = (w.n_rows as u32).min(65535);
        pass.set_pipeline(&self.q6k_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gx, (w.n_rows as u32).div_ceil(gx), 1);
    }

    /// Upload a dense f32 weight matrix (`n_rows * n_cols`, row-major).
    pub fn upload_f32(&self, data: &[f32], n_rows: usize, n_cols: usize) -> ResidentF32 {
        use wgpu::util::DeviceExt;
        assert_eq!(data.len(), n_rows * n_cols);
        ResidentF32 {
            w: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("f32-w"), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE }),
            params: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("f32-p"),
                contents: bytemuck::cast_slice(&[n_rows as u32, n_cols as u32, (n_rows as u32).min(65535), 0u32, 0u32, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM }),
            n_rows,
        }
    }

    /// Standalone Q6_K matvec (upload x + dispatch + readback) for validation.
    pub fn matmul_q6k_f32(&self, bytes: &[u8], n_rows: usize, nb_per_row: usize, x: &[f32]) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let w = self.upload_q6k(bytes, n_rows, nb_per_row);
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q6k-x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE });
        let out_buf = self.alloc_activation(n_rows, true);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.record_matvec_q6k(&mut enc, &w, &x_buf, &out_buf);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&out_buf, n_rows)
    }

    /// Allocate a GPU-resident f32 activation buffer of `len` elements.
    /// `readable=true` adds COPY_SRC so the final logits can be read back.
    pub fn alloc_activation(&self, len: usize, readable: bool) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if readable { usage |= wgpu::BufferUsages::COPY_SRC; }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activation"),
            size: (len * 4) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Record a resident matvec into `enc`: `out_buf[n_rows] = W · x_buf`.
    /// Both `x_buf` (length `nb_per_row*256`) and `out_buf` (length
    /// `n_rows`) are GPU buffers — nothing returns to the CPU, so these
    /// chain directly (out of one matmul = in of the next).
    pub fn record_matvec(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &ResidentQ4K,
        x_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.q4k_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: w.params_buf.as_entire_binding() },
            ],
        });
        let gx = (w.n_rows as u32).min(65535);
        let gy = (w.n_rows as u32).div_ceil(gx);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.q4k_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// Dispatch a matvec against a prebuilt bind group: all three kernels
    /// (Q4_K / Q6_K / f32) are coalesced workgroup-per-row with a 2D grid
    /// (row = wg.x + wg.y*gx) to clear the 65535 per-dimension limit.
    fn pass_matvec(&self, enc: &mut wgpu::CommandEncoder, w: &ResidentWeight, bg: &wgpu::BindGroup) {
        let n = w.n_rows() as u32;
        let gx = n.min(65535);
        let gy = n.div_ceil(gx);
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(self.matvec_pipeline(w));
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(gx, gy, 1);
    }

    /// Build the bind group for a fused FFN down-projection (weight + gate +
    /// up + output). Q4_K: 5 bindings; Q6_K: 8.
    fn bg_down(&self, w: &ResidentWeight, gate: &wgpu::Buffer, up: &wgpu::Buffer, out: &wgpu::Buffer) -> wgpu::BindGroup {
        match w {
            ResidentWeight::Q4(w) => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q4k_down_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.w_buf), bge(1, gate), bge(2, up), bge(3, out), bge(4, &w.params_buf)] }),
            ResidentWeight::Q6(w) => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q6k_down_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.ql), bge(1, &w.qh), bge(2, &w.scales), bge(3, &w.d), bge(4, gate), bge(5, up), bge(6, out), bge(7, &w.params)] }),
            ResidentWeight::F32(_) => unreachable!("ffn_down is never f32"),
        }
    }

    /// Dispatch a fused down-projection (coalesced workgroup-per-row, 2D grid).
    fn pass_down(&self, enc: &mut wgpu::CommandEncoder, w: &ResidentWeight, bg: &wgpu::BindGroup) {
        let pipe = match w {
            ResidentWeight::Q4(_) => &self.q4k_down_pipeline,
            ResidentWeight::Q6(_) => &self.q6k_down_pipeline,
            ResidentWeight::F32(_) => unreachable!(),
        };
        let n = w.n_rows() as u32;
        let gx = n.min(65535);
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(pipe);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(gx, n.div_ceil(gx), 1);
    }

    /// Record several **independent** matvecs in ONE compute pass (no
    /// barriers between them — caller guarantees disjoint outputs / shared
    /// reads). Cuts wgpu per-pass overhead and the conservative barriers it
    /// would otherwise insert between separate passes. Used for Q/K/V (all
    /// read the normed input, write distinct buffers) and gate/up.
    fn pass_matvec_group(&self, enc: &mut wgpu::CommandEncoder, items: &[(&ResidentWeight, &wgpu::BindGroup)]) {
        let mut p = enc.begin_compute_pass(&Default::default());
        for (w, bg) in items {
            let n = w.n_rows() as u32;
            let gx = n.min(65535);
            let gy = n.div_ceil(gx);
            p.set_pipeline(self.matvec_pipeline(w));
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(gx, gy, 1);
        }
    }

    /// Read a single u32 back from a GPU buffer (blocking) — for the argmax.
    pub fn read_u32(&self, buf: &wgpu::Buffer) -> u32 {
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-u32"), size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, 4);
        self.queue.submit([enc.finish()]);
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let v = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[0];
        read_buf.unmap();
        v
    }

    /// Read a GPU f32 buffer back to the CPU (blocking). For final logits
    /// and validation; the inference path uses this once per token.
    pub fn read_buffer(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len * 4) as u64;
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();
        out
    }
}

/// List every adapter wgpu can see, as `"name [backend, device_type]"`.
/// Diagnostic for confirming the iGPU is reachable.
pub fn enumerate() -> Vec<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .iter()
        .map(|a| {
            let i = a.get_info();
            format!("{} [{:?}, {:?}]", i.name, i.backend, i.device_type)
        })
        .collect()
}

/// WGSL Q4_K mat-vec: `out[row] = sum_k dequant(W)[row,k] * x[k]`.
///
/// One invocation per output row. Reads raw Q4_K block bytes (144 B/block,
/// `repr(C)` identical to ggml `block_q4_K`) as `array<u32>` and unpacks
/// the f16 super-scales (`unpack2x16float`), the 6-bit sub-scales/mins,
/// and the 4-bit quants entirely in-shader — the same math as the CPU
/// `dequantize_q4k_block`. f32 activation, f32 accumulation.
const Q4K_MATVEC_WGSL: &str = r#"
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
const Q6K_MATVEC_WGSL: &str = r#"
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
const Q4K_DOWN_WGSL: &str = r#"
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
const Q6K_DOWN_WGSL: &str = r#"
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
const Q4K_GEMM_WGSL: &str = r#"
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
const Q6K_GEMM_WGSL: &str = r#"
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
const BNORM_WGSL: &str = r#"
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
const BROPE_WGSL: &str = r#"
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
const BSDPA_WGSL: &str = r#"
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
const BDSDPA_WGSL: &str = r#"
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
const BDSDPA_PAGED_WGSL: &str = r#"
struct BP { n_head: u32, n_kv_head: u32, head_dim: u32, m_streams: u32, block_size: u32, max_blocks: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       kc:   array<f32>;   // block pool: n_blocks*block_size*kv_dim
@group(0) @binding(2) var<storage, read>       vc:   array<f32>;
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
        let kv_base = (phys_pos * p.n_kv_head + kvh) * hd;
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

/// Batched argmax: one workgroup per stream reduces that stream's `vocab`-wide
/// logit row of `logits[M, vocab]` to its argmax → `out_idx[s]`. Lets batched
/// decode read back M u32s instead of M*128k logits.
const BARGMAX_WGSL: &str = r#"
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
const BSAMPLE_WGSL: &str = r#"
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
const BTOPK_WGSL: &str = r#"
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
const ARGMAX_WGSL: &str = r#"
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
const F32_MATVEC_WGSL: &str = r#"
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
const SILU_MUL_WGSL: &str = r#"
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
const ROPE_WGSL: &str = r#"
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
const SDPA_DECODE_WGSL: &str = r#"
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
const RMSNORM_WGSL: &str = r#"
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
const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) { a[i] = a[i] + b[i]; }
}
"#;

impl GpuContext {
    /// GQA decode attention on the GPU (standalone, for validation).
    /// `q` is the post-RoPE query (`n_head*head_dim`); `k_cache`/`v_cache`
    /// are `seq_len*n_kv_head*head_dim` (layout `[t][kvh][d]`). Returns the
    /// per-head attention output (`n_head*head_dim`), concat-heads order.
    pub fn sdpa_decode(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        assert_eq!(q.len(), n_head * head_dim);
        assert_eq!(k_cache.len(), seq_len * n_kv_head * head_dim);
        assert_eq!(v_cache.len(), seq_len * n_kv_head * head_dim);
        assert!(head_dim <= 128);
        let mk = |label, data: &[f32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
        let q_buf = mk("q", q);
        let k_buf = mk("kc", k_cache);
        let v_buf = mk("vc", v_cache);
        let out_buf = self.alloc_activation(n_head * head_dim, true);
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdpa-p"),
            contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, seq_len as u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.sdpa_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
            ] });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.sdpa_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n_head as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit([enc.finish()]);
        self.read_buffer(&out_buf, n_head * head_dim)
    }

    /// Apply interleaved RoPE to `x` (`n_head * head_dim` f32) in place for
    /// one position, using precomputed `cos`/`sin` (length `head_dim/2`).
    /// Standalone (upload + dispatch + readback) for validation; the
    /// resident path will use a `record_rope` variant.
    pub fn rope_decode(
        &self,
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        n_head: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        assert_eq!(x.len(), n_head * head_dim);
        assert_eq!(cos.len(), head_dim / 2);
        assert_eq!(sin.len(), head_dim / 2);
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope-x"), contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let cos_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cos"), contents: bytemuck::cast_slice(cos),
            usage: wgpu::BufferUsages::STORAGE });
        let sin_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sin"), contents: bytemuck::cast_slice(sin),
            usage: wgpu::BufferUsages::STORAGE });
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope-p"), contents: bytemuck::cast_slice(&[n_head as u32, head_dim as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rope_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ] });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((n_head * head_dim / 2) as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit([enc.finish()]);
        self.read_buffer(&x_buf, n_head * head_dim)
    }

    /// Record the fused `h = silu(gate) * up` over `len` elements.
    pub fn record_silu_mul(
        &self,
        enc: &mut wgpu::CommandEncoder,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        h_buf: &wgpu::Buffer,
        len: usize,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_mul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: h_buf.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.silu_mul_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((len as u32).div_ceil(64), 1, 1);
    }

    /// Whole FFN block for one token, fully GPU-resident:
    /// `out = w2( silu(w1·x) * w3·x )`. Weights stay uploaded; the
    /// `gate`/`up`/`h` intermediates never leave the GPU — one upload of
    /// `x`, one readback of `out`. This is the FFN half of a resident
    /// decode forward.
    pub fn ffn_decode(
        &self,
        x: &[f32],
        w1: &ResidentQ4K,
        w2: &ResidentQ4K,
        w3: &ResidentQ4K,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n_inter = w1.n_rows;     // intermediate size
        let n_embd = w2.n_rows;      // output size
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ffn-x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gate_buf = self.alloc_activation(n_inter, false);
        let up_buf = self.alloc_activation(n_inter, false);
        let h_buf = self.alloc_activation(n_inter, false);
        let out_buf = self.alloc_activation(n_embd, true);

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.record_matvec(&mut enc, w1, &x_buf, &gate_buf);
        self.record_matvec(&mut enc, w3, &x_buf, &up_buf);
        self.record_silu_mul(&mut enc, &gate_buf, &up_buf, &h_buf, n_inter);
        self.record_matvec(&mut enc, w2, &h_buf, &out_buf);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&out_buf, n_embd)
    }

    /// Record RMSNorm `y = rmsnorm(x) * weight` (out-of-place).
    fn record_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        y: &wgpu::Buffer,
        np_buf: &wgpu::Buffer,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.norm_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: np_buf.as_entire_binding() },
            ] });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.norm_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Record in-place residual add `a += b` over `len` elements.
    fn record_add(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, len: usize) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.add_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            ] });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.add_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((len as u32).div_ceil(64), 1, 1);
    }

    /// Standalone RMSNorm for validation (upload + dispatch + readback).
    pub fn rmsnorm_decode(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n = x.len();
        assert_eq!(weight.len(), n);
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("norm-x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE });
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("norm-w"), contents: bytemuck::cast_slice(weight), usage: wgpu::BufferUsages::STORAGE });
        let y_buf = self.alloc_activation(n, true);
        let np_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("norm-p"),
            contents: bytemuck::cast_slice(&[n as u32, eps.to_bits()]),
            usage: wgpu::BufferUsages::UNIFORM });
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.record_rmsnorm(&mut enc, &x_buf, &w_buf, &y_buf, &np_buf);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&y_buf, n)
    }

    /// Record in-place RoPE on `buf` (`n_head*head_dim`) using caller-owned
    /// cos/sin/param buffers (so they outlive the submission).
    fn record_rope(
        &self,
        enc: &mut wgpu::CommandEncoder,
        buf: &wgpu::Buffer,
        cos_buf: &wgpu::Buffer,
        sin_buf: &wgpu::Buffer,
        p_buf: &wgpu::Buffer,
        n_head: usize,
        head_dim: usize,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rope_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ] });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.rope_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((n_head * head_dim / 2) as u32).div_ceil(64), 1, 1);
    }

    /// Record GQA decode attention into `out_buf` reading the resident KV
    /// cache buffers (only the first `seq_len` positions are used).
    fn record_sdpa(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_buf: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        p_buf: &wgpu::Buffer,
        n_head: usize,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.sdpa_pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
            ] });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.sdpa_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_head as u32).div_ceil(64), 1, 1);
    }

    /// Full GPU-resident decode attention for one token at `pos`:
    /// Q/K/V proj → RoPE(Q,K) → append K,V into the resident KV cache at
    /// `pos` → GQA SDPA over positions `0..=pos` → O proj. Everything stays
    /// on the GPU in one command buffer; only `x` goes up and the result
    /// comes back. `k_cache`/`v_cache` are caller-owned resident buffers
    /// (`>= (pos+1)*n_kv_head*head_dim` f32) holding the prior positions.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        &self,
        x: &[f32],
        wq: &ResidentQ4K, wk: &ResidentQ4K, wv: &ResidentQ4K, wo: &ResidentQ4K,
        cos: &[f32], sin: &[f32],
        k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer,
        pos: usize,
        n_head: usize, n_kv_head: usize, head_dim: usize,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n_embd = wo.n_rows;
        let kv_dim = n_kv_head * head_dim;
        let seq_len = pos + 1;

        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-x"), contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE });
        let cos_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cos"), contents: bytemuck::cast_slice(cos), usage: wgpu::BufferUsages::STORAGE });
        let sin_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sin"), contents: bytemuck::cast_slice(sin), usage: wgpu::BufferUsages::STORAGE });
        let rope_q_p = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_head as u32, head_dim as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let rope_k_p = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_kv_head as u32, head_dim as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let sdpa_p = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, seq_len as u32]),
            usage: wgpu::BufferUsages::UNIFORM });

        // q_buf/k_buf must be COPY_SRC (k copied into cache); usage STORAGE for shader r/w.
        let q_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q"), size: (n_head * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let k_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k"), size: (kv_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let v_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v"), size: (kv_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let attn_buf = self.alloc_activation(n_head * head_dim, false);
        let out_buf = self.alloc_activation(n_embd, true);

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.record_matvec(&mut enc, wq, &x_buf, &q_buf);
        self.record_matvec(&mut enc, wk, &x_buf, &k_buf);
        self.record_matvec(&mut enc, wv, &x_buf, &v_buf);
        self.record_rope(&mut enc, &q_buf, &cos_buf, &sin_buf, &rope_q_p, n_head, head_dim);
        self.record_rope(&mut enc, &k_buf, &cos_buf, &sin_buf, &rope_k_p, n_kv_head, head_dim);
        // Append current K,V into the resident cache at `pos` (V is not roped).
        let off = (pos * kv_dim * 4) as u64;
        enc.copy_buffer_to_buffer(&k_buf, 0, k_cache, off, (kv_dim * 4) as u64);
        enc.copy_buffer_to_buffer(&v_buf, 0, v_cache, off, (kv_dim * 4) as u64);
        self.record_sdpa(&mut enc, &q_buf, k_cache, v_cache, &attn_buf, &sdpa_p, n_head);
        self.record_matvec(&mut enc, wo, &attn_buf, &out_buf);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&out_buf, n_embd)
    }

    /// One full transformer decode layer, GPU-resident, in a single command
    /// buffer: `x += attn(rmsnorm(x))`, then `x += ffn(rmsnorm(x))`. The
    /// residual stream `x`, the KV cache, and all intermediates stay on the
    /// GPU; only the input row goes up and the updated row comes back. This
    /// is the orchestration unit the full forward loops over.
    /// `w1`/`w2`/`w3` are gate/down/up.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_layer_once(
        &self,
        x: &[f32],
        attn_norm_w: &[f32], ffn_norm_w: &[f32],
        wq: &ResidentQ4K, wk: &ResidentQ4K, wv: &ResidentQ4K, wo: &ResidentQ4K,
        w1: &ResidentQ4K, w2: &ResidentQ4K, w3: &ResidentQ4K,
        cos: &[f32], sin: &[f32],
        k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer,
        pos: usize,
        n_head: usize, n_kv_head: usize, head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n_embd = wo.n_rows;
        let n_inter = w1.n_rows;
        let attn_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;
        let seq_len = pos + 1;
        let sto = wgpu::BufferUsages::STORAGE;
        let init = |label, data: &[f32], usage| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: bytemuck::cast_slice(data), usage });
        let uni = |data: &[u32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::UNIFORM });

        // Residual stream (read/write in place by the adds; read back at end).
        let x_buf = init("x", x, sto | wgpu::BufferUsages::COPY_SRC);
        let an_w = init("attn-norm-w", attn_norm_w, sto);
        let fn_w = init("ffn-norm-w", ffn_norm_w, sto);
        let cos_buf = init("cos", cos, sto);
        let sin_buf = init("sin", sin, sto);
        let norm_p = uni(&[n_embd as u32, eps.to_bits()]);
        let rope_q_p = uni(&[n_head as u32, head_dim as u32, 0u32, 0u32]);
        let rope_k_p = uni(&[n_kv_head as u32, head_dim as u32, 0u32, 0u32]);
        let sdpa_p = uni(&[n_head as u32, n_kv_head as u32, head_dim as u32, seq_len as u32]);

        let normed = self.alloc_activation(n_embd, false);
        let q_buf = self.alloc_activation(attn_dim, false);
        let k_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("k"),
            size: (kv_dim * 4) as u64, usage: sto | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let v_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("v"),
            size: (kv_dim * 4) as u64, usage: sto | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let attn_out = self.alloc_activation(attn_dim, false);
        let o_buf = self.alloc_activation(n_embd, false);
        let gate = self.alloc_activation(n_inter, false);
        let up = self.alloc_activation(n_inter, false);
        let h = self.alloc_activation(n_inter, false);
        let ffn_out = self.alloc_activation(n_embd, false);

        let mut enc = self.device.create_command_encoder(&Default::default());
        // --- attention sub-block: x += Wo·SDPA(RoPE(Wq·norm(x)), cache) ---
        self.record_rmsnorm(&mut enc, &x_buf, &an_w, &normed, &norm_p);
        self.record_matvec(&mut enc, wq, &normed, &q_buf);
        self.record_matvec(&mut enc, wk, &normed, &k_buf);
        self.record_matvec(&mut enc, wv, &normed, &v_buf);
        self.record_rope(&mut enc, &q_buf, &cos_buf, &sin_buf, &rope_q_p, n_head, head_dim);
        self.record_rope(&mut enc, &k_buf, &cos_buf, &sin_buf, &rope_k_p, n_kv_head, head_dim);
        let off = (pos * kv_dim * 4) as u64;
        enc.copy_buffer_to_buffer(&k_buf, 0, k_cache, off, (kv_dim * 4) as u64);
        enc.copy_buffer_to_buffer(&v_buf, 0, v_cache, off, (kv_dim * 4) as u64);
        self.record_sdpa(&mut enc, &q_buf, k_cache, v_cache, &attn_out, &sdpa_p, n_head);
        self.record_matvec(&mut enc, wo, &attn_out, &o_buf);
        self.record_add(&mut enc, &x_buf, &o_buf, n_embd);
        // --- FFN sub-block: x += W2·(silu(W1·norm(x)) * W3·norm(x)) ---
        self.record_rmsnorm(&mut enc, &x_buf, &fn_w, &normed, &norm_p);
        self.record_matvec(&mut enc, w1, &normed, &gate);
        self.record_matvec(&mut enc, w3, &normed, &up);
        self.record_silu_mul(&mut enc, &gate, &up, &h, n_inter);
        self.record_matvec(&mut enc, w2, &h, &ffn_out);
        self.record_add(&mut enc, &x_buf, &ffn_out, n_embd);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&x_buf, n_embd)
    }

    /// Run the Q4_K mat-vec on the GPU. `weight_bytes` is row-major raw
    /// Q4_K (`n_rows * nb_per_row * 144` bytes); `x` is the `nb_per_row*256`
    /// f32 activation. Returns `n_rows` f32 outputs. Blocking (uploads,
    /// dispatches, waits, reads back) — a correctness/throughput vehicle,
    /// not yet the resident-weights inference path.
    pub fn matmul_q4k_f32(
        &self,
        weight_bytes: &[u8],
        n_rows: usize,
        nb_per_row: usize,
        x: &[f32],
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        assert_eq!(weight_bytes.len(), n_rows * nb_per_row * 144);
        assert_eq!(x.len(), nb_per_row * 256);

        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4k-weights"),
            contents: weight_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (n_rows * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gx = (n_rows as u32).min(65535);
        let params = [n_rows as u32, nb_per_row as u32, gx, 0u32, 0u32, 0u32, 0u32, 0u32];
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q4k-matvec"),
            source: wgpu::ShaderSource::Wgsl(Q4K_MATVEC_WGSL.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q4k-matvec"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(gx, (n_rows as u32).div_ceil(gx), 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
        self.queue.submit([enc.finish()]);

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();
        out
    }
}

/// A resident weight in whichever form we hold it on the GPU.
pub enum ResidentWeight {
    Q4(ResidentQ4K),
    Q6(ResidentQ6K),
    F32(ResidentF32),
}

impl ResidentWeight {
    fn n_rows(&self) -> usize {
        match self {
            ResidentWeight::Q4(w) => w.n_rows,
            ResidentWeight::Q6(w) => w.n_rows,
            ResidentWeight::F32(w) => w.n_rows,
        }
    }
    fn params(&self) -> &wgpu::Buffer {
        match self {
            ResidentWeight::Q4(w) => &w.params_buf,
            ResidentWeight::Q6(w) => &w.params,
            ResidentWeight::F32(w) => &w.params,
        }
    }
}

#[inline]
fn bge(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}

impl GpuContext {
    /// Record a matvec for either quantization.
    fn record_mv(&self, enc: &mut wgpu::CommandEncoder, w: &ResidentWeight, x: &wgpu::Buffer, out: &wgpu::Buffer) {
        let bg = self.bg_matvec(w, x, out);
        self.pass_matvec(enc, w, &bg);
    }

    fn matvec_pipeline(&self, w: &ResidentWeight) -> &wgpu::ComputePipeline {
        match w {
            ResidentWeight::Q4(_) => &self.q4k_pipeline,
            ResidentWeight::Q6(_) => &self.q6k_pipeline,
            ResidentWeight::F32(_) => &self.f32_pipeline,
        }
    }

    // --- bind-group builders (called once at model load, reused every token) ---
    fn bg_matvec(&self, w: &ResidentWeight, x: &wgpu::Buffer, out: &wgpu::Buffer) -> wgpu::BindGroup {
        match w {
            ResidentWeight::Q4(w) => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q4k_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.w_buf), bge(1, x), bge(2, out), bge(3, &w.params_buf)] }),
            ResidentWeight::Q6(w) => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.q6k_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.ql), bge(1, &w.qh), bge(2, &w.scales), bge(3, &w.d), bge(4, x), bge(5, out), bge(6, &w.params)] }),
            ResidentWeight::F32(w) => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.f32_pipeline.get_bind_group_layout(0),
                entries: &[bge(0, &w.w), bge(1, x), bge(2, out), bge(3, &w.params)] }),
        }
    }
    fn bg_norm(&self, x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer, np: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
            layout: &self.norm_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, x), bge(1, w), bge(2, y), bge(3, np)] })
    }
    fn bg_rope(&self, buf: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer, p: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
            layout: &self.rope_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, buf), bge(1, cos), bge(2, sin), bge(3, p)] })
    }
    fn bg_sdpa(&self, q: &wgpu::Buffer, kc: &wgpu::Buffer, vc: &wgpu::Buffer, out: &wgpu::Buffer, p: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
            layout: &self.sdpa_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, q), bge(1, kc), bge(2, vc), bge(3, out), bge(4, p)] })
    }
    fn bg_silu(&self, gate: &wgpu::Buffer, up: &wgpu::Buffer, h: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
            layout: &self.silu_mul_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, gate), bge(1, up), bge(2, h)] })
    }
    fn bg_add(&self, a: &wgpu::Buffer, b: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
            layout: &self.add_pipeline.get_bind_group_layout(0),
            entries: &[bge(0, a), bge(1, b)] })
    }
    /// Encode one dispatch against a prebuilt bind group.
    fn pass(&self, enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wg: u32) {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups(wg, 1, 1);
    }
}

/// Cached bind groups + dispatch sizes for one layer (built once at load).
struct LayerOps {
    attn_norm: wgpu::BindGroup,
    wq: wgpu::BindGroup, wk: wgpu::BindGroup, wv: wgpu::BindGroup,
    rope_q: wgpu::BindGroup, rope_k: wgpu::BindGroup,
    sdpa: wgpu::BindGroup, wo: wgpu::BindGroup,
    ffn_norm: wgpu::BindGroup, w1: wgpu::BindGroup, w3: wgpu::BindGroup,
    w2: wgpu::BindGroup,
}

struct GpuLayer {
    attn_norm_w: wgpu::Buffer,
    ffn_norm_w: wgpu::Buffer,
    wq: ResidentWeight, wk: ResidentWeight, wv: ResidentWeight, wo: ResidentWeight,
    w1: ResidentWeight, w2: ResidentWeight, w3: ResidentWeight,
    k_cache: wgpu::Buffer, v_cache: wgpu::Buffer,
}

/// Max prompt length the cached prefill path supports. The batched GEMM's
/// per-thread `acc[8]` over a 64-wide workgroup processes up to 8*64 = 512
/// prompt rows; longer prompts must be chunked by the caller.
pub const MAX_PREFILL_M: usize = 512;

/// One cached prefill GEMM: the bind group (built once against persistent
/// scratch + shared param buffers), which pipeline it needs, and its
/// M-independent output-row count (for the dispatch grid).
struct PrefillGemm { bg: wgpu::BindGroup, q6: bool, n_rows: u32 }

/// Cached bind groups for one layer's prefill pass (built once on first
/// prefill). Mirrors `LayerOps` but for the batched [M,*] buffers.
struct PrefillLayerBg {
    attn_norm: wgpu::BindGroup,
    wq: PrefillGemm, wk: PrefillGemm, wv: PrefillGemm, wo: PrefillGemm,
    rope_q: wgpu::BindGroup, rope_k: wgpu::BindGroup, sdpa: wgpu::BindGroup,
    ffn_norm: wgpu::BindGroup, w1: PrefillGemm, w3: PrefillGemm,
    silu: wgpu::BindGroup, w2: PrefillGemm,
}

/// Everything prefill needs that can be allocated/bound ONCE and reused
/// across calls: max-M scratch buffers, the handful of shared param uniforms
/// (only `m_rows` is rewritten per call), and per-layer cached bind groups.
/// Built lazily on the first `prefill_forward` so decode-only users don't pay
/// the ~70 MB + bind-group cost.
struct PrefillCache {
    x_b: wgpu::Buffer, normed_b: wgpu::Buffer, q_b: wgpu::Buffer,
    k_b: wgpu::Buffer, v_b: wgpu::Buffer, attn_b: wgpu::Buffer,
    gate_b: wgpu::Buffer, up_b: wgpu::Buffer, h_b: wgpu::Buffer,
    cos_b: wgpu::Buffer, sin_b: wgpu::Buffer,
    // shared param uniforms (constants baked; `m_rows` rewritten each call).
    p_wq: wgpu::Buffer, p_wkv: wgpu::Buffer, p_wo: wgpu::Buffer,
    p_w13: wgpu::Buffer, p_w2: wgpu::Buffer,
    p_rope_q: wgpu::Buffer, p_rope_k: wgpu::Buffer, p_sdpa: wgpu::Buffer,
    layers: Vec<PrefillLayerBg>,
}

/// A Llama model resident on the GPU: every weight uploaded once, KV cache
/// per layer on the GPU. `forward(token, pos)` runs the entire decode token
/// in ONE command buffer (1 CPU↔GPU sync) — the design that makes GPU
/// decode beat the CPU.
pub struct GpuModel {
    ctx: GpuContext,
    layers: Vec<GpuLayer>,
    layer_ops: Vec<LayerOps>,
    final_norm_w: wgpu::Buffer,
    lm_head: ResidentWeight,
    embed: Vec<f32>,            // dequantized token-embedding table [vocab*n_embd]
    cos: Vec<f32>, sin: Vec<f32>,   // [max_seq * head_dim/2]
    // persistent per-token buffers (allocated once, content rewritten per token)
    x_buf: wgpu::Buffer, cos_buf: wgpu::Buffer, sin_buf: wgpu::Buffer,
    normed: wgpu::Buffer, q: wgpu::Buffer, k: wgpu::Buffer, v: wgpu::Buffer,
    attn_out: wgpu::Buffer, o: wgpu::Buffer,
    gate: wgpu::Buffer, up: wgpu::Buffer, h: wgpu::Buffer, ffn_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    norm_p: wgpu::Buffer, rope_q_p: wgpu::Buffer, rope_k_p: wgpu::Buffer, sdpa_p: wgpu::Buffer,
    final_norm_op: wgpu::BindGroup, lm_op: wgpu::BindGroup,
    argmax_out: wgpu::Buffer, argmax_op: wgpu::BindGroup, argmax_read: wgpu::Buffer,
    pub n_embd: usize, n_head: usize, n_kv_head: usize, head_dim: usize,
    n_inter: usize, pub vocab: usize, eps: f32,
    max_seq: usize,
    // Built lazily on the first prefill_forward (decode-only users skip it).
    prefill_cache: std::sync::OnceLock<PrefillCache>,
}

impl GpuModel {
    /// Load a GGUF Llama model onto the GPU.
    pub fn load(path: &str, ctx: GpuContext) -> Result<Self, String> {
        use candle_core::quantized::{gguf_file, GgmlDType};
        use candle_core::Device;
        let dev = Device::Cpu;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
        let mu = |k: &str| -> Result<u32, String> {
            ct.metadata.get(k).ok_or(format!("missing {k}"))?.to_u32().map_err(|e| e.to_string()) };
        let mf = |k: &str| -> Option<f32> { ct.metadata.get(k).and_then(|v| v.to_f32().ok()) };
        let n_head = mu("llama.attention.head_count")? as usize;
        let n_kv_head = mu("llama.attention.head_count_kv")? as usize;
        let n_layers = mu("llama.block_count")? as usize;
        let n_embd = mu("llama.embedding_length")? as usize;
        let eps = mf("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_base = mf("llama.rope.freq_base").unwrap_or(10000.0);
        let head_dim = n_embd / n_head;
        let max_seq = 4096usize;

        let load_w = |ctx: &GpuContext, ct: &gguf_file::Content, file: &mut std::fs::File, name: &str| -> Result<ResidentWeight, String> {
            let qt = ct.tensor(file, name, &dev).map_err(|e| e.to_string())?;
            let dims = qt.shape().dims().to_vec();
            let (rows, cols) = (dims[0], dims[1]);
            let nb = cols / 256;
            let bytes = qt.data().map_err(|e| e.to_string())?;
            match qt.dtype() {
                GgmlDType::Q4K => Ok(ResidentWeight::Q4(ctx.upload_q4k(&bytes, rows, nb))),
                GgmlDType::Q6K => Ok(ResidentWeight::Q6(ctx.upload_q6k(&bytes, rows, nb))),
                d => Err(format!("unsupported weight dtype {d:?} for {name}")),
            }
        };
        let load_norm = |ctx: &GpuContext, ct: &gguf_file::Content, file: &mut std::fs::File, name: &str| -> Result<wgpu::Buffer, String> {
            use wgpu::util::DeviceExt;
            let qt = ct.tensor(file, name, &dev).map_err(|e| e.to_string())?;
            let v: Vec<f32> = qt.dequantize(&dev).map_err(|e| e.to_string())?
                .flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
            Ok(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::STORAGE }))
        };

        // Embedding table (dequantized for lookup) doubles as the tied LM head.
        let embed_qt = ct.tensor(&mut file, "token_embd.weight", &dev).map_err(|e| e.to_string())?;
        let vocab = embed_qt.shape().dims()[0];
        let embed: Vec<f32> = embed_qt.dequantize(&dev).map_err(|e| e.to_string())?
            .flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
        // LM head (tied to the embedding). Default: the faithful Q6_K weight
        // via the coalesced Q6_K matvec — computes exactly what candle does
        // (Q6_K dequant + dot), and at 173MB it streams ~6x less than the
        // f32 path's 1GB. `ZLLM_GPU_LM_Q4=1` re-quantizes to Q4_K (smaller,
        // slight argmax drift); `ZLLM_GPU_LM_F32=1` uses the dense f32 path.
        let lm_head = if std::env::var("ZLLM_GPU_LM_Q4").is_ok() {
            let lm_q4 = candle_core::quantized::QTensor::quantize(
                &candle_core::Tensor::from_vec(embed.clone(), (vocab, n_embd), &dev).map_err(|e| e.to_string())?,
                GgmlDType::Q4K).map_err(|e| e.to_string())?;
            ResidentWeight::Q4(ctx.upload_q4k(&lm_q4.data().map_err(|e| e.to_string())?, vocab, n_embd / 256))
        } else if std::env::var("ZLLM_GPU_LM_F32").is_ok() {
            ResidentWeight::F32(ctx.upload_f32(&embed, vocab, n_embd))
        } else {
            let lm_bytes = embed_qt.data().map_err(|e| e.to_string())?;
            match embed_qt.dtype() {
                GgmlDType::Q6K => ResidentWeight::Q6(ctx.upload_q6k(&lm_bytes, vocab, n_embd / 256)),
                GgmlDType::Q4K => ResidentWeight::Q4(ctx.upload_q4k(&lm_bytes, vocab, n_embd / 256)),
                d => return Err(format!("unsupported lm_head dtype {d:?}")),
            }
        };
        let final_norm_w = load_norm(&ctx, &ct, &mut file, "output_norm.weight")?;
        let n_inter = ct.tensor(&mut file, "blk.0.ffn_gate.weight", &dev).map_err(|e| e.to_string())?.shape().dims()[0];

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            layers.push(GpuLayer {
                attn_norm_w: load_norm(&ctx, &ct, &mut file, &format!("{p}.attn_norm.weight"))?,
                ffn_norm_w: load_norm(&ctx, &ct, &mut file, &format!("{p}.ffn_norm.weight"))?,
                wq: load_w(&ctx, &ct, &mut file, &format!("{p}.attn_q.weight"))?,
                wk: load_w(&ctx, &ct, &mut file, &format!("{p}.attn_k.weight"))?,
                wv: load_w(&ctx, &ct, &mut file, &format!("{p}.attn_v.weight"))?,
                wo: load_w(&ctx, &ct, &mut file, &format!("{p}.attn_output.weight"))?,
                w1: load_w(&ctx, &ct, &mut file, &format!("{p}.ffn_gate.weight"))?,
                w2: load_w(&ctx, &ct, &mut file, &format!("{p}.ffn_down.weight"))?,
                w3: load_w(&ctx, &ct, &mut file, &format!("{p}.ffn_up.weight"))?,
                k_cache: ctx.alloc_activation(max_seq * n_kv_head * head_dim, false),
                v_cache: ctx.alloc_activation(max_seq * n_kv_head * head_dim, false),
            });
        }
        let half = head_dim / 2;
        let mut cos = vec![0f32; max_seq * half];
        let mut sin = vec![0f32; max_seq * half];
        for pos in 0..max_seq {
            for j in 0..half {
                let th = 1.0 / rope_base.powf((2 * j) as f32 / head_dim as f32);
                cos[pos * half + j] = (pos as f32 * th).cos();
                sin[pos * half + j] = (pos as f32 * th).sin();
            }
        }

        // Persistent per-token buffers + uniforms (allocated ONCE, reused
        // every token — this is what makes decode fast).
        let kv_dim = n_kv_head * head_dim;
        let attn_dim = n_head * head_dim;
        let x_buf = ctx.alloc_activation(n_embd, false);   // COPY_DST: embedding written per token
        let cos_buf = ctx.alloc_activation(half, false);
        let sin_buf = ctx.alloc_activation(half, false);
        let normed = ctx.alloc_activation(n_embd, false);
        let q = ctx.alloc_activation(attn_dim, false);
        let k = ctx.alloc_activation(kv_dim, true);        // COPY_SRC: appended to cache
        let v = ctx.alloc_activation(kv_dim, true);
        let attn_out = ctx.alloc_activation(attn_dim, false);
        let o = ctx.alloc_activation(n_embd, false);
        let gate = ctx.alloc_activation(n_inter, false);
        let up = ctx.alloc_activation(n_inter, false);
        let hh = ctx.alloc_activation(n_inter, false);
        let ffn_out = ctx.alloc_activation(n_embd, false);
        let logits = ctx.alloc_activation(vocab, true);
        use wgpu::util::DeviceExt;
        let uni_dst = |data: &[u32]| ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let norm_p = uni_dst(&[n_embd as u32, eps.to_bits()]);
        let rope_q_p = uni_dst(&[n_head as u32, head_dim as u32, 0u32, 0u32]);
        let rope_k_p = uni_dst(&[n_kv_head as u32, head_dim as u32, 0u32, 0u32]);
        let sdpa_p = uni_dst(&[n_head as u32, n_kv_head as u32, head_dim as u32, 1u32]);

        // Pre-build all bind groups (they reference the persistent buffers).
        let mut layer_ops = Vec::with_capacity(layers.len());
        for layer in &layers {
            layer_ops.push(LayerOps {
                attn_norm: ctx.bg_norm(&x_buf, &layer.attn_norm_w, &normed, &norm_p),
                wq: ctx.bg_matvec(&layer.wq, &normed, &q),
                // wk/wv write K/V straight into this layer's resident cache at
                // the current position (out_base set per token); rope_k rotates
                // the cache slot in place — removes the two copy_buffer_to_buffer.
                wk: ctx.bg_matvec(&layer.wk, &normed, &layer.k_cache),
                wv: ctx.bg_matvec(&layer.wv, &normed, &layer.v_cache),
                rope_q: ctx.bg_rope(&q, &cos_buf, &sin_buf, &rope_q_p),
                rope_k: ctx.bg_rope(&layer.k_cache, &cos_buf, &sin_buf, &rope_k_p),
                sdpa: ctx.bg_sdpa(&q, &layer.k_cache, &layer.v_cache, &attn_out, &sdpa_p),
                // wo/w2 accumulate their result directly into the residual
                // stream x_buf (fused residual add — see acc flag below),
                // removing two dispatches+barriers per layer.
                wo: ctx.bg_matvec(&layer.wo, &attn_out, &x_buf),
                ffn_norm: ctx.bg_norm(&x_buf, &layer.ffn_norm_w, &normed, &norm_p),
                w1: ctx.bg_matvec(&layer.w1, &normed, &gate),
                w3: ctx.bg_matvec(&layer.w3, &normed, &up),
                // w2 reads gate+up and computes silu(gate)*up inline (fused
                // down-projection), accumulating into x_buf — removes the
                // separate silu_mul dispatch.
                w2: ctx.bg_down(&layer.w2, &gate, &up, &x_buf),
            });
            // Flag wo/w2 to accumulate (acc=1 at param offset 12).
            ctx.queue.write_buffer(layer.wo.params(), 12, bytemuck::cast_slice(&[1u32]));
            ctx.queue.write_buffer(layer.w2.params(), 12, bytemuck::cast_slice(&[1u32]));
        }
        let final_norm_op = ctx.bg_norm(&x_buf, &final_norm_w, &normed, &norm_p);
        let lm_op = ctx.bg_matvec(&lm_head, &normed, &logits);
        let argmax_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax-out"), size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let argmax_p = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("argmax-p"), contents: bytemuck::cast_slice(&[vocab as u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let argmax_op = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &ctx.argmax_pipeline.get_bind_group_layout(0), entries: &[
                bge(0, &logits), bge(1, &argmax_out), bge(2, &argmax_p) ] });
        let argmax_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax-read"), size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        Ok(Self {
            ctx, layers, layer_ops, final_norm_w, lm_head, embed, cos, sin,
            x_buf, cos_buf, sin_buf, normed, q, k, v, attn_out, o, gate, up, h: hh, ffn_out, logits,
            norm_p, rope_q_p, rope_k_p, sdpa_p, final_norm_op, lm_op, argmax_out, argmax_op, argmax_read,
            n_embd, n_head, n_kv_head, head_dim, n_inter, vocab, eps, max_seq,
            prefill_cache: std::sync::OnceLock::new(),
        })
    }

    /// Record the whole token forward (embedding → layers → final norm → LM
    /// head, leaving logits in `self.logits`) into `enc`, writing the per-
    /// token inputs. Shared by `forward` and `forward_argmax`.
    fn record_forward(&self, enc: &mut wgpu::CommandEncoder, token: u32, pos: usize) {
        let ctx = &self.ctx;
        let (n_embd, n_head, n_kv_head, head_dim) =
            (self.n_embd, self.n_head, self.n_kv_head, self.head_dim);
        let half = head_dim / 2;
        let kv_dim = n_kv_head * head_dim;
        let seq_len = (pos + 1) as u32;

        let row = &self.embed[token as usize * n_embd..(token as usize + 1) * n_embd];
        ctx.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(row));
        ctx.queue.write_buffer(&self.cos_buf, 0, bytemuck::cast_slice(&self.cos[pos * half..pos * half + half]));
        ctx.queue.write_buffer(&self.sin_buf, 0, bytemuck::cast_slice(&self.sin[pos * half..pos * half + half]));
        ctx.queue.write_buffer(&self.sdpa_p, 0, bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, seq_len]));

        let head_wg = (n_head as u32).div_ceil(64);
        let rope_q_wg = ((n_head * head_dim / 2) as u32).div_ceil(64);
        let rope_k_wg = ((n_kv_head * head_dim / 2) as u32).div_ceil(64);

        // KV cache write offset for this position (used by wk/wv out_base and
        // rope_k base), so K/V land in the cache directly — no copies.
        let kv_base = (pos * kv_dim) as u32;
        ctx.queue.write_buffer(&self.rope_k_p, 8, bytemuck::cast_slice(&[kv_base]));

        for (layer, op) in self.layers.iter().zip(&self.layer_ops) {
            ctx.queue.write_buffer(layer.wk.params(), 16, bytemuck::cast_slice(&[kv_base]));
            ctx.queue.write_buffer(layer.wv.params(), 16, bytemuck::cast_slice(&[kv_base]));
            ctx.pass(enc, &ctx.norm_pipeline, &op.attn_norm, 1);
            ctx.pass_matvec_group(enc, &[(&layer.wq, &op.wq), (&layer.wk, &op.wk), (&layer.wv, &op.wv)]);
            ctx.pass(enc, &ctx.rope_pipeline, &op.rope_q, rope_q_wg);
            ctx.pass(enc, &ctx.rope_pipeline, &op.rope_k, rope_k_wg);
            ctx.pass(enc, &ctx.sdpa_pipeline, &op.sdpa, head_wg);
            ctx.pass_matvec(enc, &layer.wo, &op.wo);   // accumulates into x_buf
            ctx.pass(enc, &ctx.norm_pipeline, &op.ffn_norm, 1);
            ctx.pass_matvec_group(enc, &[(&layer.w1, &op.w1), (&layer.w3, &op.w3)]);
            ctx.pass_down(enc, &layer.w2, &op.w2);   // fused silu·mul + accumulate into x_buf
        }
        ctx.pass(enc, &ctx.norm_pipeline, &self.final_norm_op, 1);
        ctx.pass_matvec(enc, &self.lm_head, &self.lm_op);
    }

    /// Forward one decode token at `pos`, returning logits over the vocab
    /// (for sampling). One command buffer; KV cache for `pos` appended.
    pub fn forward(&self, token: u32, pos: usize) -> Vec<f32> {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        self.record_forward(&mut enc, token, pos);
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.read_buffer(&self.logits, self.vocab)
    }

    /// Greedy decode: forward + GPU argmax, returning the next token id.
    /// Reads back 4 bytes instead of the 128k-wide logit vector.
    pub fn forward_argmax(&self, token: u32, pos: usize) -> u32 {
        let ctx = &self.ctx;
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.record_forward(&mut enc, token, pos);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&ctx.argmax_pipeline);
            p.set_bind_group(0, &self.argmax_op, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.argmax_out, 0, &self.argmax_read, 0, 4);
        ctx.queue.submit([enc.finish()]);
        // Draining the forward explicitly, THEN mapping the tiny argmax
        // readback, is ~15% faster than letting one map+poll drain the whole
        // queue (measured). Reuse a persistent MAP_READ buffer (no per-token
        // alloc).
        ctx.device.poll(wgpu::Maintain::Wait);
        let slice = self.argmax_read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let v = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[0];
        self.argmax_read.unmap();
        v
    }

    /// Build the persistent prefill scratch buffers, shared param uniforms,
    /// and per-layer cached bind groups ONCE (lazily, on first prefill). All
    /// op shapes are identical across layers, so only a handful of distinct
    /// param uniforms are needed; only `m_rows` is rewritten per call. This
    /// removes ~208 per-call bind-group + param-buffer creations (the
    /// M-independent TTFT floor).
    fn build_prefill_cache(&self) -> PrefillCache {
        use wgpu::util::DeviceExt;
        let max_m = MAX_PREFILL_M;
        let (n_embd, n_head, n_kv_head, head_dim, n_inter) =
            (self.n_embd, self.n_head, self.n_kv_head, self.head_dim, self.n_inter);
        let kv_dim = n_kv_head * head_dim;
        let attn_dim = n_head * head_dim;
        let half = head_dim / 2;
        let dev = &self.ctx.device;

        // `readable=true` adds COPY_SRC (buffers copied out: normed_b → last-row
        // logits input; k_b/v_b → the resident KV cache).
        let mk = |len: usize, src: bool| self.ctx.alloc_activation(len, src);
        let x_b = mk(max_m * n_embd, false);
        let normed_b = mk(max_m * n_embd, true);
        let q_b = mk(max_m * attn_dim, false);
        let k_b = mk(max_m * kv_dim, true);
        let v_b = mk(max_m * kv_dim, true);
        let attn_b = mk(max_m * attn_dim, false);
        let gate_b = mk(max_m * n_inter, false);
        let up_b = mk(max_m * n_inter, false);
        let h_b = mk(max_m * n_inter, false);
        let cos_b = mk(max_m * half, false);
        let sin_b = mk(max_m * half, false);

        // Shared GEMM/RoPE/SDPA param uniforms (m_rows placeholder 0, written
        // per call). The GP layout is [n_rows, nb, n_cols, m_rows, gx, acc, _, _].
        let gemm_p = |n_rows: usize, n_cols: usize, acc: u32| {
            let gx = (n_rows as u32).min(65535);
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefill-gemm-p"),
                contents: bytemuck::cast_slice(&[n_rows as u32, (n_cols / 256) as u32, n_cols as u32, 0u32, gx, acc, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST })
        };
        let p_wq = gemm_p(attn_dim, n_embd, 0);
        let p_wkv = gemm_p(kv_dim, n_embd, 0);
        let p_wo = gemm_p(n_embd, attn_dim, 1);
        let p_w13 = gemm_p(n_inter, n_embd, 0);
        let p_w2 = gemm_p(n_embd, n_inter, 1);
        let rope_p = |nh: usize| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefill-rope-p"),
            contents: bytemuck::cast_slice(&[nh as u32, head_dim as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let p_rope_q = rope_p(n_head);
        let p_rope_k = rope_p(n_kv_head);
        let p_sdpa = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefill-sdpa-p"),
            contents: bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, 0u32, 0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });

        let bg = |pipe: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries })
        };
        let build_gemm = |w: &ResidentWeight, x: &wgpu::Buffer, out: &wgpu::Buffer, p: &wgpu::Buffer| -> PrefillGemm {
            match w {
                ResidentWeight::Q4(q) => PrefillGemm {
                    bg: bg(&self.ctx.q4k_gemm_pipeline, &[bge(0, &q.w_buf), bge(1, x), bge(2, out), bge(3, p)]),
                    q6: false, n_rows: q.n_rows as u32 },
                ResidentWeight::Q6(q) => PrefillGemm {
                    bg: bg(&self.ctx.q6k_gemm_pipeline, &[bge(0, &q.ql), bge(1, &q.qh), bge(2, &q.scales), bge(3, &q.d), bge(4, x), bge(5, out), bge(6, p)]),
                    q6: true, n_rows: q.n_rows as u32 },
                ResidentWeight::F32(_) => unreachable!("prefill weights are Q4/Q6"),
            }
        };

        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(PrefillLayerBg {
                // bnorm shares self.norm_p ([n_embd, eps] — same NP layout).
                attn_norm: bg(&self.ctx.bnorm_pipeline, &[bge(0, &x_b), bge(1, &layer.attn_norm_w), bge(2, &normed_b), bge(3, &self.norm_p)]),
                wq: build_gemm(&layer.wq, &normed_b, &q_b, &p_wq),
                wk: build_gemm(&layer.wk, &normed_b, &k_b, &p_wkv),
                wv: build_gemm(&layer.wv, &normed_b, &v_b, &p_wkv),
                rope_q: bg(&self.ctx.brope_pipeline, &[bge(0, &q_b), bge(1, &cos_b), bge(2, &sin_b), bge(3, &p_rope_q)]),
                rope_k: bg(&self.ctx.brope_pipeline, &[bge(0, &k_b), bge(1, &cos_b), bge(2, &sin_b), bge(3, &p_rope_k)]),
                sdpa: bg(&self.ctx.bsdpa_pipeline, &[bge(0, &q_b), bge(1, &layer.k_cache), bge(2, &layer.v_cache), bge(3, &attn_b), bge(4, &p_sdpa)]),
                wo: build_gemm(&layer.wo, &attn_b, &x_b, &p_wo),
                ffn_norm: bg(&self.ctx.bnorm_pipeline, &[bge(0, &x_b), bge(1, &layer.ffn_norm_w), bge(2, &normed_b), bge(3, &self.norm_p)]),
                w1: build_gemm(&layer.w1, &normed_b, &gate_b, &p_w13),
                w3: build_gemm(&layer.w3, &normed_b, &up_b, &p_w13),
                silu: bg(&self.ctx.silu_mul_pipeline, &[bge(0, &gate_b), bge(1, &up_b), bge(2, &h_b)]),
                w2: build_gemm(&layer.w2, &h_b, &x_b, &p_w2),
            });
        }
        PrefillCache {
            x_b, normed_b, q_b, k_b, v_b, attn_b, gate_b, up_b, h_b, cos_b, sin_b,
            p_wq, p_wkv, p_wo, p_w13, p_w2, p_rope_q, p_rope_k, p_sdpa, layers,
        }
    }

    /// PREFILL: process the whole prompt in ONE batched forward. Every GEMM
    /// reuses each weight row across all M prompt rows (compute-bound — the
    /// iGPU's strength), filling the resident KV cache for positions 0..M.
    /// Returns the last token's logits (the first decode step continues from
    /// position M). Supports 1..=512 tokens (the GEMM's per-thread acc[8] caps
    /// M at 512); longer prompts must be chunked by the caller. All bind groups
    /// + scratch buffers are cached (built once), so a call only rewrites
    /// `m_rows`, the embeddings, and cos/sin, then records against the cache.
    pub fn prefill_forward(&self, prompt: &[u32]) -> Vec<f32> {
        let ctx = &self.ctx;
        let (n_embd, n_head, n_kv_head, head_dim, n_inter) =
            (self.n_embd, self.n_head, self.n_kv_head, self.head_dim, self.n_inter);
        let half = head_dim / 2;
        let kv_dim = n_kv_head * head_dim;
        let m = prompt.len();
        assert!((1..=MAX_PREFILL_M).contains(&m), "prefill supports 1..={MAX_PREFILL_M} tokens (got {m})");
        let c = self.prefill_cache.get_or_init(|| self.build_prefill_cache());

        // Only m_rows varies per call (all op shapes are baked). m_rows lives at
        // u32 index 3 (byte 12) of GP/SP and index 2 (byte 8) of RP.
        let m_u32 = m as u32;
        let mb: &[u8] = bytemuck::cast_slice(std::slice::from_ref(&m_u32));
        for (p, off) in [(&c.p_wq, 12), (&c.p_wkv, 12), (&c.p_wo, 12), (&c.p_w13, 12),
                         (&c.p_w2, 12), (&c.p_rope_q, 8), (&c.p_rope_k, 8), (&c.p_sdpa, 12)] {
            ctx.queue.write_buffer(p, off, mb);
        }
        // Gather the prompt's embedding rows → batched residual stream [M, n_embd].
        let mut x_host = vec![0f32; m * n_embd];
        for (i, &tk) in prompt.iter().enumerate() {
            x_host[i * n_embd..(i + 1) * n_embd]
                .copy_from_slice(&self.embed[tk as usize * n_embd..(tk as usize + 1) * n_embd]);
        }
        ctx.queue.write_buffer(&c.x_b, 0, bytemuck::cast_slice(&x_host));
        ctx.queue.write_buffer(&c.cos_b, 0, bytemuck::cast_slice(&self.cos[0..m * half]));
        ctx.queue.write_buffer(&c.sin_b, 0, bytemuck::cast_slice(&self.sin[0..m * half]));

        // Pass recorders over cached bind groups (M-dependent dispatch sizes).
        let pass = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, b: &wgpu::BindGroup, gx: u32| {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(pipe); p.set_bind_group(0, b, &[]); p.dispatch_workgroups(gx, 1, 1);
        };
        let rec_gemm = |enc: &mut wgpu::CommandEncoder, g: &PrefillGemm| {
            let pipe = if g.q6 { &ctx.q6k_gemm_pipeline } else { &ctx.q4k_gemm_pipeline };
            let gx = g.n_rows.min(65535);
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(pipe); p.set_bind_group(0, &g.bg, &[]);
            p.dispatch_workgroups(gx, g.n_rows.div_ceil(gx), 1);
        };
        let mu = m as u32;
        let rope_q_wg = ((m * n_head * half) as u32).div_ceil(64);
        let rope_k_wg = ((m * n_kv_head * half) as u32).div_ceil(64);
        let sdpa_wg = ((m * n_head) as u32).div_ceil(64);
        let silu_wg = ((m * n_inter) as u32).div_ceil(64);

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for (li, layer) in self.layers.iter().enumerate() {
            let b = &c.layers[li];
            pass(&mut enc, &ctx.bnorm_pipeline, &b.attn_norm, mu);   // one workgroup per row
            rec_gemm(&mut enc, &b.wq);
            rec_gemm(&mut enc, &b.wk);
            rec_gemm(&mut enc, &b.wv);
            pass(&mut enc, &ctx.brope_pipeline, &b.rope_q, rope_q_wg);
            pass(&mut enc, &ctx.brope_pipeline, &b.rope_k, rope_k_wg);
            // K/V for positions 0..M land contiguously at the front of the cache
            // ([M, kv_dim] row-major == the (t*n_kv_head+kvh)*head_dim layout BSDPA reads).
            enc.copy_buffer_to_buffer(&c.k_b, 0, &layer.k_cache, 0, (m * kv_dim * 4) as u64);
            enc.copy_buffer_to_buffer(&c.v_b, 0, &layer.v_cache, 0, (m * kv_dim * 4) as u64);
            pass(&mut enc, &ctx.bsdpa_pipeline, &b.sdpa, sdpa_wg);
            rec_gemm(&mut enc, &b.wo);                               // += residual
            pass(&mut enc, &ctx.bnorm_pipeline, &b.ffn_norm, mu);
            rec_gemm(&mut enc, &b.w1);
            rec_gemm(&mut enc, &b.w3);
            pass(&mut enc, &ctx.silu_mul_pipeline, &b.silu, silu_wg);
            rec_gemm(&mut enc, &b.w2);                               // += residual
        }
        // Final norm over all rows; LM head only needs the LAST row's logits.
        ctx.record_bnorm(&mut enc, &c.x_b, &self.final_norm_w, &c.normed_b, n_embd, self.eps, m);
        enc.copy_buffer_to_buffer(&c.normed_b, ((m - 1) * n_embd * 4) as u64, &self.normed, 0, (n_embd * 4) as u64);
        ctx.pass_matvec(&mut enc, &self.lm_head, &self.lm_op);
        ctx.queue.submit([enc.finish()]);
        ctx.read_buffer(&self.logits, self.vocab)
    }

    /// Build a batched decoder: `m_max` concurrent decode streams coalesced
    /// into one forward (weights loaded once for all M = compute-bound, the
    /// regime where aggregate serving throughput can far exceed M× serialized
    /// single-stream decode). Each stream's KV lives in a paged block pool sized
    /// so every slot can reach `max_seq` (no overcommit, contiguous-equivalent).
    pub fn batched_decoder(&self, m_max: usize, max_seq: usize) -> BatchedDecoder<'_> {
        BatchedDecoder::new(self, m_max, max_seq)
    }

    /// Like `batched_decoder` but with an explicit KV block pool of `n_blocks`
    /// physical blocks (`DEFAULT_BLOCK_SIZE` positions each), shared across all
    /// slots. `n_blocks < m_max * ceil(max_seq/block_size)` overcommits: more
    /// concurrent (short) sequences than a contiguous reservation would allow,
    /// at the cost of possible pool exhaustion if many grow long (no preemption
    /// yet — admission gates on free blocks).
    pub fn batched_decoder_paged(&self, m_max: usize, max_seq: usize, n_blocks: usize) -> BatchedDecoder<'_> {
        BatchedDecoder::new_paged(self, m_max, max_seq, DEFAULT_BLOCK_SIZE, n_blocks)
    }
}

/// M concurrent decode streams in one forward. The matmuls (which dominate)
/// run once for all M streams — their weight bandwidth is amortized across the
/// batch, so this enters the compute-bound regime. Only the attention is
/// per-stream (each stream attends its own KV cache). PoC for serving-
/// throughput exploration.
/// Prompt tokens processed per batched prefill pass (one coopmat GEMM over all
/// of them, instead of one forward per token). Larger = fewer passes but more
/// scratch; 128 keeps the GEMM comfortably compute-bound.
pub const PREFILL_CHUNK: usize = 128;

/// Positions per physical KV block in the paged cache. 16 (the vLLM default)
/// trades a little block-table indirection for fine-grained allocation.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Candidate pool size for top-k/top-p sampling: the GPU extracts each stream's
/// top-`TOPK_K` logits and the CPU samples within them. Caps top_k and the
/// top_p nucleus at 64 (covers normal distributions; flatter ones truncate).
pub const TOPK_K: usize = 64;

pub struct BatchedDecoder<'a> {
    model: &'a GpuModel,
    m_max: usize,
    max_seq: usize,
    /// Row capacity of the per-token scratch buffers = max(m_max, PREFILL_CHUNK).
    /// Decode uses ≤ m_max rows; prefill processes up to this many prompt tokens
    /// per pass, so the scratch is sized for the larger of the two.
    row_cap: usize,
    x_b: wgpu::Buffer, normed_b: wgpu::Buffer, q_b: wgpu::Buffer,
    k_b: wgpu::Buffer, v_b: wgpu::Buffer, attn_b: wgpu::Buffer,
    gate_b: wgpu::Buffer, up_b: wgpu::Buffer, h_b: wgpu::Buffer,
    cos_b: wgpu::Buffer, sin_b: wgpu::Buffer, logits_b: wgpu::Buffer,
    pos_buf: wgpu::Buffer, slots_buf: wgpu::Buffer, argmax_out: wgpu::Buffer, argmax_read: wgpu::Buffer,
    // Per-stream sampling params (temperature ≤0 = greedy; seed advances per step).
    temp_buf: wgpu::Buffer, seed_buf: wgpu::Buffer,
    // Top-K candidates (m_max × TOPK_K) for CPU-side top-k/top-p sampling.
    topk_vals: wgpu::Buffer, topk_idxs: wgpu::Buffer,
    // Paged KV: a shared pool of physical blocks (n_blocks × block_size positions
    // each) per layer, plus a per-slot block table mapping logical → physical
    // blocks. Decouples a sequence's KV from a contiguous max_seq reservation.
    block_size: usize,
    n_blocks: usize,
    max_blocks_per_seq: usize,
    blocks: std::cell::RefCell<BlockState>,
    block_table_buf: wgpu::Buffer,
    k_pool: Vec<wgpu::Buffer>, v_pool: Vec<wgpu::Buffer>,
}

/// Host-side bookkeeping for the paged KV pool, including the cross-request
/// prefix cache (blocks are reference-counted; a finished sequence's full prefix
/// blocks stay registered for reuse and are reclaimed LRU-style only when the
/// pool needs space).
struct BlockState {
    /// Truly-free physical block indices (a stack).
    free: Vec<u32>,
    /// `slot_blocks[slot]` = physical blocks owned by that slot, in logical order.
    slot_blocks: Vec<Vec<u32>>,
    /// Flattened block table `[m_max * max_blocks_per_seq]` uploaded to the GPU:
    /// `table_host[slot*max_blocks_per_seq + logical] = physical block`.
    table_host: Vec<u32>,
    /// Per physical block: number of sequences referencing it. 0 = reclaimable.
    refcount: Vec<u32>,
    /// Prefix hash → physical block holding that prefix's KV (full blocks only).
    cache_map: std::collections::HashMap<u64, u32>,
    /// Physical block → its registered prefix hash (to drop the map entry on reclaim).
    block_hash: Vec<Option<u64>>,
    /// Per physical block: the `clock` tick of its last cache use (register or
    /// reuse). Used to reclaim the LEAST-recently-used cached block first so a hot
    /// shared prefix (low-index, allocated early) survives churn from cold one-offs.
    last_use: Vec<u64>,
    /// Monotonic logical clock, bumped on every cache touch (LRU ordering key).
    clock: u64,
    /// A/B: reclaim by lowest index (old behaviour) instead of LRU. Set from
    /// `ZLLM_FIRST_FIT` at construction; tests flip it directly to compare.
    first_fit: bool,
    /// Prefix-cache stats, in blocks (reused vs freshly prefilled).
    hits: u64,
    misses: u64,
}

impl BlockState {
    /// Allocate one physical block: prefer a truly-free block; else reclaim the
    /// LEAST-recently-used unreferenced (refcount 0) cached block, dropping its
    /// prefix-cache entry. Returns None only if every block is currently referenced.
    fn alloc(&mut self) -> Option<u32> {
        if let Some(b) = self.free.pop() {
            self.refcount[b as usize] = 1;
            return Some(b);
        }
        // LRU reclaim: among reclaimable cached blocks pick the one with the
        // smallest last-use tick (first-fit by index would evict hot low-index
        // prefix blocks before cold high-index one-offs — exactly backwards).
        let reclaimable = (0..self.refcount.len())
            .filter(|&i| self.refcount[i] == 0 && self.block_hash[i].is_some());
        let reclaim = if self.first_fit { reclaimable.min() } else { reclaimable.min_by_key(|&i| self.last_use[i]) };
        if let Some(i) = reclaim {
            if let Some(h) = self.block_hash[i].take() { self.cache_map.remove(&h); }
            self.refcount[i] = 1;
            return Some(i as u32);
        }
        None
    }

    /// Stamp `phys` as just-used for LRU ordering (bump the logical clock).
    fn touch(&mut self, phys: u32) {
        self.clock += 1;
        self.last_use[phys as usize] = self.clock;
    }
}

/// Cumulative prefix hash: combine the previous block's hash with this block's
/// tokens. Two prompts with an identical token prefix get identical per-block
/// hashes (deterministic — DefaultHasher uses fixed keys).
fn prefix_block_hash(prev: u64, toks: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prev.hash(&mut h);
    toks.hash(&mut h);
    h.finish()
}

impl<'a> BatchedDecoder<'a> {
    fn new(model: &'a GpuModel, m_max: usize, max_seq: usize) -> Self {
        // Full pool: every slot can reach max_seq, so admission never starves
        // (contiguous-equivalent memory, paging mechanism exercised + validated).
        let n_blocks = m_max * max_seq.div_ceil(DEFAULT_BLOCK_SIZE);
        Self::new_paged(model, m_max, max_seq, DEFAULT_BLOCK_SIZE, n_blocks)
    }

    fn new_paged(model: &'a GpuModel, m_max: usize, max_seq: usize, block_size: usize, n_blocks: usize) -> Self {
        let ctx = &model.ctx;
        let (n_embd, n_head, n_kv_head, head_dim, n_inter, vocab) =
            (model.n_embd, model.n_head, model.n_kv_head, model.head_dim, model.n_inter, model.vocab);
        let kv_dim = n_kv_head * head_dim;
        let attn_dim = n_head * head_dim;
        let half = head_dim / 2;
        let max_blocks_per_seq = max_seq.div_ceil(block_size);
        assert!(n_blocks >= max_blocks_per_seq, "pool ({n_blocks} blocks) can't hold even one full sequence ({max_blocks_per_seq})");
        // Per-token scratch is sized for the larger of decode batch (m_max) and
        // a prefill chunk, so one prefill pass can process up to `row_cap` prompt
        // tokens. logits/argmax stay at m_max: lm_head runs ≤ m_max rows for
        // decode and exactly 1 row for prefill (only the last token's logits).
        let row_cap = m_max.max(PREFILL_CHUNK);
        let a = |len: usize| ctx.alloc_activation(len, false);
        let asrc = |len: usize| ctx.alloc_activation(len, true); // COPY_SRC: cache scatter / readback
        // Shared KV block pool: n_blocks × (block_size positions) × kv_dim, per layer.
        // COPY_SRC (asrc) so blocks can be copied out to host for swap-to-host preemption.
        let k_pool = (0..model.layers.len()).map(|_| asrc(n_blocks * block_size * kv_dim)).collect();
        let v_pool = (0..model.layers.len()).map(|_| asrc(n_blocks * block_size * kv_dim)).collect();
        let blocks = BlockState {
            free: (0..n_blocks as u32).rev().collect(),
            slot_blocks: vec![Vec::new(); m_max],
            table_host: vec![0u32; m_max * max_blocks_per_seq],
            refcount: vec![0u32; n_blocks],
            cache_map: std::collections::HashMap::new(),
            block_hash: vec![None; n_blocks],
            last_use: vec![0u64; n_blocks],
            clock: 0,
            first_fit: std::env::var("ZLLM_FIRST_FIT").is_ok(),
            hits: 0, misses: 0,
        };
        Self {
            model, m_max, max_seq, row_cap,
            x_b: a(row_cap * n_embd), normed_b: asrc(row_cap * n_embd), q_b: asrc(row_cap * attn_dim),
            k_b: asrc(row_cap * kv_dim), v_b: asrc(row_cap * kv_dim), attn_b: a(row_cap * attn_dim),
            gate_b: a(row_cap * n_inter), up_b: a(row_cap * n_inter), h_b: a(row_cap * n_inter),
            cos_b: a(row_cap * half), sin_b: a(row_cap * half), logits_b: a(m_max * vocab),
            pos_buf: a(row_cap), slots_buf: a(row_cap), temp_buf: a(m_max), seed_buf: a(m_max),
            topk_vals: asrc(m_max * TOPK_K), topk_idxs: asrc(m_max * TOPK_K), argmax_out: asrc(m_max),
            argmax_read: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bd-argmax-read"), size: (m_max * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            block_size, n_blocks, max_blocks_per_seq,
            blocks: std::cell::RefCell::new(blocks),
            block_table_buf: a(m_max * max_blocks_per_seq),
            k_pool, v_pool,
        }
    }

    /// Ensure `slot` owns enough physical blocks to hold `n_positions` positions,
    /// updating the host block table. Newly-allocated blocks come from the free
    /// list or a reclaimed unreferenced cached block. Returns false if the pool
    /// is exhausted (every block referenced).
    fn ensure_blocks(&self, slot: u32, n_positions: usize) -> bool {
        let need = n_positions.div_ceil(self.block_size);
        let mut bs = self.blocks.borrow_mut();
        let have = bs.slot_blocks[slot as usize].len();
        if need <= have { return true; }
        for lb in have..need {
            let Some(phys) = bs.alloc() else { return false; };
            bs.slot_blocks[slot as usize].push(phys);
            bs.table_host[slot as usize * self.max_blocks_per_seq + lb] = phys;
        }
        true
    }

    /// Drop `slot`'s references to its blocks (called on evict). A block whose
    /// refcount hits 0 and is NOT a registered prefix block returns to the free
    /// list; registered (cached) blocks stay for cross-request reuse and are
    /// reclaimed only under pool pressure (see `BlockState::alloc`).
    fn free_slot(&self, slot: u32) {
        let mut bs = self.blocks.borrow_mut();
        let owned = std::mem::take(&mut bs.slot_blocks[slot as usize]);
        for phys in owned {
            let rc = &mut bs.refcount[phys as usize];
            if *rc > 0 { *rc -= 1; }
            if bs.refcount[phys as usize] == 0 && bs.block_hash[phys as usize].is_none() {
                bs.free.push(phys);
            }
        }
    }

    /// Blocks available for a fresh allocation right now (free + reclaimable cached).
    fn free_blocks(&self) -> usize {
        let bs = self.blocks.borrow();
        bs.free.len() + bs.refcount.iter().zip(&bs.block_hash).filter(|(rc, h)| **rc == 0 && h.is_some()).count()
    }

    /// Whether the pool can currently fit `n_positions` of fresh KV.
    fn can_fit(&self, n_positions: usize) -> bool { self.free_blocks() >= n_positions.div_ceil(self.block_size) }

    /// How many MORE blocks `slot` needs to hold `n_positions` (0 if it already does).
    fn blocks_short(&self, slot: u32, n_positions: usize) -> usize {
        let bs = self.blocks.borrow();
        n_positions.div_ceil(self.block_size).saturating_sub(bs.slot_blocks[slot as usize].len())
    }

    /// Prefix-cache stats since construction: (reused blocks, freshly-prefilled blocks).
    fn cache_stats(&self) -> (u64, u64) { let bs = self.blocks.borrow(); (bs.hits, bs.misses) }

    /// Physical position (into the block pool, in units of kv_dim) of `pos` for `slot`.
    fn phys_pos(&self, slot: u32, pos: usize) -> usize {
        let bs = self.blocks.borrow();
        let phys_block = bs.slot_blocks[slot as usize][pos / self.block_size];
        phys_block as usize * self.block_size + pos % self.block_size
    }

    fn upload_block_table(&self) {
        let bs = self.blocks.borrow();
        self.model.ctx.queue.write_buffer(&self.block_table_buf, 0, bytemuck::cast_slice(&bs.table_host));
    }

    /// SWAP-OUT (swap-to-host preemption): copy `slot`'s KV for positions
    /// `0..n_pos` from the GPU block pool into a host blob, then free the slot's
    /// blocks. The classic vLLM alternative to recompute-preemption (restore the
    /// KV instead of re-running the forward). Restored bit-exactly by `swap_in`.
    /// NOTE: measured SLOWER than recompute on this box (see `gpu_swap_preemption`)
    /// — it's a discrete-GPU lever (small VRAM pool + large separate host RAM over
    /// PCIe); on UMA host==device memory, so it saves nothing a bigger pool wouldn't
    /// and only adds scatter/gather + mapped-readback overhead. Default-off
    /// (`ZLLM_SWAP`); kept for VRAM-constrained (non-UMA) deployments.
    fn swap_out(&self, slot: u32, n_pos: usize) -> HostKv {
        let kv_dim = self.model.n_kv_head * self.model.head_dim;
        let nb = n_pos.div_ceil(self.block_size);           // logical blocks holding the KV
        let blk = self.block_size * kv_dim;                 // f32 per block
        let blkb = (blk * 4) as u64;                        // bytes per block
        let nl = self.model.layers.len();
        let phys: Vec<u32> = { let bs = self.blocks.borrow(); bs.slot_blocks[slot as usize][..nb].to_vec() };
        let half = nl * nb * blk;                           // f32 in the K (or V) half
        let ctx = &self.model.ctx;
        // One MAP_READ staging buffer; gather scattered pool blocks into it (K then V).
        let stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv-swap-out"), size: (half * 2 * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for li in 0..nl {
            for (lb, &pb) in phys.iter().enumerate() {
                let src = pb as u64 * blkb;
                let dst = ((li * nb + lb) * blk) as u64 * 4;
                enc.copy_buffer_to_buffer(&self.k_pool[li], src, &stage, dst, blkb);
                enc.copy_buffer_to_buffer(&self.v_pool[li], src, &stage, (half * 4) as u64 + dst, blkb);
            }
        }
        ctx.queue.submit([enc.finish()]);
        let slice = stage.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let all: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        stage.unmap();
        let (k, v) = (all[..half].to_vec(), all[half..].to_vec());
        self.free_slot(slot);                               // release the blocks for others
        HostKv { n_pos, nb, k, v }
    }

    /// SWAP-IN: allocate fresh blocks for `slot` and write `host`'s saved KV back
    /// into them (host→device via the queue). The KV bytes are identical to before
    /// swap-out, so a subsequent decode step is bit-exact to never preempting.
    /// Returns false if the pool can't supply the blocks (caller checks `can_fit`).
    fn swap_in(&self, slot: u32, host: &HostKv) -> bool {
        if !self.ensure_blocks(slot, host.n_pos) { return false; }
        let kv_dim = self.model.n_kv_head * self.model.head_dim;
        let blk = self.block_size * kv_dim;
        let blkb = (blk * 4) as u64;
        let nl = self.model.layers.len();
        let nb = host.nb;
        let half = nl * nb * blk;
        let phys: Vec<u32> = { let bs = self.blocks.borrow(); bs.slot_blocks[slot as usize][..nb].to_vec() };
        let ctx = &self.model.ctx;
        // Upload the whole packed blob ONCE via a mapped staging buffer (no per-block
        // write_buffer — that costs a staging-belt write each call), then scatter to
        // the pool blocks with GPU-side copies in one submit.
        let stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv-swap-in"), size: (half * 2 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
        {
            let mut mapped = stage.slice(..).get_mapped_range_mut();
            let f: &mut [f32] = bytemuck::cast_slice_mut(&mut mapped);
            f[..half].copy_from_slice(&host.k);
            f[half..].copy_from_slice(&host.v);
        }
        stage.unmap();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for li in 0..nl {
            for (lb, &pb) in phys.iter().enumerate() {
                let soff = ((li * nb + lb) * blk) as u64 * 4;
                let dst = pb as u64 * blkb;
                enc.copy_buffer_to_buffer(&stage, soff, &self.k_pool[li], dst, blkb);
                enc.copy_buffer_to_buffer(&stage, (half * 4) as u64 + soff, &self.v_pool[li], dst, blkb);
            }
        }
        ctx.queue.submit([enc.finish()]);
        true
    }

    /// One batched step over the active streams 0..m (cache slots 0..m).
    pub fn step(&self, tokens: &[u32], positions: &[u32]) -> Vec<u32> {
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        self.step_slotted(tokens, positions, &slots)
    }

    /// One batched step: batch position i decodes `tokens[i]` at `positions[i]`
    /// using KV cache **slot** `slots[i]` — writes its K/V to slot[i]'s cache at
    /// positions[i], attends 0..=positions[i] — and returns the greedy next
    /// token per batch position. Decoupling slot from batch position lets a
    /// sequence keep its KV across admission/eviction (continuous batching).
    pub fn step_slotted(&self, tokens: &[u32], positions: &[u32], slots: &[u32]) -> Vec<u32> {
        self.step_core(tokens, positions, slots, None)
    }

    /// Like `step_slotted` but draws each next token from softmax(logits/temp[s])
    /// with per-stream `temps`/`seeds` (temp ≤ 0 → greedy). Same forward; only the
    /// final per-stream reduction changes (argmax → Gumbel-max sample).
    pub fn step_slotted_sample(&self, tokens: &[u32], positions: &[u32], slots: &[u32], temps: &[f32], seeds: &[u32]) -> Vec<u32> {
        assert!(temps.len() == tokens.len() && seeds.len() == tokens.len());
        self.step_core(tokens, positions, slots, Some((temps, seeds)))
    }

    /// Decode `m` rows through the full forward into `logits_b[m, vocab]`,
    /// returning the encoder (selection pass + submit done by the caller). Shared
    /// by the argmax/Gumbel path and the top-K path.
    fn forward_logits(&self, tokens: &[u32], positions: &[u32], slots: &[u32]) -> wgpu::CommandEncoder {
        let ctx = &self.model.ctx;
        let m = tokens.len();
        let (n_embd, eps) = (self.model.n_embd, self.model.eps);
        self.write_inputs(tokens, positions, slots);
        let phys = self.prepare_paging(m, positions, slots);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.record_layers(&mut enc, m, &phys);
        ctx.record_bnorm(&mut enc, &self.x_b, &self.model.final_norm_w, &self.normed_b, n_embd, eps, m);
        ctx.record_gemm(&mut enc, &self.model.lm_head, &self.normed_b, &self.logits_b, n_embd, m, 0);
        enc
    }

    fn step_core(&self, tokens: &[u32], positions: &[u32], slots: &[u32], sampling: Option<(&[f32], &[u32])>) -> Vec<u32> {
        let ctx = &self.model.ctx;
        let m = tokens.len();
        assert!(m <= self.m_max && m == positions.len() && m == slots.len() && m >= 1);
        let mut enc = self.forward_logits(tokens, positions, slots);
        self.record_select(&mut enc, m, sampling);
        enc.copy_buffer_to_buffer(&self.argmax_out, 0, &self.argmax_read, 0, (m * 4) as u64);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::Maintain::Wait);
        let slice = self.argmax_read.slice(..(m * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
        self.argmax_read.unmap();
        out
    }

    /// Decode step with top-k / top-p sampling: the GPU extracts each stream's
    /// top-`TOPK_K` logits, the CPU applies `params[s]` (temperature, top-k,
    /// top-p) and samples with `seeds[s]`. `params`/`seeds` are per stream.
    pub fn step_slotted_topk(&self, tokens: &[u32], positions: &[u32], slots: &[u32], params: &[SamplingParams], seeds: &[u32]) -> Vec<u32> {
        let ctx = &self.model.ctx;
        let m = tokens.len();
        assert!(m <= self.m_max && m == positions.len() && m == slots.len() && params.len() == m && seeds.len() == m && m >= 1);
        let k = TOPK_K;
        let mut enc = self.forward_logits(tokens, positions, slots);
        ctx.record_btopk(&mut enc, &self.logits_b, &self.topk_vals, &self.topk_idxs, self.model.vocab, m, k);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::Maintain::Wait);
        let vals = ctx.read_buffer(&self.topk_vals, m * k);
        let idxs: Vec<u32> = ctx.read_buffer(&self.topk_idxs, m * k).iter().map(|f| f.to_bits()).collect();
        (0..m).map(|s| {
            let o = s * k;
            sample_topk(&vals[o..o + k], &idxs[o..o + k], params[s], seeds[s])
        }).collect()
    }

    /// Record the final per-stream token selection over `logits_b[m, vocab]`:
    /// greedy argmax (None) or temperature sampling (Some((temps, seeds))).
    fn record_select(&self, enc: &mut wgpu::CommandEncoder, m: usize, sampling: Option<(&[f32], &[u32])>) {
        let ctx = &self.model.ctx;
        let vocab = self.model.vocab;
        match sampling {
            None => ctx.record_bargmax(enc, &self.logits_b, &self.argmax_out, vocab, m),
            Some((temps, seeds)) => {
                ctx.queue.write_buffer(&self.temp_buf, 0, bytemuck::cast_slice(temps));
                ctx.queue.write_buffer(&self.seed_buf, 0, bytemuck::cast_slice(seeds));
                ctx.record_bsample(enc, &self.logits_b, &self.argmax_out, &self.temp_buf, &self.seed_buf, vocab, m);
            }
        }
    }

    /// Batched prefill of a whole prompt into one KV cache `slot`, in a single
    /// forward per `PREFILL_CHUNK` tokens instead of one forward per token. The
    /// trick: feed the prompt's tokens as `m` rows with staggered positions
    /// `start..end` all pointing at `slot`; the decode SDPA then has row `i`
    /// attend the slot's cache `0..=position[i]` — which IS causal prefill
    /// attention. K/V for every prompt position is written to the slot, so the
    /// sequence can decode straight from `prompt.len()`. Returns the greedy first
    /// generated token (argmax at the last prompt position). Long prompts are
    /// chunked; chunk k attends all earlier chunks' K/V already resident in slot.
    pub fn prefill_slot(&self, prompt: &[u32], slot: u32) -> u32 {
        self.prefill_slot_from(prompt, slot, 0, None)
    }

    /// Prefix-cached prefill: reuse already-resident KV blocks for the longest
    /// shared full-block prefix of `prompt` (e.g. a system prompt repeated across
    /// requests), and only prefill the suffix. Returns (first_token, cached_len)
    /// where `cached_len` positions were served from cache (0 = cold). The reused
    /// blocks' KV is bit-identical to recomputing them (KV at position p depends
    /// only on tokens[0..=p]), so output is unchanged — just less prefill compute.
    pub fn prefill_slot_cached(&self, prompt: &[u32], slot: u32, samp: Option<(f32, u32)>) -> (u32, usize) {
        assert!(!prompt.is_empty() && prompt.len() <= self.max_seq);
        let (cached_len, cached_blocks) = self.prefill_prefix_reuse(prompt, slot);
        let first = self.prefill_slot_from(prompt, slot, cached_len, samp);
        self.prefill_register(prompt, slot, cached_blocks);
        (first, cached_len)
    }

    /// Reuse the longest cached full-block prefix of `prompt` for `slot` (refcount++
    /// each reused block, point the slot's block table at it). Returns
    /// (cached_len, cached_blocks). The block holding the last position is never
    /// reused (its logits are needed), so cached_len < prompt.len().
    fn prefill_prefix_reuse(&self, prompt: &[u32], slot: u32) -> (usize, usize) {
        let bsz = self.block_size;
        let max_reuse = (prompt.len() - 1) / bsz;
        let mut cached_blocks = 0usize;
        let mut bs = self.blocks.borrow_mut();
        let mut prev_h = 0u64;
        for lb in 0..max_reuse {
            let h = prefix_block_hash(prev_h, &prompt[lb * bsz..(lb + 1) * bsz]);
            prev_h = h;
            match bs.cache_map.get(&h).copied() {
                Some(phys) => {
                    bs.refcount[phys as usize] += 1;
                    bs.touch(phys); // LRU: a reused prefix block is hot — keep it
                    bs.slot_blocks[slot as usize].push(phys);
                    bs.table_host[slot as usize * self.max_blocks_per_seq + lb] = phys;
                    cached_blocks += 1;
                }
                None => break,
            }
        }
        bs.hits += cached_blocks as u64;
        (cached_blocks * bsz, cached_blocks)
    }

    /// Register `slot`'s newly-filled FULL blocks (logical `cached_blocks..n_full`)
    /// in the prefix cache so later prompts can reuse them. Call once prefill is
    /// complete (the blocks must be written).
    fn prefill_register(&self, prompt: &[u32], slot: u32, cached_blocks: usize) {
        let bsz = self.block_size;
        let n_full = prompt.len() / bsz;
        if cached_blocks >= n_full { return; }
        let mut bs = self.blocks.borrow_mut();
        let mut prev_h = 0u64;
        for lb in 0..cached_blocks { prev_h = prefix_block_hash(prev_h, &prompt[lb * bsz..(lb + 1) * bsz]); }
        bs.misses += (n_full - cached_blocks) as u64;
        for lb in cached_blocks..n_full {
            let h = prefix_block_hash(prev_h, &prompt[lb * bsz..(lb + 1) * bsz]);
            prev_h = h;
            let phys = bs.slot_blocks[slot as usize][lb];
            if !bs.cache_map.contains_key(&h) { bs.cache_map.insert(h, phys); bs.block_hash[phys as usize] = Some(h); bs.touch(phys); }
        }
    }

    /// Number of prompt tokens prefilled per `prefill_chunk` call.
    pub fn prefill_chunk_size(&self) -> usize { self.row_cap }

    /// Max context length per sequence (positions).
    pub fn max_seq_len(&self) -> usize { self.max_seq }

    /// Single-sequence **Prompt-Lookup speculative decode** (greedy) of up to
    /// `max_new` tokens from `prompt`. Each step finds an n-gram draft from the
    /// generated history (`lookup_len`-gram, up to `draft_k` tokens) and verifies
    /// it in ONE batched forward — `step_slotted` over `[next, draft…]` at
    /// consecutive positions in slot 0 — committing every token the model agrees
    /// with. Output is IDENTICAL to greedy single-token decode (verification only
    /// commits the model's own argmaxes), but at >1 token/forward on echo-heavy
    /// text: the weight stream is amortized across the verified positions, beating
    /// the 1-token/forward bandwidth roofline. Rejected drafts' KV is overwritten
    /// next step (SDPA reads only 0..=pos). Returns (tokens, forwards).
    pub fn generate_pld(&self, prompt: &[u32], max_new: usize, eos: u32, lookup_len: usize, draft_k: usize) -> (Vec<u32>, usize) {
        assert!(draft_k + 1 <= self.m_max, "draft_k+1 ({}) must be <= m_max ({})", draft_k + 1, self.m_max);
        let slot = 0u32;
        let g0 = self.prefill_slot(prompt, slot);
        let mut hist: Vec<u32> = prompt.to_vec();
        hist.push(g0);
        let mut produced = vec![g0];
        let mut next = g0;
        let mut pos = prompt.len(); // position where `next` will write its KV
        let mut forwards = 0usize;
        while produced.len() < max_new && next != eos {
            let k = draft_k.min(max_new - produced.len());
            match crate::engine::spec_decode::lookup_draft_best(&hist, &hist, lookup_len, k) {
                Some(d) => {
                    // Verify [next, draft…] over consecutive positions in one pass.
                    let mut inp = Vec::with_capacity(d.len() + 1);
                    inp.push(next);
                    inp.extend_from_slice(&d);
                    let positions: Vec<u32> = (0..inp.len()).map(|i| (pos + i) as u32).collect();
                    let slots = vec![slot; inp.len()];
                    let outs = self.step_slotted(&inp, &positions, &slots); // argmax per position
                    forwards += 1;
                    // out[0] is the real next token; draft d[i] is accepted iff out[i]==d[i].
                    let mut accepted = 0usize;
                    while accepted < d.len() && outs[accepted] == d[accepted] { accepted += 1; }
                    // Commit out[0..=accepted] (accepted drafts + the bonus/correction).
                    for &tok in outs.iter().take(accepted + 1) {
                        produced.push(tok); hist.push(tok); next = tok;
                        if tok == eos || produced.len() >= max_new { break; }
                    }
                    pos += accepted + 1; // KV valid through pos+accepted; bonus feeds next
                }
                None => {
                    let outs = self.step_slotted(&[next], &[pos as u32], &[slot]);
                    forwards += 1;
                    let tok = outs[0];
                    produced.push(tok); hist.push(tok); next = tok; pos += 1;
                }
            }
        }
        (produced, forwards)
    }

    /// Prefill ONE chunk — positions `[start .. min(start+chunk_size, len))` — of
    /// `prompt` into `slot` (the prefix `[0..start)` must already be resident).
    /// Returns (new_start, first_token_if_complete). When the chunk reaches the
    /// prompt's end it also runs the lm_head and returns the first generated token.
    /// Lets the scheduler interleave a long prompt's prefill with decode steps.
    pub fn prefill_chunk(&self, prompt: &[u32], slot: u32, start: usize, samp: Option<(f32, u32)>) -> (usize, Option<u32>) {
        let ctx = &self.model.ctx;
        let p = prompt.len();
        assert!(start < p);
        let (n_embd, eps) = (self.model.n_embd, self.model.eps);
        let end = (start + self.row_cap).min(p);
        let m = end - start;
        let toks = &prompt[start..end];
        let positions: Vec<u32> = (start..end).map(|x| x as u32).collect();
        let slots = vec![slot; m];
        self.write_inputs(toks, &positions, &slots);
        let phys = self.prepare_paging(m, &positions, &slots);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.record_layers(&mut enc, m, &phys);
        let is_last = end == p;
        let (tarr, sarr): ([f32; 1], [u32; 1]);
        let sampling = match samp { Some((t, s)) => { (tarr, sarr) = ([t], [s]); Some((&tarr[..], &sarr[..])) } None => None };
        if is_last {
            ctx.record_bnorm(&mut enc, &self.x_b, &self.model.final_norm_w, &self.normed_b, n_embd, eps, m);
            let src = if m > 1 {
                enc.copy_buffer_to_buffer(&self.normed_b, ((m - 1) * n_embd * 4) as u64, &self.q_b, 0, (n_embd * 4) as u64);
                &self.q_b
            } else { &self.normed_b };
            ctx.record_gemm(&mut enc, &self.model.lm_head, src, &self.logits_b, n_embd, 1, 0);
            self.record_select(&mut enc, 1, sampling);
            enc.copy_buffer_to_buffer(&self.argmax_out, 0, &self.argmax_read, 0, 4);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::Maintain::Wait);
        let g0 = if is_last {
            let slice = self.argmax_read.slice(..4);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            ctx.device.poll(wgpu::Maintain::Wait);
            let v = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[0];
            self.argmax_read.unmap();
            Some(v)
        } else { None };
        (end, g0)
    }

    /// Prefill `prompt` positions `[start_pos .. len)` into `slot`, assuming the
    /// prefix `[0 .. start_pos)` is already resident in the slot's KV (via reused
    /// cache blocks). The first generated token is greedy (`samp = None`) or drawn
    /// from softmax(logits/temp) with a single `(temp, seed)`.
    pub fn prefill_slot_from(&self, prompt: &[u32], slot: u32, start_pos: usize, samp: Option<(f32, u32)>) -> u32 {
        assert!(!prompt.is_empty() && prompt.len() <= self.max_seq && start_pos < prompt.len());
        let mut first_tok = 0u32;
        let mut start = start_pos;
        while start < prompt.len() {
            let (next, g) = self.prefill_chunk(prompt, slot, start, samp);
            if let Some(g0) = g { first_tok = g0; }
            start = next;
        }
        first_tok
    }

    /// Gather the per-row embeddings + RoPE tables for `tokens`/`positions` and
    /// upload them (plus pos/slot index buffers) — the inputs both decode and
    /// prefill feed into `record_layers`.
    fn write_inputs(&self, tokens: &[u32], positions: &[u32], slots: &[u32]) {
        let ctx = &self.model.ctx;
        let m = tokens.len();
        let n_embd = self.model.n_embd;
        let half = self.model.head_dim / 2;
        let mut x_host = vec![0f32; m * n_embd];
        let mut cos_host = vec![0f32; m * half];
        let mut sin_host = vec![0f32; m * half];
        for s in 0..m {
            let tk = tokens[s] as usize;
            x_host[s * n_embd..(s + 1) * n_embd].copy_from_slice(&self.model.embed[tk * n_embd..(tk + 1) * n_embd]);
            let p = positions[s] as usize;
            assert!(p < self.max_seq, "position {p} exceeds batched max_seq {}", self.max_seq);
            cos_host[s * half..(s + 1) * half].copy_from_slice(&self.model.cos[p * half..p * half + half]);
            sin_host[s * half..(s + 1) * half].copy_from_slice(&self.model.sin[p * half..p * half + half]);
        }
        ctx.queue.write_buffer(&self.x_b, 0, bytemuck::cast_slice(&x_host));
        ctx.queue.write_buffer(&self.cos_b, 0, bytemuck::cast_slice(&cos_host));
        ctx.queue.write_buffer(&self.sin_b, 0, bytemuck::cast_slice(&sin_host));
        ctx.queue.write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(positions));
        ctx.queue.write_buffer(&self.slots_buf, 0, bytemuck::cast_slice(slots));
    }

    /// Record the full per-layer transformer forward for `m` rows (residual left
    /// in `x_b`). Each row's new K/V is scattered to its physical pool position
    /// `phys[i]` (precomputed from the block table); the paged SDPA has row `i`
    /// attend its slot's KV `0..=position[i]` via the block table. Shared by
    /// decode (1 token/stream) and prefill (P prompt tokens, staggered positions).
    /// The caller must have run `ensure_blocks` + `upload_block_table` already.
    fn record_layers(&self, enc: &mut wgpu::CommandEncoder, m: usize, phys: &[usize]) {
        let ctx = &self.model.ctx;
        let (n_embd, n_head, n_kv_head, head_dim, n_inter, eps) = (
            self.model.n_embd, self.model.n_head, self.model.n_kv_head,
            self.model.head_dim, self.model.n_inter, self.model.eps);
        let kv_dim = n_kv_head * head_dim;
        let attn_dim = n_head * head_dim;
        for (li, layer) in self.model.layers.iter().enumerate() {
            ctx.record_bnorm(enc, &self.x_b, &layer.attn_norm_w, &self.normed_b, n_embd, eps, m);
            ctx.record_gemm(enc, &layer.wq, &self.normed_b, &self.q_b, n_embd, m, 0);
            ctx.record_gemm(enc, &layer.wk, &self.normed_b, &self.k_b, n_embd, m, 0);
            ctx.record_gemm(enc, &layer.wv, &self.normed_b, &self.v_b, n_embd, m, 0);
            ctx.record_brope(enc, &self.q_b, &self.cos_b, &self.sin_b, n_head, head_dim, m);
            ctx.record_brope(enc, &self.k_b, &self.cos_b, &self.sin_b, n_kv_head, head_dim, m);
            // scatter each row's new K/V into its physical block-pool position
            for s in 0..m {
                let dst = (phys[s] * kv_dim * 4) as u64;
                let src = (s * kv_dim * 4) as u64;
                enc.copy_buffer_to_buffer(&self.k_b, src, &self.k_pool[li], dst, (kv_dim * 4) as u64);
                enc.copy_buffer_to_buffer(&self.v_b, src, &self.v_pool[li], dst, (kv_dim * 4) as u64);
            }
            ctx.record_bdsdpa_paged(enc, &self.q_b, &self.k_pool[li], &self.v_pool[li], &self.attn_b, &self.pos_buf, &self.slots_buf, &self.block_table_buf, n_head, n_kv_head, head_dim, m, self.block_size, self.max_blocks_per_seq);
            ctx.record_gemm(enc, &layer.wo, &self.attn_b, &self.x_b, attn_dim, m, 1);
            ctx.record_bnorm(enc, &self.x_b, &layer.ffn_norm_w, &self.normed_b, n_embd, eps, m);
            ctx.record_gemm(enc, &layer.w1, &self.normed_b, &self.gate_b, n_embd, m, 0);
            ctx.record_gemm(enc, &layer.w3, &self.normed_b, &self.up_b, n_embd, m, 0);
            ctx.record_silu_mul(enc, &self.gate_b, &self.up_b, &self.h_b, m * n_inter);
            ctx.record_gemm(enc, &layer.w2, &self.h_b, &self.x_b, n_inter, m, 1);
        }
    }

    /// Ensure each row's slot has blocks for its position, compute the physical
    /// scatter positions, and upload the block table — the paging prologue both
    /// step_slotted and prefill_slot run before `record_layers`. Panics if the
    /// pool is exhausted (callers gate admission via `can_fit`/`free_blocks`).
    fn prepare_paging(&self, m: usize, positions: &[u32], slots: &[u32]) -> Vec<usize> {
        let mut phys = vec![0usize; m];
        for s in 0..m {
            assert!(self.ensure_blocks(slots[s], positions[s] as usize + 1),
                "KV block pool exhausted (slot {}, pos {})", slots[s], positions[s]);
            phys[s] = self.phys_pos(slots[s], positions[s] as usize);
        }
        self.upload_block_table();
        phys
    }
}

/// An active sequence. `tokens` is the full history (prompt ++ every produced
/// token, including the not-yet-fed last one), so derived state is:
/// next = tokens.last(); KV write pos = tokens.len()-1; gen index = tokens.len()-prompt_len.
/// Keeping the full history lets a preempted sequence be recomputed on resume.
struct CbSeq { id: u64, slot: u32, tokens: Vec<u32>, prompt_len: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32 }
impl CbSeq {
    fn next(&self) -> u32 { *self.tokens.last().unwrap() }
    fn pos(&self) -> u32 { (self.tokens.len() - 1) as u32 }
    fn gen_index(&self) -> usize { self.tokens.len() - self.prompt_len }
}

/// A sequence's KV cache copied out to host RAM (swap-to-host preemption),
/// `nb = ceil(n_pos/block_size)` logical blocks per layer, K then V packed in
/// logical-block order `[layer*nb + lb][block_size*kv_dim]`.
struct HostKv { n_pos: usize, nb: usize, k: Vec<f32>, v: Vec<f32> }

/// A sequence evicted under KV-pool pressure. On reschedule it either:
/// - `kv = None` (recompute): re-prefills `tokens` (prompt ++ produced); the
///   prefix cache reuses the prompt's blocks. Or
/// - `kv = Some(blob)` (swap-to-host): restores the saved KV and runs one decode
///   step — no recompute.
/// Both are bit-identical to never preempting (KV at p depends only on
/// tokens[0..=p]; same seed index).
struct PreemptedSeq { id: u64, tokens: Vec<u32>, prompt_len: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32, kv: Option<HostKv> }

/// A sequence mid-prefill. Its prompt is prefilled one chunk per scheduler step
/// (interleaved with decode of the active batch, so a long prompt doesn't stall
/// in-flight requests). `prefill_pos` is the next position to prefill (starts at
/// the prefix cache's reused length); `cached_blocks` = blocks reused, registered
/// once prefill completes.
struct PrefillingSeq { id: u64, slot: u32, prompt: Vec<u32>, prompt_len: usize, prefill_pos: usize, cached_blocks: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32 }

/// Per-token sampling seed: decorrelate by generation index (the kernel further
/// hashes (seed, token_id), so a cheap mix here is enough).
fn step_seed(base: u32, gen_idx: u32) -> u32 { base.wrapping_add(gen_idx.wrapping_mul(0x9E3779B1)) }

/// Per-request sampling knobs. `temp ≤ 0` = greedy; `top_k = 0` = no top-k cap;
/// `top_p ≥ 1` (or 0) = no nucleus cap. top-k/top-p sample within the GPU's
/// `TOPK_K` candidate pool, so effective top_k is capped at `TOPK_K`.
#[derive(Clone, Copy)]
pub struct SamplingParams { pub temp: f32, pub top_k: u32, pub top_p: f32 }
impl SamplingParams {
    pub fn greedy() -> Self { Self { temp: 0.0, top_k: 0, top_p: 1.0 } }
    pub fn temperature(temp: f32) -> Self { Self { temp, top_k: 0, top_p: 1.0 } }
    /// Whether this needs the top-K candidate path (vs greedy / full-dist temp).
    fn needs_topk(&self) -> bool { self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0) }
}

/// Deterministic uniform in (0,1) from a seed (splitmix64).
fn rng01(seed: u32) -> f32 {
    let mut z = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 40) as f32 + 0.5) / 16777216.0
}

/// Sample a token from a stream's descending top-K `(vals, idxs)` under
/// `params`, seeded by `seed`. Applies temperature, top-k, then top-p (nucleus),
/// renormalizes, and draws by inverse-CDF. Greedy (`temp ≤ 0`) returns the top-1.
fn sample_topk(vals: &[f32], idxs: &[u32], params: SamplingParams, seed: u32) -> u32 {
    // Drop kernel sentinels (fewer than K distinct logits — never for real vocab).
    let n = vals.iter().take_while(|&&v| v > -3.0e37).count().min(idxs.len());
    if n == 0 { return idxs[0]; }
    if params.temp <= 0.0 { return idxs[0]; } // greedy
    let kk = if params.top_k == 0 { n } else { (params.top_k as usize).min(n) };
    let maxv = vals[0]; // descending → max is first
    let mut probs: Vec<f32> = (0..kk).map(|i| ((vals[i] - maxv) / params.temp).exp()).collect();
    let z: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= z; }
    // Nucleus: smallest prefix whose cumulative prob ≥ top_p.
    let mut cut = kk;
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() { cum += p; if cum >= params.top_p { cut = i + 1; break; } }
    }
    let zc: f32 = probs[..cut].iter().sum();
    let mut acc = 0.0;
    let target = rng01(seed) * zc;
    for i in 0..cut { acc += probs[i]; if acc >= target { return idxs[i]; } }
    idxs[cut - 1]
}

/// Continuous (in-flight) batching scheduler over a `BatchedDecoder`. Sequences
/// are admitted at any time — their prompt is prefilled into a free KV slot —
/// and then all active sequences are decoded together each step regardless of
/// arrival time; finished sequences free their slot for new arrivals. This is
/// the single-device equivalent of datacenter in-flight batching: the GPU runs
/// a full batch instead of one request at a time.
pub struct ContinuousBatcher<'a> {
    dec: BatchedDecoder<'a>,
    free: Vec<u32>,
    active: Vec<CbSeq>,
    /// Sequences evicted under KV-pool pressure, awaiting reschedule (recompute).
    preempted: std::collections::VecDeque<PreemptedSeq>,
    /// Sequences mid-prefill (one chunk advanced per step, interleaved with decode).
    prefilling: std::collections::VecDeque<PrefillingSeq>,
    preemptions: u64,
    /// Preempt active sequences by SWAP-TO-HOST (copy KV out, restore on resume)
    /// instead of recompute. Set from `ZLLM_SWAP`; tests flip it directly. Default
    /// OFF: measured slower than recompute on this UMA box (see `gpu_swap_preemption`)
    /// — it's a discrete-GPU lever. Recompute is the right default here.
    swap_mode: bool,
}

impl<'a> ContinuousBatcher<'a> {
    pub fn new(model: &'a GpuModel, m_max: usize, max_seq: usize) -> Self {
        Self { dec: model.batched_decoder(m_max, max_seq), free: (0..m_max as u32).rev().collect(), active: Vec::new(), preempted: Default::default(), prefilling: Default::default(), preemptions: 0, swap_mode: std::env::var("ZLLM_SWAP").is_ok() }
    }

    /// Continuous batcher over a paged KV pool of `n_blocks` blocks shared by all
    /// `m_max` slots. `n_blocks < m_max*ceil(max_seq/block)` overcommits memory:
    /// many short sequences fit where a contiguous max_seq-per-slot reservation
    /// could not. Admission is optimistic; if a running sequence can't grow, a
    /// victim is preempted (its KV freed) and recomputed later.
    pub fn with_pool(model: &'a GpuModel, m_max: usize, max_seq: usize, n_blocks: usize) -> Self {
        Self { dec: model.batched_decoder_paged(m_max, max_seq, n_blocks), free: (0..m_max as u32).rev().collect(), active: Vec::new(), preempted: Default::default(), prefilling: Default::default(), preemptions: 0, swap_mode: std::env::var("ZLLM_SWAP").is_ok() }
    }
    pub fn has_free(&self) -> bool { !self.free.is_empty() }
    pub fn active_len(&self) -> usize { self.active.len() }
    /// True if every active sequence is greedy (temp 0, no top-k) — the precondition
    /// for step_pld (whose argmax verify is bit-identical to greedy step()).
    pub fn all_greedy(&self) -> bool { self.active.iter().all(|s| !s.params.needs_topk() && s.params.temp == 0.0) }

    /// Cancel a sequence by id (e.g. its client disconnected): remove it from
    /// wherever it is — decoding, prefilling, or preempted — and free its KV slot
    /// + blocks. Returns true if it was found. Reclaims capacity immediately
    /// instead of generating output nobody will read.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(pos) = self.active.iter().position(|s| s.id == id) {
            let s = self.active.remove(pos);
            self.dec.free_slot(s.slot);
            self.free.push(s.slot);
            return true;
        }
        if let Some(pos) = self.prefilling.iter().position(|s| s.id == id) {
            let s = self.prefilling.remove(pos).unwrap();
            self.dec.free_slot(s.slot);
            self.free.push(s.slot);
            return true;
        }
        if let Some(pos) = self.preempted.iter().position(|s| s.id == id) {
            self.preempted.remove(pos); // preempted holds no slot/blocks
            return true;
        }
        false
    }

    /// Sequences currently mid-prefill (chunked).
    pub fn prefilling_len(&self) -> usize { self.prefilling.len() }
    /// Sequences currently preempted (evicted, awaiting recompute).
    pub fn preempted_len(&self) -> usize { self.preempted.len() }
    /// Total preemptions since construction (observability).
    pub fn preemption_count(&self) -> u64 { self.preemptions }
    /// (free, total) physical KV blocks — for observing pool pressure.
    pub fn block_pool(&self) -> (usize, usize) { (self.dec.free_blocks(), self.dec.n_blocks) }
    /// Prefix-cache stats: (reused blocks, freshly-prefilled blocks) since start.
    pub fn cache_stats(&self) -> (u64, u64) { self.dec.cache_stats() }

    /// Admit a sequence: batched-prefill its prompt into a free KV slot and join
    /// the decode batch. Returns (first_token, done) — `done` is true if that one
    /// token already completes the request (EOS or max_tokens<=1), in which case
    /// the slot + its blocks are returned immediately. Returns None if no slot is
    /// free OR the KV pool can't fit prompt+max_tokens (caller should retry later;
    /// with the default full pool this never trips).
    pub fn admit(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32) -> Option<(u32, bool)> {
        self.admit_sampled(id, prompt, max_tokens, eos, 0.0, 0)
    }

    /// Admit with temperature sampling (`temp ≤ 0` = greedy), seeded by `seed`.
    pub fn admit_sampled(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32, temp: f32, seed: u32) -> Option<(u32, bool)> {
        self.admit_params(id, prompt, max_tokens, eos, SamplingParams::temperature(temp), seed)
    }

    /// Admit with full sampling params (temperature, top-k, top-p), seeded by
    /// `seed` (reproducible per request). Note: the first (prefill) token uses
    /// temperature only; top-k/top-p apply from the first decode token on.
    pub fn admit_params(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32, params: SamplingParams, seed: u32) -> Option<(u32, bool)> {
        // Optimistic admission: only the prompt's prefill must fit now; decode
        // growth is handled by preemption (make_room) if the pool fills later.
        if !self.dec.can_fit(prompt.len()) { return None; }
        let slot = self.free.pop()?;
        // First token = generation index 0 (temperature only).
        let samp = if params.temp > 0.0 { Some((params.temp, step_seed(seed, 0))) } else { None };
        let (g, _cached_len) = self.dec.prefill_slot_cached(prompt, slot, samp); // reuse shared-prefix KV; prefill the rest
        let done = max_tokens <= 1 || g == eos;
        if done {
            self.dec.free_slot(slot);
            self.free.push(slot);
        } else {
            let mut tokens = Vec::with_capacity(prompt.len() + max_tokens);
            tokens.extend_from_slice(prompt);
            tokens.push(g);
            self.active.push(CbSeq { id, slot, tokens, prompt_len: prompt.len(), max_tokens, eos, params, base_seed: seed });
        }
        Some((g, done))
    }

    /// Enqueue a sequence for CHUNKED prefill: assign a slot, reuse any cached
    /// prefix, then prefill one chunk per `step()` (interleaved with decode of the
    /// active batch, so a long prompt doesn't stall in-flight requests) until the
    /// prompt completes and it joins the decode batch. The first token is emitted
    /// by a later `step()`, not returned here. Returns false if no slot is free or
    /// the pool can't fit the prompt.
    pub fn enqueue_params(&mut self, id: u64, prompt: Vec<u32>, max_tokens: usize, eos: u32, params: SamplingParams, seed: u32) -> bool {
        if prompt.is_empty() || prompt.len() > self.dec.max_seq_len() { return false; }
        if self.free.is_empty() || !self.dec.can_fit(prompt.len()) { return false; }
        let slot = self.free.pop().unwrap();
        let prompt_len = prompt.len();
        let (cached_len, cached_blocks) = self.dec.prefill_prefix_reuse(&prompt, slot);
        self.prefilling.push_back(PrefillingSeq { id, slot, prompt, prompt_len, prefill_pos: cached_len, cached_blocks, max_tokens, eos, params, base_seed: seed });
        true
    }

    /// One scheduler step: decode the active batch, advance one chunk of the front
    /// prefilling sequence, and resume any preempted sequences that now fit.
    /// Returns (id, new_token, done) for every token produced this step.
    pub fn step(&mut self) -> Vec<(u64, u32, bool)> {
        let mut out = Vec::new();
        if !self.active.is_empty() {
            self.make_room(); // preempt victims so every active sequence can grow
            let toks: Vec<u32> = self.active.iter().map(|s| s.next()).collect();
            let pos: Vec<u32> = self.active.iter().map(|s| s.pos()).collect();
            let slots: Vec<u32> = self.active.iter().map(|s| s.slot).collect();
            let seeds: Vec<u32> = self.active.iter().map(|s| step_seed(s.base_seed, s.gen_index() as u32)).collect();
            // Cheapest path that satisfies every active stream:
            // top-k/top-p (CPU sample over GPU top-K) > temperature (Gumbel) > greedy.
            let nexts = if self.active.iter().any(|s| s.params.needs_topk()) {
                let params: Vec<SamplingParams> = self.active.iter().map(|s| s.params).collect();
                self.dec.step_slotted_topk(&toks, &pos, &slots, &params, &seeds)
            } else if self.active.iter().any(|s| s.params.temp > 0.0) {
                let temps: Vec<f32> = self.active.iter().map(|s| s.params.temp).collect();
                self.dec.step_slotted_sample(&toks, &pos, &slots, &temps, &seeds)
            } else {
                self.dec.step_slotted(&toks, &pos, &slots)
            };
            for (i, &nt) in nexts.iter().enumerate() {
                let s = &mut self.active[i];
                s.tokens.push(nt);
                let done = nt == s.eos || s.gen_index() >= s.max_tokens;
                out.push((s.id, nt, done));
            }
            let (free, dec) = (&mut self.free, &self.dec);
            self.active.retain(|s| {
                let done = s.next() == s.eos || s.gen_index() >= s.max_tokens;
                if done { dec.free_slot(s.slot); free.push(s.slot); }
                !done
            });
        }
        self.advance_prefill(&mut out); // one prefill chunk, interleaved with decode
        self.reschedule(&mut out); // resume preempted sequences that now fit
        out
    }

    /// Like `step()` but with prompt-lookup speculative decode IN the batch: every
    /// active sequence proposes an n-gram draft (from its own history), all drafts
    /// concatenate into ONE `step_slotted` forward (each row attends its sequence's
    /// slot at its position), and each sequence commits the tokens its own argmax
    /// agrees with. >1 token/stream/forward on echo-heavy workloads. GREEDY only —
    /// the verify is argmax, so output is bit-identical to `step()`; sequences with
    /// `temp>0` would diverge from sampling, so route those through `step()` instead.
    /// Assumes the full KV pool (a draft writes up to `draft_k+1` positions/step).
    pub fn step_pld(&mut self, lookup_len: usize, draft_k: usize) -> Vec<(u64, u32, bool)> {
        let mut out = Vec::new();
        if !self.active.is_empty() {
            self.make_room();
            let (mut rows, mut positions, mut slots) = (Vec::new(), Vec::new(), Vec::new());
            let mut layout: Vec<(usize, usize, Vec<u32>)> = Vec::new(); // (active_idx, start, draft)
            for (i, s) in self.active.iter().enumerate() {
                let k = draft_k.min(s.max_tokens.saturating_sub(s.gen_index()));
                let draft = if k >= 1 {
                    crate::engine::spec_decode::lookup_draft_best(&s.tokens, &s.tokens, lookup_len, k).unwrap_or_default()
                } else { Vec::new() };
                let start = rows.len();
                rows.push(s.next()); rows.extend_from_slice(&draft);
                let p0 = s.pos();
                for j in 0..=draft.len() { positions.push(p0 + j as u32); slots.push(s.slot); }
                layout.push((i, start, draft));
            }
            let outs = self.dec.step_slotted(&rows, &positions, &slots);
            for (i, start, draft) in &layout {
                let so = &outs[*start..*start + 1 + draft.len()];
                let mut acc = 0usize; while acc < draft.len() && so[acc] == draft[acc] { acc += 1; }
                let s = &mut self.active[*i];
                for &tok in so.iter().take(acc + 1) {
                    s.tokens.push(tok);
                    let done = tok == s.eos || s.gen_index() >= s.max_tokens;
                    out.push((s.id, tok, done));
                    if done { break; }
                }
            }
            let (free, dec) = (&mut self.free, &self.dec);
            self.active.retain(|s| {
                let done = s.next() == s.eos || s.gen_index() >= s.max_tokens;
                if done { dec.free_slot(s.slot); free.push(s.slot); }
                !done
            });
        }
        self.advance_prefill(&mut out);
        self.reschedule(&mut out);
        out
    }

    /// Preempt the most-recently-admitted active sequence (LIFO), queuing it for
    /// recompute. Returns false if there are no active sequences.
    fn preempt_last_active(&mut self) -> bool {
        let Some(victim) = self.active.pop() else { return false };
        self.preemptions += 1;
        // Swap-to-host: copy KV out (frees the blocks) so resume skips recompute.
        // Recompute: just free the blocks; `kv = None` re-prefills on resume.
        // Valid KV covers positions 0..tokens.len()-1 (the last token isn't fed yet).
        let kv = if self.swap_mode {
            Some(self.dec.swap_out(victim.slot, victim.tokens.len() - 1))
        } else {
            self.dec.free_slot(victim.slot);
            None
        };
        self.free.push(victim.slot);
        self.preempted.push_back(PreemptedSeq {
            id: victim.id, tokens: victim.tokens, prompt_len: victim.prompt_len,
            max_tokens: victim.max_tokens, eos: victim.eos, params: victim.params, base_seed: victim.base_seed, kv,
        });
        true
    }

    /// Preempt the most-recently-enqueued *prefilling* sequence (not the front).
    /// It has produced no tokens, so it re-prefills its prompt from scratch on
    /// reschedule. Returns false if there is at most the front prefilling sequence.
    fn preempt_last_prefill(&mut self) -> bool {
        if self.prefilling.len() < 2 { return false; }
        let pf = self.prefilling.pop_back().unwrap();
        self.dec.free_slot(pf.slot);
        self.free.push(pf.slot);
        self.preemptions += 1;
        self.preempted.push_back(PreemptedSeq {
            id: pf.id, tokens: pf.prompt, prompt_len: pf.prompt_len,
            max_tokens: pf.max_tokens, eos: pf.eos, params: pf.params, base_seed: pf.base_seed, kv: None,
        });
        true
    }

    /// Preempt (LIFO) active sequences until the pool can supply the blocks every
    /// remaining active sequence needs to grow this step. A single sequence always
    /// fits (pool ≥ one full sequence), so this terminates.
    fn make_room(&mut self) {
        let needed = |b: &BatchedDecoder, active: &[CbSeq]| -> usize {
            active.iter().map(|s| b.blocks_short(s.slot, s.tokens.len())).sum()
        };
        while self.active.len() > 1 && self.dec.free_blocks() < needed(&self.dec, &self.active) {
            self.preempt_last_active();
        }
    }

    /// Advance the front prefilling sequence by one chunk (prefill-priority: makes
    /// room by preempting active — then, if still short, other prefilling — so the
    /// front always progresses). On completion, registers its blocks and the
    /// sequence joins the decode batch, emitting its first token into `out`.
    fn advance_prefill(&mut self, out: &mut Vec<(u64, u32, bool)>) {
        if self.prefilling.is_empty() { return; }
        let (slot, start, chunk_end, temp, base_seed) = {
            let pf = &self.prefilling[0];
            let end = (pf.prefill_pos + self.dec.prefill_chunk_size()).min(pf.prompt_len);
            (pf.slot, pf.prefill_pos, end, pf.params.temp, pf.base_seed)
        };
        // Make the chunk's blocks fit (preempt active first, then other prefills).
        while self.dec.blocks_short(slot, chunk_end) > self.dec.free_blocks() {
            if !self.preempt_last_active() && !self.preempt_last_prefill() { return; } // can't fit yet; wait
        }
        let samp = if temp > 0.0 { Some((temp, step_seed(base_seed, 0))) } else { None };
        let (next, g0) = self.dec.prefill_chunk(&self.prefilling[0].prompt, slot, start, samp);
        self.prefilling[0].prefill_pos = next;
        let Some(g) = g0 else { return }; // chunk done but prompt not complete
        let pf = self.prefilling.pop_front().unwrap();
        self.dec.prefill_register(&pf.prompt, pf.slot, pf.cached_blocks);
        let done = pf.max_tokens <= 1 || g == pf.eos;
        out.push((pf.id, g, done));
        if done {
            self.dec.free_slot(pf.slot);
            self.free.push(pf.slot);
        } else {
            let mut tokens = pf.prompt;
            tokens.push(g);
            self.active.push(CbSeq { id: pf.id, slot: pf.slot, tokens, prompt_len: pf.prompt_len, max_tokens: pf.max_tokens, eos: pf.eos, params: pf.params, base_seed: pf.base_seed });
        }
    }

    /// One single-row decode step (`tok` at `pos` in `slot`), dispatching on the
    /// sequence's sampling params, returning the next token. Used to resume a
    /// swap-to-host sequence after its KV is restored.
    fn decode_one(&self, tok: u32, pos: u32, slot: u32, params: SamplingParams, seed: u32) -> u32 {
        let (t, p, s, sd) = ([tok], [pos], [slot], [seed]);
        if params.needs_topk() {
            self.dec.step_slotted_topk(&t, &p, &s, &[params], &sd)[0]
        } else if params.temp > 0.0 {
            self.dec.step_slotted_sample(&t, &p, &s, &[params.temp], &sd)[0]
        } else {
            self.dec.step_slotted(&t, &p, &s)[0]
        }
    }

    /// Resume preempted sequences (FIFO) while a slot is free and the pool can
    /// hold them. Swap-to-host: restore the saved KV + one decode step. Recompute
    /// (`kv = None`): re-prefill prompt ++ produced (prefix cache reuses the
    /// prompt). Either way, produces the exact next token they'd have produced.
    fn reschedule(&mut self, out: &mut Vec<(u64, u32, bool)>) {
        while !self.preempted.is_empty() && !self.free.is_empty() {
            let len = self.preempted.front().unwrap().tokens.len();
            if !self.dec.can_fit(len) { break; }
            let p = self.preempted.pop_front().unwrap();
            let slot = self.free.pop().unwrap();
            let gen_idx = (p.tokens.len() - p.prompt_len) as u32;
            let seed = step_seed(p.base_seed, gen_idx);
            let g = if let Some(kv) = &p.kv {
                self.dec.swap_in(slot, kv); // write KV back into fresh blocks…
                let next = *p.tokens.last().unwrap();
                self.decode_one(next, (p.tokens.len() - 1) as u32, slot, p.params, seed) // …then one step
            } else {
                let samp = if p.params.temp > 0.0 { Some((p.params.temp, seed)) } else { None };
                self.dec.prefill_slot_cached(&p.tokens, slot, samp).0
            };
            let mut tokens = p.tokens;
            tokens.push(g);
            let done = g == p.eos || (tokens.len() - p.prompt_len) >= p.max_tokens;
            out.push((p.id, g, done));
            if done {
                self.dec.free_slot(slot);
                self.free.push(slot);
            } else {
                self.active.push(CbSeq { id: p.id, slot, tokens, prompt_len: p.prompt_len, max_tokens: p.max_tokens, eos: p.eos, params: p.params, base_seed: p.base_seed });
            }
        }
    }
}

/// A generation request submitted to a [`GpuBatchServer`].
pub struct GenReq {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub eos: u32,
    /// Sampling knobs and per-request RNG seed (reproducible).
    pub params: SamplingParams,
    pub seed: u32,
    /// The server pushes `Some(token)` per produced token, then `None` at
    /// completion. Use a tokio unbounded channel so an async HTTP handler can
    /// stream from the receiver while the (sync) serving thread sends.
    pub tok_tx: tokio::sync::mpsc::UnboundedSender<Option<u32>>,
}

/// A GPU continuous-batching serving loop on its own OS thread. It OWNS the
/// `GpuModel` (and the `ContinuousBatcher` that borrows it), so there is no
/// borrow-across-`Arc<Mutex>` problem: handlers communicate only by channel.
/// `submit()` enqueues a prompt + a token channel; the loop admits it into a
/// free KV slot and decodes it together with every other in-flight request,
/// streaming tokens back and freeing the slot on completion.
/// A control message to the serving thread: a generation request, or a model
/// hot-swap (drop the batcher, load a new model, rebuild — acks when done).
enum CbMsg {
    Gen(GenReq),
    Swap { path: String, ack: std::sync::mpsc::Sender<bool> },
}

pub struct GpuBatchServer {
    tx: std::sync::mpsc::Sender<CbMsg>,
    m_max: usize,
}

impl GpuBatchServer {
    /// Spawn the serving thread. `model` is MOVED onto it (wgpu device/queue are
    /// Send). `m_max` = max concurrent sequences (KV slots), `max_seq` = max
    /// context length per slot.
    pub fn spawn(model: GpuModel, m_max: usize, max_seq: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<CbMsg>();
        std::thread::Builder::new()
            .name("gpu-batcher".into())
            .spawn(move || Self::serve(model, rx, m_max, max_seq))
            .expect("spawn gpu-batcher thread");
        Self { tx, m_max }
    }

    /// Max concurrent sequences the server can hold in flight.
    pub fn capacity(&self) -> usize { self.m_max }

    /// Submit a request. Returns a receiver yielding `Some(token)` per decode
    /// step then `None` at completion, or `Err` if the serving thread is gone.
    /// `params.temp ≤ 0` = greedy; `seed` makes sampling reproducible per request.
    pub fn submit(&self, prompt: Vec<u32>, max_tokens: usize, eos: u32, params: SamplingParams, seed: u32)
        -> Result<tokio::sync::mpsc::UnboundedReceiver<Option<u32>>, ()> {
        let (tok_tx, tok_rx) = tokio::sync::mpsc::unbounded_channel();
        self.tx.send(CbMsg::Gen(GenReq { prompt, max_tokens, eos, params, seed, tok_tx })).map_err(|_| ())?;
        Ok(tok_rx)
    }

    /// Hot-swap the served model (e.g. on `/v1/models/select`). In-flight
    /// sequences are aborted (their KV is on the old model). Blocks until the new
    /// model is loaded; returns false if the load failed or the thread is gone.
    pub fn swap_model(&self, path: String) -> bool {
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        if self.tx.send(CbMsg::Swap { path, ack: ack_tx }).is_err() { return false; }
        ack_rx.recv().unwrap_or(false)
    }

    fn serve(mut model: GpuModel, rx: std::sync::mpsc::Receiver<CbMsg>, m_max: usize, max_seq: usize) {
        use std::sync::mpsc::TryRecvError;
        let mut next_id: u64 = 0;
        let pld = std::env::var("ZLLM_CB_PLD").is_ok(); // batched spec-decode for greedy batches
        loop { // one iteration per loaded model
            let mut cb = ContinuousBatcher::new(&model, m_max, max_seq);
            let mut chans: std::collections::HashMap<u64, tokio::sync::mpsc::UnboundedSender<Option<u32>>> = Default::default();
            let mut reload: Option<(String, std::sync::mpsc::Sender<bool>)> = None;
            'serve: loop {
                // Idle (nothing active, prefilling, or preempted) → block for a message.
                let busy = cb.active_len() + cb.prefilling_len() + cb.preempted_len() > 0;
                if !busy {
                    match rx.recv() {
                        Ok(CbMsg::Gen(req)) => Self::admit_req(&mut cb, &mut chans, &mut next_id, req),
                        Ok(CbMsg::Swap { path, ack }) => { reload = Some((path, ack)); break 'serve; }
                        Err(_) => return, // every sender dropped → shut down
                    }
                }
                // Fill free slots with any waiting requests (non-blocking).
                while cb.has_free() {
                    match rx.try_recv() {
                        Ok(CbMsg::Gen(req)) => Self::admit_req(&mut cb, &mut chans, &mut next_id, req),
                        Ok(CbMsg::Swap { path, ack }) => { reload = Some((path, ack)); break 'serve; }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => { if cb.active_len() + cb.prefilling_len() + cb.preempted_len() == 0 { return; } break; }
                    }
                }
                if cb.active_len() + cb.prefilling_len() + cb.preempted_len() == 0 { continue; }
                // One scheduler step; stream + retire. ZLLM_CB_PLD turns on batched
                // prompt-lookup spec-decode when the whole active batch is greedy
                // (bit-identical to step(), fewer forwards on echo/RAG/code).
                let res = if pld && cb.all_greedy() { cb.step_pld(3, 7) } else { cb.step() };
                for (id, tok, done) in res {
                    if let Some(ch) = chans.get(&id) { let _ = ch.send(Some(tok)); }
                    if done { if let Some(ch) = chans.remove(&id) { let _ = ch.send(None); } }
                }
                // Reclaim sequences whose client disconnected (covers mid-prefill,
                // where no token has been sent yet to surface a send error).
                chans.retain(|id, ch| if ch.is_closed() { cb.cancel(*id); false } else { true });
            }
            // Swap requested: abort in-flight sequences (their KV is on the old model).
            for (_, ch) in chans.drain() { let _ = ch.send(None); }
            let Some((path, ack)) = reload else { return };
            drop(cb); // release the borrow of `model` before replacing it
            match GpuContext::new().and_then(|ctx| GpuModel::load(&path, ctx)) {
                Ok(m) => { model = m; let _ = ack.send(true); }
                Err(_) => { let _ = ack.send(false); return; } // failed load → shut down
            }
        }
    }

    fn admit_req(
        cb: &mut ContinuousBatcher,
        chans: &mut std::collections::HashMap<u64, tokio::sync::mpsc::UnboundedSender<Option<u32>>>,
        next_id: &mut u64,
        req: GenReq,
    ) {
        let id = *next_id; *next_id += 1;
        // Chunked prefill: enqueue; the first token is emitted by a later step().
        if cb.enqueue_params(id, req.prompt, req.max_tokens, req.eos, req.params, req.seed) {
            chans.insert(id, req.tok_tx);
        } else {
            let _ = req.tok_tx.send(None); // rejected (no slot / pool too small)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LRU block reclaim (pure logic, no GPU): under pool pressure `alloc` must
    /// evict the LEAST-recently-used cached block so a hot shared prefix (touched
    /// often) survives churn from cold one-offs. First-fit-by-index (the old
    /// behaviour) would evict in index order [0,1,2,3] and thrash the hot prefix.
    #[test]
    fn gpu_block_lru_reclaim() {
        let mut bs = BlockState {
            free: Vec::new(),                    // pool exhausted → every alloc reclaims
            slot_blocks: vec![Vec::new(); 1],
            table_host: vec![0; 8],
            refcount: vec![0u32; 4],             // all reclaimable
            cache_map: std::collections::HashMap::new(),
            block_hash: vec![None; 4],
            last_use: vec![0u64; 4],
            clock: 0,
            first_fit: false,
            hits: 0, misses: 0,
        };
        for (b, h) in [(0u32, 100u64), (1, 101), (2, 102), (3, 103)] {
            bs.cache_map.insert(h, b);
            bs.block_hash[b as usize] = Some(h);
        }
        // Touch order 2,3,0,1 → ticks [3,4,1,2]; block 2 is LRU, block 1 is MRU.
        for b in [2u32, 3, 0, 1] { bs.touch(b); }
        // Each alloc consumes one cached block (refcount→1, hash dropped); the next
        // picks the next-coldest. Reclaim order is LRU→MRU, NOT index order.
        let order: Vec<u32> = (0..4).map(|_| bs.alloc().expect("reclaimable")).collect();
        assert_eq!(order, vec![2, 3, 0, 1], "must reclaim least-recently-used first (hot prefix last)");
        assert!(bs.alloc().is_none(), "pool fully consumed");
    }

    /// Smallest possible end-to-end GPU compute: upload a vector, run a
    /// WGSL shader that doubles each element, read it back, verify. Proves
    /// instance → adapter → device → shader-compile → dispatch → readback
    /// all work on this box's iGPU through wgpu.
    #[test]
    fn gpu_doubles_a_vector() {
        let names = enumerate();
        eprintln!("adapters: {names:?}");

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };
        eprintln!("using {} via {:?}", ctx.adapter_name, ctx.backend);

        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let n = input.len();
        let bytes = (n * std::mem::size_of::<f32>()) as u64;

        use wgpu::util::DeviceExt;
        let in_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>        src: array<f32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) { dst[i] = src[i] * 2.0; }
}
"#;
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("double"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("double"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes);
        ctx.queue.submit([enc.finish()]);

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let got: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();

        for i in 0..n {
            assert_eq!(got[i], input[i] * 2.0, "mismatch at {i}");
        }
        eprintln!("GPU compute validated: {} elements doubled correctly", n);
    }

    /// Validate the GPU Q4_K mat-vec against the CPU dequant-then-dot
    /// oracle on real Candle-quantized Q4_K weights. Proves the in-shader
    /// f16/6-bit/4-bit unpacking is correct — the foundation for any GPU
    /// inference path.
    #[test]
    fn gpu_q4k_matvec_matches_cpu_dequant() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };

        let n_rows = 512usize;
        let nb_per_row = 8usize;            // n_cols = 2048 (Llama-3.2-1B attn width)
        let n_cols = nb_per_row * QK_K;

        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() {
            let s = ((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0;
            w[i] = s * 0.5;
        }
        let dev = Device::Cpu;
        let wt = Tensor::from_vec(w, (n_rows, n_cols), &dev).unwrap();
        let qt = QTensor::quantize(&wt, GgmlDType::Q4K).unwrap();
        let bytes = qt.data().unwrap();
        assert_eq!(bytes.len(), n_rows * nb_per_row * 144);

        let mut x = vec![0f32; n_cols];
        for i in 0..n_cols {
            let s = ((i as i64).wrapping_mul(11400714819323198485u64 as i64) & 0xFFFF)
                as f32 / 32768.0 - 1.0;
            x[i] = s * 1.3;
        }

        let gpu = ctx.matmul_q4k_f32(&bytes, n_rows, nb_per_row, &x);

        let blocks: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4K, n_rows * nb_per_row)
        };
        let mut buf = [0f32; QK_K];
        let mut max_abs = 0f32;
        let mut at = 0usize;
        for r in 0..n_rows {
            let mut acc = 0f64;
            for b in 0..nb_per_row {
                dequantize_q4k_block(&blocks[r * nb_per_row + b], &mut buf);
                for k in 0..QK_K { acc += (buf[k] as f64) * (x[b * QK_K + k] as f64); }
            }
            let e = (gpu[r] - acc as f32).abs();
            if e > max_abs { max_abs = e; at = r; }
        }
        eprintln!("GPU Q4_K matvec vs CPU dequant: max_abs_err = {max_abs:.5} at row {at} (gpu={})", gpu[at]);
        assert!(max_abs < 0.05, "GPU Q4_K matvec error too high: {max_abs}");
    }

    /// Validate the GPU GQA decode attention against a straightforward
    /// CPU softmax reference, on Llama-3.2-1B head shape (32 q / 8 kv / 64).
    #[test]
    fn gpu_sdpa_decode_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let (n_head, n_kv_head, head_dim, seq_len) = (32usize, 8usize, 64usize, 40usize);
        let group = n_head / n_kv_head;
        let f = |i: usize, salt: usize| (((i * 2654435761 + salt * 40503) % 211) as f32 - 105.0) * 0.02;
        let q: Vec<f32> = (0..n_head * head_dim).map(|i| f(i, 1)).collect();
        let kc: Vec<f32> = (0..seq_len * n_kv_head * head_dim).map(|i| f(i, 2)).collect();
        let vc: Vec<f32> = (0..seq_len * n_kv_head * head_dim).map(|i| f(i, 3)).collect();

        let gpu = ctx.sdpa_decode(&q, &kc, &vc, n_head, n_kv_head, head_dim, seq_len);

        // CPU reference: scores -> softmax -> weighted V, per head.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut cpu = vec![0f32; n_head * head_dim];
        for h in 0..n_head {
            let kvh = h / group;
            let qb = h * head_dim;
            let mut scores = vec![0f32; seq_len];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let kb = (t * n_kv_head + kvh) * head_dim;
                let mut s = 0f32;
                for d in 0..head_dim { s += q[qb + d] * kc[kb + d]; }
                s *= scale; scores[t] = s; if s > mx { mx = s; }
            }
            let mut sum = 0f32;
            for t in 0..seq_len { scores[t] = (scores[t] - mx).exp(); sum += scores[t]; }
            let inv = 1.0 / sum;
            for t in 0..seq_len {
                let w = scores[t] * inv;
                let vb = (t * n_kv_head + kvh) * head_dim;
                for d in 0..head_dim { cpu[qb + d] += w * vc[vb + d]; }
            }
        }
        let mut max_abs = 0f32;
        for i in 0..cpu.len() { max_abs = max_abs.max((gpu[i] - cpu[i]).abs()); }
        eprintln!("GPU SDPA decode vs CPU: max_abs_err = {max_abs:.6}");
        assert!(max_abs < 1e-4, "GPU SDPA mismatch: {max_abs}");
    }

    /// Resident Q6_K matvec throughput (LM-head shape), to quantify the gap
    /// vs the Q4_K kernel's ~141 GB/s.
    /// `cargo test --release --features gpu --lib gpu_q6k_bandwidth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_q6k_bandwidth() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use std::time::Instant;
        let ctx = GpuContext::new().expect("GPU");
        let n_rows = 128256usize;
        let nb = 8usize;
        let n_cols = nb * 256;
        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let bytes = QTensor::quantize(&Tensor::from_vec(w, (n_rows, n_cols), &Device::Cpu).unwrap(), GgmlDType::Q6K)
            .unwrap().data().unwrap().to_vec();
        let w6 = ctx.upload_q6k(&bytes, n_rows, nb);
        use wgpu::util::DeviceExt;
        let x_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&vec![0.7f32; n_cols]), usage: wgpu::BufferUsages::STORAGE });
        let out_buf = ctx.alloc_activation(n_rows, false);
        let dispatch = || {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            ctx.record_matvec_q6k(&mut enc, &w6, &x_buf, &out_buf);
            ctx.queue.submit([enc.finish()]);
        };
        dispatch(); ctx.device.poll(wgpu::Maintain::Wait);
        let iters = 30u32;
        let t0 = Instant::now();
        for _ in 0..iters { dispatch(); }
        ctx.device.poll(wgpu::Maintain::Wait);
        let dt = t0.elapsed();
        // SoA read traffic: ql(128)+qh(64)+scales(64)+d(4) = 260 B/block.
        let read_bytes = n_rows as f64 * nb as f64 * 260.0;
        eprintln!("GPU Q6_K resident matvec: {:.1} GB/s, {:.3} ms/matvec  [Q4_K ref ~141 GB/s]",
            read_bytes * iters as f64 / dt.as_secs_f64() / 1e9, dt.as_secs_f64() * 1e3 / iters as f64);
    }

    /// THE PAYOFF: load the real Llama-3.2-1B on the GPU, generate greedily,
    /// check it agrees with candle's CPU forward, and benchmark decode tok/s.
    /// `cargo test --release --features gpu --lib gpu_full_forward -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_full_forward_vs_candle_and_bench() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let t = Instant::now();
        let model = GpuModel::load(path, ctx).expect("load");
        eprintln!("GPU model loaded in {:.2}s ({} layers, vocab {})", t.elapsed().as_secs_f64(), model.layers.len(), model.vocab);

        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let prompt: Vec<u32> = vec![128000]; // BOS
        let n_gen = 24usize;

        // GPU greedy generation (resident KV cache persists across tokens;
        // forward_argmax does the argmax on-GPU and reads back 4 bytes).
        let _ = &argmax;
        let mut next = 0u32;
        for (i, &tk) in prompt.iter().enumerate() { next = model.forward_argmax(tk, i); }
        let mut gpu_gen = vec![next];
        let mut pos = prompt.len();
        let t0 = Instant::now();
        for _ in 1..n_gen { next = model.forward_argmax(next, pos); gpu_gen.push(next); pos += 1; }
        let dt = t0.elapsed();
        eprintln!("GPU decode: {:.1} tok/s ({} tokens in {:.2}s)", (n_gen - 1) as f64 / dt.as_secs_f64(), n_gen - 1, dt.as_secs_f64());
        eprintln!("GPU gen:    {gpu_gen:?}");

        // Candle CPU reference.
        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
        let mut clog = cb.forward_logits(&prompt).unwrap();
        let mut cnext = argmax(&clog);
        let mut cand_gen = vec![cnext];
        for _ in 1..n_gen { clog = cb.forward_logits(&[cnext]).unwrap(); cnext = argmax(&clog); cand_gen.push(cnext); }
        eprintln!("candle gen: {cand_gen:?}");

        let agree = gpu_gen.iter().zip(&cand_gen).take_while(|(a, b)| a == b).count();
        eprintln!("GPU/candle agree on first {agree}/{n_gen} tokens");
        assert!(agree >= 8, "GPU forward diverges from candle too early ({agree}); likely a kernel/wiring bug");
    }

    /// Validate the GPU Q6_K matvec against candle's dequantize-then-dot
    /// on real Candle-quantized Q6_K weights (LM-head / ffn_down dtype).
    #[test]
    fn gpu_q6k_matvec_matches_cpu() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        const QK_K: usize = 256;

        let ctx = match GpuContext::new() {
            Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let n_rows = 512usize;
        let nb = 8usize;
        let n_cols = nb * QK_K;
        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let dev = Device::Cpu;
        let qt = QTensor::quantize(&Tensor::from_vec(w.clone(), (n_rows, n_cols), &dev).unwrap(), GgmlDType::Q6K).unwrap();
        let bytes = qt.data().unwrap();
        assert_eq!(bytes.len(), n_rows * nb * 210);
        // Ground-truth weights = candle's own dequantization.
        let deq: Vec<f32> = qt.dequantize(&dev).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let x: Vec<f32> = (0..n_cols).map(|i| ((i % 23) as f32 - 11.0) * 0.06).collect();
        let gpu = ctx.matmul_q6k_f32(&bytes, n_rows, nb, &x);

        let mut max_abs = 0f32;
        for r in 0..n_rows {
            let mut acc = 0f64;
            for k in 0..n_cols { acc += (deq[r * n_cols + k] as f64) * (x[k] as f64); }
            max_abs = max_abs.max((gpu[r] - acc as f32).abs());
        }
        eprintln!("GPU Q6_K matvec vs candle dequant: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "GPU Q6_K matvec error too high: {max_abs}");
    }

    /// Validate the batched Q4_K GEMM (prefill) against the CPU dequant
    /// oracle on a real Q4_K weight, with M prompt rows.
    #[test]
    fn gpu_q4k_gemm_matches_cpu() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skip"); return; } };
        let n_rows = 512usize; let nb = 8usize; let n_cols = nb * QK_K; let m_rows = 8usize;
        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let qt = QTensor::quantize(&Tensor::from_vec(w, (n_rows, n_cols), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap();
        let bytes = qt.data().unwrap();
        let x: Vec<f32> = (0..m_rows * n_cols).map(|i| ((i % 31) as f32 - 15.0) * 0.04).collect();

        let gpu = ctx.gemm_q4k_f32(&bytes, n_rows, nb, &x, m_rows);

        let blocks: &[BlockQ4K] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4K, n_rows * nb) };
        let mut buf = [0f32; QK_K];
        let mut deq = vec![0f32; n_rows * n_cols];
        for n in 0..n_rows { for b in 0..nb { dequantize_q4k_block(&blocks[n * nb + b], &mut buf);
            for k in 0..QK_K { deq[n * n_cols + b * QK_K + k] = buf[k]; } } }
        let mut max_abs = 0f32;
        for m in 0..m_rows { for n in 0..n_rows {
            let mut acc = 0f64;
            for k in 0..n_cols { acc += (deq[n * n_cols + k] as f64) * (x[m * n_cols + k] as f64); }
            max_abs = max_abs.max((gpu[m * n_rows + n] - acc as f32).abs());
        } }
        eprintln!("GPU Q4_K GEMM vs CPU (M={m_rows}): max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "GPU Q4_K GEMM error too high: {max_abs}");
    }

    /// Prefill amortization: time the batched GEMM (M rows in one dispatch)
    /// vs running the matvec M separate times. Shows the compute-bound win.
    /// `cargo test --release --features gpu --lib gpu_prefill_amortization -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_prefill_amortization() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use std::time::Instant;
        let ctx = GpuContext::new().expect("GPU");
        let n_rows = 2048usize; let nb = 8usize; let n_cols = nb * 256; let m_rows = 128usize;
        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5; }
        let bytes = QTensor::quantize(&Tensor::from_vec(w, (n_rows, n_cols), &Device::Cpu).unwrap(), GgmlDType::Q4K).unwrap().data().unwrap().to_vec();
        let x: Vec<f32> = (0..m_rows * n_cols).map(|i| ((i % 31) as f32 - 15.0) * 0.04).collect();
        // warm + time GEMM (M rows, one dispatch)
        let _ = ctx.gemm_q4k_f32(&bytes, n_rows, nb, &x, m_rows);
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = ctx.gemm_q4k_f32(&bytes, n_rows, nb, &x, m_rows); }
        let gemm_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // matvec M times (one row each)
        let row0 = &x[0..n_cols];
        let _ = ctx.matmul_q4k_f32(&bytes, n_rows, nb, row0);
        let t1 = Instant::now();
        for _ in 0..iters { for m in 0..m_rows { let _ = ctx.matmul_q4k_f32(&bytes, n_rows, nb, &x[m*n_cols..(m+1)*n_cols]); } }
        let mv_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!("prefill {m_rows} rows: GEMM {gemm_ms:.2}ms vs {m_rows}x matvec {mv_ms:.2}ms  => {:.1}x faster (amortization)", mv_ms / gemm_ms);
    }

    /// THE PREFILL PAYOFF: load the real Llama-3.2-1B, run the whole prompt
    /// through `prefill_forward` in ONE batched pass, verify the last-token
    /// logits agree with candle's batched CPU forward (same prompt), confirm
    /// the GPU continues decoding correctly off the prefilled KV cache, and
    /// report prefill tok/s + TTFT.
    /// `cargo test --release --features gpu --lib gpu_prefill_vs_candle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_prefill_vs_candle_and_bench() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };

        // A multi-token prompt (token IDs need not be meaningful — both paths
        // process the identical sequence; this exercises the batched GEMMs +
        // causal SDPA at M>1).
        let prompt: Vec<u32> = vec![128000, 9906, 1917, 374, 264, 1296, 315, 279, 6500, 2068, 13, 758];
        let m = prompt.len();

        // GPU prefill (one batched pass fills the KV cache for 0..M).
        let warm = model.prefill_forward(&prompt);
        let _ = &warm;
        let t0 = Instant::now();
        let gpu_logits = model.prefill_forward(&prompt);
        let ttft = t0.elapsed();
        let gpu_first = argmax(&gpu_logits);

        // GPU continues decoding off the prefilled cache (pos starts at M).
        let n_gen = 12usize;
        let mut gpu_gen = vec![gpu_first];
        let mut next = gpu_first;
        let mut pos = m;
        for _ in 1..n_gen { next = model.forward_argmax(next, pos); gpu_gen.push(next); pos += 1; }

        // Candle CPU reference: batched forward over the prompt, then greedy.
        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
        let clog = cb.forward_logits(&prompt).unwrap();
        let cand_first = argmax(&clog);
        let mut cand_gen = vec![cand_first];
        let mut cnext = cand_first;
        for _ in 1..n_gen { let l = cb.forward_logits(&[cnext]).unwrap(); cnext = argmax(&l); cand_gen.push(cnext); }

        // Last-token logits comparison (the pure prefill check).
        let n = gpu_logits.len().min(clog.len());
        let (mut dot, mut ng, mut nc, mut max_abs) = (0f64, 0f64, 0f64, 0f32);
        for i in 0..n {
            dot += gpu_logits[i] as f64 * clog[i] as f64;
            ng += (gpu_logits[i] as f64).powi(2);
            nc += (clog[i] as f64).powi(2);
            max_abs = max_abs.max((gpu_logits[i] - clog[i]).abs());
        }
        let cosine = dot / (ng.sqrt() * nc.sqrt());
        eprintln!("prefill TTFT: {:.1} ms for {m} tokens => {:.1} tok/s prefill", ttft.as_secs_f64() * 1e3, m as f64 / ttft.as_secs_f64());
        eprintln!("last-token logits: argmax gpu={gpu_first} candle={cand_first}, cosine={cosine:.5}, max_abs={max_abs:.4}");
        eprintln!("GPU gen:    {gpu_gen:?}");
        eprintln!("candle gen: {cand_gen:?}");
        let agree = gpu_gen.iter().zip(&cand_gen).take_while(|(a, b)| a == b).count();
        eprintln!("GPU/candle agree on first {agree}/{n_gen} tokens after prefill");

        // Prefill latency/throughput sweep (averaged — single-shot is noisy).
        // prefill_forward reads back the logits, so each call self-drains the
        // GPU; timing N calls / N is a fair per-call number. Compare each
        // against the sequential-decode cost (M * decode_ms) for the same M.
        let decode_ms = {
            // Fill the cache 0..16 so decode SDPA scans a realistic short
            // context (decoding at pos>>0 over unfilled cache is meaningless).
            let p: Vec<u32> = (0..16).map(|i| (i as u32 * 977 + 11) % model.vocab as u32).collect();
            let _ = model.prefill_forward(&p);
            let mut tok = 1u32; let mut pos = 16usize;
            let _ = model.forward_argmax(tok, pos); pos += 1;  // warm
            let t = Instant::now();
            for _ in 0..16 { tok = model.forward_argmax(tok, pos); pos += 1; }
            t.elapsed().as_secs_f64() * 1e3 / 16.0
        };
        eprintln!("(decode reference: {decode_ms:.2} ms/token)");
        for &mm in &[12usize, 32, 128, 256] {
            let p: Vec<u32> = (0..mm).map(|i| (i as u32 * 977 + 11) % model.vocab as u32).collect();
            let _ = model.prefill_forward(&p);           // warm
            let iters = 5;
            let t = Instant::now();
            for _ in 0..iters { let _ = model.prefill_forward(&p); }
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            eprintln!("prefill M={mm:>3}: {ms:6.1} ms ({:5.0} tok/s) | {mm} seq-decodes ~{:6.1} ms => {:.2}x",
                mm as f64 / (ms / 1e3), mm as f64 * decode_ms, (mm as f64 * decode_ms) / ms);
        }

        assert_eq!(gpu_first, cand_first, "prefill last-token argmax disagrees with candle");
        // Greedy output is byte-identical to candle (the strong check); cosine
        // just guards against silent drift (f32 GEMM noise over 16 layers ~1e-3).
        assert!(cosine > 0.998, "prefill logits diverge from candle (cosine {cosine:.5})");
        assert!(agree >= 6, "GPU prefill+decode diverges from candle too early ({agree}); cache or wiring bug");
    }

    /// Diagnostic: dump every GGUF tensor's ggml dtype so we know which
    /// weights are Q4_K vs Q6_K vs F32 before building the GPU loader.
    #[test]
    #[ignore]
    fn dump_gguf_dtypes() {
        use candle_core::quantized::gguf_file;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        let mut f = std::fs::File::open(path).unwrap();
        let ct = gguf_file::Content::read(&mut f).unwrap();
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for (name, info) in &ct.tensor_infos {
            *counts.entry(format!("{:?}", info.ggml_dtype)).or_default() += 1;
            // print a few representative tensors
            if name.contains("blk.0.") || !name.starts_with("blk.") {
                eprintln!("{name}: {:?} {:?}", info.ggml_dtype, info.shape.dims());
            }
        }
        eprintln!("--- dtype counts across all tensors ---");
        for (dt, c) in &counts { eprintln!("  {dt}: {c}"); }
    }

    /// Validate the GPU RMSNorm against the explicit CPU formula
    /// `y = x * rsqrt(mean(x^2)+eps) * weight` on n_embd width.
    #[test]
    fn gpu_rmsnorm_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let n = 2048usize;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..n).map(|i| ((i % 53) as f32 - 26.0) * 0.13).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + ((i % 7) as f32) * 0.1).collect();

        let gpu = ctx.rmsnorm_decode(&x, &w, eps);

        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let cpu: Vec<f32> = (0..n).map(|i| x[i] * inv * w[i]).collect();

        let mut max_abs = 0f32;
        for i in 0..n { max_abs = max_abs.max((gpu[i] - cpu[i]).abs()); }
        eprintln!("GPU RMSNorm vs CPU: max_abs_err = {max_abs:.6}");
        assert!(max_abs < 1e-4, "GPU RMSNorm mismatch: {max_abs}");
    }

    /// End-to-end validation of the full GPU-resident attention block
    /// (Q/K/V proj → RoPE → resident-cache append → GQA SDPA → O proj)
    /// against an independent CPU reference, with a pre-filled prior KV
    /// cache. This exercises the cache-append copy + cross-pass barriers.
    #[test]
    fn gpu_attention_block_matches_cpu() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() {
            Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let (n_embd, n_head, n_kv_head, head_dim) = (512usize, 8usize, 2usize, 64usize);
        let kv_dim = n_kv_head * head_dim;   // 128
        let attn_dim = n_head * head_dim;    // 512
        let pos = 3usize;
        let seq_len = pos + 1;
        let half = head_dim / 2;
        let base = 500000f32;
        let group = n_head / n_kv_head;

        let mk_w = |rows: usize, cols: usize, seed: i64| -> Vec<u8> {
            let mut w = vec![0f32; rows * cols];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(seed) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.4; }
            QTensor::quantize(&Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap(), GgmlDType::Q4K)
                .unwrap().data().unwrap().to_vec()
        };
        let wq_b = mk_w(attn_dim, n_embd, 11);
        let wk_b = mk_w(kv_dim, n_embd, 22);
        let wv_b = mk_w(kv_dim, n_embd, 33);
        let wo_b = mk_w(n_embd, attn_dim, 44);
        let x: Vec<f32> = (0..n_embd).map(|i| ((i % 29) as f32 - 14.0) * 0.05).collect();
        let cos: Vec<f32> = (0..half).map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).cos() }).collect();
        let sin: Vec<f32> = (0..half).map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).sin() }).collect();
        let k_prior: Vec<f32> = (0..pos * kv_dim).map(|i| ((i % 37) as f32 - 18.0) * 0.03).collect();
        let v_prior: Vec<f32> = (0..pos * kv_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.025).collect();

        // GPU.
        let wq = ctx.upload_q4k(&wq_b, attn_dim, n_embd / QK_K);
        let wk = ctx.upload_q4k(&wk_b, kv_dim, n_embd / QK_K);
        let wv = ctx.upload_q4k(&wv_b, kv_dim, n_embd / QK_K);
        let wo = ctx.upload_q4k(&wo_b, n_embd, attn_dim / QK_K);
        let k_cache = ctx.alloc_activation(seq_len * kv_dim, false);
        let v_cache = ctx.alloc_activation(seq_len * kv_dim, false);
        ctx.queue.write_buffer(&k_cache, 0, bytemuck::cast_slice(&k_prior));
        ctx.queue.write_buffer(&v_cache, 0, bytemuck::cast_slice(&v_prior));
        let gpu = ctx.attention_decode(&x, &wq, &wk, &wv, &wo, &cos, &sin, &k_cache, &v_cache, pos, n_head, n_kv_head, head_dim);

        // CPU reference.
        let as_blk = |b: &[u8], n: usize| -> &[BlockQ4K] { unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4K, n) } };
        let wq_blk = as_blk(&wq_b, attn_dim * (n_embd / QK_K));
        let wk_blk = as_blk(&wk_b, kv_dim * (n_embd / QK_K));
        let wv_blk = as_blk(&wv_b, kv_dim * (n_embd / QK_K));
        let wo_blk = as_blk(&wo_b, n_embd * (attn_dim / QK_K));
        let mut buf = [0f32; QK_K];
        let dot = |blk: &[BlockQ4K], r: usize, nb: usize, v: &[f32], s: &mut [f32; QK_K]| {
            let mut a = 0f64;
            for b in 0..nb { dequantize_q4k_block(&blk[r * nb + b], s); for k in 0..QK_K { a += (s[k] as f64) * (v[b * QK_K + k] as f64); } }
            a as f32
        };
        let nb_e = n_embd / QK_K;
        let nb_a = attn_dim / QK_K;
        let q_raw: Vec<f32> = (0..attn_dim).map(|r| dot(wq_blk, r, nb_e, &x, &mut buf)).collect();
        let k_raw: Vec<f32> = (0..kv_dim).map(|r| dot(wk_blk, r, nb_e, &x, &mut buf)).collect();
        let v_cur: Vec<f32> = (0..kv_dim).map(|r| dot(wv_blk, r, nb_e, &x, &mut buf)).collect();
        let dev = Device::Cpu;
        let rope = |data: &[f32], nh: usize| -> Vec<f32> {
            let t = Tensor::from_vec(data.to_vec(), (1, nh, 1, head_dim), &dev).unwrap();
            let c = Tensor::from_vec(cos.clone(), (1, half), &dev).unwrap();
            let s = Tensor::from_vec(sin.clone(), (1, half), &dev).unwrap();
            candle_nn::rotary_emb::rope_i(&t.contiguous().unwrap(), &c, &s).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        let q = rope(&q_raw, n_head);
        let k_cur = rope(&k_raw, n_kv_head);
        let mut k_full = k_prior.clone(); k_full.extend_from_slice(&k_cur);
        let mut v_full = v_prior.clone(); v_full.extend_from_slice(&v_cur);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn = vec![0f32; attn_dim];
        for h in 0..n_head {
            let kvh = h / group; let qb = h * head_dim;
            let mut sc = vec![0f32; seq_len]; let mut mx = f32::NEG_INFINITY;
            for t in 0..seq_len { let kb = (t * n_kv_head + kvh) * head_dim; let mut s = 0f32; for d in 0..head_dim { s += q[qb + d] * k_full[kb + d]; } s *= scale; sc[t] = s; if s > mx { mx = s; } }
            let mut sum = 0f32; for t in 0..seq_len { sc[t] = (sc[t] - mx).exp(); sum += sc[t]; }
            for t in 0..seq_len { let w = sc[t] / sum; let vb = (t * n_kv_head + kvh) * head_dim; for d in 0..head_dim { attn[qb + d] += w * v_full[vb + d]; } }
        }
        let cpu: Vec<f32> = (0..n_embd).map(|r| dot(wo_blk, r, nb_a, &attn, &mut buf)).collect();

        let mut max_abs = 0f32;
        for i in 0..n_embd { max_abs = max_abs.max((gpu[i] - cpu[i]).abs()); }
        eprintln!("GPU attention block vs CPU: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "GPU attention block mismatch: {max_abs}");
    }

    /// End-to-end validation of a full GPU decode layer (the orchestration
    /// unit): `x += attn(norm(x)); x += ffn(norm(x))`, resident, one command
    /// buffer, vs a CPU reference combining RMSNorm + attention + FFN.
    #[test]
    fn gpu_decode_layer_matches_cpu() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() {
            Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let (n_embd, n_head, n_kv_head, head_dim, n_inter) = (512usize, 8, 2, 64, 1024);
        let (kv_dim, attn_dim) = (n_kv_head * head_dim, n_head * head_dim);
        let pos = 3usize; let seq_len = pos + 1; let half = head_dim / 2;
        let base = 500000f32; let eps = 1e-5f32; let group = n_head / n_kv_head;

        let mk_w = |rows: usize, cols: usize, seed: i64| -> Vec<u8> {
            let mut w = vec![0f32; rows * cols];
            for i in 0..w.len() { w[i] = (((i as i64).wrapping_mul(seed) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.4; }
            QTensor::quantize(&Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap(), GgmlDType::Q4K)
                .unwrap().data().unwrap().to_vec()
        };
        let (wq_b, wk_b, wv_b, wo_b) = (mk_w(attn_dim, n_embd, 11), mk_w(kv_dim, n_embd, 22), mk_w(kv_dim, n_embd, 33), mk_w(n_embd, attn_dim, 44));
        let (w1_b, w2_b, w3_b) = (mk_w(n_inter, n_embd, 55), mk_w(n_embd, n_inter, 66), mk_w(n_inter, n_embd, 77));
        let x: Vec<f32> = (0..n_embd).map(|i| ((i % 29) as f32 - 14.0) * 0.05).collect();
        let an_w: Vec<f32> = (0..n_embd).map(|i| 0.6 + (i % 5) as f32 * 0.07).collect();
        let fn_w: Vec<f32> = (0..n_embd).map(|i| 0.5 + (i % 9) as f32 * 0.05).collect();
        let cos: Vec<f32> = (0..half).map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).cos() }).collect();
        let sin: Vec<f32> = (0..half).map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).sin() }).collect();
        let k_prior: Vec<f32> = (0..pos * kv_dim).map(|i| ((i % 37) as f32 - 18.0) * 0.03).collect();
        let v_prior: Vec<f32> = (0..pos * kv_dim).map(|i| ((i % 41) as f32 - 20.0) * 0.025).collect();

        // GPU.
        let up = |b: &[u8], r, c| ctx.upload_q4k(b, r, c / QK_K);
        let (wq, wk, wv, wo) = (up(&wq_b, attn_dim, n_embd), up(&wk_b, kv_dim, n_embd), up(&wv_b, kv_dim, n_embd), up(&wo_b, n_embd, attn_dim));
        let (gw1, gw2, gw3) = (up(&w1_b, n_inter, n_embd), up(&w2_b, n_embd, n_inter), up(&w3_b, n_inter, n_embd));
        let k_cache = ctx.alloc_activation(seq_len * kv_dim, false);
        let v_cache = ctx.alloc_activation(seq_len * kv_dim, false);
        ctx.queue.write_buffer(&k_cache, 0, bytemuck::cast_slice(&k_prior));
        ctx.queue.write_buffer(&v_cache, 0, bytemuck::cast_slice(&v_prior));
        let gpu = ctx.decode_layer_once(&x, &an_w, &fn_w, &wq, &wk, &wv, &wo, &gw1, &gw2, &gw3,
            &cos, &sin, &k_cache, &v_cache, pos, n_head, n_kv_head, head_dim, eps);

        // CPU reference.
        let as_blk = |b: &[u8], n| -> &[BlockQ4K] { unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4K, n) } };
        let mut buf = [0f32; QK_K];
        let dot = |blk: &[BlockQ4K], r: usize, nb: usize, v: &[f32], s: &mut [f32; QK_K]| {
            let mut a = 0f64; for b in 0..nb { dequantize_q4k_block(&blk[r * nb + b], s); for k in 0..QK_K { a += (s[k] as f64) * (v[b * QK_K + k] as f64); } } a as f32 };
        let matvec = |blk_b: &[u8], rows: usize, cols: usize, v: &[f32], s: &mut [f32; QK_K]| -> Vec<f32> {
            let nb = cols / QK_K; let blk = as_blk(blk_b, rows * nb);
            (0..rows).map(|r| dot(blk, r, nb, v, s)).collect() };
        let rmsnorm = |v: &[f32], w: &[f32]| -> Vec<f32> {
            let ms: f32 = v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
            let inv = 1.0 / (ms + eps).sqrt(); (0..v.len()).map(|i| v[i] * inv * w[i]).collect() };
        let dev = Device::Cpu;
        let rope = |data: &[f32], nh: usize| -> Vec<f32> {
            let t = Tensor::from_vec(data.to_vec(), (1, nh, 1, head_dim), &dev).unwrap();
            let c = Tensor::from_vec(cos.clone(), (1, half), &dev).unwrap();
            let s = Tensor::from_vec(sin.clone(), (1, half), &dev).unwrap();
            candle_nn::rotary_emb::rope_i(&t.contiguous().unwrap(), &c, &s).unwrap().flatten_all().unwrap().to_vec1().unwrap() };

        let n1 = rmsnorm(&x, &an_w);
        let q = rope(&matvec(&wq_b, attn_dim, n_embd, &n1, &mut buf), n_head);
        let k_cur = rope(&matvec(&wk_b, kv_dim, n_embd, &n1, &mut buf), n_kv_head);
        let v_cur = matvec(&wv_b, kv_dim, n_embd, &n1, &mut buf);
        let mut k_full = k_prior.clone(); k_full.extend_from_slice(&k_cur);
        let mut v_full = v_prior.clone(); v_full.extend_from_slice(&v_cur);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn = vec![0f32; attn_dim];
        for h in 0..n_head {
            let kvh = h / group; let qb = h * head_dim;
            let mut sc = vec![0f32; seq_len]; let mut mx = f32::NEG_INFINITY;
            for t in 0..seq_len { let kb = (t * n_kv_head + kvh) * head_dim; let mut s = 0f32; for d in 0..head_dim { s += q[qb + d] * k_full[kb + d]; } s *= scale; sc[t] = s; if s > mx { mx = s; } }
            let mut sum = 0f32; for t in 0..seq_len { sc[t] = (sc[t] - mx).exp(); sum += sc[t]; }
            for t in 0..seq_len { let w = sc[t] / sum; let vb = (t * n_kv_head + kvh) * head_dim; for d in 0..head_dim { attn[qb + d] += w * v_full[vb + d]; } }
        }
        let o = matvec(&wo_b, n_embd, attn_dim, &attn, &mut buf);
        let x1: Vec<f32> = (0..n_embd).map(|i| x[i] + o[i]).collect();
        let n2 = rmsnorm(&x1, &fn_w);
        let g = matvec(&w1_b, n_inter, n_embd, &n2, &mut buf);
        let u = matvec(&w3_b, n_inter, n_embd, &n2, &mut buf);
        let hh: Vec<f32> = (0..n_inter).map(|i| (g[i] / (1.0 + (-g[i]).exp())) * u[i]).collect();
        let ffn = matvec(&w2_b, n_embd, n_inter, &hh, &mut buf);
        let cpu: Vec<f32> = (0..n_embd).map(|i| x1[i] + ffn[i]).collect();

        let mut max_abs = 0f32;
        for i in 0..n_embd { max_abs = max_abs.max((gpu[i] - cpu[i]).abs()); }
        eprintln!("GPU decode layer vs CPU: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "GPU decode layer mismatch: {max_abs}");
    }

    /// Validate the GPU interleaved-RoPE shader against candle's actual
    /// `rope_i` (the exact op the model uses), on Llama-3.2-1B head shape.
    #[test]
    fn gpu_rope_matches_candle() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };
        let n_head = 8usize;
        let head_dim = 64usize;
        let half = head_dim / 2;
        let base = 500000f32; // Llama-3.2 rope_freq_base (value irrelevant — both sides use same cos/sin)
        let pos = 7usize;

        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.07).collect();
        let cos: Vec<f32> = (0..half)
            .map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).cos() })
            .collect();
        let sin: Vec<f32> = (0..half)
            .map(|j| { let th = 1.0 / base.powf((2 * j) as f32 / head_dim as f32); (pos as f32 * th).sin() })
            .collect();

        let gpu = ctx.rope_decode(&x, &cos, &sin, n_head, head_dim);

        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let xt = Tensor::from_vec(x.clone(), (1, n_head, 1, head_dim), &dev).unwrap();
        let ct = Tensor::from_vec(cos.clone(), (1, half), &dev).unwrap();
        let st = Tensor::from_vec(sin.clone(), (1, half), &dev).unwrap();
        let refv: Vec<f32> = candle_nn::rotary_emb::rope_i(&xt.contiguous().unwrap(), &ct, &st)
            .unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let mut max_abs = 0f32;
        for i in 0..x.len() { max_abs = max_abs.max((gpu[i] - refv[i]).abs()); }
        eprintln!("GPU rope_i vs candle: max_abs_err = {max_abs:.6}");
        assert!(max_abs < 1e-4, "GPU RoPE mismatch: {max_abs}");
    }

    /// Chain two Q4_K matmuls on the GPU — `out = W2 · (W1 · x)` — with
    /// the intermediate buffer NEVER leaving GPU memory (no readback
    /// between matmuls). This is the core primitive of the resident
    /// forward: weights uploaded once, activations stay GPU-resident
    /// across ops. Validated against the CPU dequant oracle.
    #[test]
    fn gpu_chained_matmuls_stay_resident() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };

        // W1: 512x512, W2: 256x512 (W2 input dim == W1 output dim == 512).
        let (n1, c1) = (512usize, 512usize);
        let (n2, c2) = (256usize, 512usize);
        assert_eq!(c2, n1);
        let nb1 = c1 / QK_K;
        let nb2 = c2 / QK_K;

        let mk_w = |rows: usize, cols: usize, seed: i64| -> Vec<u8> {
            let mut w = vec![0f32; rows * cols];
            for i in 0..w.len() {
                w[i] = (((i as i64).wrapping_mul(seed) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.4;
            }
            let qt = QTensor::quantize(
                &Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap(),
                GgmlDType::Q4K,
            ).unwrap();
            qt.data().unwrap().to_vec()
        };
        let w1_bytes = mk_w(n1, c1, 2654435761);
        let w2_bytes = mk_w(n2, c2, 40503);
        let x: Vec<f32> = (0..c1).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

        // GPU: upload weights once, run chained matmuls in one submission.
        let w1 = ctx.upload_q4k(&w1_bytes, n1, nb1);
        let w2 = ctx.upload_q4k(&w2_bytes, n2, nb2);
        use wgpu::util::DeviceExt;
        let x_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"), contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mid_buf = ctx.alloc_activation(n1, false);     // stays on GPU
        let out_buf = ctx.alloc_activation(n2, true);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        ctx.record_matvec(&mut enc, &w1, &x_buf, &mid_buf);
        ctx.record_matvec(&mut enc, &w2, &mid_buf, &out_buf);
        ctx.queue.submit([enc.finish()]);
        let gpu_out = ctx.read_buffer(&out_buf, n2);

        // CPU oracle: dequant-then-dot, twice.
        let w1_blk: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(w1_bytes.as_ptr() as *const BlockQ4K, n1 * nb1)
        };
        let w2_blk: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(w2_bytes.as_ptr() as *const BlockQ4K, n2 * nb2)
        };
        let mut buf = [0f32; QK_K];
        let dot_row = |blocks: &[BlockQ4K], r: usize, nb: usize, v: &[f32], scratch: &mut [f32; QK_K]| {
            let mut acc = 0f64;
            for b in 0..nb {
                dequantize_q4k_block(&blocks[r * nb + b], scratch);
                for k in 0..QK_K { acc += (scratch[k] as f64) * (v[b * QK_K + k] as f64); }
            }
            acc as f32
        };
        let mid: Vec<f32> = (0..n1).map(|r| dot_row(w1_blk, r, nb1, &x, &mut buf)).collect();
        let cpu_out: Vec<f32> = (0..n2).map(|r| dot_row(w2_blk, r, nb2, &mid, &mut buf)).collect();

        let mut max_abs = 0f32;
        for r in 0..n2 { max_abs = max_abs.max((gpu_out[r] - cpu_out[r]).abs()); }
        eprintln!("GPU chained matmuls (resident intermediate) vs CPU: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "chained GPU matmul error too high: {max_abs}");
    }

    /// Validate a full GPU-resident FFN block `out = w2(silu(w1·x)*w3·x)`
    /// against a CPU oracle on real Q4_K weights. Proves the matmul +
    /// elementwise chain (the dominant decode compute) is correct end to
    /// end on the iGPU.
    #[test]
    fn gpu_ffn_block_matches_cpu() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use crate::backend::candle::q4k_repack::{BlockQ4K, dequantize_q4k_block, QK_K};

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no GPU ({e}); skipping"); return; }
        };

        let n_embd = 512usize;
        let n_inter = 1024usize;

        let mk_w = |rows: usize, cols: usize, seed: i64| -> Vec<u8> {
            let mut w = vec![0f32; rows * cols];
            for i in 0..w.len() {
                w[i] = (((i as i64).wrapping_mul(seed) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.4;
            }
            QTensor::quantize(
                &Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap(),
                GgmlDType::Q4K,
            ).unwrap().data().unwrap().to_vec()
        };
        let w1_b = mk_w(n_inter, n_embd, 2654435761); // gate
        let w3_b = mk_w(n_inter, n_embd, 40503);      // up
        let w2_b = mk_w(n_embd, n_inter, 2246822519); // down
        let x: Vec<f32> = (0..n_embd).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();

        let w1 = ctx.upload_q4k(&w1_b, n_inter, n_embd / QK_K);
        let w3 = ctx.upload_q4k(&w3_b, n_inter, n_embd / QK_K);
        let w2 = ctx.upload_q4k(&w2_b, n_embd, n_inter / QK_K);
        let gpu_out = ctx.ffn_decode(&x, &w1, &w2, &w3);

        // CPU oracle.
        let as_blk = |b: &[u8], n: usize| -> &[BlockQ4K] {
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4K, n) }
        };
        let w1_blk = as_blk(&w1_b, n_inter * (n_embd / QK_K));
        let w3_blk = as_blk(&w3_b, n_inter * (n_embd / QK_K));
        let w2_blk = as_blk(&w2_b, n_embd * (n_inter / QK_K));
        let mut buf = [0f32; QK_K];
        let dot = |blk: &[BlockQ4K], r: usize, nb: usize, v: &[f32], s: &mut [f32; QK_K]| {
            let mut a = 0f64;
            for b in 0..nb { dequantize_q4k_block(&blk[r * nb + b], s);
                for k in 0..QK_K { a += (s[k] as f64) * (v[b * QK_K + k] as f64); } }
            a as f32
        };
        let nb_in = n_embd / QK_K;
        let nb_h = n_inter / QK_K;
        let mut h = vec![0f32; n_inter];
        for r in 0..n_inter {
            let g = dot(w1_blk, r, nb_in, &x, &mut buf);
            let u = dot(w3_blk, r, nb_in, &x, &mut buf);
            h[r] = (g / (1.0 + (-g).exp())) * u;
        }
        let cpu_out: Vec<f32> = (0..n_embd).map(|r| dot(w2_blk, r, nb_h, &h, &mut buf)).collect();

        let mut max_abs = 0f32;
        for r in 0..n_embd { max_abs = max_abs.max((gpu_out[r] - cpu_out[r]).abs()); }
        eprintln!("GPU FFN block vs CPU: max_abs_err = {max_abs:.5}");
        assert!(max_abs < 0.05, "GPU FFN error too high: {max_abs}");
    }

    /// Measure resident-weight Q4_K mat-vec throughput on the iGPU
    /// (weights uploaded once, dispatched many times). Answers the
    /// pivotal question: does the iGPU reach more memory bandwidth than
    /// the CPU's ~55 GB/s? Run with:
    /// `cargo test --release --features gpu --lib gpu_q4k_bandwidth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_q4k_bandwidth() {
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        use wgpu::util::DeviceExt;
        use std::time::Instant;

        let ctx = GpuContext::new().expect("GPU");
        // LM-head shape: big enough (~148 MB) to defeat on-chip caches and
        // measure true memory streaming.
        let n_rows = 128256usize;
        let nb = 8usize;
        let n_cols = nb * 256;

        let mut w = vec![0f32; n_rows * n_cols];
        for i in 0..w.len() {
            w[i] = (((i as i64).wrapping_mul(2654435761) & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.5;
        }
        let qt = QTensor::quantize(
            &Tensor::from_vec(w, (n_rows, n_cols), &Device::Cpu).unwrap(),
            GgmlDType::Q4K,
        ).unwrap();
        let bytes = qt.data().unwrap();
        let x = vec![0.7f32; n_cols];

        let w_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: &bytes, usage: wgpu::BufferUsages::STORAGE });
        let x_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&x), usage: wgpu::BufferUsages::STORAGE });
        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (n_rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let gxb = (n_rows as u32).min(65535);
        let p_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[n_rows as u32, nb as u32, gxb, 0u32, 0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM });
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(Q4K_MATVEC_WGSL.into()) });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &shader, entry_point: "main",
            compilation_options: Default::default(), cache: None });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ] });

        let dispatch = |label: &str| {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
              pass.dispatch_workgroups(gxb, (n_rows as u32).div_ceil(gxb), 1); }
            let _ = label;
            ctx.queue.submit([enc.finish()]);
        };

        dispatch("warmup");
        ctx.device.poll(wgpu::Maintain::Wait);

        let iters = 50u32;
        let t0 = Instant::now();
        for _ in 0..iters { dispatch("timed"); }
        ctx.device.poll(wgpu::Maintain::Wait);
        let dt = t0.elapsed();

        let gbps = (bytes.len() as f64 * iters as f64) / dt.as_secs_f64() / 1e9;
        let ms = dt.as_secs_f64() * 1000.0 / iters as f64;
        eprintln!(
            "GPU Q4_K resident matvec ({} MB weights): {:.1} GB/s, {:.3} ms/matvec  [CPU ref ~55 GB/s]",
            bytes.len() / (1024 * 1024), gbps, ms,
        );
    }

    /// AGGREGATE SERVING THROUGHPUT: M concurrent decode streams coalesced into
    /// one forward. Validates batched output is bit-identical to single-stream,
    /// then measures aggregate tok/s vs concurrency — the compute-bound
    /// amortization that single-stream (bandwidth-bound) decode can't reach.
    /// `cargo test --release --features gpu --lib gpu_batched_decode -- --ignored --nocapture`
    #[test]
    #[ignore]
    /// Continuous-batching correctness + aggregate throughput: run N sequences
    /// each alone (reference), then all together through one ContinuousBatcher,
    /// and assert every sequence's greedy output is identical (the batch must not
    /// couple sequences) + report aggregate tok/s.
    /// `cargo test --release --features gpu --lib gpu_continuous_batch -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_continuous_batch() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let prompts: Vec<Vec<u32>> = vec![
            vec![128000, 791, 6864, 315, 9822, 374],
            vec![128000, 15724, 374, 264, 1296],
            vec![128000, 9906, 1917, 11],
            vec![128000, 791, 4205, 315, 2324, 374],
        ];
        let (k, eos, max_seq) = (10usize, u32::MAX, 256usize); // eos disabled → all run exactly k tokens

        // Reference: each sequence alone through its own batcher (m=1).
        let mut refs: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut toks = vec![cb.admit(0, p, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_id, t, _done) in cb.step() { toks.push(t); } }
            refs.push(toks);
        }
        // Batched: all sequences in one batcher.
        let mut cb = ContinuousBatcher::new(&model, prompts.len(), max_seq);
        let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        for (i, p) in prompts.iter().enumerate() { let g = cb.admit(i as u64, p, k, eos).unwrap().0; got.entry(i as u64).or_default().push(g); }
        let t0 = Instant::now();
        while cb.active_len() > 0 { for (id, t, _done) in cb.step() { got.get_mut(&id).unwrap().push(t); } }
        let dt = t0.elapsed().as_secs_f64();
        let total: usize = got.values().map(|v| v.len() - 1).sum(); // exclude admit token; count decode steps
        let mut all_match = true;
        for (i, r) in refs.iter().enumerate() {
            let g = &got[&(i as u64)];
            if g != r { all_match = false; eprintln!("seq {i} MISMATCH:\n  batch={g:?}\n  ref  ={r:?}"); }
        }
        eprintln!("continuous batch: {} seqs x {k} tok, decode {total} tok in {dt:.2}s => {:.0} tok/s aggregate", prompts.len(), total as f64 / dt);
        assert!(all_match, "batched sequences diverged from single-stream");
        eprintln!("all {} sequences match single-stream decode", prompts.len());
    }

    /// End-to-end GpuBatchServer: spawn the serving thread, submit several
    /// requests, drain each token channel, and assert every streamed result is
    /// identical to a single-stream reference — proving the channel/thread/admit
    /// machinery preserves per-sequence correctness under concurrency.
    /// `cargo test --release --features gpu --lib gpu_batch_server -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_batch_server() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let prompts: Vec<Vec<u32>> = vec![
            vec![128000, 791, 6864, 315, 9822, 374],
            vec![128000, 15724, 374, 264, 1296],
            vec![128000, 9906, 1917, 11],
            vec![128000, 791, 4205, 315, 2324, 374],
        ];
        let (k, eos, max_seq) = (12usize, u32::MAX, 256usize);

        // Single-stream reference for each prompt (borrows model; done before move).
        let mut refs: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut toks = vec![cb.admit(0, p, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, t, _) in cb.step() { toks.push(t); } }
            refs.push(toks);
        }

        // Spawn the server (moves the model onto its thread) and submit all reqs.
        let server = GpuBatchServer::spawn(model, prompts.len(), max_seq);
        let rxs: Vec<_> = prompts.iter().map(|p| server.submit(p.clone(), k, eos, SamplingParams::greedy(), 0).expect("submit")).collect();
        let mut got: Vec<Vec<u32>> = Vec::new();
        for mut rx in rxs {
            let mut toks = Vec::new();
            while let Some(item) = rx.blocking_recv() {
                match item { Some(t) => toks.push(t), None => break } // None = done sentinel
            }
            got.push(toks);
        }
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(&got[i], r, "server seq {i} diverged from single-stream");
        }
        eprintln!("GpuBatchServer: {} concurrent requests, all match single-stream ✓", prompts.len());
    }

    /// Batched prefill must be bit-identical to sequential (token-by-token)
    /// prefill — both the first generated token AND the KV it writes (verified
    /// by decoding several tokens out of each slot). Uses a >PREFILL_CHUNK prompt
    /// so the chunked path (chunk k attends earlier chunks' resident KV) is hit.
    /// `cargo test --release --features gpu --lib gpu_prefill_slot -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_prefill_slot() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let dec = model.batched_decoder(4, 512);

        // A 201-token prompt → two chunks (128 + 73).
        let mut prompt = vec![128000u32];
        for i in 0..200u32 { prompt.push(1000 + (i % 64)); }
        let k = 8usize;

        // Slot 0: batched prefill, then decode k tokens.
        let t0 = Instant::now();
        let g0 = dec.prefill_slot(&prompt, 0);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (mut batched, mut tok, mut pos) = (vec![g0], g0, prompt.len() as u32);
        for _ in 0..k { let n = dec.step_slotted(&[tok], &[pos], &[0])[0]; batched.push(n); tok = n; pos += 1; }

        // Slot 1: sequential prefill (the reference), then decode k tokens.
        let t1 = Instant::now();
        let mut g = 0u32;
        for (i, &t) in prompt.iter().enumerate() { g = dec.step_slotted(&[t], &[i as u32], &[1])[0]; }
        let seq_ms = t1.elapsed().as_secs_f64() * 1e3;
        let (mut seq, mut tok, mut pos) = (vec![g], g, prompt.len() as u32);
        for _ in 0..k { let n = dec.step_slotted(&[tok], &[pos], &[1])[0]; seq.push(n); tok = n; pos += 1; }

        eprintln!("prefill {} tok: batched {prefill_ms:.0} ms vs sequential {seq_ms:.0} ms => {:.1}x faster", prompt.len(), seq_ms / prefill_ms);
        assert_eq!(batched, seq, "batched prefill diverged from sequential prefill");
        eprintln!("batched prefill bit-identical to sequential (first tok + {k} decoded) ✓");
    }

    /// Paged KV overcommit: serve more concurrent sequences than a contiguous
    /// max_seq-per-slot reservation could fit, in a small shared block pool —
    /// and prove (a) every sequence is still bit-identical to single-stream and
    /// (b) blocks are recycled back to the pool on eviction.
    /// `cargo test --release --features gpu --lib gpu_paged_overcommit -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_paged_overcommit() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (m_max, max_seq, n_blocks) = (8usize, 512usize, 64usize);
        let contiguous_blocks = m_max * max_seq.div_ceil(DEFAULT_BLOCK_SIZE); // what a per-slot reservation needs
        let (k, eos) = (12usize, u32::MAX);
        let prompts: Vec<Vec<u32>> = (0..8u32).map(|i| vec![128000, 1000 + i, 2000, 3000, 4000]).collect();

        // Single-stream reference per prompt (full pool, one slot).
        let mut refs: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, p, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            refs.push(t);
        }

        // Overcommitted pool: 8 concurrent sequences through a 64-block pool.
        let mut cb = ContinuousBatcher::with_pool(&model, m_max, max_seq, n_blocks);
        let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        for (i, p) in prompts.iter().enumerate() {
            let g = cb.admit(i as u64, p, k, eos).expect("admit (pool too small?)").0;
            got.entry(i as u64).or_default().push(g);
        }
        let (free_at_peak, total) = cb.block_pool();
        while cb.active_len() > 0 { for (id, x, _) in cb.step() { got.get_mut(&id).unwrap().push(x); } }
        let (free_after, _) = cb.block_pool();

        for (i, r) in refs.iter().enumerate() {
            assert_eq!(&got[&(i as u64)], r, "paged seq {i} diverged from single-stream");
        }
        eprintln!("paged pool: {n_blocks} blocks (a contiguous {m_max}×{max_seq} reservation needs {contiguous_blocks}) — {:.0}x less KV memory", contiguous_blocks as f64 / n_blocks as f64);
        eprintln!("  served {} concurrent sequences; peak use {}/{} blocks; recycled to {}/{} after eviction", prompts.len(), total - free_at_peak, total, free_after, total);
        assert_eq!(free_after, total, "blocks were not fully recycled after all sequences finished");
        eprintln!("all {} sequences correct on the overcommitted pool, blocks fully recycled ✓", prompts.len());
    }

    /// Cross-request prefix-cache reuse: a second request sharing a long prefix
    /// with an already-processed one reuses its KV blocks (skips that prefill)
    /// and produces output bit-identical to computing it cold.
    /// `cargo test --release --features gpu --lib gpu_prefix_cache -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_prefix_cache() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (max_seq, k, eos) = (512usize, 10usize, u32::MAX);

        // 40-token shared prefix (> 2 blocks of 16) + distinct 3-token suffixes.
        let shared: Vec<u32> = (0..40u32).map(|i| 1000 + i).collect();
        let mut a = shared.clone(); a.extend([2001, 2002, 2003]);
        let mut b = shared.clone(); b.extend([3001, 3002, 3003]);

        // Reference: B computed COLD (fresh batcher, empty cache → no reuse).
        let mut ref_b = {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, &b, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            t
        };

        // Warm cache with A, finish it (frees refcounts but KEEPS A's prefix blocks
        // registered), then admit B which should reuse A's shared-prefix blocks.
        let mut cb = ContinuousBatcher::new(&model, 4, max_seq);
        cb.admit(0, &a, k, eos);
        while cb.active_len() > 0 { cb.step(); }
        let (hits_before, _) = cb.cache_stats();
        let mut got_b = vec![cb.admit(1, &b, k, eos).unwrap().0];
        let (hits_after, _) = cb.cache_stats();
        while cb.active_len() > 0 { for (_, x, _) in cb.step() { got_b.push(x); } }

        let reused = hits_after - hits_before;
        eprintln!("prefix cache: B reused {reused} blocks (~{} tokens) of the 40-token shared prefix", reused * 16);
        assert!(reused >= 2, "expected B to reuse ≥2 shared-prefix blocks, got {reused}");
        assert_eq!(got_b, ref_b, "prefix-cached output diverged from cold computation");
        eprintln!("prefix-cached B is bit-identical to cold B ✓");
    }

    /// LRU vs first-fit reclaim under overcommit pressure (the workload paged KV
    /// exists for): a HOT shared prefix is re-admitted every round while COLD
    /// one-off prompts churn a small pool. LRU keeps the hot prefix resident →
    /// it keeps hitting; first-fit evicts the low-index hot blocks first → thrash.
    /// `cargo test --release --features gpu --lib gpu_lru_vs_firstfit -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_lru_vs_firstfit() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");

        const ROUNDS: usize = 10;
        // Hot request = 48-token shared prefix (3 full blocks) + a short unique tail.
        let shared: Vec<u32> = (0..48u32).map(|i| 1000 + i).collect();
        let mut hot = shared.clone(); hot.extend([2001, 2002, 2003, 2004]);

        // Small overcommitted pool: 8 blocks, ≤4 per seq (max_seq 64). Hot needs 4,
        // each cold one-off needs 3 → cold churn forces cache-block reclaim.
        let run = |first_fit: bool| -> u64 {
            let dec = model.batched_decoder_paged(2, 64, 8);
            dec.blocks.borrow_mut().first_fit = first_fit;
            let mut hot_hits = 0u64;
            for r in 0..ROUNDS {
                let (_, cached) = dec.prefill_slot_cached(&hot, 0, None);
                if cached > 0 { hot_hits += 1; }
                dec.free_slot(0);
                let cold: Vec<u32> = (0..48u32).map(|i| 50_000 + (r as u32) * 100 + i).collect();
                dec.prefill_slot_cached(&cold, 1, None);
                dec.free_slot(1);
            }
            hot_hits
        };
        let lru = run(false);
        let ff = run(true);
        eprintln!("hot-prefix HITS over {ROUNDS} rounds under churn: LRU {lru} vs first-fit {ff}");
        assert!(lru > ff, "LRU should keep the hot prefix resident more than first-fit ({lru} vs {ff})");
        assert!(lru >= ROUNDS as u64 - 1, "LRU should hit every round after warmup, got {lru}/{ROUNDS}");
        eprintln!("LRU reclaim keeps the hot prefix resident under pressure ✓");
    }

    /// GPU temperature sampling (Gumbel-max): temp=0 must equal greedy and ignore
    /// the seed; temp>0 must be reproducible for a fixed seed and vary by seed.
    /// `cargo test --release --features gpu --lib gpu_sampling -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_sampling() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let prompt = vec![128000u32, 791, 6864, 315, 9822, 374]; // "The capital of France is"
        let (k, eos, max_seq) = (14usize, u32::MAX, 256usize);
        let run = |temp: f32, seed: u32| -> Vec<u32> {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit_sampled(0, &prompt, k, eos, temp, seed).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            t
        };

        let greedy = run(0.0, 0);
        let greedy_other_seed = run(0.0, 987654); // temp=0 must ignore the seed
        let s_a = run(1.0, 42);
        let s_b = run(1.0, 42); // same seed → reproducible
        let s_c = run(1.0, 7);  // different seed

        assert_eq!(greedy, greedy_other_seed, "temp=0 must be greedy regardless of seed");
        assert_eq!(s_a, s_b, "same (temp, seed) must reproduce the same tokens");
        eprintln!("greedy        : {greedy:?}");
        eprintln!("sample seed=42: {s_a:?}  (reproducible: {})", s_a == s_b);
        eprintln!("sample seed=7 : {s_c:?}");
        eprintln!("temp=0 ≡ greedy (seed-independent) ✓, temp>0 reproducible per seed ✓");
        eprintln!("sampling diverges from greedy: {} ; seed42≠seed7: {}", s_a != greedy, s_a != s_c);
    }

    /// Validate the GPU top-K kernel against a CPU sort (same tie-break: value
    /// descending, index ascending) on synthetic logits — no model needed.
    #[test]
    fn gpu_btopk_kernel() {
        use wgpu::util::DeviceExt;
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let (m, vocab, k) = (3usize, 5000usize, TOPK_K);
        // Deterministic pseudo-random logits, with a few exact ties to exercise the
        // index tie-break.
        let mut logits = vec![0f32; m * vocab];
        let mut st = 0x2545F4914F6CDD1Du64;
        for v in logits.iter_mut() {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            *v = ((st >> 40) as f32 / 16777216.0) - 0.5;
        }
        for s in 0..m { logits[s * vocab + 100] = 0.4999; logits[s * vocab + 4000] = 0.4999; } // tie
        let lbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&logits), usage: wgpu::BufferUsages::STORAGE });
        let vbuf = ctx.alloc_activation(m * k, true);
        let ibuf = ctx.alloc_activation(m * k, true);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        ctx.record_btopk(&mut enc, &lbuf, &vbuf, &ibuf, vocab, m, k);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::Maintain::Wait);
        let gpu_idx: Vec<u32> = ctx.read_buffer(&ibuf, m * k).iter().map(|f| f.to_bits()).collect();

        for s in 0..m {
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| {
                logits[s * vocab + b].partial_cmp(&logits[s * vocab + a]).unwrap().then(a.cmp(&b))
            });
            let cpu: Vec<u32> = order[..k].iter().map(|&i| i as u32).collect();
            assert_eq!(&gpu_idx[s * k..(s + 1) * k], &cpu[..], "stream {s} top-{k} indices mismatch");
        }
        eprintln!("GPU top-{k} matches CPU sort for {m} streams (incl. tie-break) ✓");
    }

    /// Top-k/top-p sampling end-to-end through the batcher: the top-K path with
    /// temp=0 must be bit-identical to greedy (validates routing + CPU sampler's
    /// greedy branch), sampling is reproducible per seed, and temp>0 diverges.
    /// `cargo test --release --features gpu --lib gpu_topkp -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_topkp() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let prompt = vec![128000u32, 791, 6864, 315, 9822, 374];
        let (k, eos, max_seq) = (14usize, u32::MAX, 256usize);
        let run = |params: SamplingParams, seed: u32| -> Vec<u32> {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit_params(0, &prompt, k, eos, params, seed).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            t
        };
        let greedy = run(SamplingParams::greedy(), 0);
        // top-K path (top_k+top_p set) with temp=0 → argmax everywhere → greedy.
        let topk_greedy = run(SamplingParams { temp: 0.0, top_k: 5, top_p: 0.9 }, 999);
        let a = run(SamplingParams { temp: 0.9, top_k: 40, top_p: 0.92 }, 7);
        let b = run(SamplingParams { temp: 0.9, top_k: 40, top_p: 0.92 }, 7);

        assert_eq!(topk_greedy, greedy, "top-K path with temp=0 must equal greedy");
        assert_eq!(a, b, "same (params, seed) must reproduce");
        eprintln!("greedy            : {greedy:?}");
        eprintln!("top_k=40,top_p=.92: {a:?}  (reproducible: {})", a == b);
        eprintln!("top-K path temp=0 ≡ greedy ✓, top-k/top-p reproducible ✓, diverges from greedy: {}", a != greedy);
    }

    /// Preemption (recompute) must be output-transparent: two sequences forced to
    /// preempt each other in a tiny KV pool must produce tokens bit-identical to
    /// running each alone with no preemption.
    /// `cargo test --release --features gpu --lib gpu_preemption -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_preemption() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        // max_seq 64 → 4 blocks/seq; pool = 4 holds exactly one full sequence, so
        // two growing sequences must preempt. ~36 tokens each = 3 blocks.
        let (max_seq, n_blocks, k, eos) = (64usize, 4usize, 30usize, u32::MAX);
        let prompts: Vec<Vec<u32>> = vec![
            vec![128000, 791, 6864, 315, 9822, 374],
            vec![128000, 15724, 374, 264, 1296, 1473],
        ];

        // Reference: each sequence alone, full pool, no preemption.
        let mut refs: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, p, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            refs.push(t);
        }

        // Forced preemption: both sequences in a 4-block pool.
        let mut cb = ContinuousBatcher::with_pool(&model, 2, max_seq, n_blocks);
        let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        for (i, p) in prompts.iter().enumerate() {
            let g = cb.admit(i as u64, p, k, eos).expect("admit").0;
            got.entry(i as u64).or_default().push(g);
        }
        let mut steps = 0;
        while cb.active_len() > 0 || cb.preempted_len() > 0 {
            for (id, x, _) in cb.step() { got.get_mut(&id).unwrap().push(x); }
            steps += 1;
            assert!(steps < 2000, "preemption loop made no progress");
        }
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(&got[&(i as u64)], r, "sequence {i} diverged under preemption");
        }
        assert!(cb.preemption_count() > 0, "test did not actually trigger preemption (pool too large?)");
        eprintln!("preemption: {} preemptions over {} steps; both sequences bit-identical to no-preemption ✓", cb.preemption_count(), steps);
    }

    /// Swap-to-host preemption: on eviction copy the victim's KV out to host and
    /// restore it on resume (one decode step), instead of recomputing the whole
    /// history. Verifies bit-identical to BOTH no-preemption and recompute, and
    /// times each policy's resume cost.
    /// FINDING: swap-to-host is SLOWER than recompute here (~0.9x, worse with
    /// length) — UMA means host==device memory, so swap saves nothing a bigger
    /// pool wouldn't and only adds scatter/gather + mapped-readback overhead while
    /// recompute is a fast batched prefill. Recompute stays the default; swap is a
    /// discrete-GPU (VRAM-constrained) lever, validated correct and kept default-off.
    /// `cargo test --release --features gpu --lib gpu_swap_preemption -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_swap_preemption() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        // Long-ish sequences (k=80) in a tight 10-block pool so each preempted seq
        // has a big history → recompute is expensive, swap-to-host should pay off.
        let (max_seq, n_blocks, k, eos) = (128usize, 10usize, 80usize, u32::MAX);
        let prompts: Vec<Vec<u32>> = vec![
            vec![128000, 791, 6864, 315, 9822, 374],
            vec![128000, 15724, 374, 264, 1296, 1473],
        ];

        // Reference: each sequence alone, full pool, no preemption.
        let mut refs: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, p, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            refs.push(t);
        }

        // Forced preemption in the tight pool under each policy.
        let run = |swap: bool| -> (std::collections::HashMap<u64, Vec<u32>>, u64, std::time::Duration) {
            let mut cb = ContinuousBatcher::with_pool(&model, 2, max_seq, n_blocks);
            cb.swap_mode = swap;
            let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
            for (i, p) in prompts.iter().enumerate() {
                let g = cb.admit(i as u64, p, k, eos).expect("admit").0;
                got.entry(i as u64).or_default().push(g);
            }
            let t0 = std::time::Instant::now();
            let mut steps = 0;
            while cb.active_len() > 0 || cb.preempted_len() > 0 {
                for (id, x, _) in cb.step() { got.get_mut(&id).unwrap().push(x); }
                steps += 1;
                assert!(steps < 4000, "preemption loop made no progress");
            }
            (got, cb.preemption_count(), t0.elapsed())
        };

        let (recompute, rc_pre, rc_t) = run(false);
        let (swapped, sw_pre, sw_t) = run(true);

        for (i, r) in refs.iter().enumerate() {
            assert_eq!(&recompute[&(i as u64)], r, "recompute: sequence {i} diverged");
            assert_eq!(&swapped[&(i as u64)], r, "swap-to-host: sequence {i} diverged from no-preemption");
        }
        assert!(rc_pre > 0 && sw_pre > 0, "both modes must actually preempt (rc {rc_pre}, sw {sw_pre})");
        let _ = (rc_t, sw_t); // whole-run wall is decode-dominated; isolate resume cost below.

        // Isolated resume cost across history lengths: one preempt→resume cycle.
        // Recompute re-prefills the whole history (O(L) compute); swap-to-host copies
        // KV out+back (O(L) bytes) + one decode step. Both yield the next token.
        let dec = model.batched_decoder_paged(2, 1024, 80);
        fn min5(mut f: impl FnMut() -> std::time::Duration) -> std::time::Duration {
            (0..5).map(|_| f()).min().unwrap()
        }
        for &l in &[80usize, 256, 768] {
            let tokens: Vec<u32> = (0..l as u32).map(|i| 1000 + (i % 64)).collect(); // synthetic history
            let last = *tokens.last().unwrap();
            dec.prefill_slot_cached(&tokens, 0, None); dec.free_slot(0); // warmup this length
            let recompute_resume = min5(|| {
                let t = std::time::Instant::now();
                dec.prefill_slot_cached(&tokens, 0, None);
                let d = t.elapsed(); dec.free_slot(0); d
            });
            let swap_resume = min5(|| {
                dec.prefill_slot_cached(&tokens, 0, None);            // make resident (untimed)
                let t = std::time::Instant::now();
                let kv = dec.swap_out(0, l - 1);                      // copy KV out, free blocks
                dec.swap_in(1, &kv);                                 // restore into a fresh slot
                let _ = dec.step_slotted(&[last], &[(l - 1) as u32], &[1]); // one decode step
                let d = t.elapsed(); dec.free_slot(1); d
            });
            eprintln!("isolated resume @ {l:>4} tok: recompute {recompute_resume:>10.2?} vs swap-to-host {swap_resume:>10.2?}  ({:.2}x)",
                recompute_resume.as_secs_f64() / swap_resume.as_secs_f64());
        }
        eprintln!("swap-to-host preemption bit-identical to no-preemption AND recompute ✓");
    }

    /// Chunked prefill: a long prompt prefills one chunk per step (interleaved
    /// with decode), so a short sequence keeps decoding DURING the long prefill —
    /// and both outputs are bit-identical to synchronous prefill.
    /// `cargo test --release --features gpu --lib gpu_chunked_prefill -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_chunked_prefill() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (max_seq, k, eos) = (512usize, 16usize, u32::MAX);
        let short = vec![128000u32, 791, 6864, 315, 9822, 374];
        let mut long = vec![128000u32];
        for i in 0..300u32 { long.push(1000 + (i % 64)); } // 301 tokens → 3 prefill chunks

        // References: each alone via synchronous admit.
        let run_alone = |prompt: &[u32]| -> Vec<u32> {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, prompt, k, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            t
        };
        let ref_short = run_alone(&short);
        let ref_long = run_alone(&long);

        // Interleaved: short admitted (decodes immediately), long enqueued (chunked).
        let mut cb = ContinuousBatcher::new(&model, 2, max_seq);
        let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        got.entry(0).or_default().push(cb.admit(0, &short, k, eos).unwrap().0);
        assert!(cb.enqueue_params(1, long.clone(), k, eos, SamplingParams::greedy(), 0));
        let (mut step_i, mut long_first_step, mut short_progress) = (0, 0, 0);
        while cb.active_len() + cb.prefilling_len() + cb.preempted_len() > 0 {
            step_i += 1;
            for (id, tok, _) in cb.step() {
                got.entry(id).or_default().push(tok);
                if id == 1 && long_first_step == 0 { long_first_step = step_i; short_progress = got[&0].len(); }
            }
        }
        assert_eq!(got[&0], ref_short, "short sequence diverged under chunked prefill");
        assert_eq!(got[&1], ref_long, "long sequence diverged under chunked prefill");
        assert!(long_first_step >= 2, "long prompt should span multiple prefill chunks (got {long_first_step})");
        eprintln!("chunked prefill: long ({} tok) prefilled over {} steps; short decoded {} tokens DURING that prefill; both outputs bit-identical ✓", long.len(), long_first_step, short_progress);
    }

    /// Request cancellation: a cancelled sequence frees its KV slot + blocks
    /// immediately, stops producing tokens, and does not disturb its neighbors.
    /// `cargo test --release --features gpu --lib gpu_cancel -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_cancel() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (max_seq, eos) = (256usize, u32::MAX);
        let p0 = vec![128000u32, 791, 6864, 315, 9822, 374];
        let p1 = vec![128000u32, 15724, 374, 264, 1296, 1473];

        // Reference: seq 1 run ALONE for 12 tokens (what it must produce regardless
        // of seq 0 being cancelled alongside it).
        let ref1 = {
            let mut cb = ContinuousBatcher::new(&model, 1, max_seq);
            let mut t = vec![cb.admit(0, &p1, 12, eos).unwrap().0];
            while cb.active_len() > 0 { for (_, x, _) in cb.step() { t.push(x); } }
            t
        };

        let mut cb = ContinuousBatcher::new(&model, 4, max_seq);
        let mut got1 = vec![cb.admit(1, &p1, 12, eos).unwrap().0];
        cb.admit(0, &p0, 100, eos); // long-running; will be cancelled
        for _ in 0..4 { for (id, x, _) in cb.step() { if id == 1 { got1.push(x); } } }
        let (free_before, _) = cb.block_pool();
        assert!(cb.cancel(0), "cancel of an active sequence should succeed");
        let (free_after, _) = cb.block_pool();
        assert!(free_after > free_before, "cancel must free the sequence's KV blocks ({free_before} -> {free_after})");
        assert!(!cb.cancel(0), "cancel of an unknown id returns false");

        // Seq 1 keeps going and finishes; seq 0 produces nothing more.
        let mut seq0_after = 0;
        while cb.active_len() > 0 {
            for (id, x, _) in cb.step() {
                if id == 1 { got1.push(x); } else { seq0_after += 1; }
            }
        }
        assert_eq!(seq0_after, 0, "cancelled sequence must produce no further tokens");
        assert_eq!(got1, ref1, "neighbor sequence diverged after a cancel");
        eprintln!("cancel: freed seq 0's KV (free {free_before} -> {free_after} blocks); seq 1 unaffected + bit-identical ✓");
    }

    /// Prompt-Lookup speculative decode (generate_pld): the output must be
    /// bit-identical to greedy single-token decode, but use FEWER forwards on
    /// echo-heavy text (>1 token/forward — past the 1-token/forward roofline).
    /// `cargo test --release --features gpu --lib gpu_pld -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_pld() {
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU ({e}); skipping"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (max_seq, k, eos) = (512usize, 48usize, u32::MAX);
        // Echo-heavy prompt: a repeated pattern the greedy model continues, so
        // prompt-lookup finds high-acceptance drafts.
        let mut prompt = vec![128000u32];
        for _ in 0..10 { prompt.extend_from_slice(&[264, 265, 266, 267, 268]); }

        use std::time::Instant;
        // Greedy single-token reference (timed).
        let dec_g = model.batched_decoder(8, max_seq);
        let g0 = dec_g.prefill_slot(&prompt, 0);
        let (mut greedy, mut next, mut pos) = (vec![g0], g0, prompt.len());
        let tg = Instant::now();
        while greedy.len() < k && next != eos {
            let t = dec_g.step_slotted(&[next], &[pos as u32], &[0])[0];
            greedy.push(t); next = t; pos += 1;
        }
        let g_tps = greedy.len() as f64 / tg.elapsed().as_secs_f64();

        // PLD (timed).
        let dec_p = model.batched_decoder(8, max_seq);
        let tp = Instant::now();
        let (produced, forwards) = dec_p.generate_pld(&prompt, k, eos, 3, 7);
        let p_tps = produced.len() as f64 / tp.elapsed().as_secs_f64();

        assert_eq!(produced, greedy, "PLD output diverged from greedy single-token decode");
        let toks = produced.len();
        eprintln!("PLD: {toks} tokens in {forwards} forwards = {:.2} tokens/forward; output bit-identical to greedy ✓", toks as f64 / forwards.max(1) as f64);
        eprintln!("  wall-clock: greedy {g_tps:.0} tok/s -> PLD {p_tps:.0} tok/s = {:.2}x faster (same wgpu kernels)", p_tps / g_tps);
    }

    /// PLD ACCEPTANCE SENSITIVITY: batched PLD's net win is workload-dependent — the
    /// PLD forward is bigger (M*(draft_k+1) rows), so on LOW-acceptance (open-ended)
    /// text it should LOSE to plain step(). Measures net wall-clock for both on echo
    /// vs open-ended, to find when ZLLM_CB_PLD should actually be enabled.
    /// `cargo test --release --features gpu --lib gpu_pld_acceptance -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_pld_acceptance() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (ms, max_new, eos) = (8usize, 48usize, u32::MAX);
        let mut echo = vec![128000u32]; for _ in 0..10 { echo.extend_from_slice(&[264, 265, 266, 267, 268]); }
        let open = vec![128000u32, 791, 6864, 315, 9822, 374]; // "The capital of France is" → open-ended
        for (name, prompt) in [("echo", &echo), ("open-ended", &open)] {
            let run = |pld: bool| -> (f64, usize, usize) {
                let mut cb = ContinuousBatcher::new(&model, ms * 8, 1024);
                let mut total = 0usize;
                for s in 0..ms { if cb.admit(s as u64, prompt, max_new, eos).is_some() { total += 1; } }
                let t0 = Instant::now(); let mut fwds = 0usize;
                while cb.active_len() > 0 { let r = if pld { cb.step_pld(3, 7) } else { cb.step() }; total += r.len(); fwds += 1; }
                (total as f64 / t0.elapsed().as_secs_f64(), fwds, total)
            };
            let (plain, pf, _) = run(false);
            let (pldv, pldf, toks) = run(true);
            let tpf = toks as f64 / pldf as f64;
            eprintln!("{name:>11}: plain {plain:>4.0} tok/s ({pf} fwds) | PLD {pldv:>4.0} tok/s ({pldf} fwds, {tpf:.1} tok/fwd) = {:.2}x  [{}]",
                pldv / plain, if pldv > plain * 1.05 { "PLD wins" } else if pldv < plain * 0.95 { "PLD HURTS" } else { "~neutral" });
        }
    }

    /// ContinuousBatcher::step_pld productized: drive the real batcher with batched
    /// spec-decode and confirm bit-identical output to plain step() in fewer forwards.
    /// `cargo test --release --features gpu --lib gpu_cb_pld -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_cb_pld() {
        use std::collections::HashMap;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (ms, max_new, eos) = (8usize, 40usize, u32::MAX);
        let mut prompt = vec![128000u32]; for _ in 0..10 { prompt.extend_from_slice(&[264, 265, 266, 267, 268]); }
        let run = |pld: bool| -> (HashMap<u64, Vec<u32>>, usize) {
            let mut cb = ContinuousBatcher::new(&model, ms * 8, 512);
            let mut got: HashMap<u64, Vec<u32>> = HashMap::new();
            for s in 0..ms { let (t, _) = cb.admit(s as u64, &prompt, max_new, eos).unwrap(); got.entry(s as u64).or_default().push(t); }
            let mut steps = 0usize;
            while cb.active_len() > 0 {
                let res = if pld { cb.step_pld(3, 7) } else { cb.step() };
                for (id, t, _) in res { got.get_mut(&id).unwrap().push(t); }
                steps += 1;
            }
            (got, steps)
        };
        let (ref_out, ref_steps) = run(false);
        let (pld_out, pld_steps) = run(true);
        let mut ok = true;
        for s in 0..ms { if pld_out[&(s as u64)] != ref_out[&(s as u64)] { ok = false; } }
        eprintln!("CB step_pld: {ms} streams x {} tok, {pld_steps} forwards vs plain step() {ref_steps} ({:.1}x fewer), bit-identical {}",
            ref_out[&0].len(), ref_steps as f64 / pld_steps as f64, if ok { "✓" } else { "✗ DIVERGED" });
        assert!(ok, "step_pld diverged from step()");
    }

    /// BATCHED PLD: speculative decode INSIDE the continuous batch — every stream
    /// proposes an n-gram draft, all drafts verify in ONE batched forward (step_slotted
    /// over each stream's slot), accept per stream. The serving frontier llama-server
    /// doesn't do. Bit-identical to per-stream greedy; >1 tok/forward/stream on echo text.
    /// `cargo test --release --features gpu --lib gpu_batched_pld -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_batched_pld() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        let (ms, draft_k, lookup, max_new, eos) = (8usize, 7usize, 3usize, 48usize, u32::MAX);
        let mut prompt = vec![128000u32]; for _ in 0..10 { prompt.extend_from_slice(&[264, 265, 266, 267, 268]); }

        // Single-stream greedy reference (model's own resident KV).
        let mut gn = 0u32; for (i, &t) in prompt.iter().enumerate() { gn = model.forward_argmax(t, i); }
        let mut greedy = vec![gn]; let mut gp = prompt.len();
        while greedy.len() < max_new && gn != eos { gn = model.forward_argmax(gn, gp); greedy.push(gn); gp += 1; }

        // Batched PLD over `ms` identical streams (each its own slot).
        let dec = model.batched_decoder(ms * (draft_k + 1), 512);
        let mut hist = vec![prompt.clone(); ms];
        let mut next = vec![0u32; ms]; let mut pos = vec![prompt.len(); ms];
        let mut produced: Vec<Vec<u32>> = vec![Vec::new(); ms];
        for s in 0..ms { let n = dec.prefill_slot(&prompt, s as u32); next[s] = n; hist[s].push(n); produced[s].push(n); }
        let mut done = vec![false; ms];
        let mut forwards = 0usize;
        let t0 = Instant::now();
        while !done.iter().all(|&d| d) {
            let (mut rows, mut positions, mut slots) = (Vec::new(), Vec::new(), Vec::new());
            let mut layout: Vec<(usize, usize, Vec<u32>)> = Vec::new(); // (stream, start, draft)
            for s in 0..ms {
                if done[s] { continue; }
                let k = draft_k.min(max_new - produced[s].len());
                let draft = crate::engine::spec_decode::lookup_draft_best(&hist[s], &hist[s], lookup, k).unwrap_or_default();
                let start = rows.len();
                rows.push(next[s]); rows.extend_from_slice(&draft);
                for i in 0..=draft.len() { positions.push((pos[s] + i) as u32); slots.push(s as u32); }
                layout.push((s, start, draft));
            }
            if rows.is_empty() { break; }
            let outs = dec.step_slotted(&rows, &positions, &slots);
            forwards += 1;
            for (s, start, draft) in &layout {
                let so = &outs[*start..*start + 1 + draft.len()];
                let mut acc = 0usize; while acc < draft.len() && so[acc] == draft[acc] { acc += 1; }
                for &tok in so.iter().take(acc + 1) {
                    produced[*s].push(tok); hist[*s].push(tok); next[*s] = tok;
                    if tok == eos || produced[*s].len() >= max_new { done[*s] = true; break; }
                }
                pos[*s] += acc + 1;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let total: usize = produced.iter().map(|p| p.len()).sum();
        let mut bit_ok = true;
        for s in 0..ms { let n = greedy.len().min(produced[s].len()); if produced[s][..n] != greedy[..n] { bit_ok = false; } }
        eprintln!("batched PLD: {ms} streams x {} tok, {forwards} forwards = {:.2} tok/forward, bit-identical-to-greedy {}",
            produced[0].len(), total as f64 / forwards as f64, if bit_ok { "✓" } else { "✗ DIVERGED" });
        eprintln!("  aggregate {:.0} tok/s on echo text  (plain batched M=8 was ~104; llama batched M=8 = 710)", total as f64 / dt);
        assert!(bit_ok, "batched PLD diverged from per-stream greedy");
    }

    /// HETEROGENEOUS BATCHED: iGPU in BATCHED mode is compute-bound (~11 GB/s, ~4%
    /// of the bus) — unlike single-stream (bandwidth-bound). So the CPU should add
    /// throughput from the idle bandwidth here even though it couldn't single-stream.
    /// `cargo test --release --features gpu --lib gpu_cpu_heterogeneous -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_cpu_heterogeneous_batched() {
        use std::time::{Duration, Instant};
        use std::thread;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");
        use crate::backend::candle::backend::CandleCpuBackend;
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
        let am = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let m = 8usize;
        let dec = model.batched_decoder(m, 1024);
        let mut bt = vec![128000u32; m]; let mut bp = vec![0u32; m];
        let mut cn = am(&cb.forward_logits(&[128000]).unwrap());
        let batched = |dec: &BatchedDecoder, mut bt: Vec<u32>, mut bp: Vec<u32>, dur: Duration| -> (usize, Vec<u32>, Vec<u32>) {
            let t = Instant::now(); let mut n = 0usize;
            while t.elapsed() < dur { let nx = dec.step(&bt, &bp); n += nx.len(); bt = nx; for p in bp.iter_mut() { *p += 1; } }
            (n, bt, bp)
        };
        fn cpu(mut mdl: CandleCpuBackend, mut next: u32, dur: Duration) -> (usize, CandleCpuBackend, u32) {
            use crate::backend::traits::Backend;
            let am = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
            let t = Instant::now(); let mut n = 0usize;
            while t.elapsed() < dur { next = am(&mdl.forward_logits(&[next]).unwrap()); n += 1; }
            (n, mdl, next)
        }
        let dur = Duration::from_secs(3);
        let (b_alone, bt2, bp2) = batched(&dec, bt, bp, dur); bt = bt2; bp = bp2;
        let bps_alone = b_alone as f64 / dur.as_secs_f64();
        let (c_alone, cb2, cn2) = cpu(cb, cn, dur); cb = cb2; cn = cn2;
        let cps_alone = c_alone as f64 / dur.as_secs_f64();
        let h = thread::spawn(move || cpu(cb, cn, dur));
        let (b_conc, _, _) = batched(&dec, bt, bp, dur);
        let (c_conc, _, _) = h.join().unwrap();
        let bps_conc = b_conc as f64 / dur.as_secs_f64();
        let cps_conc = c_conc as f64 / dur.as_secs_f64();
        eprintln!("ALONE:       iGPU-batched(M={m}) {bps_alone:.0} tok/s | CPU {cps_alone:.0} tok/s");
        eprintln!("CONCURRENT:  iGPU-batched {bps_conc:.0} ({:.0}%) | CPU {cps_conc:.0} ({:.0}%)", bps_conc / bps_alone * 100.0, cps_conc / cps_alone * 100.0);
        let agg = bps_conc + cps_conc;
        eprintln!("VERDICT: aggregate {agg:.0} vs iGPU-batched-alone {bps_alone:.0} = {:.2}x ({})",
            agg / bps_alone, if agg > bps_alone * 1.05 { "WIN — batched iGPU leaves bus headroom for CPU" } else { "no win" });
    }

    #[test]
    #[ignore]
    fn gpu_batched_decode_throughput() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
        let model = GpuModel::load(path, ctx).expect("load");

        // Single-stream greedy reference (the model's own resident KV cache).
        let k_val = 8usize;
        let mut single = Vec::new();
        let mut tok = 128000u32;
        for p in 0..k_val { let n = model.forward_argmax(tok, p as usize); single.push(n); tok = n; }

        // Correctness: M identical streams must reproduce the single-stream
        // tokens exactly (catches cross-stream contamination + a wrong BDSDPA).
        let m_val = 8usize;
        {
            let dec = model.batched_decoder(m_val, 256);
            let mut bt = vec![128000u32; m_val];
            let mut bp = vec![0u32; m_val];
            for k in 0..k_val {
                let nxt = dec.step(&bt, &bp);
                for s in 0..m_val {
                    assert_eq!(nxt[s], single[k], "stream {s} step {k} diverged from single-stream");
                }
                bt = nxt;
                for p in bp.iter_mut() { *p += 1; }
            }
        }
        eprintln!("batched decode validated: {m_val} streams bit-identical to single-stream over {k_val} steps");

        // Single-stream decode tok/s baseline (warm, realistic cache depth).
        let mut t = 128000u32;
        for p in 0..32 { t = model.forward_argmax(t, p); }
        let t0 = Instant::now();
        let mut pp = 32usize;
        for _ in 0..32 { t = model.forward_argmax(t, pp); pp += 1; }
        let single_tps = 32.0 / t0.elapsed().as_secs_f64();
        eprintln!("single-stream decode baseline: {single_tps:.0} tok/s");

        // Aggregate throughput vs concurrency (all streams at the same depth).
        eprintln!("--- aggregate decode throughput vs concurrency ---");
        for &m in &[1usize, 2, 4, 8, 16, 32] {
            let dec = model.batched_decoder(m, 256);
            let mut tk = vec![128000u32; m];
            let mut ps = vec![0u32; m];
            for _ in 0..32 { let n = dec.step(&tk, &ps); tk = n; for p in ps.iter_mut() { *p += 1; } } // warm + fill cache
            let steps = 24usize;
            let t0 = Instant::now();
            for _ in 0..steps { let n = dec.step(&tk, &ps); tk = n; for p in ps.iter_mut() { *p += 1; } }
            let dt = t0.elapsed().as_secs_f64();
            let agg = (m * steps) as f64 / dt;
            eprintln!("  M={m:>2}: {:5.1} ms/step, aggregate {:>5.0} tok/s ({:>4.0}/stream)  [{:.2}x single-stream]",
                dt * 1e3 / steps as f64, agg, agg / m as f64, agg / single_tps);
        }
    }
}
