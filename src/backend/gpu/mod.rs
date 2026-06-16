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
        })
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
        })
    }

    /// Record the whole token forward (embedding → layers → final norm → LM
    /// head, leaving logits in `self.logits`) into `enc`, writing the per-
    /// token inputs. Shared by `forward` and `forward_argmax`.
    fn record_forward(&self, enc: &mut wgpu::CommandEncoder, token: u32, pos: usize) {
        let ctx = &self.ctx;
        let (n_embd, n_head, n_kv_head, head_dim, n_inter) =
            (self.n_embd, self.n_head, self.n_kv_head, self.head_dim, self.n_inter);
        let half = head_dim / 2;
        let kv_dim = n_kv_head * head_dim;
        let seq_len = (pos + 1) as u32;

        let row = &self.embed[token as usize * n_embd..(token as usize + 1) * n_embd];
        ctx.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(row));
        ctx.queue.write_buffer(&self.cos_buf, 0, bytemuck::cast_slice(&self.cos[pos * half..pos * half + half]));
        ctx.queue.write_buffer(&self.sin_buf, 0, bytemuck::cast_slice(&self.sin[pos * half..pos * half + half]));
        ctx.queue.write_buffer(&self.sdpa_p, 0, bytemuck::cast_slice(&[n_head as u32, n_kv_head as u32, head_dim as u32, seq_len]));

        let inter_wg = (n_inter as u32).div_ceil(64);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
