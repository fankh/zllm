//! Phase 2.0 spike: a raw-Vulkan (`ash`) compute path that uses
//! `VK_KHR_cooperative_matrix` — the iGPU's WMMA matrix instructions, which
//! wgpu/WGSL cannot express. This module proves the whole pipeline end to end
//! (committed SPIR-V → ash device with the coopmat feature chain → RDNA3.5
//! WMMA → validated result) before the full kernel port. See VULKAN_PLAN.md.
//!
//! Shaders are GLSL compiled OFFLINE to SPIR-V (committed `.spv`, embedded via
//! `include_bytes!`), so the build needs no glslang/SDK.
#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::ffi::CStr;

const COOPMAT_MATMUL_SPV: &[u8] = include_bytes!("shaders/coopmat_matmul.spv");
const COOPMAT_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_gemm.spv");
const COOPMAT_Q4K_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_q4k_gemm.spv");
const COOPMAT_Q4K_GEMM_M16_SPV: &[u8] = include_bytes!("shaders/coopmat_q4k_gemm_m16.spv");
const DECODE_MATVEC_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q4k.spv");
const DECODE_MATVEC_Q3K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q3k.spv"); // gate+up→Q3 (naga-gen)
const DECODE_MATVEC_Q4K_ROPE_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q4k_rope.spv");
const RMSNORM_SPV: &[u8] = include_bytes!("shaders/rmsnorm.spv");
const ROPE_SPV: &[u8] = include_bytes!("shaders/rope.spv");
const SDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/sdpa_decode.spv");
const SDPA_FLASH_PARTIAL_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial.spv");
const SDPA_FLASH_COMBINE_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_combine.spv");
const SDPA_FLASH_COMBINE_H_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_combine_h.spv");
const SDPA_FLASH_PARTIAL2_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial2.spv");
const SDPA_FLASH_BLOCK: usize = 32; // must match BLOCK in the flash shaders
// Use the barrier-lean single-pass decode SDPA up to this depth (must stay < sdpa_decode's
// MAXSEQ=512); only beyond it does the flash partial/combine path's extra parallelism pay
// for its 2 dispatches + combine. The single-pass tree-reduces once, not per position.
const FLASH_MIN_SEQ: usize = 256;
// Hierarchical flash combine: each level-1 workgroup merges this many block-partials into one
// super-partial (grid n_head × ceil(nblk/SUPER) => many workgroups, fixing the flat combine's
// occupancy starvation); level 2 merges the few super-partials. ~4-6x faster combine at long ctx.
const SDPA_SUPER: usize = 8;
const SILU_MUL_SPV: &[u8] = include_bytes!("shaders/silu_mul.spv");
const DECODE_MATVEC_Q6K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k.spv");
#[cfg(test)]
const DECODE_MATVEC_Q6K_V2_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k_v2.spv");
#[cfg(test)]
const GRID_BARRIER_PROBE_SPV: &[u8] = include_bytes!("shaders/grid_barrier_probe.spv");
#[cfg(test)]
const DECODE_MATVEC_Q6K_PERSIST_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q6k_persist.spv");
#[cfg(test)]
const Q6K_MEGAKERNEL_PROBE_SPV: &[u8] = include_bytes!("shaders/q6k_megakernel_probe.spv");
#[cfg(test)]
const FFN_MEGAKERNEL_SPV: &[u8] = include_bytes!("shaders/ffn_megakernel.spv");
const KV_WRITE_SPV: &[u8] = include_bytes!("shaders/kv_write.spv");
// Test-only shaders (ignored GPU tests: slot-indirection, rp-skinny bench, barrier probes).
#[cfg(test)]
const DECODE_MATVEC_DOWN_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_down_q4k.spv");
#[cfg(test)]
const BSDPA_SLOT_SPV: &[u8] = include_bytes!("shaders/bsdpa_slot.spv");
#[cfg(test)]
const KVWRITE_SLOT_SPV: &[u8] = include_bytes!("shaders/kvwrite_slot.spv");
#[cfg(test)]
const SKINNY_GEMM_Q4K_RP_SPV: &[u8] = include_bytes!("shaders/skinny_gemm_q4k_rp.spv");
#[cfg(test)]
const INC_SPV: &[u8] = include_bytes!("shaders/inc.spv");
#[cfg(test)]
const INC_COH_SPV: &[u8] = include_bytes!("shaders/inc_coh.spv");
const KV_WRITE_HM_SPV: &[u8] = include_bytes!("shaders/kv_write_hm.spv"); // head-major write (ZLLM_HEADMAJOR_KV)
const SDPA_FLASH_PARTIAL_HM_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial_hm.spv"); // head-major partial (naga-gen)
const KV_TRANSPOSE_HM_SPV: &[u8] = include_bytes!("shaders/kv_transpose_hm.spv"); // pos→head-major cache transpose (naga-gen)
const RESIDUAL_ADD_SPV: &[u8] = include_bytes!("shaders/residual_add.spv");
const ARGMAX_SPV: &[u8] = include_bytes!("shaders/argmax.spv");
// Batched prefill kernels.
const BNORM_SPV: &[u8] = include_bytes!("shaders/bnorm.spv");
const BROPE_SPV: &[u8] = include_bytes!("shaders/brope.spv");
const BSDPA_SPV: &[u8] = include_bytes!("shaders/bsdpa.spv");
const BSDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/bsdpa_decode.spv");
const BSDPA_OFFSET_SPV: &[u8] = include_bytes!("shaders/bsdpa_offset.spv"); // chunked-prefill offset SDPA (naga-gen)
const COOPMAT_ATTN_GEMM_SPV: &[u8] = include_bytes!("shaders/coopmat_attn_gemm.spv"); // prefill attention QK^T on WMMA
const COOPMAT_ATTN_GEMM_N64_SPV: &[u8] = include_bytes!("shaders/coopmat_attn_gemm_n64.spv"); // BN=64 variant (PV, N=hd)
const CAUSAL_SOFTMAX_SPV: &[u8] = include_bytes!("shaders/causal_softmax.spv");
const COOPMAT_FLASH_ATTN_SPV: &[u8] = include_bytes!("shaders/coopmat_flash_attn.spv"); // fused FA (S never global)
const BSILU_SPV: &[u8] = include_bytes!("shaders/bsilu.spv");
const TO_F16_SPV: &[u8] = include_bytes!("shaders/to_f16.spv");
const BMV_Q4K_SPV: &[u8] = include_bytes!("shaders/batched_matvec_q4k.spv");
const SKINNY_GEMM_Q4K_SPV: &[u8] = include_bytes!("shaders/skinny_gemm_q4k.spv");
const BMV_F16_SPV: &[u8] = include_bytes!("shaders/batched_matvec_f16.spv");
const BMV_Q6K_SPV: &[u8] = include_bytes!("shaders/batched_matvec_q6k.spv");

/// Max prompt length for the raw-Vulkan server fast-lane = the batched-prefill
/// tile cap (`PREFILL_MAX_M`): prompts up to this run one coopmat-GEMM prefill
/// pass on the iGPU. (The old 128 cap predated batched prefill — it left
/// 129..=1024-token prompts on the candle CPU path, measured 9.5 s TTFT for a
/// 902-token prompt vs ~0.5 s here.) Precision note: batched prefill stages
/// activations as f16 (fp32 accumulate), so long-prompt outputs carry the same
/// f16 tolerance llama.cpp's pipeline has (cosine ~0.9995 vs the pure-decode
/// path, occasional greedy divergence) — the tolerance already shipped for
/// 33..=128-token prompts; this extends it to the full tile.
pub const MAX_PREFILL_M: usize = PREFILL_MAX_M;
/// Resident KV-cache capacity in tokens (positions). Prompts longer than one
/// prefill tile are served by CHUNKED prefill (tile-sized pieces at position
/// offsets), so the fast-lane prompt bound is this cache size, not the tile.
pub const MAX_SEQ: usize = 4096;

/// A minimal raw-Vulkan compute context with cooperative matrix enabled.
pub struct VkContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    pdev: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    pub adapter_name: String,
    /// Supported cooperative-matrix configs (M,N,K) for fp16 A/B, reported by
    /// the driver. Used to confirm the 16x16x16 shape the shaders assume.
    pub coopmat_shapes: Vec<(u32, u32, u32)>,
    pub subgroup_size: u32,
}

impl VkContext {
    pub fn new() -> Result<Self, String> {
        unsafe { Self::new_inner() }
    }

    unsafe fn new_inner() -> Result<Self, String> {
        let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader: {e}"))?;

        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .map_err(|e| format!("create_instance: {e}"))?;

        // Pick the integrated GPU (fall back to first device).
        let pdevs = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
        let pdev = pdevs
            .iter()
            .copied()
            .find(|&pd| {
                instance.get_physical_device_properties(pd).device_type
                    == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .or_else(|| pdevs.first().copied())
            .ok_or("no Vulkan physical device")?;

        let props = instance.get_physical_device_properties(pdev);
        let adapter_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        // Compute queue family.
        let queue_family = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or("no compute queue family")?;

        // Subgroup size (RDNA3 wave32 needed for the 16x16x16 KHR coopmat shape).
        let mut sg_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg_props);
        instance.get_physical_device_properties2(pdev, &mut p2);
        let subgroup_size = sg_props.min_subgroup_size.max(32).min(sg_props.max_subgroup_size.max(32));

        // Confirm the coopmat extension is present.
        let exts = instance
            .enumerate_device_extension_properties(pdev)
            .map_err(|e| format!("enumerate_device_extension_properties: {e}"))?;
        let have_coopmat = exts.iter().any(|e| {
            CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
        });
        if !have_coopmat {
            return Err("VK_KHR_cooperative_matrix not supported by device".into());
        }

        // Query the supported coopmat shapes (fp16 A/B, subgroup scope).
        let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let cm_props = cm
            .get_physical_device_cooperative_matrix_properties(pdev)
            .map_err(|e| format!("get coopmat props: {e}"))?;
        let coopmat_shapes: Vec<(u32, u32, u32)> = cm_props
            .iter()
            .filter(|p| {
                p.a_type == vk::ComponentTypeKHR::FLOAT16
                    && p.b_type == vk::ComponentTypeKHR::FLOAT16
                    && p.scope == vk::ScopeKHR::SUBGROUP
            })
            .map(|p| (p.m_size, p.n_size, p.k_size))
            .collect();

        // --- Logical device with the coopmat feature chain ---
        let mut coopmat_feat =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let mut feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut coopmat_feat)
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13);

        let ext_names = [ash::khr::cooperative_matrix::NAME.as_ptr()];
        let qprio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&qprio)];
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&ext_names)
            .push_next(&mut feats2);
        let device = instance
            .create_device(pdev, &dci, None)
            .map_err(|e| format!("create_device: {e}"))?;
        let queue = device.get_device_queue(queue_family, 0);

        Ok(Self {
            _entry: entry,
            instance,
            device,
            pdev,
            queue,
            queue_family,
            adapter_name,
            coopmat_shapes,
            subgroup_size,
        })
    }

    /// Find a memory type matching `flags` for the given `req` bits.
    unsafe fn mem_type(&self, req_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = self.instance.get_physical_device_memory_properties(self.pdev);
        (0..mp.memory_type_count).find(|&i| {
            (req_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(flags)
        })
    }

    /// Allocate a UMA storage buffer (DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT),
    /// persistently mapped — zero-copy on Strix Halo.
    unsafe fn uma_buffer(&self, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        let buf = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| format!("create_buffer: {e}"))?;
        let req = self.device.get_buffer_memory_requirements(buf);
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let idx = self
            .mem_type(req.memory_type_bits, want)
            .or_else(|| {
                self.mem_type(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .ok_or("no host-visible memory type")?;
        let mem = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx),
                None,
            )
            .map_err(|e| format!("allocate_memory: {e}"))?;
        self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind: {e}"))?;
        let ptr = self
            .device
            .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map_memory: {e}"))? as *mut u8;
        Ok((buf, mem, ptr))
    }

    /// Generic compute pipeline from SPIR-V: `n_storage` storage buffers at
    /// bindings 0.., then a uniform at binding `n_storage`. Returns the pipeline
    /// + its layouts (caller destroys). Used by the fused decode forward.
    unsafe fn make_pipeline_raw(&self, spv: &[u8], n_storage: u32) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout, vk::ShaderModule) {
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv)).unwrap();
        let module = self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).unwrap();
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(n_storage).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let set_layout = self.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = self.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).unwrap();
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
        let pipeline = self.device.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).unwrap()[0];
        (pipeline, layout, set_layout, module)
    }

    /// Like make_pipeline_raw but forces requiredSubgroupSize=32 + full
    /// subgroups — required by the cooperative-matrix GEMM kernels (wave32).
    unsafe fn make_pipeline_coopmat(&self, spv: &[u8], n_storage: u32) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout, vk::ShaderModule) {
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv)).unwrap();
        let module = self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).unwrap();
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(n_storage).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let set_layout = self.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = self.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).unwrap();
        let entry = std::ffi::CString::new("main").unwrap();
        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut req_sg);
        let pipeline = self.device.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).unwrap()[0];
        (pipeline, layout, set_layout, module)
    }

    /// Spike: compute C[16,16] = A[16,16] * B[16,16] on the GPU via a single
    /// cooperative-matrix multiply (fp16 in, fp32 accumulate). `a`/`b` are
    /// row-major length-256 f32; returns the 256 f32 of C.
    pub fn coopmat_matmul_16(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), 256);
        assert_eq!(b.len(), 256);
        unsafe { self.coopmat_matmul_16_inner(a, b) }
    }

    unsafe fn coopmat_matmul_16_inner(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let dev = &self.device;
        // Buffers: A,B fp16 (256 * 2 bytes), C fp32 (256 * 4).
        let (a_buf, a_mem, a_ptr) = self.uma_buffer(256 * 2)?;
        let (b_buf, b_mem, b_ptr) = self.uma_buffer(256 * 2)?;
        let (c_buf, c_mem, c_ptr) = self.uma_buffer(256 * 4)?;
        // Upload A,B as f16.
        let a16: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        std::ptr::copy_nonoverlapping(a16.as_ptr() as *const u8, a_ptr, 256 * 2);
        std::ptr::copy_nonoverlapping(b16.as_ptr() as *const u8, b_ptr, 256 * 2);

        // Shader module from the committed SPIR-V.
        let spv = ash::util::read_spv(&mut std::io::Cursor::new(COOPMAT_MATMUL_SPV))
            .map_err(|e| format!("read_spv: {e}"))?;
        let module = dev
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
            .map_err(|e| format!("create_shader_module: {e}"))?;

        // Descriptor set layout: 3 storage buffers.
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = dev
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| format!("create_descriptor_set_layout: {e}"))?;
        let pipeline_layout = dev
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)),
                None,
            )
            .map_err(|e| format!("create_pipeline_layout: {e}"))?;

        // Pipeline — require a full wave32 subgroup (RDNA3 coopmat needs it).
        let entry = std::ffi::CString::new("main").unwrap();
        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .push_next(&mut req_sg);
        let pipeline = dev
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                None,
            )
            .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?[0];

        // Descriptor pool + set.
        let pool = dev
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(3),
                ]),
                None,
            )
            .map_err(|e| format!("create_descriptor_pool: {e}"))?;
        let set = dev
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&set_layout)),
            )
            .map_err(|e| format!("allocate_descriptor_sets: {e}"))?[0];
        let info = |buf| [vk::DescriptorBufferInfo::default().buffer(buf).range(vk::WHOLE_SIZE)];
        let (ai, bi, ci) = (info(a_buf), info(b_buf), info(c_buf));
        let writes = [
            (0u32, &ai), (1, &bi), (2, &ci),
        ]
        .map(|(b, i)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(i)
        });
        dev.update_descriptor_sets(&writes, &[]);

        // Record + dispatch + wait.
        let cmd_pool = dev
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family),
                None,
            )
            .map_err(|e| format!("create_command_pool: {e}"))?;
        let cmd = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];
        dev.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| format!("begin: {e}"))?;
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
        dev.cmd_dispatch(cmd, 1, 1, 1);
        dev.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
        let fence = dev
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("create_fence: {e}"))?;
        let cmds = [cmd];
        dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)
            .map_err(|e| format!("queue_submit: {e}"))?;
        dev.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;

        // Read C (HOST_COHERENT — visible after the fence).
        let mut out = vec![0f32; 256];
        std::ptr::copy_nonoverlapping(c_ptr as *const f32, out.as_mut_ptr(), 256);

        // Cleanup.
        dev.destroy_fence(fence, None);
        dev.destroy_command_pool(cmd_pool, None);
        dev.destroy_descriptor_pool(pool, None);
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_pipeline_layout(pipeline_layout, None);
        dev.destroy_descriptor_set_layout(set_layout, None);
        dev.destroy_shader_module(module, None);
        for (buf, mem) in [(a_buf, a_mem), (b_buf, b_mem), (c_buf, c_mem)] {
            dev.unmap_memory(mem);
            dev.destroy_buffer(buf, None);
            dev.free_memory(mem, None);
        }
        Ok(out)
    }

    /// LDS-blocked dense fp16 coopmat GEMM: C[M,N] = A[M,K] * B[K,N], run
    /// `iters` times back-to-back (one submit). Returns the C of the last run
    /// plus the total GPU time in ms. M,N multiples of 64, K of 16. This is the
    /// coopmat compute core (Phase 2.1 minus Q4_K dequant) — used to measure
    /// raw coopmat throughput vs the current wgpu path.
    pub fn coopmat_gemm_f16(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert!(m % 64 == 0 && n % 64 == 0 && k % 16 == 0);
        unsafe { self.coopmat_gemm_f16_inner(m, n, k, a, b, iters) }
    }

    unsafe fn coopmat_gemm_f16_inner(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
        use std::time::Instant;
        let dev = &self.device;
        let (a_buf, a_mem, a_ptr) = self.uma_buffer((m * k * 2) as u64)?;
        let (b_buf, b_mem, b_ptr) = self.uma_buffer((k * n * 2) as u64)?;
        let (c_buf, c_mem, c_ptr) = self.uma_buffer((m * n * 4) as u64)?;
        let (p_buf, p_mem, p_ptr) = self.uma_buffer(16)?;
        let a16: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        std::ptr::copy_nonoverlapping(a16.as_ptr() as *const u8, a_ptr, m * k * 2);
        std::ptr::copy_nonoverlapping(b16.as_ptr() as *const u8, b_ptr, k * n * 2);
        let dims = [m as u32, n as u32, k as u32, 0u32];
        std::ptr::copy_nonoverlapping(dims.as_ptr() as *const u8, p_ptr, 16);

        let spv = ash::util::read_spv(&mut std::io::Cursor::new(COOPMAT_GEMM_SPV)).map_err(|e| format!("read_spv: {e}"))?;
        let module = dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None).map_err(|e| format!("module: {e}"))?;

        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let set_layout = dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("setlayout: {e}"))?;
        let pipeline_layout = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).map_err(|e| format!("playout: {e}"))?;

        let entry = std::ffi::CString::new("main").unwrap();
        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut req_sg);
        let pipeline = dev.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)], None)
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0];

        let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
        ]), None).map_err(|e| format!("pool: {e}"))?;
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&set_layout))).map_err(|e| format!("allocset: {e}"))?[0];
        let info = |buf| [vk::DescriptorBufferInfo::default().buffer(buf).range(vk::WHOLE_SIZE)];
        let (ai, bi, ci, pi) = (info(a_buf), info(b_buf), info(c_buf), info(p_buf));
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ai),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ci),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&pi),
        ];
        dev.update_descriptor_sets(&writes, &[]);

        let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family), None).map_err(|e| format!("cmdpool: {e}"))?;
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).map_err(|e| format!("cmdbuf: {e}"))?[0];
        let gx = (n / 64) as u32; let gy = (m / 64) as u32;
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        for _ in 0..iters {
            dev.cmd_dispatch(cmd, gx, gy, 1);
            dev.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
        dev.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let cmds = [cmd];
        let t0 = Instant::now();
        dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).map_err(|e| format!("submit: {e}"))?;
        dev.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut out = vec![0f32; m * n];
        std::ptr::copy_nonoverlapping(c_ptr as *const f32, out.as_mut_ptr(), m * n);

        dev.destroy_fence(fence, None);
        dev.destroy_command_pool(cmd_pool, None);
        dev.destroy_descriptor_pool(pool, None);
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_pipeline_layout(pipeline_layout, None);
        dev.destroy_descriptor_set_layout(set_layout, None);
        dev.destroy_shader_module(module, None);
        for (buf, mem) in [(a_buf, a_mem), (b_buf, b_mem), (c_buf, c_mem), (p_buf, p_mem)] {
            dev.unmap_memory(mem); dev.destroy_buffer(buf, None); dev.free_memory(mem, None);
        }
        Ok((out, ms))
    }

    /// Q4_K coopmat prefill GEMM (Phase 2.1): C[M,N] = x[M,K] * dequant(W)[K,N]
    /// where W is a raw Q4_K weight matrix [N,K] (`weight_bytes` = N*nb*144).
    /// The weight is dequantized into the fp16 LDS tile before the coopmat
    /// multiply. Returns C[M,N] (last run) + total GPU ms over `iters`.
    /// This is the kernel that replaces the wgpu f32 Q4_K GEMM in prefill.
    /// Small-M (M<=16) Q4_K coopmat GEMM — one 16x128 tile/workgroup, x read once
    /// per tile (fixes the weight-stationary x-reread). Output buffer holds 16 rows.
    pub fn coopmat_q4k_gemm_m16(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], m: usize, iters: u32) -> Result<(Vec<f32>, f64), String> {
        assert!(m <= 16 && n % 128 == 0 && (nb * 256) % 256 == 0);
        unsafe { self.coopmat_q4k_gemm_dispatch(weight_bytes, n, nb, x, m, iters, true) }
    }
    pub fn coopmat_q4k_gemm(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], m: usize, iters: u32) -> Result<(Vec<f32>, f64), String> {
        let k = nb * 256;
        assert_eq!(weight_bytes.len(), n * nb * 144);
        assert_eq!(x.len(), m * k);
        assert!(m % 128 == 0 && n % 128 == 0, "register-blocked GEMM tile is 128x128");
        unsafe { self.coopmat_q4k_gemm_dispatch(weight_bytes, n, nb, x, m, iters, false) }
    }

    unsafe fn coopmat_q4k_gemm_dispatch(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], m: usize, iters: u32, m16: bool) -> Result<(Vec<f32>, f64), String> {
        use std::time::Instant;
        let dev = &self.device;
        let k = nb * 256;
        let crows = if m16 { 16 } else { m }; // the M16 kernel always writes a 16-row tile
        let (a_buf, a_mem, a_ptr) = self.uma_buffer((m * k * 2) as u64)?;
        let (w_buf, w_mem, w_ptr) = self.uma_buffer(weight_bytes.len() as u64)?;
        let (c_buf, c_mem, c_ptr) = self.uma_buffer((crows * n * 4) as u64)?;
        let (p_buf, p_mem, p_ptr) = self.uma_buffer(16)?;
        let a16: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        std::ptr::copy_nonoverlapping(a16.as_ptr() as *const u8, a_ptr, m * k * 2);
        std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), w_ptr, weight_bytes.len());
        let dims = [m as u32, n as u32, k as u32, nb as u32];
        std::ptr::copy_nonoverlapping(dims.as_ptr() as *const u8, p_ptr, 16);

        let spv = ash::util::read_spv(&mut std::io::Cursor::new(if m16 { COOPMAT_Q4K_GEMM_M16_SPV } else { COOPMAT_Q4K_GEMM_SPV })).map_err(|e| format!("read_spv: {e}"))?;
        let module = dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None).map_err(|e| format!("module: {e}"))?;

        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let set_layout = dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("setlayout: {e}"))?;
        let pipeline_layout = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).map_err(|e| format!("playout: {e}"))?;

        let entry = std::ffi::CString::new("main").unwrap();
        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut req_sg);
        let pipeline = dev.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)], None)
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0];

        let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
        ]), None).map_err(|e| format!("pool: {e}"))?;
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&set_layout))).map_err(|e| format!("allocset: {e}"))?[0];
        let info = |buf| [vk::DescriptorBufferInfo::default().buffer(buf).range(vk::WHOLE_SIZE)];
        let (ai, wi, ci, pi) = (info(a_buf), info(w_buf), info(c_buf), info(p_buf));
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ai),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&wi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ci),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&pi),
        ];
        dev.update_descriptor_sets(&writes, &[]);

        let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family), None).map_err(|e| format!("cmdpool: {e}"))?;
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).map_err(|e| format!("cmdbuf: {e}"))?[0];
        let gx = (n / 128) as u32; let gy = if m16 { 1 } else { (m / 128) as u32 };
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        for _ in 0..iters {
            dev.cmd_dispatch(cmd, gx, gy, 1);
            dev.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
        dev.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let cmds = [cmd];
        let t0 = Instant::now();
        dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).map_err(|e| format!("submit: {e}"))?;
        dev.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut out = vec![0f32; m * n];
        std::ptr::copy_nonoverlapping(c_ptr as *const f32, out.as_mut_ptr(), m * n);

        dev.destroy_fence(fence, None);
        dev.destroy_command_pool(cmd_pool, None);
        dev.destroy_descriptor_pool(pool, None);
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_pipeline_layout(pipeline_layout, None);
        dev.destroy_descriptor_set_layout(set_layout, None);
        dev.destroy_shader_module(module, None);
        for (buf, mem) in [(a_buf, a_mem), (w_buf, w_mem), (c_buf, c_mem), (p_buf, p_mem)] {
            dev.unmap_memory(mem); dev.destroy_buffer(buf, None); dev.free_memory(mem, None);
        }
        Ok((out, ms))
    }

    /// Decode matvec (Q4_K): out[N] = dequant(W)[N,K] · x[K], M=1. Returns
    /// out[N] (last run) + total GPU ms over `iters`. The bandwidth-bound
    /// decode kernel (subgroup-per-row, subgroupAdd reduction, no LDS/barrier).
    pub fn decode_matvec_q4k(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
        let k = nb * 256;
        assert_eq!(weight_bytes.len(), n * nb * 144);
        assert_eq!(x.len(), k);
        let spv = ash::util::read_spv(&mut std::io::Cursor::new(DECODE_MATVEC_Q4K_SPV)).map_err(|e| format!("read_spv: {e}"))?;
        unsafe { self.decode_matvec_q4k_inner(&spv, weight_bytes, n, nb, x, iters) }
    }

    /// Run a 4-binding decode matvec (storage W,X,O + uniform{N,K,nb,gx}) from
    /// arbitrary SPIR-V `spv` (words) over a packed weight buffer — the shader
    /// owns its block stride, so this is dtype-agnostic. Used by the Q3 kernel
    /// validation (naga-compiled WGSL) and reusable for any 1-row matvec kernel.
    pub fn decode_matvec_spv(&self, spv: &[u32], weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
        unsafe { self.decode_matvec_q4k_inner(spv, weight_bytes, n, nb, x, iters) }
    }

    /// Megakernel FOUNDATION probe: run a PERSISTENT 2-matvec kernel (y1=W1·x →
    /// grid-sync → y2=W2·y1) in ONE dispatch of `n_wg` workgroups. Tests whether
    /// an atomic grid-wide barrier is COHERENT on this GPU (WGSL has no device-scope
    /// barrier; this relies on RDNA3's L2 being the coherence point). If y2 is
    /// bit-exact, a full megakernel is buildable; a race means it isn't (here).
    /// `n_wg` MUST be ≤ the max resident workgroups or the spin deadlocks (the
    /// shader has a spin cap so it can't hang the GPU).
    pub fn megakernel_poc(&self, spv: &[u32], w1: &[f32], w2: &[f32], x: &[f32], n1: usize, k: usize, n2: usize, n_wg: u32) -> Result<(Vec<f32>, f64), String> {
        use ash::vk; use std::time::Instant;
        unsafe {
            let dev = &self.device;
            let mkf = |v: &[f32]| -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
                let (b, m, p) = self.uma_buffer((v.len() * 4).max(4) as u64)?;
                std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, p, v.len() * 4); Ok((b, m, p))
            };
            let (w1b, w1m, _) = mkf(w1)?; let (w2b, w2m, _) = mkf(w2)?; let (xb, xm, _) = mkf(x)?;
            let (y1b, y1m, _) = self.uma_buffer((n1 * 4) as u64)?;
            let (y2b, y2m, y2p) = self.uma_buffer((n2 * 4) as u64)?;
            let (sb, sm, sp) = self.uma_buffer(16)?; std::ptr::write_bytes(sp, 0, 16); // sync counters = 0
            let (pb, pm, pp) = self.uma_buffer(16)?;
            std::ptr::copy_nonoverlapping([n1 as u32, k as u32, n2 as u32, n_wg].as_ptr() as *const u8, pp, 16);
            let spvb: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let (pipe, layout, sl, module) = self.make_pipeline_raw(&spvb, 6);
            let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(6),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).map_err(|e| format!("pool: {e}"))?;
            let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[sl])).map_err(|e| format!("set: {e}"))?[0];
            let info = |b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)];
            let st = [info(w1b), info(w2b), info(xb), info(y1b), info(y2b), info(sb)];
            let iu = info(pb);
            let mut wr: Vec<vk::WriteDescriptorSet> = (0..6).map(|b| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&st[b])).collect();
            wr.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&iu));
            dev.update_descriptor_sets(&wr, &[]);
            let cp = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family), None).map_err(|e| format!("cp: {e}"))?;
            let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cp).command_buffer_count(1)).map_err(|e| format!("cb: {e}"))?[0];
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            dev.cmd_dispatch(cmd, n_wg, 1, 1);
            dev.end_command_buffer(cmd).unwrap();
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let t0 = Instant::now();
            dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).map_err(|e| format!("submit: {e}"))?;
            dev.wait_for_fences(&[fence], true, 5_000_000_000).map_err(|e| format!("wait/timeout: {e}"))?; // 5s backstop
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut out = vec![0f32; n2];
            std::ptr::copy_nonoverlapping(y2p as *const f32, out.as_mut_ptr(), n2);
            dev.destroy_fence(fence, None); dev.destroy_command_pool(cp, None); dev.destroy_descriptor_pool(pool, None);
            dev.destroy_pipeline(pipe, None); dev.destroy_pipeline_layout(layout, None); dev.destroy_descriptor_set_layout(sl, None); dev.destroy_shader_module(module, None);
            for (b, m) in [(w1b, w1m), (w2b, w2m), (xb, xm), (y1b, y1m), (y2b, y2m), (sb, sm), (pb, pm)] { dev.unmap_memory(m); dev.destroy_buffer(b, None); dev.free_memory(m, None); }
            Ok((out, ms))
        }
    }

    /// Measure the cost of `n_sync` atomic grid-wide barriers in a persistent
    /// dispatch of `n_wg` workgroups (no other work). per-sync = (cost(N)-cost(1))/(N-1).
    /// Decides megakernel viability: a forward has ~194 syncs/token, so >2µs/sync
    /// (>0.4ms, >8%) would sink it. Buffers: sync array (binding 0) + uniform.
    pub fn grid_sync_cost(&self, spv: &[u32], n_wg: u32, n_sync: u32) -> Result<f64, String> {
        use ash::vk; use std::time::Instant;
        unsafe {
            let dev = &self.device;
            let (sb, sm, sp) = self.uma_buffer(1024)?; std::ptr::write_bytes(sp, 0, 1024); // 256 u32 counters
            let (pb, pm, pp) = self.uma_buffer(16)?;
            std::ptr::copy_nonoverlapping([n_wg, n_sync, 0u32, 0u32].as_ptr() as *const u8, pp, 16);
            let spvb: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let (pipe, layout, sl, module) = self.make_pipeline_raw(&spvb, 1);
            let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1),
                vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)]), None).map_err(|e| format!("pool: {e}"))?;
            let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[sl])).map_err(|e| format!("set: {e}"))?[0];
            let si = [vk::DescriptorBufferInfo::default().buffer(sb).range(vk::WHOLE_SIZE)];
            let ui = [vk::DescriptorBufferInfo::default().buffer(pb).range(vk::WHOLE_SIZE)];
            dev.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&si),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ui)], &[]);
            let cp = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family), None).unwrap();
            let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cp).command_buffer_count(1)).unwrap()[0];
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            dev.cmd_dispatch(cmd, n_wg, 1, 1);
            dev.end_command_buffer(cmd).unwrap();
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let t0 = Instant::now();
            dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).map_err(|e| format!("submit: {e}"))?;
            dev.wait_for_fences(&[fence], true, 5_000_000_000).map_err(|e| format!("wait: {e}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            dev.destroy_fence(fence, None); dev.destroy_command_pool(cp, None); dev.destroy_descriptor_pool(pool, None);
            dev.destroy_pipeline(pipe, None); dev.destroy_pipeline_layout(layout, None); dev.destroy_descriptor_set_layout(sl, None); dev.destroy_shader_module(module, None);
            for (b, m) in [(sb, sm), (pb, pm)] { dev.unmap_memory(m); dev.destroy_buffer(b, None); dev.free_memory(m, None); }
            Ok(ms)
        }
    }

    unsafe fn decode_matvec_q4k_inner(&self, spv: &[u32], weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
        use std::time::Instant;
        let dev = &self.device;
        let k = nb * 256;
        let (w_buf, w_mem, w_ptr) = self.uma_buffer(weight_bytes.len() as u64)?;
        let (x_buf, x_mem, x_ptr) = self.uma_buffer((k * 4) as u64)?;
        let (o_buf, o_mem, o_ptr) = self.uma_buffer((n * 4) as u64)?;
        let (p_buf, p_mem, p_ptr) = self.uma_buffer(16)?;
        std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), w_ptr, weight_bytes.len());
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, x_ptr, k * 4);
        let gx = (n as u32).min(65535);
        let dims = [n as u32, k as u32, nb as u32, gx];
        std::ptr::copy_nonoverlapping(dims.as_ptr() as *const u8, p_ptr, 16);

        let module = dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None).map_err(|e| format!("module: {e}"))?;

        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let set_layout = dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("setlayout: {e}"))?;
        let pipeline_layout = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).map_err(|e| format!("playout: {e}"))?;

        let entry = std::ffi::CString::new("main").unwrap();
        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut req_sg);
        let pipeline = dev.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)], None)
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0];

        let pool = dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
        ]), None).map_err(|e| format!("pool: {e}"))?;
        let set = dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&set_layout))).map_err(|e| format!("allocset: {e}"))?[0];
        let info = |buf| [vk::DescriptorBufferInfo::default().buffer(buf).range(vk::WHOLE_SIZE)];
        let (wi, xi, oi, pi) = (info(w_buf), info(x_buf), info(o_buf), info(p_buf));
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&wi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&xi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&oi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&pi),
        ];
        dev.update_descriptor_sets(&writes, &[]);

        let cmd_pool = dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family), None).map_err(|e| format!("cmdpool: {e}"))?;
        let cmd = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).map_err(|e| format!("cmdbuf: {e}"))?[0];
        let gy = (n as u32).div_ceil(gx);
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        for _ in 0..iters {
            dev.cmd_dispatch(cmd, gx, gy, 1);
            dev.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
        dev.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let cmds = [cmd];
        let t0 = Instant::now();
        dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).map_err(|e| format!("submit: {e}"))?;
        dev.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut out = vec![0f32; n];
        std::ptr::copy_nonoverlapping(o_ptr as *const f32, out.as_mut_ptr(), n);

        dev.destroy_fence(fence, None);
        dev.destroy_command_pool(cmd_pool, None);
        dev.destroy_descriptor_pool(pool, None);
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_pipeline_layout(pipeline_layout, None);
        dev.destroy_descriptor_set_layout(set_layout, None);
        dev.destroy_shader_module(module, None);
        for (buf, mem) in [(w_buf, w_mem), (x_buf, x_mem), (o_buf, o_mem), (p_buf, p_mem)] {
            dev.unmap_memory(mem); dev.destroy_buffer(buf, None); dev.free_memory(mem, None);
        }
        Ok((out, ms))
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// VkModel: a real-weight raw-Vulkan decode engine over a Llama GGUF.
// Mirrors the wgpu GpuModel but runs the validated ash decode kernels. The
// dequantized token embedding doubles as a tied Q6_K LM head; Q4_K transformer
// weights + a per-layer KV cache live in resident UMA buffers; the forward is
// re-recorded per token (cheap in raw Vulkan) so the flash-attn grid grows with
// context. Single resident KV stream.
// ---------------------------------------------------------------------------

unsafe fn vk_up_bytes(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, data: &[u8]) -> ash::vk::Buffer {
    let (b, m, p) = ctx.uma_buffer(data.len().max(4) as u64).unwrap();
    std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
    bufs.push((b, m)); b
}
unsafe fn vk_up_f32(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, data: &[f32]) -> ash::vk::Buffer {
    let (b, m, p) = ctx.uma_buffer((data.len().max(1) * 4) as u64).unwrap();
    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p, data.len() * 4);
    bufs.push((b, m)); b
}
unsafe fn vk_zeros(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, len: usize) -> (ash::vk::Buffer, *mut u8) {
    let (b, m, p) = ctx.uma_buffer((len.max(1) * 4) as u64).unwrap();
    std::ptr::write_bytes(p, 0, len * 4);
    bufs.push((b, m)); (b, p)
}
unsafe fn vk_uni(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, d: [u32; 4]) -> (ash::vk::Buffer, *mut u8) {
    let (b, m, p) = ctx.uma_buffer(16).unwrap();
    std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16);
    bufs.push((b, m)); (b, p)
}
unsafe fn vk_uni8(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, d: [u32; 8]) -> ash::vk::Buffer {
    let (b, m, p) = ctx.uma_buffer(32).unwrap();
    std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 32);
    bufs.push((b, m)); b
}
unsafe fn vk_uni8p(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, d: [u32; 8]) -> (ash::vk::Buffer, *mut u8) {
    let (b, m, p) = ctx.uma_buffer(32).unwrap();
    std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 32);
    bufs.push((b, m)); (b, p)
}
// Repack a raw Q6_K tensor (210-byte blocks) into aligned SoA buffers: ql
// (128 B/blk), qh (64 B), scales (16 i8/blk, sign-extracted in-shader), d (f32).
// Scales stay i8 (not expanded to f32) so the matvec stays bandwidth-bound.
unsafe fn vk_up_q6k(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, bytes: &[u8], nbk: usize) -> (ash::vk::Buffer, ash::vk::Buffer, ash::vk::Buffer, ash::vk::Buffer) {
    assert_eq!(bytes.len(), nbk * 210, "Q6_K byte length");
    let mut ql = vec![0u8; nbk * 128]; let mut qh = vec![0u8; nbk * 64];
    let mut scl = vec![0u8; nbk * 16]; let mut dd = vec![0f32; nbk];
    for b in 0..nbk {
        let base = b * 210;
        ql[b * 128..b * 128 + 128].copy_from_slice(&bytes[base..base + 128]);
        qh[b * 64..b * 64 + 64].copy_from_slice(&bytes[base + 128..base + 192]);
        scl[b * 16..b * 16 + 16].copy_from_slice(&bytes[base + 192..base + 208]); // raw i8 scales
        dd[b] = half::f16::from_bits(u16::from_le_bytes([bytes[base + 208], bytes[base + 209]])).to_f32();
    }
    (vk_up_bytes(ctx, bufs, &ql), vk_up_bytes(ctx, bufs, &qh), vk_up_bytes(ctx, bufs, &scl), vk_up_f32(ctx, bufs, &dd))
}

// Repack candle Q3_K blocks (110 B) into the decode_matvec_q3k layout (29 u32 /
// block, 4-aligned): d_f32 | 16 pre-shuffled 6-bit scale bytes | hmask[32] |
// qs[64], then upload as one buffer. Mirrors tests/vk_q3k_matvec::pack_q3k.
unsafe fn vk_up_q3k(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, bytes: &[u8], n_blocks: usize) -> ash::vk::Buffer {
    assert_eq!(bytes.len(), n_blocks * 110, "Q3_K byte length");
    let (km1, km2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let u = |s: &[u8], i: usize| u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]);
    let mut packed = vec![0u32; n_blocks * 29];
    for bi in 0..n_blocks {
        let blk = &bytes[bi * 110..][..110];
        let base = bi * 29;
        packed[base] = half::f16::from_bits(u16::from_le_bytes([blk[108], blk[109]])).to_f32().to_bits();
        let (s0, s1, s2) = (u(blk, 96), u(blk, 100), u(blk, 104));
        let a = [
            (s0 & km2) | ((s2 & km1) << 4),
            (s1 & km2) | (((s2 >> 2) & km1) << 4),
            ((s0 >> 4) & km2) | (((s2 >> 4) & km1) << 4),
            ((s1 >> 4) & km2) | (((s2 >> 6) & km1) << 4),
        ];
        for w in 0..4 { packed[base + 1 + w] = a[w]; }
        for w in 0..8 { packed[base + 5 + w] = u(blk, w * 4); }
        for w in 0..16 { packed[base + 13 + w] = u(blk, 32 + w * 4); }
    }
    let pb: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    vk_up_bytes(ctx, bufs, &pb)
}

// Upload an [N,K] f32 weight as a TRANSPOSED f16 buffer [K,N] (the layout the
// dense coopmat GEMM wants for B: C[M,N]=A[M,K]·B[K,N]). For the Q6 prefill path.
unsafe fn vk_up_f16t(ctx: &VkContext, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, deq: &[f32], n: usize, k: usize) -> ash::vk::Buffer {
    let mut t = vec![0u16; k * n];
    for nn in 0..n { for kk in 0..k { t[kk * n + nn] = half::f16::from_f32(deq[nn * k + kk]).to_bits(); } }
    let (b, m, p) = ctx.uma_buffer((k * n * 2) as u64).unwrap();
    std::ptr::copy_nonoverlapping(t.as_ptr() as *const u8, p, k * n * 2);
    bufs.push((b, m)); b
}

// Per-layer weights needed by the batched prefill GEMMs.
struct PrefillW {
    wq: ash::vk::Buffer, wk: ash::vk::Buffer, wo: ash::vk::Buffer, w13: ash::vk::Buffer, // Q4 raw bytes
    wqk: Option<ash::vk::Buffer>, // ZLLM_FUSED_QKV: concat wq+wk for one coopmat GEMM
    wv_f16: ash::vk::Buffer, w2_f16: ash::vk::Buffer,                                    // Q6 dequant -> f16 [K,N]
    attn_norm: ash::vk::Buffer, ffn_norm: ash::vk::Buffer,                               // f32
    kc: ash::vk::Buffer, vc: ash::vk::Buffer,                                            // shared decode KV cache
}

const PREFILL_MAX_M: usize = 1024; // single-pass batched prefill cap. The per-call tile is dynamic
// (round_up(real_m,128)), so short prompts stay small; raising the CAP lets prompts up to 1024 tok
// prefill in ONE coopmat forward (the prefill→decode handoff for RAG-scale prompts). Buffers grow
// to this (gu = 1024*2*n_inter*4 ≈ 67MB). M=128 GEMM 4.4 TFLOP/s, M=256 8.1, M=512 10.2 (amortizes).
const VERIFY_MAX_M: usize = 8;    // spec-decode verify window cap (matvec MAXM)

// Resources for the batched prefill forward (built once at load).
type Pipe3 = (ash::vk::Pipeline, ash::vk::PipelineLayout, ash::vk::DescriptorSetLayout);

// A recorded-once prefill/batched-forward command + the resources it owns. The sets
// live in `pool` (never reset); the per-dispatch uniform buffers are in `unis`.
struct PfRec {
    cmd: ash::vk::CommandBuffer,
    pool: ash::vk::DescriptorPool,
    cmd_pool: ash::vk::CommandPool,
    fence: ash::vk::Fence,
    unis: Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>,
}
impl PfRec {
    unsafe fn destroy(self, dv: &ash::Device) {
        dv.destroy_fence(self.fence, None);
        dv.destroy_command_pool(self.cmd_pool, None);
        dv.destroy_descriptor_pool(self.pool, None);
        for (b, m) in self.unis { dv.unmap_memory(m); dv.destroy_buffer(b, None); dv.free_memory(m, None); }
    }
}

struct PrefillRes {
    w: Vec<PrefillW>,
    p_q4: Pipe3, p_f16: Pipe3, p_bn: Pipe3, p_br: Pipe3, p_bs: Pipe3, p_bsi: Pipe3, p_t16: Pipe3,
    p_add: Pipe3, p_q6k: Pipe3, p_bsd: Pipe3, p_bso: Pipe3,
    // Small-M weight-stationary matvecs for spec-decode verify (M = draft window).
    p_bmq4: Pipe3, p_bmf16: Pipe3, p_bmq6: Pipe3,
    p_skinny: Pipe3,  // column-tiled skinny Q4 GEMM (faster batched matvec; ZLLM_VERIFY_BMV forces bmv)
    vlogits: ash::vk::Buffer,  // [VERIFY_MAX_M, vocab] batched LM-head logits
    lm_ql: ash::vk::Buffer, lm_qh: ash::vk::Buffer, lm_scl: ash::vk::Buffer, lm_dd: ash::vk::Buffer,
    final_norm: ash::vk::Buffer, logits: ash::vk::Buffer, logits_ptr: *mut u8,
    // scratch (sized for PREFILL_MAX_M rows)
    x32: ash::vk::Buffer, x32_ptr: *mut u8, #[allow(dead_code)] x16: ash::vk::Buffer, // kept: pre-fused-QKV staging, still allocated
    n32: ash::vk::Buffer, n16: ash::vk::Buffer, q: ash::vk::Buffer, qk: ash::vk::Buffer,
    attn32: ash::vk::Buffer, attn16: ash::vk::Buffer, gu: ash::vk::Buffer,
    h32: ash::vk::Buffer, h16: ash::vk::Buffer, o32: ash::vk::Buffer, ffn32: ash::vk::Buffer,
    cosb: ash::vk::Buffer, cosb_ptr: *mut u8, sinb: ash::vk::Buffer, sinb_ptr: *mut u8,
    // Coopmat attention (prefill SDPA on WMMA — was 34% of prefill at 2% of peak):
    // f16 views of Q/K/V, per-8-head-batch S scratch + f16 probs, GEMM + softmax pipes.
    p_cma: Pipe3, p_cma64: Pipe3, p_smax: Pipe3, p_cfa: Pipe3,
    q16: ash::vk::Buffer, k16: ash::vk::Buffer, v16: ash::vk::Buffer,
    s32: ash::vk::Buffer, prob16: ash::vk::Buffer,
}

// A loaded weight: Q4_K (raw bytes) or Q6_K (repacked SoA). nb = cols/256.
enum VkWeight {
    Q4 { buf: ash::vk::Buffer, nb: usize },
    Q6 { ql: ash::vk::Buffer, qh: ash::vk::Buffer, scl: ash::vk::Buffer, dd: ash::vk::Buffer, nb: usize },
    // Packed Q3_K (29 u32/block, vk_up_q3k layout): same 3-storage interface as Q4.
    Q3 { buf: ash::vk::Buffer, nb: usize },
}
// A matvec dispatch: which kernel pipeline (0=p_mv/Q4, 1=p_q6k/Q6, 2=p_q3/Q3) + its set.
#[derive(Clone, Copy)]
struct Mv { pipe: u8, set: ash::vk::DescriptorSet }

unsafe fn vk_alloc_set(dv: &ash::Device, pool: ash::vk::DescriptorPool, sl: ash::vk::DescriptorSetLayout, sb: &[ash::vk::Buffer], u: ash::vk::Buffer) -> ash::vk::DescriptorSet {
    use ash::vk;
    let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = sb.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
    let uinfo = [vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)];
    let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
    w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(sb.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
    dv.update_descriptor_sets(&w, &[]);
    set
}

// Like vk_alloc_set but each storage binding carries an (offset, range) — lets a
// dispatch read/write a sub-window of a buffer (e.g. write K/V into the resident
// KV cache at position `pos`).
unsafe fn vk_alloc_set_off(dv: &ash::Device, pool: ash::vk::DescriptorPool, sl: ash::vk::DescriptorSetLayout, sb: &[(ash::vk::Buffer, u64, u64)], u: ash::vk::Buffer) -> ash::vk::DescriptorSet {
    use ash::vk;
    let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = sb.iter().map(|&(b, off, range)| [vk::DescriptorBufferInfo::default().buffer(b).offset(off).range(range)]).collect();
    let uinfo = [vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)];
    let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
    w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(sb.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
    dv.update_descriptor_sets(&w, &[]);
    set
}

// Build a matvec descriptor set for a weight (Q4 or Q6) producing `rows` outputs.
// `acc` folds a residual add into the matvec output (out[row] += proj instead of =),
// removing the separate residual-add dispatch. Q4 packs acc into bit 16 of gx (kernel
// masks it); Q6 uses its spare uniform field. `out` is then bound to the residual stream.
unsafe fn vk_mk_mv(ctx: &VkContext, pool: ash::vk::DescriptorPool, mv_sl: ash::vk::DescriptorSetLayout, q6k_sl: ash::vk::DescriptorSetLayout, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, w: &VkWeight, x_in: ash::vk::Buffer, out: ash::vk::Buffer, rows: usize, acc: bool) -> Mv {
    let gx = (rows as u32).min(65535);
    match *w {
        VkWeight::Q4 { buf, nb } => {
            let (u, _) = vk_uni(ctx, bufs, [rows as u32, (nb * 256) as u32, nb as u32, gx | ((acc as u32) << 16)]);
            Mv { pipe: 0, set: vk_alloc_set(&ctx.device, pool, mv_sl, &[buf, x_in, out], u) }
        }
        VkWeight::Q6 { ql, qh, scl, dd, nb } => {
            let (u, _) = vk_uni(ctx, bufs, [rows as u32, nb as u32, gx, acc as u32]);
            Mv { pipe: 1, set: vk_alloc_set(&ctx.device, pool, q6k_sl, &[ql, qh, scl, dd, x_in, out], u) }
        }
        // Q3_K reuses the Q4 3-storage set + uniform {N,K,nb,gx}; only the pipeline differs.
        VkWeight::Q3 { buf, nb } => {
            let (u, _) = vk_uni(ctx, bufs, [rows as u32, (nb * 256) as u32, nb as u32, gx | ((acc as u32) << 16)]);
            Mv { pipe: 2, set: vk_alloc_set(&ctx.device, pool, mv_sl, &[buf, x_in, out], u) }
        }
    }
}

#[derive(Clone)]
struct VkLayerOps {
    attn_norm: ash::vk::DescriptorSet, #[allow(dead_code)] wq: Mv, #[allow(dead_code)] wk: Mv, wv: Mv, // wq/wk superseded by the fused mvrope sets
    wq_rope: ash::vk::DescriptorSet, wk_rope: ash::vk::DescriptorSet, // QKV+rope fused (q/k)
    wqk_rope: Option<ash::vk::DescriptorSet>, // ZLLM_FUSED_QKV: one mvrope over concat wq+wk
    kvw_k: ash::vk::DescriptorSet, kvw_v: ash::vk::DescriptorSet, sdpa: ash::vk::DescriptorSet,
    fp: ash::vk::DescriptorSet, fc: ash::vk::DescriptorSet, wo: Mv,
    fc1: ash::vk::DescriptorSet, fc2: ash::vk::DescriptorSet, // hierarchical combine: L1 (part->super), L2 (super->attn)
    ffn_norm: ash::vk::DescriptorSet, w13: Mv, w2: Mv,
}

/// A loaded GGUF running on the raw-Vulkan decode kernels.
pub struct VkModel {
    pub n_embd: usize, n_head: usize, n_kv: usize, hd: usize, n_inter: usize,
    pub vocab: usize, n_layers: usize, kv_dim: usize, half: usize, max_seq: usize, eps: f32,
    headmajor: bool, // ZLLM_HEADMAJOR_KV: cache [kv_head,pos,hd] + always-flash (decode-cache-fill only)
    embed: Vec<f32>, cos: Vec<f32>, sin: Vec<f32>,
    // dispatch dims
    lm_nb: usize,
    // per-token mapped uniforms / activations
    x_ptr: *mut u8, cos_ptr: *mut u8, sin_ptr: *mut u8, base_ptr: *mut u8, seq_ptr: *mut u8, logits_ptr: *mut u8,
    l1_ptr: *mut u8, l2_ptr: *mut u8, // hierarchical-combine L1/L2 uniforms (refreshed per token)
    // descriptor sets
    layers: Vec<VkLayerOps>,
    #[allow(dead_code)] s_rope_q: ash::vk::DescriptorSet, #[allow(dead_code)] s_rope_k: ash::vk::DescriptorSet, s_silu: ash::vk::DescriptorSet, // rope sets superseded by fused mvrope
    s_final_norm: ash::vk::DescriptorSet, s_lm: ash::vk::DescriptorSet, s_argmax: ash::vk::DescriptorSet,
    lm_q4: bool, // tied LM head is Q4_K (all-Q4 model) → Q4 matvec instead of Q6 (decode path)
    argmax_ptr: *mut u8,
    // Record-once cache: the command buffer only changes when the flash-attn
    // grid (n_blocks) or the lm/argmax tail changes — reuse it otherwise and
    // just refresh the mapped uniforms. -1 = not yet recorded.
    last_rec: std::cell::Cell<i64>,
    // pipelines (pipeline, layout)
    p_rms: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_mv: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_q3: (ash::vk::Pipeline, ash::vk::PipelineLayout), // Q3_K decode matvec (gate+up→Q3)
    #[allow(dead_code)] p_rope: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_kvw: (ash::vk::Pipeline, ash::vk::PipelineLayout), // p_rope superseded by fused mvrope
    p_sdpa: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_fp: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_fc: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_silu: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_ch: (ash::vk::Pipeline, ash::vk::PipelineLayout), // hierarchical (tree) flash combine
    p_fp2: (ash::vk::Pipeline, ash::vk::PipelineLayout), // 2-pass flash partial (hd<=64; faster, same output)
    p_kvw_hm: (ash::vk::Pipeline, ash::vk::PipelineLayout), // head-major KV write (ZLLM_HEADMAJOR_KV)
    p_fp2_hm: (ash::vk::Pipeline, ash::vk::PipelineLayout), // head-major flash partial (ZLLM_HEADMAJOR_KV)
    p_kvtr: (ash::vk::Pipeline, ash::vk::PipelineLayout, ash::vk::DescriptorSetLayout), // pos→head-major transpose

    #[allow(dead_code)] p_add: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_q6k: (ash::vk::Pipeline, ash::vk::PipelineLayout), // p_add superseded by acc-folded matvecs
    p_argmax: Pipe3, p_mvrope: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    // Pre-allocated scratch for verify_forward (spec-decode), reused every call so
    // it doesn't vkAllocateMemory ~280 uniforms per verification (the overhead that
    // made the first cut slower than greedy). A reusable uniform-buffer pool +
    // descriptor pool + command buffer + the per-position argmax readback buffer.
    verify_unis: Vec<(ash::vk::Buffer, *mut u8)>,
    verify_argmax: (ash::vk::Buffer, *mut u8),
    verify_pool: ash::vk::DescriptorPool,
    verify_cmd: ash::vk::CommandBuffer,
    verify_fence: ash::vk::Fence,
    // Record-once prefill/batched-forward: the 128-row command + its ~280 descriptor
    // sets/uniforms are recorded once (keyed on real_m) and reused — per call only the
    // mapped embeddings change. Kills ~118ms/call of alloc+record+cleanup (60% of the
    // forward) → ~78ms GPU-exec → ~1641 tok/s @ M=128, beating llama's 1458.
    prefill_rec: std::cell::RefCell<Option<((usize, usize), PfRec)>>, // keyed by (real_m, pos)
    /// Tokens whose K/V are resident in the cache (positions 0..len), maintained
    /// by `prefill_cached` + `note_decoded` for cross-request prefix reuse.
    /// Pos-major mode only (head-major transposes the cache after prefill, so
    /// continued chunk writes would mix layouts — reuse is disabled there).
    prompt_cache: std::cell::RefCell<Vec<u32>>,
    // owned resources for cleanup
    bufs: Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>,
    pipes: Vec<(ash::vk::Pipeline, ash::vk::PipelineLayout, ash::vk::DescriptorSetLayout, ash::vk::ShaderModule)>,
    desc_pool: ash::vk::DescriptorPool, cmd_pool: ash::vk::CommandPool, cmd: ash::vk::CommandBuffer, fence: ash::vk::Fence,
    prefill: PrefillRes,
    ctx: VkContext,
}

impl VkModel {
    /// Load a Llama GGUF onto the iGPU. `ctx` is consumed (owned by the model).
    pub fn load(path: &str, ctx: VkContext) -> Result<Self, String> {
        unsafe { Self::load_inner(path, ctx) }
    }

    unsafe fn load_inner(path: &str, ctx: VkContext) -> Result<Self, String> {
        use ash::vk;
        use candle_core::quantized::{gguf_file, GgmlDType};
        use candle_core::Device;
        let dev = Device::Cpu;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
        let mu = |k: &str| -> Result<u32, String> { ct.metadata.get(k).ok_or(format!("missing {k}"))?.to_u32().map_err(|e| e.to_string()) };
        let mf = |k: &str| -> Option<f32> { ct.metadata.get(k).and_then(|v| v.to_f32().ok()) };
        let n_head = mu("llama.attention.head_count")? as usize;
        let n_kv = mu("llama.attention.head_count_kv")? as usize;
        let n_layers = mu("llama.block_count")? as usize;
        let n_embd = mu("llama.embedding_length")? as usize;
        let eps = mf("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_base = mf("llama.rope.freq_base").unwrap_or(10000.0);
        let hd = n_embd / n_head;
        let kv_dim = n_kv * hd;
        let attn_dim = n_head * hd;
        let half = hd / 2;
        let max_seq = MAX_SEQ;

        let dv = &ctx.device;
        let mut bufs: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();

        // --- weights: Q4_K / Q6_K transformer layers (Q4_K_M upgrades attn_v +
        // ffn_down to Q6_K), f32 norms, Q6_K tied LM head ---
        let load_w = |bufs: &mut Vec<_>, file: &mut std::fs::File, name: &str| -> Result<VkWeight, String> {
            let qt = ct.tensor(file, name, &dev).map_err(|e| e.to_string())?;
            let dims = qt.shape().dims().to_vec();
            let nb = dims[1] / 256;
            let bytes = qt.data().map_err(|e| e.to_string())?;
            match qt.dtype() {
                GgmlDType::Q4K => Ok(VkWeight::Q4 { buf: vk_up_bytes(&ctx, bufs, &bytes), nb }),
                GgmlDType::Q6K => {
                    let (ql, qh, scl, dd) = vk_up_q6k(&ctx, bufs, &bytes, dims[0] * nb);
                    Ok(VkWeight::Q6 { ql, qh, scl, dd, nb })
                }
                d => Err(format!("{name}: unsupported dtype {d:?}")),
            }
        };
        // Concatenate ffn_gate + ffn_up into one Q4_K weight ([2*n_inter, n_embd])
        // so the FFN's two projections are one bigger matvec dispatch (saturates
        // the bus better than two medium ones). Both are Q4_K in Q4_K_M.
        // ZLLM_Q3_GATEUP keeps the Q4 w13 (prefill coopmat GEMM still needs it) AND
        // builds a Q3_K copy used by DECODE (validated recipe: +2.7% ppl, ~10% fewer
        // bytes/token → faster decode; see quant_sensitivity).
        let q3_gateup = std::env::var("ZLLM_Q3_GATEUP").is_ok();
        let load_gateup = |bufs: &mut Vec<_>, file: &mut std::fs::File, p: &str, as_q3: bool| -> Result<VkWeight, String> {
            let g = ct.tensor(file, &format!("{p}.ffn_gate.weight"), &dev).map_err(|e| e.to_string())?;
            let u = ct.tensor(file, &format!("{p}.ffn_up.weight"), &dev).map_err(|e| e.to_string())?;
            if g.dtype() != GgmlDType::Q4K || u.dtype() != GgmlDType::Q4K {
                return Err(format!("{p}: gate/up concat needs Q4_K (got {:?}/{:?})", g.dtype(), u.dtype()));
            }
            let nb = g.shape().dims()[1] / 256;
            if as_q3 {
                let gd = g.dequantize(&dev).map_err(|e| e.to_string())?;
                let ud = u.dequantize(&dev).map_err(|e| e.to_string())?;
                let cat = candle_core::Tensor::cat(&[&gd, &ud], 0).map_err(|e| e.to_string())?;
                let q3 = candle_core::quantized::QTensor::quantize(&cat, GgmlDType::Q3K).map_err(|e| e.to_string())?;
                let bytes = q3.data().map_err(|e| e.to_string())?;
                let n_blocks = cat.shape().dims()[0] * nb;
                return Ok(VkWeight::Q3 { buf: vk_up_q3k(&ctx, bufs, &bytes, n_blocks), nb });
            }
            let mut b = g.data().map_err(|e| e.to_string())?.to_vec();
            b.extend_from_slice(&u.data().map_err(|e| e.to_string())?);
            Ok(VkWeight::Q4 { buf: vk_up_bytes(&ctx, bufs, &b), nb })
        };
        // ZLLM_FUSED_QKV concatenates wq+wk into one Q4 weight [n_embd+kv_dim, n_embd]
        // → one p_mvrope dispatch (rope is per-dim-in-head, head-count-agnostic) instead
        // of two small grid-starved ones. wv stays separate. See vk_prefill_gemm_floor.
        let fused_qkv = std::env::var("ZLLM_FUSED_QKV").is_ok();
        let load_wqk = |bufs: &mut Vec<_>, file: &mut std::fs::File, p: &str| -> Result<vk::Buffer, String> {
            let wqt = ct.tensor(file, &format!("{p}.attn_q.weight"), &dev).map_err(|e| e.to_string())?;
            let wkt = ct.tensor(file, &format!("{p}.attn_k.weight"), &dev).map_err(|e| e.to_string())?;
            if wqt.dtype() != GgmlDType::Q4K || wkt.dtype() != GgmlDType::Q4K {
                return Err(format!("{p}: fused QKV needs Q4_K wq/wk"));
            }
            let mut b = wqt.data().map_err(|e| e.to_string())?.to_vec();
            b.extend_from_slice(&wkt.data().map_err(|e| e.to_string())?);
            Ok(vk_up_bytes(&ctx, bufs, &b))
        };
        let load_norm = |bufs: &mut Vec<_>, file: &mut std::fs::File, name: &str| -> Result<vk::Buffer, String> {
            let qt = ct.tensor(file, name, &dev).map_err(|e| e.to_string())?;
            let v: Vec<f32> = qt.dequantize(&dev).map_err(|e| e.to_string())?.flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
            Ok(vk_up_f32(&ctx, bufs, &v))
        };

        // Embedding (dequantized) — also the tied LM head (Q6_K, repacked SoA).
        let embed_qt = ct.tensor(&mut file, "token_embd.weight", &dev).map_err(|e| e.to_string())?;
        let vocab = embed_qt.shape().dims()[0];
        let embed: Vec<f32> = embed_qt.dequantize(&dev).map_err(|e| e.to_string())?.flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
        let lm_bytes = embed_qt.data().map_err(|e| e.to_string())?;
        let lm_nb = n_embd / 256;
        let lm_q4 = embed_qt.dtype() == GgmlDType::Q4K;
        let (lm_ql, lm_qh, lm_scl, lm_dd) = match embed_qt.dtype() {
            GgmlDType::Q6K => vk_up_q6k(&ctx, &mut bufs, &lm_bytes, vocab * lm_nb),
            // all-Q4 model: one Q4_K buffer; lm_ql is the weight, the other three alias
            // it (unused by the Q4 decode LM head; prefill/verify stay Q6-only).
            GgmlDType::Q4K => { let q4 = vk_up_bytes(&ctx, &mut bufs, &lm_bytes); (q4, q4, q4, q4) }
            d => return Err(format!("LM head (token_embd.weight) dtype {d:?} not supported (need Q4_K or Q6_K)")),
        };
        let final_norm = load_norm(&mut bufs, &mut file, "output_norm.weight")?;
        let n_inter = ct.tensor(&mut file, "blk.0.ffn_gate.weight", &dev).map_err(|e| e.to_string())?.shape().dims()[0];

        // Per-layer weights + KV cache.
        struct RawLayer { an: vk::Buffer, fn_: vk::Buffer, wq: VkWeight, wk: VkWeight, wv: VkWeight, wo: VkWeight, w13: VkWeight, w13_q3: Option<VkWeight>, w2: VkWeight, wqk: Option<vk::Buffer>, kc: vk::Buffer, vc: vk::Buffer }
        let mut raw: Vec<RawLayer> = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            let an = load_norm(&mut bufs, &mut file, &format!("{p}.attn_norm.weight"))?;
            let fn_ = load_norm(&mut bufs, &mut file, &format!("{p}.ffn_norm.weight"))?;
            let wq = load_w(&mut bufs, &mut file, &format!("{p}.attn_q.weight"))?;
            let wk = load_w(&mut bufs, &mut file, &format!("{p}.attn_k.weight"))?;
            let wv = load_w(&mut bufs, &mut file, &format!("{p}.attn_v.weight"))?;
            let wo = load_w(&mut bufs, &mut file, &format!("{p}.attn_output.weight"))?;
            let w13 = load_gateup(&mut bufs, &mut file, &p, false)?; // Q4 (prefill)
            let w13_q3 = if q3_gateup { Some(load_gateup(&mut bufs, &mut file, &p, true)?) } else { None }; // Q3 decode copy
            let w2 = load_w(&mut bufs, &mut file, &format!("{p}.ffn_down.weight"))?;
            let wqk = if fused_qkv { Some(load_wqk(&mut bufs, &mut file, &p)?) } else { None };
            let (kc, _) = vk_zeros(&ctx, &mut bufs, max_seq * kv_dim);
            let (vc, _) = vk_zeros(&ctx, &mut bufs, max_seq * kv_dim);
            raw.push(RawLayer { an, fn_, wq, wk, wv, wo, w13, w13_q3, w2, wqk, kc, vc });
        }

        // RoPE tables (interleaved rope_i, matching candle + rope.comp).
        let mut cos = vec![0f32; max_seq * half];
        let mut sin = vec![0f32; max_seq * half];
        for pos in 0..max_seq {
            for j in 0..half {
                let th = 1.0 / rope_base.powf((2 * j) as f32 / hd as f32);
                cos[pos * half + j] = (pos as f32 * th).cos();
                sin[pos * half + j] = (pos as f32 * th).sin();
            }
        }

        // Per-token scratch + uniforms (persistent; contents updated per token).
        let (x_buf, x_ptr) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (cos_buf, cos_ptr) = vk_zeros(&ctx, &mut bufs, half);
        let (sin_buf, sin_ptr) = vk_zeros(&ctx, &mut bufs, half);
        let (normed, _) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (q, _) = vk_zeros(&ctx, &mut bufs, attn_dim);
        let (k_buf, _) = vk_zeros(&ctx, &mut bufs, kv_dim);
        let (v_buf, _) = vk_zeros(&ctx, &mut bufs, kv_dim);
        let (qk_buf, _) = vk_zeros(&ctx, &mut bufs, attn_dim + kv_dim); // fused QKV: q | k (q,k roped together)
        let (attn, _) = vk_zeros(&ctx, &mut bufs, attn_dim);
        let (_o_buf, _) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (gu, _) = vk_zeros(&ctx, &mut bufs, n_inter * 2); // [gate(n_inter); up(n_inter)]
        let (hbuf, _) = vk_zeros(&ctx, &mut bufs, n_inter);
        let (_ffn_buf, _) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (logits, logits_ptr) = vk_zeros(&ctx, &mut bufs, vocab);
        let nblk_max = max_seq.div_ceil(SDPA_FLASH_BLOCK);
        let (part, _) = vk_zeros(&ctx, &mut bufs, n_head * nblk_max * (hd + 2));
        let n_super_max = nblk_max.div_ceil(SDPA_SUPER);
        let (superp, _) = vk_zeros(&ctx, &mut bufs, n_head * n_super_max * (hd + 2)); // hierarchical-combine scratch
        // Uniforms.
        let (u_norm, _) = vk_uni(&ctx, &mut bufs, [n_embd as u32, eps.to_bits(), 0, 0]);
        let (u_rope_q, _) = vk_uni(&ctx, &mut bufs, [n_head as u32, hd as u32, 0, 0]);
        let (u_rope_k, _) = vk_uni(&ctx, &mut bufs, [n_kv as u32, hd as u32, 0, 0]);
        let (u_base, base_ptr) = vk_uni(&ctx, &mut bufs, [kv_dim as u32, 0, 0, 0]);
        let (u_seq, seq_ptr) = vk_uni(&ctx, &mut bufs, [n_head as u32, n_kv as u32, hd as u32, 1]);
        // Hierarchical combine uniforms {n_head,n_kv,hd,n_in,chunk,final}; n_in/chunk refreshed per token.
        let (u_l1, l1_ptr) = vk_uni8p(&ctx, &mut bufs, [n_head as u32, n_kv as u32, hd as u32, 1, SDPA_SUPER as u32, 0, 0, 0]);
        let (u_l2, l2_ptr) = vk_uni8p(&ctx, &mut bufs, [n_head as u32, n_kv as u32, hd as u32, 1, 1, 1, 0, 0]);
        let (u_silu, _) = vk_uni(&ctx, &mut bufs, [n_inter as u32, 0, 0, 0]);
        let (_u_add, _) = vk_uni(&ctx, &mut bufs, [n_embd as u32, 0, 0, 0]);
        let gxof = |n: usize| (n as u32).min(65535);
        let (u_lm, _) = vk_uni(&ctx, &mut bufs, [vocab as u32, lm_nb as u32, gxof(vocab), 0]);

        // Pipelines.
        let mut pipes: Vec<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout, vk::ShaderModule)> = Vec::new();
        let mut mkpipe = |spv: &[u8], n: u32| -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
            let (p, l, sl, m) = ctx.make_pipeline_raw(spv, n);
            pipes.push((p, l, sl, m)); (p, l, sl)
        };
        let (rms_p, rms_l, rms_sl) = mkpipe(RMSNORM_SPV, 3);
        let (mv_p, mv_l, mv_sl) = mkpipe(DECODE_MATVEC_Q4K_SPV, 3);
        let (q3_p, q3_l, _q3_sl) = mkpipe(DECODE_MATVEC_Q3K_SPV, 3); // Q3_K reuses the 3-storage layout
        let (rope_p, rope_l, rope_sl) = mkpipe(ROPE_SPV, 3);
        let (kvw_p, kvw_l, kvw_sl) = mkpipe(KV_WRITE_SPV, 2);
        let (sdpa_p, sdpa_l, sdpa_sl) = mkpipe(SDPA_DECODE_SPV, 4);
        let (fp_p, fp_l, fp_sl) = mkpipe(SDPA_FLASH_PARTIAL_SPV, 4);
        let (fc_p, fc_l, fc_sl) = mkpipe(SDPA_FLASH_COMBINE_SPV, 2);
        let (ch_p, ch_l, ch_sl) = mkpipe(SDPA_FLASH_COMBINE_H_SPV, 2); // hierarchical combine (both levels)
        let (f2_p, f2_l, _f2_sl) = mkpipe(SDPA_FLASH_PARTIAL2_SPV, 4); // 2-pass partial (compatible layout with fp)
        let (kvwhm_p, kvwhm_l, _kvwhm_sl) = mkpipe(KV_WRITE_HM_SPV, 2);          // head-major KV write (same {dst,src}+uniform layout as kvw)
        let (f2hm_p, f2hm_l, _f2hm_sl) = mkpipe(SDPA_FLASH_PARTIAL_HM_SPV, 4);   // head-major partial (same {q,kc,vc,part}+uniform layout as fp2)
        let (kvtr_p, kvtr_l, kvtr_sl) = mkpipe(KV_TRANSPOSE_HM_SPV, 2);          // one-time pos→head-major cache transpose
        let (silu_p, silu_l, silu_sl) = mkpipe(SILU_MUL_SPV, 3);
        let (add_p, add_l, _add_sl) = mkpipe(RESIDUAL_ADD_SPV, 2);
        let (q6k_p, q6k_l, q6k_sl) = mkpipe(DECODE_MATVEC_Q6K_SPV, 6);
        let (argmax_p, argmax_l, argmax_sl) = mkpipe(ARGMAX_SPV, 2);
        let (mvr_p, mvr_l, mvr_sl) = mkpipe(DECODE_MATVEC_Q4K_ROPE_SPV, 5); // fused QKV+rope (q/k)

        // Descriptor pool + sets.
        let desc_pool = dv.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets((n_layers * 20 + 16) as u32).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count((n_layers * 60 + 64) as u32),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count((n_layers * 20 + 16) as u32),
        ]), None).map_err(|e| format!("desc pool: {e}"))?;
        let mkset = |sl: vk::DescriptorSetLayout, sb: &[vk::Buffer], u: vk::Buffer| -> vk::DescriptorSet {
            let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(std::slice::from_ref(&sl))).unwrap()[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = sb.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
            let uinfo = [vk::DescriptorBufferInfo::default().buffer(u).range(vk::WHOLE_SIZE)];
            let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(sb.len() as u32).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
            dv.update_descriptor_sets(&w, &[]);
            set
        };
        // Shared sets.
        let s_rope_q = mkset(rope_sl, &[q, cos_buf, sin_buf], u_rope_q);
        let s_rope_k = mkset(rope_sl, &[k_buf, cos_buf, sin_buf], u_rope_k);
        let s_final_norm = mkset(rms_sl, &[x_buf, final_norm, normed], u_norm);
        let s_lm = if lm_q4 {
            let (u, _) = vk_uni(&ctx, &mut bufs, [vocab as u32, n_embd as u32, lm_nb as u32, (vocab as u32).min(65535)]);
            mkset(mv_sl, &[lm_ql, normed, logits], u) // Q4 matvec: [weight, x, logits]
        } else {
            mkset(q6k_sl, &[lm_ql, lm_qh, lm_scl, lm_dd, normed, logits], u_lm)
        };
        let (argmax_out, argmax_ptr) = vk_zeros(&ctx, &mut bufs, 1);
        let (u_argmax, _) = vk_uni(&ctx, &mut bufs, [vocab as u32, 0, 0, 0]);
        let s_argmax = mkset(argmax_sl, &[logits, argmax_out], u_argmax);
        // Fused silu: gate = gu[0..n_inter], up = gu[n_inter..2*n_inter] (descriptor offsets).
        let s_silu = {
            let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(std::slice::from_ref(&silu_sl))).unwrap()[0];
            let gi = [vk::DescriptorBufferInfo::default().buffer(gu).offset(0).range((n_inter * 4) as u64)];
            let upi = [vk::DescriptorBufferInfo::default().buffer(gu).offset((n_inter * 4) as u64).range((n_inter * 4) as u64)];
            let hi = [vk::DescriptorBufferInfo::default().buffer(hbuf).range(vk::WHOLE_SIZE)];
            let su = [vk::DescriptorBufferInfo::default().buffer(u_silu).range(vk::WHOLE_SIZE)];
            dv.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&gi),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&upi),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&hi),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&su),
            ], &[]);
            set
        };

        let mut layers = Vec::with_capacity(n_layers);
        for r in &raw {
            let wq = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wq, normed, q, n_embd, false);
            let wk = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wk, normed, k_buf, kv_dim, false);
            let wv = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wv, normed, v_buf, kv_dim, false);
            // wo/w2 fold the residual: write x_buf += proj (no separate radd dispatch).
            let wo = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wo, attn, x_buf, n_embd, true);
            let w13 = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, r.w13_q3.as_ref().unwrap_or(&r.w13), normed, gu, n_inter * 2, false);
            let w2 = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.w2, hbuf, x_buf, n_embd, true);
            let wqb = match &r.wq { VkWeight::Q4 { buf, .. } => *buf, _ => panic!("wq not Q4") };
            let wkb = match &r.wk { VkWeight::Q4 { buf, .. } => *buf, _ => panic!("wk not Q4") };
            let u_wqr = vk_uni8(&ctx, &mut bufs, [n_embd as u32, n_embd as u32, (n_embd / 256) as u32, gxof(n_embd / 2), half as u32, 0, 0, 0]);
            let u_wkr = vk_uni8(&ctx, &mut bufs, [kv_dim as u32, n_embd as u32, (n_embd / 256) as u32, gxof(kv_dim / 2), half as u32, 0, 0, 0]);
            let wq_rope = mkset(mvr_sl, &[wqb, normed, cos_buf, sin_buf, q], u_wqr);
            let wk_rope = mkset(mvr_sl, &[wkb, normed, cos_buf, sin_buf, k_buf], u_wkr);
            // Fused QKV: one mvrope over concat wq+wk → qk_buf (q|k roped together);
            // repoint kvw_k/sdpa/fp to read k/q from qk_buf slices. wv stays separate.
            let wqk_rope = if let Some(wqkb) = r.wqk {
                let u = vk_uni8(&ctx, &mut bufs, [(n_embd + kv_dim) as u32, n_embd as u32, (n_embd / 256) as u32, gxof((n_embd + kv_dim) / 2), half as u32, 0, 0, 0]);
                Some(mkset(mvr_sl, &[wqkb, normed, cos_buf, sin_buf, qk_buf], u))
            } else { None };
            let qsrc = if fused_qkv { qk_buf } else { q };
            let kvw_k = if fused_qkv {
                vk_alloc_set_off(&ctx.device, desc_pool, kvw_sl, &[(r.kc, 0, vk::WHOLE_SIZE), (qk_buf, (n_embd * 4) as u64, (kv_dim * 4) as u64)], u_base)
            } else { mkset(kvw_sl, &[r.kc, k_buf], u_base) };
            layers.push(VkLayerOps {
                attn_norm: mkset(rms_sl, &[x_buf, r.an, normed], u_norm),
                wq, wk, wv, wq_rope, wk_rope, wqk_rope,
                kvw_k,
                kvw_v: mkset(kvw_sl, &[r.vc, v_buf], u_base),
                sdpa: mkset(sdpa_sl, &[qsrc, r.kc, r.vc, attn], u_seq),
                fp: mkset(fp_sl, &[qsrc, r.kc, r.vc, part], u_seq),
                fc: mkset(fc_sl, &[part, attn], u_seq),
                fc1: mkset(ch_sl, &[part, superp], u_l1),   // L1: block-partials -> super-partials
                fc2: mkset(ch_sl, &[superp, attn], u_l2),   // L2: super-partials -> attn (normalized)
                wo,
                ffn_norm: mkset(rms_sl, &[x_buf, r.fn_, normed], u_norm),
                w13, w2,
            });
        }

        // --- Batched prefill resources (option b: Q6 wv/ffn_down dequant -> f16 dense GEMM) ---
        let prefill = {
        let q4buf = |w: &VkWeight| -> vk::Buffer { match *w { VkWeight::Q4 { buf, .. } => buf, _ => panic!("prefill: expected Q4 weight") } };
        let mut pw: Vec<PrefillW> = Vec::with_capacity(n_layers);
        for (i, r) in raw.iter().enumerate() {
            let lp = format!("blk.{i}");
            let wv_deq: Vec<f32> = ct.tensor(&mut file, &format!("{lp}.attn_v.weight"), &dev).map_err(|e| e.to_string())?.dequantize(&dev).map_err(|e| e.to_string())?.flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
            let w2_deq: Vec<f32> = ct.tensor(&mut file, &format!("{lp}.ffn_down.weight"), &dev).map_err(|e| e.to_string())?.dequantize(&dev).map_err(|e| e.to_string())?.flatten_all().map_err(|e| e.to_string())?.to_vec1().map_err(|e| e.to_string())?;
            let wv_f16 = vk_up_f16t(&ctx, &mut bufs, &wv_deq, kv_dim, n_embd);
            let w2_f16 = vk_up_f16t(&ctx, &mut bufs, &w2_deq, n_embd, n_inter);
            pw.push(PrefillW { wq: q4buf(&r.wq), wk: q4buf(&r.wk), wqk: r.wqk, wo: q4buf(&r.wo), w13: q4buf(&r.w13), wv_f16, w2_f16, attn_norm: r.an, ffn_norm: r.fn_, kc: r.kc, vc: r.vc });
        }
        let mut mkp = |spv: &[u8], n: u32, coop: bool| -> Pipe3 {
            let (p, l, sl, m) = if coop { ctx.make_pipeline_coopmat(spv, n) } else { ctx.make_pipeline_raw(spv, n) };
            pipes.push((p, l, sl, m)); (p, l, sl)
        };
        let p_q4 = mkp(COOPMAT_Q4K_GEMM_SPV, 3, true);
        let p_f16 = mkp(COOPMAT_GEMM_SPV, 3, true);
        let p_bn = mkp(BNORM_SPV, 3, false);
        let p_br = mkp(BROPE_SPV, 3, false);
        let p_bs = mkp(BSDPA_SPV, 4, false);
        let p_bsd = mkp(BSDPA_DECODE_SPV, 4, false);
        let p_bso = mkp(BSDPA_OFFSET_SPV, 4, false); // workgroup-per-(query,head) offset SDPA for chunked prefill
        let p_bsi = mkp(BSILU_SPV, 2, false);
        let p_t16 = mkp(TO_F16_SPV, 2, false);
        let p_add = mkp(RESIDUAL_ADD_SPV, 2, false);
        let p_q6k = mkp(DECODE_MATVEC_Q6K_SPV, 6, false);
        let p_bmq4 = mkp(BMV_Q4K_SPV, 3, false);
        let p_skinny = mkp(SKINNY_GEMM_Q4K_SPV, 3, false);
        let p_bmf16 = mkp(BMV_F16_SPV, 3, false);
        let p_bmq6 = mkp(BMV_Q6K_SPV, 6, false);
        let mm = PREFILL_MAX_M;
        let f16buf = |bufs: &mut Vec<(vk::Buffer, vk::DeviceMemory)>, len: usize| -> vk::Buffer { let (b, m, p) = ctx.uma_buffer((len * 2) as u64).unwrap(); std::ptr::write_bytes(p, 0, len * 2); bufs.push((b, m)); b };
        let (pf_x32, pf_x32_ptr) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let pf_x16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_n32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let pf_n16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_q, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let (pf_qk, _) = vk_zeros(&ctx, &mut bufs, mm * (n_embd + kv_dim)); // fused-QKV GEMM out (interleaved q|k)
        let (pf_attn32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let pf_attn16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_gu, _) = vk_zeros(&ctx, &mut bufs, mm * n_inter * 2);
        let (pf_h32, _) = vk_zeros(&ctx, &mut bufs, mm * n_inter); let pf_h16 = f16buf(&mut bufs, mm * n_inter);
        let (pf_o32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let (pf_ffn32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let (pf_cosb, pf_cosb_ptr) = vk_zeros(&ctx, &mut bufs, mm * half);
        let (pf_sinb, pf_sinb_ptr) = vk_zeros(&ctx, &mut bufs, mm * half);
        let (pf_logits, pf_logits_ptr) = vk_zeros(&ctx, &mut bufs, vocab);
        let (pf_vlogits, _) = vk_zeros(&ctx, &mut bufs, VERIFY_MAX_M * vocab);
        // Coopmat attention scratch: f16 Q/K/V views + per-8-head-batch S (f32)
        // and probs (f16). ZB=8 keeps the S/prob working set at 48 MB — mostly
        // L2-resident between the QK → softmax → PV phases. (An all-32-heads
        // variant with one dispatch per phase was MEASURED SLOWER, 306 vs 289 ms:
        // 384 MB of scratch traffic per layer stops caching.)
        let p_cma = mkp(COOPMAT_ATTN_GEMM_SPV, 3, true);
        let p_cma64 = mkp(COOPMAT_ATTN_GEMM_N64_SPV, 3, true);
        let p_smax = mkp(CAUSAL_SOFTMAX_SPV, 2, false);
        let p_cfa = mkp(COOPMAT_FLASH_ATTN_SPV, 4, true); // fused flash attention (VK_FUSED_FA)
        let pf_q16 = f16buf(&mut bufs, mm * attn_dim);
        // K/V f16 views sized for the WHOLE resident prefix (max_seq rows), not
        // one tile: chunked prefill's fused attention reads keys 0..pos+m.
        let pf_k16 = f16buf(&mut bufs, max_seq * kv_dim);
        let pf_v16 = f16buf(&mut bufs, max_seq * kv_dim);
        let (pf_s32, _) = vk_zeros(&ctx, &mut bufs, 8 * mm * mm);
        let pf_prob16 = f16buf(&mut bufs, 8 * mm * mm);
        PrefillRes {
            w: pw, p_q4, p_f16, p_bn, p_br, p_bs, p_bsi, p_t16, p_add, p_q6k, p_bsd, p_bso, p_bmq4, p_skinny, p_bmf16, p_bmq6,
            vlogits: pf_vlogits,
            lm_ql, lm_qh, lm_scl, lm_dd, final_norm, logits: pf_logits, logits_ptr: pf_logits_ptr,
            x32: pf_x32, x32_ptr: pf_x32_ptr, x16: pf_x16, n32: pf_n32, n16: pf_n16, q: pf_q, qk: pf_qk,
            attn32: pf_attn32, attn16: pf_attn16, gu: pf_gu, h32: pf_h32, h16: pf_h16, o32: pf_o32, ffn32: pf_ffn32,
            cosb: pf_cosb, cosb_ptr: pf_cosb_ptr, sinb: pf_sinb, sinb_ptr: pf_sinb_ptr,
            p_cma, p_cma64, p_smax, p_cfa,
            q16: pf_q16, k16: pf_k16, v16: pf_v16, s32: pf_s32, prob16: pf_prob16,
        }
        };

        let cmd_pool = dv.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("cmd pool: {e}"))?;
        let cmds2 = dv.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(2)).map_err(|e| format!("cmd buf: {e}"))?;
        let cmd = cmds2[0];
        let verify_cmd = cmds2[1];
        let fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        // Reusable verify_forward scratch (so it doesn't allocate per call).
        let mut verify_unis: Vec<(vk::Buffer, *mut u8)> = Vec::with_capacity(384);
        for _ in 0..384 { let (b, m, p) = ctx.uma_buffer(32).map_err(|e| format!("verify uni: {e}"))?; bufs.push((b, m)); verify_unis.push((b, p)); }
        let verify_argmax = { let (b, m, p) = ctx.uma_buffer((PREFILL_MAX_M * 4) as u64).map_err(|e| format!("verify argmax: {e}"))?; bufs.push((b, m)); (b, p) };
        let verify_pool = dv.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(640).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3200),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(640),
        ]), None).map_err(|e| format!("verify pool: {e}"))?;
        let verify_fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("verify fence: {e}"))?;

        Ok(Self {
            n_embd, n_head, n_kv, hd, n_inter, vocab, n_layers, kv_dim, half, max_seq, eps,
            headmajor: std::env::var("ZLLM_HEADMAJOR_KV").is_ok(),
            embed, cos, sin, lm_nb,
            x_ptr, cos_ptr, sin_ptr, base_ptr, seq_ptr, logits_ptr, l1_ptr, l2_ptr,
            layers, s_rope_q, s_rope_k, s_silu, s_final_norm, s_lm, s_argmax, argmax_ptr, lm_q4,
            p_rms: (rms_p, rms_l), p_mv: (mv_p, mv_l), p_q3: (q3_p, q3_l), p_rope: (rope_p, rope_l), p_kvw: (kvw_p, kvw_l),
            p_sdpa: (sdpa_p, sdpa_l), p_fp: (fp_p, fp_l), p_fc: (fc_p, fc_l), p_silu: (silu_p, silu_l),
            p_ch: (ch_p, ch_l), p_fp2: (f2_p, f2_l),
            p_kvw_hm: (kvwhm_p, kvwhm_l), p_fp2_hm: (f2hm_p, f2hm_l), p_kvtr: (kvtr_p, kvtr_l, kvtr_sl),
            p_add: (add_p, add_l), p_q6k: (q6k_p, q6k_l), p_argmax: (argmax_p, argmax_l, argmax_sl), p_mvrope: (mvr_p, mvr_l),
            last_rec: std::cell::Cell::new(-1),
            verify_unis, verify_argmax, verify_pool, verify_cmd, verify_fence,
            prefill_rec: std::cell::RefCell::new(None),
            prompt_cache: std::cell::RefCell::new(Vec::new()),
            bufs, pipes, desc_pool, cmd_pool, cmd, fence, prefill, ctx,
        })
    }

    /// Decode one token at position `pos` (0-based). Writes K/V into the resident
    /// cache at `pos` and attends over 0..=pos. Returns the full logits.
    pub fn forward(&self, token: u32, pos: usize) -> Vec<f32> {
        unsafe { self.forward_inner(token, pos, true, false).0 }
    }

    /// Greedy decode: argmax on the GPU, reads back 4 bytes (avoids the slow
    /// 513 KB logit readback from write-combined host-visible memory).
    pub fn forward_argmax(&self, token: u32, pos: usize) -> u32 {
        unsafe { self.forward_inner(token, pos, true, true).1 }
    }

    /// Prefill step: run the layers to fill the KV cache at `pos`, skipping the
    /// final norm + LM head (logits not needed for non-final prompt tokens).
    pub fn prefill_step(&self, token: u32, pos: usize) {
        unsafe { self.forward_inner(token, pos, false, false); }
    }

    /// Batched prefill: process the whole prompt at once through the coopmat
    /// GEMMs, filling the resident KV cache for positions 0..prompt.len().
    /// Returns the last token's logits; decode continues from pos=prompt.len().
    /// M is padded to PREFILL_MAX_M (128) — padding rows are zero and never read.
    /// Batched prefill over the whole prompt. Prompts longer than one coopmat
    /// tile (`PREFILL_MAX_M`) run as a sequence of tile-sized CHUNKS at position
    /// offsets: each chunk writes its K/V into the resident cache at
    /// `pos*kv_dim` and its self-attention attends `[0..pos+row]` via the
    /// batched-decode SDPA (the same offset machinery spec-decode verification
    /// uses). Returns the last real token's logits (from the final chunk).
    pub fn prefill_forward(&self, prompt: &[u32]) -> Vec<f32> {
        let logits = self.prefill_from(prompt, 0);
        // Head-major: the batched prefill wrote the prompt's K/V pos-major (its own
        // prompt self-attention reads pos-major); decode reads head-major, so convert
        // the cache once here (after the LAST chunk — the full prefix at once).
        // Decode-fill paths (forward_argmax) are already head-major.
        if self.headmajor { unsafe { self.transpose_kv_headmajor(prompt.len().min(self.max_seq)); } }
        logits
    }

    /// Batched prefill of `tokens` at positions `start..start+len` against the
    /// resident cache (filled through `start`). Tile-sized chunks; returns the
    /// last token's logits. Pos-major layout only (no head-major transpose —
    /// `prefill_forward` owns that for whole-prompt fills).
    pub fn prefill_from(&self, tokens: &[u32], start: usize) -> Vec<f32> {
        let mut logits = Vec::new();
        let mut pos = 0usize;
        while pos < tokens.len() {
            let end = (pos + PREFILL_MAX_M).min(tokens.len());
            logits = unsafe { self.prefill_inner(&tokens[pos..end], start + pos) };
            pos = end;
        }
        logits
    }

    /// Prefill with CROSS-REQUEST PREFIX REUSE: K/V for the longest common
    /// prefix with the previous request's resident tokens is kept (VkModel never
    /// clears its cache); only the suffix is prefilled — sequential for short
    /// suffixes (padded-tile overhead dominates), chunked-at-offset otherwise.
    /// Always re-runs at least the last prompt token so fresh logits come back.
    /// Head-major mode falls back to a full prefill (layout mixing).
    pub fn prefill_cached(&self, prompt: &[u32]) -> Vec<f32> {
        assert!(!prompt.is_empty() && prompt.len() <= self.prefill_cap());
        if self.headmajor {
            self.prompt_cache.borrow_mut().clear();
            return self.prefill_forward(prompt);
        }
        let lcp = {
            let c = self.prompt_cache.borrow();
            prompt.iter().zip(c.iter()).take_while(|(a, b)| a == b).count()
        };
        let reuse = lcp.min(prompt.len() - 1);
        let suffix = &prompt[reuse..];
        let logits = if reuse == 0 && suffix.len() > 32 {
            self.prefill_from(prompt, 0)
        } else if suffix.len() > 32 {
            crate::metrics::prefix_cache_hits().inc();
            crate::metrics::prefix_cache_tokens_saved().inc_by(reuse as u64);
            self.prefill_from(suffix, reuse)
        } else {
            // Short suffix (or short cold prompt): sequential steps, logits from
            // the last token's full forward.
            if reuse > 0 {
                crate::metrics::prefix_cache_hits().inc();
                crate::metrics::prefix_cache_tokens_saved().inc_by(reuse as u64);
            } else {
                crate::metrics::prefix_cache_misses().inc();
            }
            for (i, &t) in suffix[..suffix.len() - 1].iter().enumerate() {
                self.prefill_step(t, reuse + i);
            }
            self.forward(prompt[prompt.len() - 1], prompt.len() - 1)
        };
        *self.prompt_cache.borrow_mut() = prompt.to_vec();
        logits
    }

    /// Record tokens decoded AFTER a `prefill_cached` prompt whose K/V landed in
    /// the cache, extending the reusable prefix (chat turns append the previous
    /// reply, so the next request's LCP runs through it). The caller passes only
    /// tokens that were actually forwarded (their KV is resident).
    pub fn note_decoded(&self, tokens: &[u32]) {
        if !self.headmajor {
            self.prompt_cache.borrow_mut().extend_from_slice(tokens);
        }
    }

    /// One-time pos-major→head-major transpose of every layer's KV cache, at the
    /// prefill→decode boundary (ZLLM_HEADMAJOR_KV). Copies each cache to a temp then
    /// scatters head-major back. Own command buffer — out of the recorded prefill.
    unsafe fn transpose_kv_headmajor(&self, seq_len: usize) {
        use ash::vk;
        if seq_len == 0 { return; }
        let dv = &self.ctx.device;
        let n = (seq_len * self.kv_dim) as u32;
        let sz = (n as u64) * 4;
        let (temp, tmem, _tp) = self.ctx.uma_buffer(sz).unwrap();
        let (ub, umem, up) = self.ctx.uma_buffer(16).unwrap();
        std::ptr::copy_nonoverlapping([n, self.hd as u32, self.kv_dim as u32, 0u32].as_ptr() as *const u8, up, 16); // slot1=hd, slot2=kv_dim
        let nsets = (self.n_layers * 2) as u32;
        let pool = dv.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(nsets).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(nsets * 2),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(nsets)]), None).unwrap();
        let cp = dv.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.ctx.queue_family), None).unwrap();
        let cmd = dv.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cp).command_buffer_count(1)).unwrap()[0];
        dv.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        let cs = vk::PipelineStageFlags::COMPUTE_SHADER; let tr = vk::PipelineStageFlags::TRANSFER;
        dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.p_kvtr.0);
        for l in &self.prefill.w {
            for cache in [l.kc, l.vc] {
                dv.cmd_copy_buffer(cmd, cache, temp, &[vk::BufferCopy::default().size(sz)]); // pos-major copy
                let b1 = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                dv.cmd_pipeline_barrier(cmd, tr, cs, vk::DependencyFlags::empty(), &[b1], &[], &[]);
                let set = vk_alloc_set(dv, pool, self.p_kvtr.2, &[cache, temp], ub); // dst=cache, src=temp
                dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.p_kvtr.1, 0, &[set], &[]);
                dv.cmd_dispatch(cmd, n.div_ceil(64), 1, 1); // scatter head-major into the cache
                let b2 = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                dv.cmd_pipeline_barrier(cmd, cs, tr, vk::DependencyFlags::empty(), &[b2], &[], &[]); // before temp is reused
            }
        }
        dv.end_command_buffer(cmd).unwrap();
        let fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        dv.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
        dv.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
        dv.destroy_fence(fence, None); dv.destroy_command_pool(cp, None); dv.destroy_descriptor_pool(pool, None);
        for (b, m) in [(temp, tmem), (ub, umem)] { dv.unmap_memory(m); dv.destroy_buffer(b, None); dv.free_memory(m, None); }
    }

    /// Longest prompt `prefill_forward` accepts. Prompts beyond one coopmat tile
    /// (`PREFILL_MAX_M`) run as CHUNKED batched prefill, so the bound is the
    /// resident-cache capacity minus the last chunk's 128-row padding headroom.
    pub fn prefill_cap(&self) -> usize { self.max_seq - 128 }

    /// One prefill chunk of ≤ PREFILL_MAX_M tokens at position offset `pos`
    /// (`pos == 0` = the classic whole-prompt tile; `pos > 0` = a continuation
    /// chunk against the resident cache). At `pos > 0` the K/V cache writes are
    /// bound at `pos*kv_dim` and the SDPA switches from the within-tile causal
    /// kernel to the batched-decode kernel (query row i at `pos+i` attends the
    /// cache `[0..=pos+i]` — same shader spec-decode verification uses).
    unsafe fn prefill_inner(&self, prompt: &[u32], pos: usize) -> Vec<f32> {
        use ash::vk;
        let dv = &self.ctx.device;
        let pf = &self.prefill;
        let (n_embd, n_head, n_kv, hd, n_inter, kv_dim, half, vocab) =
            (self.n_embd, self.n_head, self.n_kv, self.hd, self.n_inter, self.kv_dim, self.half, self.vocab);
        let lm_nb = self.lm_nb;
        let real_m = prompt.len().min(PREFILL_MAX_M);
        // Pad to the next 128-multiple (cap PREFILL_MAX_M), NOT always to the max: short prompts
        // stay M=128 (no padding waste), longer ones get the GEMM-efficient larger tile.
        let m = (real_m.div_ceil(128) * 128).min(PREFILL_MAX_M);
        assert!(pos + m <= self.max_seq, "prefill chunk (pos {pos} + padded m {m}) exceeds max_seq {}", self.max_seq);

        // Inputs: x = embeddings (padding rows zero), cos/sin for positions pos..pos+m.
        std::ptr::write_bytes(pf.x32_ptr, 0, m * n_embd * 4);
        for (i, &tk) in prompt.iter().take(real_m).enumerate() {
            std::ptr::copy_nonoverlapping(self.embed[tk as usize * n_embd..].as_ptr() as *const u8, pf.x32_ptr.add(i * n_embd * 4), n_embd * 4);
        }
        std::ptr::copy_nonoverlapping(self.cos[pos * half..(pos + m) * half].as_ptr() as *const u8, pf.cosb_ptr, m * half * 4);
        std::ptr::copy_nonoverlapping(self.sin[pos * half..(pos + m) * half].as_ptr() as *const u8, pf.sinb_ptr, m * half * 4);

        // Record the 128-row command ONCE per (real_m, pos) — only the mapped
        // embeddings above change per call — then reuse it, killing per-call
        // alloc+record+cleanup (60%). Offset chunks of a long prompt each rebuild
        // (different pos baked into uniforms/regions); GPU exec dominates there.
        let kv_off = (pos * kv_dim * 4) as u64;                  // byte offset of this chunk's KV rows
        let kv_rng = (m * kv_dim * 4) as u64;
        let need_build = self.prefill_rec.borrow().as_ref().map_or(true, |(key, _)| *key != (real_m, pos));
        if need_build {
        if let Some((_, old)) = self.prefill_rec.borrow_mut().take() { old.destroy(dv); }
        let ub = std::cell::RefCell::new(Vec::<(vk::Buffer, vk::DeviceMemory)>::new());
        let uni = |d: [u32; 4]| -> vk::Buffer {
            let (b, mm, p) = self.ctx.uma_buffer(16).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16); ub.borrow_mut().push((b, mm)); b
        };
        let uni5 = |d: [u32; 5]| -> vk::Buffer {
            let (b, mm, p) = self.ctx.uma_buffer(20).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 20); ub.borrow_mut().push((b, mm)); b
        };
        // Coopmat-attention GEMM params (15 fields, 60 B — see coopmat_attn_gemm.comp).
        let uni15 = |d: [u32; 15]| -> vk::Buffer {
            let (b, mm, p) = self.ctx.uma_buffer(64).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 60); ub.borrow_mut().push((b, mm)); b
        };
        let pool = dv.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets((self.n_layers * 40 + 8) as u32).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count((self.n_layers * 160 + 16) as u32),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count((self.n_layers * 40 + 8) as u32),
        ]), None).unwrap();
        let mkset = |sl: vk::DescriptorSetLayout, sb: &[vk::Buffer], u: vk::Buffer| vk_alloc_set(dv, pool, sl, sb, u);
        // Offset variant for chunked prefill: binds the K/V cache at this chunk's
        // byte offset so the tile's GEMM/rope writes land at rows pos..pos+m.
        // (kv_off is kv_dim*4-aligned = 2 KB for the 1B — satisfies any
        // minStorageBufferOffsetAlignment.)
        let mkset_off = |sl: vk::DescriptorSetLayout, sb: &[(vk::Buffer, u64, u64)], u: vk::Buffer| vk_alloc_set_off(dv, pool, sl, sb, u);
        let cmd_pool = dv.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.ctx.queue_family), None).unwrap();
        let cmd = dv.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dv.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        let bar = || dv.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        let tr = vk::PipelineStageFlags::TRANSFER;
        let bar_c2t = || { let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ); dv.cmd_pipeline_barrier(cmd, cs, tr, vk::DependencyFlags::empty(), &[b], &[], &[]); };
        let disp = |p: Pipe3, set: vk::DescriptorSet, gx: u32, gy: u32| {
            dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.0);
            dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.1, 0, &[set], &[]);
            dv.cmd_dispatch(cmd, gx, gy, 1);
        };
        // 3D variant for the head-batched attention GEMMs (gz = heads per batch).
        let disp3 = |p: Pipe3, set: vk::DescriptorSet, gx: u32, gy: u32, gz: u32| {
            dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.0);
            dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.1, 0, &[set], &[]);
            dv.cmd_dispatch(cmd, gx, gy, gz);
        };
        let c64 = |x: usize| ((x + 63) / 64) as u32;
        let u_eps = uni([n_embd as u32, self.eps.to_bits(), 0, 0]);
        // Prefill overhead decomposition (timing only; output is garbage when set). Skip a
        // category's dispatches (barriers kept) and diff vs full to attribute the ~64ms overhead.
        let skip_f16 = std::env::var("VK_PF_NOF16").is_ok();   // the 4 to_f16 passes/layer
        let skip_sdpa = std::env::var("VK_PF_NOSDPA").is_ok(); // the O(P^2) causal SDPA
        let skip_gemm = std::env::var("VK_PF_NOGEMM").is_ok(); // the 6 Q4/f16 GEMMs/layer

        for l in &pf.w {
            // attn rmsnorm -> n32 -> f16
            disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.attn_norm, pf.n32], u_eps), m as u32, 1); bar();
            if !skip_f16 { disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.n32, pf.n16], uni([(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); } bar();
            // QKV: wq->q (Q4), wk->kc (Q4), wv->vc (f16 dense). K=n_embd.
            // Fused (ZLLM_FUSED_QKV): one [m, n_embd+kv_dim] coopmat GEMM over concat
            // wq+wk → pf.qk (interleaved q|k per token), then de-interleave q->pf.q,
            // k->l.kc (the GEMM output is token-major; cache wants contiguous k).
            if !skip_gemm {
            if let Some(wqk) = l.wqk {
                let nqk = n_embd + kv_dim;
                disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, wqk, pf.qk], uni([m as u32, nqk as u32, n_embd as u32, (n_embd / 256) as u32])), (nqk / 128) as u32, (m / 128) as u32);
                disp(pf.p_f16, mkset_off(pf.p_f16.2, &[(pf.n16, 0, vk::WHOLE_SIZE), (l.wv_f16, 0, vk::WHOLE_SIZE), (l.vc, kv_off, kv_rng)], uni([m as u32, kv_dim as u32, n_embd as u32, 0])), (kv_dim / 64) as u32, (m / 64) as u32);
                bar_c2t();
                let qreg: Vec<vk::BufferCopy> = (0..m).map(|t| vk::BufferCopy::default().src_offset((t * nqk * 4) as u64).dst_offset((t * n_embd * 4) as u64).size((n_embd * 4) as u64)).collect();
                let kreg: Vec<vk::BufferCopy> = (0..m).map(|t| vk::BufferCopy::default().src_offset(((t * nqk + n_embd) * 4) as u64).dst_offset(kv_off + (t * kv_dim * 4) as u64).size((kv_dim * 4) as u64)).collect();
                dv.cmd_copy_buffer(cmd, pf.qk, pf.q, &qreg);
                dv.cmd_copy_buffer(cmd, pf.qk, l.kc, &kreg);
                // Final: both the de-interleave (transfer) AND wv (compute write l.vc) → rope/SDPA reads.
                let bc = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                dv.cmd_pipeline_barrier(cmd, cs | tr, cs, vk::DependencyFlags::empty(), &[bc], &[], &[]);
            } else {
            disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, l.wq, pf.q], uni([m as u32, n_embd as u32, n_embd as u32, (n_embd / 256) as u32])), (n_embd / 128) as u32, (m / 128) as u32);
            disp(pf.p_q4, mkset_off(pf.p_q4.2, &[(pf.n16, 0, vk::WHOLE_SIZE), (l.wk, 0, vk::WHOLE_SIZE), (l.kc, kv_off, kv_rng)], uni([m as u32, kv_dim as u32, n_embd as u32, (n_embd / 256) as u32])), (kv_dim / 128) as u32, (m / 128) as u32);
            disp(pf.p_f16, mkset_off(pf.p_f16.2, &[(pf.n16, 0, vk::WHOLE_SIZE), (l.wv_f16, 0, vk::WHOLE_SIZE), (l.vc, kv_off, kv_rng)], uni([m as u32, kv_dim as u32, n_embd as u32, 0])), (kv_dim / 64) as u32, (m / 64) as u32); bar();
            } } else { bar(); }
            // RoPE q, k (k in the cache, at this chunk's offset rows).
            disp(pf.p_br, mkset(pf.p_br.2, &[pf.q, pf.cosb, pf.sinb], uni([n_head as u32, hd as u32, m as u32, 0])), c64(m * n_head * half), 1);
            disp(pf.p_br, mkset_off(pf.p_br.2, &[(l.kc, kv_off, kv_rng), (pf.cosb, 0, vk::WHOLE_SIZE), (pf.sinb, 0, vk::WHOLE_SIZE)], uni([n_kv as u32, hd as u32, m as u32, 0])), c64(m * n_kv * half), 1); bar();
            // Causal SDPA -> attn. pos==0: within-tile causal kernel (unchanged
            // fast path). pos>0 (continuation chunk): offset kernel — query row i
            // (position pos+i) attends the resident cache [0..=pos+i]. One
            // workgroup per (query,head) (p_bso; the thread-per-query p_bsd
            // measured 387 tok/s chunked prefill — 64x less parallel).
            if !skip_sdpa {
                let scalar_sdpa = std::env::var("VK_SCALAR_SDPA").is_ok(); // A/B: old scalar kernels
                let fa3 = std::env::var("VK_FA3").is_ok();                 // A/B: 3-phase coopmat (pos==0 only)
                if !scalar_sdpa && !fa3 {
                    // FUSED coopmat flash attention (default, ANY pos): S never
                    // leaves LDS (online softmax + O accumulation per 64-row query
                    // block); query rows sit at positions pos..pos+m and attend the
                    // resident keys 0..pos+row. f16 views: Q = this chunk's m rows,
                    // K/V = the WHOLE prefix 0..seq. Measured at 1024 tok: 262 ms
                    // vs 295 (VK_FA3=1) vs 341 (VK_SCALAR_SDPA=1).
                    let seq = pos + m;
                    let attn_dim = n_head * hd;
                    let gqa = (n_head / n_kv) as u32;
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.q, pf.q16], uni([(m * attn_dim) as u32, 0, 0, 0])), c64(m * attn_dim), 1);
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[l.kc, pf.k16], uni([(seq * kv_dim) as u32, 0, 0, 0])), c64(seq * kv_dim), 1);
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[l.vc, pf.v16], uni([(seq * kv_dim) as u32, 0, 0, 0])), c64(seq * kv_dim), 1);
                    bar();
                    disp3(pf.p_cfa, mkset(pf.p_cfa.2, &[pf.q16, pf.k16, pf.v16, pf.attn32],
                        uni15([m as u32, seq as u32, pos as u32, gqa, attn_dim as u32, kv_dim as u32,
                               attn_dim as u32, 0, scale.to_bits(), 0, 0, 0, 0, 0, 0])),
                        (m / 64) as u32, n_head as u32, 1);
                    bar();
                } else if pos == 0 && !scalar_sdpa {
                    // 3-PHASE coopmat attention (VK_FA3=1 A/B; pos==0 only — its S
                    // scratch is sized for one tile): QK GEMM → causal softmax →
                    // PV GEMM through global S/prob planes.
                    let seq = m;
                    let attn_dim = n_head * hd;
                    let gqa = (n_head / n_kv) as u32;
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.q, pf.q16], uni([(m * attn_dim) as u32, 0, 0, 0])), c64(m * attn_dim), 1);
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[l.kc, pf.k16], uni([(m * kv_dim) as u32, 0, 0, 0])), c64(m * kv_dim), 1);
                    disp(pf.p_t16, mkset(pf.p_t16.2, &[l.vc, pf.v16], uni([(m * kv_dim) as u32, 0, 0, 0])), c64(m * kv_dim), 1);
                    bar();
                    {
                    // 8-head batches: QK → causal softmax → PV per batch. The batch
                    // barriers cost less than losing L2 residency on the S planes
                    // (all-heads-at-once measured slower — see the scratch comment).
                    for zb in (0..n_head).step_by(8) {
                        // S[z][m][seq] = scale * Q_h[m,hd] · K_h[seq,hd]^T  (A global-head, C batch-local)
                        disp3(pf.p_cma, mkset(pf.p_cma.2, &[pf.q16, pf.k16, pf.s32],
                            uni15([m as u32, seq as u32, hd as u32, attn_dim as u32, kv_dim as u32, seq as u32,
                                   hd as u32, hd as u32, (m * seq) as u32, zb as u32, gqa, 0, 1, 0, scale.to_bits()])),
                            (seq / 128) as u32, (m / 128) as u32, 8);
                        bar();
                        disp3(pf.p_smax, mkset(pf.p_smax.2, &[pf.s32, pf.prob16], uni([m as u32, seq as u32, pos as u32, 0])), m as u32, 8, 1);
                        bar();
                        // O_h[m,hd] = P[m,seq] · V_h[seq,hd]  (A batch-local, C global-head, B=V direct)
                        disp3(pf.p_cma64, mkset(pf.p_cma64.2, &[pf.prob16, pf.v16, pf.attn32],
                            uni15([m as u32, hd as u32, seq as u32, seq as u32, kv_dim as u32, attn_dim as u32,
                                   (m * seq) as u32, hd as u32, hd as u32, zb as u32, gqa, 1, 0, 1, 1.0f32.to_bits()])),
                            (hd / 64) as u32, (m / 128) as u32, 8);
                        bar();
                    }
                    }
                } else if pos == 0 {
                    disp(pf.p_bs, mkset(pf.p_bs.2, &[pf.q, l.kc, l.vc, pf.attn32], uni([n_head as u32, n_kv as u32, hd as u32, m as u32])), (m * n_head) as u32, 1); // workgroup per (query,head)
                } else {
                    disp(pf.p_bso, mkset(pf.p_bso.2, &[pf.q, l.kc, l.vc, pf.attn32], uni5([n_head as u32, n_kv as u32, hd as u32, m as u32, pos as u32])), (m * n_head) as u32, 1); // workgroup per (query,head)
                }
            } bar();
            // Wo: attn->f16, GEMM->o32, residual x += o32
            if !skip_f16 { disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.attn32, pf.attn16], uni([(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); } bar();
            if !skip_gemm { disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.attn16, l.wo, pf.o32], uni([m as u32, n_embd as u32, n_embd as u32, (n_embd / 256) as u32])), (n_embd / 128) as u32, (m / 128) as u32); } bar();
            disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.o32], uni([(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
            // ffn rmsnorm -> n32 -> f16
            disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.ffn_norm, pf.n32], u_eps), m as u32, 1); bar();
            if !skip_f16 { disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.n32, pf.n16], uni([(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); } bar();
            // W13 GEMM -> gu [M, 2*n_inter] (Q4), silu -> h32 -> f16
            if !skip_gemm { disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, l.w13, pf.gu], uni([m as u32, (2 * n_inter) as u32, n_embd as u32, (n_embd / 256) as u32])), ((2 * n_inter) / 128) as u32, (m / 128) as u32); } bar();
            disp(pf.p_bsi, mkset(pf.p_bsi.2, &[pf.gu, pf.h32], uni([n_inter as u32, m as u32, 0, 0])), c64(m * n_inter), 1); bar();
            if !skip_f16 { disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.h32, pf.h16], uni([(m * n_inter) as u32, 0, 0, 0])), c64(m * n_inter), 1); } bar();
            // W2 GEMM (f16 dense, K=n_inter) -> ffn32, residual x += ffn32
            if !skip_gemm { disp(pf.p_f16, mkset(pf.p_f16.2, &[pf.h16, l.w2_f16, pf.ffn32], uni([m as u32, n_embd as u32, n_inter as u32, 0])), (n_embd / 64) as u32, (m / 64) as u32); } bar();
            disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.ffn32], uni([(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
        }
        // final rmsnorm -> n32; LM head on the last real token's row. Q4-LM-head (all-Q4 model)
        // must use the Q4 matvec, NOT Q6 — reading the Q4 weight as Q6 gives garbage logits.
        disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, pf.final_norm, pf.n32], u_eps), m as u32, 1); bar();
        let off = ((real_m - 1) * n_embd * 4) as u64;
        let gx = (vocab as u32).min(65535);
        if self.lm_q4 {
            // Q4 LM head: batched matvec, M=1 on the last token's normed row. p_bmq4 bindings: WQ, X, out.
            let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&pf.p_bmq4.2))).unwrap()[0];
            let infos = [
                (0u32, [vk::DescriptorBufferInfo::default().buffer(pf.lm_ql).range(vk::WHOLE_SIZE)]),
                (1, [vk::DescriptorBufferInfo::default().buffer(pf.n32).offset(off).range((n_embd * 4) as u64)]),
                (2, [vk::DescriptorBufferInfo::default().buffer(pf.logits).range(vk::WHOLE_SIZE)]),
            ];
            let uq4 = uni5([1, vocab as u32, n_embd as u32, lm_nb as u32, gx]);
            let uinfo = [vk::DescriptorBufferInfo::default().buffer(uq4).range(vk::WHOLE_SIZE)];
            let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
            dv.update_descriptor_sets(&w, &[]);
            disp(pf.p_bmq4, set, gx, (vocab as u32).div_ceil(gx));
        } else {
            let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&pf.p_q6k.2))).unwrap()[0];
            let infos = [
                (0u32, [vk::DescriptorBufferInfo::default().buffer(pf.lm_ql).range(vk::WHOLE_SIZE)]),
                (1, [vk::DescriptorBufferInfo::default().buffer(pf.lm_qh).range(vk::WHOLE_SIZE)]),
                (2, [vk::DescriptorBufferInfo::default().buffer(pf.lm_scl).range(vk::WHOLE_SIZE)]),
                (3, [vk::DescriptorBufferInfo::default().buffer(pf.lm_dd).range(vk::WHOLE_SIZE)]),
                (4, [vk::DescriptorBufferInfo::default().buffer(pf.n32).offset(off).range((n_embd * 4) as u64)]),
                (5, [vk::DescriptorBufferInfo::default().buffer(pf.logits).range(vk::WHOLE_SIZE)]),
            ];
            let ulm = uni([vocab as u32, lm_nb as u32, gx, 0]);
            let uinfo = [vk::DescriptorBufferInfo::default().buffer(ulm).range(vk::WHOLE_SIZE)];
            let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
            dv.update_descriptor_sets(&w, &[]);
            disp(pf.p_q6k, set, gx, (vocab as u32).div_ceil(gx));
        }
        dv.end_command_buffer(cmd).unwrap();
        let fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        *self.prefill_rec.borrow_mut() = Some(((real_m, pos), PfRec { cmd, pool, cmd_pool, fence, unis: ub.into_inner() }));
        } // end if need_build

        // Reuse the recorded command (resources owned by prefill_rec; freed on rebuild/Drop).
        let rec = self.prefill_rec.borrow();
        let r = &rec.as_ref().unwrap().1;
        let t_gpu = std::time::Instant::now();
        dv.reset_fences(&[r.fence]).unwrap();
        dv.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[r.cmd])], r.fence).unwrap();
        dv.wait_for_fences(&[r.fence], true, u64::MAX).unwrap();
        if std::env::var("VK_PFTIME").is_ok() { eprintln!("prefill 128-row reuse: GPU-exec {:.1}ms (built={need_build})", t_gpu.elapsed().as_secs_f64() * 1e3); }
        std::slice::from_raw_parts(pf.logits_ptr as *const f32, vocab).to_vec()
    }

    /// Speculative-decode verification: run the batched coopmat forward over the
    /// `tokens` ([next, draft…]) at consecutive positions `pos..pos+len`, against
    /// the RESIDENT KV cache (filled 0..pos-1), and return the model's greedy
    /// argmax at each position. K/V for the real tokens is written into the cache
    /// at `pos` (so the sequence continues from there). Caller verifies the drafts
    /// and rolls back by position (rejected tail is overwritten next call).
    pub fn verify_forward(&self, tokens: &[u32], pos: usize) -> Vec<u32> {
        unsafe { self.verify_inner(tokens, pos) }
    }

    /// Prompt-Lookup speculative decode (greedy) on the coopmat path: prefill
    /// `prompt`, then each step verify an n-gram draft in ONE batched
    /// `verify_forward` and commit every token the model agrees with. Output is
    /// identical to greedy single-token decode; >1 token per forward on echo-heavy
    /// text. Returns (generated tokens, forwards). The prompt must already be the
    /// only thing in the resident cache (call on a fresh model / after a reset).
    pub fn generate_pld(&self, prompt: &[u32], max_new: usize, eos: u32, lookup_len: usize, draft_k: usize) -> (Vec<u32>, usize) {
        assert!(!prompt.is_empty() && prompt.len() + max_new < self.max_seq);
        // Prefill: feed the prompt; the last argmax is the first generated token.
        let mut next = 0u32;
        for (i, &t) in prompt.iter().enumerate() { next = self.forward_argmax(t, i); }
        let mut pos = prompt.len();
        let mut hist: Vec<u32> = prompt.to_vec();
        hist.push(next);
        let mut produced = vec![next];
        let mut forwards = 0usize;
        while produced.len() < max_new && next != eos {
            let k = draft_k.min(max_new - produced.len());
            match crate::engine::spec_decode::lookup_draft_best(&hist, &hist, lookup_len, k) {
                Some(d) => {
                    let mut inp = Vec::with_capacity(d.len() + 1);
                    inp.push(next);
                    inp.extend_from_slice(&d);
                    let outs = self.verify_forward(&inp, pos);
                    forwards += 1;
                    let mut accepted = 0usize;
                    while accepted < d.len() && outs[accepted] == d[accepted] { accepted += 1; }
                    for &tok in outs.iter().take(accepted + 1) {
                        produced.push(tok); hist.push(tok); next = tok;
                        if tok == eos || produced.len() >= max_new { break; }
                    }
                    pos += accepted + 1;
                }
                None => {
                    let tok = self.forward_argmax(next, pos);
                    forwards += 1;
                    produced.push(tok); hist.push(tok); next = tok; pos += 1;
                }
            }
        }
        (produced, forwards)
    }

    unsafe fn verify_inner(&self, tokens: &[u32], pos: usize) -> Vec<u32> {
        use ash::vk;
        let dv = &self.ctx.device;
        let pf = &self.prefill;
        let (n_embd, n_head, n_kv, hd, n_inter, kv_dim, half, vocab) =
            (self.n_embd, self.n_head, self.n_kv, self.hd, self.n_inter, self.kv_dim, self.half, self.vocab);
        let lm_nb = self.lm_nb;
        let real_m = tokens.len();
        assert!(real_m >= 1 && real_m <= VERIFY_MAX_M && pos + real_m <= self.max_seq);
        // Inputs: real-token embeddings + cos/sin for positions pos.. (real_m rows;
        // no padding — the small-M matvecs and batched ops all run at real_m).
        for (i, &tk) in tokens.iter().enumerate() {
            std::ptr::copy_nonoverlapping(self.embed[tk as usize * n_embd..].as_ptr() as *const u8, pf.x32_ptr.add(i * n_embd * 4), n_embd * 4);
        }
        for i in 0..real_m {
            let p = (pos + i).min(self.max_seq - 1);
            std::ptr::copy_nonoverlapping(self.cos[p * half..].as_ptr() as *const u8, pf.cosb_ptr.add(i * half * 4), half * 4);
            std::ptr::copy_nonoverlapping(self.sin[p * half..].as_ptr() as *const u8, pf.sinb_ptr.add(i * half * 4), half * 4);
        }

        // Reuse the pre-allocated verify scratch (no per-call vkAllocateMemory).
        let uni_i = std::cell::Cell::new(0usize);
        let uni = |d: [u32; 4]| -> vk::Buffer {
            let i = uni_i.get(); let (b, p) = self.verify_unis[i];
            std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16); uni_i.set(i + 1); b
        };
        let uni5 = |d: [u32; 5]| -> vk::Buffer {
            let i = uni_i.get(); let (b, p) = self.verify_unis[i];
            std::ptr::write_bytes(p, 0, 32); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 20); uni_i.set(i + 1); b
        };
        let (varg, varg_ptr) = self.verify_argmax;
        let pool = self.verify_pool;
        dv.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
        let mkset = |sl: vk::DescriptorSetLayout, sb: &[vk::Buffer], u: vk::Buffer| vk_alloc_set(dv, pool, sl, sb, u);
        let cmd = self.verify_cmd;
        dv.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        dv.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
        let tr = vk::PipelineStageFlags::TRANSFER;
        let bar = || { let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ); dv.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[b], &[], &[]); };
        let bar_c2t = || { let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ); dv.cmd_pipeline_barrier(cmd, cs, tr, vk::DependencyFlags::empty(), &[b], &[], &[]); };
        let bar_t2c = || { let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ); dv.cmd_pipeline_barrier(cmd, tr, cs, vk::DependencyFlags::empty(), &[b], &[], &[]); };
        let disp = |p: Pipe3, set: vk::DescriptorSet, gx: u32, gy: u32| {
            dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.0);
            dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.1, 0, &[set], &[]);
            dv.cmd_dispatch(cmd, gx, gy, 1);
        };
        let c64 = |x: usize| ((x + 63) / 64) as u32;
        let rm = real_m as u32;
        // Weight-stationary matvec: one workgroup per output column (W row); the
        // dequantized weight is reused across all real_m activation rows. Reads f32
        // activations directly (no f16 staging). Uniform: [M, N, K, nb/0, gx_grid].
        // ZLLM_VERIFY_SKINNY swaps in the column-tiled skinny GEMM (3.8x the matvec in isolation,
        // but ~parity end-to-end: verify_forward's ~40ms fixed overhead — re-record + ~200 barriers
        // — swamps it; needs a record-once batched forward to pay off). bmv is the validated default.
        let use_skinny = std::env::var("ZLLM_VERIFY_SKINNY").is_ok();
        // Diagnostic decomposition of the verify forward (mirror forward_inner's VK_MVONLY).
        let v_mvonly = std::env::var("VK_V_MVONLY").is_ok();              // keep only matvecs
        let v_skip_norm = v_mvonly || std::env::var("VK_V_NONORM").is_ok();
        let v_skip_attn = v_mvonly || std::env::var("VK_V_NOSDPA").is_ok(); // skip rope+kvcopy+sdpa
        let v_skip_extra = v_mvonly;                                       // skip silu+residual adds
        let mvq4 = |w: vk::Buffer, x: vk::Buffer, out: vk::Buffer, n: usize, k: usize| {
            if use_skinny { // grid = ceil(N/64) workgroups (64 columns each)
                let set = mkset(pf.p_skinny.2, &[w, x, out], uni5([rm, n as u32, k as u32, (k / 256) as u32, 0]));
                disp(pf.p_skinny, set, (n as u32).div_ceil(64), 1);
            } else {
                let gx = (n as u32).min(65535);
                let set = mkset(pf.p_bmq4.2, &[w, x, out], uni5([rm, n as u32, k as u32, (k / 256) as u32, gx]));
                disp(pf.p_bmq4, set, gx, (n as u32).div_ceil(gx));
            }
        };
        let mvf16 = |w: vk::Buffer, x: vk::Buffer, out: vk::Buffer, n: usize, k: usize| {
            let gx = (n as u32).min(65535);
            let set = mkset(pf.p_bmf16.2, &[w, x, out], uni5([rm, n as u32, k as u32, 0, gx]));
            disp(pf.p_bmf16, set, gx, (n as u32).div_ceil(gx));
        };
        let u_eps = uni([n_embd as u32, self.eps.to_bits(), 0, 0]);
        let kvoff = (pos * kv_dim * 4) as u64;
        let kvsz = (real_m * kv_dim * 4) as u64;

        for l in &pf.w {
            if !v_skip_norm { disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.attn_norm, pf.n32], u_eps), rm, 1); bar(); }
            // QKV via small-M matvec: Q -> pf.q, K -> pf.o32, V -> pf.ffn32.
            mvq4(l.wq, pf.n32, pf.q, n_embd, n_embd);
            mvq4(l.wk, pf.n32, pf.o32, kv_dim, n_embd);
            mvf16(l.wv_f16, pf.n32, pf.ffn32, kv_dim, n_embd); bar();
            if !v_skip_attn {
                // RoPE q (pf.q), k (pf.o32 scratch).
                disp(pf.p_br, mkset(pf.p_br.2, &[pf.q, pf.cosb, pf.sinb], uni([n_head as u32, hd as u32, rm, 0])), c64(real_m * n_head * half), 1);
                disp(pf.p_br, mkset(pf.p_br.2, &[pf.o32, pf.cosb, pf.sinb], uni([n_kv as u32, hd as u32, rm, 0])), c64(real_m * n_kv * half), 1); bar();
                // Copy the real K/V into the resident cache at position `pos`.
                bar_c2t();
                dv.cmd_copy_buffer(cmd, pf.o32, l.kc, &[vk::BufferCopy::default().size(kvsz).dst_offset(kvoff)]);
                dv.cmd_copy_buffer(cmd, pf.ffn32, l.vc, &[vk::BufferCopy::default().size(kvsz).dst_offset(kvoff)]);
                bar_t2c();
                // Decode SDPA over the resident cache (row i attends 0..=pos+i).
                disp(pf.p_bsd, mkset(pf.p_bsd.2, &[pf.q, l.kc, l.vc, pf.attn32], uni5([n_head as u32, n_kv as u32, hd as u32, rm, pos as u32])), c64(real_m * n_head), 1); bar();
            }
            // Wo + residual.
            mvq4(l.wo, pf.attn32, pf.o32, n_embd, n_embd); bar();
            if !v_skip_extra { disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.o32], uni([(real_m * n_embd) as u32, 0, 0, 0])), c64(real_m * n_embd), 1); bar(); }
            // FFN.
            if !v_skip_norm { disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.ffn_norm, pf.n32], u_eps), rm, 1); bar(); }
            mvq4(l.w13, pf.n32, pf.gu, 2 * n_inter, n_embd); bar();
            if !v_skip_extra { disp(pf.p_bsi, mkset(pf.p_bsi.2, &[pf.gu, pf.h32], uni([n_inter as u32, rm, 0, 0])), c64(real_m * n_inter), 1); bar(); }
            mvf16(l.w2_f16, pf.h32, pf.ffn32, n_embd, n_inter); bar();
            if !v_skip_extra { disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.ffn32], uni([(real_m * n_embd) as u32, 0, 0, 0])), c64(real_m * n_embd), 1); bar(); }
        }
        // Final norm; ONE batched Q6 LM head over all real_m rows (weight streamed
        // once, not real_m times) -> vlogits[real_m, vocab]; then per-row argmax.
        disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, pf.final_norm, pf.n32], u_eps), rm, 1); bar();
        let gx = (vocab as u32).min(65535);
        let lm_set = mkset(pf.p_bmq6.2, &[pf.lm_ql, pf.lm_qh, pf.lm_scl, pf.lm_dd, pf.n32, pf.vlogits], uni([vocab as u32, lm_nb as u32, gx, rm]));
        disp(pf.p_bmq6, lm_set, gx, (vocab as u32).div_ceil(gx)); bar();
        for i in 0..real_m {
            let uarg = uni([vocab as u32, 0, 0, 0]);
            let arg_set = vk_alloc_set_off(dv, pool, self.p_argmax.2, &[(pf.vlogits, (i * vocab * 4) as u64, (vocab * 4) as u64), (varg, (i * 4) as u64, 4)], uarg);
            disp(self.p_argmax, arg_set, 1, 1); bar();
        }
        dv.end_command_buffer(cmd).unwrap();
        dv.reset_fences(&[self.verify_fence]).unwrap();
        dv.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], self.verify_fence).unwrap();
        dv.wait_for_fences(&[self.verify_fence], true, u64::MAX).unwrap();
        // Reused resources (pool/cmd/fence/uniforms) — nothing to free per call.
        std::slice::from_raw_parts(varg_ptr as *const u32, real_m).to_vec()
    }

    unsafe fn forward_inner(&self, token: u32, pos: usize, lm: bool, argmax: bool) -> (Vec<f32>, u32) {
        use ash::vk;
        let dv = &self.ctx.device;
        let (n_embd, n_head, n_kv, hd, n_inter, kv_dim, half) = (self.n_embd, self.n_head, self.n_kv, self.hd, self.n_inter, self.kv_dim, self.half);
        let attn_dim = n_head * hd;
        let seq_len = (pos + 1) as u32;
        // Head-major KV cache (ZLLM_HEADMAJOR_KV): cache laid [kv_head, pos, hd] so
        // each SDPA workgroup reads its kv-head CONTIGUOUSLY (vs strided by n_kv*hd).
        // Requires the head-major kvwrite + partial AND always-flash (the single-pass
        // sdpa_decode is pos-major), so we force the flash path at every depth. The
        // combine reads partials (not the cache) → unchanged. Decode-cache-fill only
        // (prefill still writes pos-major), so this is a research/measurement flag.
        let headmajor = self.headmajor;
        let flash = headmajor || seq_len as usize > FLASH_MIN_SEQ;
        // Update per-token mapped buffers.
        std::ptr::copy_nonoverlapping(self.embed[token as usize * n_embd..].as_ptr() as *const u8, self.x_ptr, n_embd * 4);
        std::ptr::copy_nonoverlapping(self.cos[pos * half..].as_ptr() as *const u8, self.cos_ptr, half * 4);
        std::ptr::copy_nonoverlapping(self.sin[pos * half..].as_ptr() as *const u8, self.sin_ptr, half * 4);
        std::ptr::copy_nonoverlapping([kv_dim as u32, (pos * kv_dim) as u32, hd as u32, 0u32].as_ptr() as *const u8, self.base_ptr, 16); // slot2=hd for head-major kvwrite
        std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, seq_len].as_ptr() as *const u8, self.seq_ptr, 16);
        if flash { // refresh hierarchical-combine uniforms for this depth
            let nblk = (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK);
            let n_super = nblk.div_ceil(SDPA_SUPER);
            std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, nblk as u32, SDPA_SUPER as u32, 0u32].as_ptr() as *const u8, self.l1_ptr, 24);
            std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, n_super as u32, n_super as u32, 1u32].as_ptr() as *const u8, self.l2_ptr, 24);
        }

        // Record-once: only re-record when the SDPA grid (single-pass vs flash
        // n_blocks) or the lm/argmax tail changes; otherwise reuse self.cmd.
        let sdpa_key = if flash { (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK) as i64 } else { 0 };
        let rec_key = sdpa_key | ((lm as i64) << 20) | ((argmax as i64) << 21);
        let need_record = self.last_rec.get() != rec_key;
        let cmd = self.cmd;
        if need_record {
        dv.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        dv.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        let bar = || dv.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        let disp = |p: (vk::Pipeline, vk::PipelineLayout), set: vk::DescriptorSet, gx: u32, gy: u32| {
            dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.0);
            dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.1, 0, &[set], &[]);
            dv.cmd_dispatch(cmd, gx, gy, 1);
        };
        let gxof = |n: usize| (n as u32).min(65535);
        let mv = |m: Mv, n: usize| disp(match m.pipe { 1 => self.p_q6k, 2 => self.p_q3, _ => self.p_mv }, m.set, gxof(n), (n as u32).div_ceil(gxof(n)));
        let mvonly = std::env::var("VK_MVONLY").is_ok();      // diag: matvecs only (skip all small ops)
        let skip_extra = mvonly || std::env::var("VK_NOEXTRA").is_ok(); // diag: skip kvwrite+residual
        let skip_norm = mvonly || std::env::var("VK_NONORM").is_ok();   // diag: skip rmsnorms
        let skip_attn = mvonly;                               // diag: skip rope+sdpa
        let flat_combine = std::env::var("ZLLM_FLAT_COMBINE").is_ok(); // diag A/B: flat vs hierarchical combine
        let online_partial = std::env::var("ZLLM_ONLINE_PARTIAL").is_ok(); // diag A/B: force online (vs 2-pass) partial
        let use_p2 = hd <= 64 && !online_partial; // 2-pass partial only valid for hd<=64
        let skip_lm = std::env::var("VK_NOLM").is_ok(); // diag: skip LM-head matvec (isolate its cost)
        let skip_ffn = std::env::var("VK_NOFFN").is_ok(); // diag: skip FFN matvecs (w13/w2 + silu)
        let skip_w2 = std::env::var("VK_NOW2").is_ok();   // diag: skip just w2 (long-K down proj)
        for l in &self.layers {
            if !skip_norm { disp(self.p_rms, l.attn_norm, 1, 1); bar(); }              // attn norm
            // QKV with RoPE fused into the wq/wk output (q/k come out rotated);
            // wv normal (V isn't roped). Removes the separate RoPE dispatch + barrier.
            if let Some(wqk) = l.wqk_rope {
                // Fused: one mvrope over concat wq+wk (grid-starve fix) → qk_buf.
                let pr = (n_embd + kv_dim) / 2;
                disp(self.p_mvrope, wqk, gxof(pr), (pr as u32).div_ceil(gxof(pr)));
            } else {
                disp(self.p_mvrope, l.wq_rope, gxof(n_embd / 2), ((n_embd / 2) as u32).div_ceil(gxof(n_embd / 2)));
                disp(self.p_mvrope, l.wk_rope, gxof(kv_dim / 2), ((kv_dim / 2) as u32).div_ceil(gxof(kv_dim / 2)));
            }
            mv(l.wv, kv_dim); bar();                                                   // QKV + rope(q,k)
            if !skip_attn {
            let kvw_pipe = if headmajor { self.p_kvw_hm } else { self.p_kvw };
            if !skip_extra { disp(kvw_pipe, l.kvw_k, (kv_dim as u32).div_ceil(64), 1);
            disp(kvw_pipe, l.kvw_v, (kv_dim as u32).div_ceil(64), 1); bar(); }       // append K,V to cache
            if flash {
                let nblk = (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK) as u32;
                let n_super = (nblk as usize).div_ceil(SDPA_SUPER) as u32;
                let fp_pipe = if headmajor { self.p_fp2_hm } else if use_p2 { self.p_fp2 } else { self.p_fp };
                disp(fp_pipe, l.fp, n_head as u32, nblk); bar(); // partials per block (head-major / 2-pass if hd<=64)
                if flat_combine {
                    disp(self.p_fc, l.fc, n_head as u32, 1); bar();     // flat combine (A/B baseline)
                } else {
                    disp(self.p_ch, l.fc1, n_head as u32, n_super); bar();  // L1: blocks -> super-partials (parallel)
                    disp(self.p_ch, l.fc2, n_head as u32, 1); bar();        // L2: super-partials -> attn
                }
            } else {
                disp(self.p_sdpa, l.sdpa, n_head as u32, 1); bar();
            }
            }
            mv(l.wo, n_embd); bar();                                                   // O proj + residual (folded: x += attn)
            if !skip_norm { disp(self.p_rms, l.ffn_norm, 1, 1); bar(); }               // ffn norm
            if !skip_ffn {
                mv(l.w13, n_inter * 2); bar();                                         // gate+up (concat, one matvec)
                if !skip_attn { disp(self.p_silu, self.s_silu, (n_inter as u32).div_ceil(64), 1); bar(); } // silu·mul
                if !skip_w2 { mv(l.w2, n_embd); bar(); }                               // down proj + residual (folded: x += ffn)
            }
        }
        if lm {
            if !skip_norm { disp(self.p_rms, self.s_final_norm, 1, 1); bar(); }        // final norm
            if !skip_lm { disp(if self.lm_q4 { self.p_mv } else { self.p_q6k }, self.s_lm, gxof(self.vocab), (self.vocab as u32).div_ceil(gxof(self.vocab))); } // LM head (Q4 or Q6)
            if argmax { bar(); disp((self.p_argmax.0, self.p_argmax.1), self.s_argmax, 1, 1); }  // GPU argmax (4-byte readback)
        }
        let _ = attn_dim;
        dv.end_command_buffer(cmd).unwrap();
        self.last_rec.set(rec_key);
        } // end if need_record
        let t_rec = std::time::Instant::now();
        dv.reset_fences(&[self.fence]).unwrap();
        let cmds = [cmd];
        dv.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence).unwrap();
        dv.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
        if std::env::var("VK_TIME").is_ok() { eprintln!("  gpu {:.2}ms", t_rec.elapsed().as_secs_f64() * 1e3); }
        if !lm { (Vec::new(), 0) }
        else if argmax { (Vec::new(), *(self.argmax_ptr as *const u32)) }
        else { (std::slice::from_raw_parts(self.logits_ptr as *const f32, self.vocab).to_vec(), 0) }
    }
}

// Safe: the model is only ever accessed behind a Mutex (one thread at a time);
// the raw mapped pointers are valid for the model's lifetime.
unsafe impl Send for VkModel {}

impl Drop for VkModel {
    fn drop(&mut self) {
        unsafe {
            let dv = &self.ctx.device;
            let _ = dv.device_wait_idle();
            if let Some((_, r)) = self.prefill_rec.borrow_mut().take() { r.destroy(dv); }
            dv.destroy_fence(self.fence, None);
            dv.destroy_command_pool(self.cmd_pool, None);
            dv.destroy_descriptor_pool(self.desc_pool, None);
            for (p, l, sl, m) in self.pipes.drain(..) {
                dv.destroy_pipeline(p, None); dv.destroy_pipeline_layout(l, None);
                dv.destroy_descriptor_set_layout(sl, None); dv.destroy_shader_module(m, None);
            }
            for (b, m) in self.bufs.drain(..) {
                dv.unmap_memory(m); dv.destroy_buffer(b, None); dv.free_memory(m, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
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
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
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
        use crate::backend::traits::{Backend, QuantConfig};
        let mut cb = CandleCpuBackend::new();
        cb.load_model(std::path::Path::new(path), &QuantConfig { method: "gguf".into(), bits: 4 }).expect("candle load");
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
}

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
