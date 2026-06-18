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
const DECODE_MATVEC_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_q4k.spv");
const DECODE_MATVEC_DOWN_Q4K_SPV: &[u8] = include_bytes!("shaders/decode_matvec_down_q4k.spv");
const RMSNORM_SPV: &[u8] = include_bytes!("shaders/rmsnorm.spv");
const ROPE_SPV: &[u8] = include_bytes!("shaders/rope.spv");
const SDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/sdpa_decode.spv");
const SDPA_FLASH_PARTIAL_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_partial.spv");
const SDPA_FLASH_COMBINE_SPV: &[u8] = include_bytes!("shaders/sdpa_flash_combine.spv");
const SDPA_FLASH_BLOCK: usize = 32; // must match BLOCK in the flash shaders
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
const KV_WRITE_SPV: &[u8] = include_bytes!("shaders/kv_write.spv");
const RESIDUAL_ADD_SPV: &[u8] = include_bytes!("shaders/residual_add.spv");
const ARGMAX_SPV: &[u8] = include_bytes!("shaders/argmax.spv");
// Batched prefill kernels.
const BNORM_SPV: &[u8] = include_bytes!("shaders/bnorm.spv");
const BROPE_SPV: &[u8] = include_bytes!("shaders/brope.spv");
const BSDPA_SPV: &[u8] = include_bytes!("shaders/bsdpa.spv");
const BSILU_SPV: &[u8] = include_bytes!("shaders/bsilu.spv");
const TO_F16_SPV: &[u8] = include_bytes!("shaders/to_f16.spv");

/// Max prompt length for the raw-Vulkan server fast-lane. Prefill is sequential
/// (one forward per prompt token) so this is kept modest; batched prefill via
/// the coopmat GEMM is the follow-up that would raise it.
pub const MAX_PREFILL_M: usize = 128;
const INC_SPV: &[u8] = include_bytes!("shaders/inc.spv");
const INC_COH_SPV: &[u8] = include_bytes!("shaders/inc_coh.spv");

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
    pub fn coopmat_q4k_gemm(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], m: usize, iters: u32) -> Result<(Vec<f32>, f64), String> {
        let k = nb * 256;
        assert_eq!(weight_bytes.len(), n * nb * 144);
        assert_eq!(x.len(), m * k);
        assert!(m % 128 == 0 && n % 128 == 0, "register-blocked GEMM tile is 128x128");
        unsafe { self.coopmat_q4k_gemm_inner(weight_bytes, n, nb, x, m, iters) }
    }

    unsafe fn coopmat_q4k_gemm_inner(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], m: usize, iters: u32) -> Result<(Vec<f32>, f64), String> {
        use std::time::Instant;
        let dev = &self.device;
        let k = nb * 256;
        let (a_buf, a_mem, a_ptr) = self.uma_buffer((m * k * 2) as u64)?;
        let (w_buf, w_mem, w_ptr) = self.uma_buffer(weight_bytes.len() as u64)?;
        let (c_buf, c_mem, c_ptr) = self.uma_buffer((m * n * 4) as u64)?;
        let (p_buf, p_mem, p_ptr) = self.uma_buffer(16)?;
        let a16: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        std::ptr::copy_nonoverlapping(a16.as_ptr() as *const u8, a_ptr, m * k * 2);
        std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), w_ptr, weight_bytes.len());
        let dims = [m as u32, n as u32, k as u32, nb as u32];
        std::ptr::copy_nonoverlapping(dims.as_ptr() as *const u8, p_ptr, 16);

        let spv = ash::util::read_spv(&mut std::io::Cursor::new(COOPMAT_Q4K_GEMM_SPV)).map_err(|e| format!("read_spv: {e}"))?;
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
        let gx = (n / 128) as u32; let gy = (m / 128) as u32;
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
        unsafe { self.decode_matvec_q4k_inner(weight_bytes, n, nb, x, iters) }
    }

    unsafe fn decode_matvec_q4k_inner(&self, weight_bytes: &[u8], n: usize, nb: usize, x: &[f32], iters: u32) -> Result<(Vec<f32>, f64), String> {
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

        let spv = ash::util::read_spv(&mut std::io::Cursor::new(DECODE_MATVEC_Q4K_SPV)).map_err(|e| format!("read_spv: {e}"))?;
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
    wv_f16: ash::vk::Buffer, w2_f16: ash::vk::Buffer,                                    // Q6 dequant -> f16 [K,N]
    attn_norm: ash::vk::Buffer, ffn_norm: ash::vk::Buffer,                               // f32
    kc: ash::vk::Buffer, vc: ash::vk::Buffer,                                            // shared decode KV cache
}

const PREFILL_MAX_M: usize = 128; // fast-lane prompt cap; M padded to this

// Resources for the batched prefill forward (built once at load).
type Pipe3 = (ash::vk::Pipeline, ash::vk::PipelineLayout, ash::vk::DescriptorSetLayout);
struct PrefillRes {
    w: Vec<PrefillW>,
    p_q4: Pipe3, p_f16: Pipe3, p_bn: Pipe3, p_br: Pipe3, p_bs: Pipe3, p_bsi: Pipe3, p_t16: Pipe3,
    p_add: Pipe3, p_q6k: Pipe3,
    lm_ql: ash::vk::Buffer, lm_qh: ash::vk::Buffer, lm_scl: ash::vk::Buffer, lm_dd: ash::vk::Buffer,
    final_norm: ash::vk::Buffer, logits: ash::vk::Buffer, logits_ptr: *mut u8,
    // scratch (sized for PREFILL_MAX_M rows)
    x32: ash::vk::Buffer, x32_ptr: *mut u8, x16: ash::vk::Buffer,
    n32: ash::vk::Buffer, n16: ash::vk::Buffer, q: ash::vk::Buffer,
    attn32: ash::vk::Buffer, attn16: ash::vk::Buffer, gu: ash::vk::Buffer,
    h32: ash::vk::Buffer, h16: ash::vk::Buffer, o32: ash::vk::Buffer, ffn32: ash::vk::Buffer,
    cosb: ash::vk::Buffer, cosb_ptr: *mut u8, sinb: ash::vk::Buffer, sinb_ptr: *mut u8,
}

// A loaded weight: Q4_K (raw bytes) or Q6_K (repacked SoA). nb = cols/256.
enum VkWeight {
    Q4 { buf: ash::vk::Buffer, nb: usize },
    Q6 { ql: ash::vk::Buffer, qh: ash::vk::Buffer, scl: ash::vk::Buffer, dd: ash::vk::Buffer, nb: usize },
}
// A matvec dispatch: which kernel + its descriptor set.
#[derive(Clone, Copy)]
struct Mv { q6: bool, set: ash::vk::DescriptorSet }

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

// Build a matvec descriptor set for a weight (Q4 or Q6) producing `rows` outputs.
unsafe fn vk_mk_mv(ctx: &VkContext, pool: ash::vk::DescriptorPool, mv_sl: ash::vk::DescriptorSetLayout, q6k_sl: ash::vk::DescriptorSetLayout, bufs: &mut Vec<(ash::vk::Buffer, ash::vk::DeviceMemory)>, w: &VkWeight, x_in: ash::vk::Buffer, out: ash::vk::Buffer, rows: usize) -> Mv {
    let gx = (rows as u32).min(65535);
    match *w {
        VkWeight::Q4 { buf, nb } => {
            let (u, _) = vk_uni(ctx, bufs, [rows as u32, (nb * 256) as u32, nb as u32, gx]);
            Mv { q6: false, set: vk_alloc_set(&ctx.device, pool, mv_sl, &[buf, x_in, out], u) }
        }
        VkWeight::Q6 { ql, qh, scl, dd, nb } => {
            let (u, _) = vk_uni(ctx, bufs, [rows as u32, nb as u32, gx, 0]);
            Mv { q6: true, set: vk_alloc_set(&ctx.device, pool, q6k_sl, &[ql, qh, scl, dd, x_in, out], u) }
        }
    }
}

#[derive(Clone)]
struct VkLayerOps {
    attn_norm: ash::vk::DescriptorSet, wq: Mv, wk: Mv, wv: Mv,
    kvw_k: ash::vk::DescriptorSet, kvw_v: ash::vk::DescriptorSet, sdpa: ash::vk::DescriptorSet,
    fp: ash::vk::DescriptorSet, fc: ash::vk::DescriptorSet, wo: Mv, radd_a: ash::vk::DescriptorSet,
    ffn_norm: ash::vk::DescriptorSet, w13: Mv, w2: Mv, radd_f: ash::vk::DescriptorSet,
}

/// A loaded GGUF running on the raw-Vulkan decode kernels.
pub struct VkModel {
    pub n_embd: usize, n_head: usize, n_kv: usize, hd: usize, n_inter: usize,
    pub vocab: usize, n_layers: usize, kv_dim: usize, half: usize, max_seq: usize, eps: f32,
    embed: Vec<f32>, cos: Vec<f32>, sin: Vec<f32>,
    // dispatch dims
    lm_nb: usize,
    // per-token mapped uniforms / activations
    x_ptr: *mut u8, cos_ptr: *mut u8, sin_ptr: *mut u8, base_ptr: *mut u8, seq_ptr: *mut u8, logits_ptr: *mut u8,
    // descriptor sets
    layers: Vec<VkLayerOps>,
    s_rope_q: ash::vk::DescriptorSet, s_rope_k: ash::vk::DescriptorSet, s_silu: ash::vk::DescriptorSet,
    s_final_norm: ash::vk::DescriptorSet, s_lm: ash::vk::DescriptorSet, s_argmax: ash::vk::DescriptorSet,
    argmax_ptr: *mut u8,
    // Record-once cache: the command buffer only changes when the flash-attn
    // grid (n_blocks) or the lm/argmax tail changes — reuse it otherwise and
    // just refresh the mapped uniforms. -1 = not yet recorded.
    last_rec: std::cell::Cell<i64>,
    // pipelines (pipeline, layout)
    p_rms: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_mv: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_rope: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_kvw: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_sdpa: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_fp: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_fc: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_silu: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_add: (ash::vk::Pipeline, ash::vk::PipelineLayout), p_q6k: (ash::vk::Pipeline, ash::vk::PipelineLayout),
    p_argmax: (ash::vk::Pipeline, ash::vk::PipelineLayout),
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
        let max_seq = 4096usize;

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
        let load_gateup = |bufs: &mut Vec<_>, file: &mut std::fs::File, p: &str| -> Result<VkWeight, String> {
            let g = ct.tensor(file, &format!("{p}.ffn_gate.weight"), &dev).map_err(|e| e.to_string())?;
            let u = ct.tensor(file, &format!("{p}.ffn_up.weight"), &dev).map_err(|e| e.to_string())?;
            if g.dtype() != GgmlDType::Q4K || u.dtype() != GgmlDType::Q4K {
                return Err(format!("{p}: gate/up concat needs Q4_K (got {:?}/{:?})", g.dtype(), u.dtype()));
            }
            let nb = g.shape().dims()[1] / 256;
            let mut b = g.data().map_err(|e| e.to_string())?.to_vec();
            b.extend_from_slice(&u.data().map_err(|e| e.to_string())?);
            Ok(VkWeight::Q4 { buf: vk_up_bytes(&ctx, bufs, &b), nb })
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
        let (lm_ql, lm_qh, lm_scl, lm_dd) = match embed_qt.dtype() {
            GgmlDType::Q6K => vk_up_q6k(&ctx, &mut bufs, &lm_bytes, vocab * lm_nb),
            d => return Err(format!("LM head (token_embd.weight) dtype {d:?} not supported (need Q6_K)")),
        };
        let final_norm = load_norm(&mut bufs, &mut file, "output_norm.weight")?;
        let n_inter = ct.tensor(&mut file, "blk.0.ffn_gate.weight", &dev).map_err(|e| e.to_string())?.shape().dims()[0];

        // Per-layer weights + KV cache.
        struct RawLayer { an: vk::Buffer, fn_: vk::Buffer, wq: VkWeight, wk: VkWeight, wv: VkWeight, wo: VkWeight, w13: VkWeight, w2: VkWeight, kc: vk::Buffer, vc: vk::Buffer }
        let mut raw: Vec<RawLayer> = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            let an = load_norm(&mut bufs, &mut file, &format!("{p}.attn_norm.weight"))?;
            let fn_ = load_norm(&mut bufs, &mut file, &format!("{p}.ffn_norm.weight"))?;
            let wq = load_w(&mut bufs, &mut file, &format!("{p}.attn_q.weight"))?;
            let wk = load_w(&mut bufs, &mut file, &format!("{p}.attn_k.weight"))?;
            let wv = load_w(&mut bufs, &mut file, &format!("{p}.attn_v.weight"))?;
            let wo = load_w(&mut bufs, &mut file, &format!("{p}.attn_output.weight"))?;
            let w13 = load_gateup(&mut bufs, &mut file, &p)?;
            let w2 = load_w(&mut bufs, &mut file, &format!("{p}.ffn_down.weight"))?;
            let (kc, _) = vk_zeros(&ctx, &mut bufs, max_seq * kv_dim);
            let (vc, _) = vk_zeros(&ctx, &mut bufs, max_seq * kv_dim);
            raw.push(RawLayer { an, fn_, wq, wk, wv, wo, w13, w2, kc, vc });
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
        let (attn, _) = vk_zeros(&ctx, &mut bufs, attn_dim);
        let (o_buf, _) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (gu, _) = vk_zeros(&ctx, &mut bufs, n_inter * 2); // [gate(n_inter); up(n_inter)]
        let (hbuf, _) = vk_zeros(&ctx, &mut bufs, n_inter);
        let (ffn_buf, _) = vk_zeros(&ctx, &mut bufs, n_embd);
        let (logits, logits_ptr) = vk_zeros(&ctx, &mut bufs, vocab);
        let nblk_max = max_seq.div_ceil(SDPA_FLASH_BLOCK);
        let (part, _) = vk_zeros(&ctx, &mut bufs, n_head * nblk_max * (hd + 2));
        // Uniforms.
        let (u_norm, _) = vk_uni(&ctx, &mut bufs, [n_embd as u32, eps.to_bits(), 0, 0]);
        let (u_rope_q, _) = vk_uni(&ctx, &mut bufs, [n_head as u32, hd as u32, 0, 0]);
        let (u_rope_k, _) = vk_uni(&ctx, &mut bufs, [n_kv as u32, hd as u32, 0, 0]);
        let (u_base, base_ptr) = vk_uni(&ctx, &mut bufs, [kv_dim as u32, 0, 0, 0]);
        let (u_seq, seq_ptr) = vk_uni(&ctx, &mut bufs, [n_head as u32, n_kv as u32, hd as u32, 1]);
        let (u_silu, _) = vk_uni(&ctx, &mut bufs, [n_inter as u32, 0, 0, 0]);
        let (u_add, _) = vk_uni(&ctx, &mut bufs, [n_embd as u32, 0, 0, 0]);
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
        let (rope_p, rope_l, rope_sl) = mkpipe(ROPE_SPV, 3);
        let (kvw_p, kvw_l, kvw_sl) = mkpipe(KV_WRITE_SPV, 2);
        let (sdpa_p, sdpa_l, sdpa_sl) = mkpipe(SDPA_DECODE_SPV, 4);
        let (fp_p, fp_l, fp_sl) = mkpipe(SDPA_FLASH_PARTIAL_SPV, 4);
        let (fc_p, fc_l, fc_sl) = mkpipe(SDPA_FLASH_COMBINE_SPV, 2);
        let (silu_p, silu_l, silu_sl) = mkpipe(SILU_MUL_SPV, 3);
        let (add_p, add_l, add_sl) = mkpipe(RESIDUAL_ADD_SPV, 2);
        let (q6k_p, q6k_l, q6k_sl) = mkpipe(DECODE_MATVEC_Q6K_SPV, 6);
        let (argmax_p, argmax_l, argmax_sl) = mkpipe(ARGMAX_SPV, 2);

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
        let s_lm = mkset(q6k_sl, &[lm_ql, lm_qh, lm_scl, lm_dd, normed, logits], u_lm);
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
            let wq = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wq, normed, q, n_embd);
            let wk = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wk, normed, k_buf, kv_dim);
            let wv = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wv, normed, v_buf, kv_dim);
            let wo = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.wo, attn, o_buf, n_embd);
            let w13 = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.w13, normed, gu, n_inter * 2);
            let w2 = vk_mk_mv(&ctx, desc_pool, mv_sl, q6k_sl, &mut bufs, &r.w2, hbuf, ffn_buf, n_embd);
            layers.push(VkLayerOps {
                attn_norm: mkset(rms_sl, &[x_buf, r.an, normed], u_norm),
                wq, wk, wv,
                kvw_k: mkset(kvw_sl, &[r.kc, k_buf], u_base),
                kvw_v: mkset(kvw_sl, &[r.vc, v_buf], u_base),
                sdpa: mkset(sdpa_sl, &[q, r.kc, r.vc, attn], u_seq),
                fp: mkset(fp_sl, &[q, r.kc, r.vc, part], u_seq),
                fc: mkset(fc_sl, &[part, attn], u_seq),
                wo,
                radd_a: mkset(add_sl, &[x_buf, o_buf], u_add),
                ffn_norm: mkset(rms_sl, &[x_buf, r.fn_, normed], u_norm),
                w13, w2,
                radd_f: mkset(add_sl, &[x_buf, ffn_buf], u_add),
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
            pw.push(PrefillW { wq: q4buf(&r.wq), wk: q4buf(&r.wk), wo: q4buf(&r.wo), w13: q4buf(&r.w13), wv_f16, w2_f16, attn_norm: r.an, ffn_norm: r.fn_, kc: r.kc, vc: r.vc });
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
        let p_bsi = mkp(BSILU_SPV, 2, false);
        let p_t16 = mkp(TO_F16_SPV, 2, false);
        let p_add = mkp(RESIDUAL_ADD_SPV, 2, false);
        let p_q6k = mkp(DECODE_MATVEC_Q6K_SPV, 6, false);
        let mm = PREFILL_MAX_M;
        let f16buf = |bufs: &mut Vec<(vk::Buffer, vk::DeviceMemory)>, len: usize| -> vk::Buffer { let (b, m, p) = ctx.uma_buffer((len * 2) as u64).unwrap(); std::ptr::write_bytes(p, 0, len * 2); bufs.push((b, m)); b };
        let (pf_x32, pf_x32_ptr) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let pf_x16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_n32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let pf_n16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_q, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let (pf_attn32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let pf_attn16 = f16buf(&mut bufs, mm * n_embd);
        let (pf_gu, _) = vk_zeros(&ctx, &mut bufs, mm * n_inter * 2);
        let (pf_h32, _) = vk_zeros(&ctx, &mut bufs, mm * n_inter); let pf_h16 = f16buf(&mut bufs, mm * n_inter);
        let (pf_o32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd); let (pf_ffn32, _) = vk_zeros(&ctx, &mut bufs, mm * n_embd);
        let (pf_cosb, pf_cosb_ptr) = vk_zeros(&ctx, &mut bufs, mm * half);
        let (pf_sinb, pf_sinb_ptr) = vk_zeros(&ctx, &mut bufs, mm * half);
        let (pf_logits, pf_logits_ptr) = vk_zeros(&ctx, &mut bufs, vocab);
        PrefillRes {
            w: pw, p_q4, p_f16, p_bn, p_br, p_bs, p_bsi, p_t16, p_add, p_q6k,
            lm_ql, lm_qh, lm_scl, lm_dd, final_norm, logits: pf_logits, logits_ptr: pf_logits_ptr,
            x32: pf_x32, x32_ptr: pf_x32_ptr, x16: pf_x16, n32: pf_n32, n16: pf_n16, q: pf_q,
            attn32: pf_attn32, attn16: pf_attn16, gu: pf_gu, h32: pf_h32, h16: pf_h16, o32: pf_o32, ffn32: pf_ffn32,
            cosb: pf_cosb, cosb_ptr: pf_cosb_ptr, sinb: pf_sinb, sinb_ptr: pf_sinb_ptr,
        }
        };

        let cmd_pool = dv.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("cmd pool: {e}"))?;
        let cmd = dv.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).map_err(|e| format!("cmd buf: {e}"))?[0];
        let fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;

        Ok(Self {
            n_embd, n_head, n_kv, hd, n_inter, vocab, n_layers, kv_dim, half, max_seq, eps, embed, cos, sin, lm_nb,
            x_ptr, cos_ptr, sin_ptr, base_ptr, seq_ptr, logits_ptr,
            layers, s_rope_q, s_rope_k, s_silu, s_final_norm, s_lm, s_argmax, argmax_ptr,
            p_rms: (rms_p, rms_l), p_mv: (mv_p, mv_l), p_rope: (rope_p, rope_l), p_kvw: (kvw_p, kvw_l),
            p_sdpa: (sdpa_p, sdpa_l), p_fp: (fp_p, fp_l), p_fc: (fc_p, fc_l), p_silu: (silu_p, silu_l),
            p_add: (add_p, add_l), p_q6k: (q6k_p, q6k_l), p_argmax: (argmax_p, argmax_l),
            last_rec: std::cell::Cell::new(-1),
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
    pub fn prefill_forward(&self, prompt: &[u32]) -> Vec<f32> {
        unsafe { self.prefill_inner(prompt) }
    }

    unsafe fn prefill_inner(&self, prompt: &[u32]) -> Vec<f32> {
        use ash::vk;
        let dv = &self.ctx.device;
        let pf = &self.prefill;
        let (n_embd, n_head, n_kv, hd, n_inter, kv_dim, half, vocab) =
            (self.n_embd, self.n_head, self.n_kv, self.hd, self.n_inter, self.kv_dim, self.half, self.vocab);
        let lm_nb = self.lm_nb;
        let real_m = prompt.len().min(PREFILL_MAX_M);
        let m = PREFILL_MAX_M;

        // Inputs: x = embeddings (padding rows zero), cos/sin for positions 0..m.
        std::ptr::write_bytes(pf.x32_ptr, 0, m * n_embd * 4);
        for (i, &tk) in prompt.iter().take(real_m).enumerate() {
            std::ptr::copy_nonoverlapping(self.embed[tk as usize * n_embd..].as_ptr() as *const u8, pf.x32_ptr.add(i * n_embd * 4), n_embd * 4);
        }
        std::ptr::copy_nonoverlapping(self.cos[..m * half].as_ptr() as *const u8, pf.cosb_ptr, m * half * 4);
        std::ptr::copy_nonoverlapping(self.sin[..m * half].as_ptr() as *const u8, pf.sinb_ptr, m * half * 4);

        // Per-call uniforms + descriptor pool + command buffer (prefill is one call).
        let mut ub: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
        let uni = |ub: &mut Vec<(vk::Buffer, vk::DeviceMemory)>, d: [u32; 4]| -> vk::Buffer {
            let (b, mm, p) = self.ctx.uma_buffer(16).unwrap(); std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, p, 16); ub.push((b, mm)); b
        };
        let pool = dv.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets((self.n_layers * 20 + 8) as u32).pool_sizes(&[
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count((self.n_layers * 80 + 16) as u32),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count((self.n_layers * 20 + 8) as u32),
        ]), None).unwrap();
        let mkset = |sl: vk::DescriptorSetLayout, sb: &[vk::Buffer], u: vk::Buffer| vk_alloc_set(dv, pool, sl, sb, u);
        let cmd_pool = dv.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.ctx.queue_family), None).unwrap();
        let cmd = dv.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(1)).unwrap()[0];
        dv.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        let cs = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barr = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        let bar = || dv.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
        let disp = |p: Pipe3, set: vk::DescriptorSet, gx: u32, gy: u32| {
            dv.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.0);
            dv.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.1, 0, &[set], &[]);
            dv.cmd_dispatch(cmd, gx, gy, 1);
        };
        let c64 = |x: usize| ((x + 63) / 64) as u32;
        let u_eps = uni(&mut ub, [n_embd as u32, self.eps.to_bits(), 0, 0]);

        for l in &pf.w {
            // attn rmsnorm -> n32 -> f16
            disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.attn_norm, pf.n32], u_eps), m as u32, 1); bar();
            disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.n32, pf.n16], uni(&mut ub, [(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
            // QKV: wq->q (Q4), wk->kc (Q4), wv->vc (f16 dense). K=n_embd.
            disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, l.wq, pf.q], uni(&mut ub, [m as u32, n_embd as u32, n_embd as u32, (n_embd / 256) as u32])), (n_embd / 128) as u32, (m / 128) as u32);
            disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, l.wk, l.kc], uni(&mut ub, [m as u32, kv_dim as u32, n_embd as u32, (n_embd / 256) as u32])), (kv_dim / 128) as u32, (m / 128) as u32);
            disp(pf.p_f16, mkset(pf.p_f16.2, &[pf.n16, l.wv_f16, l.vc], uni(&mut ub, [m as u32, kv_dim as u32, n_embd as u32, 0])), (kv_dim / 64) as u32, (m / 64) as u32); bar();
            // RoPE q, k (k in the cache).
            disp(pf.p_br, mkset(pf.p_br.2, &[pf.q, pf.cosb, pf.sinb], uni(&mut ub, [n_head as u32, hd as u32, m as u32, 0])), c64(m * n_head * half), 1);
            disp(pf.p_br, mkset(pf.p_br.2, &[l.kc, pf.cosb, pf.sinb], uni(&mut ub, [n_kv as u32, hd as u32, m as u32, 0])), c64(m * n_kv * half), 1); bar();
            // causal SDPA -> attn
            disp(pf.p_bs, mkset(pf.p_bs.2, &[pf.q, l.kc, l.vc, pf.attn32], uni(&mut ub, [n_head as u32, n_kv as u32, hd as u32, m as u32])), c64(m * n_head), 1); bar();
            // Wo: attn->f16, GEMM->o32, residual x += o32
            disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.attn32, pf.attn16], uni(&mut ub, [(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
            disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.attn16, l.wo, pf.o32], uni(&mut ub, [m as u32, n_embd as u32, n_embd as u32, (n_embd / 256) as u32])), (n_embd / 128) as u32, (m / 128) as u32); bar();
            disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.o32], uni(&mut ub, [(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
            // ffn rmsnorm -> n32 -> f16
            disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, l.ffn_norm, pf.n32], u_eps), m as u32, 1); bar();
            disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.n32, pf.n16], uni(&mut ub, [(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
            // W13 GEMM -> gu [M, 2*n_inter] (Q4), silu -> h32 -> f16
            disp(pf.p_q4, mkset(pf.p_q4.2, &[pf.n16, l.w13, pf.gu], uni(&mut ub, [m as u32, (2 * n_inter) as u32, n_embd as u32, (n_embd / 256) as u32])), ((2 * n_inter) / 128) as u32, (m / 128) as u32); bar();
            disp(pf.p_bsi, mkset(pf.p_bsi.2, &[pf.gu, pf.h32], uni(&mut ub, [n_inter as u32, m as u32, 0, 0])), c64(m * n_inter), 1); bar();
            disp(pf.p_t16, mkset(pf.p_t16.2, &[pf.h32, pf.h16], uni(&mut ub, [(m * n_inter) as u32, 0, 0, 0])), c64(m * n_inter), 1); bar();
            // W2 GEMM (f16 dense, K=n_inter) -> ffn32, residual x += ffn32
            disp(pf.p_f16, mkset(pf.p_f16.2, &[pf.h16, l.w2_f16, pf.ffn32], uni(&mut ub, [m as u32, n_embd as u32, n_inter as u32, 0])), (n_embd / 64) as u32, (m / 64) as u32); bar();
            disp(pf.p_add, mkset(pf.p_add.2, &[pf.x32, pf.ffn32], uni(&mut ub, [(m * n_embd) as u32, 0, 0, 0])), c64(m * n_embd), 1); bar();
        }
        // final rmsnorm -> n32; LM head (Q6 matvec) on the last real token's row.
        disp(pf.p_bn, mkset(pf.p_bn.2, &[pf.x32, pf.final_norm, pf.n32], u_eps), m as u32, 1); bar();
        let lm_set = {
            let set = dv.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&pf.p_q6k.2))).unwrap()[0];
            let off = ((real_m - 1) * n_embd * 4) as u64;
            let infos = [
                (0u32, [vk::DescriptorBufferInfo::default().buffer(pf.lm_ql).range(vk::WHOLE_SIZE)]),
                (1, [vk::DescriptorBufferInfo::default().buffer(pf.lm_qh).range(vk::WHOLE_SIZE)]),
                (2, [vk::DescriptorBufferInfo::default().buffer(pf.lm_scl).range(vk::WHOLE_SIZE)]),
                (3, [vk::DescriptorBufferInfo::default().buffer(pf.lm_dd).range(vk::WHOLE_SIZE)]),
                (4, [vk::DescriptorBufferInfo::default().buffer(pf.n32).offset(off).range((n_embd * 4) as u64)]),
                (5, [vk::DescriptorBufferInfo::default().buffer(pf.logits).range(vk::WHOLE_SIZE)]),
            ];
            let ulm = uni(&mut ub, [vocab as u32, lm_nb as u32, (vocab as u32).min(65535), 0]);
            let uinfo = [vk::DescriptorBufferInfo::default().buffer(ulm).range(vk::WHOLE_SIZE)];
            let mut w: Vec<vk::WriteDescriptorSet> = infos.iter().map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            w.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uinfo));
            dv.update_descriptor_sets(&w, &[]);
            set
        };
        let gx = (vocab as u32).min(65535);
        disp(pf.p_q6k, lm_set, gx, (vocab as u32).div_ceil(gx));
        dv.end_command_buffer(cmd).unwrap();

        let fence = dv.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let cmds = [cmd];
        dv.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
        dv.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        let out = std::slice::from_raw_parts(pf.logits_ptr as *const f32, vocab).to_vec();
        dv.destroy_fence(fence, None);
        dv.destroy_command_pool(cmd_pool, None);
        dv.destroy_descriptor_pool(pool, None);
        for (b, mem) in ub { dv.unmap_memory(mem); dv.destroy_buffer(b, None); dv.free_memory(mem, None); }
        out
    }

    unsafe fn forward_inner(&self, token: u32, pos: usize, lm: bool, argmax: bool) -> (Vec<f32>, u32) {
        use ash::vk;
        let dv = &self.ctx.device;
        let (n_embd, n_head, n_kv, hd, n_inter, kv_dim, half) = (self.n_embd, self.n_head, self.n_kv, self.hd, self.n_inter, self.kv_dim, self.half);
        let attn_dim = n_head * hd;
        let seq_len = (pos + 1) as u32;
        // Update per-token mapped buffers.
        std::ptr::copy_nonoverlapping(self.embed[token as usize * n_embd..].as_ptr() as *const u8, self.x_ptr, n_embd * 4);
        std::ptr::copy_nonoverlapping(self.cos[pos * half..].as_ptr() as *const u8, self.cos_ptr, half * 4);
        std::ptr::copy_nonoverlapping(self.sin[pos * half..].as_ptr() as *const u8, self.sin_ptr, half * 4);
        std::ptr::copy_nonoverlapping([kv_dim as u32, (pos * kv_dim) as u32, 0u32, 0u32].as_ptr() as *const u8, self.base_ptr, 16);
        std::ptr::copy_nonoverlapping([n_head as u32, n_kv as u32, hd as u32, seq_len].as_ptr() as *const u8, self.seq_ptr, 16);

        // Record-once: only re-record when the SDPA grid (single-pass vs flash
        // n_blocks) or the lm/argmax tail changes; otherwise reuse self.cmd.
        let sdpa_key = if (seq_len as usize) > SDPA_FLASH_BLOCK { (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK) as i64 } else { 0 };
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
        let mv = |m: Mv, n: usize| disp(if m.q6 { self.p_q6k } else { self.p_mv }, m.set, gxof(n), (n as u32).div_ceil(gxof(n)));
        let mvonly = std::env::var("VK_MVONLY").is_ok();      // diag: matvecs only (skip all small ops)
        let skip_extra = mvonly || std::env::var("VK_NOEXTRA").is_ok(); // diag: skip kvwrite+residual
        let skip_norm = mvonly || std::env::var("VK_NONORM").is_ok();   // diag: skip rmsnorms
        let skip_attn = mvonly;                               // diag: skip rope+sdpa
        for l in &self.layers {
            if !skip_norm { disp(self.p_rms, l.attn_norm, 1, 1); bar(); }              // attn norm
            mv(l.wq, n_embd); mv(l.wk, kv_dim); mv(l.wv, kv_dim); bar();               // QKV
            if !skip_attn {
            disp(self.p_rope, self.s_rope_q, ((n_head * half) as u32).div_ceil(64), 1);
            disp(self.p_rope, self.s_rope_k, ((n_kv * half) as u32).div_ceil(64), 1); bar(); // RoPE q,k
            if !skip_extra { disp(self.p_kvw, l.kvw_k, (kv_dim as u32).div_ceil(64), 1);
            disp(self.p_kvw, l.kvw_v, (kv_dim as u32).div_ceil(64), 1); bar(); }       // append K,V to cache
            if seq_len as usize > SDPA_FLASH_BLOCK {
                let nblk = (seq_len as usize).div_ceil(SDPA_FLASH_BLOCK) as u32;
                disp(self.p_fp, l.fp, n_head as u32, nblk); bar();
                disp(self.p_fc, l.fc, n_head as u32, 1); bar();
            } else {
                disp(self.p_sdpa, l.sdpa, n_head as u32, 1); bar();
            }
            }
            mv(l.wo, n_embd); bar();                                                   // O proj
            if !skip_extra { disp(self.p_add, l.radd_a, (n_embd as u32).div_ceil(64), 1); bar(); } // x += attn out
            if !skip_norm { disp(self.p_rms, l.ffn_norm, 1, 1); bar(); }               // ffn norm
            mv(l.w13, n_inter * 2); bar();                                             // gate+up (concat, one matvec)
            if !skip_attn { disp(self.p_silu, self.s_silu, (n_inter as u32).div_ceil(64), 1); bar(); } // silu·mul
            mv(l.w2, n_embd); bar();                                                   // down proj
            if !skip_extra { disp(self.p_add, l.radd_f, (n_embd as u32).div_ceil(64), 1); bar(); } // x += ffn out
        }
        if lm {
            if !skip_norm { disp(self.p_rms, self.s_final_norm, 1, 1); bar(); }        // final norm
            disp(self.p_q6k, self.s_lm, gxof(self.vocab), (self.vocab as u32).div_ceil(gxof(self.vocab))); // LM head
            if argmax { bar(); disp(self.p_argmax, self.s_argmax, 1, 1); }              // GPU argmax (4-byte readback)
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
        for &m in &[256usize, 512, 2048] {
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
        use candle_core::{Device, Tensor};
        use candle_core::quantized::{QTensor, GgmlDType};
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("no Vulkan coopmat device ({e}); skipping"); return; }
        };
        unsafe { fused_decode_inner(&ctx); }
    }

    /// END-TO-END: load the real Llama-3.2-1B GGUF into the VkModel and check
    /// greedy decode matches candle CPU token-for-token (the engine the server
    /// will use). Also prints decode tok/s on real weights.
    /// `cargo test --release --features vulkan --lib vk_model_vs_candle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_model_vs_candle() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found at {path}; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan device ({e}); skipping"); return; } };
        let t = Instant::now();
        let model = VkModel::load(path, ctx).expect("load");
        eprintln!("VkModel loaded in {:.2}s (vocab {})", t.elapsed().as_secs_f64(), model.vocab);
        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let prompt: Vec<u32> = vec![128000]; // BOS
        let n_gen = 24usize;
        let _ = &argmax;
        let n_time = 128usize; // match llama-bench tg128 (avg ctx ~64) for a fair rate
        let mut next = 0u32;
        for (i, &tk) in prompt.iter().enumerate() { next = model.forward_argmax(tk, i); }
        let mut vk_gen = vec![next];
        let mut pos = prompt.len();
        let t0 = Instant::now();
        for _ in 1..n_time { next = model.forward_argmax(next, pos); vk_gen.push(next); pos += 1; }
        let dt = t0.elapsed();
        eprintln!("VkModel decode: {:.1} tok/s over {} tokens (avg ctx ~{})", (n_time - 1) as f64 / dt.as_secs_f64(), n_time - 1, n_time / 2);
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

    /// BATCHED PREFILL vs candle: prefill a multi-token prompt through the GEMM
    /// path, check the last-token logits (cosine + argmax) match candle, then a
    /// few decode steps continue correctly from pos=M. Validates the whole
    /// batched forward (norm/QKV-GEMM/rope/causal-sdpa/wo/FFN/KV-fill).
    /// `cargo test --release --features vulkan --lib vk_prefill_vs_candle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn vk_prefill_vs_candle() {
        use std::time::Instant;
        let path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() { eprintln!("model not found; skipping"); return; }
        let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no Vulkan ({e}); skipping"); return; } };
        let model = VkModel::load(path, ctx).expect("load");
        let argmax = |v: &[f32]| -> u32 { let mut bi = 0u32; let mut bv = f32::MIN; for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } } bi };
        let prompt: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374]; // "The capital of France is"
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
}

#[cfg(test)]
unsafe fn sdpa_correctness_inner(ctx: &VkContext) {
    // Exercise both paths: single-pass (short ctx) and flash 2-pass (long ctx,
    // multiple KV blocks), each vs a CPU softmax-attention reference.
    sdpa_case(ctx, 32, false);
    sdpa_case(ctx, 520, true); // > SDPA_FLASH_BLOCK and not a block multiple
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
    let rnd = |i: usize, s: usize| (((i.wrapping_mul(2654435761).wrapping_add(s.wrapping_mul(40503))) & 0xFFFF) as f32 / 32768.0 - 1.0);
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
