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
const RMSNORM_SPV: &[u8] = include_bytes!("shaders/rmsnorm.spv");
const ROPE_SPV: &[u8] = include_bytes!("shaders/rope.spv");
const SDPA_DECODE_SPV: &[u8] = include_bytes!("shaders/sdpa_decode.spv");
const SILU_MUL_SPV: &[u8] = include_bytes!("shaders/silu_mul.spv");

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
    let seq_len = 32u32; // realistic short-context decode depth
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
    let attn = buf_f32(n_embd); let gate = buf_f32(n_inter); let up = buf_f32(n_inter); let h = buf_f32(n_inter); let logits = buf_f32(vocab);
    let kc = buf_f32(max_seq * kv_dim); let vc = buf_f32(max_seq * kv_dim);

    // Pipelines (storage-buffer count per kernel).
    let (mv_p, mv_l, mv_sl, _m0) = ctx.make_pipeline_raw(DECODE_MATVEC_Q4K_SPV, 3);
    let (rn_p, rn_l, rn_sl, _m1) = ctx.make_pipeline_raw(RMSNORM_SPV, 3);
    let (ro_p, ro_l, ro_sl, _m2) = ctx.make_pipeline_raw(ROPE_SPV, 3);
    let (sd_p, sd_l, sd_sl, _m3) = ctx.make_pipeline_raw(SDPA_DECODE_SPV, 4);
    let (si_p, si_l, si_sl, _m4) = ctx.make_pipeline_raw(SILU_MUL_SPV, 3);

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
    let s_wo = mkset(mv_sl, &[wo, attn, x], mvuni(n_embd, n_embd));
    let s_w1 = mkset(mv_sl, &[w1, normed, gate], mvuni(n_inter, n_embd));
    let s_w3 = mkset(mv_sl, &[w3, normed, up], mvuni(n_inter, n_embd));
    let s_si = mkset(si_sl, &[gate, up, h], uni([n_inter as u32, 0, 0, 0]));
    let s_w2 = mkset(mv_sl, &[w2, h, x], mvuni(n_embd, n_inter));
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
    let bar = || dev.cmd_pipeline_barrier(cmd, cs, cs, vk::DependencyFlags::empty(), &[barr], &[], &[]);
    let mv = |set, n: usize| disp(mv_p, mv_l, set, gxof(n), (n as u32).div_ceil(gxof(n)));
    for _ in 0..n_layers {
        disp(rn_p, rn_l, s_rn, 1, 1); bar();                                          // attn norm
        mv(s_wq, n_embd); mv(s_wk, kv_dim); mv(s_wv, kv_dim); bar();                   // QKV
        disp(ro_p, ro_l, s_rq, ((n_head * hd / 2) as u32).div_ceil(64), 1);
        disp(ro_p, ro_l, s_rk, ((n_kv * hd / 2) as u32).div_ceil(64), 1); bar();       // RoPE q,k
        disp(sd_p, sd_l, s_sd, (n_head as u32).div_ceil(64), 1); bar();                // SDPA
        mv(s_wo, n_embd); bar();                                                       // O proj
        disp(rn_p, rn_l, s_rn, 1, 1); bar();                                          // ffn norm
        mv(s_w1, n_inter); mv(s_w3, n_inter); bar();                                   // gate, up
        disp(si_p, si_l, s_si, (n_inter as u32).div_ceil(64), 1); bar();              // silu·mul
        mv(s_w2, n_embd); bar();                                                       // down proj
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
    eprintln!("FUSED decode forward: {per:.3} ms/token => {:.0} tok/s", 1000.0 / per);
    eprintln!("  vs current wgpu decode ~80 tok/s; llama.cpp iGPU 201; matvec-only projection 237");
}
