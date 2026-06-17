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
        assert!(m % 64 == 0 && n % 64 == 0);
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
        for (buf, mem) in [(a_buf, a_mem), (w_buf, w_mem), (c_buf, c_mem), (p_buf, p_mem)] {
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
}
