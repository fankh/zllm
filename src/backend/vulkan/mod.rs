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
}
