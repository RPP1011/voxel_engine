//! GPU simulation harness — taichi-style CPU↔GPU data transfer + compute dispatch.
//!
//! ```ignore
//! let ctx = VulkanContext::new()?;
//! let mut harness = GpuHarness::new(&ctx)?;
//!
//! let positions = harness.create_field::<[f32; 4]>(1024)?;
//! harness.upload(&ctx, &positions, &initial_data)?;
//! harness.load_kernel(&ctx, "integrate", spirv_bytes, 2)?;
//! harness.dispatch(&ctx, "integrate", &[&positions, &velocities], [4, 1, 1])?;
//! let result: Vec<[f32; 4]> = harness.download(&ctx, &positions)?;
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use ash::vk;
use bytemuck::Pod;

use crate::vulkan::instance::VulkanContext;

/// Handle to a GPU field (storage buffer).
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct FieldHandle(u32);

/// A host-visible storage buffer managed by the harness.
struct Field {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
    size_bytes: u64,
}

struct CachedKernel {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
    binding_count: u32,
    push_constant_size: u32,
}

/// GPU compute harness for simulation kernels.
pub struct GpuHarness {
    fields: HashMap<u32, Field>,
    kernels: HashMap<String, CachedKernel>,
    next_id: u32,
    command_pool: vk::CommandPool,
    queue_family: u32,
}

impl GpuHarness {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let cq = ctx.compute_queue().context("No compute queue")?;
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(cq.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { ctx.device().create_command_pool(&pool_ci, None) }?;

        Ok(Self {
            fields: HashMap::new(),
            kernels: HashMap::new(),
            next_id: 0,
            command_pool,
            queue_family: cq.family_index,
        })
    }

    /// Create a host-visible storage buffer for `count` elements of type T.
    pub fn create_field<T: Pod>(&mut self, ctx: &VulkanContext, count: usize) -> Result<FieldHandle> {
        let size_bytes = (std::mem::size_of::<T>() * count) as u64;
        let device = ctx.device();

        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_ci, None) }?;
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find host-visible, host-coherent memory type
        let mem_props = unsafe {
            ctx.instance()
                .get_physical_device_memory_properties(ctx.physical_device())
        };
        let mem_type_idx = find_memory_type(
            &mem_props,
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("No suitable host-visible memory type")?;

        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_idx);
        let memory = unsafe { device.allocate_memory(&alloc_ci, None) }?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        let mapped_ptr = unsafe {
            device.map_memory(memory, 0, size_bytes, vk::MemoryMapFlags::empty())
        }? as *mut u8;

        // Zero-initialize
        unsafe {
            std::ptr::write_bytes(mapped_ptr, 0, size_bytes as usize);
        }

        let id = self.next_id;
        self.next_id += 1;
        self.fields.insert(id, Field {
            buffer,
            memory,
            mapped_ptr,
            size_bytes,
        });

        Ok(FieldHandle(id))
    }

    /// Upload data from CPU slice to GPU field.
    pub fn upload<T: Pod>(&self, handle: &FieldHandle, data: &[T]) -> Result<()> {
        let field = self.fields.get(&handle.0).context("Field not found")?;
        let bytes = bytemuck::cast_slice(data);
        if bytes.len() as u64 > field.size_bytes {
            bail!("Data ({} bytes) exceeds field size ({} bytes)", bytes.len(), field.size_bytes);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), field.mapped_ptr, bytes.len());
        }
        Ok(())
    }

    /// Download data from GPU field to CPU Vec.
    pub fn download<T: Pod + Copy>(&self, handle: &FieldHandle, count: usize) -> Result<Vec<T>> {
        let field = self.fields.get(&handle.0).context("Field not found")?;
        let byte_count = std::mem::size_of::<T>() * count;
        if byte_count as u64 > field.size_bytes {
            bail!("Requested {} bytes but field is {} bytes", byte_count, field.size_bytes);
        }
        let mut result = vec![T::zeroed(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                field.mapped_ptr,
                result.as_mut_ptr() as *mut u8,
                byte_count,
            );
        }
        Ok(result)
    }

    /// Load and cache a compute kernel from SPIR-V bytes.
    pub fn load_kernel(
        &mut self,
        ctx: &VulkanContext,
        name: &str,
        spirv: &[u8],
        binding_count: u32,
    ) -> Result<()> {
        self.load_kernel_with_push(ctx, name, spirv, binding_count, 0)
    }

    /// Load and cache a compute kernel with push constant support.
    pub fn load_kernel_with_push(
        &mut self,
        ctx: &VulkanContext,
        name: &str,
        spirv: &[u8],
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<()> {
        if self.kernels.contains_key(name) {
            return Ok(());
        }

        let device = ctx.device();
        let spirv_words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe { device.create_shader_module(&shader_ci, None) }?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_ci, None) }?;

        let set_layouts = [descriptor_set_layout];
        let push_ranges = if push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constant_size,
            }]
        } else {
            vec![]
        };
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None) }?;

        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .map_err(|(_, e)| e)?[0];

        self.kernels.insert(name.to_string(), CachedKernel {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            shader_module,
            binding_count,
            push_constant_size,
        });

        Ok(())
    }

    /// Dispatch a compute kernel with bound fields.
    pub fn dispatch(
        &self,
        ctx: &VulkanContext,
        kernel_name: &str,
        fields: &[&FieldHandle],
        groups: [u32; 3],
    ) -> Result<()> {
        let kernel = self.kernels.get(kernel_name).context("Kernel not loaded")?;
        let device = ctx.device();
        let cq = ctx.compute_queue().context("No compute queue")?;

        // Descriptor pool + set
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(kernel.binding_count);
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }?;

        let set_layouts = [kernel.descriptor_set_layout];
        let alloc_ci = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci) }?;

        // Bind fields to descriptors
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = fields
            .iter()
            .map(|fh| {
                let field = &self.fields[&fh.0];
                vk::DescriptorBufferInfo::default()
                    .buffer(field.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_sets[0])
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // Record + submit
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &desc_sets,
                &[],
            );
            device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            device.end_command_buffer(cmd)?;
        }

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(cq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(())
    }

    /// Dispatch a compute kernel with push constants.
    pub fn dispatch_push(
        &self,
        ctx: &VulkanContext,
        kernel_name: &str,
        fields: &[&FieldHandle],
        push_data: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let kernel = self.kernels.get(kernel_name).context("Kernel not loaded")?;
        let device = ctx.device();
        let cq = ctx.compute_queue().context("No compute queue")?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(kernel.binding_count.max(1));
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }?;

        let set_layouts = [kernel.descriptor_set_layout];
        let alloc_ci = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci) }?;

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = fields
            .iter()
            .map(|fh| {
                let field = &self.fields[&fh.0];
                vk::DescriptorBufferInfo::default()
                    .buffer(field.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_sets[0])
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline_layout,
                0, &desc_sets, &[],
            );
            if !push_data.is_empty() {
                device.cmd_push_constants(
                    cmd, kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, push_data,
                );
            }
            device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[barrier], &[], &[],
            );

            device.end_command_buffer(cmd)?;
        }

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(cq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(())
    }

    /// Get the raw Vulkan buffer for a field (for clearing etc.).
    pub fn field_buffer(&self, handle: &FieldHandle) -> Option<vk::Buffer> {
        self.fields.get(&handle.0).map(|f| f.buffer)
    }

    /// Zero-fill a field's memory.
    pub fn clear_field(&self, handle: &FieldHandle) -> Result<()> {
        let field = self.fields.get(&handle.0).context("Field not found")?;
        unsafe {
            std::ptr::write_bytes(field.mapped_ptr, 0, field.size_bytes as usize);
        }
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            for field in self.fields.values() {
                device.unmap_memory(field.memory);
                device.destroy_buffer(field.buffer, None);
                device.free_memory(field.memory, None);
            }
            for kernel in self.kernels.values() {
                device.destroy_pipeline(kernel.pipeline, None);
                device.destroy_pipeline_layout(kernel.pipeline_layout, None);
                device.destroy_descriptor_set_layout(kernel.descriptor_set_layout, None);
                device.destroy_shader_module(kernel.shader_module, None);
            }
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
        {
            return Some(i);
        }
    }
    None
}
