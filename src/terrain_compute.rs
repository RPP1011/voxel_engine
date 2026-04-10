//! GPU terrain materialization pipeline.
//!
//! Dispatches a compute shader that fills a chunk's worth of voxel materials
//! based on chunk position, seed, and (eventually) region plan data.

use anyhow::{Context, Result};
use ash::vk;

use crate::vulkan::allocator::{AllocatedBuffer, VulkanAllocator};
use crate::vulkan::instance::VulkanContext;

const CHUNK_SIZE: u32 = 64;
const CHUNK_VOLUME: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const OUTPUT_BYTES: u64 = CHUNK_VOLUME as u64; // 1 byte per voxel

/// GPU layout of a single region plan cell. Mirrors the GLSL `RegionCell`
/// struct in `shaders/terrain_materialize.comp`. Uses std430 layout with
/// 32-byte stride (vec4-aligned) to keep the shader side simple.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RegionCellGpu {
    pub height: f32,
    pub moisture: f32,
    pub temperature: f32,
    pub terrain: u32,
    pub sub_biome: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// GPU header describing region plan dimensions. Mirrors the GLSL
/// `RegionHeaderBuf` layout; 16-byte total (vec4-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RegionPlanHeader {
    pub cols: u32,
    pub rows: u32,
    pub cell_size: u32,
    pub _pad: u32,
}

pub struct TerrainComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    output_buffer: AllocatedBuffer,
    region_cells_buffer: Option<AllocatedBuffer>,
    region_header_buffer: Option<AllocatedBuffer>,
    placeholder_cells_buffer: AllocatedBuffer,
    placeholder_header_buffer: AllocatedBuffer,
    shader_module: vk::ShaderModule,
}

impl TerrainComputePipeline {
    pub fn new(ctx: &VulkanContext, alloc: &mut VulkanAllocator) -> Result<Self> {
        let device = ctx.device();

        // Load precompiled SPIR-V from OUT_DIR/shaders.
        let spirv_bytes = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/terrain_materialize.comp.spv"
        ));
        let spirv_words: Vec<u32> = spirv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe { device.create_shader_module(&shader_ci, None) }
            .context("create terrain compute shader")?;

        // Descriptor layout: three STORAGE_BUFFER bindings.
        //   binding 0 = output voxel buffer
        //   binding 1 = region plan cells
        //   binding 2 = region plan header
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_ci, None) }
                .context("descriptor set layout")?;

        // Push constants: ivec4 + uvec4 = 32 bytes.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(32);

        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None) }
            .context("pipeline layout")?;

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
        .map_err(|(_, e)| e)
        .context("compute pipeline")?[0];

        // Descriptor pool with three storage buffer descriptors.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .context("descriptor pool")?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("alloc descriptor set")?[0];

        // Output buffer: host-visible so we can read back without an extra copy.
        let output_buffer = alloc
            .allocate_host_visible_buffer(OUTPUT_BYTES)
            .context("output buffer")?;

        // Placeholder buffers for bindings 1 and 2 so the descriptor set is
        // always valid (prevents shader crashes before upload_region_plan).
        // Size must be >= one RegionCellGpu (32 B) / one RegionPlanHeader (16 B).
        let mut placeholder_cells_buffer = alloc
            .allocate_host_visible_buffer(std::mem::size_of::<RegionCellGpu>() as u64)
            .context("placeholder cells buffer")?;
        // Write a zeroed RegionCellGpu (terrain=0=Plains) so the shader gets
        // sane defaults if it's ever dispatched before a real upload.
        let zero_cell = RegionCellGpu {
            height: 0.3,
            moisture: 0.5,
            temperature: 0.5,
            terrain: 0,
            sub_biome: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let zero_cell_bytes = unsafe {
            std::slice::from_raw_parts(
                &zero_cell as *const _ as *const u8,
                std::mem::size_of::<RegionCellGpu>(),
            )
        };
        placeholder_cells_buffer
            .write_data(0, zero_cell_bytes)
            .context("write placeholder cells")?;

        let mut placeholder_header_buffer = alloc
            .allocate_host_visible_buffer(std::mem::size_of::<RegionPlanHeader>() as u64)
            .context("placeholder header buffer")?;
        let zero_header = RegionPlanHeader {
            cols: 1,
            rows: 1,
            cell_size: 32,
            _pad: 0,
        };
        let zero_header_bytes = unsafe {
            std::slice::from_raw_parts(
                &zero_header as *const _ as *const u8,
                std::mem::size_of::<RegionPlanHeader>(),
            )
        };
        placeholder_header_buffer
            .write_data(0, zero_header_bytes)
            .context("write placeholder header")?;

        // Bind all three buffers to the descriptor set.
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.buffer())
            .offset(0)
            .range(OUTPUT_BYTES)];
        let cells_info = [vk::DescriptorBufferInfo::default()
            .buffer(placeholder_cells_buffer.buffer())
            .offset(0)
            .range(std::mem::size_of::<RegionCellGpu>() as u64)];
        let header_info = [vk::DescriptorBufferInfo::default()
            .buffer(placeholder_header_buffer.buffer())
            .offset(0)
            .range(std::mem::size_of::<RegionPlanHeader>() as u64)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&cells_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&header_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            output_buffer,
            region_cells_buffer: None,
            region_header_buffer: None,
            placeholder_cells_buffer,
            placeholder_header_buffer,
            shader_module,
        })
    }

    /// Upload (or re-upload) the region plan cells + header to GPU storage
    /// buffers and update the descriptor set bindings 1/2 to point at them.
    /// Caller is responsible for invoking this once the plan is known, and
    /// again if the plan is replaced.
    pub fn upload_region_plan(
        &mut self,
        ctx: &VulkanContext,
        alloc: &mut VulkanAllocator,
        cols: u32,
        rows: u32,
        cell_size: u32,
        cells: &[RegionCellGpu],
    ) -> Result<()> {
        if cells.len() as u32 != cols * rows {
            anyhow::bail!(
                "cells.len() ({}) != cols*rows ({})",
                cells.len(),
                cols * rows
            );
        }

        // 1. Free any previously-uploaded buffers.
        if let Some(old) = self.region_cells_buffer.take() {
            alloc.free_buffer(old);
        }
        if let Some(old) = self.region_header_buffer.take() {
            alloc.free_buffer(old);
        }

        // 2. Allocate + fill cells buffer.
        let cells_size = (cells.len() * std::mem::size_of::<RegionCellGpu>()) as u64;
        let mut cells_buf = alloc
            .allocate_host_visible_buffer(cells_size)
            .context("region cells buffer")?;
        let cells_bytes = unsafe {
            std::slice::from_raw_parts(cells.as_ptr() as *const u8, cells_size as usize)
        };
        cells_buf
            .write_data(0, cells_bytes)
            .context("write region cells")?;

        // 3. Allocate + fill header buffer.
        let header = RegionPlanHeader {
            cols,
            rows,
            cell_size,
            _pad: 0,
        };
        let header_size = std::mem::size_of::<RegionPlanHeader>() as u64;
        let mut header_buf = alloc
            .allocate_host_visible_buffer(header_size)
            .context("region header buffer")?;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                header_size as usize,
            )
        };
        header_buf
            .write_data(0, header_bytes)
            .context("write region header")?;

        // 4. Update descriptor set bindings 1 and 2 to point at the real
        //    buffers. Safe to do because we wait on every dispatch before
        //    returning — no in-flight use of the descriptor set.
        let device = ctx.device();
        let cells_info = [vk::DescriptorBufferInfo::default()
            .buffer(cells_buf.buffer())
            .offset(0)
            .range(cells_size)];
        let header_info = [vk::DescriptorBufferInfo::default()
            .buffer(header_buf.buffer())
            .offset(0)
            .range(header_size)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&cells_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&header_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.region_cells_buffer = Some(cells_buf);
        self.region_header_buffer = Some(header_buf);
        Ok(())
    }

    /// Dispatch the compute shader for one chunk and return the materials as a flat Vec.
    /// Index ordering: `[z * cs * cs + y * cs + x]`.
    pub fn generate_chunk(
        &self,
        ctx: &VulkanContext,
        chunk_pos: [i32; 3],
        seed: u32,
    ) -> Result<Vec<u8>> {
        let device = ctx.device();
        let cq = ctx.compute_queue().context("no compute queue")?;

        // One-shot command buffer.
        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(cq.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }?;
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }?[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin)?;
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // Push constants: ivec4 chunk_pos, uvec4 params (seed, chunk_size, _, _).
            let mut push = [0u8; 32];
            push[0..4].copy_from_slice(&chunk_pos[0].to_le_bytes());
            push[4..8].copy_from_slice(&chunk_pos[1].to_le_bytes());
            push[8..12].copy_from_slice(&chunk_pos[2].to_le_bytes());
            push[16..20].copy_from_slice(&seed.to_le_bytes());
            push[20..24].copy_from_slice(&CHUNK_SIZE.to_le_bytes());
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );

            // Dispatch: shader uses 16×8×8 local size and processes cs/4 strips in x.
            // Workgroups: ceil(cs_words/16) × ceil(cs/8) × ceil(cs/8).
            // For cs=64: cs_words=16, so 1 × 8 × 8 = 64 workgroups.
            let cs_words = CHUNK_SIZE / 4;
            let groups_x = (cs_words + 15) / 16;
            let groups_y = (CHUNK_SIZE + 7) / 8;
            let groups_z = (CHUNK_SIZE + 7) / 8;
            device.cmd_dispatch(cmd, groups_x, groups_y, groups_z);

            // Memory barrier so host can read.
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

        // Submit and wait via fence.
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(cq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }

        // Read back the buffer.
        let bytes = self.output_buffer.read_data(0, OUTPUT_BYTES as usize)?;
        Ok(bytes)
    }

    pub fn destroy(self, ctx: &VulkanContext, alloc: &mut VulkanAllocator) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
        alloc.free_buffer(self.output_buffer);
        alloc.free_buffer(self.placeholder_cells_buffer);
        alloc.free_buffer(self.placeholder_header_buffer);
        if let Some(b) = self.region_cells_buffer {
            alloc.free_buffer(b);
        }
        if let Some(b) = self.region_header_buffer {
            alloc.free_buffer(b);
        }
    }
}
