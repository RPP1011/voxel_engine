//! GPU terrain materialization pipeline.
//!
//! Dispatches a compute shader that fills a chunk's worth of voxel materials
//! based on chunk position, seed, and region plan data.
//!
//! Maintains a pool of N chunk texture slots. Each slot owns a 3D R8_UINT
//! storage image that the compute shader writes directly into via imageStore,
//! plus the per-dispatch Vulkan resources (cmd pool, cmd, fence, descriptor
//! set). Slots stay GPU-resident after a dispatch completes so the renderer
//! can sample them directly; eviction is LRU based on `last_touched_frame`.

use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::vulkan::allocator::{AllocatedBuffer, VulkanAllocator};
use crate::vulkan::instance::VulkanContext;

const CHUNK_SIZE: u32 = 64;
const CHUNK_VOLUME: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const OUTPUT_BYTES: u64 = CHUNK_VOLUME as u64; // 1 byte per voxel

/// Number of chunk texture pool slots. 64³ R8_UINT = 256KB each × 256 = 64MB
/// of GPU memory dedicated to the pool.
const NUM_SLOTS: usize = 256;

/// Key used to identify a chunk in the pool — voxel chunk coordinates.
pub type ChunkKey = [i32; 3];

/// Lifecycle state of a single chunk texture pool slot.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SlotState {
    /// Ready to be assigned a new dispatch.
    Free,
    /// Dispatch submitted; waiting for fence. Carries the request id.
    InFlight(u64),
    /// Image holds valid chunk data, ready for render sampling.
    Loaded(ChunkKey),
}

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

/// GPU layout of a single river polyline point. Mirrors the GLSL
/// `RiverPoint` struct. std430: 16 bytes (vec4-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RiverPointGpu {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub _pad: f32,
}

/// GPU header describing a river's range in the flat points buffer.
/// Mirrors the GLSL `RiverHeader` struct. std430: 16 bytes (uvec4-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RiverHeaderGpu {
    pub start_idx: u32,
    pub length: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// One chunk texture pool slot. Each slot owns a full set of per-dispatch
/// Vulkan resources (image, cmd pool, fence, descriptor set) and has a
/// lifecycle `Free → InFlight → Loaded → Free (on eviction)`.
struct ComputeSlot {
    /// 3D R8_UINT image — compute shader writes here directly via imageStore.
    /// Usage: STORAGE (compute write) + SAMPLED (renderer read) +
    /// TRANSFER_SRC (sync readback path used only by tests).
    output_image: vk::Image,
    output_image_view: vk::ImageView,
    output_image_alloc: Option<Allocation>,
    /// Host-visible staging buffer used by the sync `generate_chunk` test
    /// wrapper and the deprecated `try_take_completed_with_bytes` shim. The
    /// normal fast path (renderer samples directly) never touches this.
    readback_buffer: AllocatedBuffer,
    descriptor_set: vk::DescriptorSet,
    fence: vk::Fence,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    state: SlotState,
    /// Frame index at which this slot was last referenced. Used for LRU
    /// eviction — submit_chunk picks the oldest Loaded slot when the pool
    /// is full of Loaded entries.
    last_touched_frame: u64,
    /// True until the first dispatch — used to pick UNDEFINED vs
    /// SHADER_READ_ONLY_OPTIMAL as the old layout for the pre-dispatch barrier.
    first_use: bool,
    /// Chunk this slot currently represents. Set when `state` transitions
    /// from Free → InFlight and kept until the slot is re-used. Valid
    /// whenever `state != Free`.
    chunk_pos: ChunkKey,
}

/// Create a CHUNK_SIZE³ R8_UINT 3D image for compute-write + sample + xfer.
fn create_chunk_storage_image(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
) -> Result<(vk::Image, vk::ImageView, Allocation)> {
    let device = ctx.device();
    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .format(vk::Format::R8_UINT)
        .extent(vk::Extent3D {
            width: CHUNK_SIZE,
            height: CHUNK_SIZE,
            depth: CHUNK_SIZE,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_ci, None) }
        .context("create terrain storage image")?;
    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let allocation = alloc.allocate_image_memory(mem_req)?;
    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .context("bind terrain image memory")?;
    }

    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(vk::Format::R8_UINT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_ci, None) }
        .context("create terrain storage image view")?;

    Ok((image, view, allocation))
}

pub struct TerrainComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    shader_module: vk::ShaderModule,
    // Ring of in-flight slots.
    slots: Vec<ComputeSlot>,
    next_request_id: u64,
    // Region plan + rivers (shared across all slots; re-bound to every slot's
    // descriptor set whenever they are uploaded).
    region_cells_buffer: Option<AllocatedBuffer>,
    region_header_buffer: Option<AllocatedBuffer>,
    placeholder_cells_buffer: AllocatedBuffer,
    placeholder_header_buffer: AllocatedBuffer,
    river_points_buffer: Option<AllocatedBuffer>,
    river_headers_buffer: Option<AllocatedBuffer>,
    placeholder_river_points: AllocatedBuffer,
    placeholder_river_headers: AllocatedBuffer,
}

impl TerrainComputePipeline {
    pub fn new(ctx: &VulkanContext, alloc: &mut VulkanAllocator) -> Result<Self> {
        let device = ctx.device();
        let cq = ctx.compute_queue().context("no compute queue")?;

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

        // Descriptor layout:
        //   binding 0 = output voxel 3D storage image (R8_UINT)
        //   binding 1 = region plan cells           (storage buffer)
        //   binding 2 = region plan header          (storage buffer)
        //   binding 3 = river points (flat)         (storage buffer)
        //   binding 4 = river headers               (storage buffer)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
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

        let set_layouts_one = [descriptor_set_layout];
        let push_ranges = [push_range];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_one)
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

        // Descriptor pool: NUM_SLOTS sets × (1 storage image + 4 storage buffers).
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(NUM_SLOTS as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((NUM_SLOTS * 4) as u32),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(NUM_SLOTS as u32);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .context("descriptor pool")?;

        // Allocate NUM_SLOTS descriptor sets in one call (same layout for each).
        let set_layouts_all: Vec<vk::DescriptorSetLayout> =
            vec![descriptor_set_layout; NUM_SLOTS];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts_all);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("alloc descriptor sets")?;

        // --- Placeholder buffers (shared, rebound onto every slot) -----------

        let mut placeholder_cells_buffer = alloc
            .allocate_host_visible_buffer(std::mem::size_of::<RegionCellGpu>() as u64)
            .context("placeholder cells buffer")?;
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

        let mut placeholder_river_points = alloc
            .allocate_host_visible_buffer(std::mem::size_of::<RiverPointGpu>() as u64)
            .context("placeholder river points buffer")?;
        let zero_river_pt = RiverPointGpu {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            _pad: 0.0,
        };
        let zero_river_pt_bytes = unsafe {
            std::slice::from_raw_parts(
                &zero_river_pt as *const _ as *const u8,
                std::mem::size_of::<RiverPointGpu>(),
            )
        };
        placeholder_river_points
            .write_data(0, zero_river_pt_bytes)
            .context("write placeholder river point")?;

        let mut placeholder_river_headers = alloc
            .allocate_host_visible_buffer(std::mem::size_of::<RiverHeaderGpu>() as u64)
            .context("placeholder river headers buffer")?;
        let zero_river_hdr = RiverHeaderGpu {
            start_idx: 0,
            length: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let zero_river_hdr_bytes = unsafe {
            std::slice::from_raw_parts(
                &zero_river_hdr as *const _ as *const u8,
                std::mem::size_of::<RiverHeaderGpu>(),
            )
        };
        placeholder_river_headers
            .write_data(0, zero_river_hdr_bytes)
            .context("write placeholder river header")?;

        // --- Per-slot resources ---------------------------------------------

        let mut slots: Vec<ComputeSlot> = Vec::with_capacity(NUM_SLOTS);
        for slot_idx in 0..NUM_SLOTS {
            // 3D storage image (GPU-local) that the compute shader writes to
            // via imageStore. Stays resident between dispatches; Phase 3 will
            // let the renderer sample it directly without the readback path.
            let (output_image, output_image_view, output_image_alloc) =
                create_chunk_storage_image(ctx, alloc)
                    .context("slot output image")?;

            // Host-visible staging buffer for Phase 1 parity readback. The
            // cmd_copy_image_to_buffer in submit_chunk fills this; Phase 3 can
            // drop it entirely.
            let readback_buffer = alloc
                .allocate_host_visible_buffer(OUTPUT_BYTES)
                .context("slot readback buffer")?;

            // Per-slot command pool so we can reset it independently.
            let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(cq.family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }
                .context("slot command pool")?;

            let cmd_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }
                .context("slot command buffer")?[0];

            // Start fences UNSIGNALED; we'll reset+signal on first submit.
            let fence = unsafe {
                device.create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .context("slot fence")?;

            let descriptor_set = descriptor_sets[slot_idx];

            // Initial descriptor writes: output → slot's own storage image,
            // region + river bindings → placeholders.
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(output_image_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let cells_info = [vk::DescriptorBufferInfo::default()
                .buffer(placeholder_cells_buffer.buffer())
                .offset(0)
                .range(std::mem::size_of::<RegionCellGpu>() as u64)];
            let header_info = [vk::DescriptorBufferInfo::default()
                .buffer(placeholder_header_buffer.buffer())
                .offset(0)
                .range(std::mem::size_of::<RegionPlanHeader>() as u64)];
            let river_points_info = [vk::DescriptorBufferInfo::default()
                .buffer(placeholder_river_points.buffer())
                .offset(0)
                .range(std::mem::size_of::<RiverPointGpu>() as u64)];
            let river_headers_info = [vk::DescriptorBufferInfo::default()
                .buffer(placeholder_river_headers.buffer())
                .offset(0)
                .range(std::mem::size_of::<RiverHeaderGpu>() as u64)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&output_image_info),
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
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&river_points_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&river_headers_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };

            slots.push(ComputeSlot {
                output_image,
                output_image_view,
                output_image_alloc: Some(output_image_alloc),
                readback_buffer,
                descriptor_set,
                fence,
                cmd_pool,
                cmd,
                state: SlotState::Free,
                last_touched_frame: 0,
                first_use: true,
                chunk_pos: [0, 0, 0],
            });
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            shader_module,
            slots,
            next_request_id: 1,
            region_cells_buffer: None,
            region_header_buffer: None,
            placeholder_cells_buffer,
            placeholder_header_buffer,
            river_points_buffer: None,
            river_headers_buffer: None,
            placeholder_river_points,
            placeholder_river_headers,
        })
    }

    /// Update bindings 1 + 2 on every slot's descriptor set to point at the
    /// supplied cells and header buffers.
    fn rebind_region_on_all_slots(
        &self,
        ctx: &VulkanContext,
        cells_buf: &AllocatedBuffer,
        cells_size: u64,
        header_buf: &AllocatedBuffer,
        header_size: u64,
    ) {
        let device = ctx.device();
        let cells_info = [vk::DescriptorBufferInfo::default()
            .buffer(cells_buf.buffer())
            .offset(0)
            .range(cells_size)];
        let header_info = [vk::DescriptorBufferInfo::default()
            .buffer(header_buf.buffer())
            .offset(0)
            .range(header_size)];
        for slot in &self.slots {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(slot.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&cells_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(slot.descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&header_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Update bindings 3 + 4 on every slot's descriptor set to point at the
    /// supplied river points and headers buffers.
    fn rebind_rivers_on_all_slots(
        &self,
        ctx: &VulkanContext,
        points_buf: &AllocatedBuffer,
        points_size: u64,
        headers_buf: &AllocatedBuffer,
        headers_size: u64,
    ) {
        let device = ctx.device();
        let points_info = [vk::DescriptorBufferInfo::default()
            .buffer(points_buf.buffer())
            .offset(0)
            .range(points_size)];
        let headers_info = [vk::DescriptorBufferInfo::default()
            .buffer(headers_buf.buffer())
            .offset(0)
            .range(headers_size)];
        for slot in &self.slots {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(slot.descriptor_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&points_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(slot.descriptor_set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&headers_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Wait for all in-flight slots to drain. Used before destroying or
    /// freeing buffers that slots may reference. Loaded slots are discarded
    /// to Free because the region plan / rivers they were generated against
    /// may be about to change.
    fn wait_idle(&mut self, ctx: &VulkanContext) -> Result<()> {
        let device = ctx.device();
        let fences: Vec<vk::Fence> = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::InFlight(_)))
            .map(|s| s.fence)
            .collect();
        if !fences.is_empty() {
            unsafe { device.wait_for_fences(&fences, true, u64::MAX)? };
        }
        // Mark everything free (callers are about to tear down or re-upload).
        // The images were last left in SHADER_READ_ONLY_OPTIMAL by a
        // successful dispatch, so reset first_use to false is not valid —
        // we just keep first_use as-is and rely on the normal SRO→GENERAL
        // transition on next submit.
        for slot in self.slots.iter_mut() {
            slot.state = SlotState::Free;
        }
        Ok(())
    }

    /// Upload (or re-upload) the region plan cells + header to GPU storage
    /// buffers and update the descriptor set bindings 1/2 on every slot.
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

        // Make sure nothing is reading the old bindings.
        self.wait_idle(ctx)?;

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

        // 4. Update descriptor set bindings on ALL slots.
        self.rebind_region_on_all_slots(ctx, &cells_buf, cells_size, &header_buf, header_size);

        self.region_cells_buffer = Some(cells_buf);
        self.region_header_buffer = Some(header_buf);
        Ok(())
    }

    /// Upload (or re-upload) the river polylines to GPU storage buffers and
    /// update the descriptor set bindings 3/4 on every slot.
    ///
    /// If `points` or `headers` is empty, a single dummy entry is uploaded
    /// (length=0 header) so the shader inner loop simply does nothing.
    pub fn upload_rivers(
        &mut self,
        ctx: &VulkanContext,
        alloc: &mut VulkanAllocator,
        points: &[RiverPointGpu],
        headers: &[RiverHeaderGpu],
    ) -> Result<()> {
        self.wait_idle(ctx)?;

        // 1. Free any previously-uploaded buffers.
        if let Some(old) = self.river_points_buffer.take() {
            alloc.free_buffer(old);
        }
        if let Some(old) = self.river_headers_buffer.take() {
            alloc.free_buffer(old);
        }

        // 2. Decide what we're uploading. Always upload at least one dummy
        //    entry to keep the descriptor range > 0 (empty buffers are not
        //    allowed by the allocator).
        let use_dummy = points.is_empty() || headers.is_empty();
        let (points_slice, headers_slice);
        let dummy_pt = [RiverPointGpu {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            _pad: 0.0,
        }];
        let dummy_hdr = [RiverHeaderGpu {
            start_idx: 0,
            length: 0,
            _pad0: 0,
            _pad1: 0,
        }];
        if use_dummy {
            points_slice = &dummy_pt[..];
            headers_slice = &dummy_hdr[..];
        } else {
            points_slice = points;
            headers_slice = headers;
        }

        // 3. Allocate + fill points buffer.
        let points_size =
            (points_slice.len() * std::mem::size_of::<RiverPointGpu>()) as u64;
        let mut points_buf = alloc
            .allocate_host_visible_buffer(points_size)
            .context("river points buffer")?;
        let points_bytes = unsafe {
            std::slice::from_raw_parts(
                points_slice.as_ptr() as *const u8,
                points_size as usize,
            )
        };
        points_buf
            .write_data(0, points_bytes)
            .context("write river points")?;

        // 4. Allocate + fill headers buffer.
        let headers_size =
            (headers_slice.len() * std::mem::size_of::<RiverHeaderGpu>()) as u64;
        let mut headers_buf = alloc
            .allocate_host_visible_buffer(headers_size)
            .context("river headers buffer")?;
        let headers_bytes = unsafe {
            std::slice::from_raw_parts(
                headers_slice.as_ptr() as *const u8,
                headers_size as usize,
            )
        };
        headers_buf
            .write_data(0, headers_bytes)
            .context("write river headers")?;

        // 5. Update descriptor set bindings 3 and 4 on every slot.
        self.rebind_rivers_on_all_slots(ctx, &points_buf, points_size, &headers_buf, headers_size);

        self.river_points_buffer = Some(points_buf);
        self.river_headers_buffer = Some(headers_buf);
        Ok(())
    }

    /// Submit a chunk for GPU generation.
    ///
    /// Returns:
    /// - `Ok(Some(id))` if a new dispatch was submitted.
    /// - `Ok(None)` if the chunk is already Loaded or InFlight, OR if no
    ///   slot could be acquired (all slots are InFlight — caller should
    ///   poll and retry next frame).
    pub fn submit_chunk(
        &mut self,
        ctx: &VulkanContext,
        chunk_pos: ChunkKey,
        seed: u32,
    ) -> Result<Option<u64>> {
        // 1. If this chunk is already known to the pool, no-op.
        if self
            .slots
            .iter()
            .any(|s| s.state != SlotState::Free && s.chunk_pos == chunk_pos)
        {
            return Ok(None);
        }

        // 2. Find a Free slot, or evict the least-recently-touched Loaded one.
        let slot_idx = match self.slots.iter().position(|s| s.state == SlotState::Free) {
            Some(i) => i,
            None => {
                let oldest = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s.state, SlotState::Loaded(_)))
                    .min_by_key(|(_, s)| s.last_touched_frame)
                    .map(|(i, _)| i);
                match oldest {
                    Some(i) => i,
                    // All slots are InFlight — caller should retry later.
                    None => return Ok(None),
                }
            }
        };

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let device = ctx.device();
        let cq = ctx.compute_queue().context("no compute queue")?;
        let pipeline = self.pipeline;
        let pipeline_layout = self.pipeline_layout;
        let slot = &mut self.slots[slot_idx];

        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            // Reset the command pool so we can re-record into the cmd buffer.
            device.reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(slot.cmd, &begin)?;

            // Transition storage image -> GENERAL for compute write.
            // First use: UNDEFINED -> GENERAL.
            // Subsequent (slot was Loaded or Free-after-wait_idle):
            // SHADER_READ_ONLY_OPTIMAL -> GENERAL, since the previous dispatch
            // left the image sampleable by the renderer.
            let (old_layout, src_access, src_stage) = if slot.first_use {
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::AccessFlags::empty(),
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                )
            } else {
                (
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                )
            };
            let barrier_to_general = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(slot.output_image)
                .subresource_range(full_range);
            device.cmd_pipeline_barrier(
                slot.cmd,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_general],
            );

            device.cmd_bind_pipeline(slot.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                slot.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[slot.descriptor_set],
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
                slot.cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );

            // Dispatch: 8×8×8 local size; one invocation per voxel.
            let groups = CHUNK_SIZE / 8;
            device.cmd_dispatch(slot.cmd, groups, groups, groups);

            // Transition GENERAL -> SHADER_READ_ONLY_OPTIMAL so the renderer
            // can sample the image in subsequent graphics draws.
            let barrier_to_sro = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(slot.output_image)
                .subresource_range(full_range);
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_sro],
            );

            device.end_command_buffer(slot.cmd)?;

            // Reset the fence (will be signaled on submission completion).
            device.reset_fences(&[slot.fence])?;

            let cmds = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.queue_submit(cq.queue, &[submit], slot.fence)?;
        }

        slot.state = SlotState::InFlight(request_id);
        slot.first_use = false;
        slot.chunk_pos = chunk_pos;
        Ok(Some(request_id))
    }

    /// Poll all InFlight slots for completion. Slots whose fence has signaled
    /// transition from `InFlight` → `Loaded` and are returned in the result.
    /// Non-blocking.
    ///
    /// Unlike Phase 1 this does not read back pixel data — the image stays
    /// GPU-resident in SHADER_READ_ONLY_OPTIMAL and the renderer samples it
    /// via `loaded_chunks()`. For the old readback semantics (used by the
    /// game crate before Phase 3 wires rendering), use
    /// [`try_take_completed_with_bytes`].
    pub fn try_take_completed(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<Vec<(u64, ChunkKey)>> {
        let device = ctx.device();
        let mut completed = Vec::new();
        for slot in self.slots.iter_mut() {
            let request_id = match slot.state {
                SlotState::InFlight(id) => id,
                _ => continue,
            };
            let status = unsafe { device.get_fence_status(slot.fence) };
            match status {
                Ok(true) => {
                    slot.state = SlotState::Loaded(slot.chunk_pos);
                    completed.push((request_id, slot.chunk_pos));
                }
                Ok(false) => {
                    // Still running, leave in-flight.
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(completed)
    }

    /// DEPRECATED backward-compat wrapper for callers that still expect a
    /// `(request_id, chunk_pos, bytes)` tuple. Issues an explicit one-shot
    /// readback per completed slot. Phase 3 of the pool plan will remove
    /// this once the renderer samples the images directly.
    pub fn try_take_completed_with_bytes(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<Vec<(u64, ChunkKey, Vec<u8>)>> {
        let completed = self.try_take_completed(ctx)?;
        let mut result = Vec::with_capacity(completed.len());
        for (req_id, pos) in completed {
            // Find the slot now in Loaded state for this chunk.
            let slot_idx = self
                .slots
                .iter()
                .position(|s| matches!(s.state, SlotState::Loaded(p) if p == pos))
                .ok_or_else(|| {
                    anyhow::anyhow!("completed chunk {:?} missing from pool", pos)
                })?;
            self.read_slot_to_buffer(ctx, slot_idx)?;
            let bytes = self.slots[slot_idx]
                .readback_buffer
                .read_data(0, OUTPUT_BYTES as usize)?;
            result.push((req_id, pos, bytes));
        }
        Ok(result)
    }

    /// Iterate over all slots currently holding valid chunk data, yielding
    /// `(chunk_pos, image_view)` pairs. The renderer uses this to build its
    /// per-frame sampler bind group.
    pub fn loaded_chunks(&self) -> impl Iterator<Item = (ChunkKey, vk::ImageView)> + '_ {
        self.slots.iter().filter_map(|s| match s.state {
            SlotState::Loaded(pos) => Some((pos, s.output_image_view)),
            _ => None,
        })
    }

    /// Mark a Loaded slot as touched this frame. The renderer calls this each
    /// frame for every chunk it is still sampling, to keep LRU eviction
    /// accurate. No-op if the chunk is not currently Loaded.
    pub fn mark_touched(&mut self, chunk_pos: ChunkKey, current_frame: u64) {
        for s in self.slots.iter_mut() {
            if let SlotState::Loaded(p) = s.state {
                if p == chunk_pos {
                    s.last_touched_frame = current_frame;
                    return;
                }
            }
        }
    }

    /// One-shot readback of a slot's image into its host-visible
    /// `readback_buffer`. Uses the slot's own command pool + fence, which is
    /// safe because the slot is in Loaded state (its previous dispatch has
    /// already signaled the fence).
    ///
    /// Records: SHADER_READ_ONLY_OPTIMAL → TRANSFER_SRC_OPTIMAL,
    /// cmd_copy_image_to_buffer, TRANSFER → HOST memory barrier,
    /// TRANSFER_SRC_OPTIMAL → SHADER_READ_ONLY_OPTIMAL. Submits and waits.
    fn read_slot_to_buffer(
        &mut self,
        ctx: &VulkanContext,
        slot_idx: usize,
    ) -> Result<()> {
        let device = ctx.device();
        let cq = ctx.compute_queue().context("no compute queue")?;
        let slot = &mut self.slots[slot_idx];

        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            device.reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(slot.cmd, &begin)?;

            // SHADER_READ_ONLY_OPTIMAL -> TRANSFER_SRC_OPTIMAL
            let to_xfer = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(slot.output_image)
                .subresource_range(full_range);
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_xfer],
            );

            // Copy image -> readback buffer.
            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: CHUNK_SIZE,
                    height: CHUNK_SIZE,
                    depth: CHUNK_SIZE,
                });
            device.cmd_copy_image_to_buffer(
                slot.cmd,
                slot.output_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                slot.readback_buffer.buffer(),
                &[copy_region],
            );

            // TRANSFER -> HOST so the CPU read sees the copied bytes.
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );

            // Restore image to SHADER_READ_ONLY_OPTIMAL for future rendering.
            let back_to_sro = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(slot.output_image)
                .subresource_range(full_range);
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back_to_sro],
            );

            device.end_command_buffer(slot.cmd)?;

            device.reset_fences(&[slot.fence])?;
            let cmds = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.queue_submit(cq.queue, &[submit], slot.fence)?;
            device.wait_for_fences(&[slot.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    /// Synchronous wrapper: submits a chunk, spin-polls until it completes,
    /// then issues an explicit readback into the host-visible buffer.
    /// Primarily for tests + smoke runs that want a blocking API with bytes.
    pub fn generate_chunk(
        &mut self,
        ctx: &VulkanContext,
        chunk_pos: ChunkKey,
        seed: u32,
    ) -> Result<Vec<u8>> {
        // 1. Submit. If the chunk is already in the pool this returns None
        //    even without failure, so handle that case by either re-reading
        //    from the existing slot (if Loaded) or waiting (if InFlight).
        //    For simplicity, tests use fresh chunk positions per call so we
        //    treat None from submit_chunk after retry as a hard error.
        let req_id = loop {
            match self.submit_chunk(ctx, chunk_pos, seed)? {
                Some(id) => break id,
                None => {
                    // Chunk might already be in-flight/loaded, or pool is
                    // full. If it's Loaded already, fall through to readback.
                    if self
                        .slots
                        .iter()
                        .any(|s| matches!(s.state, SlotState::Loaded(p) if p == chunk_pos))
                    {
                        // Skip waiting; use existing loaded slot.
                        return self.read_loaded_chunk(ctx, chunk_pos);
                    }
                    // In-flight: drain then retry.
                    let _ = self.try_take_completed(ctx)?;
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
        };

        // 2. Spin-poll until our submission completes.
        loop {
            let completed = self.try_take_completed(ctx)?;
            if completed.iter().any(|(rid, _)| *rid == req_id) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        // 3. Readback into the slot's host-visible buffer, then read it out.
        self.read_loaded_chunk(ctx, chunk_pos)
    }

    /// Internal: read a currently-Loaded chunk's materials back to the CPU.
    fn read_loaded_chunk(
        &mut self,
        ctx: &VulkanContext,
        chunk_pos: ChunkKey,
    ) -> Result<Vec<u8>> {
        let slot_idx = self
            .slots
            .iter()
            .position(|s| matches!(s.state, SlotState::Loaded(p) if p == chunk_pos))
            .ok_or_else(|| anyhow::anyhow!("chunk {:?} not loaded", chunk_pos))?;
        self.read_slot_to_buffer(ctx, slot_idx)?;
        let bytes = self.slots[slot_idx]
            .readback_buffer
            .read_data(0, OUTPUT_BYTES as usize)?;
        Ok(bytes)
    }

    pub fn destroy(mut self, ctx: &VulkanContext, alloc: &mut VulkanAllocator) {
        let device = ctx.device();
        // Drain any in-flight submissions first.
        let _ = self.wait_idle(ctx);

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);

            for mut slot in self.slots.drain(..) {
                device.destroy_fence(slot.fence, None);
                device.destroy_command_pool(slot.cmd_pool, None);
                device.destroy_image_view(slot.output_image_view, None);
                device.destroy_image(slot.output_image, None);
                if let Some(image_alloc) = slot.output_image_alloc.take() {
                    alloc.free_allocation(image_alloc);
                }
                alloc.free_buffer(slot.readback_buffer);
            }
        }

        alloc.free_buffer(self.placeholder_cells_buffer);
        alloc.free_buffer(self.placeholder_header_buffer);
        alloc.free_buffer(self.placeholder_river_points);
        alloc.free_buffer(self.placeholder_river_headers);
        if let Some(b) = self.region_cells_buffer {
            alloc.free_buffer(b);
        }
        if let Some(b) = self.region_header_buffer {
            alloc.free_buffer(b);
        }
        if let Some(b) = self.river_points_buffer {
            alloc.free_buffer(b);
        }
        if let Some(b) = self.river_headers_buffer {
            alloc.free_buffer(b);
        }
    }
}
