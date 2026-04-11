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

const MIP1_SIZE: u32 = CHUNK_SIZE / 2;
const MIP2_SIZE: u32 = CHUNK_SIZE / 4;
const MIP3_SIZE: u32 = CHUNK_SIZE / 8;

/// Number of chunk texture pool slots. 64³ R8_UINT main + 32³ + 16³ + 8³ mips
/// ≈ 300KB each × 256 = ~75MB of GPU memory dedicated to the pool.
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
    /// Mip1: 32³ OR-downsample of the main image.
    mip1_image: vk::Image,
    mip1_view: vk::ImageView,
    mip1_alloc: Option<Allocation>,
    /// Mip2: 16³ OR-downsample of mip1.
    mip2_image: vk::Image,
    mip2_view: vk::ImageView,
    mip2_alloc: Option<Allocation>,
    /// Mip3: 8³ OR-downsample of mip2.
    mip3_image: vk::Image,
    mip3_view: vk::ImageView,
    mip3_alloc: Option<Allocation>,
    /// Host-visible staging buffer used by the sync `generate_chunk` test
    /// wrapper and the deprecated `try_take_completed_with_bytes` shim. The
    /// normal fast path (renderer samples directly) never touches this.
    readback_buffer: AllocatedBuffer,
    descriptor_set: vk::DescriptorSet,
    /// Descriptor sets for the mip downsample passes: [mip0→mip1, mip1→mip2,
    /// mip2→mip3]. Allocated from the mip descriptor pool at slot creation
    /// and reused every dispatch (image views never change).
    mip_descriptor_sets: [vk::DescriptorSet; 3],
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

/// View of a loaded chunk's images, ready for renderer sampling. All four
/// images are in `SHADER_READ_ONLY_OPTIMAL` layout. Returned by
/// [`TerrainComputePipeline::loaded_chunk_views`] and passed to the pool
/// render entry point so the renderer can build its per-frame descriptor
/// sets directly from the pool slots without any CPU round-trip.
#[derive(Clone, Copy, Debug)]
pub struct LoadedChunkView {
    /// Index of the pool slot owning this chunk. Stable for the lifetime
    /// of the chunk in the pool, lets callers bypass the linear-scan
    /// lookup in `mark_touched` via `mark_touched_slot`.
    pub slot_idx: u32,
    pub chunk_pos: ChunkKey,
    pub main_view: vk::ImageView,
    pub mip1_view: vk::ImageView,
    pub mip2_view: vk::ImageView,
    pub mip3_view: vk::ImageView,
    pub main_dim: [u32; 3],
    pub mip1_dim: [u32; 3],
    pub mip2_dim: [u32; 3],
    pub mip3_dim: [u32; 3],
}

/// Create a `w×h×d` R8_UINT 3D image for compute-write + sample + xfer.
///
/// The image is accessed by BOTH the compute queue (write during terrain
/// materialize dispatch) AND the graphics queue (sample during render_frame_pool).
/// If those are different queue families, we need `SharingMode::CONCURRENT` with
/// both family indices, otherwise cross-queue reads hit undefined behaviour and
/// produce visible artifacts (stale cache, wrong layout assumptions).
fn create_3d_storage_image(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    w: u32,
    h: u32,
    d: u32,
) -> Result<(vk::Image, vk::ImageView, Allocation)> {
    let device = ctx.device();

    // Determine queue families that will access this image.
    let compute_qf = ctx
        .compute_queue()
        .context("no compute queue")?
        .family_index;
    let graphics_qf = ctx
        .graphics_queue()
        .context("no graphics queue")?
        .family_index;
    let queue_families: Vec<u32> = if compute_qf == graphics_qf {
        vec![] // EXCLUSIVE doesn't use this field
    } else {
        vec![compute_qf, graphics_qf]
    };
    let sharing_mode = if queue_families.is_empty() {
        vk::SharingMode::EXCLUSIVE
    } else {
        vk::SharingMode::CONCURRENT
    };

    let mut image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .format(vk::Format::R8_UINT)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: d,
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
        .sharing_mode(sharing_mode)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    if !queue_families.is_empty() {
        image_ci = image_ci.queue_family_indices(&queue_families);
    }

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
    // --- Mip downsample pipeline ---
    mip_pipeline: vk::Pipeline,
    mip_pipeline_layout: vk::PipelineLayout,
    mip_descriptor_set_layout: vk::DescriptorSetLayout,
    mip_descriptor_pool: vk::DescriptorPool,
    mip_shader_module: vk::ShaderModule,
    // Ring of in-flight slots.
    slots: Vec<ComputeSlot>,
    next_request_id: u64,
    /// Monotonically incremented whenever the set of slots returned by
    /// `loaded_chunk_views()` changes: Loaded→Free, Loaded→InFlight (via
    /// eviction in `submit_chunk_with_frame`), or InFlight→Loaded (via
    /// `try_take_completed_with_frame`). Callers can cache cull output
    /// keyed on this generation — as long as the generation is unchanged
    /// and the camera hasn't moved, last frame's cull result is still
    /// correct.
    pool_generation: u64,
    /// Cached slot-state counts, maintained incrementally on every state
    /// transition so `pool_stats()` is O(1) and `try_take_completed*`
    /// can early-exit without sweeping the slots Vec when nothing is
    /// in flight. On a 256-slot pool at 100k+ FPS, the per-frame linear
    /// scans were touching 256 cache lines × 3 call sites = ~800 cache
    /// lines per frame, by far the biggest remaining render-loop cost.
    free_count: usize,
    in_flight_count: usize,
    loaded_count: usize,
    /// Bulk-touch frame. When a caller knows that every currently Loaded
    /// slot is still in the active render set but doesn't want to pay the
    /// O(N) cost of calling `mark_touched_slot` per slot, it can call
    /// [`Self::bulk_touch_all_loaded`] which just bumps this field in
    /// O(1). The eviction LRU in `submit_chunk_with_frame` then reads
    /// `effective_age = max(slot.last_touched_frame, bulk_touched_frame)`
    /// for Loaded slots so they stay protected.
    ///
    /// On a cache-hit stable scene this replaces 256 scattered cache-line
    /// writes per frame with one register update.
    bulk_touched_frame: u64,
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
    // --- Shared palette image (uploaded once, sampled by every draw) ---
    palette_image: Option<vk::Image>,
    palette_view: Option<vk::ImageView>,
    palette_alloc: Option<Allocation>,
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

        // --- Mip downsample pipeline ----------------------------------------
        // Loads the 2x2x2 OR-downsample compute shader and creates its
        // pipeline + descriptor set layout. We allocate NUM_SLOTS × 3 descriptor
        // sets so each slot has dedicated (mip0→mip1, mip1→mip2, mip2→mip3)
        // sets baked in at construction time.
        let mip_spirv_bytes = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/chunk_mip_downsample.comp.spv"
        ));
        let mip_spirv_words: Vec<u32> = mip_spirv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mip_shader_ci = vk::ShaderModuleCreateInfo::default().code(&mip_spirv_words);
        let mip_shader_module = unsafe { device.create_shader_module(&mip_shader_ci, None) }
            .context("create mip downsample shader")?;

        let mip_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let mip_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&mip_bindings);
        let mip_descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&mip_layout_ci, None) }
                .context("mip descriptor set layout")?;

        let mip_push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16); // uvec4 out_dim
        let mip_set_layouts = [mip_descriptor_set_layout];
        let mip_push_ranges = [mip_push_range];
        let mip_pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&mip_set_layouts)
            .push_constant_ranges(&mip_push_ranges);
        let mip_pipeline_layout =
            unsafe { device.create_pipeline_layout(&mip_pipeline_layout_ci, None) }
                .context("mip pipeline layout")?;

        let mip_stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(mip_shader_module)
            .name(c"main");
        let mip_pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(mip_stage_ci)
            .layout(mip_pipeline_layout);
        let mip_pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[mip_pipeline_ci], None)
        }
        .map_err(|(_, e)| e)
        .context("mip compute pipeline")?[0];

        let mip_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count((NUM_SLOTS * 3 * 2) as u32)];
        let mip_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&mip_pool_sizes)
            .max_sets((NUM_SLOTS * 3) as u32);
        let mip_descriptor_pool = unsafe { device.create_descriptor_pool(&mip_pool_ci, None) }
            .context("mip descriptor pool")?;

        let mip_set_layouts_all: Vec<vk::DescriptorSetLayout> =
            vec![mip_descriptor_set_layout; NUM_SLOTS * 3];
        let mip_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(mip_descriptor_pool)
            .set_layouts(&mip_set_layouts_all);
        let mip_descriptor_sets_all = unsafe {
            device.allocate_descriptor_sets(&mip_alloc_info)
        }
        .context("alloc mip descriptor sets")?;

        // --- Per-slot resources ---------------------------------------------

        let mut slots: Vec<ComputeSlot> = Vec::with_capacity(NUM_SLOTS);
        for slot_idx in 0..NUM_SLOTS {
            // 3D storage image (GPU-local) that the compute shader writes to
            // via imageStore. Stays resident between dispatches; the renderer
            // samples it directly via loaded_chunk_views().
            let (output_image, output_image_view, output_image_alloc) =
                create_3d_storage_image(ctx, alloc, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)
                    .context("slot output image")?;
            let (mip1_image, mip1_view, mip1_alloc) =
                create_3d_storage_image(ctx, alloc, MIP1_SIZE, MIP1_SIZE, MIP1_SIZE)
                    .context("slot mip1 image")?;
            let (mip2_image, mip2_view, mip2_alloc) =
                create_3d_storage_image(ctx, alloc, MIP2_SIZE, MIP2_SIZE, MIP2_SIZE)
                    .context("slot mip2 image")?;
            let (mip3_image, mip3_view, mip3_alloc) =
                create_3d_storage_image(ctx, alloc, MIP3_SIZE, MIP3_SIZE, MIP3_SIZE)
                    .context("slot mip3 image")?;

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

            // --- Mip descriptor sets ----------------------------------------
            // Wire up the three (in→out) pairs for this slot once; the image
            // views never change so these sets are reused every dispatch.
            let mip_sets_base = slot_idx * 3;
            let mip_set_01 = mip_descriptor_sets_all[mip_sets_base];
            let mip_set_12 = mip_descriptor_sets_all[mip_sets_base + 1];
            let mip_set_23 = mip_descriptor_sets_all[mip_sets_base + 2];

            // Pair 0: main (binding 0) → mip1 (binding 1)
            let in_info_0 = [vk::DescriptorImageInfo::default()
                .image_view(output_image_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let out_info_0 = [vk::DescriptorImageInfo::default()
                .image_view(mip1_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // Pair 1: mip1 → mip2
            let in_info_1 = [vk::DescriptorImageInfo::default()
                .image_view(mip1_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let out_info_1 = [vk::DescriptorImageInfo::default()
                .image_view(mip2_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // Pair 2: mip2 → mip3
            let in_info_2 = [vk::DescriptorImageInfo::default()
                .image_view(mip2_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let out_info_2 = [vk::DescriptorImageInfo::default()
                .image_view(mip3_view)
                .image_layout(vk::ImageLayout::GENERAL)];

            let mip_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_01)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&in_info_0),
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_01)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_info_0),
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_12)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&in_info_1),
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_12)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_info_1),
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_23)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&in_info_2),
                vk::WriteDescriptorSet::default()
                    .dst_set(mip_set_23)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_info_2),
            ];
            unsafe { device.update_descriptor_sets(&mip_writes, &[]) };

            slots.push(ComputeSlot {
                output_image,
                output_image_view,
                output_image_alloc: Some(output_image_alloc),
                mip1_image,
                mip1_view,
                mip1_alloc: Some(mip1_alloc),
                mip2_image,
                mip2_view,
                mip2_alloc: Some(mip2_alloc),
                mip3_image,
                mip3_view,
                mip3_alloc: Some(mip3_alloc),
                readback_buffer,
                descriptor_set,
                mip_descriptor_sets: [mip_set_01, mip_set_12, mip_set_23],
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
            mip_pipeline,
            mip_pipeline_layout,
            mip_descriptor_set_layout,
            mip_descriptor_pool,
            mip_shader_module,
            slots,
            next_request_id: 1,
            pool_generation: 0,
            free_count: NUM_SLOTS,
            in_flight_count: 0,
            loaded_count: 0,
            bulk_touched_frame: 0,
            region_cells_buffer: None,
            region_header_buffer: None,
            placeholder_cells_buffer,
            placeholder_header_buffer,
            river_points_buffer: None,
            river_headers_buffer: None,
            placeholder_river_points,
            placeholder_river_headers,
            palette_image: None,
            palette_view: None,
            palette_alloc: None,
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
        // Every previously-Loaded slot just dropped out of
        // `loaded_chunk_views()` at once.
        self.pool_generation += 1;
        self.free_count = NUM_SLOTS;
        self.in_flight_count = 0;
        self.loaded_count = 0;
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
        self.submit_chunk_with_frame(ctx, chunk_pos, seed, u64::MAX)
    }

    /// Variant that refuses eviction if the least-recently-used Loaded slot
    /// was touched on or after `current_frame` — i.e. every in-pool chunk is
    /// still being sampled by the current render. In that case we return
    /// `Ok(None)` without submitting, signalling the caller to stop trying
    /// to load more (the pool is saturated with in-view chunks).
    ///
    /// Without this guard, a spiral that visits more chunks than the pool
    /// can hold keeps LRU-evicting currently-visible chunks, forcing a
    /// re-compute next frame and holding the compute queue at 100%
    /// utilization even though all the "new" chunks are chunks the pool
    /// just held a frame ago. That steady-state churn was the single
    /// biggest cost in our frame time (compute stealing GPU SMs from
    /// graphics).
    pub fn submit_chunk_with_frame(
        &mut self,
        ctx: &VulkanContext,
        chunk_pos: ChunkKey,
        seed: u32,
        current_frame: u64,
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
        //    Effective age for a Loaded slot is
        //        max(slot.last_touched_frame, self.bulk_touched_frame)
        //    so a single O(1) bulk_touch_all_loaded() call protects every
        //    currently-loaded slot without a per-slot sweep.
        let bulk_touched = self.bulk_touched_frame;
        let slot_idx = match self.slots.iter().position(|s| s.state == SlotState::Free) {
            Some(i) => i,
            None => {
                let oldest = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s.state, SlotState::Loaded(_)))
                    .map(|(i, s)| (i, s.last_touched_frame.max(bulk_touched)))
                    .min_by_key(|(_, age)| *age);
                match oldest {
                    Some((i, oldest_frame)) => {
                        // Refuse eviction if the LRU slot was touched this
                        // frame or the previous one — it's still part of
                        // the current render set.
                        if current_frame != u64::MAX && oldest_frame + 1 >= current_frame {
                            return Ok(None);
                        }
                        i
                    }
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

            // --- Mip downsample chain ---------------------------------------
            // Record 3 sequential compute dispatches that each OR-downsample
            // 2×2×2 input voxels into one output voxel. Each pass needs a
            // barrier on its output so the next one can read it. Between the
            // main dispatch and mip1, the main image stays in GENERAL — we
            // only need a memory barrier (SHADER_WRITE → SHADER_READ).

            // Bind mip pipeline once; descriptor sets change per pass.
            device.cmd_bind_pipeline(
                slot.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.mip_pipeline,
            );

            // Transition each mip image from UNDEFINED/SHADER_READ_ONLY →
            // GENERAL for compute write. For first_use we just came from
            // UNDEFINED; subsequent dispatches the mips were left in
            // SHADER_READ_ONLY_OPTIMAL by the previous submit_chunk.
            let mip_images = [slot.mip1_image, slot.mip2_image, slot.mip3_image];
            let (mip_old_layout, mip_src_access, mip_src_stage) = if slot.first_use {
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
            let mip_transition_barriers: Vec<vk::ImageMemoryBarrier> = mip_images
                .iter()
                .map(|img| {
                    vk::ImageMemoryBarrier::default()
                        .old_layout(mip_old_layout)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(mip_src_access)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(*img)
                        .subresource_range(full_range)
                })
                .collect();
            device.cmd_pipeline_barrier(
                slot.cmd,
                mip_src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &mip_transition_barriers,
            );

            // Helper: push constants + dispatch + barrier after each pass.
            let mip_sizes = [MIP1_SIZE, MIP2_SIZE, MIP3_SIZE];
            for pass in 0..3 {
                // Before this pass reads its input, make sure the previous
                // pass's write has completed on the input image.
                // Pass 0 input = main (just wrote it).
                // Pass 1 input = mip1 (written by pass 0).
                // Pass 2 input = mip2 (written by pass 1).
                // We emit a single memory barrier (SHADER_WRITE→SHADER_READ)
                // which is enough because the input and output images at each
                // stage are different and both are in GENERAL.
                let mem = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                device.cmd_pipeline_barrier(
                    slot.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[mem],
                    &[],
                    &[],
                );

                device.cmd_bind_descriptor_sets(
                    slot.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.mip_pipeline_layout,
                    0,
                    &[slot.mip_descriptor_sets[pass]],
                    &[],
                );
                let size = mip_sizes[pass];
                let mut mip_push = [0u8; 16];
                mip_push[0..4].copy_from_slice(&size.to_le_bytes());
                mip_push[4..8].copy_from_slice(&size.to_le_bytes());
                mip_push[8..12].copy_from_slice(&size.to_le_bytes());
                device.cmd_push_constants(
                    slot.cmd,
                    self.mip_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &mip_push,
                );
                // local_size = 4 → groups = ceil(size / 4)
                let groups_mip = (size + 3) / 4;
                device.cmd_dispatch(slot.cmd, groups_mip, groups_mip, groups_mip);
            }

            // Transition ALL images (main + 3 mips) from GENERAL →
            // SHADER_READ_ONLY_OPTIMAL so the renderer can sample them.
            let final_images = [
                slot.output_image,
                slot.mip1_image,
                slot.mip2_image,
                slot.mip3_image,
            ];
            let final_barriers: Vec<vk::ImageMemoryBarrier> = final_images
                .iter()
                .map(|img| {
                    vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(*img)
                        .subresource_range(full_range)
                })
                .collect();
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &final_barriers,
            );

            device.end_command_buffer(slot.cmd)?;

            // Reset the fence (will be signaled on submission completion).
            device.reset_fences(&[slot.fence])?;

            let cmds = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.queue_submit(cq.queue, &[submit], slot.fence)?;
        }

        // Update counters before overwriting state: decrement whichever
        // old-state counter the slot was contributing to.
        match slot.state {
            SlotState::Free => self.free_count -= 1,
            SlotState::Loaded(_) => self.loaded_count -= 1,
            SlotState::InFlight(_) => {
                debug_assert!(false, "submit_chunk chose an InFlight slot");
                self.in_flight_count -= 1;
            }
        }
        self.in_flight_count += 1;

        slot.state = SlotState::InFlight(request_id);
        slot.first_use = false;
        slot.chunk_pos = chunk_pos;
        // This slot just left `loaded_chunk_views()` (either it was a
        // Free slot entering the pool or a Loaded slot being evicted).
        // Either way, any cached cull output is now stale.
        self.pool_generation += 1;
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
        self.try_take_completed_with_frame(ctx, 0)
    }

    /// Variant that also stamps freshly-completed slots with the current
    /// frame as their `last_touched_frame`. Without this, a slot that's just
    /// finished its compute dispatch has a stale last_touched value from
    /// when it was last populated (or 0 if it's a new slot), which makes
    /// it look like an LRU eviction target to the very next
    /// `submit_chunk_with_frame` call — and since that call happens in
    /// the same frame as drain, fresh chunks would be evicted before the
    /// renderer even got a chance to mark them touched.
    pub fn try_take_completed_with_frame(
        &mut self,
        ctx: &VulkanContext,
        current_frame: u64,
    ) -> Result<Vec<(u64, ChunkKey)>> {
        // Hot-path fast exit: if nothing is in flight, skip the 256-slot
        // scan entirely. This is the common case on a stable camera and
        // was the single biggest render-loop cost at 150k+ FPS before
        // the counter was added.
        if self.in_flight_count == 0 {
            return Ok(Vec::new());
        }

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
                    slot.last_touched_frame = current_frame;
                    completed.push((request_id, slot.chunk_pos));
                    // A new entry just appeared in `loaded_chunk_views()`.
                    self.pool_generation += 1;
                    self.in_flight_count -= 1;
                    self.loaded_count += 1;
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

    /// Returns `(free, in_flight, loaded)` slot counts for the pool.
    /// O(1): reads cached counters maintained incrementally on state
    /// transitions, no slot scan.
    #[inline]
    pub fn pool_stats(&self) -> (usize, usize, usize) {
        (self.free_count, self.in_flight_count, self.loaded_count)
    }

    /// Iterate over all Loaded slots, yielding a full [`LoadedChunkView`] per
    /// slot. Each view is render-ready: every image is in
    /// `SHADER_READ_ONLY_OPTIMAL` and covers the main + 3 mip levels.
    ///
    /// The pool render entry point consumes this to build per-frame
    /// descriptor sets without any CPU round-trip.
    /// Monotonic counter that bumps whenever `loaded_chunk_views()` output
    /// would change (slot enters/leaves the `Loaded` state). Callers that
    /// cache iteration results can key on this to detect whether the
    /// cache is still valid.
    #[inline]
    pub fn pool_generation(&self) -> u64 {
        self.pool_generation
    }

    pub fn loaded_chunk_views(&self) -> impl Iterator<Item = LoadedChunkView> + '_ {
        self.slots.iter().enumerate().filter_map(|(idx, s)| match s.state {
            SlotState::Loaded(pos) => Some(LoadedChunkView {
                slot_idx: idx as u32,
                chunk_pos: pos,
                main_view: s.output_image_view,
                mip1_view: s.mip1_view,
                mip2_view: s.mip2_view,
                mip3_view: s.mip3_view,
                main_dim: [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE],
                mip1_dim: [MIP1_SIZE, MIP1_SIZE, MIP1_SIZE],
                mip2_dim: [MIP2_SIZE, MIP2_SIZE, MIP2_SIZE],
                mip3_dim: [MIP3_SIZE, MIP3_SIZE, MIP3_SIZE],
            }),
            _ => None,
        })
    }

    /// O(1) mark-touched by slot index. Preferred over `mark_touched` when
    /// the caller already knows the slot index (e.g. from a recent
    /// `loaded_chunk_views` iteration).
    pub fn mark_touched_slot(&mut self, slot_idx: u32, current_frame: u64) {
        if let Some(s) = self.slots.get_mut(slot_idx as usize) {
            if matches!(s.state, SlotState::Loaded(_)) {
                s.last_touched_frame = current_frame;
            }
        }
    }

    /// Bulk-touch every currently Loaded slot at `current_frame` in O(1).
    ///
    /// Use this on a stable-scene fast path where the caller knows every
    /// Loaded slot is still part of the active render set but doesn't
    /// want to pay the O(N) cost of a per-slot mark_touched loop. The
    /// LRU eviction check in `submit_chunk_with_frame` reads
    /// `max(slot.last_touched_frame, bulk_touched_frame)` for Loaded
    /// slots, so a single bulk-touch bump is equivalent to calling
    /// mark_touched_slot on every loaded slot.
    ///
    /// When the caller's stable-scene condition ends (e.g. camera moves)
    /// it should STOP calling this — future frames' full-cull path will
    /// rewrite individual slots' `last_touched_frame`, letting slots
    /// that genuinely fell out of the visible set age out normally.
    #[inline]
    pub fn bulk_touch_all_loaded(&mut self, current_frame: u64) {
        // Monotonic: never move backwards, guards against callers that
        // cycle through frame counters or call out of order.
        if current_frame > self.bulk_touched_frame {
            self.bulk_touched_frame = current_frame;
        }
    }

    /// Return the shared palette image view. Returns `vk::ImageView::null()`
    /// if [`upload_palette`] has not been called yet.
    pub fn palette_view(&self) -> vk::ImageView {
        self.palette_view.unwrap_or(vk::ImageView::null())
    }

    /// Upload (or re-upload) the shared 256×1 RGBA palette used by every
    /// chunk draw. Creates a 2D SAMPLED image on first call, then copies the
    /// RGBA bytes from a host staging buffer and transitions the image into
    /// `SHADER_READ_ONLY_OPTIMAL`. Subsequent calls replace the existing
    /// image (e.g. if the material palette changes).
    pub fn upload_palette(
        &mut self,
        ctx: &VulkanContext,
        alloc: &mut VulkanAllocator,
        palette_rgba: &[[u8; 4]; 256],
    ) -> Result<()> {
        let device = ctx.device();
        let cq = ctx.compute_queue().context("no compute queue")?;

        // Free any previously-uploaded palette.
        if let Some(v) = self.palette_view.take() {
            unsafe { device.destroy_image_view(v, None) };
        }
        if let Some(img) = self.palette_image.take() {
            unsafe { device.destroy_image(img, None) };
        }
        if let Some(a) = self.palette_alloc.take() {
            alloc.free_allocation(a);
        }

        // Create the 256×1 RGBA8 image.
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 256,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&image_ci, None) }
            .context("create palette image")?;
        let mem_req = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate_image_memory(mem_req)?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .context("bind palette image memory")?;
        }
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.create_image_view(&view_ci, None) }
            .context("create palette view")?;

        // Upload the bytes via a host-visible staging buffer.
        let data: Vec<u8> = palette_rgba.iter().flat_map(|c| c.iter().copied()).collect();
        let mut staging = alloc
            .allocate_host_visible_buffer(data.len() as u64)
            .context("palette staging buffer")?;
        staging.write_data(0, &data).context("write palette bytes")?;

        // Use a transient command buffer on the compute queue for the
        // transitions + copy. Compute queues are graphics-capable in our
        // VulkanContext setup (the rest of the pipeline also uses them for
        // layout transitions), so this is safe.
        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(cq.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }
            .context("palette cmd pool")?;
        let cmd_alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc_ci) }
            .context("palette cmd buffer")?[0];

        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd, &begin)?;

            let to_xfer = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_xfer],
            );

            let region = vk::BufferImageCopy::default()
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
                    width: 256,
                    height: 1,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            let to_sro = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_sro],
            );

            device.end_command_buffer(cmd)?;

            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence_ci = vk::FenceCreateInfo::default();
            let fence = device.create_fence(&fence_ci, None)?;
            device.queue_submit(cq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }

        alloc.free_buffer(staging);

        self.palette_image = Some(image);
        self.palette_view = Some(view);
        self.palette_alloc = Some(allocation);
        Ok(())
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

            device.destroy_pipeline(self.mip_pipeline, None);
            device.destroy_pipeline_layout(self.mip_pipeline_layout, None);
            device.destroy_descriptor_pool(self.mip_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.mip_descriptor_set_layout, None);
            device.destroy_shader_module(self.mip_shader_module, None);

            if let Some(v) = self.palette_view.take() {
                device.destroy_image_view(v, None);
            }
            if let Some(img) = self.palette_image.take() {
                device.destroy_image(img, None);
            }
            if let Some(a) = self.palette_alloc.take() {
                alloc.free_allocation(a);
            }

            for mut slot in self.slots.drain(..) {
                device.destroy_fence(slot.fence, None);
                device.destroy_command_pool(slot.cmd_pool, None);
                device.destroy_image_view(slot.output_image_view, None);
                device.destroy_image(slot.output_image, None);
                if let Some(image_alloc) = slot.output_image_alloc.take() {
                    alloc.free_allocation(image_alloc);
                }
                device.destroy_image_view(slot.mip1_view, None);
                device.destroy_image(slot.mip1_image, None);
                if let Some(a) = slot.mip1_alloc.take() {
                    alloc.free_allocation(a);
                }
                device.destroy_image_view(slot.mip2_view, None);
                device.destroy_image(slot.mip2_image, None);
                if let Some(a) = slot.mip2_alloc.take() {
                    alloc.free_allocation(a);
                }
                device.destroy_image_view(slot.mip3_view, None);
                device.destroy_image(slot.mip3_image, None);
                if let Some(a) = slot.mip3_alloc.take() {
                    alloc.free_allocation(a);
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
