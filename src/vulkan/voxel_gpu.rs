use anyhow::{bail, Context, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::voxel::grid::VoxelGrid;
use crate::voxel::mip::generate_mip;
use super::allocator::VulkanAllocator;
use super::instance::VulkanContext;

pub struct GpuVoxelTexture {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    allocation: Option<Allocation>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip1_image: vk::Image,
    pub mip1_view: vk::ImageView,
    mip1_allocation: Option<Allocation>,
    pub mip1_width: u32,
    pub mip1_height: u32,
    pub mip1_depth: u32,
    pub mip2_image: vk::Image,
    pub mip2_view: vk::ImageView,
    mip2_allocation: Option<Allocation>,
    pub mip2_width: u32,
    pub mip2_height: u32,
    pub mip2_depth: u32,
    /// 256x1 RGBA8 palette texture for per-voxel color lookup.
    pub palette_image: vk::Image,
    pub palette_view: vk::ImageView,
    palette_allocation: Option<Allocation>,
}

impl GpuVoxelTexture {
    pub fn destroy(mut self, ctx: &VulkanContext, alloc: &mut VulkanAllocator) {
        unsafe {
            ctx.device().destroy_image_view(self.image_view, None);
            ctx.device().destroy_image(self.image, None);
            ctx.device().destroy_image_view(self.mip1_view, None);
            ctx.device().destroy_image(self.mip1_image, None);
            ctx.device().destroy_image_view(self.mip2_view, None);
            ctx.device().destroy_image(self.mip2_image, None);
            ctx.device().destroy_image_view(self.palette_view, None);
            ctx.device().destroy_image(self.palette_image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            alloc.free_allocation(allocation);
        }
        if let Some(allocation) = self.mip1_allocation.take() {
            alloc.free_allocation(allocation);
        }
        if let Some(allocation) = self.mip2_allocation.take() {
            alloc.free_allocation(allocation);
        }
        if let Some(allocation) = self.palette_allocation.take() {
            alloc.free_allocation(allocation);
        }
    }

    /// Update a subregion of the 3D texture from the CPU grid.
    pub fn update_subregion(
        &mut self,
        ctx: &VulkanContext,
        alloc: &mut VulkanAllocator,
        grid: &VoxelGrid,
        offset: (u32, u32, u32),
        extent: (u32, u32, u32),
    ) -> Result<()> {
        let device = ctx.device();
        let q = ctx.graphics_queue().context("No graphics queue")?;

        // Extract subregion data from grid
        let mut subdata = Vec::with_capacity((extent.0 * extent.1 * extent.2) as usize);
        for z in offset.2..offset.2 + extent.2 {
            for y in offset.1..offset.1 + extent.1 {
                for x in offset.0..offset.0 + extent.0 {
                    subdata.push(grid.get(x, y, z).unwrap_or(0));
                }
            }
        }

        let staging_size = subdata.len() as u64;
        let mut staging = alloc.allocate_host_visible_buffer(staging_size)?;
        staging.write_data(0, &subdata)?;

        // Record copy
        let cmd = one_time_cmd_begin(ctx, q.family_index)?;

        transition_image_layout(
            device, cmd, self.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
            .image_offset(vk::Offset3D {
                x: offset.0 as i32,
                y: offset.1 as i32,
                z: offset.2 as i32,
            })
            .image_extent(vk::Extent3D {
                width: extent.0,
                height: extent.1,
                depth: extent.2,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        transition_image_layout(
            device, cmd, self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        one_time_cmd_end(ctx, cmd, q)?;

        alloc.free_buffer(staging);
        Ok(())
    }
}

/// Helper: create a 3D R8_UINT image + image view, returning (image, view, allocation).
fn create_3d_r8_image(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    w: u32,
    h: u32,
    d: u32,
) -> Result<(vk::Image, vk::ImageView, Allocation)> {
    let device = ctx.device();

    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .format(vk::Format::R8_UINT)
        .extent(vk::Extent3D { width: w, height: h, depth: d })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_ci, None) }
        .context("Failed to create 3D image")?;

    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let allocation = alloc.allocate_image_memory(mem_req)?;

    unsafe {
        device.bind_image_memory(image, allocation.memory(), allocation.offset())
    }.context("Failed to bind image memory")?;

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
    let image_view = unsafe { device.create_image_view(&view_ci, None) }
        .context("Failed to create image view")?;

    Ok((image, image_view, allocation))
}

/// Helper: upload raw bytes to a 3D image via staging buffer, with layout transitions.
fn upload_data_to_image(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    image: vk::Image,
    w: u32,
    h: u32,
    d: u32,
    data: &[u8],
) -> Result<()> {
    let device = ctx.device();
    let q = ctx.graphics_queue().context("No graphics queue")?;

    let mut staging = alloc.allocate_host_visible_buffer(data.len() as u64)?;
    staging.write_data(0, data)?;

    let cmd = one_time_cmd_begin(ctx, q.family_index)?;

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
        .image_extent(vk::Extent3D { width: w, height: h, depth: d });

    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer(),
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    one_time_cmd_end(ctx, cmd, q)?;

    alloc.free_buffer(staging);
    Ok(())
}

/// Helper: create a 256x1 R8G8B8A8_UNORM 2D image for palette lookup.
fn create_palette_image(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
) -> Result<(vk::Image, vk::ImageView, Allocation)> {
    let device = ctx.device();

    let image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D { width: 256, height: 1, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_ci, None) }
        .context("Failed to create palette image")?;

    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let allocation = alloc.allocate_image_memory(mem_req)?;

    unsafe {
        device.bind_image_memory(image, allocation.memory(), allocation.offset())
    }.context("Failed to bind palette image memory")?;

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
    let image_view = unsafe { device.create_image_view(&view_ci, None) }
        .context("Failed to create palette image view")?;

    Ok((image, image_view, allocation))
}

/// Helper: upload palette RGBA data (256x1) to a 2D image via staging buffer.
fn upload_palette_data(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    image: vk::Image,
    palette_rgba: &[[u8; 4]; 256],
) -> Result<()> {
    let device = ctx.device();
    let q = ctx.graphics_queue().context("No graphics queue")?;

    // Flatten to contiguous bytes
    let data: Vec<u8> = palette_rgba.iter().flat_map(|c| c.iter().copied()).collect();

    let mut staging = alloc.allocate_host_visible_buffer(data.len() as u64)?;
    staging.write_data(0, &data)?;

    let cmd = one_time_cmd_begin(ctx, q.family_index)?;

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
        .image_extent(vk::Extent3D { width: 256, height: 1, depth: 1 });

    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer(),
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    one_time_cmd_end(ctx, cmd, q)?;

    alloc.free_buffer(staging);
    Ok(())
}

/// Upload a VoxelGrid to a 3D R8_UINT VkImage, including MIP1, MIP2, and palette images.
pub fn upload_grid_to_gpu(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    grid: &VoxelGrid,
    palette_rgba: &[[u8; 4]; 256],
) -> Result<GpuVoxelTexture> {
    let (w, h, d) = grid.dimensions();
    let data = grid.data();

    // Full-res image
    let (image, image_view, allocation) = create_3d_r8_image(ctx, alloc, w, h, d)?;
    upload_data_to_image(ctx, alloc, image, w, h, d, data)?;

    // MIP1 (stride 2)
    let (mip1_data, mip1_w, mip1_h, mip1_d) = generate_mip(data, w, h, d, 2);
    let (mip1_image, mip1_view, mip1_alloc) = create_3d_r8_image(ctx, alloc, mip1_w, mip1_h, mip1_d)?;
    upload_data_to_image(ctx, alloc, mip1_image, mip1_w, mip1_h, mip1_d, &mip1_data)?;

    // MIP2 (stride 4)
    let (mip2_data, mip2_w, mip2_h, mip2_d) = generate_mip(data, w, h, d, 4);
    let (mip2_image, mip2_view, mip2_alloc) = create_3d_r8_image(ctx, alloc, mip2_w, mip2_h, mip2_d)?;
    upload_data_to_image(ctx, alloc, mip2_image, mip2_w, mip2_h, mip2_d, &mip2_data)?;

    // Palette (256x1 RGBA8)
    let (palette_image, palette_view, palette_alloc) = create_palette_image(ctx, alloc)?;
    upload_palette_data(ctx, alloc, palette_image, palette_rgba)?;

    Ok(GpuVoxelTexture {
        image,
        image_view,
        allocation: Some(allocation),
        width: w,
        height: h,
        depth: d,
        mip1_image,
        mip1_view,
        mip1_allocation: Some(mip1_alloc),
        mip1_width: mip1_w,
        mip1_height: mip1_h,
        mip1_depth: mip1_d,
        mip2_image,
        mip2_view,
        mip2_allocation: Some(mip2_alloc),
        mip2_width: mip2_w,
        mip2_height: mip2_h,
        mip2_depth: mip2_d,
        palette_image,
        palette_view,
        palette_allocation: Some(palette_alloc),
    })
}

/// Readback MIP1 or MIP2 texture data from the GPU.
/// `mip_level` must be 1 or 2.
pub fn readback_mip_from_gpu(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    tex: &GpuVoxelTexture,
    mip_level: u32,
) -> Result<Vec<u8>> {
    let (image, w, h, d) = match mip_level {
        1 => (tex.mip1_image, tex.mip1_width, tex.mip1_height, tex.mip1_depth),
        2 => (tex.mip2_image, tex.mip2_width, tex.mip2_height, tex.mip2_depth),
        _ => bail!("Invalid mip_level {mip_level}: must be 1 or 2"),
    };

    let device = ctx.device();
    let q = ctx.graphics_queue().context("No graphics queue")?;
    let total_bytes = (w * h * d) as u64;

    let staging = alloc.allocate_host_visible_buffer(total_bytes)?;

    let cmd = one_time_cmd_begin(ctx, q.family_index)?;

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
        .image_extent(vk::Extent3D { width: w, height: h, depth: d });

    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer(),
            &[region],
        );
    }

    transition_image_layout(
        device, cmd, image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    one_time_cmd_end(ctx, cmd, q)?;

    let data = staging.read_data(0, total_bytes as usize)?;
    alloc.free_buffer(staging);
    Ok(data)
}

/// Readback a GpuVoxelTexture into a new VoxelGrid.
pub fn readback_grid_from_gpu(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    tex: &GpuVoxelTexture,
) -> Result<VoxelGrid> {
    let device = ctx.device();
    let q = ctx.graphics_queue().context("No graphics queue")?;
    let total_bytes = (tex.width * tex.height * tex.depth) as u64;

    let staging = alloc.allocate_host_visible_buffer(total_bytes)?;

    let cmd = one_time_cmd_begin(ctx, q.family_index)?;

    transition_image_layout(
        device, cmd, tex.image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
            width: tex.width,
            height: tex.height,
            depth: tex.depth,
        });

    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            tex.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer(),
            &[region],
        );
    }

    transition_image_layout(
        device, cmd, tex.image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    one_time_cmd_end(ctx, cmd, q)?;

    let data = staging.read_data(0, total_bytes as usize)?;

    let mut grid = VoxelGrid::new(tex.width, tex.height, tex.depth);
    for z in 0..tex.depth {
        for y in 0..tex.height {
            for x in 0..tex.width {
                let idx = (z * tex.height * tex.width + y * tex.width + x) as usize;
                grid.set(x, y, z, data[idx]);
            }
        }
    }

    alloc.free_buffer(staging);
    Ok(grid)
}

// --- Helpers ---

fn one_time_cmd_begin(ctx: &VulkanContext, queue_family: u32) -> Result<vk::CommandBuffer> {
    let device = ctx.device();
    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { device.create_command_pool(&pool_ci, None) }?;

    let alloc_ci = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }
    Ok(cmd)
}

fn one_time_cmd_end(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    queue: super::instance::QueueHandle,
) -> Result<()> {
    let device = ctx.device();
    unsafe {
        device.end_command_buffer(cmd)?;
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        device.queue_submit(queue.queue, &[submit], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
        // Note: command pool cleanup is leaked here for simplicity.
        // In production code, we'd track and destroy the pool.
    }
    Ok(())
}

fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_access, src_stage, dst_access, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
