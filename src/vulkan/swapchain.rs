use anyhow::{Context, Result};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::instance::VulkanContext;

pub struct SwapchainContext {
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    current_frame: usize,
    max_frames_in_flight: usize,
    pending_overlay_cleanup: Option<(vk::Buffer, vk::DeviceMemory)>,
}

impl SwapchainContext {
    #[cfg(feature = "app-harness")]
    pub fn new(ctx: &VulkanContext, window: &winit::window::Window) -> Result<Self> {
        let display_handle = window.display_handle().unwrap();
        let window_handle = window.window_handle().unwrap();

        let surface = unsafe {
            ash_window::create_surface(
                ctx.entry(),
                ctx.instance(),
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .context("Failed to create surface")?;

        let surface_loader = ash::khr::surface::Instance::new(ctx.entry(), ctx.instance());

        let gq = ctx.graphics_queue().unwrap();

        let surface_supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                ctx.physical_device(),
                gq.family_index,
                surface,
            )
        }
        .unwrap_or(false);
        if !surface_supported {
            anyhow::bail!("Graphics queue doesn't support presentation");
        }

        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(ctx.physical_device(), surface)
        }
        .context("Failed to get surface capabilities")?;

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(ctx.physical_device(), surface)
        }
        .context("Failed to get surface formats")?;

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(ctx.physical_device(), surface)
        }
        .context("Failed to get present modes")?;

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
            .clone();

        // Prefer MAILBOX (triple-buffered, no tearing, no vsync wait) →
        // IMMEDIATE (no vsync, may tear, lowest latency) → FIFO (vsync, 60Hz cap).
        // For perf measurement we want MAILBOX/IMMEDIATE so the frame loop
        // isn't capped at the display refresh rate.
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO // always available
        };
        eprintln!("[voxel] Swapchain present mode: {:?} (available: {:?})", present_mode, present_modes);

        let size = window.inner_size();
        let extent = vk::Extent2D {
            width: size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        };

        let image_count = (capabilities.min_image_count + 1).min(
            if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            },
        );

        let swapchain_loader = ash::khr::swapchain::Device::new(ctx.instance(), ctx.device());

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_ci, None) }
            .context("Failed to create swapchain")?;

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .context("Failed to get swapchain images")?;

        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { ctx.device().create_image_view(&ci, None) }.unwrap()
            })
            .collect();

        // Command pool + buffers
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gq.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { ctx.device().create_command_pool(&pool_ci, None) }
            .context("Failed to create command pool")?;

        let max_frames_in_flight = 2;
        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_in_flight as u32);
        let command_buffers = unsafe { ctx.device().allocate_command_buffers(&alloc_ci) }
            .context("Failed to allocate command buffers")?;

        // Sync objects
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available = Vec::new();
        let mut render_finished = Vec::new();
        let mut in_flight = Vec::new();
        for _ in 0..max_frames_in_flight {
            image_available.push(unsafe { ctx.device().create_semaphore(&sem_ci, None) }?);
            render_finished.push(unsafe { ctx.device().create_semaphore(&sem_ci, None) }?);
            in_flight.push(unsafe { ctx.device().create_fence(&fence_ci, None) }?);
        }

        Ok(Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            images,
            image_views,
            extent,
            format,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            current_frame: 0,
            max_frames_in_flight,
            pending_overlay_cleanup: None,
        })
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn present_cleared_frame(&mut self, ctx: &VulkanContext, clear_color: [f32; 4]) -> Result<()> {
        let device = ctx.device();
        let frame = self.current_frame;

        unsafe {
            device.wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)?;
            device.reset_fences(&[self.in_flight[frame]])?;
        }

        let (image_index, _suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        }
        .context("Failed to acquire next image")?;

        let cmd = self.command_buffers[frame];
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            // Transition to TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.images[image_index as usize])
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Clear
            device.cmd_clear_color_image(
                cmd,
                self.images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: clear_color },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );

            // Transition to PRESENT
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.images[image_index as usize])
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(cmd)?;
        }

        let wait_semaphores = [self.image_available[frame]];
        let signal_semaphores = [self.render_finished[frame]];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores);

        let gq = ctx.graphics_queue().unwrap();
        unsafe {
            device.queue_submit(gq.queue, &[submit_info], self.in_flight[frame])?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader.queue_present(gq.queue, &present_info)
        }
        .context("Failed to present")?;

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
        Ok(())
    }

    /// Present a pixel buffer to the swapchain.
    ///
    /// Creates a staging buffer, copies `pixels` into it (swizzling R/B if the swapchain
    /// is BGRA), then blits to the swapchain image and presents.
    pub fn present_pixels(
        &mut self,
        ctx: &VulkanContext,
        pixels: &[[u8; 4]],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let device = ctx.device();
        let frame = self.current_frame;

        unsafe {
            device.wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)?;
            device.reset_fences(&[self.in_flight[frame]])?;
        }

        let (image_index, _suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        }
        .context("Failed to acquire next image")?;

        // Determine if we need to swizzle R<->B
        let need_swizzle = matches!(
            self.format.format,
            vk::Format::B8G8R8A8_SRGB
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::B8G8R8A8_SNORM
        );

        // Create staging buffer with pixel data
        let byte_size = (width * height * 4) as u64;
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { device.create_buffer(&buffer_ci, None) }
            .context("Failed to create staging buffer")?;

        let mem_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let mem_props = unsafe {
            ctx.instance()
                .get_physical_device_memory_properties(ctx.physical_device())
        };
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                (mem_req.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .context("No host-visible memory type")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let staging_memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .context("Failed to allocate staging memory")?;
        unsafe {
            device.bind_buffer_memory(staging_buffer, staging_memory, 0)
        }
        .context("Failed to bind staging buffer")?;

        // Map and copy pixel data
        let ptr = unsafe {
            device.map_memory(staging_memory, 0, byte_size, vk::MemoryMapFlags::empty())
        }
        .context("Failed to map staging memory")? as *mut u8;

        if need_swizzle {
            // Swizzle RGBA -> BGRA
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut [u8; 4], pixels.len()) };
            for (d, s) in dst.iter_mut().zip(pixels.iter()) {
                *d = [s[2], s[1], s[0], s[3]];
            }
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr() as *const u8,
                    ptr,
                    byte_size as usize,
                );
            }
        }
        unsafe { device.unmap_memory(staging_memory) };

        // Record commands
        let cmd = self.command_buffers[frame];
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            // Transition swapchain image UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.images[image_index as usize])
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy staging buffer -> swapchain image
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: width.min(self.extent.width),
                    height: height.min(self.extent.height),
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                self.images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition swapchain image TRANSFER_DST -> PRESENT_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.images[image_index as usize])
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(cmd)?;
        }

        // Submit
        let wait_semaphores = [self.image_available[frame]];
        let signal_semaphores = [self.render_finished[frame]];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores);

        let gq = ctx.graphics_queue().unwrap();
        unsafe {
            device.queue_submit(gq.queue, &[submit_info], self.in_flight[frame])?;
        }

        // Present
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(gq.queue, &present_info)
        }
        .context("Failed to present")?;

        // Cleanup staging buffer (we waited on fence at start so this is safe for the
        // *previous* frame's staging; for current frame the GPU hasn't finished yet,
        // but we'll wait at the top of the next call).
        // Actually we need to defer cleanup. For simplicity, wait idle.
        unsafe { device.queue_wait_idle(gq.queue) }?;
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
        Ok(())
    }

    /// Present by blitting a source image (already in TRANSFER_SRC_OPTIMAL) to the
    /// swapchain image. The blit handles format conversion and upscaling on the GPU.
    pub fn present_blit(
        &mut self,
        ctx: &VulkanContext,
        src_image: vk::Image,
        src_width: u32,
        src_height: u32,
    ) -> Result<()> {
        self.present_blit_with_wait(ctx, src_image, src_width, src_height, vk::Semaphore::null())
    }

    /// Like `present_blit`, but also GPU-side waits on `extra_wait` (e.g. the
    /// renderer's `render_done_semaphore`) before performing the blit. Pass
    /// `vk::Semaphore::null()` to skip.
    pub fn present_blit_with_wait(
        &mut self,
        ctx: &VulkanContext,
        src_image: vk::Image,
        src_width: u32,
        src_height: u32,
        extra_wait: vk::Semaphore,
    ) -> Result<()> {
        let device = ctx.device();
        let frame = self.current_frame;

        unsafe {
            device.wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)?;
            device.reset_fences(&[self.in_flight[frame]])?;
        }

        let (image_index, _suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        }
        .context("Failed to acquire next image")?;

        let cmd = self.command_buffers[frame];
        let dst_image = self.images[image_index as usize];
        let color_subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            // Transition swapchain image UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier_dst = vk::ImageMemoryBarrier::default()
                .image(dst_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(color_subresource);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier_dst],
            );

            // Blit from source image to swapchain image (handles upscaling + format conversion)
            let src_subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let blit_region = vk::ImageBlit {
                src_subresource,
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: src_width as i32, y: src_height as i32, z: 1 },
                ],
                dst_subresource: src_subresource,
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ],
            };
            device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition swapchain image TRANSFER_DST -> PRESENT_SRC_KHR
            let barrier_present = vk::ImageMemoryBarrier::default()
                .image(dst_image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(color_subresource);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier_present],
            );

            device.end_command_buffer(cmd)?;
        }

        // Submit — wait for both the swapchain image to be acquired AND
        // (optionally) the renderer's output to have finished writing.
        let wait_semaphores_full: [vk::Semaphore; 2] = [self.image_available[frame], extra_wait];
        let wait_stages_full: [vk::PipelineStageFlags; 2] =
            [vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER];
        let wait_count = if extra_wait == vk::Semaphore::null() { 1 } else { 2 };
        let wait_semaphores = &wait_semaphores_full[..wait_count];
        let wait_stages = &wait_stages_full[..wait_count];
        let signal_semaphores = [self.render_finished[frame]];
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores);

        let gq = ctx.graphics_queue().unwrap();
        unsafe {
            device.queue_submit(gq.queue, &[submit_info], self.in_flight[frame])?;
        }

        // Present
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(gq.queue, &present_info)
        }
        .context("Failed to present")?;

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.device_wait_idle().ok();
            for &fence in &self.in_flight {
                device.destroy_fence(fence, None);
            }
            for &sem in &self.render_finished {
                device.destroy_semaphore(sem, None);
            }
            for &sem in &self.image_available {
                device.destroy_semaphore(sem, None);
            }
            device.destroy_command_pool(self.command_pool, None);
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
