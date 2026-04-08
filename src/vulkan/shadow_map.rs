use anyhow::{Context, Result};
use ash::vk;

use super::graphics_pipeline::GraphicsPipeline;
use super::instance::VulkanContext;

/// Depth-only render target for shadow map generation.
pub struct ShadowMap {
    pub width: u32,
    pub height: u32,
    pub render_pass: vk::RenderPass,
    pub depth_image: vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    command_pool: vk::CommandPool,
}

impl ShadowMap {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().context("No graphics queue")?;

        let depth_att = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let attachments = [depth_att];
        let depth_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);
        let subpasses = [subpass];
        let rp_ci = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        let render_pass = unsafe { device.create_render_pass(&rp_ci, None) }?;

        let (depth_image, depth_memory) = create_image(
            ctx, width, height, vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let depth_view = create_image_view(device, depth_image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH)?;

        let views = [depth_view];
        let fb_ci = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&views)
            .width(width).height(height).layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&fb_ci, None) }?;

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gq.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }?;

        Ok(Self { width, height, render_pass, depth_image, depth_memory, depth_view, framebuffer, command_pool })
    }

    /// Record shadow map rendering commands into an existing command buffer.
    pub fn record_render(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline: &GraphicsPipeline,
        shared_vb: vk::Buffer,
        shared_ib: vk::Buffer,
        index_count: u32,
        push_constants: &[u8],
        descriptor_set: vk::DescriptorSet,
    ) {
        let clear_values = [vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }];

        unsafe {
            device.cmd_begin_render_pass(cmd, &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass).framebuffer(self.framebuffer)
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } })
                .clear_values(&clear_values), vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } }]);

            let ds = [descriptor_set];
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &ds, &[]);
            device.cmd_bind_vertex_buffers(cmd, 0, &[shared_vb], &[0]);
            device.cmd_bind_index_buffer(cmd, shared_ib, 0, vk::IndexType::UINT32);
            device.cmd_push_constants(cmd, pipeline.layout, pipeline.push_constant_stages, 0, push_constants);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);

            device.cmd_end_render_pass(cmd);
        }
    }

    /// Render into shadow map (depth-only) with DDA pipeline.
    /// Render using pre-existing shared vertex/index buffers.
    pub fn render(
        &self, ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        shared_vb: vk::Buffer,
        shared_ib: vk::Buffer,
        index_count: u32,
        push_constants: &[u8],
        descriptor_set: vk::DescriptorSet,
    ) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        let clear_values = [vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            device.cmd_begin_render_pass(cmd, &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass).framebuffer(self.framebuffer)
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } })
                .clear_values(&clear_values), vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } }]);

            let ds = [descriptor_set];
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &ds, &[]);
            device.cmd_bind_vertex_buffers(cmd, 0, &[shared_vb], &[0]);
            device.cmd_bind_index_buffer(cmd, shared_ib, 0, vk::IndexType::UINT32);
            device.cmd_push_constants(cmd, pipeline.layout, pipeline.push_constant_stages, 0, push_constants);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd)?;
        }

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(())
    }

    /// Read back shadow map depth values.
    pub fn read_depth(&self, ctx: &VulkanContext) -> Result<Vec<f32>> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();
        let n = (self.width * self.height) as usize;
        let bytes = n * 4;

        let (staging, staging_mem) = create_host_buffer(ctx, bytes as u64, vk::BufferUsageFlags::TRANSFER_DST)?;
        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            // Transition SHADER_READ_ONLY → TRANSFER_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.depth_image)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::DEPTH, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });
            device.cmd_copy_image_to_buffer(cmd, self.depth_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, staging, &[region]);

            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        let mut result = vec![0.0f32; n];
        unsafe {
            let ptr = device.map_memory(staging_mem, 0, bytes as u64, vk::MemoryMapFlags::empty())? as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), n);
            device.unmap_memory(staging_mem);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(staging, None); device.free_memory(staging_mem, None);
        }
        Ok(result)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn create_image(ctx: &VulkanContext, w: u32, h: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> Result<(vk::Image, vk::DeviceMemory)> {
    let device = ctx.device();
    let ci = vk::ImageCreateInfo::default().image_type(vk::ImageType::TYPE_2D).format(format)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL).usage(usage);
    let image = unsafe { device.create_image(&ci, None) }?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mt = (0..props.memory_type_count).find(|&i| (req.memory_type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)).context("No device mem")?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
    let memory = unsafe { device.allocate_memory(&ai, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;
    Ok((image, memory))
}

fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags) -> Result<vk::ImageView> {
    let ci = vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    Ok(unsafe { device.create_image_view(&ci, None) }?)
}

fn create_host_buffer(ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let device = ctx.device();
    let ci = vk::BufferCreateInfo::default().size(size).usage(usage);
    let buffer = unsafe { device.create_buffer(&ci, None) }?;
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mt = (0..props.memory_type_count).find(|&i| (req.memory_type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)).context("No host mem")?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
    let memory = unsafe { device.allocate_memory(&ai, None) }?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}
