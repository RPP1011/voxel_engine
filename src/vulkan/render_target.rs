use anyhow::{Context, Result};
use ash::vk;

use super::graphics_pipeline::GraphicsPipeline;
use super::instance::VulkanContext;

/// Offscreen render target with color (RGBA8) and depth (D32_SFLOAT) attachments.
pub struct OffscreenTarget {
    width: u32,
    height: u32,
    color_image: vk::Image,
    color_memory: vk::DeviceMemory,
    color_view: vk::ImageView,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    command_pool: vk::CommandPool,
}

impl OffscreenTarget {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().context("No graphics queue")?;

        // Render pass: color + depth
        let color_att = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        let depth_att = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        let attachments = [color_att, depth_att];

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_ref);

        let subpasses = [subpass];
        let rp_ci = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let render_pass = unsafe { device.create_render_pass(&rp_ci, None) }?;

        // Create color image
        let (color_image, color_memory) = create_image(
            ctx, width, height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let color_view = create_image_view(
            device, color_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        )?;

        // Create depth image
        let (depth_image, depth_memory) = create_image(
            ctx, width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let depth_view = create_image_view(
            device, depth_image,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        )?;

        // Framebuffer
        let views = [color_view, depth_view];
        let fb_ci = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&fb_ci, None) }?;

        // Command pool
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gq.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }?;

        Ok(Self {
            width, height,
            color_image, color_memory, color_view,
            depth_image, depth_memory, depth_view,
            render_pass, framebuffer, command_pool,
        })
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn color_view(&self) -> vk::ImageView {
        self.color_view
    }

    pub fn color_image(&self) -> vk::Image {
        self.color_image
    }

    /// Record and submit a draw call with clearing.
    pub fn draw(
        &mut self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        vertices: &[[f32; 3]],
        indices: &[u32],
        push_constants: &[f32; 16],
    ) -> Result<()> {
        self.draw_inner(ctx, pipeline, vertices, indices, push_constants, true)
    }

    /// Draw without clearing (for multi-draw).
    pub fn draw_no_clear(
        &mut self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        vertices: &[[f32; 3]],
        indices: &[u32],
        push_constants: &[f32; 16],
    ) -> Result<()> {
        self.draw_inner(ctx, pipeline, vertices, indices, push_constants, false)
    }

    fn draw_inner(
        &mut self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        vertices: &[[f32; 3]],
        indices: &[u32],
        push_constants: &[f32; 16],
        clear: bool,
    ) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        // Upload vertex + index data to host-visible buffers
        let vert_bytes = bytemuck::cast_slice::<[f32; 3], u8>(vertices);
        let idx_bytes = bytemuck::cast_slice::<u32, u8>(indices);

        let (vb, vb_mem) = create_host_buffer(ctx, vert_bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (ib, ib_mem) = create_host_buffer(ctx, idx_bytes.len() as u64, vk::BufferUsageFlags::INDEX_BUFFER)?;

        unsafe {
            let vptr = device.map_memory(vb_mem, 0, vert_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(vert_bytes.as_ptr(), vptr, vert_bytes.len());
            device.unmap_memory(vb_mem);

            let iptr = device.map_memory(ib_mem, 0, idx_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(idx_bytes.as_ptr(), iptr, idx_bytes.len());
            device.unmap_memory(ib_mem);
        }

        // Record command buffer
        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        // For no-clear, we use LOAD_OP_LOAD by starting a new render pass variant
        // Simplification: always use the same render pass (LOAD_OP_CLEAR), but for
        // multi-draw we just don't clear by using a second render pass.
        // For now, use the same render pass — the second draw will clear again.
        // TODO: proper no-clear variant. For the test, we'll draw both in one cmd buffer.

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

            let viewport = vk::Viewport {
                x: 0.0, y: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            }]);

            device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
            device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);

            let pc_bytes = bytemuck::cast_slice::<f32, u8>(push_constants);
            device.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::VERTEX, 0, pc_bytes);

            device.cmd_draw_indexed(cmd, indices.len() as u32, 1, 0, 0, 0);

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd)?;
        }

        // Submit and wait
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(vb, None);
            device.free_memory(vb_mem, None);
            device.destroy_buffer(ib, None);
            device.free_memory(ib_mem, None);
        }

        Ok(())
    }

    /// Read back the depth buffer as a Vec<f32>.
    pub fn read_depth(&self, ctx: &VulkanContext) -> Result<Vec<f32>> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();
        let pixel_count = (self.width * self.height) as usize;
        let byte_count = pixel_count * 4; // D32_SFLOAT = 4 bytes

        let (staging, staging_mem) = create_host_buffer(
            ctx, byte_count as u64, vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });

            device.cmd_copy_image_to_buffer(
                cmd, self.depth_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging, &[region],
            );

            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        let mut result = vec![0.0f32; pixel_count];
        unsafe {
            let ptr = device.map_memory(staging_mem, 0, byte_count as u64, vk::MemoryMapFlags::empty())? as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), pixel_count);
            device.unmap_memory(staging_mem);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_mem, None);
        }

        Ok(result)
    }

    /// Expose the command pool for external command buffer allocation.
    pub fn command_pool(&self) -> vk::CommandPool { self.command_pool }

    /// Record fullscreen draw commands into an existing command buffer.
    pub fn record_draw_fullscreen(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline: &GraphicsPipeline,
        push_constants: &[u8],
        descriptor_set: vk::DescriptorSet,
    ) {
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        unsafe {
            device.cmd_begin_render_pass(cmd, &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_viewport(cmd, 0, &[vk::Viewport {
                x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            }]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            }]);

            let desc_sets = [descriptor_set];
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &desc_sets, &[]);

            if !push_constants.is_empty() {
                device.cmd_push_constants(cmd, pipeline.layout,
                    pipeline.push_constant_stages,
                    0, push_constants);
            }

            device.cmd_draw(cmd, 3, 1, 0, 0);

            device.cmd_end_render_pass(cmd);
        }
    }

    /// Draw a fullscreen triangle with no vertex buffer (3 vertices generated in shader).
    pub fn draw_fullscreen(
        &mut self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        push_constants: &[u8],
        descriptor_set: vk::DescriptorSet,
    ) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            device.cmd_begin_render_pass(cmd, &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_viewport(cmd, 0, &[vk::Viewport {
                x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            }]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            }]);

            let desc_sets = [descriptor_set];
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &desc_sets, &[]);

            if !push_constants.is_empty() {
                device.cmd_push_constants(cmd, pipeline.layout,
                    pipeline.push_constant_stages,
                    0, push_constants);
            }

            device.cmd_draw(cmd, 3, 1, 0, 0); // fullscreen triangle

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

    /// Read back the color buffer as Vec<[u8; 4]> (RGBA).
    pub fn read_color(&self, ctx: &VulkanContext) -> Result<Vec<[u8; 4]>> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();
        let pixel_count = (self.width * self.height) as usize;
        let byte_count = pixel_count * 4;

        let (staging, staging_mem) = create_host_buffer(
            ctx, byte_count as u64, vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });

            device.cmd_copy_image_to_buffer(
                cmd, self.color_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging, &[region],
            );

            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        let mut result = vec![[0u8; 4]; pixel_count];
        unsafe {
            let ptr = device.map_memory(staging_mem, 0, byte_count as u64, vk::MemoryMapFlags::empty())? as *const [u8; 4];
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), pixel_count);
            device.unmap_memory(staging_mem);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_mem, None);
        }

        Ok(result)
    }

    /// Draw with a descriptor set bound (for DDA shader with 3D texture).
    pub fn draw_with_descriptors(
        &mut self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        vertices: &[[f32; 3]],
        indices: &[u32],
        push_constants: &[u8],
        descriptor_set: vk::DescriptorSet,
    ) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        let vert_bytes = bytemuck::cast_slice::<[f32; 3], u8>(vertices);
        let idx_bytes = bytemuck::cast_slice::<u32, u8>(indices);

        let (vb, vb_mem) = create_host_buffer(ctx, vert_bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (ib, ib_mem) = create_host_buffer(ctx, idx_bytes.len() as u64, vk::BufferUsageFlags::INDEX_BUFFER)?;

        unsafe {
            let vptr = device.map_memory(vb_mem, 0, vert_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(vert_bytes.as_ptr(), vptr, vert_bytes.len());
            device.unmap_memory(vb_mem);

            let iptr = device.map_memory(ib_mem, 0, idx_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(idx_bytes.as_ptr(), iptr, idx_bytes.len());
            device.unmap_memory(ib_mem);
        }

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

            let viewport = vk::Viewport {
                x: 0.0, y: 0.0,
                width: self.width as f32, height: self.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            }]);

            // Bind descriptor set
            let desc_sets = [descriptor_set];
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout, 0, &desc_sets, &[],
            );

            device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
            device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);

            device.cmd_push_constants(
                cmd, pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0, push_constants,
            );

            device.cmd_draw_indexed(cmd, indices.len() as u32, 1, 0, 0, 0);

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
            device.destroy_buffer(vb, None);
            device.free_memory(vb_mem, None);
            device.destroy_buffer(ib, None);
            device.free_memory(ib_mem, None);
        }

        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.color_image, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.color_memory, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn create_image(
    ctx: &VulkanContext, w: u32, h: u32,
    format: vk::Format, usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let device = ctx.device();
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1).array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage);
    let image = unsafe { device.create_image(&ci, None) }?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mem_type = find_memory_type(&props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .context("No device-local memory")?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&ai, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;
    Ok((image, memory))
}

fn create_image_view(
    device: &ash::Device, image: vk::Image,
    format: vk::Format, aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect, base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        });
    Ok(unsafe { device.create_image_view(&ci, None) }?)
}

fn create_host_buffer(
    ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let device = ctx.device();
    let ci = vk::BufferCreateInfo::default().size(size).usage(usage);
    let buffer = unsafe { device.create_buffer(&ci, None) }?;
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mem_type = find_memory_type(
        &props, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).context("No host-visible memory")?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&ai, None) }?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32, required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(required)
        {
            return Some(i);
        }
    }
    None
}
