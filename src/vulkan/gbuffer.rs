use anyhow::{Context, Result};
use ash::vk;

use super::graphics_pipeline::GraphicsPipeline;
use super::instance::VulkanContext;

/// G-buffer with 3 color attachments + depth.
/// (Previously 5 — velocity and linear-depth RTs were written per frame but
/// never sampled by any downstream pass, so they were pure ROP overhead.)
pub struct GBuffer {
    pub width: u32,
    pub height: u32,
    pub render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    // RT0: Albedo RGBA8, RT1: Normal RGBA16F, RT2: Material RGBA8
    rt_images: Vec<vk::Image>,
    rt_memories: Vec<vk::DeviceMemory>,
    rt_views: Vec<vk::ImageView>,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    command_pool: vk::CommandPool,
    // Shared unit cube [0,1]^3 — reused for every object
    unit_vb: vk::Buffer,
    unit_vb_mem: vk::DeviceMemory,
    unit_ib: vk::Buffer,
    unit_ib_mem: vk::DeviceMemory,
}

struct RtSpec {
    format: vk::Format,
}

const RT_SPECS: [RtSpec; 3] = [
    RtSpec { format: vk::Format::R8G8B8A8_UNORM },     // albedo
    RtSpec { format: vk::Format::R16G16B16A16_SFLOAT },// normal
    RtSpec { format: vk::Format::R8G8B8A8_UNORM },     // material
];

impl GBuffer {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().context("No graphics queue")?;

        // Build attachment descriptions
        let mut attachments = Vec::new();
        for spec in &RT_SPECS {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(spec.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            );
        }
        // Depth attachment (index 3)
        attachments.push(
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        );

        let color_refs: Vec<vk::AttachmentReference> = (0..3)
            .map(|i| vk::AttachmentReference {
                attachment: i,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect();
        let depth_ref = vk::AttachmentReference {
            attachment: 3,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_ref);

        let subpasses = [subpass];
        let rp_ci = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        let render_pass = unsafe { device.create_render_pass(&rp_ci, None) }?;

        // Create RT images
        let mut rt_images = Vec::new();
        let mut rt_memories = Vec::new();
        let mut rt_views = Vec::new();

        for spec in &RT_SPECS {
            let (img, mem) = create_image(
                ctx, width, height, spec.format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
            )?;
            let view = create_image_view(device, img, spec.format, vk::ImageAspectFlags::COLOR)?;
            rt_images.push(img);
            rt_memories.push(mem);
            rt_views.push(view);
        }

        let (depth_image, depth_memory) = create_image(
            ctx, width, height, vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let depth_view = create_image_view(device, depth_image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH)?;

        let mut all_views: Vec<vk::ImageView> = rt_views.clone();
        all_views.push(depth_view);

        let fb_ci = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&all_views)
            .width(width).height(height).layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&fb_ci, None) }?;

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gq.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }?;

        // Shared unit cube [0,1]^3 vertex + index buffers (created once, reused every frame)
        let cube_verts: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let cube_indices: [u32; 36] = [
            0,1,2, 2,3,0,  4,6,5, 6,4,7,  0,3,7, 7,4,0,
            1,5,6, 6,2,1,  3,2,6, 6,7,3,  0,4,5, 5,1,0,
        ];
        let vb_bytes = bytemuck::cast_slice::<[f32; 3], u8>(&cube_verts);
        let ib_bytes = bytemuck::cast_slice::<u32, u8>(&cube_indices);
        let (unit_vb, unit_vb_mem) = create_host_buffer(ctx, vb_bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (unit_ib, unit_ib_mem) = create_host_buffer(ctx, ib_bytes.len() as u64, vk::BufferUsageFlags::INDEX_BUFFER)?;
        unsafe {
            let p = device.map_memory(unit_vb_mem, 0, vb_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(vb_bytes.as_ptr(), p, vb_bytes.len());
            device.unmap_memory(unit_vb_mem);
            let p = device.map_memory(unit_ib_mem, 0, ib_bytes.len() as u64, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(ib_bytes.as_ptr(), p, ib_bytes.len());
            device.unmap_memory(unit_ib_mem);
        }

        Ok(Self {
            width, height, render_pass, framebuffer,
            rt_images, rt_memories, rt_views,
            depth_image, depth_memory, depth_view,
            command_pool,
            unit_vb, unit_vb_mem, unit_ib, unit_ib_mem,
        })
    }

    /// Get image view for a specific RT (0=albedo, 1=normal, 2=material, 3=velocity, 4=depth).
    pub fn rt_view(&self, index: usize) -> vk::ImageView {
        self.rt_views[index]
    }

    /// Shared unit cube vertex buffer (8 verts, [0,1]^3).
    pub fn unit_vb(&self) -> vk::Buffer { self.unit_vb }

    /// Shared unit cube index buffer (36 indices, 12 triangles).
    pub fn unit_ib(&self) -> vk::Buffer { self.unit_ib }

    /// Transition color RTs from TRANSFER_SRC to SHADER_READ_ONLY for sampling.
    pub fn transition_for_sampling(&self, ctx: &VulkanContext) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();
        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];
        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            for &image in &self.rt_images {
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1,
                    });
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
            }
            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(())
    }

    /// Expose the command pool for external command buffer allocation.
    pub fn command_pool(&self) -> vk::CommandPool { self.command_pool }

    /// Record G-buffer batch rendering commands into an existing command buffer.
    /// Does NOT begin/end the command buffer or submit it.
    pub fn record_batch(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline: &GraphicsPipeline,
        objects: &[(&[u8], vk::DescriptorSet)],
    ) {
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // albedo
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // normal
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // material
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        unsafe {
            device.cmd_begin_render_pass(cmd, &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass).framebuffer(self.framebuffer)
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } })
                .clear_values(&clear_values), vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } }]);

            device.cmd_bind_vertex_buffers(cmd, 0, &[self.unit_vb], &[0]);
            device.cmd_bind_index_buffer(cmd, self.unit_ib, 0, vk::IndexType::UINT32);

            for (pc, ds) in objects {
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &[*ds], &[]);
                device.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc);
                device.cmd_draw_indexed(cmd, 36, 1, 0, 0, 0);
            }

            device.cmd_end_render_pass(cmd);
        }
    }

    /// Record transition barriers into an existing command buffer.
    pub fn record_transition_for_sampling(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            for &image in &self.rt_images {
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1,
                    });
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
            }
        }
    }

    /// Render multiple objects into the G-buffer in a single render pass (proper depth testing between objects).
    /// Render all objects using the shared unit cube geometry.
    /// Each object only needs push constants and a descriptor set.
    pub fn render_batch(
        &self,
        ctx: &VulkanContext,
        pipeline: &GraphicsPipeline,
        objects: &[(&[u8], vk::DescriptorSet)], // (push_constants, desc_set)
    ) -> Result<()> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // albedo
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // normal
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // material
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

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

            // Bind shared unit cube once
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.unit_vb], &[0]);
            device.cmd_bind_index_buffer(cmd, self.unit_ib, 0, vk::IndexType::UINT32);

            for (pc, ds) in objects {
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.layout, 0, &[*ds], &[]);
                device.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc);
                device.cmd_draw_indexed(cmd, 36, 1, 0, 0, 0);
            }

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

    /// Render into the G-buffer with a pipeline that outputs to all 5 color attachments.
    pub fn render(
        &self,
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
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // albedo
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // normal
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }, // material
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
            device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
            device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
            device.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, push_constants);
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
            device.destroy_buffer(vb, None); device.free_memory(vb_mem, None);
            device.destroy_buffer(ib, None); device.free_memory(ib_mem, None);
        }
        Ok(())
    }

    /// Read back a color RT as raw bytes. `rt_index` is 0..4.
    pub fn read_rt(&self, ctx: &VulkanContext, rt_index: usize) -> Result<Vec<u8>> {
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();
        let pixel_count = (self.width * self.height) as usize;
        let bpp = match RT_SPECS[rt_index].format {
            vk::Format::R8G8B8A8_UNORM => 4,
            vk::Format::R16G16B16A16_SFLOAT => 8,
            vk::Format::R32_SFLOAT => 4,
            _ => 4,
        };
        let byte_count = pixel_count * bpp;

        let (staging, staging_mem) = create_host_buffer(ctx, byte_count as u64, vk::BufferUsageFlags::TRANSFER_DST)?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            // Transition from SHADER_READ_ONLY to TRANSFER_SRC for copy
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.rt_images[rt_index])
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1,
                });
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });
            device.cmd_copy_image_to_buffer(cmd, self.rt_images[rt_index], vk::ImageLayout::TRANSFER_SRC_OPTIMAL, staging, &[region]);
            device.end_command_buffer(cmd)?;

            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(gq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        let mut result = vec![0u8; byte_count];
        unsafe {
            let ptr = device.map_memory(staging_mem, 0, byte_count as u64, vk::MemoryMapFlags::empty())? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), byte_count);
            device.unmap_memory(staging_mem);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_mem, None);
        }
        Ok(result)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            for v in &self.rt_views { device.destroy_image_view(*v, None); }
            for i in &self.rt_images { device.destroy_image(*i, None); }
            for m in &self.rt_memories { device.free_memory(*m, None); }
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_buffer(self.unit_vb, None);
            device.free_memory(self.unit_vb_mem, None);
            device.destroy_buffer(self.unit_ib, None);
            device.free_memory(self.unit_ib_mem, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn create_image(ctx: &VulkanContext, w: u32, h: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> Result<(vk::Image, vk::DeviceMemory)> {
    let device = ctx.device();
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D).format(format)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL).usage(usage);
    let image = unsafe { device.create_image(&ci, None) }?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mt = (0..props.memory_type_count).find(|&i| (req.memory_type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)).context("No device-local mem")?;
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
    let mt = (0..props.memory_type_count).find(|&i| (req.memory_type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)).context("No host-visible mem")?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
    let memory = unsafe { device.allocate_memory(&ai, None) }?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}
