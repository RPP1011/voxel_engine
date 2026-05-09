//! `voxel_engine::ui` — egui overlay rendering.
//!
//! Bridges three crates:
//! - [`egui`] for immediate-mode UI state.
//! - [`egui_winit`] for routing winit input events into egui.
//! - [`egui_ash_renderer`] for painting egui's `ClippedPrimitive`
//!   list via Vulkan.
//!
//! Lives behind the `app-harness` cargo feature so non-windowed
//! consumers don't pay the dep cost.
//!
//! ## Lifecycle (per frame)
//!
//! ```text
//! handle_window_event(...)            // each WindowEvent → egui input
//! begin_frame(window)                 // start collecting raw input
//! ctx.run(input, |ctx| user.paint())  // user emits panels / plots
//! end_frame(window)                   // → ClippedPrimitive list
//! cmd_paint(cmd_buf, framebuffer, …)  // record draw commands
//! ```
//!
//! The paint pass runs INSIDE its own render pass on the swapchain
//! image, with `LOAD_OP_LOAD` on the colour attachment so anything
//! already there (the voxel render that the swapchain blit just
//! produced) stays underneath.

#![cfg(feature = "app-harness")]

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use ash::vk;
use egui::{ClippedPrimitive, FullOutput};
use egui_ash_renderer::{Options, Renderer as EguiRenderer};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::vulkan::instance::VulkanContext;

/// Owns the egui context, the winit input bridge, the Vulkan
/// painter, the render pass + per-swapchain-image framebuffers, and
/// a private gpu_allocator (separate from voxel_engine's main one
/// so the lifetimes are fully owned by this struct).
pub struct EguiState {
    ctx: egui::Context,
    winit: egui_winit::State,
    renderer: EguiRenderer,
    /// Arc<Mutex<>> required by egui-ash-renderer's allocator API.
    allocator: Arc<Mutex<Allocator>>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    extent: vk::Extent2D,
    /// Re-acquired each `begin_frame` from winit's pixel scale.
    pixels_per_point: f32,
    /// Carries paint output between `end_frame` and `cmd_paint` —
    /// allows callers to call them in separate borrow scopes.
    pending: Option<PendingPaint>,
}

struct PendingPaint {
    primitives: Vec<ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
}

impl EguiState {
    /// Construct given an existing VulkanContext + the swapchain's
    /// colour format, image views, and per-image extent.
    pub fn new(
        ctx: &VulkanContext,
        swapchain_format: vk::Format,
        swapchain_image_views: &[vk::ImageView],
        extent: vk::Extent2D,
        window: &Window,
    ) -> Result<Self> {
        let device = ctx.device();
        let physical_device = ctx.physical_device();
        let instance = ctx.instance();

        // Private allocator scoped to this state. Shared with
        // egui-ash-renderer via Arc<Mutex<>> because that's the
        // shape its `with_gpu_allocator` API requires.
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .context("egui allocator")?;
        let allocator = Arc::new(Mutex::new(allocator));

        let render_pass = make_egui_render_pass(device, swapchain_format)?;
        let framebuffers =
            make_framebuffers(device, render_pass, swapchain_image_views, extent)?;

        let renderer = EguiRenderer::with_gpu_allocator(
            allocator.clone(),
            device.clone(),
            render_pass,
            Options {
                in_flight_frames: swapchain_image_views.len(),
                srgb_framebuffer: false,
                enable_depth_test: false,
                enable_depth_write: false,
            },
        )
        .map_err(|e| anyhow::anyhow!("egui renderer init: {e:?}"))?;

        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();
        let winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        Ok(Self {
            ctx: egui_ctx,
            winit: winit_state,
            renderer,
            allocator,
            render_pass,
            framebuffers,
            extent,
            pixels_per_point: window.scale_factor() as f32,
            pending: None,
        })
    }

    /// Forward a winit event to egui. Returns the response —
    /// callers should consume `response.consumed` to decide whether
    /// to also process the event themselves.
    pub fn handle_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit.on_window_event(window, event)
    }

    /// Run egui for the frame. `paint` is called with the egui
    /// context inside the frame; emit panels, plots, etc.
    /// Stores the resulting paint output for [`Self::cmd_paint`].
    pub fn run<F: FnMut(&egui::Context)>(&mut self, window: &Window, paint: F) {
        let raw_input = self.winit.take_egui_input(window);
        let FullOutput {
            shapes,
            textures_delta,
            pixels_per_point,
            platform_output,
            ..
        } = self.ctx.run(raw_input, paint);
        self.winit.handle_platform_output(window, platform_output);
        self.pixels_per_point = pixels_per_point;
        let primitives = self.ctx.tessellate(shapes, pixels_per_point);
        self.pending = Some(PendingPaint {
            primitives,
            textures_delta,
        });
    }

    /// Record egui draw commands into `cmd_buffer` for the swapchain
    /// image at `image_index`. Caller must have transitioned the
    /// swapchain image to COLOR_ATTACHMENT_OPTIMAL with `LOAD_OP_LOAD`
    /// semantics already implied by our render pass setup.
    pub fn cmd_paint(
        &mut self,
        ctx: &VulkanContext,
        cmd_buffer: vk::CommandBuffer,
        image_index: usize,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(()); // no UI emitted this frame; nothing to paint
        };

        // Texture upload (font atlas, user images) MUST happen
        // before cmd_draw — egui-ash-renderer requires its own
        // queue submit for staging copies, so it runs synchronously
        // outside our cmd_buffer.
        if !pending.textures_delta.is_empty() {
            self.renderer
                .set_textures(
                    queue,
                    command_pool,
                    pending.textures_delta.set.as_slice(),
                )
                .map_err(|e| anyhow::anyhow!("egui set_textures: {e:?}"))?;
            // Process texture frees AFTER set so any newly-uploaded
            // texture isn't immediately freed.
            if !pending.textures_delta.free.is_empty() {
                self.renderer
                    .free_textures(pending.textures_delta.free.as_slice())
                    .map_err(|e| anyhow::anyhow!("egui free_textures: {e:?}"))?;
            }
        }

        let device = ctx.device();
        let framebuffer = self.framebuffers[image_index];
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });
        unsafe {
            device.cmd_begin_render_pass(
                cmd_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
        }
        self.renderer
            .cmd_draw(
                cmd_buffer,
                self.extent,
                self.pixels_per_point,
                &pending.primitives,
            )
            .map_err(|e| anyhow::anyhow!("egui cmd_draw: {e:?}"))?;
        unsafe {
            device.cmd_end_render_pass(cmd_buffer);
        }
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        // Renderer has its own Drop impl; just dropping it cleans
        // up its Vulkan resources via the allocator we shared.
        drop(self.renderer);
        let device = ctx.device();
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_render_pass(self.render_pass, None);
        }
        // self.allocator drops here.
    }
}

fn make_egui_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        // LOAD: keep what's already in the swapchain image (the
        // voxel render the present_blit just put there).
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // We expect the image to already be in COLOR_ATTACHMENT_OPTIMAL
        // when this render pass starts (or compatible — the caller's
        // present_blit transitions to TRANSFER_DST, then to whatever
        // we pick here). Going COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC
        // for the final layout so the post-pass barrier matches the
        // present expectation.
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_refs = [color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);

    let attachments = [color_attachment];
    let subpasses = [subpass];
    // External → subpass 0: wait for any previous color writes /
    // transfer ops to complete before egui starts drawing.
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
        )
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::TRANSFER_WRITE,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )];

    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let pass = unsafe { device.create_render_pass(&info, None) }.context("egui render pass")?;
    Ok(pass)
}

fn make_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    let mut out = Vec::with_capacity(image_views.len());
    for &view in image_views {
        let attachments = [view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        out.push(unsafe { device.create_framebuffer(&info, None) }.context("egui framebuffer")?);
    }
    Ok(out)
}
