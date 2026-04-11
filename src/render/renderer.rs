use anyhow::Result;
use ash::vk;
use ash::vk::Handle;

use crate::vulkan::gbuffer::GBuffer;
use crate::vulkan::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineBuilder};
use crate::vulkan::instance::VulkanContext;
use crate::vulkan::render_target::OffscreenTarget;
use crate::vulkan::shadow_map::ShadowMap;
use crate::terrain_compute::LoadedChunkView;
use crate::vulkan::voxel_gpu::GpuVoxelTexture;

use crate::camera::OrbitCamera;
use crate::camera::FreeCamera;
use glam::Vec3;

/// Trait for cameras compatible with the renderer.
pub trait RenderCamera {
    fn eye_position(&self) -> [f32; 3];
    fn view_matrix_array(&self) -> [f32; 16];
    fn projection_matrix_array(&self, aspect: f32) -> [f32; 16];
    fn center(&self) -> Vec3;
}

impl RenderCamera for OrbitCamera {
    fn eye_position(&self) -> [f32; 3] { self.eye_position() }
    fn view_matrix_array(&self) -> [f32; 16] { self.view_matrix_array() }
    fn projection_matrix_array(&self, aspect: f32) -> [f32; 16] { self.projection_matrix_array(aspect) }
    fn center(&self) -> Vec3 { self.center() }
}

impl RenderCamera for FreeCamera {
    fn eye_position(&self) -> [f32; 3] { self.eye_position() }
    fn view_matrix_array(&self) -> [f32; 16] { self.view_matrix_array() }
    fn projection_matrix_array(&self, aspect: f32) -> [f32; 16] { self.projection_matrix_array(aspect) }
    fn center(&self) -> Vec3 { self.center() }
}

/// Full rendering pipeline orchestrator.
/// Owns GBuffer, ShadowMap, offscreen targets, all pipelines, samplers, and descriptor pools.
pub struct VoxelRenderer {
    width: u32,
    height: u32,
    gbuffer: GBuffer,
    shadow_map: ShadowMap,
    light_target: OffscreenTarget,
    tonemap_target: OffscreenTarget,
    gbuffer_pipeline: GraphicsPipeline,
    shadow_pipeline: GraphicsPipeline,
    light_pipeline: GraphicsPipeline,
    tonemap_pipeline: GraphicsPipeline,
    sampler: vk::Sampler,
    // Descriptor pools for each pass
    light_desc_pool: vk::DescriptorPool,
    light_desc_set: vk::DescriptorSet,
    tonemap_desc_pool: vk::DescriptorPool,
    tonemap_desc_set: vk::DescriptorSet,
    // Cached per-object descriptor pool + sets (reused across frames)
    obj_desc_pool: Option<vk::DescriptorPool>,
    obj_gbuf_desc_sets: Vec<vk::DescriptorSet>,
    obj_shadow_desc_set: Option<vk::DescriptorSet>,
    /// Hash of image view handles used to build the cached descriptor sets.
    /// When this changes (texture upload/evict), we rebuild.
    cached_obj_hash: u64,
    /// Persistent per-slot cache for the pool-rendering path. Keyed by the
    /// chunk's main image view handle (stable per pool slot). Populated
    /// lazily by `rebuild_pool_descriptors`. This replaces the old destroy-
    /// and-recreate pool-rebuild on every visible-set change, which was the
    /// source of multi-millisecond raycast spikes during pool fill-up.
    obj_pool_desc_cache: std::collections::HashMap<u64, vk::DescriptorSet>,
    voxel_sampler: vk::Sampler,
    // Pre-allocated Vulkan objects to avoid per-frame alloc/free
    render_cmd: [vk::CommandBuffer; 2],
    render_fence: vk::Fence,
    /// Binary semaphore signaled by each `render_frame_pool` submit and
    /// consumed by the subsequent `present_blit` submit. Lets the CPU return
    /// from `render_frame_pool` without waiting for the GPU render to finish,
    /// while still ensuring `present_blit`'s blit reads a finished frame.
    /// Safe as a single binary semaphore because the start-of-frame
    /// `render_fence` wait caps us at one render in flight at a time.
    render_done_semaphore: vk::Semaphore,
    frame_index: usize,
    // Reusable per-frame Vec buffers (cleared each frame, capacity preserved)
    batch_push_buf: Vec<[u8; 176]>,
}

impl VoxelRenderer {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        let device = ctx.device();

        let gbuffer = GBuffer::new(ctx, width, height)?;
        let shadow_map = ShadowMap::new(ctx, width, height)?;
        let mut light_target = OffscreenTarget::new(ctx, width, height)?;
        let mut tonemap_target = OffscreenTarget::new(ctx, width, height)?;

        // Create samplers
        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_ci, None) }?;

        // Nearest sampler for voxel 3D texture (integer format needs nearest)
        let voxel_sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let voxel_sampler = unsafe { device.create_sampler(&voxel_sampler_ci, None) }?;

        // Build gbuffer pipeline
        let gbuffer_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gbuffer.vert.spv"));
        let gbuffer_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gbuffer.frag.spv"));
        let gbuffer_pipeline = GraphicsPipelineBuilder::new(ctx)
            .vertex_shader(gbuffer_vert_spv)
            .fragment_shader(gbuffer_frag_spv)
            .render_pass(gbuffer.render_pass)
            .push_constant_size(176, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .descriptor(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .color_attachment_count(1)
            .cull_mode(vk::CullModeFlags::NONE) // no HW culling; back faces discarded in frag shader
            .build()?;

        // Build shadow pipeline (reuses gbuffer.vert + shadow_map.frag)
        let shadow_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/shadow_map.frag.spv"));
        let shadow_pipeline = GraphicsPipelineBuilder::new(ctx)
            .vertex_shader(gbuffer_vert_spv)
            .fragment_shader(shadow_frag_spv)
            .render_pass(shadow_map.render_pass)
            .push_constant_size(128, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .descriptor(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .color_attachment_count(0)
            .build()?;

        // Build deferred light pipeline
        let fullscreen_vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/fullscreen.vert.spv"));
        let deferred_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/deferred_sun.frag.spv"));
        let light_pipeline = GraphicsPipelineBuilder::new(ctx)
            .vertex_shader(fullscreen_vert_spv)
            .fragment_shader(deferred_frag_spv)
            .render_pass(light_target.render_pass())
            .push_constant_size(48, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .no_vertex_input()
            .no_depth_test()
            .build()?;

        // Build tonemap pipeline
        let tonemap_frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/tonemap.frag.spv"));
        let tonemap_pipeline = GraphicsPipelineBuilder::new(ctx)
            .vertex_shader(fullscreen_vert_spv)
            .fragment_shader(tonemap_frag_spv)
            .render_pass(tonemap_target.render_pass())
            .push_constant_size(16, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .descriptor(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .no_vertex_input()
            .no_depth_test()
            .build()?;

        // Allocate light pass descriptor set (3 GBuffer RTs: albedo, normal, material)
        let light_desc_pool = {
            let ps = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(3)];
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&ps),
                    None,
                )
            }?
        };
        let light_desc_set = {
            let layouts = [light_pipeline.descriptor_set_layout.unwrap()];
            let sets = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(light_desc_pool)
                        .set_layouts(&layouts),
                )
            }?;
            sets[0]
        };

        // Allocate tonemap descriptor set (HDR input + history)
        let tonemap_desc_pool = {
            let ps = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2)];
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&ps),
                    None,
                )
            }?
        };
        let tonemap_desc_set = {
            let layouts = [tonemap_pipeline.descriptor_set_layout.unwrap()];
            let sets = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(tonemap_desc_pool)
                        .set_layouts(&layouts),
                )
            }?;
            sets[0]
        };

        // Pre-allocate 2 command buffers from the gbuffer command pool (supports reset)
        let render_cmd = {
            let alloc_ci = vk::CommandBufferAllocateInfo::default()
                .command_pool(gbuffer.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(2);
            let cmds = unsafe { device.allocate_command_buffers(&alloc_ci) }?;
            [cmds[0], cmds[1]]
        };

        // Pre-allocate a fence (created signaled so the first frame's wait succeeds)
        let render_fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }?;

        // Binary semaphore for render→present handoff. Unsignaled at creation;
        // the first frame must NOT have anyone waiting on it before it's first
        // signaled. The call graph guarantees render_frame_pool runs before
        // present_blit every frame, so this holds.
        let render_done_semaphore = unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }?;

        // NOTE: the light pass has been merged into gbuffer.frag and is no
        // longer called. We still allocate light_desc_set for API compat
        // but we write it to point at the (single) gbuffer RT0 three times
        // — the light pipeline is never bound so the contents don't matter,
        // but Vulkan validation requires the set be initialized.
        let light_img_infos = [
            vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(gbuffer.rt_view(0))
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            3
        ];
        let light_writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(light_desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&light_img_infos[i]))
            })
            .collect();
        unsafe { device.update_descriptor_sets(&light_writes, &[]) };

        Ok(Self {
            width,
            height,
            gbuffer,
            shadow_map,
            light_target,
            tonemap_target,
            gbuffer_pipeline,
            shadow_pipeline,
            light_pipeline,
            tonemap_pipeline,
            sampler,
            light_desc_pool,
            light_desc_set,
            tonemap_desc_pool,
            tonemap_desc_set,
            obj_desc_pool: None,
            obj_gbuf_desc_sets: Vec::new(),
            obj_shadow_desc_set: None,
            cached_obj_hash: 0,
            obj_pool_desc_cache: std::collections::HashMap::new(),
            voxel_sampler,
            render_cmd,
            render_fence,
            render_done_semaphore,
            frame_index: 0,
            batch_push_buf: Vec::new(),
        })
    }

    /// Rebuild cached per-object descriptor sets when the object list changes.
    /// Skips rebuild if the same set of textures is passed as last time.
    fn rebuild_object_descriptors(
        &mut self,
        ctx: &VulkanContext,
        objects: &[(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])],
    ) -> Result<()> {
        // Hash the image view handles to detect changes (FNV-1a style).
        let mut obj_hash: u64 = 0xcbf29ce484222325;
        obj_hash ^= objects.len() as u64;
        obj_hash = obj_hash.wrapping_mul(0x100000001b3);
        for (tex, _, _, _) in objects {
            // Use the raw Vulkan handle as a unique ID for each texture.
            obj_hash ^= tex.image_view.as_raw() as u64;
            obj_hash = obj_hash.wrapping_mul(0x100000001b3);
        }
        if obj_hash == self.cached_obj_hash && self.obj_desc_pool.is_some() {
            return Ok(());
        }

        let device = ctx.device();

        // Destroy old pool if it exists (frees all sets allocated from it)
        if let Some(pool) = self.obj_desc_pool.take() {
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }
        self.obj_gbuf_desc_sets.clear();
        self.obj_shadow_desc_set = None;

        let obj_count = objects.len().max(1) as u32;

        // Need: obj_count gbuffer sets (5 samplers each) + 1 shadow set (1 sampler)
        let pool = {
            let ps = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(obj_count * 5 + 1)];
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(obj_count + 1)
                        .pool_sizes(&ps),
                    None,
                )
            }?
        };

        // Allocate gbuffer descriptor sets and write image bindings
        let gbuf_dsl = self.gbuffer_pipeline.descriptor_set_layout.unwrap();
        for (gpu_tex, _palette_color, _position, _grid_dims) in objects.iter() {
            let layouts = [gbuf_dsl];
            let desc_set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }?[0];

            let img_infos = [
                vk::DescriptorImageInfo::default()
                    .sampler(self.voxel_sampler)
                    .image_view(gpu_tex.image_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                vk::DescriptorImageInfo::default()
                    .sampler(self.voxel_sampler)
                    .image_view(gpu_tex.mip1_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                vk::DescriptorImageInfo::default()
                    .sampler(self.voxel_sampler)
                    .image_view(gpu_tex.mip2_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                vk::DescriptorImageInfo::default()
                    .sampler(self.sampler)
                    .image_view(gpu_tex.palette_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                vk::DescriptorImageInfo::default()
                    .sampler(self.voxel_sampler)
                    .image_view(gpu_tex.mip3_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..5)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&img_infos[i]))
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };

            self.obj_gbuf_desc_sets.push(desc_set);
        }

        // Allocate shadow descriptor set (first object's texture)
        if let Some((gpu_tex, _, _, _)) = objects.first() {
            let shadow_dsl = self.shadow_pipeline.descriptor_set_layout.unwrap();
            let shadow_layouts = [shadow_dsl];
            let shadow_desc = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&shadow_layouts),
                )
            }?[0];
            let shadow_img = [vk::DescriptorImageInfo::default()
                .sampler(self.voxel_sampler)
                .image_view(gpu_tex.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let shadow_write = [vk::WriteDescriptorSet::default()
                .dst_set(shadow_desc)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_img)];
            unsafe { device.update_descriptor_sets(&shadow_write, &[]) };
            self.obj_shadow_desc_set = Some(shadow_desc);
        }

        self.obj_desc_pool = Some(pool);
        self.cached_obj_hash = obj_hash;

        Ok(())
    }

    /// Render a complete frame and read back the final color buffer.
    ///
    /// `objects` is a slice of (gpu_texture, palette_color_rgba, world_position, grid_dim).
    pub fn render_frame(
        &mut self,
        ctx: &VulkanContext,
        camera: &impl RenderCamera,
        objects: &[(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])],
    ) -> Result<Vec<[u8; 4]>> {
        let aspect = self.width as f32 / self.height as f32;
        let view = camera.view_matrix_array();
        let proj = camera.projection_matrix_array(aspect);
        let eye = camera.eye_position();

        // Rebuild descriptor sets (textures may have changed even if count hasn't)
        self.rebuild_object_descriptors(ctx, objects)?;

        // ---- Step 1: GBuffer pass ----
        // Build push constants per object (descriptor sets are cached)
        self.batch_push_buf.clear();

        for (gpu_tex, palette_color, position, dims) in objects.iter() {
            // MVP bakes the scale (dims) into the model matrix so the unit cube
            // becomes [0, dims.x] x [0, dims.y] x [0, dims.z] in world space at the object's position.
            let model = scale3_translate_matrix(dims[0], dims[1], dims[2], position[0], position[1], position[2]);
            let mv = mat4_mul(&view, &model);
            let mvp = mat4_mul(&proj, &mv);

            let mut push_data = [0u8; 176];
            push_data[0..64].copy_from_slice(bytemuck::cast_slice(&mvp));
            let grid_dim_v = [dims[0], dims[1], dims[2], 0.0f32];
            push_data[64..80].copy_from_slice(bytemuck::cast_slice(&grid_dim_v));
            let cam_pos = [eye[0] - position[0], eye[1] - position[1], eye[2] - position[2], 0.0f32];
            push_data[80..96].copy_from_slice(bytemuck::cast_slice(&cam_pos));
            push_data[96..112].copy_from_slice(bytemuck::cast_slice(palette_color));
            let mip1_dim_v = [gpu_tex.mip1_width as f32, gpu_tex.mip1_height as f32, gpu_tex.mip1_depth as f32, 0.0f32];
            push_data[112..128].copy_from_slice(bytemuck::cast_slice(&mip1_dim_v));
            let mip2_dim_v = [gpu_tex.mip2_width as f32, gpu_tex.mip2_height as f32, gpu_tex.mip2_depth as f32, 0.0f32];
            push_data[128..144].copy_from_slice(bytemuck::cast_slice(&mip2_dim_v));
            let mip3_dim_v = [gpu_tex.mip3_width as f32, gpu_tex.mip3_height as f32, gpu_tex.mip3_depth as f32, 0.0f32];
            push_data[144..160].copy_from_slice(bytemuck::cast_slice(&mip3_dim_v));

            self.batch_push_buf.push(push_data);
        }

        let batch_objects: Vec<(&[u8], vk::DescriptorSet)> = self.batch_push_buf
            .iter()
            .zip(self.obj_gbuf_desc_sets.iter())
            .map(|(p, d)| (p.as_ref(), *d))
            .collect();

        // ---- Single command buffer for passes 1-4 (gbuffer, shadow, transition, light) ----
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        // Wait for previous frame using this command buffer to finish, then reset
        let cmd = self.render_cmd[self.frame_index % 2];
        unsafe {
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
            device.reset_fences(&[self.render_fence])?;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }

        // Pass 1: GBuffer
        self.gbuffer.record_batch(device, cmd, &self.gbuffer_pipeline, &batch_objects);

        // Pass 2: Shadow map (use first object as representative)
        if let Some((_gpu_tex, _palette_color, position, dims)) = objects.first() {
            let max_dim = dims[0].max(dims[1]).max(dims[2]);
            let sun_dir = normalize_v([0.5, 0.8, 0.3]);
            let sun_model = scale3_translate_matrix(dims[0], dims[1], dims[2], position[0], position[1], position[2]);
            let sun_view_mat = build_sun_view(&sun_dir, position, max_dim);
            let half = max_dim * 2.0;
            let dist = max_dim * 3.0;
            let sun_proj = ortho(-half, half, -half, half, 0.1, dist * 2.0);
            let sun_mvp = mat4_mul(&sun_proj, &mat4_mul(&sun_view_mat, &sun_model));
            let grid_dim_v = [dims[0], dims[1], dims[2], 0.0f32];

            let mut shadow_push = [0u8; 128];
            shadow_push[0..64].copy_from_slice(bytemuck::cast_slice(&sun_mvp));
            shadow_push[64..80].copy_from_slice(bytemuck::cast_slice(&grid_dim_v));
            let sun_eye = [position[0]+sun_dir[0]*dist, position[1]+sun_dir[1]*dist, position[2]+sun_dir[2]*dist];
            let sun_cam = [sun_eye[0], sun_eye[1], sun_eye[2], 0.0f32];
            shadow_push[80..96].copy_from_slice(bytemuck::cast_slice(&sun_cam));

            let shadow_desc = self.obj_shadow_desc_set.unwrap();
            self.shadow_map.record_render(
                device, cmd, &self.shadow_pipeline,
                self.gbuffer.unit_vb(), self.gbuffer.unit_ib(), 36,
                &shadow_push, shadow_desc,
            );
        }

        // Pass 3: Transition GBuffer RTs for sampling
        self.gbuffer.record_transition_for_sampling(device, cmd);

        // Pass 4: Deferred sun light pass (descriptor set written once in new())
        let sun_dir = normalize_v([0.5, 0.8, 0.3]);
        let center = camera.center();
        let view_dir = normalize_v([
            center.x - eye[0],
            center.y - eye[1],
            center.z - eye[2],
        ]);
        let mut light_push = [0u8; 48];
        let sun_dir_v = [sun_dir[0], sun_dir[1], sun_dir[2], 2.0f32]; // w = intensity
        let sun_color_v = [1.0f32, 0.95, 0.9, 1.0];
        let cam_dir_v = [view_dir[0], view_dir[1], view_dir[2], 0.0f32];
        light_push[0..16].copy_from_slice(bytemuck::cast_slice(&sun_dir_v));
        light_push[16..32].copy_from_slice(bytemuck::cast_slice(&sun_color_v));
        light_push[32..48].copy_from_slice(bytemuck::cast_slice(&cam_dir_v));

        self.light_target.record_draw_fullscreen(
            device, cmd, &self.light_pipeline, &light_push, self.light_desc_set,
        );

        // End and submit using pre-allocated fence
        unsafe {
            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.queue_submit(gq.queue, &[submit], self.render_fence)?;
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
        }
        self.frame_index += 1;

        // ---- Step 5: Read back from light target (separate submission for CPU readback) ----
        let color = self.light_target.read_color(ctx)?;

        Ok(color)
    }

    /// Return the light target's color image handle (for GPU-only blit presentation).
    pub fn light_output_image(&self) -> vk::Image {
        self.light_target.color_image()
    }

    /// Return the gbuffer's final color RT image (for GPU-only blit
    /// presentation). After the light-pass merge this is the post-light
    /// output and is in TRANSFER_SRC_OPTIMAL layout after `render_frame_pool`.
    pub fn gbuffer_output_image(&self) -> vk::Image {
        self.gbuffer.rt_image(0)
    }

    /// Render all passes on the GPU without reading back pixels.
    /// The result is left in `light_output_image()` in TRANSFER_SRC_OPTIMAL layout.
    pub fn render_frame_gpu(
        &mut self,
        ctx: &VulkanContext,
        camera: &impl RenderCamera,
        objects: &[(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])],
    ) -> Result<()> {
        let aspect = self.width as f32 / self.height as f32;
        let view = camera.view_matrix_array();
        let proj = camera.projection_matrix_array(aspect);
        let eye = camera.eye_position();

        // Rebuild descriptor sets (textures may have changed even if count hasn't)
        self.rebuild_object_descriptors(ctx, objects)?;

        // ---- Step 1: GBuffer pass ----
        self.batch_push_buf.clear();

        for (gpu_tex, palette_color, position, dims) in objects.iter() {
            let model = scale3_translate_matrix(dims[0], dims[1], dims[2], position[0], position[1], position[2]);
            let mv = mat4_mul(&view, &model);
            let mvp = mat4_mul(&proj, &mv);

            let mut push_data = [0u8; 176];
            push_data[0..64].copy_from_slice(bytemuck::cast_slice(&mvp));
            let grid_dim_v = [dims[0], dims[1], dims[2], 0.0f32];
            push_data[64..80].copy_from_slice(bytemuck::cast_slice(&grid_dim_v));
            let cam_pos = [eye[0] - position[0], eye[1] - position[1], eye[2] - position[2], 0.0f32];
            push_data[80..96].copy_from_slice(bytemuck::cast_slice(&cam_pos));
            push_data[96..112].copy_from_slice(bytemuck::cast_slice(palette_color));
            let mip1_dim_v = [gpu_tex.mip1_width as f32, gpu_tex.mip1_height as f32, gpu_tex.mip1_depth as f32, 0.0f32];
            push_data[112..128].copy_from_slice(bytemuck::cast_slice(&mip1_dim_v));
            let mip2_dim_v = [gpu_tex.mip2_width as f32, gpu_tex.mip2_height as f32, gpu_tex.mip2_depth as f32, 0.0f32];
            push_data[128..144].copy_from_slice(bytemuck::cast_slice(&mip2_dim_v));
            let mip3_dim_v = [gpu_tex.mip3_width as f32, gpu_tex.mip3_height as f32, gpu_tex.mip3_depth as f32, 0.0f32];
            push_data[144..160].copy_from_slice(bytemuck::cast_slice(&mip3_dim_v));

            self.batch_push_buf.push(push_data);
        }

        let batch_objects: Vec<(&[u8], vk::DescriptorSet)> = self.batch_push_buf
            .iter()
            .zip(self.obj_gbuf_desc_sets.iter())
            .map(|(p, d)| (p.as_ref(), *d))
            .collect();

        // ---- Single command buffer for passes 1-4 ----
        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        // Wait for previous frame using this command buffer to finish, then reset
        let cmd = self.render_cmd[self.frame_index % 2];
        unsafe {
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
            device.reset_fences(&[self.render_fence])?;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }

        // Pass 1: GBuffer
        self.gbuffer.record_batch(device, cmd, &self.gbuffer_pipeline, &batch_objects);

        // Pass 2: Shadow map
        if let Some((_gpu_tex, _palette_color, position, dims)) = objects.first() {
            let max_dim = dims[0].max(dims[1]).max(dims[2]);
            let sun_dir = normalize_v([0.5, 0.8, 0.3]);
            let sun_model = scale3_translate_matrix(dims[0], dims[1], dims[2], position[0], position[1], position[2]);
            let sun_view_mat = build_sun_view(&sun_dir, position, max_dim);
            let half = max_dim * 2.0;
            let dist = max_dim * 3.0;
            let sun_proj = ortho(-half, half, -half, half, 0.1, dist * 2.0);
            let sun_mvp = mat4_mul(&sun_proj, &mat4_mul(&sun_view_mat, &sun_model));
            let grid_dim_v = [dims[0], dims[1], dims[2], 0.0f32];

            let mut shadow_push = [0u8; 128];
            shadow_push[0..64].copy_from_slice(bytemuck::cast_slice(&sun_mvp));
            shadow_push[64..80].copy_from_slice(bytemuck::cast_slice(&grid_dim_v));
            let sun_eye = [position[0]+sun_dir[0]*dist, position[1]+sun_dir[1]*dist, position[2]+sun_dir[2]*dist];
            let sun_cam = [sun_eye[0], sun_eye[1], sun_eye[2], 0.0f32];
            shadow_push[80..96].copy_from_slice(bytemuck::cast_slice(&sun_cam));

            let shadow_desc = self.obj_shadow_desc_set.unwrap();
            self.shadow_map.record_render(
                device, cmd, &self.shadow_pipeline,
                self.gbuffer.unit_vb(), self.gbuffer.unit_ib(), 36,
                &shadow_push, shadow_desc,
            );
        }

        // Pass 3: Transition GBuffer RTs for sampling
        self.gbuffer.record_transition_for_sampling(device, cmd);

        // Pass 4: Deferred sun light pass (descriptor set written once in new())
        let sun_dir = normalize_v([0.5, 0.8, 0.3]);
        let center = camera.center();
        let view_dir = normalize_v([
            center.x - eye[0],
            center.y - eye[1],
            center.z - eye[2],
        ]);
        let mut light_push = [0u8; 48];
        let sun_dir_v = [sun_dir[0], sun_dir[1], sun_dir[2], 2.0f32];
        let sun_color_v = [1.0f32, 0.95, 0.9, 1.0];
        let cam_dir_v = [view_dir[0], view_dir[1], view_dir[2], 0.0f32];
        light_push[0..16].copy_from_slice(bytemuck::cast_slice(&sun_dir_v));
        light_push[16..32].copy_from_slice(bytemuck::cast_slice(&sun_color_v));
        light_push[32..48].copy_from_slice(bytemuck::cast_slice(&cam_dir_v));

        self.light_target.record_draw_fullscreen(
            device, cmd, &self.light_pipeline, &light_push, self.light_desc_set,
        );

        // End and submit using pre-allocated fence
        unsafe {
            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.queue_submit(gq.queue, &[submit], self.render_fence)?;
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
        }
        self.frame_index += 1;

        Ok(())
    }

    /// Populate the per-visible-chunk descriptor set list for the pool
    /// render path using a persistent cache keyed by main image view.
    ///
    /// The terrain compute pool holds at most 256 slots and each slot owns
    /// a stable set of image views (main + 3 mip levels) for the lifetime
    /// of the renderer. We lazily allocate and write one descriptor set
    /// per unique main_view the first time we see it, then reuse it on
    /// every subsequent frame. The previous implementation destroyed and
    /// rebuilt the entire pool every time the visible set changed — which
    /// was every frame during pool fill-up, costing up to ~2ms of raycast
    /// latency spikes from allocating 100+ descriptor sets at once.
    fn rebuild_pool_descriptors(
        &mut self,
        ctx: &VulkanContext,
        views: &[(LoadedChunkView, [f32; 4], [f32; 3], [f32; 3])],
        palette_view: vk::ImageView,
    ) -> Result<()> {
        let device = ctx.device();

        // One-time pool creation: size generously for the full pool slot
        // count (256) × 5 bindings each. Allocations are reused forever.
        if self.obj_desc_pool.is_none() {
            const POOL_SLOT_COUNT: u32 = 256;
            const BINDINGS_PER_SET: u32 = 5;
            let ps = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(POOL_SLOT_COUNT * BINDINGS_PER_SET + 8)];
            let pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(POOL_SLOT_COUNT + 8)
                        .pool_sizes(&ps),
                    None,
                )
            }?;
            self.obj_desc_pool = Some(pool);
        }
        let pool = self.obj_desc_pool.unwrap();
        let gbuf_dsl = self.gbuffer_pipeline.descriptor_set_layout.unwrap();

        // Rebuild the visible-chunk descriptor set list from the cache,
        // allocating + writing new entries only for chunks we haven't seen.
        self.obj_gbuf_desc_sets.clear();
        self.obj_gbuf_desc_sets.reserve(views.len());
        for (v, _palette_color, _position, _grid_dims) in views.iter() {
            let key = v.main_view.as_raw();
            let desc_set = match self.obj_pool_desc_cache.get(&key) {
                Some(&ds) => ds,
                None => {
                    let layouts = [gbuf_dsl];
                    let ds = unsafe {
                        device.allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(pool)
                                .set_layouts(&layouts),
                        )
                    }?[0];

                    // NOTE: binding layout matches gbuffer.frag:
                    //   0 = voxel_grid (main), 1 = mip1, 2 = mip2, 3 = palette, 4 = mip3
                    let img_infos = [
                        vk::DescriptorImageInfo::default()
                            .sampler(self.voxel_sampler)
                            .image_view(v.main_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::DescriptorImageInfo::default()
                            .sampler(self.voxel_sampler)
                            .image_view(v.mip1_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::DescriptorImageInfo::default()
                            .sampler(self.voxel_sampler)
                            .image_view(v.mip2_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::DescriptorImageInfo::default()
                            .sampler(self.sampler)
                            .image_view(palette_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::DescriptorImageInfo::default()
                            .sampler(self.voxel_sampler)
                            .image_view(v.mip3_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    ];
                    let writes: Vec<vk::WriteDescriptorSet> = (0..5)
                        .map(|i| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(i as u32)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(std::slice::from_ref(&img_infos[i]))
                        })
                        .collect();
                    unsafe { device.update_descriptor_sets(&writes, &[]) };

                    self.obj_pool_desc_cache.insert(key, ds);
                    ds
                }
            };
            self.obj_gbuf_desc_sets.push(desc_set);
        }
        Ok(())
    }

    /// Render all passes on the GPU sampling from pool-resident
    /// [`LoadedChunkView`] textures. Same contract as
    /// [`render_frame_gpu`] — the result is left in `light_output_image()`
    /// in TRANSFER_SRC_OPTIMAL, ready for a blit-to-swapchain.
    ///
    /// `views` is `(view, palette_color_rgba, world_position, grid_dim)` per
    /// drawn chunk. `palette_view` is a single 256×1 RGBA image shared by
    /// every chunk (typically owned by the `TerrainComputePipeline`).
    pub fn render_frame_pool(
        &mut self,
        ctx: &VulkanContext,
        camera: &impl RenderCamera,
        views: &[(LoadedChunkView, [f32; 4], [f32; 3], [f32; 3])],
        palette_view: vk::ImageView,
    ) -> Result<()> {
        let aspect = self.width as f32 / self.height as f32;
        let view_mat = camera.view_matrix_array();
        let proj = camera.projection_matrix_array(aspect);
        let eye = camera.eye_position();

        self.rebuild_pool_descriptors(ctx, views, palette_view)?;

        // ---- Build push constants per chunk view ----
        self.batch_push_buf.clear();

        for (v, palette_color, position, dims) in views.iter() {
            let model = scale3_translate_matrix(dims[0], dims[1], dims[2], position[0], position[1], position[2]);
            let mv = mat4_mul(&view_mat, &model);
            let mvp = mat4_mul(&proj, &mv);

            let mut push_data = [0u8; 176];
            push_data[0..64].copy_from_slice(bytemuck::cast_slice(&mvp));
            let grid_dim_v = [dims[0], dims[1], dims[2], 0.0f32];
            push_data[64..80].copy_from_slice(bytemuck::cast_slice(&grid_dim_v));
            let cam_pos = [eye[0] - position[0], eye[1] - position[1], eye[2] - position[2], 0.0f32];
            push_data[80..96].copy_from_slice(bytemuck::cast_slice(&cam_pos));
            push_data[96..112].copy_from_slice(bytemuck::cast_slice(palette_color));
            let mip1_dim_v = [v.mip1_dim[0] as f32, v.mip1_dim[1] as f32, v.mip1_dim[2] as f32, 0.0f32];
            push_data[112..128].copy_from_slice(bytemuck::cast_slice(&mip1_dim_v));
            let mip2_dim_v = [v.mip2_dim[0] as f32, v.mip2_dim[1] as f32, v.mip2_dim[2] as f32, 0.0f32];
            push_data[128..144].copy_from_slice(bytemuck::cast_slice(&mip2_dim_v));
            let mip3_dim_v = [v.mip3_dim[0] as f32, v.mip3_dim[1] as f32, v.mip3_dim[2] as f32, 0.0f32];
            push_data[144..160].copy_from_slice(bytemuck::cast_slice(&mip3_dim_v));

            self.batch_push_buf.push(push_data);
        }

        let batch_objects: Vec<(&[u8], vk::DescriptorSet)> = self.batch_push_buf
            .iter()
            .zip(self.obj_gbuf_desc_sets.iter())
            .map(|(p, d)| (p.as_ref(), *d))
            .collect();

        let device = ctx.device();
        let gq = ctx.graphics_queue().unwrap();

        let cmd = self.render_cmd[self.frame_index % 2];
        unsafe {
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
            device.reset_fences(&[self.render_fence])?;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }

        // Single "forward" pass: gbuffer.frag now writes the final lit
        // color directly into a single color attachment. The former
        // deferred-light fullscreen pass has been removed — lighting is
        // inlined in the fragment shader (see gbuffer.frag). Also removed:
        // the dead shadow pass (nothing sampled it) and a redundant
        // record_transition_for_sampling call (the render pass's
        // final_layout + subpass dependency handle the transition).
        self.gbuffer.record_batch(device, cmd, &self.gbuffer_pipeline, &batch_objects);
        let _ = eye; // unused now that lighting is in-shader
        let _ = camera;

        unsafe {
            device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let signals = [self.render_done_semaphore];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&cmds)
                .signal_semaphores(&signals);
            device.queue_submit(gq.queue, &[submit], self.render_fence)?;
            // No end wait_for_fences: the CPU returns immediately after
            // queuing the render, letting the next frame's CPU work
            // (drain/gen/sim) overlap with GPU render. The start-of-frame
            // wait on render_fence throttles us to one render-in-flight,
            // and present_blit waits on render_done_semaphore before its
            // blit reads the light target.
        }
        self.frame_index += 1;

        Ok(())
    }

    /// Semaphore signaled by each `render_frame_pool` submit. Callers pass
    /// this to the swapchain's present/blit step so the blit GPU-side waits
    /// for render to finish before reading the light target.
    pub fn render_done_semaphore(&self) -> vk::Semaphore {
        self.render_done_semaphore
    }

    /// Block the CPU until the in-flight render submit (if any) has finished.
    /// Normally this wait sits at the top of `render_frame_pool` where its cost
    /// is lumped into the raycast bucket; callers that want to measure the
    /// wait separately can invoke it manually, then `render_frame_pool` will
    /// skip the internal wait (the fence is already signaled).
    pub fn wait_for_previous_frame(&self, ctx: &VulkanContext) -> Result<()> {
        let device = ctx.device();
        unsafe {
            device.wait_for_fences(&[self.render_fence], true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            // Wait for any in-flight work before destroying
            let _ = device.wait_for_fences(&[self.render_fence], true, u64::MAX);
            device.destroy_fence(self.render_fence, None);
            device.destroy_semaphore(self.render_done_semaphore, None);
            device.free_command_buffers(self.gbuffer.command_pool(), &self.render_cmd);
            if let Some(pool) = self.obj_desc_pool {
                device.destroy_descriptor_pool(pool, None);
            }
            device.destroy_descriptor_pool(self.light_desc_pool, None);
            device.destroy_descriptor_pool(self.tonemap_desc_pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.voxel_sampler, None);
        }
        self.gbuffer_pipeline.destroy(ctx);
        self.shadow_pipeline.destroy(ctx);
        self.light_pipeline.destroy(ctx);
        self.tonemap_pipeline.destroy(ctx);
        self.gbuffer.destroy(ctx);
        self.shadow_map.destroy(ctx);
        self.light_target.destroy(ctx);
        self.tonemap_target.destroy(ctx);
    }
}

/// Build the view matrix for sun shadow mapping.
fn build_sun_view(sun_dir: &[f32; 3], center: &[f32; 3], dim: f32) -> [f32; 16] {
    let dist = dim * 3.0;
    let eye = [
        center[0] + sun_dir[0] * dist,
        center[1] + sun_dir[1] * dist,
        center[2] + sun_dir[2] * dist,
    ];
    look_at_simple(eye, *center, [0.0, 1.0, 0.0])
}

/// Simple look-at producing a column-major view matrix.
fn look_at_simple(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize_v([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
    let s = normalize_v(cross_v(f, up));
    let u = cross_v(s, f);
    [
        s[0],  u[0],  -f[0], 0.0,
        s[1],  u[1],  -f[1], 0.0,
        s[2],  u[2],  -f[2], 0.0,
        -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]),
        -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]),
        (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]),
        1.0,
    ]
}

/// Orthographic projection (column-major, Vulkan depth [0,1]).
fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    let rl = right - left;
    let tb = top - bottom;
    let fn_ = far - near;
    [
        2.0 / rl, 0.0,      0.0,         0.0,
        0.0,      -2.0 / tb, 0.0,         0.0,
        0.0,      0.0,      -1.0 / fn_,   0.0,
        -(right + left) / rl, -(top + bottom) / tb, -near / fn_, 1.0,
    ]
}

fn translation_matrix(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        x,   y,   z,   1.0,
    ]
}

/// Scale by `s` then translate — maps unit cube [0,1]^3 to [x, x+s] × [y, y+s] × [z, z+s].
fn scale_translate_matrix(s: f32, x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        s,   0.0, 0.0, 0.0,
        0.0, s,   0.0, 0.0,
        0.0, 0.0, s,   0.0,
        x,   y,   z,   1.0,
    ]
}

/// Non-uniform scale then translate — maps unit cube [0,1]^3 to [x, x+sx] × [y, y+sy] × [z, z+sz].
fn scale3_translate_matrix(sx: f32, sy: f32, sz: f32, x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        sx,  0.0, 0.0, 0.0,
        0.0, sy,  0.0, 0.0,
        0.0, 0.0, sz,  0.0,
        x,   y,   z,   1.0,
    ]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn normalize_v(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross_v(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
