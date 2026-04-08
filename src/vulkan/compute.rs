use anyhow::{Context, Result};
use ash::vk;

use super::instance::VulkanContext;

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
}

impl ComputePipeline {
    pub fn new(ctx: &VulkanContext, spirv: &[u8]) -> Result<Self> {
        let device = ctx.device();

        // Create shader module
        let spirv_words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe { device.create_shader_module(&shader_ci, None) }
            .context("Failed to create shader module")?;

        // Descriptor set layout: single storage buffer at binding 0
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let bindings = [binding];
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_ci, None) }
                .context("Failed to create descriptor set layout")?;

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None) }
            .context("Failed to create pipeline layout")?;

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
        .context("Failed to create compute pipeline")?[0];

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            shader_module,
        })
    }

    pub fn dispatch_with_buffer(
        &self,
        ctx: &VulkanContext,
        buffer: vk::Buffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<()> {
        let device = ctx.device();
        let cq = ctx.compute_queue().context("No compute queue")?;

        // Create descriptor pool and set
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .context("Failed to create descriptor pool")?;

        let set_layouts = [self.descriptor_set_layout];
        let alloc_ci = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_ci) }
            .context("Failed to allocate descriptor sets")?;

        // Update descriptor set
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let buffer_infos = [buffer_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        // Command buffer
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(cq.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .context("Failed to create command pool")?;

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }
            .context("Failed to allocate command buffer")?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &descriptor_sets,
                &[],
            );
            device.cmd_dispatch(cmd, group_count_x, group_count_y, group_count_z);

            // Memory barrier so host can read the result
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

        // Submit and wait
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence = unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(cq.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            fence
        };

        // Cleanup
        unsafe {
            device.destroy_command_pool(cmd_pool, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(())
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}
