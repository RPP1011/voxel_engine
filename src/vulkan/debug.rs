use anyhow::{Context, Result};
use ash::vk;

use super::instance::{QueueHandle, VulkanContext};

pub struct GpuProfiler {
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    query_count: u32,
}

impl GpuProfiler {
    pub fn new(ctx: &VulkanContext, query_count: u32) -> Result<Self> {
        let props = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
        };
        let timestamp_period = props.limits.timestamp_period;

        let pool_ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count);
        let query_pool = unsafe { ctx.device().create_query_pool(&pool_ci, None) }
            .context("Failed to create timestamp query pool")?;

        Ok(Self {
            query_pool,
            timestamp_period,
            query_count,
        })
    }

    /// Submit an empty command buffer bracketed by timestamp queries.
    /// Returns the elapsed time in nanoseconds.
    pub fn time_empty_submit(&mut self, ctx: &VulkanContext, queue: QueueHandle) -> Result<f64> {
        let device = ctx.device();

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None) }?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_ci) }?[0];

        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);

            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );

            // Debug label (RenderDoc compatible)
            // Only insert if debug_utils is available via the context
            // For now, just the timestamps are the core functionality

            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );

            device.end_command_buffer(cmd)?;
        }

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence = unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(queue.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            fence
        };

        // Read timestamps
        let mut timestamps = [0u64; 2];
        unsafe {
            device.get_query_pool_results(
                self.query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }

        let delta_ticks = timestamps[1].wrapping_sub(timestamps[0]);
        let delta_ns = delta_ticks as f64 * self.timestamp_period as f64;

        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }

        Ok(delta_ns)
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device().destroy_query_pool(self.query_pool, None);
        }
    }
}
