use anyhow::{Context, Result};
use ash::vk;

use super::instance::{QueueHandle, VulkanContext};

pub struct TimelineSemaphore {
    semaphore: vk::Semaphore,
}

impl TimelineSemaphore {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let mut timeline_ci =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_ci);
        let semaphore = unsafe { ctx.device().create_semaphore(&sem_ci, None) }
            .context("Failed to create timeline semaphore")?;
        Ok(Self { semaphore })
    }

    /// Submit a command that writes `value` (u32) to `buffer` at byte `offset`,
    /// then signals this timeline semaphore to `signal_value`.
    pub fn submit_buffer_write(
        &self,
        ctx: &VulkanContext,
        queue: QueueHandle,
        buffer: vk::Buffer,
        value: u32,
        offset: usize,
        signal_value: u64,
    ) -> Result<()> {
        self.submit_buffer_write_inner(ctx, queue, buffer, value, offset, None, signal_value)
    }

    /// Submit a command that waits on `wait_value`, writes `value` to `buffer` at byte
    /// offset `slot * 4`, then signals `signal_value`.
    pub fn submit_buffer_write_after(
        &self,
        ctx: &VulkanContext,
        queue: QueueHandle,
        buffer: vk::Buffer,
        value: u32,
        slot: usize,
        wait_value: u64,
        signal_value: u64,
    ) -> Result<()> {
        self.submit_buffer_write_inner(
            ctx,
            queue,
            buffer,
            value,
            slot * 4,
            Some(wait_value),
            signal_value,
        )
    }

    fn submit_buffer_write_inner(
        &self,
        ctx: &VulkanContext,
        queue: QueueHandle,
        buffer: vk::Buffer,
        value: u32,
        byte_offset: usize,
        wait_value: Option<u64>,
        signal_value: u64,
    ) -> Result<()> {
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
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_fill_buffer(
                cmd,
                buffer,
                byte_offset as u64,
                4,
                value,
            );

            // Barrier so host can read
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            device.end_command_buffer(cmd)?;
        }

        let cmd_bufs = [cmd];
        let signal_sems = [self.semaphore];
        let signal_values = [signal_value];

        let wait_sems;
        let wait_values;
        let wait_stages;

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_values);

        let mut submit = vk::SubmitInfo::default()
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_sems);

        if let Some(wv) = wait_value {
            wait_sems = [self.semaphore];
            wait_values = [wv];
            wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            timeline_info = timeline_info.wait_semaphore_values(&wait_values);
            submit = submit
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages);
        }

        submit = submit.push_next(&mut timeline_info);

        unsafe {
            device.queue_submit(queue.queue, &[submit], vk::Fence::null())?;
        }

        // Cleanup pool after GPU is done (we'll wait externally)
        // For correctness, we need to defer cleanup — store for later or wait here
        // For tests, we wait on host after all submits, so just leak the pool for now
        // and clean it up at the end. Actually let's use a fence for cleanup.
        // Simpler: just track and clean up. For test simplicity, wait idle.
        unsafe {
            device.queue_wait_idle(queue.queue)?;
            device.destroy_command_pool(cmd_pool, None);
        }

        Ok(())
    }

    /// Wait on the host for the timeline semaphore to reach `value`.
    pub fn wait_on_host(&self, ctx: &VulkanContext, value: u64, timeout_ns: u64) -> Result<()> {
        let sems = [self.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);
        unsafe { ctx.device().wait_semaphores(&wait_info, timeout_ns) }
            .context("Timeline semaphore wait timed out")?;
        Ok(())
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe { ctx.device().destroy_semaphore(self.semaphore, None) };
    }
}
