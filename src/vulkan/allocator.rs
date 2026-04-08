use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use super::instance::VulkanContext;

pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
    mapped_ptr: Option<*mut u8>,
    device: ash::Device,
}

impl AllocatedBuffer {
    pub fn is_valid(&self) -> bool {
        self.buffer != vk::Buffer::null()
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn write_data(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let alloc = self.allocation.as_ref().context("No allocation")?;
        let mapped = alloc.mapped_ptr().context("Buffer not host-mapped")?;
        let dst = mapped.as_ptr() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst.add(offset), data.len());
        }
        Ok(())
    }

    pub fn read_data(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let alloc = self.allocation.as_ref().context("No allocation")?;
        let mapped = alloc.mapped_ptr().context("Buffer not host-mapped")?;
        let src = mapped.as_ptr() as *const u8;
        let mut result = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(offset), result.as_mut_ptr(), len);
        }
        Ok(result)
    }
}

pub struct VulkanAllocator {
    allocator: Allocator,
    device: ash::Device,
}

impl VulkanAllocator {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance().clone(),
            device: ctx.device().clone(),
            physical_device: ctx.physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .context("Failed to create gpu-allocator")?;

        Ok(Self {
            allocator,
            device: ctx.device().clone(),
        })
    }

    pub fn allocate_device_local_buffer(&mut self, size: u64) -> Result<AllocatedBuffer> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_ci, None) }
            .context("Failed to create buffer")?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.allocate(&AllocationCreateDesc {
            name: "device_local_buffer",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).context("Failed to allocate device-local memory")?;

        unsafe {
            self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }.context("Failed to bind buffer memory")?;

        Ok(AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            mapped_ptr: None,
            device: self.device.clone(),
        })
    }

    pub fn allocate_host_visible_buffer(&mut self, size: u64) -> Result<AllocatedBuffer> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_ci, None) }
            .context("Failed to create buffer")?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.allocate(&AllocationCreateDesc {
            name: "host_visible_buffer",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).context("Failed to allocate host-visible memory")?;

        unsafe {
            self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }.context("Failed to bind buffer memory")?;

        Ok(AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            mapped_ptr: None,
            device: self.device.clone(),
        })
    }

    pub fn free_buffer(&mut self, mut buffer: AllocatedBuffer) {
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
        if let Some(allocation) = buffer.allocation.take() {
            self.allocator.free(allocation).ok();
        }
    }

    pub fn allocate_image_memory(
        &mut self,
        requirements: vk::MemoryRequirements,
    ) -> Result<Allocation> {
        self.allocator
            .allocate(&AllocationCreateDesc {
                name: "image_memory",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .context("Failed to allocate image memory")
    }

    pub fn free_allocation(&mut self, allocation: Allocation) {
        self.allocator.free(allocation).ok();
    }
}
