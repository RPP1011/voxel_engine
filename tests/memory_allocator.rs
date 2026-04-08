use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

fn setup() -> (VulkanContext, VulkanAllocator) {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let alloc = VulkanAllocator::new(&ctx).expect("Failed to create allocator");
    (ctx, alloc)
}

#[test]
fn allocate_device_local_buffer() {
    let (_ctx, mut alloc) = setup();

    let buffer = alloc
        .allocate_device_local_buffer(1024 * 1024) // 1MB
        .expect("Failed to allocate device-local buffer");

    assert!(buffer.is_valid(), "Buffer handle should be valid");
    assert!(buffer.size() >= 1024 * 1024, "Buffer should be at least 1MB");
}

#[test]
fn allocate_host_visible_buffer_write_read() {
    let (_ctx, mut alloc) = setup();

    let mut buffer = alloc
        .allocate_host_visible_buffer(1024 * 1024) // 1MB
        .expect("Failed to allocate host-visible buffer");

    // Write a pattern
    let pattern: Vec<u8> = (0..256).map(|i| i as u8).collect();
    buffer.write_data(0, &pattern).expect("Failed to write data");

    // Read back and verify
    let readback = buffer.read_data(0, 256).expect("Failed to read data");
    assert_eq!(readback, pattern, "Read data should match written pattern");
}

#[test]
fn allocate_and_free_no_leaks() {
    let (ctx, mut alloc) = setup();

    let buffer = alloc
        .allocate_device_local_buffer(1024 * 1024)
        .expect("Failed to allocate buffer");

    alloc.free_buffer(buffer);

    // If we reach here without validation errors, no leaks
    assert_eq!(
        ctx.validation_error_count(),
        0,
        "Should have no validation errors after alloc+free"
    );
}
