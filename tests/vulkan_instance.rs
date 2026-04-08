use voxel_engine::vulkan::instance::VulkanContext;

#[test]
fn create_instance_no_validation_errors() {
    // Create Vulkan instance with validation layers and verify clean creation/destruction
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    assert!(
        ctx.validation_error_count() == 0,
        "Validation errors during instance creation: {}",
        ctx.validation_error_count()
    );
    drop(ctx);
    // If we get here without panic, creation and destruction succeeded cleanly
}

#[test]
fn selects_discrete_gpu_with_required_extensions() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

    // Must be a discrete GPU
    assert!(
        ctx.is_discrete_gpu(),
        "Selected device is not a discrete GPU"
    );

    // Must support required extensions
    let required_extensions = [
        "VK_KHR_buffer_device_address",
        "VK_KHR_synchronization2",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_acceleration_structure",
    ];
    for ext in &required_extensions {
        assert!(
            ctx.has_device_extension(ext),
            "Missing required extension: {ext}"
        );
    }
}

#[test]
fn logical_device_has_required_queue_families() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");

    // Must have graphics, compute, and transfer queue families
    assert!(
        ctx.graphics_queue().is_some(),
        "No graphics queue available"
    );
    assert!(
        ctx.compute_queue().is_some(),
        "No compute queue available"
    );
    assert!(
        ctx.transfer_queue().is_some(),
        "No transfer queue available"
    );
}
