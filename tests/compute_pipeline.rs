use voxel_engine::vulkan::compute::ComputePipeline;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::allocator::VulkanAllocator;

#[test]
fn compute_shader_writes_thread_id_times_two() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let mut alloc = VulkanAllocator::new(&ctx).expect("Failed to create allocator");

    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/test_multiply.comp.spv"));
    let pipeline = ComputePipeline::new(&ctx, spirv).expect("Failed to create compute pipeline");

    // Allocate output buffer (256 u32s = 1024 bytes)
    let mut output = alloc
        .allocate_host_visible_buffer(256 * 4)
        .expect("Failed to allocate output buffer");

    // Dispatch 256 threads (1 workgroup of 256)
    pipeline
        .dispatch_with_buffer(&ctx, output.buffer(), 1, 1, 1)
        .expect("Failed to dispatch compute shader");

    // Read back results
    let data = output.read_data(0, 256 * 4).expect("Failed to read back");
    let results: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();

    for (i, &val) in results.iter().enumerate() {
        assert_eq!(
            val,
            (i as u32) * 2,
            "Element {i}: expected {}, got {val}",
            i * 2
        );
    }

    assert_eq!(ctx.validation_error_count(), 0);
    alloc.free_buffer(output);
}
