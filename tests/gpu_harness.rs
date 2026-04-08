use voxel_engine::compute::GpuHarness;
use voxel_engine::vulkan::instance::VulkanContext;

#[test]
fn harness_upload_dispatch_download() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let mut harness = GpuHarness::new(&ctx).expect("Failed to create harness");

    // Create input and output fields (256 f32s each)
    let input = harness.create_field::<f32>(&ctx, 256).expect("create input");
    let output = harness.create_field::<f32>(&ctx, 256).expect("create output");

    // Upload: [1.0, 2.0, 3.0, ..., 256.0]
    let input_data: Vec<f32> = (1..=256).map(|i| i as f32).collect();
    harness.upload(&input, &input_data).expect("upload");

    // Load kernel: doubles each element
    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/test_double.comp.spv"));
    harness
        .load_kernel(&ctx, "double", spirv, 2)
        .expect("load kernel");

    // Dispatch: 1 workgroup of 256 threads
    harness
        .dispatch(&ctx, "double", &[&input, &output], [1, 1, 1])
        .expect("dispatch");

    // Download and verify
    let result: Vec<f32> = harness.download(&output, 256).expect("download");
    for (i, &val) in result.iter().enumerate() {
        let expected = (i + 1) as f32 * 2.0;
        assert!(
            (val - expected).abs() < 0.001,
            "Element {i}: expected {expected}, got {val}"
        );
    }

    harness.destroy(&ctx);
}

#[test]
fn harness_multiple_dispatches() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let mut harness = GpuHarness::new(&ctx).expect("Failed to create harness");

    let buf_a = harness.create_field::<f32>(&ctx, 256).expect("create a");
    let buf_b = harness.create_field::<f32>(&ctx, 256).expect("create b");

    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/test_double.comp.spv"));
    harness.load_kernel(&ctx, "double", spirv, 2).expect("load");

    // Upload initial values
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    harness.upload(&buf_a, &data).expect("upload");

    // Double: a → b
    harness.dispatch(&ctx, "double", &[&buf_a, &buf_b], [1, 1, 1]).expect("dispatch 1");
    // Double again: b → a
    harness.dispatch(&ctx, "double", &[&buf_b, &buf_a], [1, 1, 1]).expect("dispatch 2");

    // a should now be original * 4
    let result: Vec<f32> = harness.download(&buf_a, 256).expect("download");
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32 * 4.0;
        assert!(
            (val - expected).abs() < 0.01,
            "Element {i}: expected {expected}, got {val}"
        );
    }

    harness.destroy(&ctx);
}
