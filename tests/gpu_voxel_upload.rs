use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{GpuVoxelTexture, upload_grid_to_gpu, readback_grid_from_gpu};
use voxel_engine::voxel::material::default_palette;

#[test]
fn upload_and_readback_exact_match() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let mut alloc = VulkanAllocator::new(&ctx).expect("Failed to create allocator");

    let mut grid = VoxelGrid::new(8, 8, 8);
    // Write a known pattern
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, ((x + y * 8 + z * 64) % 255 + 1) as u8);
            }
        }
    }

    let gpu_tex = upload_grid_to_gpu(&ctx, &mut alloc, &grid, &default_palette())
        .expect("Failed to upload grid to GPU");

    let readback = readback_grid_from_gpu(&ctx, &mut alloc, &gpu_tex)
        .expect("Failed to readback grid");

    // Byte-exact match
    assert_eq!(
        grid.data(),
        readback.data(),
        "Readback should match uploaded data exactly"
    );

    assert_eq!(ctx.validation_error_count(), 0);
    gpu_tex.destroy(&ctx, &mut alloc);
}

#[test]
fn partial_subregion_update() {
    let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
    let mut alloc = VulkanAllocator::new(&ctx).expect("Failed to create allocator");

    // Start with a fully solid grid
    let mut grid = VoxelGrid::new(8, 8, 8);
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, 1);
            }
        }
    }

    let mut gpu_tex = upload_grid_to_gpu(&ctx, &mut alloc, &grid, &default_palette())
        .expect("Failed to upload grid");

    // Modify a 4x4x4 subregion on CPU
    for x in 2..6 {
        for y in 2..6 {
            for z in 2..6 {
                grid.set(x, y, z, 42);
            }
        }
    }

    // Upload only the modified subregion
    gpu_tex
        .update_subregion(&ctx, &mut alloc, &grid, (2, 2, 2), (4, 4, 4))
        .expect("Failed to update subregion");

    // Readback full grid
    let readback = readback_grid_from_gpu(&ctx, &mut alloc, &gpu_tex)
        .expect("Failed to readback");

    // Verify only the modified region changed
    for x in 0..8u32 {
        for y in 0..8u32 {
            for z in 0..8u32 {
                let expected = grid.get(x, y, z).unwrap();
                let actual = readback.get(x, y, z).unwrap();
                assert_eq!(
                    actual, expected,
                    "Mismatch at ({x},{y},{z}): expected {expected}, got {actual}"
                );
            }
        }
    }

    assert_eq!(ctx.validation_error_count(), 0);
    gpu_tex.destroy(&ctx, &mut alloc);
}
