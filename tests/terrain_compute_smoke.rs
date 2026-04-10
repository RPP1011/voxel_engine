//! Smoke test: GPU compute generates a chunk with the expected skeleton output
//! (stone below z=0, air above). Skips if no Vulkan GPU available.

use voxel_engine::terrain_compute::TerrainComputePipeline;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

const CHUNK_SIZE: usize = 64;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

const MAT_AIR: u8 = 0;
const MAT_STONE: u8 = 2;

#[test]
fn gpu_skeleton_chunk_below_zero_is_stone() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    // Chunk at cz=-1 covers vz = -64..-1, all below zero → all stone.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, -1], 42).expect("dispatch");
    assert_eq!(mats.len(), CHUNK_VOLUME, "wrong buffer size");

    let stone = mats.iter().filter(|&&m| m == MAT_STONE).count();
    let air = mats.iter().filter(|&&m| m == MAT_AIR).count();
    assert_eq!(stone, CHUNK_VOLUME, "below-zero chunk should be all stone, got {} stone / {} air", stone, air);

    pipeline.destroy(&ctx, &mut alloc);
}

#[test]
fn gpu_skeleton_chunk_above_zero_is_air() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    // Chunk at cz=1 covers vz = 64..127, all above zero → all air.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 1], 42).expect("dispatch");
    let air = mats.iter().filter(|&&m| m == MAT_AIR).count();
    assert_eq!(air, CHUNK_VOLUME, "above-zero chunk should be all air");

    pipeline.destroy(&ctx, &mut alloc);
}

#[test]
fn gpu_skeleton_chunk_straddling_zero_has_both() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    // Chunk at cz=-1 covers vz = -64..-1, all below zero → all stone.
    // Chunk at cz=0 covers vz = 0..63, all >= 0 → all air (skeleton uses vz < 0).
    // Verify boundary: top layer of cz=-1 (lz=63 → vz=-1) is stone,
    // and bottom layer of cz=0 (lz=0 → vz=0) is air.

    let mats_below = pipeline.generate_chunk(&ctx, [0, 0, -1], 42).expect("dispatch");
    // Top layer of below chunk: lz=63 → indices [63*64*64 .. 64*64*64)
    let top_layer = &mats_below[63 * CHUNK_SIZE * CHUNK_SIZE .. CHUNK_VOLUME];
    assert!(top_layer.iter().all(|&m| m == MAT_STONE), "top layer of cz=-1 should be stone (vz=-1)");

    let mats_above = pipeline.generate_chunk(&ctx, [0, 0, 0], 42).expect("dispatch");
    // Bottom layer of above chunk: lz=0
    let bottom_layer = &mats_above[0 .. CHUNK_SIZE * CHUNK_SIZE];
    assert!(bottom_layer.iter().all(|&m| m == MAT_AIR), "bottom layer of cz=0 should be air (vz=0)");

    pipeline.destroy(&ctx, &mut alloc);
}
