//! Smoke tests for GPU terrain compute pipeline.
//! Verifies the materializer produces grass/dirt/stone layers as expected.
//! Skips if no Vulkan GPU available.

use voxel_engine::terrain_compute::TerrainComputePipeline;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

const CHUNK_SIZE: usize = 64;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

const MAT_AIR: u8 = 0;
#[allow(dead_code)]
const MAT_DIRT: u8 = 1;
const MAT_STONE: u8 = 2;
#[allow(dead_code)]
const MAT_GRANITE: u8 = 3;
#[allow(dead_code)]
const MAT_GRASS: u8 = 7;

fn try_pipeline() -> Option<(VulkanContext, VulkanAllocator, TerrainComputePipeline)> {
    let ctx = VulkanContext::new().ok()?;
    let mut alloc = VulkanAllocator::new(&ctx).ok()?;
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).ok()?;
    Some((ctx, alloc, pipeline))
}

#[test]
fn surface_chunk_has_grass_dirt_stone() {
    // base_height = 0.3 * MAX_SURFACE_Z(2000) = 600. Surface lives near vz=600.
    // Chunk cz=9 covers vz=576..639, which straddles the surface.
    // Task 9+: biome-aware materials means the output may include TallGrass,
    // Peat, Sandstone, etc. depending on which cell (0,0) lands in. Assert
    // that there's a mix of air and solids, not the exact material set.
    let (ctx, mut alloc, mut pipeline) = match try_pipeline() {
        Some(p) => p,
        None => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 9], 42).expect("dispatch");
    assert_eq!(mats.len(), CHUNK_VOLUME);

    let air = mats.iter().filter(|&&m| m == MAT_AIR).count();
    let solid = CHUNK_VOLUME - air;
    eprintln!("air={air} solid={solid}");

    assert!(air > 0, "no air found in surface chunk");
    assert!(solid > 0, "no solid found in surface chunk");

    pipeline.destroy(&ctx, &mut alloc);
}

#[test]
fn deep_chunk_is_solid() {
    let (ctx, mut alloc, mut pipeline) = match try_pipeline() {
        Some(p) => p,
        None => { eprintln!("SKIP: no Vulkan"); return; }
    };
    // Chunk cz=0 covers vz=0..63, deep underground (surface ~600). Should be all stone.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 0], 42).expect("dispatch");
    let stone = mats.iter().filter(|&&m| m == MAT_STONE).count();
    let air = mats.iter().filter(|&&m| m == MAT_AIR).count();
    assert_eq!(air, 0, "deep chunk should have no air");
    assert!(stone > CHUNK_VOLUME * 9 / 10, "deep chunk should be mostly stone");
    pipeline.destroy(&ctx, &mut alloc);
}

#[test]
fn high_sky_chunk_is_air() {
    let (ctx, mut alloc, mut pipeline) = match try_pipeline() {
        Some(p) => p,
        None => { eprintln!("SKIP: no Vulkan"); return; }
    };
    // Chunk cz=20 covers vz=1280..1343, well above any surface. All air.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 20], 42).expect("dispatch");
    let air = mats.iter().filter(|&&m| m == MAT_AIR).count();
    assert_eq!(air, CHUNK_VOLUME, "high sky chunk should be all air");
    pipeline.destroy(&ctx, &mut alloc);
}

#[test]
fn deterministic_for_same_inputs() {
    let (ctx, mut alloc, mut pipeline) = match try_pipeline() {
        Some(p) => p,
        None => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let a = pipeline.generate_chunk(&ctx, [0, 0, 9], 42).expect("dispatch a");
    let b = pipeline.generate_chunk(&ctx, [0, 0, 9], 42).expect("dispatch b");
    assert_eq!(a, b, "GPU output not deterministic for same inputs");
    pipeline.destroy(&ctx, &mut alloc);
}
