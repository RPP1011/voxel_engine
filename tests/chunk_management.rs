use voxel_engine::world::chunks::{ChunkManager, ChunkTier};

#[test]
fn camera_at_origin_nearby_chunks_active() {
    let mut mgr = ChunkManager::new(64.0, 100.0, 500.0);
    mgr.update_camera([32.0, 32.0, 32.0]); // center of chunk (0,0,0)

    let tier = mgr.chunk_tier(0, 0, 0);
    assert_eq!(tier, ChunkTier::Active, "Chunk containing camera should be active");
}

#[test]
fn distant_chunks_streaming() {
    let mut mgr = ChunkManager::new(64.0, 50.0, 500.0);
    mgr.update_camera([0.0, 0.0, 0.0]);

    // Chunk at ~200m (chunk coords 3,0,0 → center at 192m + 32m = 224m)
    let tier = mgr.chunk_tier(3, 0, 0);
    assert_eq!(tier, ChunkTier::Streaming, "Chunk at ~200m should be streaming");
}

#[test]
fn very_distant_chunks_unloaded() {
    let mut mgr = ChunkManager::new(64.0, 50.0, 500.0);
    mgr.update_camera([0.0, 0.0, 0.0]);

    // Chunk at ~600m
    let tier = mgr.chunk_tier(10, 0, 0);
    assert_eq!(tier, ChunkTier::Unloaded, "Chunk at ~600m should be unloaded");
}

#[test]
fn camera_move_promotes_demotes_chunks() {
    let mut mgr = ChunkManager::new(64.0, 100.0, 500.0);

    // Start at center of chunk (0,0,0)
    mgr.update_camera([32.0, 0.0, 32.0]);
    assert_eq!(mgr.chunk_tier(0, 0, 0), ChunkTier::Active);
    assert_eq!(mgr.chunk_tier(10, 0, 0), ChunkTier::Unloaded);

    // Move camera to chunk (10,0,0) center
    mgr.update_camera([672.0, 0.0, 32.0]);
    assert_eq!(mgr.chunk_tier(0, 0, 0), ChunkTier::Unloaded, "Origin should be unloaded now");
    assert_eq!(mgr.chunk_tier(10, 0, 0), ChunkTier::Active, "Chunk near new camera should be active");
}

#[test]
fn chunk_size_64m() {
    let mgr = ChunkManager::new(64.0, 50.0, 500.0);
    assert_eq!(mgr.chunk_size(), 64.0);
}
