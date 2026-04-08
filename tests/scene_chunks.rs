use glam::IVec3;
use voxel_engine::scene::{Scene, SceneConfig};
use voxel_engine::voxel::grid::VoxelGrid;

fn test_scene() -> Scene { Scene::new_headless(SceneConfig::default()) }
fn chunk_grid() -> VoxelGrid { let mut g = VoxelGrid::new(64,64,64); g.set(32,32,32,1); g }

#[test]
fn load_chunk_returns_handle() {
    let mut scene = test_scene();
    let handle = scene.load_chunk(IVec3::ZERO, &chunk_grid());
    assert!(scene.is_chunk_loaded(handle));
}

#[test]
fn unload_chunk_removes_it() {
    let mut scene = test_scene();
    let handle = scene.load_chunk(IVec3::ZERO, &chunk_grid());
    scene.unload_chunk(handle);
    assert!(!scene.is_chunk_loaded(handle));
}

#[test]
fn load_multiple_chunks_at_different_positions() {
    let mut scene = test_scene();
    let h1 = scene.load_chunk(IVec3::new(0,0,0), &chunk_grid());
    let h2 = scene.load_chunk(IVec3::new(1,0,0), &chunk_grid());
    let h3 = scene.load_chunk(IVec3::new(0,1,0), &chunk_grid());
    assert_ne!(h1, h2); assert_ne!(h2, h3);
    assert_eq!(scene.loaded_chunk_count(), 3);
}

#[test]
fn double_unload_is_noop() {
    let mut scene = test_scene();
    let handle = scene.load_chunk(IVec3::ZERO, &chunk_grid());
    scene.unload_chunk(handle);
    scene.unload_chunk(handle);
    assert_eq!(scene.loaded_chunk_count(), 0);
}
