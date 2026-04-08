use glam::Vec3;
use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

fn test_scene() -> Scene {
    Scene::new_headless(SceneConfig::default())
}

fn small_grid() -> VoxelGrid {
    let mut g = VoxelGrid::new(4, 4, 4);
    g.set(1, 1, 1, 1);
    g.set(2, 2, 2, 1);
    g
}

#[test]
fn spawn_returns_unique_handles() {
    let mut scene = test_scene();
    let h1 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    let h2 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    assert_ne!(h1, h2);
}

#[test]
fn despawn_removes_entity() {
    let mut scene = test_scene();
    let h = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    assert!(scene.is_alive(h));
    scene.despawn(h);
    assert!(!scene.is_alive(h));
}

#[test]
fn despawn_invalid_handle_is_noop() {
    let mut scene = test_scene();
    let h = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.despawn(h);
    // Double despawn must not panic.
    scene.despawn(h);
    assert!(!scene.is_alive(h));
}

#[test]
fn set_transform_updates_entity() {
    let mut scene = test_scene();
    let h = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    let new_transform = Transform {
        position: Vec3::new(10.0, 20.0, 30.0),
        ..Transform::default()
    };
    scene.set_transform(h, new_transform);
    let t = scene.get_transform(h).expect("entity should be alive");
    assert_eq!(t.position, Vec3::new(10.0, 20.0, 30.0));
}

#[test]
fn set_transform_on_dead_handle_is_noop() {
    let mut scene = test_scene();
    let h = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.despawn(h);
    let new_transform = Transform {
        position: Vec3::new(1.0, 2.0, 3.0),
        ..Transform::default()
    };
    // Must not panic.
    scene.set_transform(h, new_transform);
    // Should return None — entity is dead.
    assert!(scene.get_transform(h).is_none());
}

#[test]
fn entity_count_tracks_spawns_and_despawns() {
    let mut scene = test_scene();
    assert_eq!(scene.entity_count(), 0);
    let h1 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    let _h2 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    assert_eq!(scene.entity_count(), 2);
    scene.despawn(h1);
    assert_eq!(scene.entity_count(), 1);
}

#[test]
fn handle_reuse_with_generation() {
    let mut scene = test_scene();
    let old = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.despawn(old);
    // Spawning again will reuse the freed slot but bump the generation.
    let new_h = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    assert!(!scene.is_alive(old), "old handle must be dead after slot reuse");
    assert!(scene.is_alive(new_h), "new handle must be alive");
    // They share the same slot index but differ in generation.
    assert_eq!(old.slot_index(), new_h.slot_index());
    assert_ne!(old.generation(), new_h.generation());
}
