use glam::{IVec3, Vec3};
use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

fn test_scene() -> Scene {
    Scene::new_headless(SceneConfig::default())
}

fn filled_grid() -> VoxelGrid {
    let mut g = VoxelGrid::new(16, 16, 16);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                g.set(x, y, z, 1);
            }
        }
    }
    g
}

#[test]
fn damage_sphere_removes_voxels() {
    let mut scene = test_scene();
    let handle = scene.spawn(filled_grid(), Transform::default(), MaterialPalette::new());
    let before = scene.voxel_count(handle).unwrap();
    scene.damage_sphere(handle, Vec3::new(8.0, 8.0, 8.0), 3.0);
    scene.tick_sim();
    let after = scene.voxel_count(handle).unwrap();
    assert!(after < before, "before={before}, after={after}");
}

#[test]
fn damage_box_removes_voxels() {
    let mut scene = test_scene();
    let handle = scene.spawn(filled_grid(), Transform::default(), MaterialPalette::new());
    let before = scene.voxel_count(handle).unwrap();
    scene.damage_box(handle, Vec3::new(4.0, 4.0, 4.0), Vec3::new(8.0, 8.0, 8.0));
    scene.tick_sim();
    let after = scene.voxel_count(handle).unwrap();
    assert!(after < before);
}

#[test]
fn set_voxel_modifies_grid() {
    let mut scene = test_scene();
    let handle = scene.spawn(VoxelGrid::new(8, 8, 8), Transform::default(), MaterialPalette::new());
    assert_eq!(scene.voxel_count(handle), Some(0));
    scene.set_voxel(handle, IVec3::new(2, 2, 2), 5);
    scene.tick_sim();
    assert_eq!(scene.voxel_count(handle), Some(1));
}

#[test]
fn mutations_on_dead_handle_are_noop() {
    let mut scene = test_scene();
    let handle = scene.spawn(filled_grid(), Transform::default(), MaterialPalette::new());
    scene.despawn(handle);
    scene.damage_sphere(handle, Vec3::new(8.0, 8.0, 8.0), 3.0);
    scene.set_voxel(handle, IVec3::new(0, 0, 0), 1);
    scene.tick_sim(); // should not panic
}

#[test]
fn mutations_are_batched_and_applied_on_tick() {
    let mut scene = test_scene();
    let handle = scene.spawn(filled_grid(), Transform::default(), MaterialPalette::new());
    let initial = scene.voxel_count(handle).unwrap();
    scene.damage_sphere(handle, Vec3::new(4.0, 4.0, 4.0), 2.0);
    scene.damage_sphere(handle, Vec3::new(12.0, 12.0, 12.0), 2.0);
    assert_eq!(scene.voxel_count(handle), Some(initial)); // not applied yet
    scene.tick_sim();
    let after = scene.voxel_count(handle).unwrap();
    assert!(after < initial);
}
