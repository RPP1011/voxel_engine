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
    g
}

#[test]
fn tick_snapshots_transforms_for_interpolation() {
    let mut scene = test_scene();
    let grid = small_grid();
    let palette = MaterialPalette::new();
    let handle = scene.spawn(grid, Transform::default(), palette);
    scene.tick_sim();
    scene.set_transform(
        handle,
        Transform {
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        },
    );
    scene.tick_sim();
    let interp = scene.interpolated_transform(handle, 0.5).unwrap();
    assert!((interp.position.x - 5.0).abs() < 1e-4);
}

#[test]
fn interpolate_at_zero_returns_previous() {
    let mut scene = test_scene();
    let handle = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.tick_sim();
    scene.set_transform(
        handle,
        Transform {
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        },
    );
    scene.tick_sim();
    let interp = scene.interpolated_transform(handle, 0.0).unwrap();
    assert!((interp.position.x).abs() < 1e-4);
}

#[test]
fn interpolate_at_one_returns_current() {
    let mut scene = test_scene();
    let handle = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.tick_sim();
    scene.set_transform(
        handle,
        Transform {
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        },
    );
    scene.tick_sim();
    let interp = scene.interpolated_transform(handle, 1.0).unwrap();
    assert!((interp.position.x - 10.0).abs() < 1e-4);
}

#[test]
fn interpolate_dead_entity_returns_none() {
    let mut scene = test_scene();
    let handle = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.tick_sim();
    scene.despawn(handle);
    assert!(scene.interpolated_transform(handle, 0.5).is_none());
}
