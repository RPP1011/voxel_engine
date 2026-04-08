use glam::Vec3;
use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::physics::PhysicsBody;
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

fn test_scene() -> Scene { Scene::new_headless(SceneConfig::default()) }
fn small_grid() -> VoxelGrid { let mut g = VoxelGrid::new(4,4,4); g.set(1,1,1,1); g }

#[test]
fn attach_physics_to_entity() {
    let mut scene = test_scene();
    let handle = scene.spawn(&small_grid(), Transform::default(), &MaterialPalette::new());
    assert!(scene.get_physics(handle).is_none());
    scene.set_physics(handle, PhysicsBody::dynamic(5.0));
    assert!(scene.get_physics(handle).is_some());
    assert_eq!(scene.get_physics(handle).unwrap().mass, 5.0);
}

#[test]
fn remove_physics_from_entity() {
    let mut scene = test_scene();
    let handle = scene.spawn(&small_grid(), Transform::default(), &MaterialPalette::new());
    scene.set_physics(handle, PhysicsBody::dynamic(5.0));
    scene.remove_physics(handle);
    assert!(scene.get_physics(handle).is_none());
}

#[test]
fn apply_impulse_changes_velocity() {
    let mut scene = test_scene();
    let handle = scene.spawn(&small_grid(), Transform::default(), &MaterialPalette::new());
    scene.set_physics(handle, PhysicsBody::dynamic(1.0));
    scene.apply_impulse(handle, Vec3::new(10.0, 0.0, 0.0));
    let body = scene.get_physics(handle).unwrap();
    assert!((body.linear_velocity.x - 10.0).abs() < 1e-4);
}

#[test]
fn physics_on_dead_handle_is_noop() {
    let mut scene = test_scene();
    let handle = scene.spawn(&small_grid(), Transform::default(), &MaterialPalette::new());
    scene.despawn(handle);
    scene.set_physics(handle, PhysicsBody::dynamic(1.0));
    assert!(scene.get_physics(handle).is_none());
}
