use glam::Vec3;
use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

fn test_scene() -> Scene { Scene::new_headless(SceneConfig::default()) }
fn solid_grid() -> VoxelGrid {
    let mut g = VoxelGrid::new(8,8,8);
    for z in 0..8 { for y in 0..8 { for x in 0..8 { g.set(x,y,z,1); } } }
    g
}

#[test]
fn raycast_hits_entity() {
    let mut scene = test_scene();
    let _handle = scene.spawn(&solid_grid(), Transform {
        position: Vec3::new(0.0, 0.0, -20.0), ..Default::default()
    }, &MaterialPalette::new());
    let hit = scene.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0);
    assert!(hit.is_some());
    let hit = hit.unwrap();
    assert!(hit.distance > 0.0);
    assert!(hit.distance < 30.0);
}

#[test]
fn raycast_misses_empty_space() {
    let mut scene = test_scene();
    let _handle = scene.spawn(&solid_grid(), Transform {
        position: Vec3::new(100.0, 0.0, 0.0), ..Default::default()
    }, &MaterialPalette::new());
    let hit = scene.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0);
    assert!(hit.is_none());
}

#[test]
fn query_sphere_finds_nearby_entities() {
    let mut scene = test_scene();
    let h1 = scene.spawn(&solid_grid(), Transform {
        position: Vec3::ZERO, ..Default::default()
    }, &MaterialPalette::new());
    let _h2 = scene.spawn(&solid_grid(), Transform {
        position: Vec3::new(1000.0, 0.0, 0.0), ..Default::default()
    }, &MaterialPalette::new());
    let results = scene.query_sphere(Vec3::ZERO, 50.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], h1);
}

#[test]
fn query_aabb_finds_overlapping_entities() {
    let mut scene = test_scene();
    let h1 = scene.spawn(&solid_grid(), Transform {
        position: Vec3::new(5.0, 5.0, 5.0), ..Default::default()
    }, &MaterialPalette::new());
    let results = scene.query_aabb(Vec3::ZERO, Vec3::splat(20.0));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], h1);
}
