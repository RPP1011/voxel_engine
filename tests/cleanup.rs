use glam::Vec3;
use voxel_engine::scene::{Scene, SceneConfig, Transform, CleanupPolicy};
use voxel_engine::scene::events::{SceneEvent, CleanupReason};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

fn tiny_grid() -> VoxelGrid { let mut g = VoxelGrid::new(2,2,2); g.set(0,0,0,1); g }
fn scene_with_policy(policy: CleanupPolicy) -> Scene {
    Scene::new_headless(SceneConfig { cleanup: policy, ..Default::default() })
}

#[test]
fn too_small_fragments_cleaned_on_tick() {
    let mut scene = scene_with_policy(CleanupPolicy { min_fragment_voxels: 4, ..Default::default() });
    let handle = scene.spawn_fragment(&tiny_grid(), Transform::default(), &MaterialPalette::new());
    scene.tick_sim();
    assert!(!scene.is_alive(handle));
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert!(events.iter().any(|e| matches!(e, SceneEvent::FragmentCleaned { reason: CleanupReason::TooSmall, .. })));
}

#[test]
fn kill_plane_removes_entities_below() {
    let mut scene = scene_with_policy(CleanupPolicy { kill_plane_y: -50.0, min_fragment_voxels: 0, ..Default::default() });
    let handle = scene.spawn_fragment(&tiny_grid(), Transform { position: Vec3::new(0.0, -100.0, 0.0), ..Default::default() }, &MaterialPalette::new());
    scene.tick_sim();
    assert!(!scene.is_alive(handle));
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert!(events.iter().any(|e| matches!(e, SceneEvent::FragmentCleaned { reason: CleanupReason::BelowKillPlane, .. })));
}

#[test]
fn fragment_limit_evicts_oldest_at_rest() {
    let mut scene = scene_with_policy(CleanupPolicy {
        max_fragments: 2, min_fragment_voxels: 0, rest_timeout_secs: f32::MAX,
        kill_plane_y: f32::NEG_INFINITY, ..Default::default()
    });
    let palette = MaterialPalette::new();
    let h1 = scene.spawn_fragment(&tiny_grid(), Transform::default(), &palette);
    let h2 = scene.spawn_fragment(&tiny_grid(), Transform::default(), &palette);
    let h3 = scene.spawn_fragment(&tiny_grid(), Transform::default(), &palette);
    scene.mark_at_rest(h1); scene.mark_at_rest(h2); scene.mark_at_rest(h3);
    scene.tick_sim();
    let alive = [h1,h2,h3].iter().filter(|h| scene.is_alive(**h)).count();
    assert_eq!(alive, 2);
}

#[test]
fn non_fragment_entities_are_not_cleaned() {
    let mut scene = scene_with_policy(CleanupPolicy { min_fragment_voxels: 100, ..Default::default() });
    let handle = scene.spawn(&tiny_grid(), Transform::default(), &MaterialPalette::new());
    scene.tick_sim();
    assert!(scene.is_alive(handle));
}
