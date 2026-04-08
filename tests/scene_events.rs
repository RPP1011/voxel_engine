use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::scene::events::SceneEvent;
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
fn no_events_initially() {
    let mut scene = test_scene();
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert!(events.is_empty());
}

#[test]
fn drain_events_clears_queue() {
    let mut scene = test_scene();
    let handle = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.push_event(SceneEvent::EntityAtRest { handle });
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert_eq!(events.len(), 1);
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert!(events.is_empty());
}

#[test]
fn multiple_events_preserved_in_order() {
    let mut scene = test_scene();
    let h1 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    let h2 = scene.spawn(small_grid(), Transform::default(), MaterialPalette::new());
    scene.push_event(SceneEvent::EntityAtRest { handle: h1 });
    scene.push_event(SceneEvent::EntityAtRest { handle: h2 });
    let events: Vec<SceneEvent> = scene.drain_events().collect();
    assert_eq!(events.len(), 2);
    match &events[0] {
        SceneEvent::EntityAtRest { handle } => assert_eq!(*handle, h1),
        _ => panic!(),
    }
    match &events[1] {
        SceneEvent::EntityAtRest { handle } => assert_eq!(*handle, h2),
        _ => panic!(),
    }
}
