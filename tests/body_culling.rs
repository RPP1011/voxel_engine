use voxel_engine::physics::culling::{BodyCuller, CullableBody};

#[test]
fn body_below_world_bounds_removed() {
    let mut culler = BodyCuller::new(-100.0, 1, 100_000);
    let bodies = vec![
        CullableBody { id: 0, y: 5.0, voxel_count: 100 },
        CullableBody { id: 1, y: -200.0, voxel_count: 100 },
        CullableBody { id: 2, y: 0.0, voxel_count: 100 },
    ];

    let to_remove = culler.find_bodies_to_cull(&bodies);
    assert!(to_remove.contains(&1), "Body below y=-100 should be culled");
    assert!(!to_remove.contains(&0));
    assert!(!to_remove.contains(&2));
}

#[test]
fn tiny_debris_removed() {
    let mut culler = BodyCuller::new(-100.0, 25, 100_000);
    let bodies = vec![
        CullableBody { id: 0, y: 5.0, voxel_count: 100 },
        CullableBody { id: 1, y: 5.0, voxel_count: 10 },  // < 25 threshold
        CullableBody { id: 2, y: 5.0, voxel_count: 3 },   // < 25 threshold
    ];

    let to_remove = culler.find_bodies_to_cull(&bodies);
    assert!(to_remove.contains(&1));
    assert!(to_remove.contains(&2));
    assert!(!to_remove.contains(&0));
}

#[test]
fn cap_exceeded_smallest_culled_first() {
    let mut culler = BodyCuller::new(-100.0, 1, 3); // cap of 3 bodies
    let bodies = vec![
        CullableBody { id: 0, y: 5.0, voxel_count: 500 },
        CullableBody { id: 1, y: 5.0, voxel_count: 100 },
        CullableBody { id: 2, y: 5.0, voxel_count: 200 },
        CullableBody { id: 3, y: 5.0, voxel_count: 50 },  // smallest
        CullableBody { id: 4, y: 5.0, voxel_count: 300 },
    ];

    let to_remove = culler.find_bodies_to_cull(&bodies);

    // Need to remove 2 bodies (5 - cap of 3). Smallest first: id=3 (50), id=1 (100)
    assert_eq!(to_remove.len(), 2, "Should cull 2 to reach cap of 3");
    assert!(to_remove.contains(&3), "Smallest body should be culled first");
    assert!(to_remove.contains(&1), "Second smallest should be culled");
}
