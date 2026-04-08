use voxel_engine::voxel::grid::VoxelGrid;

#[test]
fn set_and_get_voxel() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    grid.set(3, 7, 11, 5);
    assert_eq!(grid.get(3, 7, 11), Some(5));
}

#[test]
fn out_of_bounds_returns_none() {
    let grid = VoxelGrid::new(16, 16, 16);
    assert_eq!(grid.get(16, 0, 0), None);
    assert_eq!(grid.get(0, 16, 0), None);
    assert_eq!(grid.get(0, 0, 16), None);
    assert_eq!(grid.get(100, 100, 100), None);
}

#[test]
fn count_nonempty() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    assert_eq!(grid.count_nonempty(), 0);

    grid.set(0, 0, 0, 1);
    grid.set(5, 5, 5, 2);
    grid.set(15, 15, 15, 3);
    assert_eq!(grid.count_nonempty(), 3);

    // Remove one
    grid.set(5, 5, 5, 0);
    assert_eq!(grid.count_nonempty(), 2);
}

#[test]
fn bounding_box_of_nonempty() {
    let mut grid = VoxelGrid::new(16, 16, 16);

    // Empty grid has no bounding box
    assert!(grid.bounding_box_of_nonempty().is_none());

    grid.set(3, 2, 5, 1);
    grid.set(10, 8, 12, 2);

    let (min, max) = grid.bounding_box_of_nonempty().unwrap();
    assert_eq!(min, (3, 2, 5));
    assert_eq!(max, (10, 8, 12));
}

#[test]
fn index_zero_is_air() {
    let grid = VoxelGrid::new(8, 8, 8);
    // All voxels should be 0 (air) by default
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                assert_eq!(grid.get(x, y, z), Some(0));
            }
        }
    }
}
