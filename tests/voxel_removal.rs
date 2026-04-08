use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::destruction::{remove_sphere, remove_box};

#[test]
fn single_voxel_removal() {
    let mut grid = VoxelGrid::new(8, 8, 8);
    // Fill the grid
    for x in 0..8 { for y in 0..8 { for z in 0..8 { grid.set(x, y, z, 1); } } }

    grid.set(4, 4, 4, 0); // remove single voxel

    assert_eq!(grid.get(4, 4, 4), Some(0), "Removed voxel should be air");
    assert_eq!(grid.get(3, 4, 4), Some(1), "Neighbor should be unchanged");
    assert_eq!(grid.get(5, 4, 4), Some(1), "Neighbor should be unchanged");
}

#[test]
fn remove_sphere_correct_count() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    // Fill entire grid
    for x in 0..16 { for y in 0..16 { for z in 0..16 { grid.set(x, y, z, 1); } } }

    let initial = grid.count_nonempty();
    assert_eq!(initial, 16 * 16 * 16);

    let removed = remove_sphere(&mut grid, (8, 8, 8), 3.0);

    let remaining = grid.count_nonempty();
    assert_eq!(remaining, initial - removed);
    // Approximate sphere volume: 4/3 * pi * r^3 ≈ 113 for r=3
    assert!(removed > 80, "Should remove a reasonable number of voxels, got {removed}");
    assert!(removed < 150, "Should not remove too many, got {removed}");
}

#[test]
fn remove_box_within_bounds() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    for x in 0..16 { for y in 0..16 { for z in 0..16 { grid.set(x, y, z, 1); } } }

    let removed = remove_box(&mut grid, (2, 2, 2), (5, 5, 5));

    // Box is 4x4x4 = 64 voxels
    assert_eq!(removed, 64);
    assert_eq!(grid.get(2, 2, 2), Some(0));
    assert_eq!(grid.get(5, 5, 5), Some(0));
    assert_eq!(grid.get(1, 2, 2), Some(1)); // outside box
    assert_eq!(grid.get(6, 5, 5), Some(1)); // outside box
}

#[test]
fn remove_box_on_grid_edge_no_panic() {
    let mut grid = VoxelGrid::new(8, 8, 8);
    for x in 0..8 { for y in 0..8 { for z in 0..8 { grid.set(x, y, z, 1); } } }

    // Box extends beyond grid bounds
    let removed = remove_box(&mut grid, (5, 5, 5), (20, 20, 20));

    // Should only remove voxels within bounds: 3x3x3 = 27
    assert_eq!(removed, 27);
}

#[test]
fn remove_from_empty_region_is_noop() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    // Grid is all zeros (air) by default

    let removed = remove_sphere(&mut grid, (8, 8, 8), 3.0);
    assert_eq!(removed, 0, "Nothing to remove from empty grid");
}
