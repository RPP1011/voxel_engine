use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::ai::spatial::{query_distance, query_line_of_sight};

fn solid_cube_grid() -> VoxelGrid {
    let mut grid = VoxelGrid::new(8, 8, 8);
    for x in 2..6 { for y in 2..6 { for z in 2..6 { grid.set(x, y, z, 1); } } }
    grid
}

#[test]
fn point_inside_solid_negative_distance() {
    let grid = solid_cube_grid();
    let dist = query_distance(&grid, [4.0, 4.0, 4.0], 1.0);
    assert!(dist < 0.0, "Point inside solid should have negative distance, got {dist}");
}

#[test]
fn point_outside_positive_distance() {
    let grid = solid_cube_grid();
    // Point 5 voxels away from surface (cube spans 2..6, surface at x=6, point at x=11)
    let dist = query_distance(&grid, [11.0, 4.0, 4.0], 1.0);
    assert!(
        (dist - 5.0).abs() < 1.5,
        "Point ~5 voxels from surface should have distance ≈5, got {dist}"
    );
}

#[test]
fn los_blocked_by_wall() {
    let mut grid = VoxelGrid::new(16, 8, 8);
    // Build a wall at x=8
    for y in 0..8 { for z in 0..8 { grid.set(8, y, z, 1); } }

    let visible = query_line_of_sight(&grid, [2.0, 4.0, 4.0], [14.0, 4.0, 4.0], 1.0);
    assert!(!visible, "LOS should be blocked by wall");
}

#[test]
fn los_clear_in_open_space() {
    let grid = VoxelGrid::new(16, 8, 8); // empty grid

    let visible = query_line_of_sight(&grid, [2.0, 4.0, 4.0], [14.0, 4.0, 4.0], 1.0);
    assert!(visible, "LOS should be clear in empty space");
}
