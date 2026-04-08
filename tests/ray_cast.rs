use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::raycast::{ray_cast_grid, RayHit, FaceDir};

#[test]
fn ray_hits_specific_voxel() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    grid.set(5, 5, 5, 1);

    // Ray from (5, 5, 0) going +Z
    let hit = ray_cast_grid(&grid, [5.5, 5.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = hit.expect("Should hit voxel (5,5,5)");
    assert_eq!(hit.voxel, (5, 5, 5));
    assert_eq!(hit.face, FaceDir::NegZ); // hit the -Z face of the voxel
}

#[test]
fn ray_misses_empty_grid() {
    let grid = VoxelGrid::new(8, 8, 8);
    let hit = ray_cast_grid(&grid, [4.5, 4.5, -5.0], [0.0, 0.0, 1.0]);
    assert!(hit.is_none(), "Should miss empty grid");
}

#[test]
fn ray_hits_closer_voxel_first() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    grid.set(5, 5, 3, 1); // closer
    grid.set(5, 5, 10, 2); // farther

    let hit = ray_cast_grid(&grid, [5.5, 5.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = hit.expect("Should hit a voxel");
    assert_eq!(hit.voxel, (5, 5, 3), "Should hit closer voxel first");
}

#[test]
fn ray_from_outside_grid_entering() {
    let mut grid = VoxelGrid::new(8, 8, 8);
    // Fill front face
    for x in 0..8 { for y in 0..8 { grid.set(x, y, 0, 1); } }

    let hit = ray_cast_grid(&grid, [4.5, 4.5, -5.0], [0.0, 0.0, 1.0]);
    let hit = hit.expect("Should hit front face");
    assert_eq!(hit.voxel.2, 0, "Should hit z=0 face");
}

#[test]
fn diagonal_ray() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    grid.set(8, 8, 8, 1);

    // Diagonal ray from origin toward (8,8,8)
    let dir = [1.0f32, 1.0, 1.0];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let dir = [dir[0] / len, dir[1] / len, dir[2] / len];

    let hit = ray_cast_grid(&grid, [0.5, 0.5, 0.5], dir);
    let hit = hit.expect("Should hit voxel on diagonal");
    assert_eq!(hit.voxel, (8, 8, 8));
}
