use voxel_engine::ai::navmesh::NavMesh;
use voxel_engine::voxel::grid::VoxelGrid;

fn flat_open_grid() -> VoxelGrid {
    let mut grid = VoxelGrid::new(32, 8, 32);
    // Floor at y=0
    for x in 0..32 { for z in 0..32 { grid.set(x, 0, z, 1); } }
    grid
}

#[test]
fn flat_open_area_single_walkable_region() {
    let grid = flat_open_grid();
    let mut nav = NavMesh::new(16, 2);
    nav.build_tile(&grid, 0, 0);
    nav.build_tile(&grid, 1, 0);
    nav.build_tile(&grid, 0, 1);
    nav.build_tile(&grid, 1, 1);

    // All tiles should be walkable
    assert!(nav.is_walkable(8, 8));
    assert!(nav.is_walkable(24, 24));
}

#[test]
fn destroy_floor_tile_not_walkable() {
    let mut grid = flat_open_grid();
    let mut nav = NavMesh::new(16, 2);
    nav.build_tile(&grid, 0, 0);

    assert!(nav.is_walkable(8, 8));

    // Destroy floor under (8, 8)
    grid.set(8, 0, 8, 0);
    nav.mark_dirty(0, 0);

    // Rebuild dirty tile
    nav.build_tile(&grid, 0, 0);

    assert!(!nav.is_walkable(8, 8), "Destroyed floor should not be walkable");
}

#[test]
fn astar_finds_shortest_path() {
    let grid = flat_open_grid();
    let mut nav = NavMesh::new(16, 2);
    nav.build_tile(&grid, 0, 0);
    nav.build_tile(&grid, 1, 0);

    let path = nav.find_path((2, 2), (28, 2));
    let path = path.expect("Should find path on flat terrain");

    // Path should be roughly 26 steps (Manhattan distance)
    assert!(path.len() >= 20, "Path too short: {} steps", path.len());
    assert_eq!(path.first(), Some(&(2, 2)));
    assert_eq!(path.last(), Some(&(28, 2)));
}

#[test]
fn pathfinding_around_obstacle() {
    let mut grid = flat_open_grid();
    // Wall blocking direct path at x=16 (except gap at z=0..1)
    // Wall: remove the floor at x=16 and replace with full solid column
    // The walkability check needs no valid floor+headroom combination
    for z in 2..30 {
        // Remove floor
        grid.set(16, 0, z, 0);
        // No floor means not walkable at x=16
    }

    let mut nav = NavMesh::new(16, 2);
    nav.build_tile(&grid, 0, 0);
    nav.build_tile(&grid, 1, 0);

    let path = nav.find_path((8, 15), (24, 15));

    // Wall blocks direct path at x=16, z=15 — must go around through z=0..1
    let path = path.expect("Should find path around wall");
    assert!(path.len() > 20, "Path around wall should be longer than direct");
}
