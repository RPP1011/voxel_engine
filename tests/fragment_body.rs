use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialType;
use voxel_engine::physics::body_from_fragment::{PhysicsBodyInfo, compute_body_info};

#[test]
fn solid_wood_cube_mass() {
    // 4x4x4 solid wood cube at 10cm resolution
    let mut grid = VoxelGrid::new(4, 4, 4);
    for x in 0..4 { for y in 0..4 { for z in 0..4 { grid.set(x, y, z, 1); } } }

    let info = compute_body_info(&grid, MaterialType::Wood, 0.1);

    // mass = 64 voxels * 0.1^3 m^3 * 500 kg/m^3 = 64 * 0.001 * 500 = 32 kg
    assert!(
        (info.mass - 32.0).abs() < 0.1,
        "Mass should be ~32 kg, got {}",
        info.mass
    );
}

#[test]
fn symmetric_object_com_at_center() {
    let mut grid = VoxelGrid::new(4, 4, 4);
    for x in 0..4 { for y in 0..4 { for z in 0..4 { grid.set(x, y, z, 1); } } }

    let info = compute_body_info(&grid, MaterialType::Wood, 0.1);

    // Center of mass should be at geometric center: (2, 2, 2) * 0.1 = (0.2, 0.2, 0.2)
    assert!((info.center_of_mass[0] - 0.2).abs() < 0.01);
    assert!((info.center_of_mass[1] - 0.2).abs() < 0.01);
    assert!((info.center_of_mass[2] - 0.2).abs() < 0.01);
}

#[test]
fn asymmetric_l_shape_com_offset() {
    // L-shape: heavy base + thin arm
    let mut grid = VoxelGrid::new(8, 8, 4);
    // Base: 8x4x4
    for x in 0..8 { for y in 0..4 { for z in 0..4 { grid.set(x, y, z, 1); } } }
    // Arm: 2x8x4 at the left end
    for x in 0..2 { for y in 4..8 { for z in 0..4 { grid.set(x, y, z, 1); } } }

    let info = compute_body_info(&grid, MaterialType::Metal, 0.1);

    // CoM should be offset toward the base (where most mass is)
    // Base has 128 voxels, arm has 32 voxels
    // CoM x should be > 2.0 * scale (center of arm) — offset toward base center
    assert!(
        info.center_of_mass[0] > 0.15,
        "CoM should be offset toward heavy base, got x={}",
        info.center_of_mass[0]
    );
}

#[test]
fn velocity_inheritance_from_parent() {
    let mut grid = VoxelGrid::new(4, 4, 4);
    for x in 0..4 { for y in 0..4 { for z in 0..4 { grid.set(x, y, z, 1); } } }

    let info = compute_body_info(&grid, MaterialType::Wood, 0.1);

    // Fragment at offset (10, 0, 0) from parent with parent angular velocity (0, 1, 0)
    let parent_vel = [0.0f64, 0.0, 0.0];
    let parent_angular = [0.0f64, 1.0, 0.0]; // rotating around Y
    let fragment_offset = [1.0f64, 0.0, 0.0]; // 1m to the right

    let inherited = info.inherited_velocity(parent_vel, parent_angular, fragment_offset);

    // v = v_parent + ω × r = (0,0,0) + (0,1,0) × (1,0,0) = (0,0,-1)
    assert!((inherited[0]).abs() < 0.01, "vx should be ~0, got {}", inherited[0]);
    assert!((inherited[1]).abs() < 0.01, "vy should be ~0, got {}", inherited[1]);
    assert!((inherited[2] - (-1.0)).abs() < 0.01, "vz should be ~-1, got {}", inherited[2]);
}
