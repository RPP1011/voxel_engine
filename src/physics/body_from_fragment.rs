use crate::voxel::grid::VoxelGrid;
use crate::voxel::material::MaterialType;

/// Physics body properties computed from a voxel fragment.
pub struct PhysicsBodyInfo {
    pub mass: f64,
    pub center_of_mass: [f64; 3],
    /// 3x3 inertia tensor (stored as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]).
    pub inertia: [f64; 6],
}

impl PhysicsBodyInfo {
    /// Compute inherited linear velocity for this fragment given parent's
    /// velocity, angular velocity, and the fragment's offset from parent CoM.
    pub fn inherited_velocity(
        &self,
        parent_linear_vel: [f64; 3],
        parent_angular_vel: [f64; 3],
        offset_from_parent: [f64; 3],
    ) -> [f64; 3] {
        // v = v_parent + ω × r
        let wx = parent_angular_vel[0];
        let wy = parent_angular_vel[1];
        let wz = parent_angular_vel[2];
        let rx = offset_from_parent[0];
        let ry = offset_from_parent[1];
        let rz = offset_from_parent[2];

        [
            parent_linear_vel[0] + (wy * rz - wz * ry),
            parent_linear_vel[1] + (wz * rx - wx * rz),
            parent_linear_vel[2] + (wx * ry - wy * rx),
        ]
    }
}

/// Compute mass, center of mass, and inertia tensor for a voxel grid fragment.
/// `voxel_scale` is meters per voxel (e.g., 0.1 for 10cm voxels).
pub fn compute_body_info(
    grid: &VoxelGrid,
    material: MaterialType,
    voxel_scale: f64,
) -> PhysicsBodyInfo {
    let (w, h, d) = grid.dimensions();
    let density = material.density_kg_m3() as f64;
    let voxel_volume = voxel_scale * voxel_scale * voxel_scale;
    let voxel_mass = density * voxel_volume;

    let mut total_mass = 0.0;
    let mut com = [0.0f64; 3];
    let mut count = 0u64;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if grid.get(x, y, z) != Some(0) {
                    let px = (x as f64 + 0.5) * voxel_scale;
                    let py = (y as f64 + 0.5) * voxel_scale;
                    let pz = (z as f64 + 0.5) * voxel_scale;

                    total_mass += voxel_mass;
                    com[0] += px * voxel_mass;
                    com[1] += py * voxel_mass;
                    com[2] += pz * voxel_mass;
                    count += 1;
                }
            }
        }
    }

    if total_mass > 0.0 {
        com[0] /= total_mass;
        com[1] /= total_mass;
        com[2] /= total_mass;
    }

    // Inertia tensor (relative to center of mass)
    let mut ixx = 0.0;
    let mut iyy = 0.0;
    let mut izz = 0.0;
    let mut ixy = 0.0;
    let mut ixz = 0.0;
    let mut iyz = 0.0;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if grid.get(x, y, z) != Some(0) {
                    let dx = (x as f64 + 0.5) * voxel_scale - com[0];
                    let dy = (y as f64 + 0.5) * voxel_scale - com[1];
                    let dz = (z as f64 + 0.5) * voxel_scale - com[2];

                    ixx += voxel_mass * (dy * dy + dz * dz);
                    iyy += voxel_mass * (dx * dx + dz * dz);
                    izz += voxel_mass * (dx * dx + dy * dy);
                    ixy -= voxel_mass * dx * dy;
                    ixz -= voxel_mass * dx * dz;
                    iyz -= voxel_mass * dy * dz;
                }
            }
        }
    }

    PhysicsBodyInfo {
        mass: total_mass,
        center_of_mass: com,
        inertia: [ixx, iyy, izz, ixy, ixz, iyz],
    }
}
