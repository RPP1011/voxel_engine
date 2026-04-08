use super::grid::VoxelGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceDir {
    PosX, NegX,
    PosY, NegY,
    PosZ, NegZ,
}

#[derive(Debug)]
pub struct RayHit {
    pub voxel: (u32, u32, u32),
    pub face: FaceDir,
    pub t: f32,
}

/// Amanatides-Woo DDA ray march through a VoxelGrid.
/// `origin` and `dir` are in grid-local space (voxel units).
/// Returns the first non-empty voxel hit, if any.
pub fn ray_cast_grid(grid: &VoxelGrid, origin: [f32; 3], dir: [f32; 3]) -> Option<RayHit> {
    let (w, h, d) = grid.dimensions();
    let (w, h, d) = (w as f32, h as f32, d as f32);

    // Find entry point into grid AABB
    let (t_enter, t_exit) = ray_aabb(origin, dir, [0.0, 0.0, 0.0], [w, h, d])?;
    let t_start = t_enter.max(0.0) + 1e-4;

    if t_start >= t_exit {
        return None;
    }

    // Starting position
    let px = origin[0] + dir[0] * t_start;
    let py = origin[1] + dir[1] * t_start;
    let pz = origin[2] + dir[2] * t_start;

    let mut x = (px.floor() as i32).clamp(0, w as i32 - 1);
    let mut y = (py.floor() as i32).clamp(0, h as i32 - 1);
    let mut z = (pz.floor() as i32).clamp(0, d as i32 - 1);

    let step_x: i32 = if dir[0] >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir[1] >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir[2] >= 0.0 { 1 } else { -1 };

    let t_delta_x = if dir[0].abs() < 1e-10 { f32::MAX } else { (1.0 / dir[0]).abs() };
    let t_delta_y = if dir[1].abs() < 1e-10 { f32::MAX } else { (1.0 / dir[1]).abs() };
    let t_delta_z = if dir[2].abs() < 1e-10 { f32::MAX } else { (1.0 / dir[2]).abs() };

    let next_boundary_x = if step_x > 0 { (x + 1) as f32 } else { x as f32 };
    let next_boundary_y = if step_y > 0 { (y + 1) as f32 } else { y as f32 };
    let next_boundary_z = if step_z > 0 { (z + 1) as f32 } else { z as f32 };

    let mut t_max_x = if dir[0].abs() < 1e-10 { f32::MAX } else { (next_boundary_x - px) / dir[0] };
    let mut t_max_y = if dir[1].abs() < 1e-10 { f32::MAX } else { (next_boundary_y - py) / dir[1] };
    let mut t_max_z = if dir[2].abs() < 1e-10 { f32::MAX } else { (next_boundary_z - pz) / dir[2] };

    let mut last_face = FaceDir::NegZ; // default, will be set by first step
    let max_steps = (w + h + d) as usize * 2;

    // Check starting voxel
    if x >= 0 && y >= 0 && z >= 0
        && (x as u32) < w as u32
        && (y as u32) < h as u32
        && (z as u32) < d as u32
    {
        if grid.get(x as u32, y as u32, z as u32) != Some(0) {
            // Determine which face we entered through
            last_face = entry_face(origin, dir, t_start, x, y, z);
            return Some(RayHit {
                voxel: (x as u32, y as u32, z as u32),
                face: last_face,
                t: t_start,
            });
        }
    }

    for _ in 0..max_steps {
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                x += step_x;
                t_max_x += t_delta_x;
                last_face = if step_x > 0 { FaceDir::NegX } else { FaceDir::PosX };
            } else {
                z += step_z;
                t_max_z += t_delta_z;
                last_face = if step_z > 0 { FaceDir::NegZ } else { FaceDir::PosZ };
            }
        } else {
            if t_max_y < t_max_z {
                y += step_y;
                t_max_y += t_delta_y;
                last_face = if step_y > 0 { FaceDir::NegY } else { FaceDir::PosY };
            } else {
                z += step_z;
                t_max_z += t_delta_z;
                last_face = if step_z > 0 { FaceDir::NegZ } else { FaceDir::PosZ };
            }
        }

        // Out of bounds check
        if x < 0 || y < 0 || z < 0
            || x >= w as i32
            || y >= h as i32
            || z >= d as i32
        {
            return None;
        }

        if grid.get(x as u32, y as u32, z as u32) != Some(0) {
            let t = t_max_x.min(t_max_y).min(t_max_z) + t_start;
            return Some(RayHit {
                voxel: (x as u32, y as u32, z as u32),
                face: last_face,
                t,
            });
        }
    }

    None
}

fn entry_face(origin: [f32; 3], dir: [f32; 3], t: f32, vx: i32, vy: i32, vz: i32) -> FaceDir {
    let px = origin[0] + dir[0] * t;
    let py = origin[1] + dir[1] * t;
    let pz = origin[2] + dir[2] * t;

    let dx_min = (px - vx as f32).abs();
    let dx_max = (px - (vx + 1) as f32).abs();
    let dy_min = (py - vy as f32).abs();
    let dy_max = (py - (vy + 1) as f32).abs();
    let dz_min = (pz - vz as f32).abs();
    let dz_max = (pz - (vz + 1) as f32).abs();

    let min_d = dx_min.min(dx_max).min(dy_min).min(dy_max).min(dz_min).min(dz_max);

    if (min_d - dx_min).abs() < 1e-3 { FaceDir::NegX }
    else if (min_d - dx_max).abs() < 1e-3 { FaceDir::PosX }
    else if (min_d - dy_min).abs() < 1e-3 { FaceDir::NegY }
    else if (min_d - dy_max).abs() < 1e-3 { FaceDir::PosY }
    else if (min_d - dz_min).abs() < 1e-3 { FaceDir::NegZ }
    else { FaceDir::PosZ }
}

fn ray_aabb(
    origin: [f32; 3],
    dir: [f32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> Option<(f32, f32)> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for i in 0..3 {
        if dir[i].abs() < 1e-10 {
            if origin[i] < aabb_min[i] || origin[i] > aabb_max[i] {
                return None;
            }
        } else {
            let inv_d = 1.0 / dir[i];
            let mut t1 = (aabb_min[i] - origin[i]) * inv_d;
            let mut t2 = (aabb_max[i] - origin[i]) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    Some((t_min, t_max))
}
