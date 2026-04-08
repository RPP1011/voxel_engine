use crate::voxel::grid::VoxelGrid;

/// Query approximate signed distance from a world-space point to the nearest surface.
/// Negative = inside solid, positive = outside.
/// `voxel_scale` = meters per voxel.
pub fn query_distance(grid: &VoxelGrid, point: [f64; 3], voxel_scale: f64) -> f64 {
    let (w, h, d) = grid.dimensions();

    // Convert to voxel space
    let vx = point[0] / voxel_scale;
    let vy = point[1] / voxel_scale;
    let vz = point[2] / voxel_scale;

    // Check if the point's voxel is solid
    let ix = vx.floor() as i32;
    let iy = vy.floor() as i32;
    let iz = vz.floor() as i32;

    let point_is_solid = if ix >= 0 && iy >= 0 && iz >= 0
        && (ix as u32) < w && (iy as u32) < h && (iz as u32) < d
    {
        grid.get(ix as u32, iy as u32, iz as u32) != Some(0)
    } else {
        false
    };

    // Brute-force nearest surface distance (for correctness; optimize later with JFA)
    let search_radius = 16i32;
    let mut min_dist_sq = f64::MAX;

    let cx = ix.clamp(0, w as i32 - 1);
    let cy = iy.clamp(0, h as i32 - 1);
    let cz = iz.clamp(0, d as i32 - 1);

    for dz in -search_radius..=search_radius {
        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let sx = cx + dx;
                let sy = cy + dy;
                let sz = cz + dz;
                if sx < 0 || sy < 0 || sz < 0
                    || sx >= w as i32 || sy >= h as i32 || sz >= d as i32
                {
                    continue;
                }

                let is_solid = grid.get(sx as u32, sy as u32, sz as u32) != Some(0);

                // We want distance to the surface: find voxels that are at the
                // solid/air boundary
                if is_solid != point_is_solid {
                    // This is a surface voxel (different state from query point)
                    let center_x = sx as f64 + 0.5;
                    let center_y = sy as f64 + 0.5;
                    let center_z = sz as f64 + 0.5;
                    let ddx = vx - center_x;
                    let ddy = vy - center_y;
                    let ddz = vz - center_z;
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    min_dist_sq = min_dist_sq.min(dist_sq);
                }
            }
        }
    }

    let dist = min_dist_sq.sqrt() * voxel_scale;
    if point_is_solid { -dist } else { dist }
}

/// Query line-of-sight between two points using DDA ray marching.
/// Returns true if the ray from `from` to `to` does not hit any solid voxel.
pub fn query_line_of_sight(
    grid: &VoxelGrid,
    from: [f64; 3],
    to: [f64; 3],
    voxel_scale: f64,
) -> bool {
    let (w, h, d) = grid.dimensions();

    // Convert to voxel space
    let ox = from[0] / voxel_scale;
    let oy = from[1] / voxel_scale;
    let oz = from[2] / voxel_scale;
    let tx = to[0] / voxel_scale;
    let ty = to[1] / voxel_scale;
    let tz = to[2] / voxel_scale;

    let dx = tx - ox;
    let dy = ty - oy;
    let dz = tz - oz;
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len < 1e-10 {
        return true;
    }

    let dir_x = dx / len;
    let dir_y = dy / len;
    let dir_z = dz / len;

    // March through voxels using DDA
    let mut x = ox.floor() as i32;
    let mut y = oy.floor() as i32;
    let mut z = oz.floor() as i32;

    let step_x: i32 = if dir_x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir_y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir_z >= 0.0 { 1 } else { -1 };

    let t_delta_x = if dir_x.abs() < 1e-10 { f64::MAX } else { (1.0 / dir_x).abs() };
    let t_delta_y = if dir_y.abs() < 1e-10 { f64::MAX } else { (1.0 / dir_y).abs() };
    let t_delta_z = if dir_z.abs() < 1e-10 { f64::MAX } else { (1.0 / dir_z).abs() };

    let nb_x = if step_x > 0 { (x + 1) as f64 } else { x as f64 };
    let nb_y = if step_y > 0 { (y + 1) as f64 } else { y as f64 };
    let nb_z = if step_z > 0 { (z + 1) as f64 } else { z as f64 };

    let mut t_max_x = if dir_x.abs() < 1e-10 { f64::MAX } else { (nb_x - ox) / dir_x };
    let mut t_max_y = if dir_y.abs() < 1e-10 { f64::MAX } else { (nb_y - oy) / dir_y };
    let mut t_max_z = if dir_z.abs() < 1e-10 { f64::MAX } else { (nb_z - oz) / dir_z };

    let max_steps = (len.ceil() as usize + 2) * 3;

    for _ in 0..max_steps {
        // Check current voxel
        if x >= 0 && y >= 0 && z >= 0
            && (x as u32) < w && (y as u32) < h && (z as u32) < d
        {
            if grid.get(x as u32, y as u32, z as u32) != Some(0) {
                return false; // Blocked
            }
        }

        // Advance
        let t_min = t_max_x.min(t_max_y).min(t_max_z);
        if t_min > len {
            break; // Reached destination
        }

        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            x += step_x;
            t_max_x += t_delta_x;
        } else if t_max_y <= t_max_z {
            y += step_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t_max_z += t_delta_z;
        }
    }

    true // No obstruction found
}
