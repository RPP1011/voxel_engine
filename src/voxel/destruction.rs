use super::grid::VoxelGrid;

/// Remove all voxels within a sphere centered at `center` with given `radius`.
/// Returns the number of voxels actually removed (were non-zero).
pub fn remove_sphere(grid: &mut VoxelGrid, center: (u32, u32, u32), radius: f32) -> usize {
    let (w, h, d) = grid.dimensions();
    let r2 = radius * radius;
    let cx = center.0 as f32;
    let cy = center.1 as f32;
    let cz = center.2 as f32;

    let min_x = (cx - radius).max(0.0) as u32;
    let max_x = ((cx + radius) as u32).min(w - 1);
    let min_y = (cy - radius).max(0.0) as u32;
    let max_y = ((cy + radius) as u32).min(h - 1);
    let min_z = (cz - radius).max(0.0) as u32;
    let max_z = ((cz + radius) as u32).min(d - 1);

    let mut removed = 0;
    for z in min_z..=max_z {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dz = z as f32 - cz;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    if grid.get(x, y, z) != Some(0) {
                        grid.set(x, y, z, 0);
                        removed += 1;
                    }
                }
            }
        }
    }
    removed
}

/// Remove all voxels in the axis-aligned box from `min` to `max` (inclusive).
/// Clamps to grid bounds. Returns the number of voxels actually removed.
pub fn remove_box(grid: &mut VoxelGrid, min: (u32, u32, u32), max: (u32, u32, u32)) -> usize {
    let (w, h, d) = grid.dimensions();
    let x0 = min.0;
    let y0 = min.1;
    let z0 = min.2;
    let x1 = max.0.min(w - 1);
    let y1 = max.1.min(h - 1);
    let z1 = max.2.min(d - 1);

    let mut removed = 0;
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if grid.get(x, y, z) != Some(0) {
                    grid.set(x, y, z, 0);
                    removed += 1;
                }
            }
        }
    }
    removed
}
