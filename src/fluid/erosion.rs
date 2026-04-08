use crate::voxel::grid::VoxelGrid;
use super::swe::SweGrid;

/// Update SWE terrain heights from the voxel grid.
/// Terrain height at each (x, z) cell = highest solid voxel y + 1.
pub fn update_terrain_from_voxels(swe: &mut SweGrid, grid: &VoxelGrid) {
    let (vw, vh, vd) = grid.dimensions();
    let (sw, sh) = swe.dimensions();

    for z in 0..sh.min(vd as usize) {
        for x in 0..sw.min(vw as usize) {
            let mut max_y = 0.0;
            for y in (0..vh).rev() {
                if grid.get(x as u32, y, z as u32) != Some(0) {
                    max_y = (y + 1) as f64;
                    break;
                }
            }
            swe.set_terrain_height(x, z, max_y);
        }
    }
}

/// Apply erosion: fast-flowing water removes surface voxels.
/// `erosion_rate`: how much flow speed causes erosion per step.
/// `threshold`: minimum flow speed to erode.
pub fn apply_erosion(
    swe: &SweGrid,
    grid: &mut VoxelGrid,
    erosion_rate: f64,
    threshold: f64,
) {
    let (sw, sh) = swe.dimensions();
    let (vw, vh, vd) = grid.dimensions();

    for z in 0..sh.min(vd as usize) {
        for x in 0..sw.min(vw as usize) {
            let speed = swe.flow_speed(x, z);
            if speed < threshold {
                continue;
            }

            // Find topmost solid voxel
            let mut top_y = None;
            for y in (0..vh).rev() {
                if grid.get(x as u32, y, z as u32) != Some(0) {
                    top_y = Some(y);
                    break;
                }
            }

            if let Some(y) = top_y {
                // Erode with probability proportional to flow speed
                let erode_strength = (speed - threshold) * erosion_rate;
                if erode_strength > 0.5 {
                    grid.set(x as u32, y, z as u32, 0);
                }
            }
        }
    }
}
