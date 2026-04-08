use crate::voxel::grid::VoxelGrid;
use super::spatial::query_line_of_sight;

/// Check if a cell at (x, floor_y, z) is walkable — ground exists and headroom sufficient.
/// `agent_height` is in voxel units.
pub fn is_walkable(grid: &VoxelGrid, x: u32, floor_y: u32, z: u32, agent_height: f64) -> bool {
    let (_, h, _) = grid.dimensions();

    // Must have solid ground at floor_y
    if grid.get(x, floor_y, z) != Some(0) {
        // floor_y is solid — check headroom above it
        let start_y = floor_y + 1;
        let needed = agent_height.ceil() as u32;

        for dy in 0..needed {
            let check_y = start_y + dy;
            if check_y >= h {
                return true; // Open sky = enough headroom
            }
            if grid.get(x, check_y, z) != Some(0) {
                return false; // Blocked by ceiling
            }
        }
        return true;
    }

    false // No ground
}

/// Check if a position has cover from a threat direction.
/// Uses LOS check — if LOS is blocked, position has cover.
pub fn has_cover(grid: &VoxelGrid, position: [f64; 3], threat: [f64; 3], voxel_scale: f64) -> bool {
    !query_line_of_sight(grid, position, threat, voxel_scale)
}
