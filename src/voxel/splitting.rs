use super::connectivity::find_connected_components;
use super::grid::VoxelGrid;

/// A fragment produced by splitting a disconnected VoxelGrid.
pub struct Fragment {
    /// The new grid containing only this fragment's voxels, sized to its bounding box.
    pub grid: VoxelGrid,
    /// Offset of this fragment's grid origin relative to the original grid's origin.
    pub offset: (u32, u32, u32),
}

/// Split a VoxelGrid into fragments based on connectivity.
/// Components with fewer than `min_voxels` are discarded.
/// Returns one Fragment per surviving connected component.
pub fn split_into_fragments(grid: &VoxelGrid, min_voxels: usize) -> Vec<Fragment> {
    let components = find_connected_components(grid);

    components
        .into_iter()
        .filter(|comp| comp.len() >= min_voxels.max(1))
        .map(|comp| {
            // Find bounding box of this component
            let mut min_x = u32::MAX;
            let mut min_y = u32::MAX;
            let mut min_z = u32::MAX;
            let mut max_x = 0u32;
            let mut max_y = 0u32;
            let mut max_z = 0u32;

            for &(x, y, z) in &comp {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                min_z = min_z.min(z);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                max_z = max_z.max(z);
            }

            let w = max_x - min_x + 1;
            let h = max_y - min_y + 1;
            let d = max_z - min_z + 1;
            let mut new_grid = VoxelGrid::new(w, h, d);

            for &(x, y, z) in &comp {
                let val = grid.get(x, y, z).unwrap_or(0);
                new_grid.set(x - min_x, y - min_y, z - min_z, val);
            }

            Fragment {
                grid: new_grid,
                offset: (min_x, min_y, min_z),
            }
        })
        .collect()
}
