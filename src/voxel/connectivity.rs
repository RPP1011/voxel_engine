use std::collections::{HashSet, VecDeque};
use super::grid::VoxelGrid;

/// 6-connected neighbors (face-adjacent).
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
];

/// Find all connected components of non-empty voxels using 6-connected flood fill.
/// Returns a list of components, each being a set of (x, y, z) coordinates.
pub fn find_connected_components(grid: &VoxelGrid) -> Vec<HashSet<(u32, u32, u32)>> {
    let (w, h, d) = grid.dimensions();
    let mut visited = vec![false; (w * h * d) as usize];
    let mut components = Vec::new();

    let idx = |x: u32, y: u32, z: u32| -> usize {
        (z * h * w + y * w + x) as usize
    };

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let i = idx(x, y, z);
                if visited[i] || grid.get(x, y, z) == Some(0) {
                    continue;
                }

                // BFS flood fill from this voxel
                let mut component = HashSet::new();
                let mut queue = VecDeque::new();
                queue.push_back((x, y, z));
                visited[i] = true;

                while let Some((cx, cy, cz)) = queue.pop_front() {
                    component.insert((cx, cy, cz));

                    for &(dx, dy, dz) in &NEIGHBORS {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        let nz = cz as i32 + dz;

                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                        if nx >= w || ny >= h || nz >= d {
                            continue;
                        }

                        let ni = idx(nx, ny, nz);
                        if !visited[ni] && grid.get(nx, ny, nz) != Some(0) {
                            visited[ni] = true;
                            queue.push_back((nx, ny, nz));
                        }
                    }
                }

                components.push(component);
            }
        }
    }

    components
}

/// Given a set of anchor voxels (e.g., ground-level), find which non-empty voxels
/// are reachable from anchors (structurally supported) and which are unsupported fragments.
pub fn find_anchored_and_fragments(
    grid: &VoxelGrid,
    anchors: &HashSet<(u32, u32, u32)>,
) -> (HashSet<(u32, u32, u32)>, Vec<HashSet<(u32, u32, u32)>>) {
    let (w, h, d) = grid.dimensions();
    let mut visited = vec![false; (w * h * d) as usize];
    let idx = |x: u32, y: u32, z: u32| -> usize {
        (z * h * w + y * w + x) as usize
    };

    // BFS from all anchors simultaneously
    let mut supported = HashSet::new();
    let mut queue = VecDeque::new();

    for &(x, y, z) in anchors {
        if grid.get(x, y, z) != Some(0) {
            let i = idx(x, y, z);
            if !visited[i] {
                visited[i] = true;
                queue.push_back((x, y, z));
            }
        }
    }

    while let Some((cx, cy, cz)) = queue.pop_front() {
        supported.insert((cx, cy, cz));

        for &(dx, dy, dz) in &NEIGHBORS {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            let nz = cz as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 { continue; }
            let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
            if nx >= w || ny >= h || nz >= d { continue; }

            let ni = idx(nx, ny, nz);
            if !visited[ni] && grid.get(nx, ny, nz) != Some(0) {
                visited[ni] = true;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    // Now find unsupported fragments (non-empty voxels not in `supported`)
    let mut fragments = Vec::new();
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let i = idx(x, y, z);
                if visited[i] || grid.get(x, y, z) == Some(0) {
                    continue;
                }

                // BFS for this unsupported component
                let mut fragment = HashSet::new();
                let mut fq = VecDeque::new();
                fq.push_back((x, y, z));
                visited[i] = true;

                while let Some((cx, cy, cz)) = fq.pop_front() {
                    fragment.insert((cx, cy, cz));
                    for &(dx, dy, dz) in &NEIGHBORS {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        let nz = cz as i32 + dz;
                        if nx < 0 || ny < 0 || nz < 0 { continue; }
                        let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                        if nx >= w || ny >= h || nz >= d { continue; }

                        let ni = idx(nx, ny, nz);
                        if !visited[ni] && grid.get(nx, ny, nz) != Some(0) {
                            visited[ni] = true;
                            fq.push_back((nx, ny, nz));
                        }
                    }
                }

                fragments.push(fragment);
            }
        }
    }

    (supported, fragments)
}

/// For static objects, bottom layer (y=0) non-empty voxels are anchors.
pub fn ground_anchors(grid: &VoxelGrid) -> HashSet<(u32, u32, u32)> {
    let (w, _h, d) = grid.dimensions();
    let mut anchors = HashSet::new();
    for z in 0..d {
        for x in 0..w {
            if grid.get(x, 0, z) != Some(0) {
                anchors.insert((x, 0, z));
            }
        }
    }
    anchors
}
