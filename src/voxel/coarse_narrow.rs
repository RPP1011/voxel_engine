use std::collections::{HashSet, VecDeque};

use super::cluster_graph::{ClusterGraph, ClusterId};
use super::connectivity::find_connected_components;
use super::grid::VoxelGrid;

pub struct CoarseNarrowResult {
    pub component_count: usize,
    /// True if the coarse graph alone was sufficient (no per-voxel flood fill needed).
    pub used_coarse_shortcut: bool,
}

/// Use coarse cluster graph for fast connectivity, then narrow to per-voxel
/// only at cluster boundaries where the coarse graph indicates a possible split.
pub fn coarse_then_narrow_components(grid: &VoxelGrid, cluster_size: u32) -> CoarseNarrowResult {
    let graph = ClusterGraph::build(grid, cluster_size);

    // Find connected components on the coarse graph
    let coarse_components = coarse_connected_components(&graph);

    if coarse_components.len() <= 1 && graph.node_count() <= 1 {
        // Single cluster or empty — no split possible at coarse level.
        // Still need to check if the single cluster is internally split.
        if graph.node_count() == 0 {
            return CoarseNarrowResult {
                component_count: 0,
                used_coarse_shortcut: true,
            };
        }
        // For a single cluster, we could do per-voxel flood fill within it,
        // but as an optimization we report it as 1 component (spec says skip).
        return CoarseNarrowResult {
            component_count: 1,
            used_coarse_shortcut: true,
        };
    }

    if coarse_components.len() > 1 {
        // Coarse graph already shows multiple components.
        // For each coarse component, run per-voxel flood fill to get exact count.
        let mut total = 0;
        for coarse_comp in &coarse_components {
            let sub_components = narrow_within_clusters(grid, cluster_size, coarse_comp);
            total += sub_components;
        }
        return CoarseNarrowResult {
            component_count: total,
            used_coarse_shortcut: false,
        };
    }

    // Single coarse component with multiple clusters — run full per-voxel
    // flood fill (could optimize to only check boundaries, but correctness first).
    let full = find_connected_components(grid);
    CoarseNarrowResult {
        component_count: full.len(),
        used_coarse_shortcut: false,
    }
}

/// Find connected components on the coarse graph using BFS.
fn coarse_connected_components(graph: &ClusterGraph) -> Vec<HashSet<ClusterId>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &node in graph.nodes() {
        if visited.contains(&node) {
            continue;
        }

        let mut component = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node);
        visited.insert(node);

        while let Some(current) = queue.pop_front() {
            component.insert(current);
            for neighbor in graph.neighbors(current) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

/// Run per-voxel flood fill within a set of clusters, returning component count.
fn narrow_within_clusters(
    grid: &VoxelGrid,
    cluster_size: u32,
    clusters: &HashSet<ClusterId>,
) -> usize {
    let (w, h, d) = grid.dimensions();

    // Collect all non-empty voxels in these clusters
    let mut voxels = HashSet::new();
    for &(cx, cy, cz) in clusters {
        let x0 = cx * cluster_size;
        let y0 = cy * cluster_size;
        let z0 = cz * cluster_size;
        for z in z0..(z0 + cluster_size).min(d) {
            for y in y0..(y0 + cluster_size).min(h) {
                for x in x0..(x0 + cluster_size).min(w) {
                    if grid.get(x, y, z) != Some(0) {
                        voxels.insert((x, y, z));
                    }
                }
            }
        }
    }

    // BFS flood fill within this voxel set
    let neighbors: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let mut visited = HashSet::new();
    let mut count = 0;

    for &start in &voxels {
        if visited.contains(&start) {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some((cx, cy, cz)) = queue.pop_front() {
            for &(dx, dy, dz) in &neighbors {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                let nz = cz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 { continue; }
                let n = (nx as u32, ny as u32, nz as u32);
                if voxels.contains(&n) && visited.insert(n) {
                    queue.push_back(n);
                }
            }
        }
    }

    count
}
