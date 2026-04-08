use std::collections::{HashMap, HashSet};
use super::grid::VoxelGrid;

/// Identifier for a cluster: grid coordinates of the cluster (cx, cy, cz).
pub type ClusterId = (u32, u32, u32);

/// Coarse connectivity graph over 3D clusters of the voxel grid.
pub struct ClusterGraph {
    /// Set of non-empty cluster IDs.
    nodes: HashSet<ClusterId>,
    /// Undirected edges between face-adjacent non-empty clusters
    /// that share at least one face-adjacent pair of non-empty voxels.
    edges: HashSet<(ClusterId, ClusterId)>,
    /// Cluster size in voxels per axis.
    cluster_size: u32,
}

impl ClusterGraph {
    /// Build a coarse connectivity graph by dividing the grid into cubic clusters.
    pub fn build(grid: &VoxelGrid, cluster_size: u32) -> Self {
        let (w, h, d) = grid.dimensions();
        let cx_count = (w + cluster_size - 1) / cluster_size;
        let cy_count = (h + cluster_size - 1) / cluster_size;
        let cz_count = (d + cluster_size - 1) / cluster_size;

        // Identify non-empty clusters
        let mut nodes = HashSet::new();
        for cz in 0..cz_count {
            for cy in 0..cy_count {
                for cx in 0..cx_count {
                    if cluster_has_voxels(grid, cluster_size, cx, cy, cz) {
                        nodes.insert((cx, cy, cz));
                    }
                }
            }
        }

        // Find edges: two adjacent non-empty clusters share an edge if they have
        // face-adjacent non-empty voxels across the cluster boundary.
        let mut edges = HashSet::new();
        let neighbor_dirs: [(i32, i32, i32); 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];

        for &(cx, cy, cz) in &nodes {
            for &(dx, dy, dz) in &neighbor_dirs {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                let nz = cz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 {
                    continue;
                }
                let neighbor = (nx as u32, ny as u32, nz as u32);
                if !nodes.contains(&neighbor) {
                    continue;
                }

                // Check if there's at least one pair of face-adjacent non-empty voxels
                // across the boundary.
                if clusters_share_boundary_voxels(
                    grid,
                    cluster_size,
                    (cx, cy, cz),
                    neighbor,
                    (dx, dy, dz),
                ) {
                    let a = (cx, cy, cz).min(neighbor);
                    let b = (cx, cy, cz).max(neighbor);
                    edges.insert((a, b));
                }
            }
        }

        Self {
            nodes,
            edges,
            cluster_size,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn nodes(&self) -> &HashSet<ClusterId> {
        &self.nodes
    }

    pub fn edges(&self) -> &HashSet<(ClusterId, ClusterId)> {
        &self.edges
    }

    pub fn cluster_size(&self) -> u32 {
        self.cluster_size
    }

    /// Get neighbors of a cluster node.
    pub fn neighbors(&self, id: ClusterId) -> Vec<ClusterId> {
        self.edges
            .iter()
            .filter_map(|&(a, b)| {
                if a == id {
                    Some(b)
                } else if b == id {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: ClusterId) {
        self.nodes.remove(&id);
        self.edges.retain(|&(a, b)| a != id && b != id);
    }

    /// Recompute a single cluster node's presence and its edges with neighbors.
    pub fn recompute_cluster(
        &mut self,
        grid: &VoxelGrid,
        cluster_id: ClusterId,
    ) {
        let (cx, cy, cz) = cluster_id;

        // Remove old edges involving this cluster
        self.edges.retain(|&(a, b)| a != cluster_id && b != cluster_id);

        if cluster_has_voxels(grid, self.cluster_size, cx, cy, cz) {
            self.nodes.insert(cluster_id);

            // Recheck edges with all 6 neighbors
            let dirs: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ];
            for &(dx, dy, dz) in &dirs {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                let nz = cz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 { continue; }
                let neighbor = (nx as u32, ny as u32, nz as u32);
                if !self.nodes.contains(&neighbor) { continue; }

                if clusters_share_boundary_voxels(
                    grid, self.cluster_size, cluster_id, neighbor, (dx, dy, dz),
                ) {
                    let a = cluster_id.min(neighbor);
                    let b = cluster_id.max(neighbor);
                    self.edges.insert((a, b));
                }
            }
        } else {
            self.nodes.remove(&cluster_id);
        }
    }
}

fn cluster_has_voxels(grid: &VoxelGrid, cs: u32, cx: u32, cy: u32, cz: u32) -> bool {
    let (w, h, d) = grid.dimensions();
    let x0 = cx * cs;
    let y0 = cy * cs;
    let z0 = cz * cs;
    for z in z0..(z0 + cs).min(d) {
        for y in y0..(y0 + cs).min(h) {
            for x in x0..(x0 + cs).min(w) {
                if grid.get(x, y, z) != Some(0) {
                    return true;
                }
            }
        }
    }
    false
}

fn clusters_share_boundary_voxels(
    grid: &VoxelGrid,
    cs: u32,
    _from: ClusterId,
    _to: ClusterId,
    dir: (i32, i32, i32),
) -> bool {
    // The boundary face is defined by the direction.
    // For dir=(1,0,0): boundary is at x = from.cx*cs + cs - 1 (from side)
    //                   and x = to.cx*cs (to side)
    let (w, h, d) = grid.dimensions();
    let (from_cx, from_cy, from_cz) = _from;
    let (to_cx, to_cy, to_cz) = _to;

    let from_x0 = from_cx * cs;
    let from_y0 = from_cy * cs;
    let from_z0 = from_cz * cs;
    let to_x0 = to_cx * cs;
    let to_y0 = to_cy * cs;
    let to_z0 = to_cz * cs;

    match dir {
        (1, 0, 0) | (-1, 0, 0) => {
            // Boundary along X: check the face between the two clusters
            let (face_from_x, face_to_x) = if dir.0 > 0 {
                ((from_x0 + cs - 1).min(w - 1), to_x0.min(w - 1))
            } else {
                (from_x0, (to_x0 + cs - 1).min(w - 1))
            };
            for z in 0..cs {
                for y in 0..cs {
                    let fy = from_y0 + y;
                    let fz = from_z0 + z;
                    let ty = to_y0 + y;
                    let tz = to_z0 + z;
                    if fy < h && fz < d && ty < h && tz < d {
                        if grid.get(face_from_x, fy, fz) != Some(0)
                            && grid.get(face_to_x, ty, tz) != Some(0)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        (0, 1, 0) | (0, -1, 0) => {
            let (face_from_y, face_to_y) = if dir.1 > 0 {
                ((from_y0 + cs - 1).min(h - 1), to_y0.min(h - 1))
            } else {
                (from_y0, (to_y0 + cs - 1).min(h - 1))
            };
            for z in 0..cs {
                for x in 0..cs {
                    let fx = from_x0 + x;
                    let fz = from_z0 + z;
                    let tx = to_x0 + x;
                    let tz = to_z0 + z;
                    if fx < w && fz < d && tx < w && tz < d {
                        if grid.get(fx, face_from_y, fz) != Some(0)
                            && grid.get(tx, face_to_y, tz) != Some(0)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        (0, 0, 1) | (0, 0, -1) => {
            let (face_from_z, face_to_z) = if dir.2 > 0 {
                ((from_z0 + cs - 1).min(d - 1), to_z0.min(d - 1))
            } else {
                (from_z0, (to_z0 + cs - 1).min(d - 1))
            };
            for y in 0..cs {
                for x in 0..cs {
                    let fx = from_x0 + x;
                    let fy = from_y0 + y;
                    let tx = to_x0 + x;
                    let ty = to_y0 + y;
                    if fx < w && fy < h && tx < w && ty < h {
                        if grid.get(fx, fy, face_from_z) != Some(0)
                            && grid.get(tx, ty, face_to_z) != Some(0)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        _ => {}
    }
    false
}
