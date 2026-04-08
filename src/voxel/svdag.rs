use std::collections::HashMap;
use super::grid::VoxelGrid;

/// A node in the sparse voxel DAG.
/// Leaf nodes store an 8-bit child mask (2x2x2 occupancy).
/// Internal nodes store indices to 8 children (or 0 for empty subtree).
#[derive(Clone, Hash, Eq, PartialEq)]
enum SvdagNode {
    /// Leaf: 2x2x2 occupancy bitmask.
    Leaf(u8),
    /// Internal: indices into node pool for 8 children.
    Internal([u32; 8]),
}

/// Sparse Voxel Directed Acyclic Graph — compressed octree with subtree deduplication.
pub struct Svdag {
    nodes: Vec<SvdagNode>,
    root: u32,
    grid_size: u32,
}

impl Svdag {
    /// Build an SVDAG from a dense voxel grid. Grid dimensions must be power of 2.
    pub fn from_grid(grid: &VoxelGrid) -> Self {
        let (w, h, d) = grid.dimensions();
        let size = w.max(h).max(d).next_power_of_two();

        let mut nodes = Vec::new();
        let mut dedup: HashMap<SvdagNode, u32> = HashMap::new();

        let root = build_recursive(grid, &mut nodes, &mut dedup, 0, 0, 0, size);

        Self {
            nodes,
            root,
            grid_size: size,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Query whether a voxel is solid at (x, y, z).
    pub fn query(&self, x: u32, y: u32, z: u32) -> bool {
        if x >= self.grid_size || y >= self.grid_size || z >= self.grid_size {
            return false;
        }
        query_recursive(&self.nodes, self.root, x, y, z, self.grid_size)
    }
}

fn build_recursive(
    grid: &VoxelGrid,
    nodes: &mut Vec<SvdagNode>,
    dedup: &mut HashMap<SvdagNode, u32>,
    ox: u32, oy: u32, oz: u32,
    size: u32,
) -> u32 {
    if size == 2 {
        // Leaf: encode 2x2x2 as bitmask
        let mut mask = 0u8;
        for dz in 0..2u32 {
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let bit = dz * 4 + dy * 2 + dx;
                    let v = grid.get(ox + dx, oy + dy, oz + dz).unwrap_or(0);
                    if v != 0 {
                        mask |= 1 << bit;
                    }
                }
            }
        }
        let node = SvdagNode::Leaf(mask);
        return intern(nodes, dedup, node);
    }

    let half = size / 2;
    let mut children = [0u32; 8];

    for i in 0..8u32 {
        let dx = (i & 1) * half;
        let dy = ((i >> 1) & 1) * half;
        let dz = ((i >> 2) & 1) * half;
        children[i as usize] = build_recursive(grid, nodes, dedup, ox + dx, oy + dy, oz + dz, half);
    }

    let node = SvdagNode::Internal(children);
    intern(nodes, dedup, node)
}

fn intern(
    nodes: &mut Vec<SvdagNode>,
    dedup: &mut HashMap<SvdagNode, u32>,
    node: SvdagNode,
) -> u32 {
    if let Some(&idx) = dedup.get(&node) {
        return idx;
    }
    let idx = nodes.len() as u32;
    dedup.insert(node.clone(), idx);
    nodes.push(node);
    idx
}

fn query_recursive(nodes: &[SvdagNode], node_idx: u32, x: u32, y: u32, z: u32, size: u32) -> bool {
    match &nodes[node_idx as usize] {
        SvdagNode::Leaf(mask) => {
            let lx = x & 1;
            let ly = y & 1;
            let lz = z & 1;
            let bit = lz * 4 + ly * 2 + lx;
            (mask >> bit) & 1 != 0
        }
        SvdagNode::Internal(children) => {
            let half = size / 2;
            let cx = if x >= half { 1 } else { 0 };
            let cy = if y >= half { 1 } else { 0 };
            let cz = if z >= half { 1 } else { 0 };
            let child_idx = cx + cy * 2 + cz * 4;
            let local_x = x - cx as u32 * half;
            let local_y = y - cy as u32 * half;
            let local_z = z - cz as u32 * half;
            query_recursive(nodes, children[child_idx], local_x, local_y, local_z, half)
        }
    }
}
