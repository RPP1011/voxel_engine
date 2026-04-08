use std::collections::{HashMap, HashSet, VecDeque};
use super::cluster_graph::{ClusterGraph, ClusterId};
use super::grid::VoxelGrid;
use super::material::MaterialType;

/// Node in the structural graph with mass and strength info.
struct ClusterNode {
    id: ClusterId,
    mass: f64,
    strength: f64,
    is_ground_anchored: bool,
    voxel_count: u32,
}

/// Result of structural stress analysis.
pub struct StressResult {
    stress: HashMap<ClusterId, f64>,
    fractured: Vec<ClusterId>,
    max_cascade_depth: u32,
}

impl StressResult {
    pub fn cluster_stress(&self, id: &ClusterId) -> f64 {
        self.stress.get(id).copied().unwrap_or(0.0)
    }

    pub fn fractured_clusters(&self) -> &[ClusterId] {
        &self.fractured
    }

    pub fn max_stress_ratio(&self) -> f64 {
        self.stress.values().copied().fold(0.0, f64::max)
    }

    pub fn max_cascade_depth(&self) -> u32 {
        self.max_cascade_depth
    }
}

/// Structural analysis over the coarse cluster graph.
pub struct StructuralGraph {
    graph: ClusterGraph,
    nodes: HashMap<ClusterId, ClusterNode>,
}

impl StructuralGraph {
    /// Build a structural graph from a voxel grid.
    /// All voxels are assumed to be of the given material type.
    pub fn build(grid: &VoxelGrid, cluster_size: u32, material: MaterialType) -> Self {
        let graph = ClusterGraph::build(grid, cluster_size);
        let (w, h, d) = grid.dimensions();
        let density = material.density_kg_m3() as f64;
        let strength = material.strength_pa() as f64;
        let voxel_volume = 0.1 * 0.1 * 0.1; // 10cm voxels

        let mut nodes = HashMap::new();
        for &id in graph.nodes() {
            let (cx, cy, cz) = id;
            let x0 = cx * cluster_size;
            let y0 = cy * cluster_size;
            let z0 = cz * cluster_size;

            let mut voxel_count = 0u32;
            for z in z0..(z0 + cluster_size).min(d) {
                for y in y0..(y0 + cluster_size).min(h) {
                    for x in x0..(x0 + cluster_size).min(w) {
                        if grid.get(x, y, z) != Some(0) {
                            voxel_count += 1;
                        }
                    }
                }
            }

            let mass = voxel_count as f64 * voxel_volume * density;
            let cluster_strength = strength * voxel_count as f64 * 0.01; // Simplified

            nodes.insert(id, ClusterNode {
                id,
                mass,
                strength: cluster_strength,
                is_ground_anchored: cy == 0,
                voxel_count,
            });
        }

        Self { graph, nodes }
    }

    /// BFS from ground-anchored nodes upward, accumulating child mass as stress.
    pub fn analyze_stress(&self) -> StressResult {
        let mut stress: HashMap<ClusterId, f64> = HashMap::new();
        let mut fractured: Vec<ClusterId> = Vec::new();

        // Build adjacency for BFS — only upward edges (from lower y to higher y)
        // First, find ground-anchored nodes
        let anchors: Vec<ClusterId> = self.nodes.values()
            .filter(|n| n.is_ground_anchored)
            .map(|n| n.id)
            .collect();

        if anchors.is_empty() {
            // Everything is unsupported
            return StressResult { stress, fractured, max_cascade_depth: 0 };
        }

        // BFS from top down to accumulate mass (stress flows downward)
        // First, find topological order by BFS from ground up
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        for &a in &anchors {
            if visited.insert(a) {
                queue.push_back(a);
            }
        }

        while let Some(current) = queue.pop_front() {
            order.push(current);
            for neighbor in self.graph.neighbors(current) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        // Accumulate stress bottom-up: each node's stress = sum of mass above it
        // Process in reverse order (top-to-bottom), propagating mass downward
        let mut supported_mass: HashMap<ClusterId, f64> = HashMap::new();

        // Initialize each node with its own mass
        for &id in &order {
            if let Some(node) = self.nodes.get(&id) {
                supported_mass.insert(id, node.mass);
            }
        }

        // Process top-down: for each node, distribute its total supported mass
        // to lower neighbors
        for &id in order.iter().rev() {
            let total_mass = supported_mass.get(&id).copied().unwrap_or(0.0);
            let neighbors = self.graph.neighbors(id);

            // Find lower neighbors (lower y)
            let lower_neighbors: Vec<ClusterId> = neighbors
                .iter()
                .filter(|n| n.1 < id.1)
                .copied()
                .collect();

            if !lower_neighbors.is_empty() {
                let share = total_mass / lower_neighbors.len() as f64;
                for ln in &lower_neighbors {
                    *supported_mass.entry(*ln).or_insert(0.0) += share;
                }
            }
        }

        // Compute stress ratio (supported mass / strength) for each node
        let mut max_cascade = 0u32;
        let mut cascade_depth = 0u32;

        for &id in &order {
            let total = supported_mass.get(&id).copied().unwrap_or(0.0);
            let node = self.nodes.get(&id).unwrap();
            // Stress = accumulated mass above (excluding own mass)
            let stress_val = (total - node.mass).max(0.0);
            stress.insert(id, stress_val);

            // Check fracture
            if node.strength > 0.0 && stress_val > node.strength {
                fractured.push(id);
                cascade_depth += 1;
                if cascade_depth > 3 {
                    break; // Limit cascade per frame
                }
            }
        }

        max_cascade = cascade_depth.min(3);

        StressResult {
            stress,
            fractured,
            max_cascade_depth: max_cascade,
        }
    }
}
