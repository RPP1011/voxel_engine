use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::Reverse;

use crate::voxel::articulation::{find_articulation_points, ArticulationInfo};
use crate::voxel::cluster_graph::{ClusterGraph, ClusterId};
use crate::voxel::grid::VoxelGrid;
use crate::voxel::material::MaterialType;

pub struct RemovalResult {
    pub fragment_count: usize,
    pub fragment_masses: Vec<f64>,
}

pub struct StructuralQueryApi {
    graph: ClusterGraph,
    cluster_strength: HashMap<ClusterId, f64>,
    cluster_mass: HashMap<ClusterId, f64>,
}

impl StructuralQueryApi {
    pub fn new(grid: &VoxelGrid, cluster_size: u32, material: MaterialType) -> Self {
        Self::new_with_materials(grid, cluster_size, &|_| material)
    }

    pub fn new_with_materials(
        grid: &VoxelGrid,
        cluster_size: u32,
        material_fn: &dyn Fn(u8) -> MaterialType,
    ) -> Self {
        let graph = ClusterGraph::build(grid, cluster_size);
        let (w, h, d) = grid.dimensions();
        let voxel_volume = 0.1 * 0.1 * 0.1;

        let mut cluster_strength = HashMap::new();
        let mut cluster_mass = HashMap::new();

        for &id in graph.nodes() {
            let (cx, cy, cz) = id;
            let x0 = cx * cluster_size;
            let y0 = cy * cluster_size;
            let z0 = cz * cluster_size;

            let mut total_mass = 0.0f64;
            let mut min_strength = f64::MAX;
            let mut count = 0u32;

            for z in z0..(z0 + cluster_size).min(d) {
                for y in y0..(y0 + cluster_size).min(h) {
                    for x in x0..(x0 + cluster_size).min(w) {
                        let v = grid.get(x, y, z).unwrap_or(0);
                        if v != 0 {
                            let mat = material_fn(v);
                            total_mass += mat.density_kg_m3() as f64 * voxel_volume;
                            let s = mat.strength_pa() as f64;
                            if s < min_strength {
                                min_strength = s;
                            }
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                cluster_strength.insert(id, min_strength * count as f64 * 0.01);
                cluster_mass.insert(id, total_mass);
            }
        }

        Self { graph, cluster_strength, cluster_mass }
    }

    /// Query articulation points in the coarse connectivity graph.
    pub fn query_articulation_points(&self) -> Vec<ArticulationInfo> {
        // Convert ClusterGraph to adjacency HashMap<u32, Vec<u32>>
        let (adj, _id_map, _rev_map) = self.graph_to_adj();
        find_articulation_points(&adj)
    }

    /// Simulate removing a set of clusters and predict resulting fragments.
    pub fn simulate_removal(&self, clusters_to_remove: &[ClusterId]) -> RemovalResult {
        let remove_set: HashSet<ClusterId> = clusters_to_remove.iter().copied().collect();

        // BFS on remaining nodes to count components
        let remaining: HashSet<ClusterId> = self.graph.nodes()
            .iter()
            .filter(|n| !remove_set.contains(n))
            .copied()
            .collect();

        let mut visited = HashSet::new();
        let mut fragments = Vec::new();

        for &node in &remaining {
            if visited.contains(&node) { continue; }

            let mut component_mass = 0.0;
            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                component_mass += self.cluster_mass.get(&current).copied().unwrap_or(0.0);
                for neighbor in self.graph.neighbors(current) {
                    if remaining.contains(&neighbor) && visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            fragments.push(component_mass);
        }

        RemovalResult {
            fragment_count: fragments.len(),
            fragment_masses: fragments,
        }
    }

    /// Find the weakest path between two clusters (minimizing total strength along path).
    pub fn find_weakest_path(&self, from: ClusterId, to: ClusterId) -> Vec<ClusterId> {
        // Dijkstra with strength as edge weight
        let mut dist: HashMap<ClusterId, f64> = HashMap::new();
        let mut prev: HashMap<ClusterId, ClusterId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(from, 0.0);
        heap.push(Reverse((OrderedF64(0.0), from)));

        while let Some(Reverse((OrderedF64(d), node))) = heap.pop() {
            if node == to {
                // Reconstruct path
                let mut path = vec![to];
                let mut current = to;
                while let Some(&p) = prev.get(&current) {
                    path.push(p);
                    current = p;
                    if current == from { break; }
                }
                path.reverse();
                return path;
            }

            if d > dist.get(&node).copied().unwrap_or(f64::MAX) {
                continue;
            }

            for neighbor in self.graph.neighbors(node) {
                let edge_cost = self.cluster_strength.get(&neighbor).copied().unwrap_or(f64::MAX);
                let new_dist = d + edge_cost;
                if new_dist < dist.get(&neighbor).copied().unwrap_or(f64::MAX) {
                    dist.insert(neighbor, new_dist);
                    prev.insert(neighbor, node);
                    heap.push(Reverse((OrderedF64(new_dist), neighbor)));
                }
            }
        }

        Vec::new() // No path found
    }

    fn graph_to_adj(&self) -> (HashMap<u32, Vec<u32>>, HashMap<ClusterId, u32>, HashMap<u32, ClusterId>) {
        let mut id_map: HashMap<ClusterId, u32> = HashMap::new();
        let mut rev_map: HashMap<u32, ClusterId> = HashMap::new();
        let mut next_id = 0u32;

        for &node in self.graph.nodes() {
            id_map.insert(node, next_id);
            rev_map.insert(next_id, node);
            next_id += 1;
        }

        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &node in self.graph.nodes() {
            let nid = id_map[&node];
            adj.entry(nid).or_default();
            for neighbor in self.graph.neighbors(node) {
                if let Some(&nb_id) = id_map.get(&neighbor) {
                    adj.entry(nid).or_default().push(nb_id);
                }
            }
        }

        (adj, id_map, rev_map)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);
impl Eq for OrderedF64 {}
impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}
