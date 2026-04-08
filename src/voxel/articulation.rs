use std::collections::{HashMap, HashSet};

/// Information about an articulation point.
#[derive(Debug)]
pub struct ArticulationInfo {
    pub node: u32,
    /// How many additional components would be created by removing this node.
    pub disconnect_count: u32,
    /// Total mass that would be disconnected (placeholder — set by caller).
    pub disconnected_mass: f64,
}

/// Tarjan's algorithm for finding articulation points in an undirected graph.
/// Input: adjacency list. Output: list of articulation points with metadata.
pub fn find_articulation_points(adj: &HashMap<u32, Vec<u32>>) -> Vec<ArticulationInfo> {
    let nodes: Vec<u32> = adj.keys().copied().collect();
    if nodes.is_empty() {
        return Vec::new();
    }

    let mut disc: HashMap<u32, u32> = HashMap::new();
    let mut low: HashMap<u32, u32> = HashMap::new();
    let mut parent: HashMap<u32, Option<u32>> = HashMap::new();
    let mut ap_set: HashSet<u32> = HashSet::new();
    let mut timer = 0u32;

    // Run DFS from each unvisited node (handles disconnected graphs)
    for &start in &nodes {
        if disc.contains_key(&start) {
            continue;
        }
        tarjan_dfs(
            start, adj, &mut disc, &mut low, &mut parent, &mut ap_set, &mut timer,
        );
    }

    // For each articulation point, compute disconnect_count
    ap_set
        .into_iter()
        .map(|node| {
            let dc = compute_disconnect_count(node, adj);
            ArticulationInfo {
                node,
                disconnect_count: dc,
                disconnected_mass: 0.0,
            }
        })
        .collect()
}

fn tarjan_dfs(
    u: u32,
    adj: &HashMap<u32, Vec<u32>>,
    disc: &mut HashMap<u32, u32>,
    low: &mut HashMap<u32, u32>,
    parent: &mut HashMap<u32, Option<u32>>,
    ap_set: &mut HashSet<u32>,
    timer: &mut u32,
) {
    disc.insert(u, *timer);
    low.insert(u, *timer);
    *timer += 1;
    parent.entry(u).or_insert(None);

    let mut child_count = 0u32;

    let neighbors = adj.get(&u).cloned().unwrap_or_default();
    for v in neighbors {
        if !disc.contains_key(&v) {
            child_count += 1;
            parent.insert(v, Some(u));
            tarjan_dfs(v, adj, disc, low, parent, ap_set, timer);

            // Update low value
            let low_v = *low.get(&v).unwrap();
            let low_u = low.get(&u).copied().unwrap();
            low.insert(u, low_u.min(low_v));

            // u is an articulation point if:
            // 1. u is root of DFS tree and has ≥2 children
            if parent.get(&u) == Some(&None) && child_count > 1 {
                ap_set.insert(u);
            }
            // 2. u is not root and low[v] >= disc[u]
            if parent.get(&u) != Some(&None) && low_v >= disc[&u] {
                ap_set.insert(u);
            }
        } else if Some(v) != parent.get(&u).copied().flatten() {
            // Back edge — update low
            let disc_v = *disc.get(&v).unwrap();
            let low_u = low.get(&u).copied().unwrap();
            low.insert(u, low_u.min(disc_v));
        }
    }
}

/// Count how many components would exist if `node` were removed.
fn compute_disconnect_count(node: u32, adj: &HashMap<u32, Vec<u32>>) -> u32 {
    // BFS/DFS on the graph minus `node`
    let mut visited = HashSet::new();
    visited.insert(node);
    let mut components = 0u32;

    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            if let Some(neighbors) = adj.get(&n) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        stack.push(nb);
                    }
                }
            }
        }
    }

    // disconnect_count = components - 1 (compared to removing node)
    // or just report how many fragments there are
    components.saturating_sub(1)
}
