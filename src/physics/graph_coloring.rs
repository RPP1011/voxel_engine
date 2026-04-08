use std::collections::{HashMap, HashSet};

/// Result of greedy graph coloring.
pub struct ColoringResult {
    colors: HashMap<u32, u32>,
    num_colors: u32,
}

impl ColoringResult {
    pub fn num_colors(&self) -> u32 {
        self.num_colors
    }

    pub fn assignments(&self) -> &HashMap<u32, u32> {
        &self.colors
    }

    pub fn color_of(&self, node: u32) -> Option<u32> {
        self.colors.get(&node).copied()
    }
}

/// Greedy graph coloring: no two adjacent nodes share a color.
/// Uses smallest-available-color heuristic.
pub fn greedy_color(adj: &HashMap<u32, Vec<u32>>) -> ColoringResult {
    let mut colors: HashMap<u32, u32> = HashMap::new();
    let mut max_color = 0u32;

    // Sort nodes by degree (descending) for better coloring
    let mut nodes: Vec<u32> = adj.keys().copied().collect();
    nodes.sort_by(|a, b| {
        let da = adj.get(a).map_or(0, |v| v.len());
        let db = adj.get(b).map_or(0, |v| v.len());
        db.cmp(&da)
    });

    for &node in &nodes {
        let neighbors = adj.get(&node).cloned().unwrap_or_default();
        let used: HashSet<u32> = neighbors
            .iter()
            .filter_map(|nb| colors.get(nb).copied())
            .collect();

        // Find smallest available color
        let mut color = 0u32;
        while used.contains(&color) {
            color += 1;
        }

        colors.insert(node, color);
        if color + 1 > max_color {
            max_color = color + 1;
        }
    }

    ColoringResult {
        colors,
        num_colors: max_color,
    }
}
