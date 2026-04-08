use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use crate::voxel::grid::VoxelGrid;

/// A navmesh tile covering a 16x16 XZ area.
pub struct NavTile {
    pub tile_x: i32,
    pub tile_z: i32,
    pub walkable: Vec<bool>, // 16x16 grid of walkability
    pub dirty: bool,
}

/// Tile-based navmesh with incremental rebuild and A* pathfinding.
pub struct NavMesh {
    tiles: HashMap<(i32, i32), NavTile>,
    tile_size: usize,
    agent_height: u32,
}

impl NavMesh {
    pub fn new(tile_size: usize, agent_height: u32) -> Self {
        Self { tiles: HashMap::new(), tile_size, agent_height }
    }

    /// Build or rebuild a navmesh tile from the voxel grid.
    pub fn build_tile(&mut self, grid: &VoxelGrid, tile_x: i32, tile_z: i32) {
        let ts = self.tile_size;
        let (gw, gh, gd) = grid.dimensions();
        let mut walkable = vec![false; ts * ts];

        let base_x = (tile_x * ts as i32).max(0) as u32;
        let base_z = (tile_z * ts as i32).max(0) as u32;

        for lz in 0..ts {
            for lx in 0..ts {
                let gx = base_x + lx as u32;
                let gz = base_z + lz as u32;
                if gx >= gw || gz >= gd { continue; }

                // Find highest solid voxel (floor)
                let mut floor_y = None;
                for y in (0..gh).rev() {
                    if grid.get(gx, y, gz) != Some(0) {
                        floor_y = Some(y);
                        break;
                    }
                }

                if let Some(fy) = floor_y {
                    // Check headroom
                    let mut has_headroom = true;
                    for dy in 1..=self.agent_height {
                        let check_y = fy + dy;
                        if check_y < gh && grid.get(gx, check_y, gz) != Some(0) {
                            has_headroom = false;
                            break;
                        }
                    }
                    walkable[lz * ts + lx] = has_headroom;
                }
            }
        }

        self.tiles.insert((tile_x, tile_z), NavTile {
            tile_x, tile_z, walkable, dirty: false,
        });
    }

    /// Mark a tile as dirty (needs rebuild after destruction).
    pub fn mark_dirty(&mut self, tile_x: i32, tile_z: i32) {
        if let Some(tile) = self.tiles.get_mut(&(tile_x, tile_z)) {
            tile.dirty = true;
        }
    }

    pub fn is_walkable(&self, world_x: u32, world_z: u32) -> bool {
        let ts = self.tile_size;
        let tx = (world_x as usize / ts) as i32;
        let tz = (world_z as usize / ts) as i32;
        let lx = world_x as usize % ts;
        let lz = world_z as usize % ts;
        self.tiles.get(&(tx, tz))
            .map_or(false, |t| t.walkable[lz * ts + lx])
    }

    /// A* pathfinding on the navmesh grid.
    /// Returns a list of (world_x, world_z) waypoints.
    pub fn find_path(&self, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<(u32, u32)>> {
        if !self.is_walkable(start.0, start.1) || !self.is_walkable(goal.0, goal.1) {
            return None;
        }

        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<(u32,u32), f64> = HashMap::new();
        let mut came_from: HashMap<(u32,u32), (u32,u32)> = HashMap::new();
        let mut closed = HashSet::new();

        g_score.insert(start, 0.0);
        let h = heuristic(start, goal);
        open.push(Reverse((OrderedF64(h), start)));

        let neighbors = |pos: (u32,u32)| -> Vec<(u32,u32)> {
            let mut nb = Vec::new();
            for (dx, dz) in [(-1i32,0),(1,0),(0,-1),(0,1)] {
                let nx = pos.0 as i32 + dx;
                let nz = pos.1 as i32 + dz;
                if nx >= 0 && nz >= 0 {
                    let n = (nx as u32, nz as u32);
                    if self.is_walkable(n.0, n.1) {
                        nb.push(n);
                    }
                }
            }
            nb
        };

        while let Some(Reverse((_, current))) = open.pop() {
            if current == goal {
                let mut path = vec![goal];
                let mut c = goal;
                while let Some(&prev) = came_from.get(&c) {
                    path.push(prev);
                    c = prev;
                    if c == start { break; }
                }
                path.reverse();
                return Some(path);
            }

            if !closed.insert(current) { continue; }

            let current_g = g_score[&current];
            for nb in neighbors(current) {
                if closed.contains(&nb) { continue; }
                let tentative_g = current_g + 1.0;
                if tentative_g < *g_score.get(&nb).unwrap_or(&f64::MAX) {
                    g_score.insert(nb, tentative_g);
                    came_from.insert(nb, current);
                    let f = tentative_g + heuristic(nb, goal);
                    open.push(Reverse((OrderedF64(f), nb)));
                }
            }
        }
        None
    }
}

fn heuristic(a: (u32,u32), b: (u32,u32)) -> f64 {
    let dx = (a.0 as f64 - b.0 as f64).abs();
    let dz = (a.1 as f64 - b.1 as f64).abs();
    dx + dz // Manhattan distance
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
