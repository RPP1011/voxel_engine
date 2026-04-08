use std::collections::HashSet;

pub struct CullableBody {
    pub id: u64,
    pub y: f64,
    pub voxel_count: u32,
}

pub struct BodyCuller {
    min_y: f64,
    min_voxels: u32,
    max_bodies: usize,
}

impl BodyCuller {
    pub fn new(min_y: f64, min_voxels: u32, max_bodies: usize) -> Self {
        Self { min_y, min_voxels, max_bodies }
    }

    pub fn find_bodies_to_cull(&mut self, bodies: &[CullableBody]) -> HashSet<u64> {
        let mut to_remove = HashSet::new();

        // Rule 1: below world bounds
        for b in bodies {
            if b.y < self.min_y {
                to_remove.insert(b.id);
            }
        }

        // Rule 2: below minimum voxel count
        for b in bodies {
            if b.voxel_count < self.min_voxels {
                to_remove.insert(b.id);
            }
        }

        // Rule 3: hard cap — remove smallest bodies first
        let surviving: Vec<&CullableBody> = bodies
            .iter()
            .filter(|b| !to_remove.contains(&b.id))
            .collect();

        if surviving.len() > self.max_bodies {
            let excess = surviving.len() - self.max_bodies;
            let mut sorted: Vec<&CullableBody> = surviving;
            sorted.sort_by_key(|b| b.voxel_count);
            for b in sorted.iter().take(excess) {
                to_remove.insert(b.id);
            }
        }

        to_remove
    }
}
