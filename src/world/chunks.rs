/// Tier for a world chunk based on distance to camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkTier {
    /// Full resolution, physics enabled.
    Active,
    /// SVDAG, read-only.
    Streaming,
    /// On disk, not in memory.
    Unloaded,
}

/// Manages world chunks and their tier assignments based on camera position.
pub struct ChunkManager {
    chunk_size: f64,
    active_radius: f64,
    streaming_radius: f64,
    camera_pos: [f64; 3],
}

impl ChunkManager {
    pub fn new(chunk_size: f64, active_radius: f64, streaming_radius: f64) -> Self {
        Self {
            chunk_size,
            active_radius,
            streaming_radius,
            camera_pos: [0.0; 3],
        }
    }

    pub fn chunk_size(&self) -> f64 {
        self.chunk_size
    }

    pub fn update_camera(&mut self, pos: [f64; 3]) {
        self.camera_pos = pos;
    }

    /// Determine the tier for a chunk at grid coordinates (cx, cy, cz).
    pub fn chunk_tier(&self, cx: i32, cy: i32, cz: i32) -> ChunkTier {
        let center_x = cx as f64 * self.chunk_size + self.chunk_size * 0.5;
        let center_y = cy as f64 * self.chunk_size + self.chunk_size * 0.5;
        let center_z = cz as f64 * self.chunk_size + self.chunk_size * 0.5;

        let dx = center_x - self.camera_pos[0];
        let dy = center_y - self.camera_pos[1];
        let dz = center_z - self.camera_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist <= self.active_radius {
            ChunkTier::Active
        } else if dist <= self.streaming_radius {
            ChunkTier::Streaming
        } else {
            ChunkTier::Unloaded
        }
    }
}
