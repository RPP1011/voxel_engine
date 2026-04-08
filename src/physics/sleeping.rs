use std::collections::HashMap;

pub struct BodyState {
    pub vx: f64,
    pub vy: f64,
    pub angular_vel: f64,
    pub mass: f64,
}

impl BodyState {
    fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * (self.vx * self.vx + self.vy * self.vy)
            + 0.5 * self.mass * self.angular_vel * self.angular_vel
    }
}

struct SleepEntry {
    low_ke_frames: u32,
    sleeping: bool,
    position: [f64; 3],
}

pub struct SleepSystem {
    entries: HashMap<u64, SleepEntry>,
    threshold_frames: u32,
    ke_threshold: f64,
}

impl SleepSystem {
    pub fn new(threshold_frames: u32, ke_threshold: f64) -> Self {
        Self {
            entries: HashMap::new(),
            threshold_frames,
            ke_threshold,
        }
    }

    pub fn update(&mut self, id: u64, state: &BodyState) {
        let entry = self.entries.entry(id).or_insert(SleepEntry {
            low_ke_frames: 0,
            sleeping: false,
            position: [0.0; 3],
        });

        if entry.sleeping {
            return; // Don't update sleeping bodies
        }

        if state.kinetic_energy() < self.ke_threshold {
            entry.low_ke_frames += 1;
            if entry.low_ke_frames >= self.threshold_frames {
                entry.sleeping = true;
            }
        } else {
            entry.low_ke_frames = 0;
        }
    }

    pub fn is_sleeping(&self, id: u64) -> bool {
        self.entries.get(&id).map_or(false, |e| e.sleeping)
    }

    pub fn wake(&mut self, id: u64) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.sleeping = false;
            entry.low_ke_frames = 0;
        }
    }

    pub fn set_position(&mut self, id: u64, pos: [f64; 3]) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.position = pos;
        }
    }

    pub fn wake_in_radius(&mut self, center: [f64; 3], radius: f64) {
        let r2 = radius * radius;
        for entry in self.entries.values_mut() {
            let dx = entry.position[0] - center[0];
            let dy = entry.position[1] - center[1];
            let dz = entry.position[2] - center[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                entry.sleeping = false;
                entry.low_ke_frames = 0;
            }
        }
    }

    pub fn sleeping_count(&self) -> usize {
        self.entries.values().filter(|e| e.sleeping).count()
    }

    pub fn awake_count(&self) -> usize {
        self.entries.values().filter(|e| !e.sleeping).count()
    }
}
