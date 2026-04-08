/// Parameters for the shallow water pipe model.
pub struct SweParams {
    pub gravity: f64,
    pub pipe_area: f64,
}

impl Default for SweParams {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            pipe_area: 1.0,
        }
    }
}

struct Source {
    x: usize,
    z: usize,
    rate: f64,
}

struct Sink {
    x: usize,
    z: usize,
}

/// 2D shallow water grid using the pipe model (flux between neighbors).
pub struct SweGrid {
    width: usize,
    height: usize,
    /// Water height per cell.
    water: Vec<f64>,
    /// Terrain height per cell.
    terrain: Vec<f64>,
    /// Horizontal fluxes: right (+x) and down (+z) per cell.
    flux_right: Vec<f64>,
    flux_down: Vec<f64>,
    params: SweParams,
    sources: Vec<Source>,
    sinks: Vec<Sink>,
}

impl SweGrid {
    pub fn new(width: usize, height: usize, params: SweParams) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            water: vec![0.0; n],
            terrain: vec![0.0; n],
            flux_right: vec![0.0; n],
            flux_down: vec![0.0; n],
            params,
            sources: Vec::new(),
            sinks: Vec::new(),
        }
    }

    fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    pub fn set_water_height(&mut self, x: usize, z: usize, h: f64) {
        let i = self.idx(x, z);
        self.water[i] = h;
    }

    pub fn water_height(&self, x: usize, z: usize) -> f64 {
        self.water[self.idx(x, z)]
    }

    pub fn set_terrain_height(&mut self, x: usize, z: usize, h: f64) {
        let i = self.idx(x, z);
        self.terrain[i] = h;
    }

    pub fn terrain_height(&self, x: usize, z: usize) -> f64 {
        self.terrain[self.idx(x, z)]
    }

    pub fn add_source(&mut self, x: usize, z: usize, rate: f64) {
        self.sources.push(Source { x, z, rate });
    }

    pub fn add_sink(&mut self, x: usize, z: usize) {
        self.sinks.push(Sink { x, z });
    }

    pub fn total_water_volume(&self) -> f64 {
        self.water.iter().sum()
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Approximate flow speed at a cell based on flux magnitude.
    pub fn flow_speed(&self, x: usize, z: usize) -> f64 {
        let i = self.idx(x, z);
        let fr = self.flux_right.get(i).copied().unwrap_or(0.0);
        let fd = self.flux_down.get(i).copied().unwrap_or(0.0);
        (fr * fr + fd * fd).sqrt()
    }

    /// Advance one timestep using the pipe model.
    pub fn step(&mut self, dt: f64) {
        let g = self.params.gravity;
        let a = self.params.pipe_area;
        let w = self.width;
        let h = self.height;

        // Update fluxes based on height differences
        for z in 0..h {
            for x in 0..w {
                let i = self.idx(x, z);
                let surface_i = self.terrain[i] + self.water[i];

                // Right neighbor
                if x + 1 < w {
                    let j = self.idx(x + 1, z);
                    let surface_j = self.terrain[j] + self.water[j];
                    let dh = surface_i - surface_j;
                    self.flux_right[i] += g * a * dh * dt;
                    self.flux_right[i] = self.flux_right[i].max(0.0).min(self.water[i] / (dt * 4.0 + 1e-10));
                    // Also allow negative flux (flow from right to left)
                    if dh < 0.0 {
                        self.flux_right[i] = -((-self.flux_right[i]).max(0.0).min(self.water[j] / (dt * 4.0 + 1e-10)));
                    }
                }

                // Down neighbor
                if z + 1 < h {
                    let j = self.idx(x, z + 1);
                    let surface_j = self.terrain[j] + self.water[j];
                    let dh = surface_i - surface_j;
                    self.flux_down[i] += g * a * dh * dt;
                    self.flux_down[i] = self.flux_down[i].max(0.0).min(self.water[i] / (dt * 4.0 + 1e-10));
                    if dh < 0.0 {
                        self.flux_down[i] = -((-self.flux_down[i]).max(0.0).min(self.water[j] / (dt * 4.0 + 1e-10)));
                    }
                }
            }
        }

        // Apply flux — simple approach: recompute from scratch each step
        // Actually, let me use a cleaner pipe model implementation:
        self.flux_right.fill(0.0);
        self.flux_down.fill(0.0);

        // Compute fluxes
        for z in 0..h {
            for x in 0..w {
                let i = self.idx(x, z);
                let surface_i = self.terrain[i] + self.water[i];

                if x + 1 < w {
                    let j = self.idx(x + 1, z);
                    let surface_j = self.terrain[j] + self.water[j];
                    let dh = surface_i - surface_j;
                    // Flux proportional to height difference
                    let flux = g * a * dh * dt;
                    // Limit flux to available water
                    let max_flux = if flux > 0.0 {
                        self.water[i] * 0.25 // max 25% of water can flow per direction
                    } else {
                        self.water[j] * 0.25
                    };
                    self.flux_right[i] = flux.clamp(-max_flux, max_flux);
                }

                if z + 1 < h {
                    let j = self.idx(x, z + 1);
                    let surface_j = self.terrain[j] + self.water[j];
                    let dh = surface_i - surface_j;
                    let flux = g * a * dh * dt;
                    let max_flux = if flux > 0.0 {
                        self.water[i] * 0.25
                    } else {
                        self.water[j] * 0.25
                    };
                    self.flux_down[i] = flux.clamp(-max_flux, max_flux);
                }
            }
        }

        // Apply fluxes to update water heights
        let mut delta = vec![0.0f64; w * h];

        for z in 0..h {
            for x in 0..w {
                let i = self.idx(x, z);

                // Right flux
                if x + 1 < w {
                    let j = self.idx(x + 1, z);
                    delta[i] -= self.flux_right[i];
                    delta[j] += self.flux_right[i];
                }

                // Down flux
                if z + 1 < h {
                    let j = self.idx(x, z + 1);
                    delta[i] -= self.flux_down[i];
                    delta[j] += self.flux_down[i];
                }
            }
        }

        for i in 0..(w * h) {
            self.water[i] = (self.water[i] + delta[i]).max(0.0);
        }

        // Apply sources and sinks
        for src in &self.sources {
            let i = self.idx(src.x, src.z);
            self.water[i] += src.rate * dt;
        }
        for sink in &self.sinks {
            let i = self.idx(sink.x, sink.z);
            self.water[i] = (self.water[i] - self.water[i] * 0.5 * dt).max(0.0);
        }
    }
}
