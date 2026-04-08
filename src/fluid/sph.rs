use std::collections::HashMap;

pub struct SphParticle {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub density: f64,
}

pub struct SphSolver {
    pub particles: Vec<SphParticle>,
    pub rest_density: f64,
    pub kernel_radius: f64,
    pub gravity: f64,
    pub stiffness: f64,
    pub viscosity: f64,
    pub dt: f64,
    pub bounds: ([f64; 3], [f64; 3]),
}

impl SphSolver {
    pub fn new(rest_density: f64, kernel_radius: f64) -> Self {
        Self {
            particles: Vec::new(),
            rest_density,
            kernel_radius,
            gravity: -9.81,
            stiffness: 50.0,
            viscosity: 0.01,
            dt: 0.005,
            bounds: ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        }
    }

    pub fn add_particle(&mut self, x: f64, y: f64, z: f64) {
        self.particles.push(SphParticle {
            pos: [x, y, z],
            vel: [0.0; 3],
            density: self.rest_density,
        });
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn step(&mut self) {
        let n = self.particles.len();
        let h = self.kernel_radius;

        // Build spatial hash for neighbor search
        let neighbors = self.find_neighbors();

        // Compute densities
        for i in 0..n {
            let mut density = 0.0;
            for &j in neighbors.get(&i).unwrap_or(&vec![]) {
                let r = dist(&self.particles[i].pos, &self.particles[j].pos);
                density += poly6(r, h);
            }
            // Self contribution
            density += poly6(0.0, h);
            self.particles[i].density = density;
        }

        // Compute pressure forces + viscosity + gravity → update velocities
        let mut forces = vec![[0.0f64; 3]; n];

        for i in 0..n {
            let pi = pressure(self.particles[i].density, self.rest_density, self.stiffness);

            for &j in neighbors.get(&i).unwrap_or(&vec![]) {
                if i == j { continue; }
                let r = dist(&self.particles[i].pos, &self.particles[j].pos);
                if r < 1e-10 { continue; }

                let pj = pressure(self.particles[j].density, self.rest_density, self.stiffness);

                // Pressure force (symmetric)
                let grad = spiky_gradient(r, h);
                let pressure_term = (pi + pj) / (2.0 * self.particles[j].density.max(1e-6));

                let dir = [
                    (self.particles[i].pos[0] - self.particles[j].pos[0]) / r,
                    (self.particles[i].pos[1] - self.particles[j].pos[1]) / r,
                    (self.particles[i].pos[2] - self.particles[j].pos[2]) / r,
                ];

                for d in 0..3 {
                    forces[i][d] -= pressure_term * grad * dir[d];
                }

                // Viscosity
                let visc = viscosity_laplacian(r, h);
                for d in 0..3 {
                    forces[i][d] += self.viscosity * (self.particles[j].vel[d] - self.particles[i].vel[d])
                        * visc / self.particles[j].density.max(1e-6);
                }
            }

            // Gravity
            forces[i][1] += self.gravity;
        }

        // Integration
        for i in 0..n {
            for d in 0..3 {
                self.particles[i].vel[d] += forces[i][d] * self.dt;
                self.particles[i].pos[d] += self.particles[i].vel[d] * self.dt;
            }

            // Boundary clamping
            for d in 0..3 {
                if self.particles[i].pos[d] < self.bounds.0[d] {
                    self.particles[i].pos[d] = self.bounds.0[d];
                    self.particles[i].vel[d] *= -0.3;
                }
                if self.particles[i].pos[d] > self.bounds.1[d] {
                    self.particles[i].pos[d] = self.bounds.1[d];
                    self.particles[i].vel[d] *= -0.3;
                }
            }
        }
    }

    fn find_neighbors(&self) -> HashMap<usize, Vec<usize>> {
        let h = self.kernel_radius;
        let cell_size = h;
        let mut grid: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();

        for (i, p) in self.particles.iter().enumerate() {
            let cx = (p.pos[0] / cell_size).floor() as i32;
            let cy = (p.pos[1] / cell_size).floor() as i32;
            let cz = (p.pos[2] / cell_size).floor() as i32;
            grid.entry((cx,cy,cz)).or_default().push(i);
        }

        let mut neighbors = HashMap::new();
        for (i, p) in self.particles.iter().enumerate() {
            let cx = (p.pos[0] / cell_size).floor() as i32;
            let cy = (p.pos[1] / cell_size).floor() as i32;
            let cz = (p.pos[2] / cell_size).floor() as i32;
            let mut nb = Vec::new();
            for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                if let Some(cell) = grid.get(&(cx+dx, cy+dy, cz+dz)) {
                    for &j in cell {
                        if dist(&p.pos, &self.particles[j].pos) < h {
                            nb.push(j);
                        }
                    }
                }
            }}}
            neighbors.insert(i, nb);
        }
        neighbors
    }
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    (dx*dx + dy*dy + dz*dz).sqrt()
}

fn poly6(r: f64, h: f64) -> f64 {
    if r >= h { return 0.0; }
    let c = 315.0 / (64.0 * std::f64::consts::PI * h.powi(9));
    c * (h*h - r*r).powi(3)
}

fn spiky_gradient(r: f64, h: f64) -> f64 {
    if r >= h || r < 1e-10 { return 0.0; }
    let c = -45.0 / (std::f64::consts::PI * h.powi(6));
    c * (h - r).powi(2)
}

fn viscosity_laplacian(r: f64, h: f64) -> f64 {
    if r >= h { return 0.0; }
    let c = 45.0 / (std::f64::consts::PI * h.powi(6));
    c * (h - r)
}

fn pressure(density: f64, rest: f64, k: f64) -> f64 {
    k * (density - rest)
}
