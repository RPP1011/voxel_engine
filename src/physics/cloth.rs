/// A cloth mesh with position-based dynamics.
pub struct ClothMesh {
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    pinned: Vec<bool>,
    width: usize,
    height: usize,
    rest_spacing: f64,
    stretch_constraints: Vec<(usize, usize, f64)>, // (a, b, rest_len)
    bend_constraints: Vec<(usize, usize, f64)>,
    lra_constraints: Vec<(usize, usize, f64)>, // (vertex, pin, max_dist)
}

impl ClothMesh {
    pub fn new_grid(width: usize, height: usize, spacing: f64) -> Self {
        let n = width * height;
        let mut positions = Vec::with_capacity(n);
        for y in 0..height {
            for x in 0..width {
                positions.push([x as f64 * spacing, 0.0, y as f64 * spacing]);
            }
        }

        let mut stretch = Vec::new();
        // Horizontal
        for y in 0..height {
            for x in 0..(width - 1) {
                let a = y * width + x;
                let b = a + 1;
                stretch.push((a, b, spacing));
            }
        }
        // Vertical
        for y in 0..(height - 1) {
            for x in 0..width {
                let a = y * width + x;
                let b = a + width;
                stretch.push((a, b, spacing));
            }
        }

        let mut bend = Vec::new();
        // Horizontal skip-one
        if width >= 3 {
            for y in 0..height {
                for x in 0..(width - 2) {
                    let a = y * width + x;
                    let b = a + 2;
                    bend.push((a, b, spacing * 2.0));
                }
            }
        }
        // Vertical skip-one
        if height >= 3 {
            for y in 0..(height - 2) {
                for x in 0..width {
                    let a = y * width + x;
                    let b = a + 2 * width;
                    bend.push((a, b, spacing * 2.0));
                }
            }
        }

        Self {
            positions,
            velocities: vec![[0.0; 3]; n],
            pinned: vec![false; n],
            width,
            height,
            rest_spacing: spacing,
            stretch_constraints: stretch,
            bend_constraints: bend,
            lra_constraints: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn stretch_constraint_count(&self) -> usize {
        self.stretch_constraints.len()
    }

    pub fn bend_constraint_count(&self) -> usize {
        self.bend_constraints.len()
    }

    pub fn pin_row(&mut self, row: usize) {
        for x in 0..self.width {
            self.pinned[row * self.width + x] = true;
        }
    }

    pub fn pin_vertex(&mut self, idx: usize) {
        self.pinned[idx] = true;
    }

    pub fn is_pinned(&self, x: usize, y: usize) -> bool {
        self.pinned[y * self.width + x]
    }

    pub fn position(&self, idx: usize) -> [f64; 3] {
        self.positions[idx]
    }

    pub fn set_position(&mut self, idx: usize, pos: [f64; 3]) {
        self.positions[idx] = pos;
    }

    /// Pre-compute LRA constraints: geodesic distance from each free vertex to nearest pin.
    pub fn compute_lra(&mut self) {
        use std::collections::VecDeque;

        let n = self.positions.len();
        let mut dist_to_pin = vec![f64::MAX; n];
        let mut nearest_pin = vec![usize::MAX; n];
        let mut queue = VecDeque::new();

        // BFS from all pinned vertices
        for i in 0..n {
            if self.pinned[i] {
                dist_to_pin[i] = 0.0;
                nearest_pin[i] = i;
                queue.push_back(i);
            }
        }

        while let Some(current) = queue.pop_front() {
            let cx = current % self.width;
            let cy = current / self.width;

            let neighbors = [
                if cx > 0 { Some(current - 1) } else { None },
                if cx + 1 < self.width { Some(current + 1) } else { None },
                if cy > 0 { Some(current - self.width) } else { None },
                if cy + 1 < self.height { Some(current + self.width) } else { None },
            ];

            for nb in neighbors.iter().flatten() {
                let new_dist = dist_to_pin[current] + self.rest_spacing;
                if new_dist < dist_to_pin[*nb] {
                    dist_to_pin[*nb] = new_dist;
                    nearest_pin[*nb] = nearest_pin[current];
                    queue.push_back(*nb);
                }
            }
        }

        self.lra_constraints.clear();
        for i in 0..n {
            if !self.pinned[i] && nearest_pin[i] != usize::MAX {
                self.lra_constraints.push((i, nearest_pin[i], dist_to_pin[i]));
            }
        }
    }
}

pub struct ClothSolver {
    gravity: f64,
    use_lra: bool,
}

impl ClothSolver {
    pub fn new(gravity: f64) -> Self {
        Self { gravity, use_lra: false }
    }

    pub fn enable_lra(&mut self, enable: bool) {
        self.use_lra = enable;
    }

    pub fn step(&self, cloth: &mut ClothMesh, dt: f64, iterations: usize) {
        let n = cloth.positions.len();

        // Apply gravity and predict
        let mut old_pos = cloth.positions.clone();
        for i in 0..n {
            if cloth.pinned[i] { continue; }
            cloth.velocities[i][1] += self.gravity * dt;
            for d in 0..3 {
                cloth.positions[i][d] += cloth.velocities[i][d] * dt;
            }
        }

        // Solve constraints
        self.solve_constraints(cloth, iterations);

        // Derive velocity
        for i in 0..n {
            if cloth.pinned[i] { continue; }
            for d in 0..3 {
                cloth.velocities[i][d] = (cloth.positions[i][d] - old_pos[i][d]) / dt;
                cloth.velocities[i][d] *= 0.99; // damping
            }
        }
    }

    pub fn solve_constraints(&self, cloth: &mut ClothMesh, iterations: usize) {
        for _ in 0..iterations {
            // Stretch constraints
            for &(a, b, rest) in &cloth.stretch_constraints.clone() {
                solve_distance(cloth, a, b, rest);
            }

            // Bend constraints (weaker)
            for &(a, b, rest) in &cloth.bend_constraints.clone() {
                solve_distance_weighted(cloth, a, b, rest, 0.3);
            }

            // LRA constraints
            if self.use_lra {
                for &(vert, pin, max_dist) in &cloth.lra_constraints.clone() {
                    let p = cloth.positions[vert];
                    let q = cloth.positions[pin];
                    let dx = p[0] - q[0];
                    let dy = p[1] - q[1];
                    let dz = p[2] - q[2];
                    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                    if dist > max_dist && dist > 1e-10 {
                        // Pull vertex back to max_dist
                        let factor = max_dist / dist;
                        if !cloth.pinned[vert] {
                            cloth.positions[vert][0] = q[0] + dx * factor;
                            cloth.positions[vert][1] = q[1] + dy * factor;
                            cloth.positions[vert][2] = q[2] + dz * factor;
                        }
                    }
                }
            }
        }
    }
}

fn solve_distance(cloth: &mut ClothMesh, a: usize, b: usize, rest: f64) {
    solve_distance_weighted(cloth, a, b, rest, 1.0);
}

fn solve_distance_weighted(cloth: &mut ClothMesh, a: usize, b: usize, rest: f64, stiffness: f64) {
    let pa = cloth.positions[a];
    let pb = cloth.positions[b];
    let dx = pb[0] - pa[0];
    let dy = pb[1] - pa[1];
    let dz = pb[2] - pa[2];
    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
    if dist < 1e-10 { return; }

    let diff = (dist - rest) / dist * stiffness;
    let wa = if cloth.pinned[a] { 0.0 } else { 1.0 };
    let wb = if cloth.pinned[b] { 0.0 } else { 1.0 };
    let total = wa + wb;
    if total < 1e-10 { return; }

    let ca = wa / total * diff;
    let cb = wb / total * diff;

    if !cloth.pinned[a] {
        cloth.positions[a][0] += dx * ca;
        cloth.positions[a][1] += dy * ca;
        cloth.positions[a][2] += dz * ca;
    }
    if !cloth.pinned[b] {
        cloth.positions[b][0] -= dx * cb;
        cloth.positions[b][1] -= dy * cb;
        cloth.positions[b][2] -= dz * cb;
    }
}
