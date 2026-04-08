/// 2D rigid body for the AVBD (Augmented Vertex Block Descent) prototype solver.
#[derive(Clone, Debug)]
pub struct Body2D {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub vx: f64,
    pub vy: f64,
    pub angular_vel: f64,
    pub half_w: f64,
    pub half_h: f64,
    pub mass: f64,
    pub inertia: f64,
    pub is_static: bool,
}

impl Body2D {
    fn inv_mass(&self) -> f64 {
        if self.is_static { 0.0 } else { 1.0 / self.mass }
    }
    fn inv_inertia(&self) -> f64 {
        if self.is_static { 0.0 } else { 1.0 / self.inertia }
    }
}

struct Contact {
    body_a: usize,
    body_b: Option<usize>, // None = ground
    normal_x: f64,
    normal_y: f64,
    penetration: f64,
    contact_x: f64,
    contact_y: f64,
}

/// Minimal 2D AVBD solver: position-based dynamics with substeps.
pub struct AvbdSolver2D {
    bodies: Vec<Body2D>,
    pub gravity: f64,
    pub ground_y: f64,
}

impl AvbdSolver2D {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: -9.81,
            ground_y: 0.0,
        }
    }

    pub fn add_body(&mut self, body: Body2D) {
        self.bodies.push(body);
    }

    pub fn bodies(&self) -> &[Body2D] {
        &self.bodies
    }

    pub fn total_kinetic_energy(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| !b.is_static)
            .map(|b| {
                0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy)
                    + 0.5 * b.inertia * b.angular_vel * b.angular_vel
            })
            .sum()
    }

    pub fn step(&mut self, dt: f64, iterations: usize) {
        // Use substeps for stability
        let substeps = 10;
        let sub_dt = dt / substeps as f64;

        for _ in 0..substeps {
            self.substep(sub_dt, iterations);
        }
    }

    fn substep(&mut self, dt: f64, iterations: usize) {
        let n = self.bodies.len();

        // Store old positions
        let mut old_x = vec![0.0; n];
        let mut old_y = vec![0.0; n];
        let mut old_angle = vec![0.0; n];

        for i in 0..n {
            old_x[i] = self.bodies[i].x;
            old_y[i] = self.bodies[i].y;
            old_angle[i] = self.bodies[i].angle;
        }

        // Predict positions (semi-implicit Euler)
        for b in self.bodies.iter_mut() {
            if b.is_static {
                continue;
            }
            b.vy += self.gravity * dt;
            b.x += b.vx * dt;
            b.y += b.vy * dt;
            b.angle += b.angular_vel * dt;
        }

        // Iterative position projection
        for _iter in 0..iterations {
            let contacts = self.detect_contacts();

            for contact in &contacts {
                if contact.penetration <= 0.0 {
                    continue;
                }

                self.solve_contact(contact);
            }
        }

        // Derive velocities from position change
        for i in 0..n {
            if self.bodies[i].is_static {
                continue;
            }
            self.bodies[i].vx = (self.bodies[i].x - old_x[i]) / dt;
            self.bodies[i].vy = (self.bodies[i].y - old_y[i]) / dt;
            self.bodies[i].angular_vel = (self.bodies[i].angle - old_angle[i]) / dt;

            // Velocity damping for stability
            self.bodies[i].vx *= 0.998;
            self.bodies[i].vy *= 0.998;
            self.bodies[i].angular_vel *= 0.98;
        }
    }

    fn solve_contact(&mut self, contact: &Contact) {
        let nx = contact.normal_x;
        let ny = contact.normal_y;

        // Compute generalized inverse mass
        let ba = &self.bodies[contact.body_a];
        let ra_x = contact.contact_x - ba.x;
        let ra_y = contact.contact_y - ba.y;
        let ra_cross_n = ra_x * ny - ra_y * nx;
        let mut w = ba.inv_mass() + ba.inv_inertia() * ra_cross_n * ra_cross_n;

        let mut rb_cross_n = 0.0;
        if let Some(bi) = contact.body_b {
            let bb = &self.bodies[bi];
            let rb_x = contact.contact_x - bb.x;
            let rb_y = contact.contact_y - bb.y;
            rb_cross_n = rb_x * ny - rb_y * nx;
            w += bb.inv_mass() + bb.inv_inertia() * rb_cross_n * rb_cross_n;
        }

        if w < 1e-12 {
            return;
        }

        let delta = contact.penetration / w;

        if contact.body_b.is_none() {
            // Ground contact: normal points away from ground (up), push body along normal
            let ba = &mut self.bodies[contact.body_a];
            if !ba.is_static {
                ba.x += nx * delta * ba.inv_mass();
                ba.y += ny * delta * ba.inv_mass();
                ba.angle += ra_cross_n * delta * ba.inv_inertia();
            }
        } else {
            // Body-body: normal points A→B, push A back, push B forward
            {
                let ba = &mut self.bodies[contact.body_a];
                if !ba.is_static {
                    ba.x -= nx * delta * ba.inv_mass();
                    ba.y -= ny * delta * ba.inv_mass();
                    ba.angle -= ra_cross_n * delta * ba.inv_inertia();
                }
            }
            if let Some(bi) = contact.body_b {
                let bb = &mut self.bodies[bi];
                if !bb.is_static {
                    bb.x += nx * delta * bb.inv_mass();
                    bb.y += ny * delta * bb.inv_mass();
                    bb.angle += rb_cross_n * delta * bb.inv_inertia();
                }
            }
        }
    }

    fn detect_contacts(&self) -> Vec<Contact> {
        let mut contacts = Vec::new();

        // Ground contacts: test corners of each box
        for (i, body) in self.bodies.iter().enumerate() {
            if body.is_static {
                continue;
            }
            let corners = box_corners(body);
            for (cx, cy) in corners {
                let pen = self.ground_y - cy;
                if pen > 0.0 {
                    contacts.push(Contact {
                        body_a: i,
                        body_b: None,
                        normal_x: 0.0,
                        normal_y: 1.0,
                        penetration: pen,
                        contact_x: cx,
                        contact_y: cy,
                    });
                }
            }
        }

        // Body-body: SAT-based contact detection
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let pair_contacts = box_box_contacts(&self.bodies[i], &self.bodies[j], i, j);
                contacts.extend(pair_contacts);
            }
        }

        contacts
    }
}

fn box_corners(b: &Body2D) -> [(f64, f64); 4] {
    let c = b.angle.cos();
    let s = b.angle.sin();
    let hw = b.half_w;
    let hh = b.half_h;
    let offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
    offsets.map(|(lx, ly)| (b.x + lx * c - ly * s, b.y + lx * s + ly * c))
}

/// Test if a point is inside an OBB.
fn point_in_box(px: f64, py: f64, b: &Body2D) -> bool {
    let c = b.angle.cos();
    let s = b.angle.sin();
    let dx = px - b.x;
    let dy = py - b.y;
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;
    lx.abs() < b.half_w + 1e-6 && ly.abs() < b.half_h + 1e-6
}

/// Compute contacts between two boxes using SAT + clipping approach.
fn box_box_contacts(a: &Body2D, b: &Body2D, ai: usize, bi: usize) -> Vec<Contact> {
    // SAT to find minimum penetration axis
    let a_corners = box_corners(a);
    let b_corners = box_corners(b);

    let axes = [
        axis_from_angle(a.angle),
        axis_from_angle(a.angle + std::f64::consts::FRAC_PI_2),
        axis_from_angle(b.angle),
        axis_from_angle(b.angle + std::f64::consts::FRAC_PI_2),
    ];

    let mut min_pen = f64::MAX;
    let mut best_normal = (0.0, 0.0);

    for &(ax, ay) in &axes {
        let (a_min, a_max) = project_corners(&a_corners, ax, ay);
        let (b_min, b_max) = project_corners(&b_corners, ax, ay);

        let overlap = a_max.min(b_max) - a_min.max(b_min);
        if overlap <= 0.0 {
            return Vec::new(); // Separated
        }

        if overlap < min_pen {
            min_pen = overlap;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            if dx * ax + dy * ay < 0.0 {
                best_normal = (-ax, -ay);
            } else {
                best_normal = (ax, ay);
            }
        }
    }

    // Generate contact points: vertices of each box that are inside the other
    let mut contacts = Vec::new();
    let (nx, ny) = best_normal;

    for (cx, cy) in a_corners {
        if point_in_box(cx, cy, b) {
            contacts.push(Contact {
                body_a: ai,
                body_b: Some(bi),
                normal_x: nx,
                normal_y: ny,
                penetration: min_pen,
                contact_x: cx,
                contact_y: cy,
            });
        }
    }

    for (cx, cy) in b_corners {
        if point_in_box(cx, cy, a) {
            contacts.push(Contact {
                body_a: ai,
                body_b: Some(bi),
                normal_x: nx,
                normal_y: ny,
                penetration: min_pen,
                contact_x: cx,
                contact_y: cy,
            });
        }
    }

    // If no vertex-in-box contacts found but SAT confirms overlap, use centroid contact
    if contacts.is_empty() && min_pen > 0.0 {
        contacts.push(Contact {
            body_a: ai,
            body_b: Some(bi),
            normal_x: nx,
            normal_y: ny,
            penetration: min_pen,
            contact_x: (a.x + b.x) * 0.5,
            contact_y: (a.y + b.y) * 0.5,
        });
    }

    // Avoid duplicate overcorrection — use only one representative contact
    // by averaging if there are many vertex contacts
    if contacts.len() > 2 {
        let avg_cx: f64 = contacts.iter().map(|c| c.contact_x).sum::<f64>() / contacts.len() as f64;
        let avg_cy: f64 = contacts.iter().map(|c| c.contact_y).sum::<f64>() / contacts.len() as f64;
        let pen = contacts[0].penetration;
        contacts.truncate(1);
        contacts[0].contact_x = avg_cx;
        contacts[0].contact_y = avg_cy;
        contacts[0].penetration = pen;
    }

    contacts
}

fn axis_from_angle(angle: f64) -> (f64, f64) {
    (angle.cos(), angle.sin())
}

fn project_corners(corners: &[(f64, f64); 4], ax: f64, ay: f64) -> (f64, f64) {
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    for &(cx, cy) in corners {
        let proj = cx * ax + cy * ay;
        min = min.min(proj);
        max = max.max(proj);
    }
    (min, max)
}
