/// Orbit camera that revolves around a center point.
pub struct OrbitCamera {
    pub center: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            distance: 80.0,
            yaw: 0.5,
            pitch: 0.3,
        }
    }
}

impl OrbitCamera {
    /// Compute the eye (camera) position in world space.
    pub fn eye_position(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        [
            self.center[0] + self.distance * cp * sy,
            self.center[1] + self.distance * sp,
            self.center[2] + self.distance * cp * cy,
        ]
    }

    /// Produce a column-major 4x4 view matrix (look_at).
    pub fn view_matrix(&self) -> [f32; 16] {
        let eye = self.eye_position();
        look_at(eye, self.center, [0.0, 1.0, 0.0])
    }

    /// Produce a column-major 4x4 perspective projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> [f32; 16] {
        perspective(std::f32::consts::FRAC_PI_4, aspect, 0.1, 5000.0)
    }

    /// Produce projection * view as a column-major 4x4 matrix.
    pub fn mvp_matrix(&self, aspect: f32) -> [f32; 16] {
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        mat4_mul(&proj, &view)
    }

    /// Rotate the camera by delta yaw and pitch (radians).
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-1.4, 1.4);
    }

    /// Zoom in/out by adjusting distance.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(1.0, 3000.0);
    }
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize(sub(center, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        s[0],  u[0],  -f[0], 0.0,
        s[1],  u[1],  -f[1], 0.0,
        s[2],  u[2],  -f[2], 0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy / 2.0).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0,  0.0,               0.0,
        0.0,       -f,    0.0,               0.0,
        0.0,        0.0,  far * nf,          -1.0,
        0.0,        0.0,  near * far * nf,    0.0,
    ]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}
