use glam::{Mat4, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 5000.0,
        }
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(self.fov, aspect, self.near, self.far);
        proj.y_axis.y *= -1.0; // Vulkan Y flip
        proj
    }

    pub fn view_projection_matrix(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}
