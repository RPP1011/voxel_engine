use glam::Vec3;
use super::camera::Camera;
use super::traits::{CameraController, InputState};

pub struct OrbitCamera {
    center: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    mouse_sensitivity: f32,
    zoom_sensitivity: f32,
    camera: Camera,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let mut cam = Self {
            center: Vec3::ZERO,
            distance: 80.0,
            yaw: 0.5,
            pitch: 0.3,
            mouse_sensitivity: 0.005,
            zoom_sensitivity: 1.0,
            camera: Camera::default(),
        };
        cam.rebuild_camera();
        cam
    }
}

impl OrbitCamera {
    pub fn new(center: Vec3, distance: f32) -> Self {
        let mut cam = Self { center, distance, ..Default::default() };
        cam.rebuild_camera();
        cam
    }

    pub fn set_center(&mut self, center: Vec3) {
        self.center = center;
        self.rebuild_camera();
    }

    pub fn center(&self) -> Vec3 {
        self.center
    }

    fn rebuild_camera(&mut self) {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.camera.position = self.center + Vec3::new(
            self.distance * cp * sy,
            self.distance * sp,
            self.distance * cp * cy,
        );
        self.camera.target = self.center;
    }

    // Legacy compatibility for renderer
    pub fn eye_position(&self) -> [f32; 3] {
        self.camera.position.into()
    }

    pub fn view_matrix_array(&self) -> [f32; 16] {
        self.camera.view_matrix().to_cols_array()
    }

    pub fn projection_matrix_array(&self, aspect: f32) -> [f32; 16] {
        self.camera.projection_matrix(aspect).to_cols_array()
    }

    pub fn mvp_matrix(&self, aspect: f32) -> [f32; 16] {
        self.camera.view_projection_matrix(aspect).to_cols_array()
    }

    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-1.4, 1.4);
        self.rebuild_camera();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(1.0, 3000.0);
        self.rebuild_camera();
    }
}

impl CameraController for OrbitCamera {
    fn update(&mut self, input: &InputState, _dt: f32) {
        self.yaw += input.mouse_dx * self.mouse_sensitivity;
        self.pitch = (self.pitch + input.mouse_dy * self.mouse_sensitivity).clamp(-1.4, 1.4);
        self.distance = (self.distance + input.scroll_delta * self.zoom_sensitivity).clamp(1.0, 3000.0);
        self.rebuild_camera();
    }

    fn camera(&self) -> &Camera {
        &self.camera
    }
}
