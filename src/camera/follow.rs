use glam::Vec3;
use super::camera::Camera;
use super::traits::{CameraController, InputState};

pub struct FollowCamera {
    offset: Vec3,
    target_pos: Vec3,
    smoothing: f32,
    camera: Camera,
}

impl FollowCamera {
    pub fn new(offset: Vec3) -> Self {
        Self {
            offset,
            target_pos: Vec3::ZERO,
            smoothing: 5.0,
            camera: Camera { position: offset, target: Vec3::ZERO, ..Default::default() },
        }
    }

    pub fn set_target(&mut self, pos: Vec3) {
        self.target_pos = pos;
    }

    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.smoothing = smoothing;
    }
}

impl CameraController for FollowCamera {
    fn update(&mut self, _input: &InputState, dt: f32) {
        let t = (self.smoothing * dt).min(1.0);
        self.camera.target = self.camera.target.lerp(self.target_pos, t);
        self.camera.position = self.camera.position.lerp(self.target_pos + self.offset, t);
    }

    fn camera(&self) -> &Camera {
        &self.camera
    }
}
