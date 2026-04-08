use glam::Vec3;
use super::camera::Camera;
use super::traits::{CameraController, InputState};

pub struct FreeCamera {
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    mouse_sensitivity: f32,
    camera: Camera,
}

impl FreeCamera {
    pub fn new(position: Vec3, look_at: Vec3) -> Self {
        let dir = (look_at - position).normalize_or_zero();
        let yaw = dir.x.atan2(-dir.z);
        let pitch = dir.y.asin();
        let mut cam = Self {
            yaw,
            pitch,
            move_speed: 10.0,
            mouse_sensitivity: 0.003,
            camera: Camera { position, target: look_at, ..Default::default() },
        };
        cam.rebuild_camera();
        cam
    }

    pub fn set_move_speed(&mut self, speed: f32) {
        self.move_speed = speed;
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    fn rebuild_camera(&mut self) {
        self.camera.target = self.camera.position + self.forward();
    }
}

impl CameraController for FreeCamera {
    fn update(&mut self, input: &InputState, dt: f32) {
        self.yaw += input.mouse_dx * self.mouse_sensitivity;
        self.pitch = (self.pitch - input.mouse_dy * self.mouse_sensitivity).clamp(-1.4, 1.4);
        let forward = self.forward();
        let right = self.right();
        let speed = self.move_speed * dt;
        self.camera.position += forward * input.move_forward * speed;
        self.camera.position += right * input.move_right * speed;
        self.camera.position += Vec3::Y * input.move_up * speed;
        self.rebuild_camera();
    }

    fn camera(&self) -> &Camera {
        &self.camera
    }
}
