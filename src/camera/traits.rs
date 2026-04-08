use super::camera::Camera;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputState {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub scroll_delta: f32,
}

pub trait CameraController {
    fn update(&mut self, input: &InputState, dt: f32);
    fn camera(&self) -> &Camera;
}
