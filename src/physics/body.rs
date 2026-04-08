use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsBody {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub is_static: bool,
    pub restitution: f32,
    pub friction: f32,
}

impl Default for PhysicsBody {
    fn default() -> Self {
        Self {
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            is_static: true,
            restitution: 0.3,
            friction: 0.5,
        }
    }
}

impl PhysicsBody {
    pub fn dynamic(mass: f32) -> Self {
        Self {
            mass,
            is_static: false,
            ..Default::default()
        }
    }
}
