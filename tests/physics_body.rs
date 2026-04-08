use glam::Vec3;
use voxel_engine::physics::PhysicsBody;

#[test]
fn default_physics_body_is_static_at_rest() {
    let body = PhysicsBody::default();
    assert!(body.is_static);
    assert_eq!(body.linear_velocity, Vec3::ZERO);
    assert_eq!(body.angular_velocity, Vec3::ZERO);
    assert_eq!(body.mass, 1.0);
}

#[test]
fn dynamic_body_constructor() {
    let body = PhysicsBody::dynamic(5.0);
    assert!(!body.is_static);
    assert_eq!(body.mass, 5.0);
    assert_eq!(body.restitution, 0.3);
    assert_eq!(body.friction, 0.5);
}
