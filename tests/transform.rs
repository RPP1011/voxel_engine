use glam::{Quat, Vec3};
use voxel_engine::scene::Transform;

#[test]
fn default_transform_is_identity() {
    let t = Transform::default();
    assert_eq!(t.position, Vec3::ZERO);
    assert_eq!(t.rotation, Quat::IDENTITY);
    assert_eq!(t.scale, Vec3::ONE);
}

#[test]
fn transform_lerp_halfway() {
    let a = Transform {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    let b = Transform {
        position: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
    };
    let mid = a.lerp(&b, 0.5);
    assert!((mid.position.x - 5.0).abs() < 1e-5);
    assert!((mid.scale.x - 1.5).abs() < 1e-5);
}

#[test]
fn transform_lerp_at_zero_returns_self() {
    let a = Transform {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    let b = Transform {
        position: Vec3::new(10.0, 20.0, 30.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(5.0),
    };
    let result = a.lerp(&b, 0.0);
    assert!((result.position - a.position).length() < 1e-5);
}

#[test]
fn transform_lerp_at_one_returns_other() {
    let a = Transform::default();
    let b = Transform {
        position: Vec3::new(10.0, 20.0, 30.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(5.0),
    };
    let result = a.lerp(&b, 1.0);
    assert!((result.position - b.position).length() < 1e-5);
}
