use voxel_engine::physics::shape_matching::{extract_rigid_transform, RigidTransform};

#[test]
fn pure_translation_extracted() {
    // Original points
    let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // Translated by (1, 0, 0)
    let deformed: Vec<[f64; 3]> = rest.iter().map(|p| [p[0] + 1.0, p[1], p[2]]).collect();

    let t = extract_rigid_transform(&rest, &deformed);

    assert!((t.translation[0] - 1.0).abs() < 0.01, "dx should be 1, got {}", t.translation[0]);
    assert!((t.translation[1]).abs() < 0.01);
    assert!((t.translation[2]).abs() < 0.01);

    // Rotation should be near identity
    assert!((t.rotation[0][0] - 1.0).abs() < 0.01, "Should be identity rotation");
}

#[test]
fn rotation_45_deg_around_y() {
    let rest = vec![
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ];

    let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
    let c = angle.cos();
    let s = angle.sin();

    // Rotate around Y: x' = x*cos + z*sin, z' = -x*sin + z*cos
    let deformed: Vec<[f64; 3]> = rest.iter().map(|p| {
        [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
    }).collect();

    let t = extract_rigid_transform(&rest, &deformed);

    // Translation should be near zero (centered)
    assert!((t.translation[0]).abs() < 0.1);
    assert!((t.translation[1]).abs() < 0.1);

    // Rotation matrix should approximate Y-rotation by 45 deg
    assert!((t.rotation[0][0] - c).abs() < 0.1, "R[0][0] should be cos(45°)≈{c}, got {}", t.rotation[0][0]);
}

#[test]
fn noisy_perturbation_recovers_best_fit() {
    let rest = vec![
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ];

    // Translate + small noise
    let mut rng = 42u64;
    let deformed: Vec<[f64; 3]> = rest.iter().map(|p| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng >> 33) as f64 / u32::MAX as f64 * 0.1 - 0.05;
        [p[0] + 2.0 + noise, p[1] + 3.0, p[2] + 1.0]
    }).collect();

    let t = extract_rigid_transform(&rest, &deformed);

    // Should recover approximate translation (2, 3, 1)
    assert!((t.translation[0] - 2.0).abs() < 0.2, "dx≈2, got {}", t.translation[0]);
    assert!((t.translation[1] - 3.0).abs() < 0.2, "dy≈3, got {}", t.translation[1]);
    assert!((t.translation[2] - 1.0).abs() < 0.2, "dz≈1, got {}", t.translation[2]);
}
