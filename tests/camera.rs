use glam::Vec3;
use voxel_engine::camera::{Camera, InputState, CameraController, OrbitCamera, FreeCamera, FollowCamera};

#[test]
fn camera_default_looks_forward() {
    let cam = Camera::default();
    assert_eq!(cam.position, Vec3::ZERO);
    assert_eq!(cam.target, Vec3::new(0.0, 0.0, -1.0));
    assert_eq!(cam.up, Vec3::Y);
    assert!((cam.fov - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
}

#[test]
fn camera_view_matrix_at_origin() {
    let cam = Camera::default();
    let view = cam.view_matrix();
    let det = view.determinant();
    assert!((det - 1.0).abs() < 1e-3, "det={det}");
}

#[test]
fn camera_projection_matrix_valid() {
    let cam = Camera::default();
    let proj = cam.projection_matrix(16.0 / 9.0);
    assert!(proj.determinant().abs() > 1e-6);
}

#[test]
fn input_state_default_is_idle() {
    let input = InputState::default();
    assert_eq!(input.move_forward, 0.0);
    assert_eq!(input.move_right, 0.0);
    assert_eq!(input.move_up, 0.0);
    assert_eq!(input.mouse_dx, 0.0);
    assert_eq!(input.mouse_dy, 0.0);
    assert_eq!(input.scroll_delta, 0.0);
}

#[test]
fn orbit_camera_default_position() {
    let orbit = OrbitCamera::default();
    let cam = orbit.camera();
    let dist = cam.position.length();
    assert!(dist > 1.0, "dist={dist}");
    assert!((cam.target - Vec3::ZERO).length() < 1e-5);
}

#[test]
fn orbit_camera_zoom_changes_distance() {
    let mut orbit = OrbitCamera::new(Vec3::ZERO, 50.0);
    let before = orbit.camera().position.length();
    let input = InputState { scroll_delta: -10.0, ..Default::default() };
    orbit.update(&input, 1.0 / 60.0);
    let after = orbit.camera().position.length();
    assert!(after < before, "before={before}, after={after}");
}

#[test]
fn orbit_camera_rotate_changes_position() {
    let mut orbit = OrbitCamera::new(Vec3::ZERO, 50.0);
    let before = orbit.camera().position;
    let input = InputState { mouse_dx: 100.0, ..Default::default() };
    orbit.update(&input, 1.0 / 60.0);
    let after = orbit.camera().position;
    assert!((before - after).length() > 0.1);
    let dist_before = before.length();
    let dist_after = after.length();
    assert!((dist_before - dist_after).abs() < 0.1);
}

#[test]
fn free_camera_moves_forward() {
    let mut free = FreeCamera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
    let input = InputState { move_forward: 1.0, ..Default::default() };
    free.update(&input, 1.0);
    let cam = free.camera();
    assert!(cam.position.z < 0.0, "z={}", cam.position.z);
}

#[test]
fn free_camera_strafes_right() {
    let mut free = FreeCamera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
    let input = InputState { move_right: 1.0, ..Default::default() };
    free.update(&input, 1.0);
    let cam = free.camera();
    assert!(cam.position.x > 0.0, "x={}", cam.position.x);
}

#[test]
fn free_camera_mouse_look_changes_target() {
    let mut free = FreeCamera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
    let before = free.camera().target;
    let input = InputState { mouse_dx: 100.0, ..Default::default() };
    free.update(&input, 1.0 / 60.0);
    let after = free.camera().target;
    assert!((before - after).length() > 0.01);
}

#[test]
fn follow_camera_tracks_target() {
    let mut follow = FollowCamera::new(Vec3::new(0.0, 5.0, 10.0));
    follow.set_target(Vec3::new(100.0, 0.0, 0.0));
    let input = InputState::default();
    for _ in 0..600 {
        follow.update(&input, 1.0 / 60.0);
    }
    let cam = follow.camera();
    assert!(cam.target.x > 50.0, "x={}", cam.target.x);
}

#[test]
fn follow_camera_maintains_offset() {
    let offset = Vec3::new(0.0, 5.0, 10.0);
    let mut follow = FollowCamera::new(offset);
    let target = Vec3::new(10.0, 0.0, 0.0);
    follow.set_target(target);
    for _ in 0..600 {
        follow.update(&InputState::default(), 1.0 / 60.0);
    }
    let cam = follow.camera();
    let actual_offset = cam.position - cam.target;
    assert!((actual_offset - offset).length() < 1.0, "offset={actual_offset}");
}
