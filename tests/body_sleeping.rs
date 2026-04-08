use voxel_engine::physics::sleeping::{SleepSystem, BodyState};

#[test]
fn body_at_rest_goes_to_sleep() {
    let mut sys = SleepSystem::new(60, 0.01);

    // Body with near-zero KE for 60+ frames
    for _ in 0..70 {
        sys.update(0, &BodyState { vx: 0.001, vy: 0.001, angular_vel: 0.0, mass: 1.0 });
    }

    assert!(sys.is_sleeping(0), "Body should be sleeping after 70 low-KE frames");
}

#[test]
fn moving_body_stays_awake() {
    let mut sys = SleepSystem::new(60, 0.01);

    for _ in 0..100 {
        sys.update(0, &BodyState { vx: 5.0, vy: 0.0, angular_vel: 0.0, mass: 1.0 });
    }

    assert!(!sys.is_sleeping(0), "Moving body should stay awake");
}

#[test]
fn sleeping_body_wakes_on_collision() {
    let mut sys = SleepSystem::new(60, 0.01);

    // Put to sleep
    for _ in 0..70 {
        sys.update(0, &BodyState { vx: 0.0, vy: 0.0, angular_vel: 0.0, mass: 1.0 });
    }
    assert!(sys.is_sleeping(0));

    // Wake it
    sys.wake(0);
    assert!(!sys.is_sleeping(0), "Should be awake after wake()");
}

#[test]
fn nearby_destruction_wakes_bodies() {
    let mut sys = SleepSystem::new(60, 0.01);

    // Put bodies 0,1,2 to sleep
    for id in 0..3 {
        for _ in 0..70 {
            sys.update(id, &BodyState { vx: 0.0, vy: 0.0, angular_vel: 0.0, mass: 1.0 });
        }
    }

    // Register body positions
    sys.set_position(0, [0.0, 0.0, 0.0]);
    sys.set_position(1, [3.0, 0.0, 0.0]);
    sys.set_position(2, [20.0, 0.0, 0.0]);

    // Destruction event at origin, radius 5
    sys.wake_in_radius([0.0, 0.0, 0.0], 5.0);

    assert!(!sys.is_sleeping(0), "Body 0 near explosion should wake");
    assert!(!sys.is_sleeping(1), "Body 1 within radius should wake");
    assert!(sys.is_sleeping(2), "Body 2 far away should stay asleep");
}

#[test]
fn sleeping_body_count() {
    let mut sys = SleepSystem::new(60, 0.01);

    // 5 bodies, put 3 to sleep
    for id in 0..5u64 {
        let vel = if id < 3 { 0.0 } else { 5.0 };
        for _ in 0..70 {
            sys.update(id, &BodyState { vx: vel, vy: 0.0, angular_vel: 0.0, mass: 1.0 });
        }
    }

    assert_eq!(sys.sleeping_count(), 3);
    assert_eq!(sys.awake_count(), 2);
}
