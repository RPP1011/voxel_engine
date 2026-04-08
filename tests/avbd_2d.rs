use voxel_engine::physics::avbd2d::{AvbdSolver2D, Body2D};

#[test]
fn single_box_falls_and_rests_on_ground() {
    let mut solver = AvbdSolver2D::new();
    solver.gravity = -9.81;
    solver.ground_y = 0.0;

    // Drop box from height 5, box is 1x1
    solver.add_body(Body2D {
        x: 0.0,
        y: 5.0,
        angle: 0.0,
        vx: 0.0,
        vy: 0.0,
        angular_vel: 0.0,
        half_w: 0.5,
        half_h: 0.5,
        mass: 1.0,
        inertia: 1.0 / 6.0, // box inertia
        is_static: false,
    });

    // Simulate for 500 timesteps
    for _ in 0..500 {
        solver.step(1.0 / 60.0, 4);
    }

    let body = &solver.bodies()[0];
    // Should have come to rest near ground (y ≈ half_h = 0.5)
    assert!(
        (body.y - 0.5).abs() < 0.1,
        "Box should rest at y≈0.5, got y={}",
        body.y
    );
    assert!(
        body.vy.abs() < 0.1,
        "Box should be nearly at rest, vy={}",
        body.vy
    );
}

#[test]
fn box_stacks_on_another_box() {
    let mut solver = AvbdSolver2D::new();
    solver.gravity = -9.81;
    solver.ground_y = 0.0;

    // Bottom box (already on ground)
    solver.add_body(Body2D {
        x: 0.0,
        y: 0.5,
        angle: 0.0,
        vx: 0.0,
        vy: 0.0,
        angular_vel: 0.0,
        half_w: 0.5,
        half_h: 0.5,
        mass: 1.0,
        inertia: 1.0 / 6.0,
        is_static: false,
    });

    // Top box (dropped from height 3)
    solver.add_body(Body2D {
        x: 0.0,
        y: 3.0,
        angle: 0.0,
        vx: 0.0,
        vy: 0.0,
        angular_vel: 0.0,
        half_w: 0.5,
        half_h: 0.5,
        mass: 1.0,
        inertia: 1.0 / 6.0,
        is_static: false,
    });

    for _ in 0..500 {
        solver.step(1.0 / 60.0, 4);
    }

    let bottom = &solver.bodies()[0];
    let top = &solver.bodies()[1];

    // Bottom should be on ground
    assert!(
        (bottom.y - 0.5).abs() < 0.15,
        "Bottom box should be at y≈0.5, got {}",
        bottom.y
    );
    // Top should be stacked on bottom
    assert!(
        (top.y - 1.5).abs() < 0.15,
        "Top box should be at y≈1.5, got {}",
        top.y
    );
}

#[test]
fn ten_box_tower_stable() {
    let mut solver = AvbdSolver2D::new();
    solver.gravity = -9.81;
    solver.ground_y = 0.0;

    for i in 0..10 {
        solver.add_body(Body2D {
            x: 0.0,
            y: 0.5 + i as f64 * 1.0,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_vel: 0.0,
            half_w: 0.5,
            half_h: 0.5,
            mass: 1.0,
            inertia: 1.0 / 6.0,
            is_static: false,
        });
    }

    for _ in 0..1000 {
        solver.step(1.0 / 60.0, 4);
    }

    // Check no interpenetration and no explosion
    for (i, body) in solver.bodies().iter().enumerate() {
        assert!(
            body.y > -0.1,
            "Body {i} fell through ground: y={}",
            body.y
        );
        assert!(
            body.y < 20.0,
            "Body {i} exploded: y={}",
            body.y
        );
    }

    // Check bodies are vertically separated (not all at same y)
    let mut ys: Vec<f64> = solver.bodies().iter().map(|b| b.y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spread = ys.last().unwrap() - ys.first().unwrap();
    assert!(
        spread > 5.0,
        "10-box tower should have vertical spread > 5, got {spread} (ys: {ys:?})"
    );
}

#[test]
fn energy_non_increasing() {
    let mut solver = AvbdSolver2D::new();
    solver.gravity = 0.0; // No external forces
    solver.ground_y = -100.0; // Far away

    // Two boxes moving toward each other
    solver.add_body(Body2D {
        x: -3.0,
        y: 0.0,
        angle: 0.0,
        vx: 1.0,
        vy: 0.0,
        angular_vel: 0.0,
        half_w: 0.5,
        half_h: 0.5,
        mass: 1.0,
        inertia: 1.0 / 6.0,
        is_static: false,
    });
    solver.add_body(Body2D {
        x: 3.0,
        y: 0.0,
        angle: 0.0,
        vx: -1.0,
        vy: 0.0,
        angular_vel: 0.0,
        half_w: 0.5,
        half_h: 0.5,
        mass: 1.0,
        inertia: 1.0 / 6.0,
        is_static: false,
    });

    let initial_ke = solver.total_kinetic_energy();
    for _ in 0..200 {
        solver.step(1.0 / 60.0, 4);
    }
    let final_ke = solver.total_kinetic_energy();

    assert!(
        final_ke <= initial_ke + 1e-6,
        "Energy should not increase: initial={initial_ke}, final={final_ke}"
    );
}
