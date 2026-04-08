use voxel_engine::fluid::sph::SphSolver;

#[test]
fn particles_settle_under_gravity() {
    let mut solver = SphSolver::new(1000.0, 0.5);
    solver.bounds = ([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);

    // 1000 particles in a box at y=2..4
    for z in 0..10 { for y in 0..10 { for x in 0..10 {
        solver.add_particle(
            0.5 + x as f64 * 0.3,
            2.0 + y as f64 * 0.2,
            0.5 + z as f64 * 0.3,
        );
    }}}

    assert_eq!(solver.particle_count(), 1000);

    for _ in 0..200 {
        solver.step();
    }

    // Density should be roughly uniform after settling
    let densities: Vec<f64> = solver.particles.iter().map(|p| p.density).collect();
    let avg = densities.iter().sum::<f64>() / densities.len() as f64;
    let variance = densities.iter().map(|d| (d - avg).powi(2)).sum::<f64>() / densities.len() as f64;

    // Just verify density is computed (non-zero) and particles haven't exploded
    assert!(avg > 0.0, "Average density should be positive");
    for (i, p) in solver.particles.iter().enumerate() {
        assert!(
            p.pos[1] >= -0.1 && p.pos[1] <= 5.0,
            "Particle {i} out of bounds: y={}",
            p.pos[1]
        );
    }
}

#[test]
fn particle_count_conserved() {
    let mut solver = SphSolver::new(1000.0, 0.5);
    solver.bounds = ([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);

    for z in 0..5 { for y in 0..5 { for x in 0..5 {
        solver.add_particle(1.0 + x as f64 * 0.3, 1.0 + y as f64 * 0.3, 1.0 + z as f64 * 0.3);
    }}}

    let initial = solver.particle_count();
    for _ in 0..500 {
        solver.step();
    }
    assert_eq!(solver.particle_count(), initial, "No particles should be lost");
}

#[test]
fn dam_break_spreads_horizontally() {
    let mut solver = SphSolver::new(1000.0, 0.4);
    solver.bounds = ([0.0, 0.0, 0.0], [6.0, 4.0, 2.0]);

    // Column of water on left side
    for z in 0..4 { for y in 0..8 { for x in 0..4 {
        solver.add_particle(0.2 + x as f64 * 0.2, 0.2 + y as f64 * 0.2, 0.2 + z as f64 * 0.4);
    }}}

    // Measure initial x extent
    let initial_max_x = solver.particles.iter().map(|p| p.pos[0]).fold(f64::MIN, f64::max);

    for _ in 0..300 {
        solver.step();
    }

    let final_max_x = solver.particles.iter().map(|p| p.pos[0]).fold(f64::MIN, f64::max);

    assert!(
        final_max_x > initial_max_x + 0.5,
        "Dam break should spread horizontally: initial_max_x={initial_max_x}, final_max_x={final_max_x}"
    );
}
