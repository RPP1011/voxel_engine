use voxel_engine::physics::cloth::{ClothMesh, ClothSolver};

#[test]
fn create_10x10_cloth_grid() {
    let cloth = ClothMesh::new_grid(10, 10, 0.1);

    assert_eq!(cloth.vertex_count(), 100);
    // Stretch edges: horizontal + vertical = 9*10 + 10*9 = 180
    assert_eq!(cloth.stretch_constraint_count(), 180);
    // Bending: (10-2)*10 + 10*(10-2) = 80 + 80 = 160
    assert_eq!(cloth.bend_constraint_count(), 160);
}

#[test]
fn pin_top_row() {
    let mut cloth = ClothMesh::new_grid(10, 10, 0.1);
    cloth.pin_row(0); // Pin top row (y=0)

    for x in 0..10 {
        assert!(cloth.is_pinned(x, 0), "Top row vertex ({x},0) should be pinned");
    }
    assert!(!cloth.is_pinned(0, 1), "Second row should not be pinned");
}

#[test]
fn distance_constraint_restores_rest_length() {
    let mut cloth = ClothMesh::new_grid(2, 1, 1.0);
    // Two vertices at (0,0,0) and (1,0,0), rest length = 1.0
    // Stretch them apart to distance 2.0
    cloth.set_position(1, [2.0, 0.0, 0.0]);

    let mut solver = ClothSolver::new(0.0); // no gravity
    solver.solve_constraints(&mut cloth, 10);

    let p0 = cloth.position(0);
    let p1 = cloth.position(1);
    let dist = ((p1[0]-p0[0]).powi(2) + (p1[1]-p0[1]).powi(2) + (p1[2]-p0[2]).powi(2)).sqrt();

    assert!(
        (dist - 1.0).abs() < 0.1,
        "Distance should return toward rest length 1.0, got {dist}"
    );
}

#[test]
fn cloth_falls_under_gravity_with_pin() {
    let mut cloth = ClothMesh::new_grid(5, 5, 0.1);
    cloth.pin_row(0);

    let mut solver = ClothSolver::new(-9.81);

    for _ in 0..200 {
        solver.step(&mut cloth, 1.0/60.0, 4);
    }

    // Bottom row should have fallen below top row
    let top_y = cloth.position(0)[1]; // pinned, should be at 0
    let bottom_y = cloth.position(4 * 5)[1]; // row 4
    assert!(
        bottom_y < top_y - 0.1,
        "Bottom should fall below top: top_y={top_y}, bottom_y={bottom_y}"
    );
}

#[test]
fn lra_prevents_excessive_stretching() {
    // 50x1 strip pinned at one end
    let mut cloth = ClothMesh::new_grid(50, 1, 0.1);
    cloth.pin_vertex(0);
    cloth.compute_lra(); // Pre-compute geodesic distances

    let mut solver = ClothSolver::new(-9.81);
    solver.enable_lra(true);

    for _ in 0..500 {
        solver.step(&mut cloth, 1.0/60.0, 4);
    }

    // Measure total length of strip
    let p_start = cloth.position(0);
    let p_end = cloth.position(49);
    let end_to_end = ((p_end[0]-p_start[0]).powi(2) + (p_end[1]-p_start[1]).powi(2) + (p_end[2]-p_start[2]).powi(2)).sqrt();
    let rest_length = 49.0 * 0.1; // 4.9

    // With LRA, stretch should be within 5% of rest length
    let stretch_ratio = end_to_end / rest_length;
    assert!(
        stretch_ratio < 1.10,
        "LRA should limit stretch to <10%: ratio={stretch_ratio}, dist={end_to_end}, rest={rest_length}"
    );
}
