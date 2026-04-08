use voxel_engine::fluid::swe::{SweGrid, SweParams};

#[test]
fn water_column_spreads_uniformly_on_flat_terrain() {
    let mut grid = SweGrid::new(16, 16, SweParams::default());

    // Place water column in center
    grid.set_water_height(8, 8, 5.0);

    // Simulate
    for _ in 0..500 {
        grid.step(0.01);
    }

    // Water should have spread — center should be lower, edges higher
    let center_h = grid.water_height(8, 8);
    let edge_h = grid.water_height(0, 0);

    assert!(
        center_h < 5.0,
        "Center should have lost water, got {center_h}"
    );
    // Eventually approaches uniform height
    let avg = grid.total_water_volume() / (16.0 * 16.0);
    assert!(
        (center_h - avg).abs() < 1.0,
        "Center should approach average {avg}, got {center_h}"
    );
}

#[test]
fn water_flows_downhill() {
    let mut grid = SweGrid::new(16, 16, SweParams::default());

    // Sloped terrain: terrain_height increases with x
    for x in 0..16 {
        for z in 0..16 {
            grid.set_terrain_height(x, z, x as f64 * 0.5);
        }
    }

    // Place water at the top of the slope
    grid.set_water_height(14, 8, 3.0);

    for _ in 0..500 {
        grid.step(0.01);
    }

    // Water should accumulate at the bottom (low x)
    let bottom_h = grid.water_height(0, 8);
    let top_h = grid.water_height(14, 8);

    assert!(
        bottom_h > top_h,
        "Water should flow downhill: bottom={bottom_h}, top={top_h}"
    );
}

#[test]
fn volume_conservation() {
    let mut grid = SweGrid::new(16, 16, SweParams::default());
    grid.set_water_height(8, 8, 10.0);

    let initial_vol = grid.total_water_volume();
    assert!(initial_vol > 0.0);

    for _ in 0..1000 {
        grid.step(0.01);
    }

    let final_vol = grid.total_water_volume();
    let error = (final_vol - initial_vol).abs() / initial_vol;
    assert!(
        error < 0.01,
        "Volume should be conserved within 1%: initial={initial_vol}, final={final_vol}, error={error}"
    );
}

#[test]
fn wall_blocks_water() {
    let mut grid = SweGrid::new(16, 16, SweParams::default());

    // Build a wall of high terrain at x=8
    for z in 0..16 {
        grid.set_terrain_height(8, z, 100.0);
    }

    // Water on left side
    grid.set_water_height(4, 8, 5.0);

    for _ in 0..500 {
        grid.step(0.01);
    }

    // No water should cross the wall
    let right_water: f64 = (9..16).map(|x| grid.water_height(x, 8)).sum();
    assert!(
        right_water < 0.01,
        "Water should not cross wall, got {right_water} on right side"
    );
}
