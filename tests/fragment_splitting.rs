use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::splitting::{split_into_fragments, Fragment};

#[test]
fn remove_center_column_splits_into_two() {
    let mut grid = VoxelGrid::new(16, 8, 8);
    // Fill entirely
    for x in 0..16 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, 1);
            }
        }
    }
    // Remove center column at x=8
    for y in 0..8 {
        for z in 0..8 {
            grid.set(8, y, z, 0);
        }
    }

    let fragments = split_into_fragments(&grid, 0);
    assert_eq!(fragments.len(), 2, "Should split into 2 fragments");

    // Each fragment should have the correct voxels
    let sizes: Vec<usize> = fragments.iter().map(|f| f.grid.count_nonempty()).collect();
    assert!(sizes.contains(&(8 * 8 * 8)), "Left half: 8*8*8 = 512");
    assert!(sizes.contains(&(7 * 8 * 8)), "Right half: 7*8*8 = 448");
}

#[test]
fn fragment_inherits_correct_world_offset() {
    let mut grid = VoxelGrid::new(16, 8, 8);
    for x in 0..16 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, 1);
            }
        }
    }
    // Remove column at x=8
    for y in 0..8 {
        for z in 0..8 {
            grid.set(8, y, z, 0);
        }
    }

    let fragments = split_into_fragments(&grid, 0);

    // Find the right-side fragment (starts at x=9)
    let right_frag = fragments
        .iter()
        .find(|f| f.offset.0 == 9)
        .expect("Should have fragment starting at x=9");
    assert_eq!(right_frag.offset, (9, 0, 0));

    // The left fragment starts at x=0
    let left_frag = fragments
        .iter()
        .find(|f| f.offset.0 == 0)
        .expect("Should have fragment starting at x=0");
    assert_eq!(left_frag.offset, (0, 0, 0));
}

#[test]
fn small_fragment_below_threshold_is_deleted() {
    let mut grid = VoxelGrid::new(16, 16, 16);
    // Place a large block
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, 1);
            }
        }
    }
    // Place a tiny separate block (10 voxels, below threshold of 25)
    for x in 12..14 {
        for y in 0..5 {
            grid.set(x, y, 0, 2);
        }
    }
    assert_eq!(grid.count_nonempty(), 512 + 10);

    let fragments = split_into_fragments(&grid, 25);
    // Only the large block should survive
    assert_eq!(fragments.len(), 1, "Small fragment should be auto-deleted");
    assert_eq!(fragments[0].grid.count_nonempty(), 512);
}

#[test]
fn single_component_returns_one_fragment() {
    let mut grid = VoxelGrid::new(8, 8, 8);
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                grid.set(x, y, z, 1);
            }
        }
    }

    let fragments = split_into_fragments(&grid, 0);
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].grid.count_nonempty(), 512);
    assert_eq!(fragments[0].offset, (0, 0, 0));
}

#[test]
fn fragment_preserves_palette_indices() {
    let mut grid = VoxelGrid::new(8, 8, 1);
    // Left block with palette 5
    for x in 0..3 {
        grid.set(x, 0, 0, 5);
    }
    // Right block with palette 10 (gap at x=3)
    for x in 4..7 {
        grid.set(x, 0, 0, 10);
    }

    let fragments = split_into_fragments(&grid, 0);
    assert_eq!(fragments.len(), 2);

    for frag in &fragments {
        let (w, h, d) = frag.grid.dimensions();
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    if let Some(v) = frag.grid.get(x, y, z) {
                        if v != 0 {
                            // Should be 5 or 10, not something else
                            assert!(
                                v == 5 || v == 10,
                                "Palette index should be preserved, got {v}"
                            );
                        }
                    }
                }
            }
        }
    }
}
