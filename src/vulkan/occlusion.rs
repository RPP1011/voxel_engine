/// Test if an AABB is fully occluded by the depth buffer.
/// Projects the AABB to screen space, finds the max depth of the AABB,
/// and compares against the depth buffer values at those pixels.
/// Returns true if the AABB is fully behind existing geometry.
pub fn is_aabb_occluded(
    depth_buffer: &[f32],
    width: u32,
    height: u32,
    mvp: &[f32; 16],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> bool {
    // Project 8 corners of AABB to screen space
    let corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ];

    let mut min_sx = f32::MAX;
    let mut max_sx = f32::MIN;
    let mut min_sy = f32::MAX;
    let mut max_sy = f32::MIN;
    let mut min_depth = f32::MAX; // nearest depth of AABB

    for corner in &corners {
        let clip = mat4_mul_vec4(mvp, [corner[0], corner[1], corner[2], 1.0]);
        if clip[3] <= 0.0 { return false; } // behind camera

        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let ndc_z = clip[2] / clip[3];

        let sx = (ndc_x * 0.5 + 0.5) * width as f32;
        let sy = (ndc_y * 0.5 + 0.5) * height as f32;

        min_sx = min_sx.min(sx);
        max_sx = max_sx.max(sx);
        min_sy = min_sy.min(sy);
        max_sy = max_sy.max(sy);
        min_depth = min_depth.min(ndc_z);
    }

    // Clamp to screen bounds
    let x0 = (min_sx.floor() as i32).max(0) as u32;
    let x1 = (max_sx.ceil() as i32).min(width as i32 - 1).max(0) as u32;
    let y0 = (min_sy.floor() as i32).max(0) as u32;
    let y1 = (max_sy.ceil() as i32).min(height as i32 - 1).max(0) as u32;

    if x0 > x1 || y0 > y1 { return false; } // off-screen

    // Check if ALL pixels in the projected rect have depth < min_depth of AABB
    // (meaning existing geometry is closer everywhere the AABB projects to)
    for y in y0..=y1 {
        for x in x0..=x1 {
            let idx = (y * width + x) as usize;
            if idx < depth_buffer.len() {
                if depth_buffer[idx] >= min_depth {
                    return false; // at least one pixel has no occluder
                }
            }
        }
    }

    true // fully occluded
}

/// Sort objects by distance (front-to-back) for depth-based culling.
pub fn sort_front_to_back(objects: &mut [(usize, f32)]) {
    objects.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
}

fn mat4_mul_vec4(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    // Column-major multiplication
    [
        m[0]*v[0] + m[4]*v[1] + m[8]*v[2]  + m[12]*v[3],
        m[1]*v[0] + m[5]*v[1] + m[9]*v[2]  + m[13]*v[3],
        m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
        m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3],
    ]
}
