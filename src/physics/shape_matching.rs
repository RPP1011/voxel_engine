/// A rigid transform: rotation matrix + translation.
pub struct RigidTransform {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

/// Extract the best-fit rigid transform from rest positions to deformed positions.
/// Uses shape matching via SVD of the covariance matrix (simplified polar decomposition).
pub fn extract_rigid_transform(rest: &[[f64; 3]], deformed: &[[f64; 3]]) -> RigidTransform {
    assert_eq!(rest.len(), deformed.len());
    let n = rest.len() as f64;

    // Compute centroids
    let mut rest_com = [0.0; 3];
    let mut def_com = [0.0; 3];
    for i in 0..rest.len() {
        for d in 0..3 {
            rest_com[d] += rest[i][d];
            def_com[d] += deformed[i][d];
        }
    }
    for d in 0..3 {
        rest_com[d] /= n;
        def_com[d] /= n;
    }

    // Compute covariance matrix H = Σ (qi - q_com)(pi - p_com)^T
    // Where q = rest (centered), p = deformed (centered)
    // H[i][j] = Σ q_centered[k][i] * p_centered[k][j]
    let mut h = [[0.0; 3]; 3];
    for k in 0..rest.len() {
        let q = [
            rest[k][0] - rest_com[0],
            rest[k][1] - rest_com[1],
            rest[k][2] - rest_com[2],
        ];
        let p = [
            deformed[k][0] - def_com[0],
            deformed[k][1] - def_com[1],
            deformed[k][2] - def_com[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += q[i] * p[j];
            }
        }
    }

    // Extract rotation via iterative polar decomposition:
    // R = H * (H^T * H)^{-1/2}
    // Simplified: use iterative approach R_{n+1} = 0.5 * (R_n + (R_n^{-T}))
    let mut r = h;

    for _ in 0..20 {
        let r_inv_t = inverse_transpose_3x3(&r);
        if let Some(rit) = r_inv_t {
            for i in 0..3 {
                for j in 0..3 {
                    r[i][j] = 0.5 * (r[i][j] + rit[i][j]);
                }
            }
        } else {
            break;
        }
    }

    // Ensure proper rotation (det > 0)
    let det = det_3x3(&r);
    if det < 0.0 {
        for j in 0..3 {
            r[2][j] = -r[2][j];
        }
    }

    // Translation = def_com - R * rest_com
    let r_rest_com = mat_vec_3x3(&r, &rest_com);
    let translation = [
        def_com[0] - r_rest_com[0],
        def_com[1] - r_rest_com[1],
        def_com[2] - r_rest_com[2],
    ];

    RigidTransform {
        rotation: r,
        translation,
    }
}

fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse_transpose_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = det_3x3(m);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;

    // Inverse of m, then transpose = cofactor / det
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
        ],
        [
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
        ],
        [
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

fn mat_vec_3x3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
