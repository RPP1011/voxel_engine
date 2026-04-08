/// Generate a MIP level from raw voxel data by taking the max over NxNxN blocks.
///
/// `stride` is the block size (2 for MIP1, 4 for MIP2).
/// Returns (mip_data, mip_width, mip_height, mip_depth).
pub fn generate_mip(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    stride: u32,
) -> (Vec<u8>, u32, u32, u32) {
    let mw = (width + stride - 1) / stride;
    let mh = (height + stride - 1) / stride;
    let md = (depth + stride - 1) / stride;
    let mut mip = vec![0u8; (mw * mh * md) as usize];

    for mz in 0..md {
        for my in 0..mh {
            for mx in 0..mw {
                let mut max_val: u8 = 0;
                for dz in 0..stride {
                    for dy in 0..stride {
                        for dx in 0..stride {
                            let x = mx * stride + dx;
                            let y = my * stride + dy;
                            let z = mz * stride + dz;
                            if x < width && y < height && z < depth {
                                let idx = (z * height * width + y * width + x) as usize;
                                max_val = max_val.max(data[idx]);
                            }
                        }
                    }
                }
                let mi = (mz * mh * mw + my * mw + mx) as usize;
                mip[mi] = max_val;
            }
        }
    }

    (mip, mw, mh, md)
}
