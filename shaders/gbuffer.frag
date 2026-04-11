#version 450

// Force depth test BEFORE fragment shader runs. Without this, the
// discard statements below force late fragment tests, meaning every
// occluded fragment still runs the full DDA raymarch before being
// thrown away. With early_fragment_tests + front-to-back draw order,
// back chunks that are occluded by front chunks skip the DDA entirely.
// Safe here because depth is purely rasterizer-interpolated
// (gl_Position.z from the vertex shader) — we never write gl_FragDepth.
layout(early_fragment_tests) in;

layout(location = 0) in vec3 frag_entry_pos;
layout(location = 1) flat in vec3 frag_camera_pos;

layout(location = 0) out vec4 out_albedo;     // RT0: RGBA8
layout(location = 1) out vec4 out_normal;      // RT1: RGB16F (xyz = world normal)
layout(location = 2) out vec4 out_material;    // RT2: RGBA8 (roughness, metallic, emissive, material_type)
// Former RT3 (velocity) and RT4 (linear depth) removed — neither was read
// by any downstream pass, and the ROP writes cost real per-fragment time.

layout(set = 0, binding = 0) uniform usampler3D voxel_grid;
layout(set = 0, binding = 1) uniform usampler3D mip1_grid;
layout(set = 0, binding = 2) uniform usampler3D mip2_grid;
layout(set = 0, binding = 3) uniform sampler2D palette_tex; // 256x1 RGBA
layout(set = 0, binding = 4) uniform usampler3D mip3_grid;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;
    vec4 camera_pos;
    vec4 palette_color;  // .rgb = albedo, .a = roughness
    vec4 mip1_dim;       // xyz = MIP1 dimensions
    vec4 mip2_dim;       // xyz = MIP2 dimensions
    vec4 mip3_dim;       // xyz = MIP3 dimensions (8x8x8 blocks)
} pc;

void main() {
    vec3 dim = pc.grid_dim.xyz;
    // If camera is inside the volume, start ray from camera position.
    bool camera_inside = all(greaterThanEqual(frag_camera_pos, vec3(0.0)))
                      && all(lessThan(frag_camera_pos, dim));

    // Discard back-face fragments when camera is outside the volume.
    // gl_FrontFacing is true for front faces (camera-facing).
    // Back faces from adjacent cubes cause z-fighting; discarding them
    // eliminates the flickering at chunk boundaries.
    if (!camera_inside && !gl_FrontFacing) {
        discard;
    }

    vec3 entry = camera_inside
        ? clamp(frag_camera_pos, vec3(0.001), dim - vec3(0.001))
        : clamp(frag_entry_pos, vec3(0.001), dim - vec3(0.001));
    vec3 ray_dir = camera_inside
        ? normalize(frag_entry_pos - frag_camera_pos)
        : normalize(entry - frag_camera_pos);

    ivec3 pos = ivec3(floor(entry));
    ivec3 step_dir = ivec3(sign(ray_dir));
    vec3 t_delta = abs(vec3(1.0) / ray_dir);
    vec3 next_boundary = vec3(pos) + max(vec3(step_dir), vec3(0.0));
    vec3 t_max = (next_boundary - entry) / ray_dir;

    if (abs(ray_dir.x) < 1e-8) t_max.x = 1e30;
    if (abs(ray_dir.y) < 1e-8) t_max.y = 1e30;
    if (abs(ray_dir.z) < 1e-8) t_max.z = 1e30;

    // Determine which box face the ray entered through for correct initial normal
    vec3 raw = frag_entry_pos;
    vec3 dist_lo = abs(raw);
    vec3 dist_hi = abs(raw - dim);
    // Find minimum distance to any box face
    float d0 = min(dist_lo.x, dist_hi.x);
    float d1 = min(dist_lo.y, dist_hi.y);
    float d2 = min(dist_lo.z, dist_hi.z);
    int last_axis;
    int last_sign;
    if (d0 < d1 && d0 < d2) {
        last_axis = 0;
        last_sign = (dist_lo.x < dist_hi.x) ? 1 : -1;
    } else if (d1 < d2) {
        last_axis = 1;
        last_sign = (dist_lo.y < dist_hi.y) ? 1 : -1;
    } else {
        last_axis = 2;
        last_sign = (dist_lo.z < dist_hi.z) ? 1 : -1;
    }

    // With MIP3 jump-skip, rays traverse empty space in O(1) per block.
    // 128 steps is sufficient for 64³ grids (diagonal ≈ 110 voxels).
    int max_steps = 128;
    int step_count = 0;
    // Visualization modes via palette_color.a:
    // < 1.5 = normal rendering
    // 2.0   = DDA step-count heatmap
    // 3.0   = physics velocity heatmap (.r=speed 0-1, .g=1 if sleeping)
    // 4.0   = normal debug (white=camera-facing, pink=backface)
    bool heatmap_mode = (pc.palette_color.a > 1.5 && pc.palette_color.a < 2.5);
    bool physics_mode = (pc.palette_color.a > 2.5 && pc.palette_color.a < 3.5);
    bool normal_debug = (pc.palette_color.a > 3.5);
    bool any_debug = heatmap_mode || physics_mode || normal_debug;

    for (int i = 0; i < max_steps; i++) {
        step_count++;

        // Bounds check
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= int(dim.x) || pos.y >= int(dim.y) || pos.z >= int(dim.z)) {
            if (any_debug) {
                vec3 sun_n = normalize(vec3(0.5, 0.8, 0.3));
                out_albedo = vec4(0.25, 0.25, 0.25, 1.0);
                out_normal = vec4(sun_n * 0.5 + 0.5, 0.0);
                out_material = vec4(0.0);
                gl_FragDepth = 0.9999;
                return;
            }
            discard;
        }

        // Test current voxel FIRST (last_axis/last_sign from the step that brought us here)
        uint voxel = texelFetch(voxel_grid, pos, 0).r;
        if (voxel != 0u) {
            // Compute face normal from last step axis
            vec3 normal = vec3(0.0);
            if (last_axis == 0) normal.x = -float(last_sign);
            else if (last_axis == 1) normal.y = -float(last_sign);
            else normal.z = -float(last_sign);

            // Hit position
            float t_hit = min(min(t_max.x, t_max.y), t_max.z);
            vec3 hit_pos = entry + ray_dir * max(t_hit - 0.01, 0.0);

            // Write G-buffer outputs
            vec3 final_color;
            if (heatmap_mode) {
                // DDA step-count heatmap: fewer steps = redder (faster)
                float max_expected = dim.x + dim.y + dim.z;
                float t = clamp(float(step_count) / max_expected, 0.0, 1.0);
                float r = clamp(1.5 - t * 3.0, 0.0, 1.0);
                float g = t < 0.5 ? clamp(t * 3.0, 0.0, 1.0) : clamp(3.0 - t * 3.0, 0.0, 1.0);
                float b = clamp(t * 3.0 - 1.5, 0.0, 1.0);
                final_color = vec3(r, g, b);
                normal = normalize(vec3(0.5, 0.8, 0.3));
            } else if (physics_mode) {
                // Physics velocity heatmap
                // .r = normalized speed (0-1), .g = 1.0 if sleeping
                bool is_sleeping = (pc.palette_color.g > 0.5);
                if (is_sleeping) {
                    final_color = vec3(0.3, 0.3, 0.3); // grey = sleeping
                } else {
                    float speed = clamp(pc.palette_color.r, 0.0, 1.0);
                    // Blue (still) → Cyan → Green → Yellow → Red (fast)
                    float r = clamp(speed * 3.0 - 1.0, 0.0, 1.0);
                    float g = speed < 0.5 ? clamp(speed * 3.0, 0.0, 1.0) : clamp(3.0 - speed * 3.0, 0.0, 1.0);
                    float b = clamp(1.0 - speed * 3.0, 0.0, 1.0);
                    final_color = vec3(r, g, b);
                }
                normal = normalize(vec3(0.5, 0.8, 0.3));
            } else if (normal_debug) {
                // White = normal faces camera, hot pink = backface
                vec3 cam_dir = normalize(frag_camera_pos - hit_pos);
                float facing = dot(normal, cam_dir);
                if (facing > 0.0) {
                    final_color = vec3(0.95, 0.95, 0.95); // white = front-facing
                } else {
                    final_color = vec3(1.0, 0.0, 0.5); // hot pink = backface
                }
                // Keep actual normal for shading so front/back are distinguishable
            } else {
                vec4 pal = texelFetch(palette_tex, ivec2(voxel, 0), 0);
                final_color = pal.rgb;
            }
            out_albedo = vec4(final_color, 1.0);
            out_normal = vec4(normal * 0.5 + 0.5, 0.0); // encode [-1,1] → [0,1]
            out_material = vec4(pc.palette_color.a, 0.0, 0.0, float(voxel) / 255.0);

            // Write hardware depth (MVP expects unit-space [0,1], hit_pos is grid-space [0,dim])
            vec4 clip = pc.mvp * vec4(hit_pos / pc.grid_dim.xyz, 1.0);
            gl_FragDepth = clip.z / clip.w;
            return;
        }

        // Advance via single DDA step.
        // MIP acceleration: if we land on an aligned empty block, do rapid single
        // steps through it (the block is guaranteed empty so we can step without checking).
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                last_axis = 0; last_sign = step_dir.x;
                pos.x += step_dir.x; t_max.x += t_delta.x;
            } else {
                last_axis = 2; last_sign = step_dir.z;
                pos.z += step_dir.z; t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                last_axis = 1; last_sign = step_dir.y;
                pos.y += step_dir.y; t_max.y += t_delta.y;
            } else {
                last_axis = 2; last_sign = step_dir.z;
                pos.z += step_dir.z; t_max.z += t_delta.z;
            }
        }

        // MIP acceleration: skip known-empty blocks at multiple scales.
        // When a block is empty, jump the DDA to the block boundary in O(1)
        // instead of stepping voxel-by-voxel.

        // MIP3: 8x8x8 block — jump skip
        {
            ivec3 block = pos >> 3;
            if (block.x >= 0 && block.y >= 0 && block.z >= 0 &&
                block.x < int(pc.mip3_dim.x) && block.y < int(pc.mip3_dim.y) && block.z < int(pc.mip3_dim.z)) {
                if (texelFetch(mip3_grid, block, 0).r == 0u) {
                    // Compute how many voxels to the exit boundary of this 8-block
                    ivec3 block_min = block * 8;
                    ivec3 block_max = block_min + 8;
                    // For each axis, distance to exit boundary
                    int dx = (step_dir.x > 0) ? (block_max.x - pos.x) : (pos.x - block_min.x + 1);
                    int dy = (step_dir.y > 0) ? (block_max.y - pos.y) : (pos.y - block_min.y + 1);
                    int dz = (step_dir.z > 0) ? (block_max.z - pos.z) : (pos.z - block_min.z + 1);
                    // Advance DDA state by jumping across the block
                    float tx = t_max.x + t_delta.x * float(dx - 1);
                    float ty = t_max.y + t_delta.y * float(dy - 1);
                    float tz = t_max.z + t_delta.z * float(dz - 1);
                    if (tx < ty && tx < tz) {
                        last_axis = 0; last_sign = step_dir.x;
                        pos.x += step_dir.x * dx; t_max.x += t_delta.x * float(dx);
                    } else if (ty < tz) {
                        last_axis = 1; last_sign = step_dir.y;
                        pos.y += step_dir.y * dy; t_max.y += t_delta.y * float(dy);
                    } else {
                        last_axis = 2; last_sign = step_dir.z;
                        pos.z += step_dir.z * dz; t_max.z += t_delta.z * float(dz);
                    }
                    step_count++;
                    continue;
                }
            }
        }

        // MIP2: 4x4x4 block — jump skip
        {
            ivec3 block = pos >> 2;
            if (block.x >= 0 && block.y >= 0 && block.z >= 0 &&
                block.x < int(pc.mip2_dim.x) && block.y < int(pc.mip2_dim.y) && block.z < int(pc.mip2_dim.z)) {
                if (texelFetch(mip2_grid, block, 0).r == 0u) {
                    ivec3 block_min = block * 4;
                    ivec3 block_max = block_min + 4;
                    int dx = (step_dir.x > 0) ? (block_max.x - pos.x) : (pos.x - block_min.x + 1);
                    int dy = (step_dir.y > 0) ? (block_max.y - pos.y) : (pos.y - block_min.y + 1);
                    int dz = (step_dir.z > 0) ? (block_max.z - pos.z) : (pos.z - block_min.z + 1);
                    float tx = t_max.x + t_delta.x * float(dx - 1);
                    float ty = t_max.y + t_delta.y * float(dy - 1);
                    float tz = t_max.z + t_delta.z * float(dz - 1);
                    if (tx < ty && tx < tz) {
                        last_axis = 0; last_sign = step_dir.x;
                        pos.x += step_dir.x * dx; t_max.x += t_delta.x * float(dx);
                    } else if (ty < tz) {
                        last_axis = 1; last_sign = step_dir.y;
                        pos.y += step_dir.y * dy; t_max.y += t_delta.y * float(dy);
                    } else {
                        last_axis = 2; last_sign = step_dir.z;
                        pos.z += step_dir.z * dz; t_max.z += t_delta.z * float(dz);
                    }
                    step_count++;
                    continue;
                }
            }
        }

        // MIP1: 2x2x2 block — jump skip
        {
            ivec3 block = pos >> 1;
            if (block.x >= 0 && block.y >= 0 && block.z >= 0 &&
                block.x < int(pc.mip1_dim.x) && block.y < int(pc.mip1_dim.y) && block.z < int(pc.mip1_dim.z)) {
                if (texelFetch(mip1_grid, block, 0).r == 0u) {
                    ivec3 block_min = block * 2;
                    ivec3 block_max = block_min + 2;
                    int dx = (step_dir.x > 0) ? (block_max.x - pos.x) : (pos.x - block_min.x + 1);
                    int dy = (step_dir.y > 0) ? (block_max.y - pos.y) : (pos.y - block_min.y + 1);
                    int dz = (step_dir.z > 0) ? (block_max.z - pos.z) : (pos.z - block_min.z + 1);
                    float tx = t_max.x + t_delta.x * float(dx - 1);
                    float ty = t_max.y + t_delta.y * float(dy - 1);
                    float tz = t_max.z + t_delta.z * float(dz - 1);
                    if (tx < ty && tx < tz) {
                        last_axis = 0; last_sign = step_dir.x;
                        pos.x += step_dir.x * dx; t_max.x += t_delta.x * float(dx);
                    } else if (ty < tz) {
                        last_axis = 1; last_sign = step_dir.y;
                        pos.y += step_dir.y * dy; t_max.y += t_delta.y * float(dy);
                    } else {
                        last_axis = 2; last_sign = step_dir.z;
                        pos.z += step_dir.z * dz; t_max.z += t_delta.z * float(dz);
                    }
                    step_count++;
                }
            }
        }
    }

    discard;
}
