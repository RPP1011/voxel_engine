#version 450

// DDA step-count heatmap visualization.
// Red = few steps (fast / MIP helped), Blue = many steps (slow), Grey = miss.

layout(location = 0) in vec3 frag_entry_pos;
layout(location = 1) flat in vec3 frag_camera_pos;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;
layout(location = 3) out vec4 out_velocity;
layout(location = 4) out float out_depth;

layout(set = 0, binding = 0) uniform usampler3D voxel_grid;
layout(set = 0, binding = 1) uniform usampler3D mip1_grid;
layout(set = 0, binding = 2) uniform usampler3D mip2_grid;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;
    vec4 camera_pos;
    vec4 palette_color;  // .w reused: 0.0 = heatmap mode (MIP on), 1.0 = heatmap mode (MIP off)
    vec4 mip1_dim;
    vec4 mip2_dim;
} pc;

// Blue-to-red heatmap: t=0 → red (fast), t=1 → blue (slow)
vec3 heatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    // red at 0, yellow at 0.25, green at 0.5, cyan at 0.75, blue at 1.0
    float r = clamp(1.5 - t * 3.0, 0.0, 1.0);
    float g = t < 0.5 ? clamp(t * 3.0, 0.0, 1.0) : clamp(3.0 - t * 3.0, 0.0, 1.0);
    float b = clamp(t * 3.0 - 1.5, 0.0, 1.0);
    return vec3(r, g, b);
}

void main() {
    vec3 dim = pc.grid_dim.xyz;
    vec3 entry = clamp(frag_entry_pos, vec3(0.001), dim - vec3(0.001));
    vec3 ray_dir = normalize(entry - frag_camera_pos);

    ivec3 pos = ivec3(floor(entry));
    ivec3 step_dir = ivec3(sign(ray_dir));
    vec3 t_delta = abs(vec3(1.0) / ray_dir);
    vec3 next_boundary = vec3(pos) + max(vec3(step_dir), vec3(0.0));
    vec3 t_max = (next_boundary - entry) / ray_dir;

    if (abs(ray_dir.x) < 1e-8) t_max.x = 1e30;
    if (abs(ray_dir.y) < 1e-8) t_max.y = 1e30;
    if (abs(ray_dir.z) < 1e-8) t_max.z = 1e30;

    int last_axis = 0;
    int last_sign = 1;
    // Entry face detection (same as gbuffer.frag)
    vec3 raw = frag_entry_pos;
    vec3 dist_lo = abs(raw);
    vec3 dist_hi = abs(raw - dim);
    float d0 = min(dist_lo.x, dist_hi.x);
    float d1 = min(dist_lo.y, dist_hi.y);
    float d2 = min(dist_lo.z, dist_hi.z);
    if (d0 < d1 && d0 < d2) {
        last_axis = 0; last_sign = (dist_lo.x < dist_hi.x) ? -1 : 1;
    } else if (d1 < d2) {
        last_axis = 1; last_sign = (dist_lo.y < dist_hi.y) ? -1 : 1;
    } else {
        last_axis = 2; last_sign = (dist_lo.z < dist_hi.z) ? -1 : 1;
    }

    bool mip_enabled = (pc.palette_color.w < 0.5); // 0.0 = MIP on, 1.0 = MIP off
    int max_steps = int(dim.x + dim.y + dim.z) * 2;
    int step_count = 0;

    for (int i = 0; i < max_steps; i++) {
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= int(dim.x) || pos.y >= int(dim.y) || pos.z >= int(dim.z)) {
            break; // miss — will output grey below
        }

        // MIP skip (only when enabled)
        if (mip_enabled) {
            if ((pos.x & 3) == 0 && (pos.y & 3) == 0 && (pos.z & 3) == 0) {
                ivec3 m2pos = pos >> 2;
                if (m2pos.x >= 0 && m2pos.y >= 0 && m2pos.z >= 0 &&
                    m2pos.x < int(pc.mip2_dim.x) && m2pos.y < int(pc.mip2_dim.y) && m2pos.z < int(pc.mip2_dim.z)) {
                    if (texelFetch(mip2_grid, m2pos, 0).r == 0u) {
                        if (t_max.x < t_max.y && t_max.x < t_max.z) {
                            pos.x += step_dir.x * 4; t_max.x += t_delta.x * 4.0;
                            last_axis = 0; last_sign = step_dir.x;
                        } else if (t_max.y < t_max.z) {
                            pos.y += step_dir.y * 4; t_max.y += t_delta.y * 4.0;
                            last_axis = 1; last_sign = step_dir.y;
                        } else {
                            pos.z += step_dir.z * 4; t_max.z += t_delta.z * 4.0;
                            last_axis = 2; last_sign = step_dir.z;
                        }
                        step_count++;
                        continue;
                    }
                }
            }
            else if ((pos.x & 1) == 0 && (pos.y & 1) == 0 && (pos.z & 1) == 0) {
                ivec3 m1pos = pos >> 1;
                if (m1pos.x >= 0 && m1pos.y >= 0 && m1pos.z >= 0 &&
                    m1pos.x < int(pc.mip1_dim.x) && m1pos.y < int(pc.mip1_dim.y) && m1pos.z < int(pc.mip1_dim.z)) {
                    if (texelFetch(mip1_grid, m1pos, 0).r == 0u) {
                        if (t_max.x < t_max.y && t_max.x < t_max.z) {
                            pos.x += step_dir.x * 2; t_max.x += t_delta.x * 2.0;
                            last_axis = 0; last_sign = step_dir.x;
                        } else if (t_max.y < t_max.z) {
                            pos.y += step_dir.y * 2; t_max.y += t_delta.y * 2.0;
                            last_axis = 1; last_sign = step_dir.y;
                        } else {
                            pos.z += step_dir.z * 2; t_max.z += t_delta.z * 2.0;
                            last_axis = 2; last_sign = step_dir.z;
                        }
                        step_count++;
                        continue;
                    }
                }
            }
        }

        uint voxel = texelFetch(voxel_grid, pos, 0).r;
        step_count++;

        if (voxel != 0u) {
            // HIT — color by step count
            float max_expected = dim.x + dim.y + dim.z; // diagonal = max reasonable steps
            float t = float(step_count) / max_expected;
            vec3 color = heatmap(t);

            float t_hit = min(min(t_max.x, t_max.y), t_max.z);
            vec3 hit_pos = entry + ray_dir * max(t_hit - 0.01, 0.0);

            out_albedo = vec4(color, 1.0);
            out_normal = vec4(0.5, 0.5, 1.0, 0.0);
            out_material = vec4(0.5, 0.0, 0.0, 0.0);
            out_velocity = vec4(0.0);
            out_depth = length(hit_pos - frag_camera_pos);

            vec4 clip = pc.mvp * vec4(hit_pos, 1.0);
            gl_FragDepth = clip.z / clip.w;
            return;
        }

        // Normal DDA step
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
    }

    // MISS — grey
    out_albedo = vec4(0.15, 0.15, 0.15, 1.0);
    out_normal = vec4(0.5, 0.5, 1.0, 0.0);
    out_material = vec4(0.0);
    out_velocity = vec4(0.0);
    out_depth = 1e6;
    gl_FragDepth = 1.0;
}
