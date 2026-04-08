#version 450

layout(location = 0) in vec3 frag_ray_origin;   // camera-space ray origin
layout(location = 1) in vec3 frag_entry_pos;     // entry point into bbox in grid space [0..dim]

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform usampler3D voxel_grid; // R8_UINT
layout(set = 0, binding = 1) uniform usampler3D mip1_grid;
layout(set = 0, binding = 2) uniform usampler3D mip2_grid;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;       // xyz = dimensions, w = unused
    vec4 camera_pos;     // xyz = camera position in grid space
    vec4 palette_color;  // color for the first non-zero palette entry
    vec4 mip1_dim;       // xyz = MIP1 dimensions
    vec4 mip2_dim;       // xyz = MIP2 dimensions
} pc;

void main() {
    vec3 dim = pc.grid_dim.xyz;
    vec3 entry = frag_entry_pos;

    // Ray direction from camera to entry point
    vec3 ray_dir = normalize(entry - pc.camera_pos.xyz);

    // Clamp entry into grid bounds
    entry = clamp(entry, vec3(0.001), dim - vec3(0.001));

    // DDA setup (Amanatides-Woo)
    ivec3 pos = ivec3(floor(entry));
    ivec3 step_dir = ivec3(sign(ray_dir));
    vec3 t_delta = abs(vec3(1.0) / ray_dir);

    vec3 next_boundary = vec3(pos) + max(vec3(step_dir), vec3(0.0));
    vec3 t_max = (next_boundary - entry) / ray_dir;

    // Fix for zero direction components
    if (abs(ray_dir.x) < 1e-8) t_max.x = 1e30;
    if (abs(ray_dir.y) < 1e-8) t_max.y = 1e30;
    if (abs(ray_dir.z) < 1e-8) t_max.z = 1e30;

    int max_steps = int(dim.x + dim.y + dim.z) * 2;

    for (int i = 0; i < max_steps; i++) {
        // Bounds check
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= int(dim.x) || pos.y >= int(dim.y) || pos.z >= int(dim.z)) {
            discard;
        }

        // MIP2 skip: check 4x4x4 block if position is 4-aligned
        if ((pos.x & 3) == 0 && (pos.y & 3) == 0 && (pos.z & 3) == 0) {
            ivec3 m2pos = pos >> 2;
            if (m2pos.x >= 0 && m2pos.y >= 0 && m2pos.z >= 0 &&
                m2pos.x < int(pc.mip2_dim.x) && m2pos.y < int(pc.mip2_dim.y) && m2pos.z < int(pc.mip2_dim.z)) {
                if (texelFetch(mip2_grid, m2pos, 0).r == 0u) {
                    if (t_max.x < t_max.y && t_max.x < t_max.z) {
                        pos.x += step_dir.x * 4;
                        t_max.x += t_delta.x * 4.0;
                    } else if (t_max.y < t_max.z) {
                        pos.y += step_dir.y * 4;
                        t_max.y += t_delta.y * 4.0;
                    } else {
                        pos.z += step_dir.z * 4;
                        t_max.z += t_delta.z * 4.0;
                    }
                    continue;
                }
            }
        }
        // MIP1 skip: check 2x2x2 block if position is 2-aligned
        else if ((pos.x & 1) == 0 && (pos.y & 1) == 0 && (pos.z & 1) == 0) {
            ivec3 m1pos = pos >> 1;
            if (m1pos.x >= 0 && m1pos.y >= 0 && m1pos.z >= 0 &&
                m1pos.x < int(pc.mip1_dim.x) && m1pos.y < int(pc.mip1_dim.y) && m1pos.z < int(pc.mip1_dim.z)) {
                if (texelFetch(mip1_grid, m1pos, 0).r == 0u) {
                    if (t_max.x < t_max.y && t_max.x < t_max.z) {
                        pos.x += step_dir.x * 2;
                        t_max.x += t_delta.x * 2.0;
                    } else if (t_max.y < t_max.z) {
                        pos.y += step_dir.y * 2;
                        t_max.y += t_delta.y * 2.0;
                    } else {
                        pos.z += step_dir.z * 2;
                        t_max.z += t_delta.z * 2.0;
                    }
                    continue;
                }
            }
        }

        // Sample voxel
        uint voxel = texelFetch(voxel_grid, pos, 0).r;
        if (voxel != 0u) {
            // Hit! Output palette color and write depth
            out_color = pc.palette_color;

            // Compute depth from hit distance
            float t_hit = min(min(t_max.x, t_max.y), t_max.z);
            vec3 hit_pos = entry + ray_dir * max(t_hit - 0.01, 0.0);
            vec4 clip = pc.mvp * vec4(hit_pos / pc.grid_dim.xyz, 1.0);
            gl_FragDepth = clip.z / clip.w;
            return;
        }

        // Step to next voxel
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                pos.x += step_dir.x;
                t_max.x += t_delta.x;
            } else {
                pos.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                pos.y += step_dir.y;
                t_max.y += t_delta.y;
            } else {
                pos.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        }
    }

    discard; // No hit
}
