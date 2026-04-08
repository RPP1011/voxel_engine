#version 450

// Depth-only fragment shader for shadow map generation.
// Uses DDA ray march to find the first hit and writes gl_FragDepth.

layout(location = 0) in vec3 frag_entry_pos;
layout(location = 1) flat in vec3 frag_camera_pos;

layout(set = 0, binding = 0) uniform usampler3D voxel_grid;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;
    vec4 camera_pos;
    vec4 unused;
} pc;

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

    int max_steps = int(dim.x + dim.y + dim.z) * 2;

    for (int i = 0; i < max_steps; i++) {
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= int(dim.x) || pos.y >= int(dim.y) || pos.z >= int(dim.z)) {
            discard;
        }

        uint voxel = texelFetch(voxel_grid, pos, 0).r;
        if (voxel != 0u) {
            // Use voxel center for depth computation
            vec3 voxel_center = vec3(pos) + vec3(0.5);
            vec4 clip = pc.mvp * vec4(voxel_center, 1.0);
            gl_FragDepth = clamp(clip.z / clip.w, 0.0, 1.0);
            return;
        }

        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) { pos.x += step_dir.x; t_max.x += t_delta.x; }
            else { pos.z += step_dir.z; t_max.z += t_delta.z; }
        } else {
            if (t_max.y < t_max.z) { pos.y += step_dir.y; t_max.y += t_delta.y; }
            else { pos.z += step_dir.z; t_max.z += t_delta.z; }
        }
    }
    discard;
}
