#version 450

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 frag_entry_pos;
layout(location = 1) flat out vec3 frag_camera_pos;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;
    vec4 camera_pos;
    vec4 palette_color;
    vec4 mip1_dim;       // xyz = MIP1 dimensions
    vec4 mip2_dim;       // xyz = MIP2 dimensions
} pc;

void main() {
    // in_position is [0,1]^3 unit cube. Scale to grid space for DDA.
    vec3 grid_pos = in_position * pc.grid_dim.xyz;
    frag_entry_pos = grid_pos;
    frag_camera_pos = pc.camera_pos.xyz;
    // MVP already includes scale, so use unit position for clip space
    gl_Position = pc.mvp * vec4(in_position, 1.0);
}
