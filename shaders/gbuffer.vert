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

    // NOTE: a previous revision inflated the cube by `eps = 1/grid_dim`
    // to close low-res rasterization gaps, but that caused visible
    // camera warping because `frag_entry_pos` was computed from the
    // uninflated `in_position` while `gl_Position` used the inflated
    // one. The rasterizer interpolated `frag_entry_pos` across the
    // 3%-larger cube footprint, stretching the DDA ray entry
    // positions and producing geometric warping near cube edges.
    // Using the raw unit cube eliminates the mismatch; any
    // remaining rasterization seams can be addressed separately.
    gl_Position = pc.mvp * vec4(in_position, 1.0);
}
