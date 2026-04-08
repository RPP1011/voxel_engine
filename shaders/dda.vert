#version 450

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 frag_ray_origin;
layout(location = 1) out vec3 frag_entry_pos;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 grid_dim;
    vec4 camera_pos;
    vec4 palette_color;
    vec4 mip1_dim;
    vec4 mip2_dim;
} pc;

void main() {
    // Scale unit cube [0,1]^3 to grid space [0, dim]^3 for DDA entry point
    frag_entry_pos = in_position * pc.grid_dim.xyz;
    frag_ray_origin = pc.camera_pos.xyz;
    gl_Position = pc.mvp * vec4(in_position, 1.0);
}
