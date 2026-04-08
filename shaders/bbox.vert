#version 450

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(in_position, 1.0);
}
