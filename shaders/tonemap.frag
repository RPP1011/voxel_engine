#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D hdr_input;
layout(set = 0, binding = 1) uniform sampler2D history;  // previous frame for TAA

layout(push_constant) uniform PushConstants {
    vec4 params; // x = frame_index (0 = no TAA), y = blend_factor, z,w unused
} pc;

// ACES tone mapping (simplified fit)
vec3 aces_tonemap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdr_input, frag_uv).rgb;

    // TAA: blend with history
    float frame = pc.params.x;
    if (frame > 0.5) {
        vec3 hist = texture(history, frag_uv).rgb;
        float blend = pc.params.y;
        hdr = mix(hist, hdr, blend);
    }

    // Tone map HDR → LDR
    vec3 ldr = aces_tonemap(hdr);

    // Gamma correction (linear → sRGB approximation)
    ldr = pow(ldr, vec3(1.0 / 2.2));

    out_color = vec4(ldr, 1.0);
}
