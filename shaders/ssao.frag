#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out float out_ao;

layout(set = 0, binding = 0) uniform sampler2D gbuf_depth;   // RT4: linear depth (R32F)
layout(set = 0, binding = 1) uniform sampler2D gbuf_normal;  // RT1: world normal (RGBA16F)

layout(push_constant) uniform PushConstants {
    vec4 params; // x = radius, y = bias, z = intensity, w = resolution
} pc;

// Simple HBAO-like: sample depth in a hemisphere around the pixel
// and compute occlusion based on depth differences.
void main() {
    float center_depth = texture(gbuf_depth, frag_uv).r;
    vec3 center_normal = texture(gbuf_normal, frag_uv).rgb * 2.0 - 1.0;

    if (center_depth <= 0.001) {
        out_ao = 1.0; // No geometry
        return;
    }

    float radius = pc.params.x;
    float bias = pc.params.y;
    float intensity = pc.params.z;
    float resolution = pc.params.w;

    float texel = 1.0 / resolution;
    float occlusion = 0.0;
    float total_weight = 0.0;

    // 8-tap sampling pattern
    const vec2 offsets[8] = vec2[](
        vec2( 1.0,  0.0), vec2(-1.0,  0.0),
        vec2( 0.0,  1.0), vec2( 0.0, -1.0),
        vec2( 0.7,  0.7), vec2(-0.7,  0.7),
        vec2( 0.7, -0.7), vec2(-0.7, -0.7)
    );

    for (int i = 0; i < 8; i++) {
        vec2 sample_uv = frag_uv + offsets[i] * radius * texel;
        float sample_depth = texture(gbuf_depth, sample_uv).r;

        if (sample_depth <= 0.001) continue;

        float depth_diff = center_depth - sample_depth;

        // If the sample is closer to the camera (smaller depth),
        // it might be occluding us
        if (depth_diff > bias) {
            float range_check = smoothstep(0.0, 1.0, radius / abs(depth_diff));
            occlusion += range_check;
        }
        total_weight += 1.0;
    }

    if (total_weight > 0.0) {
        occlusion = occlusion / total_weight;
    }

    out_ao = clamp(1.0 - occlusion * intensity, 0.0, 1.0);
}
