#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D gbuf_albedo;   // RT0
layout(set = 0, binding = 1) uniform sampler2D gbuf_normal;   // RT1
layout(set = 0, binding = 2) uniform sampler2D gbuf_material; // RT2

layout(push_constant) uniform PushConstants {
    vec4 sun_dir;       // xyz = normalized direction TO sun, w = intensity
    vec4 sun_color;     // rgb
    vec4 camera_dir;    // xyz = normalized view direction
} pc;

// GGX distribution
float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159 * denom * denom + 0.0001);
}

void main() {
    vec4 albedo_sample = texture(gbuf_albedo, frag_uv);
    vec4 normal_sample = texture(gbuf_normal, frag_uv);
    vec4 material_sample = texture(gbuf_material, frag_uv);

    // Skip pixels with no geometry (albedo alpha = 0)
    if (albedo_sample.a < 0.01) {
        out_color = vec4(0.0);
        return;
    }

    vec3 albedo = albedo_sample.rgb;
    vec3 N = normalize(normal_sample.rgb * 2.0 - 1.0); // decode from [0,1] to [-1,1]
    float roughness = material_sample.r;

    vec3 L = normalize(pc.sun_dir.xyz);
    float intensity = pc.sun_dir.w;
    vec3 sun_col = pc.sun_color.rgb;
    vec3 V = normalize(-pc.camera_dir.xyz);

    // Lambertian diffuse
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * NdotL;

    // GGX specular
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float spec = D_GGX(NdotH, max(roughness, 0.04)) * NdotL;
    vec3 specular = sun_col * spec * 0.04; // F0=0.04 for dielectric

    // Ambient (hemisphere: sky blue above, ground brown below)
    float sky_factor = N.y * 0.5 + 0.5; // 1 for up-facing, 0 for down-facing
    vec3 sky_color = vec3(0.35, 0.40, 0.55);
    vec3 ground_color = vec3(0.20, 0.15, 0.10);
    vec3 ambient = albedo * mix(ground_color, sky_color, sky_factor);

    out_color = vec4((diffuse + specular) * sun_col * intensity + ambient, 1.0);
}
