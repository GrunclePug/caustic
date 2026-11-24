// Shared Pass 3: Exposure + Troy Sobotka AgX Filmic tone mapping (Punchy look)
// for any simulation writing a linear HDR Rgba16Float / Rgba32Float texture.

struct AgxUniforms {
    params: vec4<f32>, // x = exposure, yzw = reserved
};

@group(0) @binding(0) var<uniform> u: AgxUniforms;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) << 2u) - 1.0;
    let y = f32((vid & 2u) << 1u) - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * x + 0.5, 0.5 - 0.5 * y);
    return out;
}

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
}

fn agx_tonemap(linear_rgb: vec3<f32>) -> vec3<f32> {
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>( 1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368,  1.15190312990417,   -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433,  1.15107367264116),
    );

    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var val = agx_mat * max(linear_rgb, vec3<f32>(1e-10));
    val = clamp(log2(val), vec3<f32>(min_ev), vec3<f32>(max_ev));
    val = (val - min_ev) / (max_ev - min_ev);
    val = agx_default_contrast_approx(val);
    val = agx_mat_inv * val;
    val = clamp(val, vec3<f32>(0.0), vec3<f32>(1.0));
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma = dot(val, lw);
    val = pow(val, vec3<f32>(1.28));
    val = mix(vec3<f32>(luma), val, 1.32);
    val = clamp(val, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(val, vec3<f32>(2.2));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb * u.params.x;
    return vec4<f32>(agx_tonemap(hdr), 1.0);
}
