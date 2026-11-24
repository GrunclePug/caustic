// Pass 3: Exposure, Troy Sobotka AgX Filmic tone mapping, and physics debug overlays.
//
// Struct layouts match `src/engine/post.rs` (`PostUniforms`) and `gpu_tracer.rs` (`RayHit`).

struct PostUniforms {
    params: vec4<f32>, // x = exposure, y = max_steps_scale, z = 0, w = 0
    dims: vec4<u32>,   // x = width, y = height, z = debug_mode (0..3), w = 0
};

struct RayHit {
    r_hit: f32,
    phi_hit: f32,
    redshift_g: f32,
    flags: u32,
};

@group(0) @binding(0) var<uniform> u: PostUniforms;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;
@group(0) @binding(3) var<storage, read> ray_hits: array<RayHit>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle covering NDC [-1, 1] x [-1, 1].
    let x = f32((vid & 1u) << 2u) - 1.0;
    let y = f32((vid & 2u) << 1u) - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * x + 0.5, 0.5 - 0.5 * y);
    return out;
}

// AgX Filmic tone mapping (Troy Sobotka's AgX default curve).
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
    // Troy Sobotka's AgX "Punchy" look (preserves incandescent blackbody hue & contrast).
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma = dot(val, lw);
    val = pow(val, vec3<f32>(1.28));
    val = mix(vec3<f32>(luma), val, 1.32);
    val = clamp(val, vec3<f32>(0.0), vec3<f32>(1.0));
    // Output linear sRGB (surface/offscreen target is Rgba8UnormSrgb which applies sRGB OETF).
    return pow(val, vec3<f32>(2.2));
}

// Turbo-like perceptual heatmap for step counts t in [0, 1].
fn turbo_colormap(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let debug_mode = u.dims.z;
    if (debug_mode == 0u) {
        let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb * u.params.x;
        return vec4<f32>(agx_tonemap(hdr), 1.0);
    }

    let px = min(u32(in.uv.x * f32(u.dims.x)), u.dims.x - 1u);
    let py = min(u32(in.uv.y * f32(u.dims.y)), u.dims.y - 1u);
    let hit = ray_hits[py * u.dims.x + px];
    let status = hit.flags & 3u;
    let order = (hit.flags >> 2u) & 3u;
    let steps = hit.flags >> 4u;

    if (debug_mode == 1u) {
        // Step-count heatmap
        let norm = f32(steps) / max(u.params.y, 1.0);
        return vec4<f32>(turbo_colormap(norm), 1.0);
    } else if (debug_mode == 2u) {
        // Relativistic redshift g-factor: g = 1 white, g > 1 blue, g < 1 red.
        if (status == 0u || status == 3u) {
            return vec4<f32>(0.02, 0.02, 0.02, 1.0);
        }
        let lg = clamp(log2(max(hit.redshift_g, 0.05)), -1.5, 1.5) / 1.5;
        let col = select(
            mix(vec3<f32>(0.9, 0.9, 0.9), vec3<f32>(0.1, 0.45, 1.0), lg),
            mix(vec3<f32>(0.9, 0.9, 0.9), vec3<f32>(1.0, 0.15, 0.05), -lg),
            lg < 0.0,
        );
        return vec4<f32>(col, 1.0);
    } else {
        // Hit-class map (bright magenta flags any BudgetExhausted pixels immediately).
        if (status == 0u) {
            return vec4<f32>(0.01, 0.01, 0.01, 1.0);
        } else if (status == 1u) {
            if (order == 0u) {
                return vec4<f32>(0.95, 0.65, 0.12, 1.0);
            } else if (order == 1u) {
                return vec4<f32>(0.15, 0.85, 0.95, 1.0);
            } else {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
        } else if (status == 2u) {
            return vec4<f32>(0.04, 0.07, 0.18, 1.0);
        } else {
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }
}
