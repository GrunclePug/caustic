// Pass 2: Stationary Page-Thorne thin accretion disk + CIE log-XYZ blackbody LUT
// + neighbor-Jacobian-filtered procedural sky -> linear HDR Rgba16Float target.
//
// Struct layouts match `src/sim/blackhole/gpu_tracer.rs` (`ShadeUniforms`, `RayHit`).

struct ShadeUniforms {
    disk_params: vec4<f32>, // x = r_isco, y = disk_outer, z = t_peak_k, w = exposure_scale
    lut_params: vec4<f32>,  // x = ln(T_min), y = ln(T_max), z = T_min, w = sky_intensity
    dims: vec4<u32>,        // x = width, y = height, z = disk_lut_len, w = bb_lut_len
};

struct RayHit {
    r_hit: f32,
    phi_hit: f32,
    redshift_g: f32,
    flags: u32,
};

@group(0) @binding(0) var<uniform> u: ShadeUniforms;
@group(0) @binding(1) var<storage, read> ray_hits: array<RayHit>;
@group(0) @binding(2) var<storage, read> ray_skies: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> disk_temp_lut: array<f32>;
@group(0) @binding(4) var<storage, read> blackbody_lut: array<vec4<f32>>;
@group(0) @binding(5) var hdr_out: texture_storage_2d<rgba16float, write>;

const STATUS_CAPTURED: u32 = 0u;
const STATUS_DISK_HIT: u32 = 1u;
const STATUS_ESCAPED: u32 = 2u;

fn hit_status(flags: u32) -> u32 {
    return flags & 3u;
}

fn sample_disk_temp_norm(r_hit: f32) -> f32 {
    let r_in = u.disk_params.x;
    let r_out = u.disk_params.y;
    let t = clamp((r_hit - r_in) / max(r_out - r_in, 1e-6), 0.0, 1.0);
    // LUT is sampled uniformly in s = sqrt((r - r_ISCO) / (r_out - r_ISCO)).
    let s = sqrt(t);
    let max_idx = f32(u.dims.z - 1u);
    let pos = s * max_idx;
    let i0 = min(u32(pos), u.dims.z - 2u);
    let frac = pos - f32(i0);
    return mix(disk_temp_lut[i0], disk_temp_lut[i0 + 1u], frac);
}

// Convert strictly positive CIE XYZ to linear Rec.709 RGB, gently clipping out-of-gamut
// low-temperature Planckian colors while preserving luminance.
fn xyz_to_rec709(xyz: vec3<f32>) -> vec3<f32> {
    let rgb = vec3<f32>(
         3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
         0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
    );
    let min_c = min(rgb.r, min(rgb.g, rgb.b));
    if (min_c < 0.0) {
        // Desaturate softly toward D65 luminance Y to stay in-gamut without negative channels.
        let y = max(xyz.y, 0.0);
        let alpha = y / max(y - min_c, 1e-6);
        return max(mix(vec3<f32>(y), rgb, clamp(alpha, 0.0, 1.0)), vec3<f32>(0.0));
    }
    return rgb;
}

// Evaluate absolute CIE blackbody radiance at T_obs = g * T_emit (NO extra g^4 multiplier!).
// Exposure is applied in log space before exp() so Rgba16Float never overflows at high T_obs.
fn eval_blackbody_rgb(t_obs: f32, log_exposure: f32) -> vec3<f32> {
    let t_min = u.lut_params.z;
    if (t_obs < t_min) {
        return vec3<f32>(0.0);
    }
    let ln_min = u.lut_params.x;
    let ln_max = u.lut_params.y;
    let ln_t = clamp(log(t_obs), ln_min, ln_max);
    let frac_u = (ln_t - ln_min) / max(ln_max - ln_min, 1e-6);
    let max_idx = f32(u.dims.w - 1u);
    let pos = frac_u * max_idx;
    let i0 = min(u32(pos), u.dims.w - 2u);
    let frac = pos - f32(i0);
    let ln_xyz = mix(blackbody_lut[i0].xyz, blackbody_lut[i0 + 1u].xyz, frac);
    // Clamp exponent below f16 overflow (~ln(30000) ~ 10.3) after exposure scaling.
    let xyz = exp(min(ln_xyz + vec3<f32>(log_exposure), vec3<f32>(10.2)));
    return xyz_to_rec709(xyz);
}

// Deterministic PCG hash for procedural celestial sphere stars.
fn hash3(p: vec3<u32>) -> vec3<f32> {
    var v = p * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return vec3<f32>(v & vec3<u32>(0x00FFFFFFu)) * (1.0 / 16777216.0);
}

// Procedural starfield surface brightness filtered by angular pixel footprint `foot_rad`.
fn sample_filtered_sky(dir: vec3<f32>, foot_rad: f32, sky_g: f32) -> vec3<f32> {
    let d = normalize(dir);
    // Subtle deep-space background gradient along an inclined galactic band.
    let gal_axis = normalize(vec3<f32>(0.25, 0.35, 0.9));
    let b_lat = abs(dot(d, gal_axis));
    let band = exp(-b_lat * b_lat * 18.0);
    var col = vec3<f32>(0.0025, 0.0032, 0.0055) + band * vec3<f32>(0.012, 0.014, 0.022);

    // Two layers of angularly filtered stars on a cube-projected grid.
    let abs_d = abs(d);
    var face = 0u;
    var uv = vec2<f32>(0.0);
    if (abs_d.x >= abs_d.y && abs_d.x >= abs_d.z) {
        face = select(1u, 0u, d.x >= 0.0);
        uv = vec2<f32>(d.y, d.z) / max(abs_d.x, 1e-6);
    } else if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
        face = select(3u, 2u, d.y >= 0.0);
        uv = vec2<f32>(d.x, d.z) / max(abs_d.y, 1e-6);
    } else {
        face = select(5u, 4u, d.z >= 0.0);
        uv = vec2<f32>(d.x, d.y) / max(abs_d.z, 1e-6);
    }

    // Filter width in grid cells (larger Jacobian footprint smooths high-order Einstein arcs).
    let foot_clamped = clamp(foot_rad, 0.0008, 0.08);

    // Layer 0 (brighter sparse stars) and Layer 1 (fainter dense stars).
    for (var layer = 0u; layer < 2u; layer = layer + 1u) {
        let grid_n = select(110.0, 220.0, layer == 1u);
        let p_grid = (uv * 0.5 + 0.5) * grid_n;
        let cell = vec2<i32>(floor(p_grid));
        let f_uv = fract(p_grid);
        let sigma_cell = max(0.06, foot_clamped * grid_n * 0.35);
        let inv_2sig2 = 1.0 / (2.0 * sigma_cell * sigma_cell);
        // Energy-conserving peak attenuation as footprint grows.
        let energy_norm = (0.06 * 0.06) / (sigma_cell * sigma_cell);

        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = cell + vec2<i32>(dx, dy);
                let h = hash3(vec3<u32>(
                    u32(nc.x + 1024) + layer * 4096u,
                    u32(nc.y + 1024),
                    face * 97u + layer * 13u,
                ));
                let thresh = select(0.935, 0.90, layer == 1u);
                if (h.x > thresh) {
                    let center = vec2<f32>(f32(dx), f32(dy)) + vec2<f32>(0.15 + 0.7 * h.y, 0.15 + 0.7 * h.z);
                    let diff = f_uv - center;
                    let r2 = dot(diff, diff);
                    let mag = pow((h.x - thresh) / (1.0 - thresh), 2.2);
                    // Assign physical stellar temperature (3200 K - 18000 K), shifted by observer sky_g.
                    let t_star = mix(3200.0, 18000.0, h.y) * max(sky_g, 0.2);
                    let star_rgb = eval_blackbody_rgb(t_star, -3.2);
                    let lum = max(dot(star_rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-6);
                    let star_chroma = star_rgb / lum;
                    let intensity = select(1.8, 0.7, layer == 1u) * (0.3 + 2.5 * mag) * energy_norm;
                    col = col + star_chroma * (intensity * exp(-r2 * inv_2sig2));
                }
            }
        }
    }

    // Modulate sky surface brightness gently by observer gravitational shift.
    let g_mod = clamp(sky_g, 0.25, 4.0);
    return col * (u.lut_params.w * g_mod * g_mod);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u.dims.x;
    let height = u.dims.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let idx = gid.y * width + gid.x;
    let hit = ray_hits[idx];
    let status = hit_status(hit.flags);

    var hdr = vec3<f32>(0.0);

    if (status == STATUS_DISK_HIT) {
        let t_norm = sample_disk_temp_norm(hit.r_hit);
        let t_emit = u.disk_params.z * t_norm;
        let t_obs = hit.redshift_g * t_emit;
        let log_exp = log(max(u.disk_params.w, 1e-6));
        hdr = eval_blackbody_rgb(t_obs, log_exp);
    } else if (status == STATUS_ESCAPED) {
        let sky = ray_skies[idx];
        let dir = sky.xyz;

        // Estimate sky angular footprint from Escaped neighbor finite differences,
        // strictly masking out non-Escaped neighbors at shadow/disk boundaries.
        var foot = 0.002;
        var accum = 0.0;
        var count = 0.0;
        if (gid.x + 1u < width) {
            let n_idx = idx + 1u;
            if (hit_status(ray_hits[n_idx].flags) == STATUS_ESCAPED) {
                accum = accum + length(ray_skies[n_idx].xyz - dir);
                count = count + 1.0;
            }
        }
        if (gid.x > 0u) {
            let n_idx = idx - 1u;
            if (hit_status(ray_hits[n_idx].flags) == STATUS_ESCAPED) {
                accum = accum + length(ray_skies[n_idx].xyz - dir);
                count = count + 1.0;
            }
        }
        if (gid.y + 1u < height) {
            let n_idx = idx + width;
            if (hit_status(ray_hits[n_idx].flags) == STATUS_ESCAPED) {
                accum = accum + length(ray_skies[n_idx].xyz - dir);
                count = count + 1.0;
            }
        }
        if (gid.y > 0u) {
            let n_idx = idx - width;
            if (hit_status(ray_hits[n_idx].flags) == STATUS_ESCAPED) {
                accum = accum + length(ray_skies[n_idx].xyz - dir);
                count = count + 1.0;
            }
        }
        if (count > 0.0) {
            foot = accum / count;
        }
        hdr = sample_filtered_sky(dir, foot, sky.w);
    } else if (status == STATUS_CAPTURED) {
        hdr = vec3<f32>(0.0);
    }

    textureStore(hdr_out, vec2<i32>(gid.xy), vec4<f32>(hdr, 1.0));
}
