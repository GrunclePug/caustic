// Pass 1: Real-time forward spectral photon simulation supporting:
// - Multiple simultaneous physical emitters (Beam, Point, Spotlight, Star Disk) with
//   White, Planck Blackbody (K), or Monochromatic Laser (nm) spectra
// - PHYS_OPTICS (0): Dispersive glass (Cauchy/Snell/Fresnel/TIR), mirrors, 50/50 beam splitters,
//   diffuse Lambertian surfaces/planets with spectral albedo, and Rayleigh (1/lambda^4) atmospheres
// - PHYS_KERR (1): Equatorial Ingoing Cartesian Kerr-Schild curved spacetime
// - PHYS_GRIN_LUNEBURG (2), PHYS_GRIN_BH (3), PHYS_GRIN_MIRAGE (4): Continuous GRIN eikonal RK4

struct SimUniforms {
    // xy = view_center, z = view_half_height, w = aspect (width / height)
    view: vec4<f32>,
    // x = total_power, yzw = reserved
    emitter: vec4<f32>,
    // x = dt_substep, y = spin a, z = r_plus, w = fixed_scale
    physics: vec4<f32>,
    // x = splat_stride, y = grin_strength, z = dispersion_scale, w = reserved
    extra: vec4<f32>,
    // Optional rogue Schwarzschild black hole in Space: xy = pos, z = geometric mass M (<=0 disables), w = reserved
    bh: vec4<f32>,
    // x = width, y = height, z = active_photons, w = substeps
    dims: vec4<u32>,
    // x = physics_kind, y = mode (0=Accumulate, 1=Pulse), z = num_emitters, w = pulse_trigger
    flags: vec4<u32>,
    // x = num_segments, y = num_circles, z = frame_idx, w = seed
    counts: vec4<u32>,
};

struct GpuEmitter {
    geom: vec4<f32>, // xy = pos, z = angle, w = spread/radius
    spec: vec4<f32>, // x = spec_param (temp_k or lambda_nm), y = power, z = cdf_hi, w = reserved
    info: vec4<u32>, // x = emitter_kind, y = spectrum_kind, zw = reserved
};

// 32-byte Photon state record (used persistently in MODE_PULSE).
struct Photon {
    pos: vec2<f32>,
    // Optics: unit direction d. Kerr: covariant spatial momentum. GRIN: eikonal p (|p| = n).
    p: vec2<f32>,
    // Optics & GRIN: wavelength in nm [380..780]. Kerr: emitted local ZAMO wavelength (nu_emit = 1).
    lambda_nm: f32,
    // Optics: current medium IOR n_cur (or atmo_density when in_atmo bit set). Kerr: energy E = -p_t.
    energy: f32,
    rng: u32,
    // bit 0: active, bit 1: in_atmo, bits 2..6: emitter_idx (0..15), bits 6..32: bounces/age
    flags: u32,
};

struct OptSegment {
    endpoints: vec4<f32>, // xy = p0, zw = p1
    material: vec4<f32>,  // Glass: (a, b). Diffuse/Splitter: (albedo_r, g, b, prob_refl)
    info: vec4<u32>,      // x = mat_kind, y = subkind (1 = icy diffuse ring on splitter)
};

struct OptCircle {
    geom: vec4<f32>,     // xy = center, z = radius, w = clip_radius
    clip_arc: vec4<f32>, // xy = clip_center, z = arc_cos_half, w = arc_angle_center
    material: vec4<f32>, // Glass: (a, b). Diffuse: (r, g, b). Atmo: (density, n_delta, r_planet)
    info: vec4<u32>,     // x = mat_kind, y = invert_normal
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(2) var<storage, read_write> accum_xyz: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> spectral_xyz_lut: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> segments: array<OptSegment>;
@group(0) @binding(5) var<storage, read> circles: array<OptCircle>;
@group(0) @binding(6) var<storage, read> emitters: array<GpuEmitter>;

const PHYS_OPTICS: u32 = 0u;
const PHYS_KERR: u32 = 1u;
const PHYS_GRIN_LUNEBURG: u32 = 2u;
const PHYS_GRIN_BH: u32 = 3u;
const PHYS_GRIN_MIRAGE: u32 = 4u;

const MAT_MIRROR: u32 = 1u;
const MAT_GLASS: u32 = 2u;
const MAT_ABSORBER: u32 = 3u;
const MAT_DIFFUSE: u32 = 4u;
const MAT_ATMOSPHERE: u32 = 5u;
const MAT_SPLITTER: u32 = 6u;

const EMITTER_BEAM: u32 = 0u;
const EMITTER_POINT: u32 = 1u;
const EMITTER_SPOT: u32 = 2u;
const EMITTER_DISK: u32 = 3u;
const EMITTER_SOLAR_FAR: u32 = 4u;

const SPEC_WHITE_FLAT: u32 = 0u;
const SPEC_BLACKBODY: u32 = 1u;
const SPEC_LASER: u32 = 2u;

const MODE_ACCUMULATE: u32 = 0u;

const LAMBDA_MIN: f32 = 380.0;
const LAMBDA_MAX: f32 = 780.0;
const TAU: f32 = 6.283185307;

const LUNEBURG_R: f32 = 3.2;
const OPT_BH_BC_BASE: f32 = 2.4;
const OPT_BH_CORE_R: f32 = 0.72;
const MIRAGE_Y_GROUND: f32 = -2.2;
const MIRAGE_SCALE_H: f32 = 1.35;

fn hash_u32(x: u32) -> u32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

// Pure value-based PCG-RXS-M-XS-32 step (zero `ptr<function, ...>` args to avoid driver alias bugs).
fn pcg_step(state: u32) -> u32 {
    return state * 747796405u + 2891336453u;
}

fn pcg_f32(state: u32) -> f32 {
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let out = (word >> 22u) ^ word;
    return f32(out & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

fn sample_spectral_xyz(lambda_nm: f32) -> vec3<f32> {
    if (lambda_nm < LAMBDA_MIN || lambda_nm > LAMBDA_MAX) {
        return vec3<f32>(0.0);
    }
    let t = (lambda_nm - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN);
    let pos = clamp(t, 0.0, 1.0) * 255.0;
    let i0 = min(u32(pos), 254u);
    let frac = pos - f32(i0);
    return mix(spectral_xyz_lut[i0].xyz, spectral_xyz_lut[i0 + 1u].xyz, frac);
}

// Exact Planck spectral radiance in nm, normalized over [380, 780] nm.
fn planck_raw(lambda_nm: f32, temp_k: f32) -> f32 {
    let lam = clamp(lambda_nm, 380.0, 780.0);
    let t = clamp(temp_k, 1500.0, 25000.0);
    let x = clamp(1.4387769e7 / (lam * t), 1e-4, 60.0);
    let inv = 550.0 / lam;
    let p5 = inv * inv * inv * inv * inv;
    return p5 / max(exp(x) - 1.0, 1e-6);
}

fn planck_visible_weight(lambda_nm: f32, temp_k: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let l = 405.0 + f32(i) * 50.0;
        sum = sum + planck_raw(l, temp_k);
    }
    let avg = max(sum * 0.125, 1e-20);
    return planck_raw(lambda_nm, temp_k) / avg;
}

fn emitter_spectral_weight(em_idx: u32, lambda_nm: f32) -> f32 {
    let em = emitters[em_idx];
    let sk = em.info.y;
    if (sk == SPEC_BLACKBODY) {
        return planck_visible_weight(lambda_nm, em.spec.x);
    }
    if (sk == SPEC_LASER) {
        return 0.55;
    }
    return 1.0;
}

// Smooth physical spectral reflectance R(lambda) in [0, 0.96] from RGB albedo weights.
fn spectral_albedo(lambda_nm: f32, albedo: vec3<f32>) -> f32 {
    let db = (lambda_nm - 450.0) / 55.0;
    let dg = (lambda_nm - 545.0) / 55.0;
    let dr = (lambda_nm - 635.0) / 65.0;
    let wb = exp(-0.5 * db * db);
    let wg = exp(-0.5 * dg * dg);
    let wr = exp(-0.5 * dr * dr);
    let denom = max(wb + wg + wr, 1e-5);
    return clamp((albedo.r * wr + albedo.g * wg + albedo.b * wb) / denom, 0.0, 0.96);
}

fn rayleigh_sigma(lambda_nm: f32, base_density: f32) -> f32 {
    let ratio = 550.0 / clamp(lambda_nm, 380.0, 780.0);
    let r2 = ratio * ratio;
    return max(base_density, 0.0) * r2 * r2;
}

fn cauchy_ior(lambda_nm: f32, a: f32, b: f32) -> f32 {
    let lam_um = lambda_nm * 1e-3;
    return a + b / (lam_um * lam_um);
}

fn world_to_pixel(world: vec2<f32>) -> vec2<f32> {
    let half_h = u.view.z;
    let half_w = half_h * u.view.w;
    let rel = world - u.view.xy;
    let u_px = (rel.x / half_w * 0.5 + 0.5) * f32(u.dims.x);
    let v_px = (0.5 - rel.y / half_h * 0.5) * f32(u.dims.y);
    return vec2<f32>(u_px, v_px);
}

fn splat_pixel_xyz(ix: i32, iy: i32, xyz: vec3<f32>) {
    let w = i32(u.dims.x);
    let h = i32(u.dims.y);
    if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
        return;
    }
    let scale = u.physics.w;
    let qx = u32(max(xyz.x * scale + 0.5, 0.0));
    let qy = u32(max(xyz.y * scale + 0.5, 0.0));
    let qz = u32(max(xyz.z * scale + 0.5, 0.0));
    let base = (u32(iy) * u.dims.x + u32(ix)) * 3u;
    if (qx > 0u) {
        atomicAdd(&accum_xyz[base], qx);
    }
    if (qy > 0u) {
        atomicAdd(&accum_xyz[base + 1u], qy);
    }
    if (qz > 0u) {
        atomicAdd(&accum_xyz[base + 2u], qz);
    }
}

fn splat_segment_xyz(w0: vec2<f32>, w1: vec2<f32>, xyz: vec3<f32>, density_w: f32, jitter: f32) {
    let s0 = world_to_pixel(w0);
    let s1 = world_to_pixel(w1);
    let d = s1 - s0;
    let fw = f32(u.dims.x);
    let fh = f32(u.dims.y);

    var t0 = 0.0;
    var t1 = 1.0;

    if (abs(d.x) < 1e-6) {
        if (s0.x < 0.0 || s0.x >= fw) {
            return;
        }
    } else {
        let inv_dx = 1.0 / d.x;
        let tx0 = (0.0 - s0.x) * inv_dx;
        let tx1 = (fw - s0.x) * inv_dx;
        t0 = max(t0, min(tx0, tx1));
        t1 = min(t1, max(tx0, tx1));
    }

    if (abs(d.y) < 1e-6) {
        if (s0.y < 0.0 || s0.y >= fh) {
            return;
        }
    } else {
        let inv_dy = 1.0 / d.y;
        let ty0 = (0.0 - s0.y) * inv_dy;
        let ty1 = (fh - s0.y) * inv_dy;
        t0 = max(t0, min(ty0, ty1));
        t1 = min(t1, max(ty0, ty1));
    }

    if (t0 > t1) {
        return;
    }

    let c0 = s0 + d * t0;
    let c1 = s0 + d * t1;
    let dist_px = length(c1 - c0);
    if (dist_px < 1e-5) {
        return;
    }

    let stride = max(u.extra.x, 1.0);
    let n = clamp(u32(ceil(dist_px / stride)), 1u, 2048u);
    let step_px = dist_px / f32(n);
    let per_sample = xyz * (density_w * step_px);
    let step_vec = (c1 - c0) / f32(n);
    var cur = c0 + step_vec * jitter;
    for (var i = 0u; i < n; i = i + 1u) {
        splat_pixel_xyz(i32(floor(cur.x)), i32(floor(cur.y)), per_sample);
        cur = cur + step_vec;
    }
}

// ---------- Continuous Graded-Index (GRIN) Eikonal Optics ----------

struct GrinEval {
    n2: f32,
    half_grad_n2: vec2<f32>,
};

fn grin_chromatic_offset(lambda_nm: f32) -> f32 {
    let lam_um = clamp(lambda_nm * 1e-3, 0.35, 0.85);
    return 0.3025 / (lam_um * lam_um) - 1.0;
}

fn eval_grin(phys: u32, pos: vec2<f32>, lambda_nm: f32) -> GrinEval {
    let strength = max(u.extra.y, 0.05);
    let disp = max(u.extra.z, 0.0);
    let chi = grin_chromatic_offset(lambda_nm);

    if (phys == PHYS_GRIN_LUNEBURG) {
        let r2 = dot(pos, pos);
        let big_r2 = LUNEBURG_R * LUNEBURG_R;
        if (r2 >= big_r2) {
            return GrinEval(1.0, vec2<f32>(0.0));
        }
        let c = max(strength * (1.0 + 0.16 * disp * chi), 0.05);
        let n2 = 1.0 + c * (1.0 - r2 / big_r2);
        return GrinEval(n2, -(c / big_r2) * pos);
    }

    if (phys == PHYS_GRIN_BH) {
        let bc0 = OPT_BH_BC_BASE * sqrt(strength);
        let bc = max(bc0 * (1.0 + 0.24 * disp * chi), 0.2);
        let bc2 = bc * bc;
        let r2 = max(dot(pos, pos), 1e-6);
        let n2 = 1.0 + bc2 / r2;
        return GrinEval(n2, -(bc2 / (r2 * r2)) * pos);
    }

    let n_air = max(1.08 + 0.045 * disp * chi, 1.01);
    let amp = clamp(0.30 * strength * (1.0 + 0.22 * disp * chi), 0.02, n_air * n_air - 0.25);
    let dy = max(pos.y - MIRAGE_Y_GROUND, 0.0);
    let ex = exp(-dy / MIRAGE_SCALE_H);
    let n2 = n_air * n_air - amp * ex;
    return GrinEval(n2, vec2<f32>(0.0, (amp / (2.0 * MIRAGE_SCALE_H)) * ex));
}

fn grin_absorbed(phys: u32, pos: vec2<f32>) -> bool {
    if (phys == PHYS_GRIN_BH) {
        return dot(pos, pos) <= OPT_BH_CORE_R * OPT_BH_CORE_R;
    }
    if (phys == PHYS_GRIN_MIRAGE) {
        return pos.y <= MIRAGE_Y_GROUND;
    }
    return false;
}

// ---------- Exact Isotropic Schwarzschild Gravitational Lensing (for Rogue Black Hole in Space) ----------
// In isotropic coordinates, null geodesics satisfy Fermat's principle with exact optical index
// n(r) = (1 + u)^3 / (1 - u), where u = M / (2r).
// Event horizon: r_H = 0.5 M | Photon sphere: r_ph = (2 + sqrt(3))/2 M = 1.8660 M | b_crit = 3*sqrt(3)*M.

fn bh_n(r: f32, m: f32) -> f32 {
    let u_val = min(0.5 * m / max(r, 0.54 * m), 0.92);
    let op = 1.0 + u_val;
    return (op * op * op) / (1.0 - u_val);
}

fn bh_half_grad_n2(rel: vec2<f32>, m: f32) -> vec2<f32> {
    let r2 = max(dot(rel, rel), (0.54 * m) * (0.54 * m));
    let r = sqrt(r2);
    let u_val = min(0.5 * m / r, 0.92);
    let op = 1.0 + u_val;
    let om = 1.0 - u_val;
    let op2 = op * op;
    let op5 = op2 * op2 * op;
    let om3 = om * om * om;
    let coeff = -2.0 * u_val * op5 * (2.0 - u_val) / (om3 * r2);
    return rel * coeff;
}

// One RK4 step of dx/dl = p, dp/dl = 0.5 * grad(n^2), returning xy = next pos, zw = unit next dir.
fn bh_rk4_step(pos: vec2<f32>, dir: vec2<f32>, bh_pos: vec2<f32>, m: f32, h: f32) -> vec4<f32> {
    let r0 = length(pos - bh_pos);
    let p0 = dir * bh_n(r0, m);

    let k1x = p0;
    let k1p = bh_half_grad_n2(pos - bh_pos, m);

    let pos2 = pos + 0.5 * h * k1x;
    let k2x = p0 + 0.5 * h * k1p;
    let k2p = bh_half_grad_n2(pos2 - bh_pos, m);

    let pos3 = pos + 0.5 * h * k2x;
    let k3x = p0 + 0.5 * h * k2p;
    let k3p = bh_half_grad_n2(pos3 - bh_pos, m);

    let pos4 = pos + h * k3x;
    let k4x = p0 + h * k3p;
    let k4p = bh_half_grad_n2(pos4 - bh_pos, m);

    let next_pos = pos + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    let next_p = p0 + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
    let plen = max(length(next_p), 1e-8);
    return vec4<f32>(next_pos, next_p / plen);
}

fn grin_rk4_step(phys: u32, pos0: vec2<f32>, p0: vec2<f32>, lambda_nm: f32, h: f32) -> vec4<f32> {
    let g1 = eval_grin(phys, pos0, lambda_nm).half_grad_n2;
    let p2 = p0 + 0.5 * h * g1;
    let g2 = eval_grin(phys, pos0 + 0.5 * h * p0, lambda_nm).half_grad_n2;
    let p3 = p0 + 0.5 * h * g2;
    let g3 = eval_grin(phys, pos0 + 0.5 * h * p2, lambda_nm).half_grad_n2;
    let p4 = p0 + h * g3;
    let g4 = eval_grin(phys, pos0 + h * p3, lambda_nm).half_grad_n2;

    let pos1 = pos0 + (h / 6.0) * (p0 + 2.0 * p2 + 2.0 * p3 + p4);
    let p1 = p0 + (h / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4);
    return vec4<f32>(pos1.x, pos1.y, p1.x, p1.y);
}

// ---------- Ingoing Cartesian Kerr-Schild Equatorial (z = 0) Physics ----------

fn ks_r(pos: vec2<f32>, a2: f32) -> f32 {
    return sqrt(max(dot(pos, pos) - a2, 1e-8));
}

struct KerrRhs {
    dx: vec2<f32>,
    dp: vec2<f32>,
    dt: f32,
};

fn ingoing_ks_rhs(pos: vec2<f32>, p: vec2<f32>, energy: f32, a: f32, a2: f32) -> KerrRhs {
    let r = ks_r(pos, a2);
    let r2 = r * r;
    let d = r2 + a2;
    let inv_d = 1.0 / d;
    let h = 1.0 / r;
    let l = vec2<f32>((r * pos.x + a * pos.y) * inv_d, (r * pos.y - a * pos.x) * inv_d);

    let k = energy + dot(l, p);
    let two_hk = 2.0 * h * k;
    let dx = p - two_hk * l;
    let dt = energy + two_hk;

    let grad_r = pos * h;
    let grad_h = -pos * (h * h * h);

    let p_dl_fixed = vec2<f32>((r * p.x - a * p.y) * inv_d, (a * p.x + r * p.y) * inv_d);
    let p_dl_dr = (p.x * (pos.x - 2.0 * r * l.x) + p.y * (pos.y - 2.0 * r * l.y)) * inv_d;
    let p_grad_l = p_dl_fixed + p_dl_dr * grad_r;
    let dp = (k * k) * grad_h + two_hk * p_grad_l;
    return KerrRhs(dx, dp, dt);
}

fn equatorial_zamo_nu(pos: vec2<f32>, p: vec2<f32>, energy: f32, a: f32, a2: f32, rp: f32) -> f32 {
    let r = ks_r(pos, a2);
    let rm = 2.0 - rp;
    let delta = max((r - rp) * (r - rm), 1e-5);
    let r2 = r * r;
    let big_a = (r2 + a2) * (r2 + a2) - a2 * delta;
    let alpha = sqrt(max((r2 * delta) / big_a, 1e-6));
    let omega = (2.0 * a * r) / big_a;
    let lz = pos.x * p.y - pos.y * p.x;
    return max((energy - omega * lz) / alpha, 1e-4);
}

fn init_kerr_photon_momentum(pos: vec2<f32>, aim: vec2<f32>, a: f32, a2: f32, rp: f32) -> vec3<f32> {
    let r = ks_r(pos, a2);
    let r2 = r * r;
    let d = r2 + a2;
    let h = 1.0 / r;
    let l = vec2<f32>((r * pos.x + a * pos.y) / d, (r * pos.y - a * pos.x) / d);

    let rm = 2.0 - rp;
    let delta = max((r - rp) * (r - rm), 1e-5);
    let big_a = d * d - a2 * delta;
    let alpha = sqrt(max((r2 * delta) / big_a, 1e-6));
    let omega = (2.0 * a * r) / big_a;

    let e_t = vec3<f32>(1.0, -omega * pos.y, omega * pos.x) / alpha;

    let xi_phi = vec3<f32>(0.0, -pos.y, pos.x);
    let lp_phi = dot(l, xi_phi.yz);
    let g_phiphi = dot(xi_phi.yz, xi_phi.yz) + 2.0 * h * lp_phi * lp_phi;
    let e_phi = xi_phi / sqrt(max(g_phiphi, 1e-10));

    let raw_r = vec3<f32>(0.0, pos.x, pos.y);
    let lr = dot(l, raw_r.yz);
    let lt = e_t.x + dot(l, e_t.yz);
    let lphi = e_phi.x + dot(l, e_phi.yz);

    let dot_rt = dot(raw_r.yz, e_t.yz) + 2.0 * h * lr * lt;
    let dot_rphi = dot(raw_r.yz, e_phi.yz) + 2.0 * h * lr * lphi;
    let er_orth = raw_r + dot_rt * e_t - dot_rphi * e_phi;
    let ler = er_orth.x + dot(l, er_orth.yz);
    let g_rr = -er_orth.x * er_orth.x + dot(er_orth.yz, er_orth.yz) + 2.0 * h * ler * ler;
    let e_r = er_orth / sqrt(max(g_rr, 1e-10));

    let rad_hat = normalize(pos);
    let phi_hat = vec2<f32>(-rad_hat.y, rad_hat.x);
    let c_r = dot(aim, rad_hat);
    let c_phi = dot(aim, phi_hat);

    let p_up = e_t + c_r * e_r + c_phi * e_phi;
    let lp = p_up.x + dot(l, p_up.yz);
    let p_t = -p_up.x + 2.0 * h * lp;
    let p_xy = p_up.yz + (2.0 * h * lp) * l;
    return vec3<f32>(-p_t, p_xy.x, p_xy.y);
}

fn kerr_spectral_xyz(lambda_emit: f32, nu_loc: f32) -> vec4<f32> {
    let g_vis = pow(clamp(nu_loc, 0.35, 10.0), 0.42);
    let lam_loc = clamp(lambda_emit / g_vis, 432.0, 675.0);
    let xyz = 0.5 * sample_spectral_xyz(lam_loc)
        + 0.25 * sample_spectral_xyz(max(lam_loc - 14.0, 410.0))
        + 0.25 * sample_spectral_xyz(min(lam_loc + 14.0, 700.0));
    let energy_boost = clamp(pow(nu_loc, 0.35), 0.65, 2.4);
    return vec4<f32>(xyz, energy_boost);
}

// ---------- Flat-Space Optical & Planetary Intersection Helpers ----------

struct HitInfo {
    t: f32,
    normal: vec2<f32>,
    mat_kind: u32,
    subkind: u32,
    mat: vec4<f32>,
};

fn intersect_scene(pos: vec2<f32>, dir: vec2<f32>, t_max: f32) -> HitInfo {
    var best = HitInfo(t_max + 1.0, vec2<f32>(0.0, 1.0), 0u, 0u, vec4<f32>(1.0, 0.0, 0.0, 0.0));

    let n_seg = u.counts.x;
    for (var i = 0u; i < n_seg; i = i + 1u) {
        let seg = segments[i];
        let a = seg.endpoints.xy;
        let b = seg.endpoints.zw;
        let e = b - a;
        let denom = dir.x * e.y - dir.y * e.x;
        if (abs(denom) < 1e-8) {
            continue;
        }
        let ap = a - pos;
        let t = (ap.x * e.y - ap.y * e.x) / denom;
        let s = (ap.x * dir.y - ap.y * dir.x) / denom;
        if (t > 1e-6 && t < best.t && s >= 0.0 && s <= 1.0) {
            let n = normalize(vec2<f32>(e.y, -e.x));
            best = HitInfo(t, n, seg.info.x, seg.info.y, seg.material);
        }
    }

    let n_cir = u.counts.y;
    for (var i = 0u; i < n_cir; i = i + 1u) {
        let cir = circles[i];
        let center = cir.geom.xy;
        let radius = cir.geom.z;
        let clip_r = cir.geom.w;
        let clip_c = cir.clip_arc.xy;
        let arc_cos = cir.clip_arc.z;
        let arc_ang = cir.clip_arc.w;
        let arc_dir = vec2<f32>(cos(arc_ang), sin(arc_ang));

        let oc = pos - center;
        let b_dot = dot(oc, dir);
        let c_val = dot(oc, oc) - radius * radius;
        let disc = b_dot * b_dot - c_val;
        if (disc <= 0.0) {
            continue;
        }
        let sq = sqrt(disc);
        let roots = vec2<f32>(-b_dot - sq, -b_dot + sq);
        for (var k = 0u; k < 2u; k = k + 1u) {
            let t = select(roots.y, roots.x, k == 0u);
            if (t <= 1e-6 || t >= best.t) {
                continue;
            }
            let hit = pos + dir * t;
            let n_out = (hit - center) / radius;
            if (clip_r > 0.0) {
                let dc = hit - clip_c;
                if (dot(dc, dc) > clip_r * clip_r) {
                    continue;
                }
            }
            if (arc_cos > -1.5) {
                if (dot(n_out, arc_dir) < arc_cos) {
                    continue;
                }
            }
            let n = select(n_out, -n_out, cir.info.y != 0u);
            best = HitInfo(t, n, cir.info.x, 0u, cir.material);
        }
    }

    return best;
}

fn lambertian_reflect(n_in: vec2<f32>, dir: vec2<f32>, u_rand: f32) -> vec2<f32> {
    var n = n_in;
    if (dot(dir, n) > 0.0) {
        n = -n;
    }
    let sin_a = u_rand * 2.0 - 1.0;
    let cos_a = sqrt(max(1.0 - sin_a * sin_a, 0.0));
    let tang = vec2<f32>(-n.y, n.x);
    return normalize(n * cos_a + tang * sin_a);
}

fn rayleigh_scatter_dir(dir: vec2<f32>, u0: f32, u1: f32, u2: f32) -> vec2<f32> {
    // 2D Rayleigh dipole phase function P(theta) proportional to 1 + cos^2(theta).
    var theta = u0 * TAU;
    let c = cos(theta);
    if (u1 > 0.5 * (1.0 + c * c)) {
        theta = u2 * TAU;
    }
    let ct = cos(theta);
    let st = sin(theta);
    return normalize(vec2<f32>(dir.x * ct - dir.y * st, dir.x * st + dir.y * ct));
}

fn spawn_photon(gid: u32, rng_in: u32) -> Photon {
    let phys = u.flags.x;
    let n_em = max(u.flags.z, 1u);

    // Stratified emitter selection across gid proportional to emitter power CDF.
    let u_gid = (f32(gid) + 0.5) / f32(u.dims.z);
    var em_idx = 0u;
    var cdf_lo = 0.0;
    var cdf_hi = 1.0;
    for (var i = 0u; i < n_em; i = i + 1u) {
        let hi = emitters[i].spec.z;
        if (u_gid <= hi || i + 1u == n_em) {
            em_idx = i;
            cdf_hi = hi;
            break;
        }
        cdf_lo = hi;
    }

    let em = emitters[em_idx];
    let local_u = clamp((u_gid - cdf_lo) / max(cdf_hi - cdf_lo, 1e-6), 0.0, 0.999999);
    let base_ang = em.geom.z;
    let spread = em.geom.w;
    let em_kind = em.info.x;
    let spec_kind = em.info.y;

    var rng = pcg_step(rng_in);
    let r0 = pcg_f32(rng);
    rng = pcg_step(rng);
    let r1 = pcg_f32(rng);
    rng = pcg_step(rng);
    let r2 = pcg_f32(rng);

    var pos = em.geom.xy;
    var aim = vec2<f32>(cos(base_ang), sin(base_ang));

    if (em_kind == EMITTER_BEAM) {
        let perp = vec2<f32>(-aim.y, aim.x);
        let lane = (local_u + (r0 - 0.5) / max(f32(u.dims.z) * (cdf_hi - cdf_lo), 1.0)) * 2.0 - 1.0;
        pos = pos + perp * (clamp(lane, -1.0, 1.0) * spread);
    } else if (em_kind == EMITTER_POINT) {
        let ang = local_u * TAU + (r0 - 0.5) * 0.02;
        aim = vec2<f32>(cos(ang), sin(ang));
    } else if (em_kind == EMITTER_SPOT) {
        let ang_off = (local_u * 2.0 - 1.0) * spread + (r0 - 0.5) * 0.004;
        let ang = base_ang + ang_off;
        aim = vec2<f32>(cos(ang), sin(ang));
    } else if (em_kind == EMITTER_SOLAR_FAR) {
        // Frustum importance-sampled distant Sun with exact finite angular half-radius `alpha_sun`.
        let alpha_sun = em.spec.w;
        let d_spawn = bitcast<f32>(em.info.z);
        let base_dir = vec2<f32>(cos(base_ang), sin(base_ang));
        let perp = vec2<f32>(-base_dir.y, base_dir.x);
        let lane_step = (2.0 * spread) / max(f32(u.dims.z) * (cdf_hi - cdf_lo), 1.0);
        let u_lat = (local_u * 2.0 - 1.0) * spread + (r0 - 0.5) * lane_step;
        let ang = base_ang + (r2 * 2.0 - 1.0) * alpha_sun;
        aim = vec2<f32>(cos(ang), sin(ang));
        pos = perp * u_lat - aim * d_spawn;
    } else {
        // EMITTER_DISK: glowing physical star/Sun photosphere casting real umbra & soft penumbra!
        let rim_ang = local_u * TAU + r0 * 0.02;
        let out_n = vec2<f32>(cos(rim_ang), sin(rim_ang));
        pos = pos + out_n * (max(spread, 1e-5) + 1e-5);
        let tang = vec2<f32>(-out_n.y, out_n.x);
        let sin_rel = select(r2 * 2.0 - 1.0, (r2 * 2.0 - 1.0) * 0.35, u.flags.y != MODE_ACCUMULATE);
        let cos_rel = sqrt(max(1.0 - sin_rel * sin_rel, 0.0));
        aim = normalize(out_n * cos_rel + tang * sin_rel);
    }

    let base_flags = 1u | ((em_idx & 15u) << 2u);

    if (phys == PHYS_KERR) {
        let a = u.physics.y;
        let a2 = a * a;
        let rp = u.physics.z;
        if (ks_r(pos, a2) <= rp * 1.15) {
            pos = normalize(pos + vec2<f32>(1e-4, 0.0)) * (rp * 1.3 + abs(a));
        }
        let ep = init_kerr_photon_momentum(pos, aim, a, a2, rp);
        let energy = ep.x;
        let p = ep.yz;
        let lambda_emit = mix(585.0, 610.0, r1);
        return Photon(pos, p, lambda_emit, energy, rng, base_flags);
    }

    var lambda_nm = mix(LAMBDA_MIN, LAMBDA_MAX, r1);
    if (spec_kind == SPEC_LASER) {
        lambda_nm = clamp(em.spec.x + (r1 - 0.5) * 4.0, LAMBDA_MIN, LAMBDA_MAX);
    }

    if (phys >= PHYS_GRIN_LUNEBURG) {
        if (phys == PHYS_GRIN_BH && length(pos) <= OPT_BH_CORE_R * 1.15) {
            pos = normalize(pos + vec2<f32>(1e-4, 0.0)) * (OPT_BH_CORE_R * 1.35);
        }
        if (phys == PHYS_GRIN_MIRAGE && pos.y <= MIRAGE_Y_GROUND + 0.05) {
            pos.y = MIRAGE_Y_GROUND + 0.08;
        }
        let ge = eval_grin(phys, pos, lambda_nm);
        let p = aim * sqrt(max(ge.n2, 0.1));
        return Photon(pos, p, lambda_nm, 1.0, rng, base_flags);
    }

    return Photon(pos, aim, lambda_nm, 1.0, rng, base_flags);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid_v: vec3<u32>) {
    let gid = gid_v.x;
    if (gid >= u.dims.z) {
        return;
    }

    let phys = u.flags.x;
    let mode = u.flags.y;
    let pulse_trigger = u.flags.w != 0u;
    let substeps = u.dims.w;
    let dt_step = u.physics.x;

    let half_h = u.view.z;
    let half_w = half_h * u.view.w;
    let bound_x = half_w * 2.4;
    let bound_y = half_h * 2.4;

    if (mode == MODE_ACCUMULATE) {
        let init_rng = pcg_step(hash_u32(gid ^ hash_u32(u.counts.z * 0x9E3779B9u ^ hash_u32(u.counts.w))));
        var ph = spawn_photon(gid, init_rng);
        var rng = ph.rng;
        let em_idx = (ph.flags >> 2u) & 15u;

        if (phys == PHYS_KERR) {
            let a = u.physics.y;
            let a2 = a * a;
            let rp = u.physics.z;
            let r_cap = rp * 0.96;

            for (var step = 0u; step < 420u; step = step + 1u) {
                let pos0 = ph.pos;
                let p0 = ph.p;
                let energy = ph.energy;
                let r_cur = ks_r(pos0, a2);
                let h = clamp(0.024 * r_cur, 0.012, 0.28);

                let k1 = ingoing_ks_rhs(pos0, p0, energy, a, a2);
                let k2 = ingoing_ks_rhs(pos0 + 0.5 * h * k1.dx, p0 + 0.5 * h * k1.dp, energy, a, a2);
                let k3 = ingoing_ks_rhs(pos0 + 0.5 * h * k2.dx, p0 + 0.5 * h * k2.dp, energy, a, a2);
                let k4 = ingoing_ks_rhs(pos0 + h * k3.dx, p0 + h * k3.dp, energy, a, a2);

                let pos1 = pos0 + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
                let p1 = p0 + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
                if (ks_r(pos1, a2) <= r_cap) {
                    break;
                }

                let nu_loc = equatorial_zamo_nu(pos1, p1, energy, a, a2, rp);
                let col = kerr_spectral_xyz(ph.lambda_nm, nu_loc);
                rng = pcg_step(rng);
                splat_segment_xyz(pos0, pos1, col.xyz, col.w, pcg_f32(rng));

                ph.pos = pos1;
                ph.p = p1;
                let rel = abs(pos1 - u.view.xy);
                if (rel.x > bound_x || rel.y > bound_y) {
                    break;
                }
            }
        } else if (phys >= PHYS_GRIN_LUNEBURG) {
            let spec_w = emitter_spectral_weight(em_idx, ph.lambda_nm);
            let xyz = sample_spectral_xyz(ph.lambda_nm) * spec_w;
            rng = pcg_step(rng);
            let h_jit = mix(0.82, 1.18, pcg_f32(rng));
            for (var step = 0u; step < 440u; step = step + 1u) {
                let pos0 = ph.pos;
                let p0 = ph.p;
                let ge = eval_grin(phys, pos0, ph.lambda_nm);
                let n_cur = sqrt(max(ge.n2, 0.1));
                var h = 0.06;
                if (phys == PHYS_GRIN_LUNEBURG) {
                    h = select(0.14, 0.025, length(pos0) < LUNEBURG_R * 1.12) * h_jit;
                } else if (phys == PHYS_GRIN_BH) {
                    h = clamp(0.028 * length(pos0) / n_cur, 0.004, 0.16);
                } else {
                    h = 0.07;
                }

                let next = grin_rk4_step(phys, pos0, p0, ph.lambda_nm, h);
                let pos1 = next.xy;
                let p1 = next.zw;
                rng = pcg_step(rng);
                if (grin_absorbed(phys, pos1)) {
                    splat_segment_xyz(pos0, pos1, xyz, 1.0, pcg_f32(rng));
                    break;
                }

                splat_segment_xyz(pos0, pos1, xyz, 1.0, pcg_f32(rng));
                ph.pos = pos1;
                ph.p = p1;
                let rel = abs(pos1 - u.view.xy);
                if (rel.x > bound_x || rel.y > bound_y) {
                    break;
                }
            }
        } else {
            // PHYS_OPTICS: full playground (glass, mirrors, beam splitters, diffuse surfaces,
            // planetary Rayleigh atmospheres, absorbers, plus optional Schwarzschild black hole lensing).
            let spec_w = emitter_spectral_weight(em_idx, ph.lambda_nm);
            let xyz = sample_spectral_xyz(ph.lambda_nm) * spec_w;
            let spawn_dist = length(ph.pos - u.view.xy);
            let max_dist = spawn_dist + length(vec2<f32>(bound_x, bound_y)) * 2.5;
            var in_atmo = false;
            var atmo_density = 0.0;

            if (u.bh.z > 0.0) {
                let bh_pos = u.bh.xy;
                let bh_m = u.bh.z;
                let bh_cap = 0.62 * bh_m;
                var bounces = 0u;
                for (var step = 0u; step < 340u; step = step + 1u) {
                    let r_bh = length(ph.pos - bh_pos);
                    if (r_bh <= bh_cap) {
                        break;
                    }
                    let n_cur = bh_n(r_bh, bh_m);
                    // Adaptive affine step keeping spatial arcs crisp near photon sphere and fast far away.
                    let max_dx = clamp(0.036 * r_bh, 0.005 * bh_m, 0.14 * half_h);
                    let h = max_dx / max(n_cur, 1.0);
                    let next = bh_rk4_step(ph.pos, ph.p, bh_pos, bh_m, h);
                    let pos1 = next.xy;
                    let dir1 = next.zw;
                    let chord = pos1 - ph.pos;
                    let chord_len = length(chord);
                    if (chord_len < 1e-7) {
                        break;
                    }
                    let chord_dir = chord / chord_len;
                    let hit = intersect_scene(ph.pos, chord_dir, chord_len);
                    let seg_len = select(chord_len, hit.t, hit.t <= chord_len);

                    var did_scatter = false;
                    if (in_atmo && atmo_density > 0.0) {
                        let sigma = rayleigh_sigma(ph.lambda_nm, atmo_density);
                        rng = pcg_step(rng);
                        let s_scat = -log(max(1.0 - pcg_f32(rng), 1e-6)) / max(sigma, 1e-6);
                        if (s_scat < seg_len) {
                            let scat_pos = ph.pos + chord_dir * s_scat;
                            rng = pcg_step(rng);
                            splat_segment_xyz(ph.pos, scat_pos, xyz, 1.0, pcg_f32(rng));
                            rng = pcg_step(rng);
                            let u0 = pcg_f32(rng);
                            rng = pcg_step(rng);
                            let u1 = pcg_f32(rng);
                            rng = pcg_step(rng);
                            let u2 = pcg_f32(rng);
                            ph.pos = scat_pos;
                            ph.p = rayleigh_scatter_dir(chord_dir, u0, u1, u2);
                            bounces = bounces + 1u;
                            did_scatter = true;
                        }
                    }

                    if (did_scatter) {
                        if (bounces > 24u) {
                            break;
                        }
                    } else if (hit.t <= chord_len) {
                        let hit_pos = ph.pos + chord_dir * hit.t;
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, hit_pos, xyz, 1.0, pcg_f32(rng));
                        bounces = bounces + 1u;
                        if (bounces > 24u || hit.mat_kind == MAT_ABSORBER) {
                            break;
                        } else if (hit.mat_kind == MAT_DIFFUSE) {
                            let refl = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                            rng = pcg_step(rng);
                            if (pcg_f32(rng) < refl) {
                                var n = hit.normal;
                                if (dot(chord_dir, n) > 0.0) {
                                    n = -n;
                                }
                                rng = pcg_step(rng);
                                ph.p = lambertian_reflect(n, chord_dir, pcg_f32(rng));
                                ph.pos = hit_pos + n * 2e-5;
                            } else {
                                break;
                            }
                        } else if (hit.mat_kind == MAT_ATMOSPHERE) {
                            let entering = dot(chord_dir, hit.normal) < 0.0;
                            let n_face = select(-hit.normal, hit.normal, entering);
                            let n_atmo = 1.0 + hit.mat.y;
                            let n1 = select(n_atmo, 1.0, entering);
                            let n2 = select(1.0, n_atmo, entering);
                            let cos_i = clamp(-dot(chord_dir, n_face), 0.0, 1.0);
                            let eta = n1 / n2;
                            let sin2_t = eta * eta * max(1.0 - cos_i * cos_i, 0.0);
                            let atmo_eps = clamp(hit.mat.z * 0.0015, 1e-5, 1e-4);
                            if (sin2_t < 1.0) {
                                let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
                                ph.p = normalize(eta * chord_dir + (eta * cos_i - cos_t) * n_face);
                                ph.pos = hit_pos - n_face * atmo_eps;
                                in_atmo = entering;
                                atmo_density = select(0.0, hit.mat.x, entering);
                            } else {
                                ph.pos = hit_pos + chord_dir * (atmo_eps * 2.0);
                                ph.p = chord_dir;
                                in_atmo = false;
                                atmo_density = 0.0;
                            }
                        } else if (hit.mat_kind == MAT_SPLITTER) {
                            let prob_refl = hit.mat.w;
                            rng = pcg_step(rng);
                            if (pcg_f32(rng) < prob_refl) {
                                var n = hit.normal;
                                if (dot(chord_dir, n) > 0.0) {
                                    n = -n;
                                }
                                let alb = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                                rng = pcg_step(rng);
                                if (pcg_f32(rng) < alb) {
                                    rng = pcg_step(rng);
                                    ph.p = lambertian_reflect(n, chord_dir, pcg_f32(rng));
                                    ph.pos = hit_pos + n * 2e-5;
                                } else {
                                    break;
                                }
                            } else {
                                ph.pos = hit_pos + chord_dir * 5e-5;
                                ph.p = dir1;
                            }
                        } else {
                            ph.pos = hit_pos + chord_dir * 5e-5;
                            ph.p = dir1;
                        }
                    } else {
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, pos1, xyz, 1.0, pcg_f32(rng));
                        ph.pos = pos1;
                        ph.p = dir1;
                    }

                    if (length(ph.pos - bh_pos) <= bh_cap) {
                        break;
                    }
                    let offset = ph.pos - u.view.xy;
                    let rel = abs(offset);
                    if (dot(offset, ph.p) > 0.0 && (rel.x > bound_x || rel.y > bound_y)) {
                        break;
                    }
                }
            } else {
            for (var bounce = 0u; bounce < 28u; bounce = bounce + 1u) {
                let hit = intersect_scene(ph.pos, ph.p, max_dist);
                let seg_len = select(max_dist, hit.t, hit.t <= max_dist);

                // Beer-Lambert + Rayleigh (1 / lambda^4) scattering inside planetary atmospheres!
                var did_scatter = false;
                if (in_atmo && atmo_density > 0.0) {
                    let sigma = rayleigh_sigma(ph.lambda_nm, atmo_density);
                    rng = pcg_step(rng);
                    let s_scat = -log(max(1.0 - pcg_f32(rng), 1e-6)) / max(sigma, 1e-6);
                    if (s_scat < seg_len) {
                        let scat_pos = ph.pos + ph.p * s_scat;
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, scat_pos, xyz, 1.0, pcg_f32(rng));
                        rng = pcg_step(rng);
                        let u0 = pcg_f32(rng);
                        rng = pcg_step(rng);
                        let u1 = pcg_f32(rng);
                        rng = pcg_step(rng);
                        let u2 = pcg_f32(rng);
                        ph.pos = scat_pos;
                        ph.p = rayleigh_scatter_dir(ph.p, u0, u1, u2);
                        did_scatter = true;
                    }
                }

                if (did_scatter) {
                    // Continue directly to next bounce iteration.
                } else if (hit.t <= max_dist) {
                    let hit_pos = ph.pos + ph.p * hit.t;
                    rng = pcg_step(rng);
                    splat_segment_xyz(ph.pos, hit_pos, xyz, 1.0, pcg_f32(rng));

                    if (hit.mat_kind == MAT_ABSORBER) {
                        break;
                    } else if (hit.mat_kind == MAT_MIRROR) {
                        var n = hit.normal;
                        if (dot(ph.p, n) > 0.0) {
                            n = -n;
                        }
                        ph.p = normalize(ph.p - 2.0 * dot(ph.p, n) * n);
                        ph.pos = hit_pos + n * 5e-5;
                    } else if (hit.mat_kind == MAT_GLASS) {
                        let entering = dot(ph.p, hit.normal) < 0.0;
                        let n_face = select(-hit.normal, hit.normal, entering);
                        let n_glass = cauchy_ior(ph.lambda_nm, hit.mat.x, hit.mat.y);
                        let n1 = select(n_glass, 1.0, entering);
                        let n2 = select(1.0, n_glass, entering);
                        let cos_i = clamp(-dot(ph.p, n_face), 0.0, 1.0);
                        let eta = n1 / n2;
                        let sin2_t = eta * eta * max(1.0 - cos_i * cos_i, 0.0);

                        var refl = 1.0;
                        var refr_dir = vec2<f32>(0.0);
                        if (sin2_t < 1.0) {
                            let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
                            let rs = (n1 * cos_i - n2 * cos_t) / max(n1 * cos_i + n2 * cos_t, 1e-6);
                            let rp = (n1 * cos_t - n2 * cos_i) / max(n1 * cos_t + n2 * cos_i, 1e-6);
                            refl = 0.5 * (rs * rs + rp * rp);
                            refr_dir = normalize(eta * ph.p + (eta * cos_i - cos_t) * n_face);
                        }

                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < refl) {
                            ph.p = normalize(ph.p - 2.0 * dot(ph.p, n_face) * n_face);
                            ph.pos = hit_pos + n_face * 5e-5;
                        } else {
                            ph.p = refr_dir;
                            ph.pos = hit_pos - n_face * 5e-5;
                        }
                    } else if (hit.mat_kind == MAT_DIFFUSE) {
                        let refl = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < refl) {
                            var n = hit.normal;
                            if (dot(ph.p, n) > 0.0) {
                                n = -n;
                            }
                            rng = pcg_step(rng);
                            ph.p = lambertian_reflect(n, ph.p, pcg_f32(rng));
                            ph.pos = hit_pos + n * 2e-5;
                        } else {
                            break;
                        }
                    } else if (hit.mat_kind == MAT_ATMOSPHERE) {
                        let entering = dot(ph.p, hit.normal) < 0.0;
                        let n_face = select(-hit.normal, hit.normal, entering);
                        // Gentle physical atmospheric limb refraction (bends grazing sunset light into umbra).
                        let n_atmo = 1.0 + hit.mat.y;
                        let n1 = select(n_atmo, 1.0, entering);
                        let n2 = select(1.0, n_atmo, entering);
                        let cos_i = clamp(-dot(ph.p, n_face), 0.0, 1.0);
                        let eta = n1 / n2;
                        let sin2_t = eta * eta * max(1.0 - cos_i * cos_i, 0.0);
                        let atmo_eps = clamp(hit.mat.z * 0.0015, 1e-5, 1e-4);
                        if (sin2_t < 1.0) {
                            let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
                            ph.p = normalize(eta * ph.p + (eta * cos_i - cos_t) * n_face);
                            ph.pos = hit_pos - n_face * atmo_eps;
                            in_atmo = entering;
                            atmo_density = select(0.0, hit.mat.x, entering);
                        } else {
                            ph.pos = hit_pos + ph.p * (atmo_eps * 2.0);
                            in_atmo = false;
                            atmo_density = 0.0;
                        }
                    } else if (hit.mat_kind == MAT_SPLITTER) {
                        let prob_refl = hit.mat.w;
                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < prob_refl) {
                            var n = hit.normal;
                            if (dot(ph.p, n) > 0.0) {
                                n = -n;
                            }
                            if (hit.subkind == 1u) {
                                // Translucent icy planetary ring: spectral scatter or absorb.
                                let alb = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                                rng = pcg_step(rng);
                                if (pcg_f32(rng) < alb) {
                                    rng = pcg_step(rng);
                                    ph.p = lambertian_reflect(n, ph.p, pcg_f32(rng));
                                    ph.pos = hit_pos + n * 2e-5;
                                } else {
                                    break;
                                }
                            } else {
                                // Clean 50/50 optical beam splitter specular reflection.
                                ph.p = normalize(ph.p - 2.0 * dot(ph.p, n) * n);
                                ph.pos = hit_pos + n * 5e-5;
                            }
                        } else {
                            // Transmit straight through!
                            ph.pos = hit_pos + ph.p * 5e-5;
                        }
                    }

                    let offset = ph.pos - u.view.xy;
                    let rel = abs(offset);
                    if (dot(offset, ph.p) > 0.0 && (rel.x > bound_x || rel.y > bound_y)) {
                        break;
                    }
                } else {
                    let end_pos = ph.pos + ph.p * max_dist;
                    rng = pcg_step(rng);
                    splat_segment_xyz(ph.pos, end_pos, xyz, 1.0, pcg_f32(rng));
                    break;
                }
            }
            }
        }
        return;
    }

    // ---------- MODE_PULSE: Stateful real-time wavefront flight at c / n(lambda) ----------
    var ph = photons[gid];
    var rng = ph.rng;
    if (u.counts.z == 0u || pulse_trigger) {
        let init_rng = pcg_step(hash_u32(gid ^ hash_u32(u.counts.z * 0x9E3779B9u ^ hash_u32(u.counts.w))));
        ph = spawn_photon(gid, init_rng);
        rng = ph.rng;
    }

    if ((ph.flags & 1u) == 0u) {
        ph.rng = rng;
        photons[gid] = ph;
        return;
    }

    let em_idx = (ph.flags >> 2u) & 15u;

    if (phys == PHYS_KERR) {
        let a = u.physics.y;
        let a2 = a * a;
        let rp = u.physics.z;
        let r_cap = rp * 0.96;

        for (var s = 0u; s < substeps; s = s + 1u) {
            if ((ph.flags & 1u) == 0u) {
                break;
            }
            let pos0 = ph.pos;
            let p0 = ph.p;
            let energy = ph.energy;
            let r_cur = ks_r(pos0, a2);

            let k1 = ingoing_ks_rhs(pos0, p0, energy, a, a2);
            let dt_dl = clamp(k1.dt, 0.25, 16.0);
            let h = min(dt_step / dt_dl, clamp(0.025 * r_cur, 0.01, dt_step));

            let k2 = ingoing_ks_rhs(pos0 + 0.5 * h * k1.dx, p0 + 0.5 * h * k1.dp, energy, a, a2);
            let k3 = ingoing_ks_rhs(pos0 + 0.5 * h * k2.dx, p0 + 0.5 * h * k2.dp, energy, a, a2);
            let k4 = ingoing_ks_rhs(pos0 + h * k3.dx, p0 + h * k3.dp, energy, a, a2);

            let pos1 = pos0 + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
            let p1 = p0 + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
            if (ks_r(pos1, a2) <= r_cap) {
                ph.flags = ph.flags & (~1u);
                break;
            }

            let nu_loc = equatorial_zamo_nu(pos1, p1, energy, a, a2, rp);
            let col = kerr_spectral_xyz(ph.lambda_nm, nu_loc);
            rng = pcg_step(rng);
            splat_segment_xyz(pos0, pos1, col.xyz, col.w, pcg_f32(rng));

            ph.pos = pos1;
            ph.p = p1;
            let age = (ph.flags >> 6u) + 1u;
            let rel = abs(pos1 - u.view.xy);
            if (rel.x > bound_x || rel.y > bound_y || age > 4000u) {
                ph.flags = ph.flags & (~1u);
            } else {
                ph.flags = (ph.flags & 63u) | (age << 6u);
            }
        }
    } else if (phys >= PHYS_GRIN_LUNEBURG) {
        let spec_w = emitter_spectral_weight(em_idx, ph.lambda_nm);
        let xyz = sample_spectral_xyz(ph.lambda_nm) * spec_w;
        for (var s = 0u; s < substeps; s = s + 1u) {
            if ((ph.flags & 1u) == 0u) {
                break;
            }
            let pos0 = ph.pos;
            let p0 = ph.p;
            let ge = eval_grin(phys, pos0, ph.lambda_nm);
            let h_time = dt_step / max(ge.n2, 0.25);
            let h_safe = select(0.08, clamp(0.03 * length(pos0) / sqrt(max(ge.n2, 0.25)), 0.004, 0.08), phys == PHYS_GRIN_BH);
            let h = min(h_time, h_safe);

            let next = grin_rk4_step(phys, pos0, p0, ph.lambda_nm, h);
            let pos1 = next.xy;
            let p1 = next.zw;
            rng = pcg_step(rng);
            if (grin_absorbed(phys, pos1)) {
                splat_segment_xyz(pos0, pos1, xyz, 1.0, pcg_f32(rng));
                ph.flags = ph.flags & (~1u);
                break;
            }

            splat_segment_xyz(pos0, pos1, xyz, 1.0, pcg_f32(rng));
            ph.pos = pos1;
            ph.p = p1;
            let age = (ph.flags >> 6u) + 1u;
            let rel = abs(pos1 - u.view.xy);
            if (rel.x > bound_x || rel.y > bound_y || age > 4000u) {
                ph.flags = ph.flags & (~1u);
            } else {
                ph.flags = (ph.flags & 63u) | (age << 6u);
            }
        }
    } else {
        let spec_w = emitter_spectral_weight(em_idx, ph.lambda_nm);
        let xyz = sample_spectral_xyz(ph.lambda_nm) * spec_w;
        for (var s = 0u; s < substeps; s = s + 1u) {
            if ((ph.flags & 1u) == 0u) {
                break;
            }
            var rem_t = dt_step;
            for (var bounce_iter = 0u; bounce_iter < 4u; bounce_iter = bounce_iter + 1u) {
                if (rem_t <= 1e-5 || (ph.flags & 1u) == 0u) {
                    break;
                }
                let in_atmo = (ph.flags & 2u) != 0u;
                let n_cur = select(max(ph.energy, 1.0), 1.0, in_atmo);
                let dist_budget = rem_t / n_cur;
                let hit = intersect_scene(ph.pos, ph.p, dist_budget + 2e-6);
                let seg_len = select(dist_budget, min(hit.t, dist_budget), hit.t <= dist_budget + 2e-6);

                var did_scatter = false;
                if (in_atmo && ph.energy > 0.0) {
                    let sigma = rayleigh_sigma(ph.lambda_nm, ph.energy);
                    rng = pcg_step(rng);
                    let s_scat = -log(max(1.0 - pcg_f32(rng), 1e-6)) / max(sigma, 1e-6);
                    if (s_scat < seg_len) {
                        let scat_pos = ph.pos + ph.p * s_scat;
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, scat_pos, xyz, 1.0, pcg_f32(rng));
                        rng = pcg_step(rng);
                        let u0 = pcg_f32(rng);
                        rng = pcg_step(rng);
                        let u1 = pcg_f32(rng);
                        rng = pcg_step(rng);
                        let u2 = pcg_f32(rng);
                        ph.pos = scat_pos;
                        ph.p = rayleigh_scatter_dir(ph.p, u0, u1, u2);
                        rem_t = max(rem_t - s_scat, 0.0);
                        did_scatter = true;
                    }
                }

                if (did_scatter) {
                    // Proceed to next bounce iteration within this substep.
                } else if (hit.t <= dist_budget + 2e-6) {
                    let hit_t = min(hit.t, dist_budget);
                    let hit_pos = ph.pos + ph.p * hit_t;
                    rng = pcg_step(rng);
                    splat_segment_xyz(ph.pos, hit_pos, xyz, 1.0, pcg_f32(rng));
                    rem_t = max(rem_t - hit_t * n_cur, 0.0);

                    if (hit.mat_kind == MAT_ABSORBER) {
                        ph.flags = ph.flags & (~1u);
                        break;
                    } else if (hit.mat_kind == MAT_MIRROR) {
                        var n = hit.normal;
                        if (dot(ph.p, n) > 0.0) {
                            n = -n;
                        }
                        ph.p = normalize(ph.p - 2.0 * dot(ph.p, n) * n);
                        ph.pos = hit_pos + n * 5e-5;
                    } else if (hit.mat_kind == MAT_GLASS) {
                        let entering = dot(ph.p, hit.normal) < 0.0;
                        let n_face = select(-hit.normal, hit.normal, entering);
                        let n_glass = cauchy_ior(ph.lambda_nm, hit.mat.x, hit.mat.y);
                        let n1 = select(n_glass, 1.0, entering);
                        let n2 = select(1.0, n_glass, entering);
                        let cos_i = clamp(-dot(ph.p, n_face), 0.0, 1.0);
                        let eta = n1 / n2;
                        let sin2_t = eta * eta * max(1.0 - cos_i * cos_i, 0.0);

                        var refl = 1.0;
                        var refr_dir = vec2<f32>(0.0);
                        if (sin2_t < 1.0) {
                            let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
                            let rs = (n1 * cos_i - n2 * cos_t) / max(n1 * cos_i + n2 * cos_t, 1e-6);
                            let rp = (n1 * cos_t - n2 * cos_i) / max(n1 * cos_t + n2 * cos_i, 1e-6);
                            refl = 0.5 * (rs * rs + rp * rp);
                            refr_dir = normalize(eta * ph.p + (eta * cos_i - cos_t) * n_face);
                        }

                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < refl) {
                            ph.p = normalize(ph.p - 2.0 * dot(ph.p, n_face) * n_face);
                            ph.pos = hit_pos + n_face * 5e-5;
                            ph.energy = n1;
                        } else {
                            ph.p = refr_dir;
                            ph.pos = hit_pos - n_face * 5e-5;
                            ph.energy = n2;
                        }
                        ph.flags = ph.flags & (~2u);
                    } else if (hit.mat_kind == MAT_DIFFUSE) {
                        let refl = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < refl) {
                            var n = hit.normal;
                            if (dot(ph.p, n) > 0.0) {
                                n = -n;
                            }
                            rng = pcg_step(rng);
                            ph.p = lambertian_reflect(n, ph.p, pcg_f32(rng));
                            ph.pos = hit_pos + n * 2e-5;
                        } else {
                            ph.flags = ph.flags & (~1u);
                            break;
                        }
                    } else if (hit.mat_kind == MAT_ATMOSPHERE) {
                        let entering = dot(ph.p, hit.normal) < 0.0;
                        let n_face = select(-hit.normal, hit.normal, entering);
                        let n_atmo = 1.0 + hit.mat.y;
                        let n1 = select(n_atmo, 1.0, entering);
                        let n2 = select(1.0, n_atmo, entering);
                        let cos_i = clamp(-dot(ph.p, n_face), 0.0, 1.0);
                        let eta = n1 / n2;
                        let sin2_t = eta * eta * max(1.0 - cos_i * cos_i, 0.0);
                        let atmo_eps = clamp(hit.mat.z * 0.0015, 1e-5, 1e-4);
                        if (sin2_t < 1.0) {
                            let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
                            ph.p = normalize(eta * ph.p + (eta * cos_i - cos_t) * n_face);
                            ph.pos = hit_pos - n_face * atmo_eps;
                            if (entering) {
                                ph.flags = ph.flags | 2u;
                                ph.energy = hit.mat.x;
                            } else {
                                ph.flags = ph.flags & (~2u);
                                ph.energy = 1.0;
                            }
                        } else {
                            ph.pos = hit_pos + ph.p * (atmo_eps * 2.0);
                            ph.flags = ph.flags & (~2u);
                            ph.energy = 1.0;
                        }
                    } else if (hit.mat_kind == MAT_SPLITTER) {
                        let prob_refl = hit.mat.w;
                        rng = pcg_step(rng);
                        if (pcg_f32(rng) < prob_refl) {
                            var n = hit.normal;
                            if (dot(ph.p, n) > 0.0) {
                                n = -n;
                            }
                            if (hit.subkind == 1u) {
                                let alb = spectral_albedo(ph.lambda_nm, hit.mat.xyz);
                                rng = pcg_step(rng);
                                if (pcg_f32(rng) < alb) {
                                    rng = pcg_step(rng);
                                    ph.p = lambertian_reflect(n, ph.p, pcg_f32(rng));
                                    ph.pos = hit_pos + n * 2e-5;
                                } else {
                                    ph.flags = ph.flags & (~1u);
                                    break;
                                }
                            } else {
                                ph.p = normalize(ph.p - 2.0 * dot(ph.p, n) * n);
                                ph.pos = hit_pos + n * 5e-5;
                            }
                        } else {
                            ph.pos = hit_pos + ph.p * 5e-5;
                        }
                    }

                    let bounces = (ph.flags >> 6u) + 1u;
                    if (bounces > 32u) {
                        ph.flags = ph.flags & (~1u);
                        break;
                    }
                    ph.flags = (ph.flags & 63u) | (bounces << 6u);
                } else {
                    if (u.bh.z > 0.0) {
                        let bh_pos = u.bh.xy;
                        let bh_m = u.bh.z;
                        let r_bh = length(ph.pos - bh_pos);
                        if (r_bh <= 0.62 * bh_m) {
                            ph.flags = ph.flags & (~1u);
                            break;
                        }
                        let n_bh = bh_n(r_bh, bh_m);
                        // Exact Shapiro coordinate-time delay: dt / dl = n(r)^2 -> dl = dt / n^2.
                        let h_t = rem_t / max(n_bh * n_bh, 1.0);
                        let h_safe = clamp(0.036 * r_bh / max(n_bh, 1.0), 0.004 * bh_m, 0.14 * half_h);
                        let h = min(h_t, h_safe);
                        let next = bh_rk4_step(ph.pos, ph.p, bh_pos, bh_m, h);
                        let pos1 = next.xy;
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, pos1, xyz, 1.0, pcg_f32(rng));
                        ph.pos = pos1;
                        ph.p = next.zw;
                        rem_t = max(rem_t - h * n_bh * n_bh, 0.0);
                        if (length(ph.pos - bh_pos) <= 0.62 * bh_m) {
                            ph.flags = ph.flags & (~1u);
                            break;
                        }
                    } else {
                        let next_pos = ph.pos + ph.p * dist_budget;
                        rng = pcg_step(rng);
                        splat_segment_xyz(ph.pos, next_pos, xyz, 1.0, pcg_f32(rng));
                        ph.pos = next_pos;
                        rem_t = 0.0;
                    }
                }
            }

            let offset = ph.pos - u.view.xy;
            let rel = abs(offset);
            if ((dot(offset, ph.p) > 0.0 && (rel.x > bound_x || rel.y > bound_y))
                || max(rel.x, rel.y) > max(bound_x, bound_y) * 16.0)
            {
                ph.flags = ph.flags & (~1u);
            }
        }
    }

    ph.rng = rng;
    photons[gid] = ph;
}
