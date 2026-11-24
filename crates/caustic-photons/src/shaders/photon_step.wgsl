// Pass 1: Real-time forward spectral photon simulation in 2D dispersive optics
// OR equatorial Ingoing Cartesian Kerr-Schild curved spacetime.
//
// Struct layouts match `crates/caustic-photons/src/scene.rs` and `gpu_sim.rs`.
// Deposits CIE 1931 XYZ (strictly non-negative for all wavelengths) into a fixed-point
// atomic<u32> accumulation buffer so arbitrary wavelengths mix physically into white
// without requiring f32 atomics.

struct SimUniforms {
    // xy = view_center, z = view_half_height, w = aspect (width / height)
    view: vec4<f32>,
    // xy = emitter_pos, z = emitter_angle, w = emitter_spread
    emitter: vec4<f32>,
    // x = dt_substep, y = spin a, z = r_plus, w = fixed_scale
    physics: vec4<f32>,
    // x = width, y = height, z = num_photons, w = substeps
    dims: vec4<u32>,
    // x = is_kerr (0/1), y = mode (0=Accumulate, 1=Pulse), z = emitter_kind (0=Beam, 1=Point), w = pulse_trigger (0/1)
    flags: vec4<u32>,
    // x = num_segments, y = num_circles, z = frame_idx, w = seed
    counts: vec4<u32>,
};

// 32-byte Photon state record.
struct Photon {
    pos: vec2<f32>,
    // In optics mode: unit direction d. In Kerr mode: covariant spatial momentum (p_x, p_y).
    p: vec2<f32>,
    // In optics mode: wavelength in nm [380..780].
    // In Kerr mode: emitted local ZAMO wavelength lambda_emit (with nu_emit = 1.0).
    lambda_nm: f32,
    // In optics mode: 1.0 (or current medium IOR n_cur). In Kerr mode: conserved energy E = -p_t.
    energy: f32,
    rng: u32,
    // bit 0: active, bit 1: inside_glass, bits 2..32: bounces/age
    flags: u32,
};

struct OptSegment {
    endpoints: vec4<f32>, // xy = p0, zw = p1
    material: vec4<f32>,  // x = cauchy_a, y = cauchy_b
    info: vec4<u32>,      // x = mat_kind
};

struct OptCircle {
    geom: vec4<f32>,     // xy = center, z = radius, w = clip_radius
    clip_arc: vec4<f32>, // xy = clip_center, z = arc_cos_half, w = arc_angle_center
    material: vec4<f32>, // x = cauchy_a, y = cauchy_b
    info: vec4<u32>,     // x = mat_kind, y = invert_normal
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(2) var<storage, read_write> accum_xyz: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> spectral_xyz_lut: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> segments: array<OptSegment>;
@group(0) @binding(5) var<storage, read> circles: array<OptCircle>;

const MAT_MIRROR: u32 = 1u;
const MAT_GLASS: u32 = 2u;
const MAT_ABSORBER: u32 = 3u;

const EMITTER_BEAM: u32 = 0u;
const EMITTER_POINT: u32 = 1u;

const MODE_ACCUMULATE: u32 = 0u;
const MODE_PULSE: u32 = 1u;

const LAMBDA_MIN: f32 = 380.0;
const LAMBDA_MAX: f32 = 780.0;

fn pcg_next(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    return f32(pcg_next(state) & 0x00FFFFFFu) * (1.0 / 16777216.0);
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

// Liang-Barsky screen-clipped, step-size-invariant spectral segment splat.
// Every pixel crossed receives exact geometric arc-length weighting (step_px * density_w)
// with stochastic phase jitter along the segment to prevent step-boundary Moire.
fn splat_segment_xyz(w0: vec2<f32>, w1: vec2<f32>, xyz: vec3<f32>, density_w: f32, jitter: f32) {
    let s0 = world_to_pixel(w0);
    let s1 = world_to_pixel(w1);
    let d = s1 - s0;
    let fw = f32(u.dims.x);
    let fh = f32(u.dims.y);

    var t0 = 0.0;
    var t1 = 1.0;

    // Clip X against [0, fw]
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

    // Clip Y against [0, fh]
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

    let n = clamp(u32(ceil(dist_px)), 1u, 4096u);
    let step_px = dist_px / f32(n);
    let per_sample = xyz * (density_w * step_px);
    let step_vec = (c1 - c0) / f32(n);
    var cur = c0 + step_vec * jitter;
    for (var i = 0u; i < n; i = i + 1u) {
        splat_pixel_xyz(i32(floor(cur.x)), i32(floor(cur.y)), per_sample);
        cur = cur + step_vec;
    }
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

// Initialize a future-directed null covector (E, p) in Ingoing KS at `pos` aiming along `aim`.
// Normalized to local ZAMO frequency nu_emit = 1.0.
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

    // Metric dot helper in (t, x, y): g(u, v) = -u.x*v.x + dot(u.yz, v.yz) + 2h*(l.u)*(l.v)
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

// ---------- Flat-Space Optical Intersection Helpers ----------

struct HitInfo {
    t: f32,
    normal: vec2<f32>,
    mat_kind: u32,
    cauchy_a: f32,
    cauchy_b: f32,
};

fn intersect_scene(pos: vec2<f32>, dir: vec2<f32>, t_max: f32) -> HitInfo {
    var best = HitInfo(t_max + 1.0, vec2<f32>(0.0, 1.0), 0u, 1.0, 0.0);

    // 1. Line segments (mirrors, prism faces, absorbers)
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
            // Outward normal for CCW polygon edge a -> b is (e.y, -e.x).
            let n = normalize(vec2<f32>(e.y, -e.x));
            best = HitInfo(t, n, seg.info.x, seg.material.x, seg.material.y);
        }
    }

    // 2. Circular arcs and clipped biconvex lens surfaces
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
            best = HitInfo(t, n, cir.info.x, cir.material.x, cir.material.y);
        }
    }

    return best;
}

fn spawn_photon(gid: u32, rng: ptr<function, u32>) -> Photon {
    let base_ang = u.emitter.z;
    let spread = u.emitter.w;
    let is_kerr = u.flags.x != 0u;
    let mode = u.flags.y;
    let emitter_kind = u.flags.z;

    let r0 = rand_f32(rng);
    let r1 = rand_f32(rng);

    var pos = u.emitter.xy;
    var aim = vec2<f32>(cos(base_ang), sin(base_ang));

    if (emitter_kind == EMITTER_BEAM) {
        let perp = vec2<f32>(-aim.y, aim.x);
        // Stratified across photon id for low-noise live dragging AND smooth pulse wavefronts!
        let lane = ((f32(gid) + r0) / f32(u.dims.z)) * 2.0 - 1.0;
        pos = pos + perp * (lane * spread);
    } else {
        let ang_off = (((f32(gid) + r0) / f32(u.dims.z)) * 2.0 - 1.0) * spread;
        let ang = base_ang + ang_off;
        aim = vec2<f32>(cos(ang), sin(ang));
    }

    if (is_kerr) {
        let a = u.physics.y;
        let a2 = a * a;
        let rp = u.physics.z;
        // Keep spawn safely outside horizon.
        if (ks_r(pos, a2) <= rp * 1.15) {
            pos = normalize(pos + vec2<f32>(1e-4, 0.0)) * (rp * 1.3 + abs(a));
        }
        let ep = init_kerr_photon_momentum(pos, aim, a, a2, rp);
        let energy = ep.x;
        let p = ep.yz;
        // Warm golden-amber emission band (585..610 nm, nu_emit = 1.0) so gravitational
        // blueshift shifts smoothly gold -> emerald -> cyan -> electric violet while
        // gravitational redshift shifts gold -> orange -> deep crimson.
        let lambda_emit = mix(585.0, 610.0, r1);
        return Photon(pos, p, lambda_emit, energy, *rng, 1u);
    } else {
        // Uniform full visible spectrum [380, 780] nm -> integrates to pure white light,
        // splits into full physical rainbow inside Cauchy dispersive glass!
        let lambda_nm = mix(LAMBDA_MIN, LAMBDA_MAX, r1);
        return Photon(pos, aim, lambda_nm, 1.0, *rng, 1u);
    }
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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid_v: vec3<u32>) {
    let gid = gid_v.x;
    if (gid >= u.dims.z) {
        return;
    }

    let is_kerr = u.flags.x != 0u;
    let mode = u.flags.y;
    let pulse_trigger = u.flags.w != 0u;
    let substeps = u.dims.w;
    let dt_step = u.physics.x;

    // World bounds with generous margin so curved or reflected paths return cleanly.
    let half_h = u.view.z;
    let half_w = half_h * u.view.w;
    let bound_x = half_w * 2.4;
    let bound_y = half_h * 2.4;

    if (mode == MODE_ACCUMULATE) {
        // Full-path forward photon tracing per frame: every frame delivers complete
        // end-to-end trajectories so live emitter dragging and aiming update the entire
        // beam, dispersion rainbow, and caustics instantaneously at interactive framerates,
        // while stationary frames accumulate cleanly to multi-million-photon HDR quality!
        var rng = gid * 1973u + u.counts.w * 9277u + u.counts.z * 26699u + 1u;
        let _w0 = pcg_next(&rng);
        let _w1 = pcg_next(&rng);
        var ph = spawn_photon(gid, &rng);

        if (is_kerr) {
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
                let r_next = ks_r(pos1, a2);
                if (r_next <= r_cap) {
                    break;
                }

                let nu_loc = equatorial_zamo_nu(pos1, p1, energy, a, a2, rp);
                let col = kerr_spectral_xyz(ph.lambda_nm, nu_loc);
                splat_segment_xyz(pos0, pos1, col.xyz, col.w, rand_f32(&rng));

                ph.pos = pos1;
                ph.p = p1;
                let rel = abs(pos1 - u.view.xy);
                if (rel.x > bound_x || rel.y > bound_y) {
                    break;
                }
            }
        } else {
            // Exact straight-line multi-bounce dispersive optics in one frame.
            let xyz = sample_spectral_xyz(ph.lambda_nm);
            let max_dist = length(vec2<f32>(bound_x, bound_y)) * 2.5;

            for (var bounce = 0u; bounce < 24u; bounce = bounce + 1u) {
                let hit = intersect_scene(ph.pos, ph.p, max_dist);
                if (hit.t <= max_dist) {
                    let hit_pos = ph.pos + ph.p * hit.t;
                    splat_segment_xyz(ph.pos, hit_pos, xyz, 1.0, rand_f32(&rng));

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
                        let n_glass = cauchy_ior(ph.lambda_nm, hit.cauchy_a, hit.cauchy_b);
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

                        if (rand_f32(&rng) < refl) {
                            ph.p = normalize(ph.p - 2.0 * dot(ph.p, n_face) * n_face);
                            ph.pos = hit_pos + n_face * 5e-5;
                        } else {
                            ph.p = refr_dir;
                            ph.pos = hit_pos - n_face * 5e-5;
                        }
                    }

                    let rel = abs(ph.pos - u.view.xy);
                    if (rel.x > bound_x || rel.y > bound_y) {
                        break;
                    }
                } else {
                    let end_pos = ph.pos + ph.p * max_dist;
                    splat_segment_xyz(ph.pos, end_pos, xyz, 1.0, rand_f32(&rng));
                    break;
                }
            }
        }
        return;
    }

    // ---------- MODE_PULSE: Stateful real-time wavefront flight at c / n(lambda) ----------
    var ph = photons[gid];
    var rng = ph.rng;
    if (u.counts.z == 0u || pulse_trigger) {
        rng = gid * 1973u + u.counts.w * 9277u + u.counts.z * 26699u + 1u;
        let _w = pcg_next(&rng);
        ph = spawn_photon(gid, &rng);
    }

    if ((ph.flags & 1u) == 0u) {
        ph.rng = rng;
        photons[gid] = ph;
        return;
    }

    if (is_kerr) {
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
            // Step by coordinate time dt so real Shapiro delay wraps the wavefront near r_+.
            let dt_dl = clamp(k1.dt, 0.25, 16.0);
            let h = min(dt_step / dt_dl, clamp(0.025 * r_cur, 0.01, dt_step));

            let k2 = ingoing_ks_rhs(pos0 + 0.5 * h * k1.dx, p0 + 0.5 * h * k1.dp, energy, a, a2);
            let k3 = ingoing_ks_rhs(pos0 + 0.5 * h * k2.dx, p0 + 0.5 * h * k2.dp, energy, a, a2);
            let k4 = ingoing_ks_rhs(pos0 + h * k3.dx, p0 + h * k3.dp, energy, a, a2);

            let pos1 = pos0 + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
            let p1 = p0 + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
            let r_next = ks_r(pos1, a2);

            if (r_next <= r_cap) {
                ph.flags = ph.flags & (~1u);
                break;
            }

            let nu_loc = equatorial_zamo_nu(pos1, p1, energy, a, a2, rp);
            let col = kerr_spectral_xyz(ph.lambda_nm, nu_loc);
            splat_segment_xyz(pos0, pos1, col.xyz, col.w, rand_f32(&rng));

            ph.pos = pos1;
            ph.p = p1;
            let age = (ph.flags >> 2u) + 1u;
            let rel = abs(pos1 - u.view.xy);
            if (rel.x > bound_x || rel.y > bound_y || age > 4000u) {
                ph.flags = ph.flags & (~1u);
            } else {
                ph.flags = (ph.flags & 3u) | (age << 2u);
            }
        }
    } else {
        for (var s = 0u; s < substeps; s = s + 1u) {
            if ((ph.flags & 1u) == 0u) {
                break;
            }

            let xyz = sample_spectral_xyz(ph.lambda_nm);
            var rem_t = dt_step;
            for (var bounce_iter = 0u; bounce_iter < 4u; bounce_iter = bounce_iter + 1u) {
                if (rem_t <= 1e-5 || (ph.flags & 1u) == 0u) {
                    break;
                }
                let n_cur = max(ph.energy, 1.0);
                let dist_budget = rem_t / n_cur;
                let hit = intersect_scene(ph.pos, ph.p, dist_budget + 2e-6);

                if (hit.t <= dist_budget + 2e-6) {
                    let hit_t = min(hit.t, dist_budget);
                    let hit_pos = ph.pos + ph.p * hit_t;
                    splat_segment_xyz(ph.pos, hit_pos, xyz, 1.0, rand_f32(&rng));
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
                        let n_glass = cauchy_ior(ph.lambda_nm, hit.cauchy_a, hit.cauchy_b);
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

                        if (rand_f32(&rng) < refl) {
                            ph.p = normalize(ph.p - 2.0 * dot(ph.p, n_face) * n_face);
                            ph.pos = hit_pos + n_face * 5e-5;
                            ph.energy = n1;
                        } else {
                            ph.p = refr_dir;
                            ph.pos = hit_pos - n_face * 5e-5;
                            ph.energy = n2;
                        }
                    }

                    let bounces = (ph.flags >> 2u) + 1u;
                    if (bounces > 32u) {
                        ph.flags = ph.flags & (~1u);
                        break;
                    }
                    ph.flags = (ph.flags & 3u) | (bounces << 2u);
                } else {
                    let next_pos = ph.pos + ph.p * dist_budget;
                    splat_segment_xyz(ph.pos, next_pos, xyz, 1.0, rand_f32(&rng));
                    ph.pos = next_pos;
                    rem_t = 0.0;
                }
            }

            let rel = abs(ph.pos - u.view.xy);
            if (rel.x > bound_x || rel.y > bound_y) {
                ph.flags = ph.flags & (~1u);
            }
        }
    }

    ph.rng = rng;
    photons[gid] = ph;
}
