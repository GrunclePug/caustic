// Pass 1: Outgoing Cartesian Kerr-Schild Hamiltonian null geodesic tracer (RK4).
//
// Struct layouts match `src/sim/blackhole/gpu_tracer.rs` (`TraceUniforms`, `RayHit`).
// Conventions (G = c = M = 1, signature (-, +, +, +)):
// - Future-directed 4-momentum p_mu = (-E, p) normalized to camera-frame frequency nu_obs = 1,
//   integrated backward with step `-h` so shadow rays cross the past outer horizon H^- smoothly.
// - Outgoing KS null covector: l_mu = (1, -(r x - a y)/(r^2 + a^2), -(r y + a x)/(r^2 + a^2), -z/r).
// - Split 4K-safe SoA outputs: `ray_hits` (16 B/px) and `ray_skies` (16 B/px).

struct TraceUniforms {
    cam_pos: vec4<f32>,   // xyz = Outgoing KS camera pos, w = spin a
    tet_et: vec4<f32>,    // (t, x, y, z) covariant ZAMO e_(0)
    tet_er: vec4<f32>,    // (t, x, y, z) covariant ZAMO e_(r)
    tet_eth: vec4<f32>,   // (t, x, y, z) covariant ZAMO e_(theta)
    tet_ephi: vec4<f32>,  // (t, x, y, z) covariant ZAMO e_(phi)
    params0: vec4<f32>,   // x = r_plus, y = r_isco, z = disk_outer, w = r_escape
    params1: vec4<f32>,   // x = tan_half_vfov, y = aspect, z = eta_near, w = eta_far
    dims: vec4<u32>,      // x = width, y = height, z = max_steps, w = disk_enabled (0/1)
};

// 16-byte aligned RayHit record (native u32 flags to avoid float bitcast NaN/FTZ issues).
struct RayHit {
    r_hit: f32,
    phi_hit: f32,
    redshift_g: f32,
    flags: u32, // bits 0..2: status (0=Captured, 1=DiskHit, 2=Escaped, 3=BudgetExhausted),
                // bits 2..4: crossing order (clamped 0..3), bits 4..32: step count
};

@group(0) @binding(0) var<uniform> u: TraceUniforms;
@group(0) @binding(1) var<storage, read_write> ray_hits: array<RayHit>;
@group(0) @binding(2) var<storage, read_write> ray_skies: array<vec4<f32>>;

const STATUS_CAPTURED: u32 = 0u;
const STATUS_DISK_HIT: u32 = 1u;
const STATUS_ESCAPED: u32 = 2u;
const STATUS_BUDGET_EXHAUSTED: u32 = 3u;

fn pack_flags(status: u32, order: u32, steps: u32) -> u32 {
    return (status & 3u) | ((min(order, 3u) & 3u) << 2u) | (min(steps, 0x0FFFFFFFu) << 4u);
}

fn ks_radius(pos: vec3<f32>, a2: f32) -> f32 {
    let b = dot(pos, pos) - a2;
    let disc = sqrt(max(b * b + 4.0 * a2 * pos.z * pos.z, 0.0));
    return sqrt(max(0.5 * (b + disc), 1e-12));
}

struct RhsOut {
    dx: vec3<f32>,
    dp: vec3<f32>,
};

// Future-directed Hamiltonian RHS (dx/dlambda, dp/dlambda) in Outgoing Cartesian KS.
fn ks_rhs(pos: vec3<f32>, p: vec3<f32>, energy: f32, a: f32, a2: f32) -> RhsOut {
    let r = ks_radius(pos, a2);
    let r2 = r * r;
    let d = r2 + a2;
    let inv_d = 1.0 / d;
    let inv_r = 1.0 / r;
    let denom = r2 * r2 + a2 * pos.z * pos.z;
    let inv_denom = 1.0 / denom;
    let h = (r2 * r) * inv_denom;

    let l = vec3<f32>(
        -(r * pos.x - a * pos.y) * inv_d,
        -(r * pos.y + a * pos.x) * inv_d,
        -pos.z * inv_r,
    );

    let k = energy + dot(l, p);
    let two_hk = 2.0 * h * k;
    let dx = p - two_hk * l;

    let grad_r_scale = r * inv_denom;
    let grad_r = vec3<f32>(
        grad_r_scale * r2 * pos.x,
        grad_r_scale * r2 * pos.y,
        grad_r_scale * d * pos.z,
    );

    let inv_denom2 = inv_denom * inv_denom;
    let dh_dr = r2 * (3.0 * a2 * pos.z * pos.z - r2 * r2) * inv_denom2;
    let dh_dz_fixed = -2.0 * a2 * r2 * r * pos.z * inv_denom2;
    let grad_h = dh_dr * grad_r + vec3<f32>(0.0, 0.0, dh_dz_fixed);

    let p_dl_dr = -(p.x * (pos.x + 2.0 * r * l.x) + p.y * (pos.y + 2.0 * r * l.y)) * inv_d
        - (p.z * l.z) * inv_r;
    let p_dl_fixed = vec3<f32>(
        -(r * p.x + a * p.y) * inv_d,
        (a * p.x - r * p.y) * inv_d,
        -p.z * inv_r,
    );
    let p_grad_l = p_dl_fixed + p_dl_dr * grad_r;
    let dp = (k * k) * grad_h + two_hk * p_grad_l;

    return RhsOut(dx, dp);
}

fn disk_redshift_g(r_hit: f32, a: f32, energy: f32, lz: f32) -> f32 {
    let r_sqrt = sqrt(max(r_hit, 1e-6));
    let r32 = r_hit * r_sqrt;
    let omega = 1.0 / (r32 + a);
    let u_t = (r32 + a) / sqrt(max(r32 * (r32 - 3.0 * r_sqrt + 2.0 * a), 1e-12));
    let nu_emit = u_t * (energy - omega * lz);
    return select(0.0, 1.0 / nu_emit, nu_emit > 1e-6);
}

fn hermite_eval_f32(x0: f32, v0: f32, x1: f32, v1: f32, h: f32, tau: f32) -> vec2<f32> {
    let t2 = tau * tau;
    let t3 = t2 * tau;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + tau;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let val = h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1;

    let dh00 = 6.0 * t2 - 6.0 * tau;
    let dh10 = 3.0 * t2 - 4.0 * tau + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * tau;
    let dh11 = 3.0 * t2 - 2.0 * tau;
    let dval = (dh00 * x0 + dh01 * x1) / max(h, 1e-12) + dh10 * v0 + dh11 * v1;
    return vec2<f32>(val, dval);
}

fn hermite_eval_vec3(x0: vec3<f32>, v0: vec3<f32>, x1: vec3<f32>, v1: vec3<f32>, h: f32, tau: f32) -> vec3<f32> {
    let t2 = tau * tau;
    let t3 = t2 * tau;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + tau;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    return h00 * x0 + (h10 * h) * v0 + h01 * x1 + (h11 * h) * v1;
}

// Safeguarded Newton + bisection root-finder for z(tau) = 0 to |z| < 1e-5 M.
fn refine_z_root(z0: f32, vz0: f32, z1: f32, vz1: f32, h: f32) -> f32 {
    var lo = 0.0;
    var hi = 1.0;
    var f_lo = z0;
    var tau = clamp(-z0 / select(1.0, z1 - z0, abs(z1 - z0) > 1e-12), 0.0, 1.0);

    for (var it = 0u; it < 10u; it = it + 1u) {
        let ev = hermite_eval_f32(z0, vz0, z1, vz1, h, tau);
        let f = ev.x;
        if (abs(f) < 1e-5) {
            return tau;
        }
        if ((f_lo > 0.0 && f > 0.0) || (f_lo < 0.0 && f < 0.0)) {
            lo = tau;
            f_lo = f;
        } else {
            hi = tau;
        }
        let df_dtau = ev.y * h;
        let cand = select(0.5 * (lo + hi), tau - f / df_dtau, abs(df_dtau) > 1e-12);
        tau = select(0.5 * (lo + hi), cand, cand > lo && cand < hi);
        if (hi - lo < 1e-6) {
            break;
        }
    }
    return 0.5 * (lo + hi);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u.dims.x;
    let height = u.dims.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let idx = gid.y * width + gid.x;

    let a = u.cam_pos.w;
    let a2 = a * a;
    let r_plus = u.params0.x;
    let r_isco = u.params0.y;
    let r_out = u.params0.z;
    let r_esc = u.params0.w;
    let tan_half = u.params1.x;
    let aspect = u.params1.y;
    let eta_near = u.params1.z;
    let eta_far = u.params1.w;
    let max_steps = u.dims.z;
    let disk_enabled = u.dims.w != 0u;

    // Pixel center in NDC [-1, 1] (+x right, +y up).
    let ndc_x = ((f32(gid.x) + 0.5) / f32(width)) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(gid.y) + 0.5) / f32(height)) * 2.0;

    // Local ZAMO camera look direction: forward = -e_r, right = +e_phi, up = -e_theta.
    let local_dir = normalize(vec3<f32>(-1.0, -ndc_y * tan_half, ndc_x * aspect * tan_half));

    // Future-directed null 4-momentum p^mu normalized to camera-frame frequency nu_obs = 1.
    let p_up = u.tet_et
        - local_dir.x * u.tet_er
        - local_dir.y * u.tet_eth
        - local_dir.z * u.tet_ephi;

    var pos = u.cam_pos.xyz;
    let r0 = ks_radius(pos, a2);
    let r0_2 = r0 * r0;
    let h0 = (r0_2 * r0) / (r0_2 * r0_2 + a2 * pos.z * pos.z);
    let d0 = r0_2 + a2;
    let l0 = vec3<f32>(
        -(r0 * pos.x - a * pos.y) / d0,
        -(r0 * pos.y + a * pos.x) / d0,
        -pos.z / r0,
    );
    let l_dot_pup = p_up.x + dot(l0, p_up.yzw);
    let p_down_t = -p_up.x + 2.0 * h0 * l_dot_pup;
    let energy = -p_down_t;
    var p = p_up.yzw + (2.0 * h0 * l_dot_pup) * l0;

    // Exact Killing axial angular momentum L_z = x p_y - y p_x (constant along geodesic).
    let lz = pos.x * p.y - pos.y * p.x;

    // Default outputs.
    var status = STATUS_BUDGET_EXHAUSTED;
    var out_r = 0.0;
    var out_phi = 0.0;
    var out_g = 0.0;
    var out_sky = vec4<f32>(0.0);
    var order = 0u;
    var steps = 0u;

    let r_capture = r_plus * 0.95;
    let r_out_guard2 = (r_out * 1.1) * (r_out * 1.1);
    let h_min = 0.005;
    let h_max = 60.0;

    // Exact coplanar check if camera starts directly on the equatorial disk.
    if (disk_enabled && abs(pos.z) < 1e-5) {
        let rhs0 = ks_rhs(pos, p, energy, a, a2);
        if (abs(rhs0.dx.z) < 1e-5 && r0 >= r_isco && r0 <= r_out) {
            ray_hits[idx] = RayHit(
                r0,
                atan2(pos.y, pos.x),
                disk_redshift_g(r0, a, energy, lz),
                pack_flags(STATUS_DISK_HIT, 0u, 0u),
            );
            ray_skies[idx] = vec4<f32>(0.0);
            return;
        }
    }

    var r_cur = r0;
    var k1_fwd = ks_rhs(pos, p, energy, a, a2);

    for (var step = 0u; step < max_steps; step = step + 1u) {
        steps = step + 1u;

        // Smooth radial step law h = eta(r) * r; NO (r - r_+) or raw |z| stall limiters.
        let blend = smoothstep(3.5, 14.0, r_cur);
        let eta = mix(eta_near, eta_far, blend);
        var h = clamp(eta * r_cur, h_min, h_max);

        // Backward velocity is -k1_fwd.dx.
        let v0_bwd = -k1_fwd.dx;

        // Optional time-to-plane guard ONLY when approaching z = 0 inside 1.1 * r_out.
        let rho2 = pos.x * pos.x + pos.y * pos.y;
        if (disk_enabled && rho2 <= r_out_guard2 && pos.z * v0_bwd.z < 0.0) {
            let t_plane = -pos.z / v0_bwd.z;
            if (t_plane > 0.0) {
                h = min(h, max(1.25 * t_plane, h_min));
            }
        }

        // Classical RK4 stepping backward by `-h` (in-place register accumulation).
        let hs = -h;
        let half_hs = 0.5 * hs;

        let x2 = pos + half_hs * k1_fwd.dx;
        let p2 = p + half_hs * k1_fwd.dp;
        let k2_fwd = ks_rhs(x2, p2, energy, a, a2);

        let x3 = pos + half_hs * k2_fwd.dx;
        let p3 = p + half_hs * k2_fwd.dp;
        let k3_fwd = ks_rhs(x3, p3, energy, a, a2);

        let x4 = pos + hs * k3_fwd.dx;
        let p4 = p + hs * k3_fwd.dp;
        let k4_fwd = ks_rhs(x4, p4, energy, a, a2);

        let next_pos = pos + (hs / 6.0) * (k1_fwd.dx + 2.0 * k2_fwd.dx + 2.0 * k3_fwd.dx + k4_fwd.dx);
        let next_p = p + (hs / 6.0) * (k1_fwd.dp + 2.0 * k2_fwd.dp + 2.0 * k3_fwd.dp + k4_fwd.dp);
        let next_k1_fwd = ks_rhs(next_pos, next_p, energy, a, a2);
        let v1_bwd = -next_k1_fwd.dx;
        let r_next = ks_radius(next_pos, a2);

        // 1. Coplanar equatorial ray radial entry into [r_isco, r_out].
        if (disk_enabled && abs(pos.z) < 1e-5 && abs(next_pos.z) < 1e-5 && abs(v0_bwd.z) < 1e-5) {
            let r_lo = min(r_cur, r_next);
            let r_hi = max(r_cur, r_next);
            if (r_hi >= r_isco && r_lo <= r_out && r_hi > r_plus) {
                let entry_r = clamp(r_cur, r_isco, r_out);
                let frac = select(0.0, clamp((entry_r - r_cur) / (r_next - r_cur), 0.0, 1.0), abs(r_next - r_cur) > 1e-7);
                let hit_pos = mix(pos, next_pos, frac);
                status = STATUS_DISK_HIT;
                out_r = entry_r;
                out_phi = atan2(hit_pos.y, hit_pos.x);
                out_g = disk_redshift_g(entry_r, a, energy, lz);
                break;
            }
        }

        // 2. Equatorial plane crossing z = 0 (checked BEFORE capture so ISCO hits near r_+ are never clipped).
        if (disk_enabled && (pos.z * next_pos.z < 0.0 || (pos.z != 0.0 && next_pos.z == 0.0))) {
            let tau = refine_z_root(pos.z, v0_bwd.z, next_pos.z, v1_bwd.z, h);
            let hit_pos = hermite_eval_vec3(pos, v0_bwd, next_pos, v1_bwd, h, tau);
            // Exact BL radius on z = 0 is sqrt(x^2 + y^2 - a^2).
            let rho_hit2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
            let r_hit = sqrt(max(rho_hit2 - a2, 0.0));
            if (r_hit > r_plus) {
                if (r_hit >= r_isco && r_hit <= r_out) {
                    status = STATUS_DISK_HIT;
                    out_r = r_hit;
                    out_phi = atan2(hit_pos.y, hit_pos.x);
                    out_g = disk_redshift_g(r_hit, a, energy, lz);
                    break;
                }
                order = order + 1u;
            }
        }

        // 3. Horizon capture in Outgoing KS (smooth across past horizon r_+ down to 0.95 r_+).
        if (r_next <= r_capture) {
            status = STATUS_CAPTURED;
            out_r = r_plus;
            break;
        }

        // 4. Escape at r >= r_esc.
        if (r_next >= r_esc && r_next > r_cur) {
            let frac = clamp((r_esc - r_cur) / max(r_next - r_cur, 1e-6), 0.0, 1.0);
            let dir_esc = normalize(mix(v0_bwd, v1_bwd, frac));
            let sky_g = select(0.0, 1.0 / energy, energy > 1e-6);
            status = STATUS_ESCAPED;
            out_r = r_esc;
            out_g = sky_g;
            out_sky = vec4<f32>(dir_esc, sky_g);
            break;
        }

        pos = next_pos;
        p = next_p;
        r_cur = r_next;
        k1_fwd = next_k1_fwd;
    }

    ray_hits[idx] = RayHit(out_r, out_phi, out_g, pack_flags(status, order, steps));
    ray_skies[idx] = out_sky;
}
