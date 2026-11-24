//! Independent CPU `f64` Boyer-Lindquist Mino-time geodesic oracle using
//! non-singular second-order polynomial ODEs in inverse radius `w = 1 / r` and
//! `u = cos(theta)` with adaptive Dormand-Prince RK45.
//!
//! Using `w = 1 / r` keeps `W(w) = w^4 R(1/w) = O(1)` from `r = infinity` (`w = 0`)
//! down to the event horizon (`w = 1 / r_+`), avoiding `r^4` cancellation at large
//! radii and reducing identically to the exact Binet equation `w'' + w = 3 M w^2`
//! for Schwarzschild (`a = 0`).

use glam::DVec3;

use crate::kerr::{
    BlInitialRay, Camera, KerrParams, bl_velocity_to_outgoing_ks_dir, delta_factored,
    disk_redshift_g, outgoing_ks_phi_twist, r_isco, r_plus,
};

/// 4-way ray termination classification (never conflate `BudgetExhausted` with `Captured`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RayStatus {
    Captured = 0,
    DiskHit = 1,
    Escaped = 2,
    BudgetExhausted = 3,
}

/// Complete trace result from the CPU `f64` Boyer-Lindquist oracle.
#[derive(Debug, Clone, Copy)]
pub struct OracleRayResult {
    pub status: RayStatus,
    /// Boyer-Lindquist hit radius `r` (valid for `DiskHit`).
    pub r_hit: f64,
    /// Outgoing Kerr-Schild Cartesian azimuth `atan2(y, x)` at disk hit (for direct GPU comparison).
    pub phi_ks_cart: f64,
    /// Boyer-Lindquist azimuth `phi_BL` at disk hit or escape.
    pub phi_bl: f64,
    /// Ergosphere-safe relativistic redshift factor `g` (disk or sky).
    pub redshift_g: f64,
    /// Unit escape direction in Outgoing Cartesian KS (valid for `Escaped`).
    pub escape_dir_ks: DVec3,
    /// Equatorial crossing order (`0` = primary direct image, `1` = secondary, `2` = tertiary+).
    pub order: u32,
    /// Number of accepted DOPRI5 steps.
    pub steps: u32,
    /// Maximum radial potential residual `|vw^2 - W(w)| / (1 + |W(w)|)` along the trajectory.
    pub max_radial_residual: f64,
    /// Maximum polar potential / Carter `Q` residual along the trajectory.
    pub max_carter_drift: f64,
}

/// Non-singular second-order Mino-time state in inverse radius `w = 1 / r` and `u = cos(theta)`
/// integrated backward (`s = -lambda_M` increases along the camera ray).
#[derive(Debug, Clone, Copy)]
struct MinoState {
    w: f64,
    vw: f64, // d w / d (-lambda_M) = -(1/r^2) * (-dr/dlambda_M) = +dr/dlambda_M / r^2
    u: f64,
    vu: f64, // d u / d (-lambda_M) = -du/dlambda_M
    phi: f64,
}

impl MinoState {
    fn add_scaled(self, d: Self, s: f64) -> Self {
        Self {
            w: self.w + s * d.w,
            vw: self.vw + s * d.vw,
            u: self.u + s * d.u,
            vu: self.vu + s * d.vu,
            phi: self.phi + s * d.phi,
        }
    }
}

/// Constants of motion along a null Kerr geodesic.
#[derive(Debug, Clone, Copy)]
struct GeodesicConsts {
    a: f64,
    a2: f64,
    e: f64,
    lz: f64,
    q: f64,
    k_carter: f64, // Q + (L - a E)^2
    c_u2: f64,     // Q + L^2 - a^2 E^2
    a2_e2: f64,    // a^2 E^2
    a_ae_m_l: f64, // a * (a E - L)
}

impl GeodesicConsts {
    fn new(a: f64, e: f64, lz: f64, q: f64) -> Self {
        let a2 = a * a;
        let lae = lz - a * e;
        let a2_e2 = a2 * e * e;
        Self {
            a,
            a2,
            e,
            lz,
            q,
            k_carter: q + lae * lae,
            c_u2: q + lz * lz - a2_e2,
            a2_e2,
            a_ae_m_l: a * (a * e - lz),
        }
    }

    /// Inverse-radius potential `W(w) = w^4 R(1/w)`.
    fn inv_radial_potential(&self, w: f64) -> f64 {
        let w2 = w * w;
        let pw = self.e + self.a_ae_m_l * w2;
        let dw = 1.0 - 2.0 * w + self.a2 * w2;
        pw * pw - dw * self.k_carter * w2
    }

    /// Polar potential `Theta_u(u)`.
    fn polar_potential(&self, u: f64) -> f64 {
        let u2 = u * u;
        self.q - self.c_u2 * u2 - self.a2_e2 * u2 * u2
    }

    /// RHS for backward Mino-time evolution (`s = -lambda_M` increases along backward ray).
    fn rhs(&self, st: MinoState) -> MinoState {
        let w = st.w;
        let w2 = w * w;
        let pw = self.e + self.a_ae_m_l * w2;
        // d^2 w / ds^2 = 0.5 * W'(w)
        let aw =
            2.0 * self.a_ae_m_l * w * pw - self.k_carter * (w - 3.0 * w2 + 2.0 * self.a2 * w2 * w);

        // d^2 u / ds^2 = 0.5 * Theta_u'(u)
        let u = st.u;
        let au = -self.c_u2 * u - 2.0 * self.a2_e2 * u * u * u;

        // dphi / ds = -dphi / dlambda_M
        // Note: a * P(r) / Delta(r) = a * (E + a(aE - L) w^2) / (1 - 2 w + a^2 w^2)
        // which stays completely smooth at w -> 0 (r -> infinity)!
        let dw = 1.0 - 2.0 * w + self.a2 * w2;
        let term_r = if dw.abs() > 1e-12 {
            (self.a * pw) / dw
        } else {
            0.0
        };
        let one_minus_u2 = 1.0 - u * u;
        // Special-case L_z = 0 (exact polar ray) so there is zero 1/(1-u^2) pole.
        let term_u = if self.lz.abs() < 1e-14 {
            0.0
        } else if one_minus_u2 > 1e-14 {
            self.lz / one_minus_u2
        } else {
            0.0
        };
        let dphi_ds = -(term_r + term_u - self.a * self.e);

        MinoState {
            w: st.vw,
            vw: aw,
            u: st.vu,
            vu: au,
            phi: dphi_ds,
        }
    }
}

/// Evaluate cubic Hermite position and velocity at `tau in [0, 1]` over step `h`.
fn hermite_eval(x0: f64, v0: f64, x1: f64, v1: f64, h: f64, tau: f64) -> (f64, f64) {
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
    let dval = (dh00 * x0 + dh01 * x1) / h + dh10 * v0 + dh11 * v1;
    (val, dval)
}

/// Refine a root of `f(tau) = 0` on `tau in [0, 1]` via safeguarded Newton + bisection.
fn refine_hermite_root(x0: f64, v0: f64, x1: f64, v1: f64, h: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut f_lo = x0;
    let mut tau = if (x1 - x0).abs() > 1e-18 {
        (-x0 / (x1 - x0)).clamp(0.0, 1.0)
    } else {
        0.5
    };

    for _ in 0..24 {
        let (f, df_ds) = hermite_eval(x0, v0, x1, v1, h, tau);
        if f.abs() < 1e-13 {
            return tau;
        }
        if (f_lo > 0.0 && f > 0.0) || (f_lo < 0.0 && f < 0.0) {
            lo = tau;
            f_lo = f;
        } else {
            hi = tau;
        }
        let df_dtau = df_ds * h;
        let next = if df_dtau.abs() > 1e-15 {
            tau - f / df_dtau
        } else {
            0.5 * (lo + hi)
        };
        tau = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
        if (hi - lo) < 1e-13 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Trace a single ray using the independent CPU `f64` Boyer-Lindquist Mino-time oracle.
pub fn trace_oracle_ray(
    init: BlInitialRay,
    params: KerrParams,
    obs_alpha: f64,
    disk_enabled: bool,
) -> OracleRayResult {
    let a = params.spin;
    let rp = r_plus(a);
    let wp = 1.0 / rp;
    let r_in = r_isco(a);
    let r_out = params.disk_outer;
    let r_esc = params.r_escape;
    let w_esc = 1.0 / r_esc;
    let consts = GeodesicConsts::new(a, init.energy, init.lz, init.q);

    // Backward Mino time s = -lambda_M:
    // w = 1/r, so dw/ds = (-1/r^2) * dr/ds = (-1/r^2) * (-dr/dlambda_M) = +dr_dlm / r^2.
    let w0 = 1.0 / init.r;
    let mut st = MinoState {
        w: w0,
        vw: init.dr_dlm * w0 * w0,
        u: init.u,
        vu: -init.du_dlm,
        phi: init.phi_bl,
    };

    // Dormand-Prince 5(4) Butcher tableau coefficients.
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    const E1: f64 = 71.0 / 57600.0;
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let rtol = 1e-12_f64;
    let atol = 1e-13_f64;
    let mut h = 0.01_f64;
    let max_steps: u32 = 25_000;
    let mut accepted: u32 = 0;
    let mut crossings: u32 = 0;
    let mut max_rad_res = 0.0_f64;
    let mut max_carter_res = 0.0_f64;

    // Exact coplanar check if camera starts directly on the equatorial plane inside the disk.
    if disk_enabled
        && st.u.abs() < 1e-12
        && st.vu.abs() < 1e-12
        && init.r >= r_in
        && init.r <= r_out
    {
        let phi_out = st.phi - outgoing_ks_phi_twist(init.r, a);
        let phi_ks_cart = phi_out - (a / init.r).atan();
        return OracleRayResult {
            status: RayStatus::DiskHit,
            r_hit: init.r,
            phi_ks_cart,
            phi_bl: st.phi,
            redshift_g: disk_redshift_g(init.r, a, consts.e, consts.lz),
            escape_dir_ks: DVec3::ZERO,
            order: 0,
            steps: 0,
            max_radial_residual: 0.0,
            max_carter_drift: 0.0,
        };
    }

    let mut k1 = consts.rhs(st);

    for _iter in 0..(max_steps * 4) {
        if accepted >= max_steps {
            break;
        }

        // Prevent giant steps across turning points or the horizon.
        let h_max = 0.2_f64;
        if h > h_max {
            h = h_max;
        }

        let s2 = st.add_scaled(k1, h * A21);
        let k2 = consts.rhs(s2);

        let s3 = st.add_scaled(k1, h * A31).add_scaled(k2, h * A32);
        let k3 = consts.rhs(s3);

        let s4 = st
            .add_scaled(k1, h * A41)
            .add_scaled(k2, h * A42)
            .add_scaled(k3, h * A43);
        let k4 = consts.rhs(s4);

        let s5 = st
            .add_scaled(k1, h * A51)
            .add_scaled(k2, h * A52)
            .add_scaled(k3, h * A53)
            .add_scaled(k4, h * A54);
        let k5 = consts.rhs(s5);

        let s6 = st
            .add_scaled(k1, h * A61)
            .add_scaled(k2, h * A62)
            .add_scaled(k3, h * A63)
            .add_scaled(k4, h * A64)
            .add_scaled(k5, h * A65);
        let k6 = consts.rhs(s6);

        let next = MinoState {
            w: st.w + h * (B1 * k1.w + B3 * k3.w + B4 * k4.w + B5 * k5.w + B6 * k6.w),
            vw: st.vw + h * (B1 * k1.vw + B3 * k3.vw + B4 * k4.vw + B5 * k5.vw + B6 * k6.vw),
            u: st.u + h * (B1 * k1.u + B3 * k3.u + B4 * k4.u + B5 * k5.u + B6 * k6.u),
            vu: st.vu + h * (B1 * k1.vu + B3 * k3.vu + B4 * k4.vu + B5 * k5.vu + B6 * k6.vu),
            phi: st.phi + h * (B1 * k1.phi + B3 * k3.phi + B4 * k4.phi + B5 * k5.phi + B6 * k6.phi),
        };
        let k7 = consts.rhs(next);

        let err_w = h * (E1 * k1.w + E3 * k3.w + E4 * k4.w + E5 * k5.w + E6 * k6.w + E7 * k7.w);
        let err_vw =
            h * (E1 * k1.vw + E3 * k3.vw + E4 * k4.vw + E5 * k5.vw + E6 * k6.vw + E7 * k7.vw);
        let err_u = h * (E1 * k1.u + E3 * k3.u + E4 * k4.u + E5 * k5.u + E6 * k6.u + E7 * k7.u);
        let err_vu =
            h * (E1 * k1.vu + E3 * k3.vu + E4 * k4.vu + E5 * k5.vu + E6 * k6.vu + E7 * k7.vu);
        // Include phi error whenever outside the horizon so near-polar L/(1-u^2) swings
        // are automatically refined instead of skipped!
        let err_phi = if next.w < wp * 0.99 {
            h * (E1 * k1.phi + E3 * k3.phi + E4 * k4.phi + E5 * k5.phi + E6 * k6.phi + E7 * k7.phi)
        } else {
            0.0
        };

        let sc_w = atol + rtol * st.w.abs().max(next.w.abs()).max(1e-6);
        let sc_vw = atol + rtol * st.vw.abs().max(next.vw.abs()).max(1.0);
        let sc_u = atol + rtol * st.u.abs().max(next.u.abs()).max(1.0);
        let sc_vu = atol + rtol * st.vu.abs().max(next.vu.abs()).max(1.0);
        // Slightly looser tolerance on phi during razor-sharp polar passes so steps stay fast.
        let sc_phi = 1e-9 + 1e-8 * st.phi.abs().max(next.phi.abs()).max(1.0);

        let err_norm = (((err_w / sc_w).powi(2)
            + (err_vw / sc_vw).powi(2)
            + (err_u / sc_u).powi(2)
            + (err_vu / sc_vu).powi(2)
            + (err_phi / sc_phi).powi(2))
            / 5.0)
            .sqrt();

        if err_norm <= 1.0 || h <= 1e-12 {
            let prev = st;
            let mut cur = next;
            accepted += 1;

            // Axis crossing pi-jump for exact polar rays (L_z = 0):
            // when vu changes sign near |u| == 1, the geodesic passed through the pole.
            if consts.lz.abs() < 1e-14 && prev.vu * cur.vu < 0.0 && cur.u.abs() > 0.999 {
                cur.phi += std::f64::consts::PI;
            }
            cur.u = cur.u.clamp(-1.0, 1.0);

            // Track independent invariant diagnostics while outside horizon.
            if cur.w > 0.0 && cur.w < wp * 0.999 {
                let w_pot = consts.inv_radial_potential(cur.w);
                let w_scale = 1.0 + cur.vw * cur.vw + w_pot.abs();
                max_rad_res = max_rad_res.max((cur.vw * cur.vw - w_pot).abs() / w_scale);

                let u_pot = consts.polar_potential(cur.u);
                let u_scale = 1.0 + cur.vu * cur.vu + u_pot.abs();
                max_carter_res = max_carter_res.max((cur.vu * cur.vu - u_pot).abs() / u_scale);
            }

            // 1. Check coplanar equatorial ray radial entry into [r_in, r_out].
            if disk_enabled
                && prev.u.abs() < 1e-11
                && cur.u.abs() < 1e-11
                && prev.vu.abs() < 1e-11
                && prev.w > 0.0
                && cur.w > 0.0
            {
                let prev_r = 1.0 / prev.w;
                let cur_r = 1.0 / cur.w;
                let (r_min_seg, r_max_seg) = if prev_r <= cur_r {
                    (prev_r, cur_r)
                } else {
                    (cur_r, prev_r)
                };
                if r_max_seg >= r_in && r_min_seg <= r_out {
                    let entry_r = prev_r.clamp(r_in, r_out);
                    let frac = if (cur_r - prev_r).abs() > 1e-15 {
                        ((entry_r - prev_r) / (cur_r - prev_r)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let phi_hit = prev.phi + frac * (cur.phi - prev.phi);
                    let phi_out = phi_hit - outgoing_ks_phi_twist(entry_r, a);
                    let phi_ks_cart = phi_out - (a / entry_r).atan();
                    return OracleRayResult {
                        status: RayStatus::DiskHit,
                        r_hit: entry_r,
                        phi_ks_cart,
                        phi_bl: phi_hit,
                        redshift_g: disk_redshift_g(entry_r, a, consts.e, consts.lz),
                        escape_dir_ks: DVec3::ZERO,
                        order: crossings,
                        steps: accepted,
                        max_radial_residual: max_rad_res,
                        max_carter_drift: max_carter_res,
                    };
                }
            }

            // 2. Check equatorial plane crossing u = 0 (checked BEFORE capture!).
            if disk_enabled && (prev.u * cur.u < 0.0 || (prev.u != 0.0 && cur.u == 0.0)) {
                let tau = refine_hermite_root(prev.u, prev.vu, cur.u, cur.vu, h);
                let (w_hit, _) = hermite_eval(prev.w, prev.vw, cur.w, cur.vw, h, tau);
                if w_hit > 0.0 && w_hit < wp {
                    let r_hit = 1.0 / w_hit;
                    let (phi_hit, _) = hermite_eval(prev.phi, k1.phi, cur.phi, k7.phi, h, tau);
                    if r_hit >= r_in && r_hit <= r_out {
                        let phi_out = phi_hit - outgoing_ks_phi_twist(r_hit, a);
                        let phi_ks_cart = phi_out - (a / r_hit).atan();
                        return OracleRayResult {
                            status: RayStatus::DiskHit,
                            r_hit,
                            phi_ks_cart,
                            phi_bl: phi_hit,
                            redshift_g: disk_redshift_g(r_hit, a, consts.e, consts.lz),
                            escape_dir_ks: DVec3::ZERO,
                            order: crossings,
                            steps: accepted,
                            max_radial_residual: max_rad_res,
                            max_carter_drift: max_carter_res,
                        };
                    }
                    crossings += 1;
                }
            }

            // 3. Horizon capture check: w >= 1 / r_+ (moving inward: vw > 0).
            if cur.w >= wp * (1.0 - 1e-6) && cur.vw > 0.0 || cur.w >= wp {
                return OracleRayResult {
                    status: RayStatus::Captured,
                    r_hit: rp,
                    phi_ks_cart: 0.0,
                    phi_bl: cur.phi,
                    redshift_g: 0.0,
                    escape_dir_ks: DVec3::ZERO,
                    order: crossings,
                    steps: accepted,
                    max_radial_residual: max_rad_res,
                    max_carter_drift: max_carter_res,
                };
            }

            // 4. Escape check at w <= w_esc (moving outward: vw < 0).
            if cur.w <= w_esc && cur.vw < 0.0 {
                let tau = refine_hermite_root(prev.w - w_esc, prev.vw, cur.w - w_esc, cur.vw, h);
                let (w_e, vw_e) = hermite_eval(prev.w, prev.vw, cur.w, cur.vw, h, tau);
                let (u_e, vu_e) = hermite_eval(prev.u, prev.vu, cur.u, cur.vu, h, tau);
                let (phi_e, dphi_e) = hermite_eval(prev.phi, k1.phi, cur.phi, k7.phi, h, tau);
                let r_e = 1.0 / w_e.max(1e-12);
                // Convert dw/ds -> dr/ds = -r^2 * dw/ds.
                let vr_e = -r_e * r_e * vw_e;
                let u_clamped = u_e.clamp(-1.0, 1.0);
                let dir_ks =
                    bl_velocity_to_outgoing_ks_dir(r_e, u_clamped, phi_e, vr_e, vu_e, dphi_e, a)
                        .normalize();
                let _ = obs_alpha;
                let sky_g = if consts.e > 1e-12 {
                    1.0 / consts.e
                } else {
                    0.0
                };
                return OracleRayResult {
                    status: RayStatus::Escaped,
                    r_hit: r_e,
                    phi_ks_cart: phi_e - outgoing_ks_phi_twist(r_e, a) - (a / r_e).atan(),
                    phi_bl: phi_e,
                    redshift_g: sky_g,
                    escape_dir_ks: dir_ks,
                    order: crossings,
                    steps: accepted,
                    max_radial_residual: max_rad_res,
                    max_carter_drift: max_carter_res,
                };
            }

            st = cur;
            k1 = consts.rhs(st);

            let factor = if err_norm < 1e-16 {
                4.0
            } else {
                (0.9 * err_norm.powf(-0.2)).clamp(0.2, 4.0)
            };
            h *= factor;
        } else {
            let factor = (0.9 * err_norm.powf(-0.25)).clamp(0.1, 0.5);
            h *= factor;
        }
    }

    OracleRayResult {
        status: RayStatus::BudgetExhausted,
        r_hit: if st.w > 0.0 { 1.0 / st.w } else { r_esc },
        phi_ks_cart: 0.0,
        phi_bl: st.phi,
        redshift_g: 0.0,
        escape_dir_ks: DVec3::ZERO,
        order: crossings,
        steps: accepted,
        max_radial_residual: max_rad_res,
        max_carter_drift: max_carter_res,
    }
}

/// Convenience helper: trace a single pixel from `Camera` using the CPU `f64` oracle.
pub fn trace_oracle_pixel(
    camera: &Camera,
    params: KerrParams,
    ndc_x: f64,
    ndc_y: f64,
    aspect: f64,
    disk_enabled: bool,
) -> OracleRayResult {
    let tetrad = camera.zamo_tetrad(params.spin);
    let local_dir = camera.pixel_local_dir(ndc_x, ndc_y, aspect);
    let bl_init = tetrad.init_bl_ray(local_dir, params.spin);
    trace_oracle_ray(bl_init, params, tetrad.alpha, disk_enabled)
}

/// Exact Kerr critical shadow curve in finite-distance ZAMO screen angles `(alpha_z, beta_z)`.
pub fn finite_distance_kerr_shadow_angle(
    r_ph: f64,
    a: f64,
    r_obs: f64,
    theta_obs: f64,
) -> Option<(f64, f64)> {
    let rp_minus_1 = r_ph - 1.0;

    if a.abs() < 1e-12 {
        return None;
    }
    let xi = -(r_ph * r_ph * (r_ph - 3.0) + a * a * (r_ph + 1.0)) / (a * rp_minus_1);
    let eta = -(r_ph * r_ph * r_ph * (r_ph * (r_ph - 3.0).powi(2) - 4.0 * a * a))
        / (a * a * rp_minus_1 * rp_minus_1);

    let sin_t = theta_obs.sin();
    let cos_t = theta_obs.cos();
    let r2 = r_obs * r_obs;
    let a2 = a * a;
    let sigma = r2 + a2 * cos_t * cos_t;
    let delta_obs = delta_factored(r_obs, a);
    let big_a = (r2 + a2) * (r2 + a2) - a2 * delta_obs * sin_t * sin_t;

    let alpha_obs = (sigma * delta_obs / big_a).sqrt();
    let omega_obs = (2.0 * a * r_obs) / big_a;
    let sqrt_g_phiphi = (big_a / sigma).sqrt() * sin_t;

    let denom = 1.0 - omega_obs * xi;
    if denom <= 0.0 {
        return None;
    }
    let e = alpha_obs / denom;
    let lz = e * xi;
    let q = e * e * eta;

    let d_phi = -lz / sqrt_g_phiphi;
    let p_theta2 = q - cos_t * cos_t * ((lz / sin_t).powi(2) - a2 * e * e);
    if p_theta2 < -1e-12 {
        return None;
    }
    let d_theta = p_theta2.max(0.0).sqrt() / sigma.sqrt();
    let d_r2 = 1.0 - d_phi * d_phi - d_theta * d_theta;
    if d_r2 < -1e-12 {
        return None;
    }
    let d_r = -d_r2.max(0.0).sqrt();

    Some((d_phi.atan2(-d_r), d_theta.atan2(-d_r)))
}

/// Exact Schwarzschild critical shadow angular radius at finite static/ZAMO distance `r_obs`.
pub fn finite_distance_schwarzschild_shadow_angle(r_obs: f64) -> f64 {
    let sin_psi = (3.0 * 3.0_f64.sqrt() * (1.0 - 2.0 / r_obs).sqrt()) / r_obs;
    sin_psi.clamp(-1.0, 1.0).asin()
}

/// Exact equatorial null deflection `Delta phi - pi` between `r_start` and `r_end`
/// via Simpson quadrature of the exact Kerr equatorial orbit integral.
pub fn exact_equatorial_deflection(b: f64, a: f64, r_start: f64, r_end: f64) -> f64 {
    let r_pot = |r: f64| -> f64 {
        let p = r * r + a * a - a * b;
        let delta = delta_factored(r, a);
        let ba = b - a;
        p * p - delta * ba * ba
    };

    let mut hi = b.abs().max(r_plus(a) + 0.5);
    while r_pot(hi) <= 0.0 {
        hi *= 1.5;
    }
    // Exact equatorial spherical photon orbit radius for sign(b) = prograde (+b) / retrograde (-b):
    // R(r) attains its local minimum at the photon orbit, so R(r) is strictly increasing on [r_ph, hi].
    let a_eff = if b >= 0.0 { a } else { -a };
    let r_ph = 2.0 * (1.0 + ((2.0 / 3.0) * (-a_eff).acos()).cos());
    let mut lo = r_ph;
    if r_pot(lo) > 0.0 {
        // Fallback bracket if impact parameter is very close to a different root.
        lo = r_plus(a) + 1e-6;
    }
    for _ in 0..90 {
        let mid = 0.5 * (lo + hi);
        if r_pot(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let r_min = hi;

    let integrate_leg = |r_max: f64| -> f64 {
        let s_max = (1.0 - r_min / r_max).max(0.0).sqrt();
        let n = 4096_usize;
        let ds = s_max / (n as f64);
        let integrand = |s: f64| -> f64 {
            let one_m_s2 = 1.0 - s * s;
            let r = r_min / one_m_s2;
            let dr_ds = 2.0 * r_min * s / (one_m_s2 * one_m_s2);
            let rp_val = r_pot(r);
            let p = r * r + a * a - a * b;
            let delta = delta_factored(r, a);
            let dphi_dlm = (a * p) / delta + b - a;
            if s < 1e-9 {
                let dr_pot = 4.0 * r_min * (r_min * r_min + a * a - a * b)
                    - 2.0 * (r_min - 1.0) * (b - a) * (b - a);
                let limit_ratio = 2.0 * r_min.sqrt() / dr_pot.sqrt();
                dphi_dlm * limit_ratio
            } else {
                dphi_dlm * dr_ds / rp_val.max(1e-30).sqrt()
            }
        };
        let mut sum = integrand(0.0) + integrand(s_max);
        for i in 1..n {
            let s = (i as f64) * ds;
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += w * integrand(s);
        }
        sum * ds / 3.0
    };

    let total_phi = integrate_leg(r_start) + integrate_leg(r_end);
    total_phi.abs() - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test2_weak_field_deflection_matches_exact_quadrature() {
        let r_start = 10_000.0;
        let r_end = 10_000.0;
        let params_s = KerrParams::new(0.0, 20.0, r_end);

        for b in [20.0_f64, 50.0] {
            let e = 1.0;
            let lz = b;
            let p = r_start * r_start * e;
            let delta = delta_factored(r_start, 0.0);
            let r_pot = p * p - delta * lz * lz;
            let init = BlInitialRay {
                r: r_start,
                u: 0.0,
                phi_bl: 0.0,
                dr_dlm: r_pot.sqrt(),
                du_dlm: 0.0,
                energy: e,
                lz,
                q: 0.0,
            };
            let res = trace_oracle_ray(init, params_s, 1.0, false);
            assert_eq!(res.status, RayStatus::Escaped);
            let got = res.phi_bl.abs() - std::f64::consts::PI;
            let exact = exact_equatorial_deflection(b, 0.0, r_start, r_end);
            let rel_err = (got - exact).abs() / exact.abs();
            assert!(
                rel_err < 1e-8,
                "Schwarzschild b={b}: got={got}, exact={exact}, rel_err={rel_err}"
            );
        }

        // Kerr frame-dragging prograde vs retrograde asymmetry at a = +0.9, b = +/-30M
        let a = 0.9;
        let b_mag = 30.0;
        let params_k = KerrParams::new(a, 20.0, r_end);
        let mut defl = [0.0_f64; 2];
        for (idx, b) in [b_mag, -b_mag].iter().enumerate() {
            let e = 1.0;
            let lz = *b;
            let p = (r_start * r_start + a * a) * e - a * lz;
            let delta = delta_factored(r_start, a);
            let r_pot = p * p - delta * (lz - a * e).powi(2);
            let init = BlInitialRay {
                r: r_start,
                u: 0.0,
                phi_bl: 0.0,
                dr_dlm: r_pot.sqrt(),
                du_dlm: 0.0,
                energy: e,
                lz,
                q: 0.0,
            };
            let res = trace_oracle_ray(init, params_k, 1.0, false);
            assert_eq!(res.status, RayStatus::Escaped);
            let got = res.phi_bl.abs() - std::f64::consts::PI;
            let exact = exact_equatorial_deflection(*b, a, r_start, r_end);
            let rel_err = (got - exact).abs() / exact.abs();
            assert!(
                rel_err < 1e-8,
                "Kerr a={a} b={b}: got={got}, exact={exact}, rel_err={rel_err}"
            );
            defl[idx] = got;
        }
        let asym = defl[1] - defl[0];
        let asym_2pn =
            8.0 * a / (b_mag * b_mag) + 20.0 * std::f64::consts::PI * a / (b_mag * b_mag * b_mag);
        assert!(asym > 0.0, "expected retrograde deflection > prograde");
        assert!(
            (asym - asym_2pn).abs() / asym_2pn < 0.08,
            "Kerr asymmetry={asym} vs 2PN={asym_2pn}"
        );
    }

    #[test]
    fn test3_finite_distance_shadow_boundary_schwarzschild_and_kerr() {
        let r_obs = 25.0;
        let cam_s = Camera {
            radius: r_obs,
            inclination: std::f64::consts::FRAC_PI_2,
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        };
        let params_s = KerrParams::new(0.0, 20.0, 1000.0);
        let expected_psi = finite_distance_schwarzschild_shadow_angle(r_obs);
        let mut lo = expected_psi - 0.02;
        let mut hi = expected_psi + 0.02;
        let tet_s = cam_s.zamo_tetrad(0.0);
        for _ in 0..32 {
            let mid = 0.5 * (lo + hi);
            let dir = DVec3::new(-mid.cos(), 0.0, mid.sin());
            let res = trace_oracle_ray(tet_s.init_bl_ray(dir, 0.0), params_s, tet_s.alpha, false);
            if res.status == RayStatus::Captured {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let got_psi = 0.5 * (lo + hi);
        assert!(
            (got_psi - expected_psi).abs() < 1e-6,
            "Schwarzschild shadow psi got={got_psi} vs exact={expected_psi}"
        );

        for a in [0.998_f64, -0.998] {
            let params = KerrParams::new(a, 20.0, 1000.0);
            let tet = cam_s.zamo_tetrad(a);
            let r_ph_pro = 2.0 * (1.0 + ((2.0 / 3.0) * (-a.abs()).acos()).cos());
            let r_ph_ret = 2.0 * (1.0 + ((2.0 / 3.0) * a.abs().acos()).cos());
            for k in 1..=5 {
                let frac = (k as f64) / 6.0;
                let r_ph = r_ph_pro + frac * (r_ph_ret - r_ph_pro);
                let Some((ang_x, ang_y)) =
                    finite_distance_kerr_shadow_angle(r_ph, a, r_obs, std::f64::consts::FRAC_PI_2)
                else {
                    continue;
                };
                let eps = 3e-4;
                let dir_in = DVec3::new(
                    -1.0,
                    -((1.0 - eps) * ang_y).tan(),
                    ((1.0 - eps) * ang_x).tan(),
                )
                .normalize();
                let dir_out = DVec3::new(
                    -1.0,
                    -((1.0 + eps) * ang_y).tan(),
                    ((1.0 + eps) * ang_x).tan(),
                )
                .normalize();
                let res_in = trace_oracle_ray(tet.init_bl_ray(dir_in, a), params, tet.alpha, false);
                let res_out =
                    trace_oracle_ray(tet.init_bl_ray(dir_out, a), params, tet.alpha, false);
                assert_eq!(res_in.status, RayStatus::Captured);
                assert_eq!(res_out.status, RayStatus::Escaped);
            }
        }
    }

    #[test]
    fn test4_exact_on_axis_finite_redshift_and_logarithmic_divergence() {
        // 1. Exact on-axis (theta_obs = 0, L_z = 0) Schwarzschild finite-distance redshift:
        // At r_obs = 30M, shadow edge is ndc_y ~ 0.42; pick rays hitting disk r in [6M, 20M].
        let r_obs = 30.0;
        let cam = Camera {
            radius: r_obs,
            inclination: 0.0,
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        };
        let params = KerrParams::new(0.0, 20.0, 1000.0);
        for ndc_y in [0.6_f64, 0.75, 0.9] {
            let res = trace_oracle_pixel(&cam, params, 0.0, ndc_y, 1.0, true);
            assert_eq!(res.status, RayStatus::DiskHit);
            let exact_g = ((1.0 - 3.0 / res.r_hit) / (1.0 - 2.0 / r_obs)).sqrt();
            let rel_err = (res.redshift_g - exact_g).abs() / exact_g;
            assert!(
                rel_err < 1e-10,
                "on-axis Schwarzschild g={} vs exact={exact_g} (r={})",
                res.redshift_g,
                res.r_hit
            );
        }

        // 2. Logarithmic strong-deflection slope as b -> b_c^+ in Schwarzschild:
        let b_c = 3.0 * 3.0_f64.sqrt();
        let r_end = 10_000.0;
        let eps1 = 1e-4;
        let eps2 = 1e-6;
        let d1 = exact_equatorial_deflection(b_c + eps1, 0.0, r_end, r_end);
        let d2 = exact_equatorial_deflection(b_c + eps2, 0.0, r_end, r_end);
        let slope = (d2 - d1) / (eps1 / eps2).ln();
        assert!(
            (slope - 1.0).abs() < 0.02,
            "strong-deflection log slope={slope}, expected ~1.0"
        );
    }

    #[test]
    fn test5_oracle_invariants_and_subring_disk_edges() {
        // 1. Invariant conservation across Kerr (a = 0.95) and Schwarzschild (a = 0.0).
        let cam_inc = Camera {
            radius: 25.0,
            inclination: 80.0_f64.to_radians(),
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        };
        for a in [0.0_f64, 0.95] {
            let params = KerrParams::new(a, 12.0, 1000.0);
            let mut saw_n0 = false;
            let mut saw_n1 = false;
            for i in 0..400 {
                let ndc_y = -0.8 + 1.6 * (i as f64) / 399.0;
                let res = trace_oracle_pixel(&cam_inc, params, 0.12, ndc_y, 1.0, true);
                assert!(
                    res.max_radial_residual < 1e-9,
                    "a={a} radial residual too large: {}",
                    res.max_radial_residual
                );
                assert!(
                    res.max_carter_drift < 1e-9,
                    "a={a} Carter residual too large: {}",
                    res.max_carter_drift
                );
                if res.status == RayStatus::DiskHit {
                    if res.order == 0 {
                        saw_n0 = true;
                    } else if res.order == 1 {
                        saw_n1 = true;
                    }
                }
            }
            assert!(saw_n0, "a={a}: expected primary n=0 disk hits");
            assert!(saw_n1, "a={a}: expected lensed secondary n=1 disk hits");
        }

        // 2. Direct logarithmic targeting of n = 0, 1, 2 subring disk hits near critical curve
        //    (face-on Schwarzschild at r_obs = 25M: every crossing happens at phi_pol = pi/2 + n*pi).
        let r_obs = 25.0;
        let cam_face = Camera {
            radius: r_obs,
            inclination: 0.0,
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        };
        let params_s = KerrParams::new(0.0, 20.0, 1000.0);
        let tet = cam_face.zamo_tetrad(0.0);
        let psi_c = finite_distance_schwarzschild_shadow_angle(r_obs);
        let mut saw_orders = [false; 3];
        for i in 0..300 {
            // Logarithmic sweep from psi_c + 1e-6 rad out to psi_c + 0.4 rad.
            let log_off = -6.0 + 5.6 * (i as f64) / 299.0;
            let psi = psi_c + 10.0_f64.powf(log_off);
            let dir = DVec3::new(-psi.cos(), psi.sin(), 0.0);
            let res = trace_oracle_ray(tet.init_bl_ray(dir, 0.0), params_s, tet.alpha, true);
            assert!(res.max_radial_residual < 1e-9);
            if res.status == RayStatus::DiskHit && (res.order as usize) < 3 {
                saw_orders[res.order as usize] = true;
            }
        }
        assert!(
            saw_orders == [true, true, true],
            "expected n=0, n=1, and n=2 subring disk hits, got {saw_orders:?}"
        );
    }
}
