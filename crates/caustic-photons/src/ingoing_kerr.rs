//! Exact equatorial (`z = 0`) Ingoing Cartesian Kerr-Schild metric and Hamiltonian
//! for forward-in-time relativistic photon flight.
//!
//! Conventions (`G = c = M = 1`, signature `(-, +, +, +)`):
//! - Ingoing KS coordinates `(t_in, x, y)` satisfy:
//!   `dt_in = dt_BL + (2r / Delta) dr`, `dphi_in = dphi_BL + (a / Delta) dr`,
//!   `x + i y = (r + i a) exp(i phi_in)` on `z = 0`, so `r = sqrt(x^2 + y^2 - a^2)`.
//! - Null covector `l_mu = (1, (r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2))`.
//! - Future-directed infalling photons cross the **future** event horizon `H^+` smoothly
//!   without coordinate blowup.

use glam::{DMat3, DVec2, DVec3};

/// Outer event horizon radius `r_+ = 1 + sqrt((1 - a)(1 + a))`.
pub fn r_plus(a: f64) -> f64 {
    1.0 + ((1.0 - a) * (1.0 + a)).max(0.0).sqrt()
}

/// Inner Cauchy horizon radius `r_- = 1 - sqrt((1 - a)(1 + a))`.
pub fn r_minus(a: f64) -> f64 {
    1.0 - ((1.0 - a) * (1.0 + a)).max(0.0).sqrt()
}

/// Factored `Delta(r) = (r - r_+)(r - r_-)`.
pub fn delta_factored(r: f64, a: f64) -> f64 {
    (r - r_plus(a)) * (r - r_minus(a))
}

/// Equatorial Boyer-Lindquist radius `r = sqrt(x^2 + y^2 - a^2)` from Cartesian `(x, y)`.
pub fn ingoing_ks_radius(pos: DVec2, a: f64) -> f64 {
    (pos.length_squared() - a * a).max(1e-18).sqrt()
}

/// Ingoing Kerr-Schild equatorial `r`, scalar `H = 1/r`, and spatial null covector `(l_x, l_y)`.
pub fn ingoing_ks_hl(pos: DVec2, a: f64) -> (f64, f64, DVec2) {
    let r = ingoing_ks_radius(pos, a);
    let d = r * r + a * a;
    let h = 1.0 / r;
    let l = DVec2::new((r * pos.x + a * pos.y) / d, (r * pos.y - a * pos.x) / d);
    (r, h, l)
}

/// Equatorial Ingoing Cartesian KS covariant metric `g_{mu nu}` in `(t, x, y)`.
pub fn ingoing_ks_metric_2d(pos: DVec2, a: f64) -> DMat3 {
    let (_, h, l) = ingoing_ks_hl(pos, a);
    let lv = [1.0, l.x, l.y];
    let eta = [-1.0, 1.0, 1.0];
    let mut cols = [DVec3::ZERO; 3];
    for c in 0..3 {
        let mut col = [0.0; 3];
        for r in 0..3 {
            let diag = if r == c { eta[r] } else { 0.0 };
            col[r] = diag + 2.0 * h * lv[r] * lv[c];
        }
        cols[c] = DVec3::from_array(col);
    }
    DMat3::from_cols(cols[0], cols[1], cols[2])
}

/// Equatorial Ingoing Cartesian KS null Hamiltonian `H_geo = 0.5 * (-E^2 + |p|^2) - H (E + l.p)^2`.
pub fn ingoing_ks_hamiltonian(pos: DVec2, p: DVec2, energy: f64, a: f64) -> f64 {
    let (_, h, l) = ingoing_ks_hl(pos, a);
    let k = energy + l.dot(p);
    0.5 * (-energy * energy + p.length_squared()) - h * k * k
}

/// Future-directed equatorial Hamiltonian RHS `(dx/dlambda, dp/dlambda, dt/dlambda)` in Ingoing KS.
pub fn ingoing_ks_rhs(pos: DVec2, p: DVec2, energy: f64, a: f64) -> (DVec2, DVec2, f64) {
    let r = ingoing_ks_radius(pos, a);
    let r2 = r * r;
    let d = r2 + a * a;
    let inv_d = 1.0 / d;
    let h = 1.0 / r;
    let l = DVec2::new(
        (r * pos.x + a * pos.y) * inv_d,
        (r * pos.y - a * pos.x) * inv_d,
    );

    let k = energy + l.dot(p);
    let two_hk = 2.0 * h * k;
    let dx = p - two_hk * l;
    let dt = energy + two_hk;

    // On z = 0: r^2 = x^2 + y^2 - a^2 => grad r = pos / r, H = 1/r => grad H = -pos / r^3.
    let grad_r = pos / r;
    let grad_h = -pos / (r2 * r);

    let p_dl_fixed = DVec2::new((r * p.x - a * p.y) * inv_d, (a * p.x + r * p.y) * inv_d);
    let p_dl_dr = (p.x * (pos.x - 2.0 * r * l.x) + p.y * (pos.y - 2.0 * r * l.y)) * inv_d;
    let p_grad_l = p_dl_fixed + p_dl_dr * grad_r;

    let dp = (k * k) * grad_h + two_hk * p_grad_l;
    (dx, dp, dt)
}

/// Local equatorial ZAMO lapse `alpha` and frame-dragging angular velocity `omega` at BL radius `r`.
pub fn equatorial_zamo_alpha_omega(r: f64, a: f64) -> (f64, f64) {
    let r2 = r * r;
    let a2 = a * a;
    let delta = delta_factored(r, a).max(1e-12);
    let big_a = (r2 + a2) * (r2 + a2) - a2 * delta;
    let alpha = (r2 * delta / big_a).max(1e-12).sqrt();
    let omega = (2.0 * a * r) / big_a;
    (alpha, omega)
}

/// Construct a future-directed null covector `(E, p)` at equatorial Ingoing KS `pos`
/// launched along a unit 2D Euclidean aim direction `aim_dir` in the local ZAMO frame.
///
/// Normalized so local ZAMO frequency `nu_emit = -(p_mu u^mu_ZAMO) = 1.0`.
pub fn init_forward_kerr_photon(pos: DVec2, aim_dir: DVec2, a: f64) -> (f64, DVec2) {
    let d_aim = aim_dir.normalize();
    let r = ingoing_ks_radius(pos, a);
    let g = ingoing_ks_metric_2d(pos, a);
    let (alpha, omega) = equatorial_zamo_alpha_omega(r, a);

    // Covariant ZAMO 4-velocity u = (1, -omega y, omega x) / alpha.
    let e_t = DVec3::new(1.0, -omega * pos.y, omega * pos.x) / alpha;

    // Orthonormal spatial basis in local ZAMO equatorial slice:
    // e_phi along xi_phi = (0, -y, x), e_r along (dr)^sharp orthogonal to e_t and e_phi.
    let xi_phi = DVec3::new(0.0, -pos.y, pos.x);
    let e_phi_len = xi_phi.dot(g * xi_phi).max(1e-18).sqrt();
    let e_phi = xi_phi / e_phi_len;

    // Outward radial direction orthogonal to e_t and e_phi in metric g.
    let raw_r = DVec3::new(0.0, pos.x, pos.y);
    let er_orth = raw_r + (raw_r.dot(g * e_t)) * e_t - (raw_r.dot(g * e_phi)) * e_phi;
    let e_r = er_orth / er_orth.dot(g * er_orth).max(1e-18).sqrt();

    // Project desired Cartesian aim direction onto local radial/azimuthal unit vectors:
    let rad_hat = pos.normalize();
    let phi_hat = DVec2::new(-rad_hat.y, rad_hat.x);
    let c_r = d_aim.dot(rad_hat);
    let c_phi = d_aim.dot(phi_hat);

    // Future-directed forward photon travels ALONG +aim in ZAMO frame: p^mu = e_t + c_r e_r + c_phi e_phi.
    let p_up = e_t + c_r * e_r + c_phi * e_phi;
    let p_down = g * p_up;
    (-p_down.x, DVec2::new(p_down.y, p_down.z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use caustic_blackhole::oracle::exact_equatorial_deflection;

    #[test]
    fn ingoing_ks_metric_pullback_equals_boyer_lindquist() {
        let cases = [
            (0.0, 5.0, 0.3),
            (0.7, 3.8, 1.1),
            (0.998, 2.2, -0.8),
            (-0.998, 3.0, 2.0),
        ];
        for (a, r, phi_bl) in cases {
            let rp = r_plus(a);
            let rm = r_minus(a);
            let twist = if a.abs() > 1e-15 {
                (a / (rp - rm)) * ((r - rp) / (r - rm)).ln()
            } else {
                0.0
            };
            let phi_in = phi_bl + twist;
            let (sp, cp) = phi_in.sin_cos();
            // x + i y = (r + i a) exp(i phi_in) => x = r cp - a sp, y = r sp + a cp
            let pos = DVec2::new(r * cp - a * sp, r * sp + a * cp);
            assert!((ingoing_ks_radius(pos, a) - r).abs() < 1e-12);

            let delta = delta_factored(r, a);
            let dx_dphi = -r * sp - a * cp;
            let dy_dphi = r * cp - a * sp;
            let col_t = DVec3::new(1.0, 0.0, 0.0);
            let col_r = DVec3::new(
                2.0 * r / delta,
                cp + (a / delta) * dx_dphi,
                sp + (a / delta) * dy_dphi,
            );
            let col_phi = DVec3::new(0.0, dx_dphi, dy_dphi);
            let jac = DMat3::from_cols(col_t, col_r, col_phi);

            let g_ks = ingoing_ks_metric_2d(pos, a);
            let pullback = jac.transpose() * g_ks * jac;

            let g_tt = -(1.0 - 2.0 / r);
            let g_tphi = -2.0 * a / r;
            let g_rr = r * r / delta;
            let g_phiphi = r * r + a * a + 2.0 * a * a / r;
            let g_bl = DMat3::from_cols(
                DVec3::new(g_tt, 0.0, g_tphi),
                DVec3::new(0.0, g_rr, 0.0),
                DVec3::new(g_tphi, 0.0, g_phiphi),
            );

            let max_err = (pullback - g_bl)
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_err < 1e-12,
                "Ingoing KS pullback err={max_err} for a={a}"
            );

            let (_, omega) = equatorial_zamo_alpha_omega(r, a);
            if a > 0.0 {
                assert!(omega > 0.0);
            } else if a < 0.0 {
                assert!(omega < 0.0);
            }
        }
    }

    #[test]
    fn ingoing_ks_rhs_matches_finite_differences() {
        let a = 0.92;
        let pos = DVec2::new(3.5, -2.1);
        let p = DVec2::new(-0.7, 0.45);
        let energy = 1.05;
        let (dx, dp, _) = ingoing_ks_rhs(pos, p, energy, a);
        let eps = 1e-6;

        for axis in 0..2 {
            let mut p_hi = p;
            let mut p_lo = p;
            p_hi[axis] += eps;
            p_lo[axis] -= eps;
            let fd_dx = (ingoing_ks_hamiltonian(pos, p_hi, energy, a)
                - ingoing_ks_hamiltonian(pos, p_lo, energy, a))
                / (2.0 * eps);
            assert!((dx[axis] - fd_dx).abs() < 1e-9);

            let mut x_hi = pos;
            let mut x_lo = pos;
            x_hi[axis] += eps;
            x_lo[axis] -= eps;
            let fd_dp = -(ingoing_ks_hamiltonian(x_hi, p, energy, a)
                - ingoing_ks_hamiltonian(x_lo, p, energy, a))
                / (2.0 * eps);
            assert!((dp[axis] - fd_dp).abs() < 1e-9);
        }
    }

    #[test]
    fn forward_ingoing_ks_photon_matches_exact_equatorial_deflection() {
        // Trace forward-in-time photons in Ingoing KS RK4 and verify deflection against
        // exact equatorial quadrature for Schwarzschild (a=0) and Kerr (a=+0.9, prograde/retrograde).
        let cases = [(0.0_f64, 20.0_f64), (0.9, 30.0), (0.9, -30.0)];
        let r_start = 400.0_f64;
        let r_end = 400.0_f64;

        for (a, b) in cases {
            // Start photon at x = +r_start, y = b (or exact BL impact parameter b = L_z / E):
            // Initialize at pos = (-r_start, b) aiming +x, then read exact conserved b = L_z / E!
            let mut pos = DVec2::new(-(r_start * r_start - b * b).sqrt(), b);
            let (energy, mut p) = init_forward_kerr_photon(pos, DVec2::new(1.0, 0.0), a);
            let lz = pos.x * p.y - pos.y * p.x;
            let b_exact = lz / energy;

            let r0 = ingoing_ks_radius(pos, a);
            let start_dir = {
                let (dx0, _, _) = ingoing_ks_rhs(pos, p, energy, a);
                dx0.normalize()
            };

            let mut r_prev = r0;
            let mut end_dir = start_dir;
            for _ in 0..8000 {
                let r_cur = ingoing_ks_radius(pos, a);
                let h = (0.015 * r_cur).clamp(0.01, 12.0);
                let (dx1, dp1, _) = ingoing_ks_rhs(pos, p, energy, a);
                let (dx2, dp2, _) =
                    ingoing_ks_rhs(pos + 0.5 * h * dx1, p + 0.5 * h * dp1, energy, a);
                let (dx3, dp3, _) =
                    ingoing_ks_rhs(pos + 0.5 * h * dx2, p + 0.5 * h * dp2, energy, a);
                let (dx4, dp4, _) = ingoing_ks_rhs(pos + h * dx3, p + h * dp3, energy, a);
                pos += (h / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
                p += (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
                let r_next = ingoing_ks_radius(pos, a);
                if r_next >= r_end && r_next > r_prev {
                    let (dxf, _, _) = ingoing_ks_rhs(pos, p, energy, a);
                    end_dir = dxf.normalize();
                    break;
                }
                r_prev = r_next;
            }

            // Asymptotic deflection angle between incoming and outgoing tangents vs exact quadrature
            // (exact_equatorial_deflection measures azimuthal Delta_phi - pi; at finite r_start=r_end=400,
            // tangent angle differs from coordinate phi swing by 2*asin(b/r_end)).
            let got_tang = start_dir.dot(end_dir).clamp(-1.0, 1.0).acos();
            let exact_phi = exact_equatorial_deflection(b_exact, a, r0, r_end);
            let exact_tang = exact_phi + 2.0 * (b_exact.abs() / r_end).asin();
            let rel_err = (got_tang - exact_tang).abs() / exact_tang;
            assert!(
                rel_err < 5e-3,
                "Ingoing KS forward deflection a={a} b={b_exact:.2}: got={got_tang:.5} vs exact={exact_tang:.5} (rel={rel_err:.2e})"
            );
        }
    }
}
