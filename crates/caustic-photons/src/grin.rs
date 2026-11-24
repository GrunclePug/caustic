//! Graded-index (GRIN) continuous eikonal optics physics and CPU `f64` oracle.
//!
//! Solves the exact eikonal Hamiltonian derived from Fermat's principle
//! `delta integral n(x, lambda) ds = 0`:
//!
//!   H(x, p) = 1/2 * (|p|^2 - n(x, lambda)^2) = 0
//!   dx / dtau = p
//!   dp / dtau = 1/2 * grad(n(x, lambda)^2)
//!
//! Physical arc length is `ds = n * dtau` and coordinate time at `c = 1` is
//! `dt = n * ds = n^2 * dtau`, matching group-velocity slowing inside high-index media.

use glam::DVec2;

/// Continuous refractive-index field model.
#[derive(Debug, Clone, Copy)]
pub enum GrinField {
    /// Analytical Luneburg sphere `n^2(r) = 1 + c(lambda) * (1 - r^2 / R^2)` for `r < R`.
    /// At `c = 1` (`dispersion_scale = 0, strength = 1`), every parallel ray focuses to `(+R, 0)`
    /// with zero spherical aberration.
    Luneburg {
        radius: f64,
        strength: f64,
        dispersion_scale: f64,
    },
    /// Smooth chromatic inverse-square optical black hole `n^2(r) = 1 + (b_c(lambda) / r)^2`.
    /// Rays with impact parameter `b < b_c(lambda)` spiral inevitably into `core_radius`;
    /// rays with `b > b_c(lambda)` deflect by exact angle `pi * (b / sqrt(b^2 - b_c^2) - 1)`.
    OpticalBlackHole {
        bc_base: f64,
        core_radius: f64,
        dispersion_scale: f64,
    },
    /// Exponential thermal atmospheric mirage above `y_ground`:
    /// `n^2(y) = n_air^2 - A * exp(-(y - y_ground) / scale_h)`.
    Mirage {
        y_ground: f64,
        scale_h: f64,
        strength: f64,
        dispersion_scale: f64,
    },
}

/// Chromatic dispersion factor relative to `550 nm` reference wavelength:
/// negative for red (`~ -0.38` at `700 nm`), zero at `550 nm`, positive for violet (`~ +0.89` at `400 nm`).
#[inline]
pub fn chromatic_offset(lambda_nm: f64) -> f64 {
    let lam_um = (lambda_nm * 1e-3).clamp(0.35, 0.85);
    (0.55 * 0.55) / (lam_um * lam_um) - 1.0
}

impl GrinField {
    /// Evaluate `(n^2, 0.5 * grad(n^2))` at position `pos` and wavelength `lambda_nm`.
    #[inline]
    pub fn eval(self, pos: DVec2, lambda_nm: f64) -> (f64, DVec2) {
        let chi = chromatic_offset(lambda_nm);
        match self {
            Self::Luneburg {
                radius,
                strength,
                dispersion_scale,
            } => {
                let r2 = pos.length_squared();
                let big_r2 = radius * radius;
                if r2 >= big_r2 {
                    (1.0, DVec2::ZERO)
                } else {
                    let c = (strength * (1.0 + 0.16 * dispersion_scale * chi)).max(0.05);
                    let n2 = 1.0 + c * (1.0 - r2 / big_r2);
                    let half_grad = -(c / big_r2) * pos;
                    (n2, half_grad)
                }
            }
            Self::OpticalBlackHole {
                bc_base,
                dispersion_scale,
                ..
            } => {
                let bc = self.optical_bh_bc(bc_base, dispersion_scale, chi);
                let bc2 = bc * bc;
                let r2 = pos.length_squared().max(1e-8);
                let n2 = 1.0 + bc2 / r2;
                let half_grad = -(bc2 / (r2 * r2)) * pos;
                (n2, half_grad)
            }
            Self::Mirage {
                y_ground,
                scale_h,
                strength,
                dispersion_scale,
            } => {
                let n_air = (1.08 + 0.045 * dispersion_scale * chi).max(1.01);
                let amp = (0.30 * strength * (1.0 + 0.22 * dispersion_scale * chi))
                    .clamp(0.02, n_air * n_air - 0.25);
                let dy = (pos.y - y_ground).max(0.0);
                let ex = (-dy / scale_h).exp();
                let n2 = n_air * n_air - amp * ex;
                let half_grad = DVec2::new(0.0, (amp / (2.0 * scale_h)) * ex);
                (n2, half_grad)
            }
        }
    }

    #[inline]
    fn optical_bh_bc(self, bc_base: f64, dispersion_scale: f64, chi: f64) -> f64 {
        (bc_base * (1.0 + 0.24 * dispersion_scale * chi)).max(0.2)
    }

    /// True if position lies inside an absorbing core or below ground.
    #[inline]
    pub fn is_absorbed(self, pos: DVec2) -> bool {
        match self {
            Self::Luneburg { .. } => false,
            Self::OpticalBlackHole { core_radius, .. } => {
                pos.length_squared() <= core_radius * core_radius
            }
            Self::Mirage { y_ground, .. } => pos.y <= y_ground,
        }
    }
}

/// Outcome of CPU `f64` GRIN geodesic trace.
#[derive(Debug, Clone, Copy)]
pub struct GrinTraceResult {
    pub pos: DVec2,
    pub p: DVec2,
    pub absorbed: bool,
    pub max_hamiltonian_err: f64,
}

/// Trace a single GRIN ray with RK4 in `f64` from `pos0` aiming along unit `dir0`.
pub fn trace_grin_rk4(
    field: GrinField,
    pos0: DVec2,
    dir0: DVec2,
    lambda_nm: f64,
    bound: f64,
    max_steps: usize,
) -> GrinTraceResult {
    let dir = dir0.normalize();
    let (n2_0, _) = field.eval(pos0, lambda_nm);
    let mut pos = pos0;
    let mut p = dir * n2_0.sqrt();
    let mut max_h_err = 0.0_f64;

    for _ in 0..max_steps {
        if field.is_absorbed(pos) {
            return GrinTraceResult {
                pos,
                p,
                absorbed: true,
                max_hamiltonian_err: max_h_err,
            };
        }
        let (n2_cur, _) = field.eval(pos, lambda_nm);
        let n_cur = n2_cur.sqrt();
        let h_err = (0.5 * (p.length_squared() - n2_cur)).abs();
        max_h_err = max_h_err.max(h_err);

        let h = match field {
            GrinField::Luneburg { radius, .. } => {
                let r = pos.length();
                if r < radius * 1.15 { 0.015 } else { 0.05 }
            }
            GrinField::OpticalBlackHole { .. } => (0.015 * pos.length() / n_cur).clamp(0.001, 0.08),
            GrinField::Mirage { .. } => 0.02,
        };

        let (_, g1) = field.eval(pos, lambda_nm);
        let k1x = p;
        let k1p = g1;

        let p2 = p + 0.5 * h * k1p;
        let (_, g2) = field.eval(pos + 0.5 * h * k1x, lambda_nm);
        let k2x = p2;
        let k2p = g2;

        let p3 = p + 0.5 * h * k2p;
        let (_, g3) = field.eval(pos + 0.5 * h * k2x, lambda_nm);
        let k3x = p3;
        let k3p = g3;

        let p4 = p + h * k3p;
        let (_, g4) = field.eval(pos + h * k3x, lambda_nm);
        let k4x = p4;
        let k4p = g4;

        pos += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        p += (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);

        if pos.x.abs() > bound || pos.y.abs() > bound {
            break;
        }
    }

    GrinTraceResult {
        pos,
        p,
        absorbed: field.is_absorbed(pos),
        max_hamiltonian_err: max_h_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luneburg_lens_focuses_parallel_rays_to_exact_opposite_rim() {
        let r = 3.2;
        let field = GrinField::Luneburg {
            radius: r,
            strength: 1.0,
            dispersion_scale: 0.0,
        };
        for frac in [0.15_f64, 0.45, 0.75, 0.90] {
            let y0 = frac * r;
            let mut pos = DVec2::new(-1.5 * r, y0);
            let mut p = DVec2::new(1.0, 0.0);
            let mut crossed_y = None;
            let h = 0.005;
            for _ in 0..4000 {
                let prev = pos;
                let (_, g1) = field.eval(pos, 550.0);
                let p2 = p + 0.5 * h * g1;
                let (_, g2) = field.eval(pos + 0.5 * h * p, 550.0);
                let p3 = p + 0.5 * h * g2;
                let (_, g3) = field.eval(pos + 0.5 * h * p2, 550.0);
                let p4 = p + h * g3;
                let (_, g4) = field.eval(pos + h * p3, 550.0);
                pos += (h / 6.0) * (p + 2.0 * p2 + 2.0 * p3 + p4);
                p += (h / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4);
                if prev.x < r && pos.x >= r {
                    let t = (r - prev.x) / (pos.x - prev.x);
                    crossed_y = Some(prev.y + t * (pos.y - prev.y));
                    break;
                }
            }
            let y_focus = crossed_y.expect("ray crosses x = +R");
            assert!(
                y_focus.abs() < 2e-3,
                "Luneburg ray at y0={y0:.2} missed rim focus (+R, 0): y_focus={y_focus:.3e}"
            );
        }
    }

    #[test]
    fn optical_black_hole_matches_exact_deflection_and_chromatic_capture() {
        let bc_base = 2.4;
        let field = GrinField::OpticalBlackHole {
            bc_base,
            core_radius: 0.65,
            dispersion_scale: 1.0,
        };

        // 1. Exact analytical deflection at 550 nm (where bc == bc_base) for b > bc.
        for b in [3.2_f64, 4.5, 6.0] {
            let res = trace_grin_rk4(
                field,
                DVec2::new(-80.0, b),
                DVec2::new(1.0, 0.0),
                550.0,
                100.0,
                20_000,
            );
            assert!(!res.absorbed, "b={b} > bc={bc_base} must escape");
            let d = res.p.normalize();
            // Deflected clockwise (downward) from +x.
            let measured = (-d.y).atan2(d.x).rem_euclid(std::f64::consts::TAU);
            let exact = std::f64::consts::PI * (b / (b * b - bc_base * bc_base).sqrt() - 1.0);
            let err = (measured - exact).abs();
            assert!(
                err < 2e-2,
                "b={b}: measured={measured:.5}, exact={exact:.5}, err={err:.3e}"
            );
            assert!(
                res.max_hamiltonian_err < 1e-4,
                "Hamiltonian drift too large: {:.3e}",
                res.max_hamiltonian_err
            );
        }

        // 2. Chromatic capture split: at b = 2.55, red (680 nm, bc ~ 2.20) escapes while
        //    violet (400 nm, bc ~ 2.91) spirals into the absorbing core!
        let b_split = 2.55;
        let red = trace_grin_rk4(
            field,
            DVec2::new(-16.0, b_split),
            DVec2::new(1.0, 0.0),
            680.0,
            30.0,
            10_000,
        );
        let violet = trace_grin_rk4(
            field,
            DVec2::new(-16.0, b_split),
            DVec2::new(1.0, 0.0),
            400.0,
            30.0,
            10_000,
        );
        assert!(!red.absorbed, "red ray at b={b_split} should escape");
        assert!(
            violet.absorbed,
            "violet ray at b={b_split} should be captured"
        );
    }

    #[test]
    fn thermal_mirage_bends_shallow_rays_upward_and_conserves_px() {
        let field = GrinField::Mirage {
            y_ground: -2.2,
            scale_h: 1.35,
            strength: 1.0,
            dispersion_scale: 1.0,
        };
        let pos0 = DVec2::new(-6.8, 1.0);
        let ang = -0.25_f64;
        let dir0 = DVec2::new(ang.cos(), ang.sin());
        let (n2_0, _) = field.eval(pos0, 550.0);
        let px_expected = dir0.x * n2_0.sqrt();

        let res = trace_grin_rk4(field, pos0, dir0, 550.0, 12.0, 4000);
        assert!(!res.absorbed, "shallow mirage ray should curve back upward");
        assert!(
            res.p.y > 0.0,
            "mirage ray should exit heading upward, py={}",
            res.p.y
        );
        assert!(
            (res.p.x - px_expected).abs() < 1e-7,
            "Snell/Noether invariant p_x drifted: got {}, expected {}",
            res.p.x,
            px_expected
        );
        assert!(res.max_hamiltonian_err < 1e-6);
    }
}
