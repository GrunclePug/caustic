//! Physically-based Page-Thorne (1974) relativistic thin accretion disk flux/temperature
//! profile, re-exporting shared CIE 1931 blackbody LUT types from `caustic-core`.

pub use caustic_core::spectrum::{
    BLACKBODY_LUT_LEN, BLACKBODY_T_MAX, BLACKBODY_T_MIN, BLACKBODY_T_REF, BlackbodyLut,
    cie_1931_xyz, integrate_blackbody_xyz, planck_lambda,
};

use crate::kerr::r_isco;

/// Number of entries in the Page-Thorne radial temperature LUT.
pub const DISK_LUT_LEN: usize = 512;

/// Exact Page-Thorne (1974, Eq. 15n) dimensionless relativistic accretion disk flux `f(r)`.
///
/// Returns `0.0` for `r <= r_ISCO(a)`.
pub fn page_thorne_flux(r: f64, a: f64) -> f64 {
    let r_in = r_isco(a);
    if r <= r_in {
        return 0.0;
    }
    let x = r.sqrt();
    let x0 = r_in.sqrt();
    let denom = x * x * (x * x * x - 3.0 * x + 2.0 * a);
    if denom <= 0.0 {
        return 0.0;
    }

    let bracket = if a.abs() < 1e-6 {
        let s3 = 3.0_f64.sqrt();
        let num = (x - s3) * (x0 + s3);
        let den = (x + s3) * (x0 - s3);
        (x - x0) - 0.5 * s3 * (num / den).ln()
    } else {
        let theta = a.clamp(-0.999999, 0.999999).acos() / 3.0;
        let x1 = 2.0 * (theta - std::f64::consts::FRAC_PI_3).cos();
        let x2 = 2.0 * (theta + std::f64::consts::FRAC_PI_3).cos();
        let x3 = -2.0 * theta.cos();

        let term = |xi: f64, xj: f64, xk: f64| -> f64 {
            let coeff = 3.0 * (xi - a) * (xi - a) / (xi * (xi - xj) * (xi - xk));
            coeff * ((x - xi) / (x0 - xi)).abs().ln()
        };

        (x - x0) - 1.5 * a * (x / x0).ln() - term(x1, x2, x3) - term(x2, x1, x3) - term(x3, x1, x2)
    };

    (1.5 * bracket / denom).max(0.0)
}

/// Build the normalized Page-Thorne effective temperature LUT `T_emit(r) / T_peak` in `f32`.
///
/// Sampled uniformly in `s = sqrt((r - r_ISCO) / (r_out - r_ISCO))` in `[0, 1]`.
pub fn build_disk_temp_lut(a: f64, r_out: f64) -> Vec<f32> {
    let r_in = r_isco(a);
    let span = (r_out - r_in).max(1e-6);

    let mut f_max = 0.0_f64;
    let mut raw = Vec::with_capacity(DISK_LUT_LEN);
    for i in 0..DISK_LUT_LEN {
        let s = (i as f64) / ((DISK_LUT_LEN - 1) as f64);
        let r = r_in + span * s * s;
        let f = page_thorne_flux(r, a);
        f_max = f_max.max(f);
        raw.push(f);
    }

    if f_max <= 0.0 {
        return vec![0.0; DISK_LUT_LEN];
    }

    raw.into_iter()
        .map(|f| (f / f_max).powf(0.25) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_thorne_flux_properties() {
        for a in [-0.998_f64, 0.0, 0.9, 0.998] {
            let rin = r_isco(a);
            assert_eq!(page_thorne_flux(rin, a), 0.0);
            assert_eq!(page_thorne_flux(rin - 0.1, a), 0.0);
            assert!(page_thorne_flux(rin + 0.5, a) > 0.0);
            let lut = build_disk_temp_lut(a, 20.0);
            assert_eq!(lut[0], 0.0);
            let max_v = lut.iter().copied().fold(0.0_f32, f32::max);
            assert!((max_v - 1.0).abs() < 1e-6);
        }
    }
}
