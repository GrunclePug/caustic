//! Physically-based Page-Thorne (1974) relativistic thin accretion disk flux/temperature
//! profile and CIE 1931 -> linear Rec.709 log-XYZ blackbody spectral radiance LUT.
//!
//! Because `I_{nu, obs} = g^3 B_{nu / g}(T_emit) = B_nu(g * T_emit)`, evaluating the
//! absolute spectral blackbody radiance at `T_obs = g * T_emit` carries both the exact
//! Planckian color shift AND the relativistic `g^4` bolometric beaming with zero
//! double-counting.

use super::kerr::r_isco;

/// Number of entries in the Page-Thorne radial temperature LUT.
pub const DISK_LUT_LEN: usize = 512;
/// Number of entries in the log-temperature CIE XYZ blackbody LUT.
pub const BLACKBODY_LUT_LEN: usize = 1024;
/// Minimum blackbody temperature (K) before clamping to black.
pub const BLACKBODY_T_MIN: f64 = 300.0;
/// Maximum blackbody temperature (K) covered by the LUT.
pub const BLACKBODY_T_MAX: f64 = 400_000.0;
/// Reference temperature (K) whose luminance `Y(T_ref)` normalizes log-XYZ storage.
pub const BLACKBODY_T_REF: f64 = 6_500.0;

/// Physical constants (SI) for Planck's law.
const PLANCK_H: f64 = 6.626_070_15e-34;
const SPEED_C: f64 = 2.997_924_58e8;
const BOLTZMANN_K: f64 = 1.380_649e-23;

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
        // Smooth Schwarzschild limit (avoids 0/0 in the x2 -> 0 root term).
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
/// Sampled uniformly in `s = sqrt((r - r_ISCO) / (r_out - r_ISCO))` in `[0, 1]` because
/// `T(r) ~ sqrt(r - r_ISCO)` near the inner edge, making `T(s)` linear at `s = 0`!
pub fn build_disk_temp_lut(a: f64, r_out: f64) -> Vec<f32> {
    let r_in = r_isco(a);
    let span = (r_out - r_in).max(1e-6);

    // Find global peak flux over the full disk (and slightly beyond in case r_out is compact).
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

/// Piecewise Gaussian approximation to CIE 1931 2-degree color matching functions
/// (Wyman, Sloan, & Shirley, JCGT 2013), wavelength `lambda_nm` in nanometers.
pub fn cie_1931_xyz(lambda_nm: f64) -> [f64; 3] {
    let t1 = (lambda_nm - 442.0) * if lambda_nm < 442.0 { 0.0624 } else { 0.0374 };
    let t2 = (lambda_nm - 599.8) * if lambda_nm < 599.8 { 0.0264 } else { 0.0323 };
    let t3 = (lambda_nm - 501.1) * if lambda_nm < 501.1 { 0.0490 } else { 0.0382 };
    let x = 0.362 * (-0.5 * t1 * t1).exp() + 1.056 * (-0.5 * t2 * t2).exp()
        - 0.065 * (-0.5 * t3 * t3).exp();

    let u1 = (lambda_nm - 568.8) * if lambda_nm < 568.8 { 0.0213 } else { 0.0247 };
    let u2 = (lambda_nm - 530.9) * if lambda_nm < 530.9 { 0.0613 } else { 0.0322 };
    let y = 0.821 * (-0.5 * u1 * u1).exp() + 0.286 * (-0.5 * u2 * u2).exp();

    let v1 = (lambda_nm - 437.0) * if lambda_nm < 437.0 { 0.0845 } else { 0.0278 };
    let v2 = (lambda_nm - 459.0) * if lambda_nm < 459.0 { 0.0385 } else { 0.0725 };
    let z = 1.217 * (-0.5 * v1 * v1).exp() + 0.681 * (-0.5 * v2 * v2).exp();

    [x.max(0.0), y.max(0.0), z.max(0.0)]
}

/// Planck spectral radiance `B_lambda(lambda_m, T)` in `W / (sr m^3)`.
pub fn planck_lambda(lambda_m: f64, temp_k: f64) -> f64 {
    if temp_k <= 0.0 || lambda_m <= 0.0 {
        return 0.0;
    }
    let exponent = (PLANCK_H * SPEED_C) / (lambda_m * BOLTZMANN_K * temp_k);
    if exponent > 700.0 {
        return 0.0;
    }
    let c1 = 2.0 * PLANCK_H * SPEED_C * SPEED_C;
    c1 / (lambda_m.powi(5) * exponent.exp_m1())
}

/// Integrate absolute visible-band CIE XYZ radiance of a blackbody at temperature `temp_k` (K)
/// over `[380, 780] nm` using 400-panel Simpson quadrature.
pub fn integrate_blackbody_xyz(temp_k: f64) -> [f64; 3] {
    if temp_k < 1.0 {
        return [0.0; 3];
    }
    let n = 400_usize;
    let l0 = 380.0_f64;
    let l1 = 780.0_f64;
    let dl_nm = (l1 - l0) / (n as f64);
    let mut xyz = [0.0_f64; 3];

    for i in 0..=n {
        let lam_nm = l0 + (i as f64) * dl_nm;
        let lam_m = lam_nm * 1e-9;
        let b = planck_lambda(lam_m, temp_k);
        let cmf = cie_1931_xyz(lam_nm);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        for c in 0..3 {
            xyz[c] += w * b * cmf[c];
        }
    }
    let scale = (dl_nm * 1e-9) / 3.0;
    [xyz[0] * scale, xyz[1] * scale, xyz[2] * scale]
}

/// CPU-precomputed `f32` log-CIE-XYZ blackbody LUT sampled log-uniformly in `T`.
///
/// Each entry stores `[ln(X / Y_ref), ln(Y / Y_ref), ln(Z / Y_ref), 0.0]`.
/// Storing strictly positive log-XYZ prevents `log(negative)` on out-of-gamut low-T
/// linear Rec.709 channels and avoids `Rgba16Float` overflow before exposure scaling.
pub struct BlackbodyLut {
    /// Packed `[f32; 4]` entries for direct GPU `array<vec4<f32>>` upload.
    pub entries: Vec<[f32; 4]>,
    /// Reference luminance `Y(BLACKBODY_T_REF)` used to normalize the LUT.
    pub y_ref: f64,
}

impl BlackbodyLut {
    /// Precompute the 1024-entry log-CIE-XYZ blackbody LUT in `f64`.
    pub fn build() -> Self {
        let ref_xyz = integrate_blackbody_xyz(BLACKBODY_T_REF);
        let y_ref = ref_xyz[1];
        let ln_t_min = BLACKBODY_T_MIN.ln();
        let ln_t_max = BLACKBODY_T_MAX.ln();

        let mut entries = Vec::with_capacity(BLACKBODY_LUT_LEN);
        for i in 0..BLACKBODY_LUT_LEN {
            let frac = (i as f64) / ((BLACKBODY_LUT_LEN - 1) as f64);
            let t = (ln_t_min + frac * (ln_t_max - ln_t_min)).exp();
            let xyz = integrate_blackbody_xyz(t);
            let lx = (xyz[0] / y_ref).max(1e-300).ln() as f32;
            let ly = (xyz[1] / y_ref).max(1e-300).ln() as f32;
            let lz = (xyz[2] / y_ref).max(1e-300).ln() as f32;
            entries.push([lx, ly, lz, 0.0]);
        }

        Self { entries, y_ref }
    }

    /// Evaluate normalized CIE XYZ radiance at observed temperature `temp_k` (K)
    /// using the exact same log-temperature interpolation as the WGSL shader.
    pub fn sample_xyz(&self, temp_k: f64) -> [f64; 3] {
        if temp_k < BLACKBODY_T_MIN {
            return [0.0; 3];
        }
        let ln_min = BLACKBODY_T_MIN.ln();
        let ln_max = BLACKBODY_T_MAX.ln();
        let u = ((temp_k.clamp(BLACKBODY_T_MIN, BLACKBODY_T_MAX).ln() - ln_min)
            / (ln_max - ln_min))
            * ((BLACKBODY_LUT_LEN - 1) as f64);
        let i0 = (u.floor() as usize).min(BLACKBODY_LUT_LEN - 2);
        let frac = (u - (i0 as f64)) as f32;
        let a = self.entries[i0];
        let b = self.entries[i0 + 1];
        let lx = (a[0] + frac * (b[0] - a[0])) as f64;
        let ly = (a[1] + frac * (b[1] - a[1])) as f64;
        let lz = (a[2] + frac * (b[2] - a[2])) as f64;
        [
            lx.exp() * self.y_ref,
            ly.exp() * self.y_ref,
            lz.exp() * self.y_ref,
        ]
    }
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

    #[test]
    fn test1_blackbody_lut_matches_shifted_spectral_quadrature() {
        // Verify that evaluating the log-XYZ LUT at T_obs = g * T_emit matches direct
        // numerical quadrature of int g^5 B_{g*lambda}(T_emit) * CMF(lambda) dlambda
        // (equivalently int g^3 B_{nu/g}(T_emit) * CMF(nu) dnu) within < 0.5% relative error,
        // proving both spectral shift AND g^4 bolometric beaming with NO g^8 double-count!
        let lut = BlackbodyLut::build();

        let temps = [500.0_f64, 1_200.0, 3_500.0, 6_500.0, 15_000.0, 100_000.0];
        let gs = [0.25_f64, 0.5, 1.0, 1.8, 3.5];

        for &t_emit in &temps {
            for &g in &gs {
                let t_obs = g * t_emit;
                if !(400.0..=BLACKBODY_T_MAX).contains(&t_obs) {
                    continue;
                }

                // Direct Lorentz-invariant spectral quadrature:
                // I_{lambda, obs} = g^5 * B_lambda(g * lambda_obs, T_emit)
                let n = 400_usize;
                let l0 = 380.0_f64;
                let l1 = 780.0_f64;
                let dl_nm = (l1 - l0) / (n as f64);
                let mut direct = [0.0_f64; 3];
                let g5 = g.powi(5);
                for i in 0..=n {
                    let lam_nm = l0 + (i as f64) * dl_nm;
                    let lam_obs_m = lam_nm * 1e-9;
                    let b_obs = g5 * planck_lambda(g * lam_obs_m, t_emit);
                    let cmf = cie_1931_xyz(lam_nm);
                    let w = if i == 0 || i == n {
                        1.0
                    } else if i % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    for c in 0..3 {
                        direct[c] += w * b_obs * cmf[c];
                    }
                }
                let scale = (dl_nm * 1e-9) / 3.0;
                for v in &mut direct {
                    *v *= scale;
                }

                let got = lut.sample_xyz(t_obs);
                for c in 0..3 {
                    // Check relative error for non-underflowed channels (> 1e-12 of peak luminance).
                    if direct[c] > 1e-12 * direct[1].max(1e-30) {
                        let rel_err = (got[c] - direct[c]).abs() / direct[c];
                        assert!(
                            rel_err < 0.005,
                            "CIE LUT error {rel_err} > 0.5% at T={t_emit}, g={g}, ch={c}"
                        );
                    }
                }
            }
        }
    }
}
