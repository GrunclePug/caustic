//! CIE 1931 2-degree colorimetry, spectral wavelength-to-RGB LUTs, and
//! Planckian log-XYZ blackbody spectral radiance LUTs.

/// Number of entries in the log-temperature CIE XYZ blackbody LUT.
pub const BLACKBODY_LUT_LEN: usize = 1024;
/// Minimum blackbody temperature (K) before clamping to black.
pub const BLACKBODY_T_MIN: f64 = 300.0;
/// Maximum blackbody temperature (K) covered by the LUT.
pub const BLACKBODY_T_MAX: f64 = 400_000.0;
/// Reference temperature (K) whose luminance `Y(T_ref)` normalizes log-XYZ storage.
pub const BLACKBODY_T_REF: f64 = 6_500.0;

/// Number of entries in the visible wavelength (`380..780 nm`) spectral Rec.709 LUT.
pub const SPECTRAL_LUT_LEN: usize = 256;
/// Minimum visible wavelength in nanometers.
pub const LAMBDA_MIN_NM: f64 = 380.0;
/// Maximum visible wavelength in nanometers.
pub const LAMBDA_MAX_NM: f64 = 780.0;

/// Physical constants (SI) for Planck's law.
const PLANCK_H: f64 = 6.626_070_15e-34;
const SPEED_C: f64 = 2.997_924_58e8;
const BOLTZMANN_K: f64 = 1.380_649e-23;

/// Piecewise Gaussian approximation to CIE 1931 2-degree color matching functions
/// (Wyman, Sloan, & Shirley, JCGT 2013), wavelength `lambda_nm` in nanometers.
pub fn cie_1931_xyz(lambda_nm: f64) -> [f64; 3] {
    if !(360.0..=830.0).contains(&lambda_nm) {
        return [0.0; 3];
    }
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

/// Convert CIE 1931 XYZ to linear Rec.709 RGB with soft luminance-preserving gamut mapping.
pub fn xyz_to_rec709_soft(xyz: [f64; 3]) -> [f64; 3] {
    let r = 3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2];
    let g = -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2];
    let b = 0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2];
    let min_c = r.min(g).min(b);
    if min_c < 0.0 {
        let y = xyz[1].max(0.0);
        let alpha = (y / (y - min_c).max(1e-12)).clamp(0.0, 1.0);
        [
            (y + alpha * (r - y)).max(0.0),
            (y + alpha * (g - y)).max(0.0),
            (y + alpha * (b - y)).max(0.0),
        ]
    } else {
        [r, g, b]
    }
}

/// Build a 256-entry `f32` spectral LUT mapping wavelength `lambda in [380, 780] nm`
/// to CIE 1931 XYZ tristimulus `[X, Y, Z, 0.0]` normalized so equal-energy white
/// integrates to `(1, 1, 1)` luminance-balanced XYZ (converting XYZ -> Rec.709 after
/// summing across wavelengths preserves true physical color mixing!).
pub fn build_spectral_xyz_lut() -> Vec<[f32; 4]> {
    let mut raw = Vec::with_capacity(SPECTRAL_LUT_LEN);
    let mut sum_y = 0.0_f64;
    for i in 0..SPECTRAL_LUT_LEN {
        let frac = (i as f64) / ((SPECTRAL_LUT_LEN - 1) as f64);
        let lam = LAMBDA_MIN_NM + frac * (LAMBDA_MAX_NM - LAMBDA_MIN_NM);
        let xyz = cie_1931_xyz(lam);
        sum_y += xyz[1];
        raw.push(xyz);
    }
    let scale = (SPECTRAL_LUT_LEN as f64) / sum_y.max(1e-12);
    raw.into_iter()
        .map(|xyz| {
            [
                (xyz[0] * scale) as f32,
                (xyz[1] * scale) as f32,
                (xyz[2] * scale) as f32,
                0.0,
            ]
        })
        .collect()
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
    let l0 = LAMBDA_MIN_NM;
    let l1 = LAMBDA_MAX_NM;
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
pub struct BlackbodyLut {
    /// Packed `[f32; 4]` entries `[ln(X/Y_ref), ln(Y/Y_ref), ln(Z/Y_ref), 0.0]`.
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

    /// Evaluate normalized CIE XYZ radiance at observed temperature `temp_k` (K).
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
    fn spectral_xyz_lut_integrates_to_neutral_white() {
        let lut = build_spectral_xyz_lut();
        let mut sum = [0.0_f64; 3];
        for e in &lut {
            for c in 0..3 {
                assert!(e[c] >= 0.0);
                sum[c] += f64::from(e[c]);
            }
        }
        let mean_y = sum[1] / (lut.len() as f64);
        assert!((mean_y - 1.0).abs() < 1e-6);
        // Equal-energy white has X ~ Y ~ Z ~ 1.0
        assert!((sum[0] / sum[1] - 1.0).abs() < 0.05);
        assert!((sum[2] / sum[1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn blackbody_lut_matches_shifted_spectral_quadrature() {
        let lut = BlackbodyLut::build();
        let temps = [500.0_f64, 1_200.0, 3_500.0, 6_500.0, 15_000.0, 100_000.0];
        let gs = [0.25_f64, 0.5, 1.0, 1.8, 3.5];

        for &t_emit in &temps {
            for &g in &gs {
                let t_obs = g * t_emit;
                if !(400.0..=BLACKBODY_T_MAX).contains(&t_obs) {
                    continue;
                }
                let n = 400_usize;
                let dl_nm = (LAMBDA_MAX_NM - LAMBDA_MIN_NM) / (n as f64);
                let mut direct = [0.0_f64; 3];
                let g5 = g.powi(5);
                for i in 0..=n {
                    let lam_nm = LAMBDA_MIN_NM + (i as f64) * dl_nm;
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
