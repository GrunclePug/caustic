//! Exact flat-space dispersive optics: Cauchy glass dispersion `n(lambda)`,
//! exact unpolarized Fresnel equations, Snell refraction, and total internal reflection.

use glam::DVec2;

/// Cauchy refractive index `n(lambda_nm) = A + B / (lambda_um)^2`.
pub fn cauchy_ior(lambda_nm: f64, a: f64, b: f64) -> f64 {
    let lam_um = lambda_nm * 1e-3;
    a + b / (lam_um * lam_um)
}

/// Exact unpolarized Fresnel reflectance `R = 0.5 * (R_s + R_p)` and refracted unit direction.
///
/// `dir` is the unit incoming ray direction, `normal` points into the incident medium
/// (`dot(dir, normal) < 0`), `n1` is the incident medium IOR, and `n2` is the transmitted IOR.
/// Returns `(reflectance, Option<refracted_dir>)` (`None` on total internal reflection, `R = 1.0`).
pub fn fresnel_snell(dir: DVec2, normal: DVec2, n1: f64, n2: f64) -> (f64, Option<DVec2>) {
    let cos_i = (-dir.dot(normal)).clamp(0.0, 1.0);
    let eta = n1 / n2;
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return (1.0, None);
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    let r = 0.5 * (rs * rs + rp * rp);
    let refr = (eta * dir + (eta * cos_i - cos_t) * normal).normalize();
    (r, Some(refr))
}

/// Exact specular reflection of `dir` across unit `normal`.
pub fn reflect(dir: DVec2, normal: DVec2) -> DVec2 {
    dir - 2.0 * dir.dot(normal) * normal
}

/// Exact Rayleigh atmospheric scattering coefficient scaling as `(550 nm / lambda)^4`.
///
/// Short blue/violet wavelengths scatter ~4-6x more strongly than red wavelengths,
/// producing blue planetary skies and red-orange sunset / lunar-eclipse limbs.
#[inline]
pub fn rayleigh_sigma(lambda_nm: f64, base_density: f64) -> f64 {
    let ratio = 550.0 / lambda_nm.clamp(380.0, 780.0);
    let r2 = ratio * ratio;
    base_density.max(0.0) * r2 * r2
}

/// Unnormalized Planck spectral radiance `B_lambda(T)` in visible nanometers (`380..780 nm`).
#[inline]
pub fn planck_radiance_nm(lambda_nm: f64, temp_k: f64) -> f64 {
    const C2_NM_K: f64 = 1.438_776_877e7;
    let lam = lambda_nm.clamp(380.0, 780.0);
    let t = temp_k.clamp(1200.0, 40_000.0);
    let x = C2_NM_K / (lam * t);
    let inv_lam = 550.0 / lam;
    let p5 = inv_lam.powi(5);
    p5 / (x.exp_m1().max(1e-12))
}

/// Normalized Planck spectral weight over `[380, 780] nm` (mean value `~1.0` across the visible band).
pub fn planck_visible_weight(lambda_nm: f64, temp_k: f64) -> f64 {
    // 16-point midpoint normalization over [380, 780] nm (matches GPU shader).
    let mut sum = 0.0_f64;
    for i in 0..16 {
        let l = 380.0 + (f64::from(i) + 0.5) * (400.0 / 16.0);
        sum += planck_radiance_nm(l, temp_k);
    }
    let avg = (sum / 16.0).max(1e-30);
    planck_radiance_nm(lambda_nm, temp_k) / avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_and_planck_laws_match_physics() {
        // Rayleigh 1 / lambda^4 scaling: 450 nm vs 650 nm must equal (650/450)^4.
        let s_blue = rayleigh_sigma(450.0, 1.0);
        let s_red = rayleigh_sigma(650.0, 1.0);
        let expected_ratio = (650.0_f64 / 450.0).powi(4);
        assert!(
            ((s_blue / s_red) - expected_ratio).abs() < 1e-12,
            "Rayleigh ratio mismatch: got {}, expected {}",
            s_blue / s_red,
            expected_ratio
        );

        // Wien peak of 5778 K sunlight is near 501.5 nm (green-cyan), well above 400 and 700 nm.
        let w_500 = planck_visible_weight(501.5, 5778.0);
        let w_400 = planck_visible_weight(400.0, 5778.0);
        let w_700 = planck_visible_weight(700.0, 5778.0);
        assert!(
            w_500 > w_400 && w_500 > w_700,
            "5778 K should peak near 501 nm"
        );

        // Cool 3000 K star is red-heavy; hot 12000 K star is blue-heavy.
        assert!(planck_visible_weight(700.0, 3000.0) > 5.0 * planck_visible_weight(420.0, 3000.0));
        assert!(
            planck_visible_weight(420.0, 12000.0) > 3.0 * planck_visible_weight(700.0, 12000.0)
        );
    }

    #[test]
    fn snell_fresnel_brewster_and_tir() {
        let n1 = 1.0;
        let n2 = 1.5;
        let normal = DVec2::new(0.0, 1.0);

        // 1. Normal incidence: R = ((n1 - n2)/(n1 + n2))^2 = 0.04
        let (r0, refr0) = fresnel_snell(DVec2::new(0.0, -1.0), normal, n1, n2);
        assert!((r0 - 0.04).abs() < 1e-12);
        assert!((refr0.unwrap() - DVec2::new(0.0, -1.0)).length() < 1e-12);

        // 2. Brewster's angle: tan(theta_B) = n2 / n1 => R_p = 0, R = 0.5 * R_s
        let theta_b = (n2 / n1).atan();
        let dir_b = DVec2::new(theta_b.sin(), -theta_b.cos());
        let (rb, refr_b) = fresnel_snell(dir_b, normal, n1, n2);
        let refr_b = refr_b.unwrap();
        let sin_t = refr_b.x;
        assert!((n1 * theta_b.sin() - n2 * sin_t).abs() < 1e-12);
        let cos_t = -refr_b.y;
        let rs_b = (n1 * theta_b.cos() - n2 * cos_t) / (n1 * theta_b.cos() + n2 * cos_t);
        assert!((rb - 0.5 * rs_b * rs_b).abs() < 1e-12);

        // 3. Total internal reflection from glass -> air above critical angle arcsin(1/1.5)
        let theta_c = (n1 / n2).asin();
        let theta_tir = theta_c + 0.05;
        let dir_tir = DVec2::new(theta_tir.sin(), -theta_tir.cos());
        let (r_tir, refr_tir) = fresnel_snell(dir_tir, normal, n2, n1);
        assert_eq!(r_tir, 1.0);
        assert!(refr_tir.is_none());
    }

    #[test]
    fn equilateral_prism_minimum_deviation_dispersion() {
        // Verify exact textbook minimum deviation angle delta_min(lambda) = 2 arcsin(n sin(alpha/2)) - alpha
        // across red (650 nm), green (532 nm), and violet (420 nm) for an equilateral flint glass prism.
        let alpha = 60.0_f64.to_radians();
        let a = 1.58;
        let b = 0.032;

        // Outward normals of left and right faces of a symmetric apex-up equilateral prism:
        let n_left = DVec2::new(-(alpha * 0.5).cos(), (alpha * 0.5).sin());
        let n_right = DVec2::new((alpha * 0.5).cos(), (alpha * 0.5).sin());

        let mut prev_dev = 0.0_f64;
        for lam in [650.0_f64, 532.0, 420.0] {
            let n = cauchy_ior(lam, a, b);
            // Symmetric minimum-deviation incidence angle: sin(theta_i) = n * sin(alpha / 2)
            let theta_i = (n * (0.5 * alpha).sin()).asin();
            // Ray enters left face at angle theta_i to -n_left:
            let in_angle = (-n_left).y.atan2((-n_left).x) + theta_i;
            let dir_in = DVec2::new(in_angle.cos(), in_angle.sin());
            let (_, dir_glass) = fresnel_snell(dir_in, n_left, 1.0, n);
            let dir_glass = dir_glass.expect("enters prism");
            // Inside prism, ray hits right face from inside (normal pointing into glass is -n_right):
            let (_, dir_out) = fresnel_snell(dir_glass, -n_right, n, 1.0);
            let dir_out = dir_out.expect("exits prism");

            let got_dev = dir_in.dot(dir_out).clamp(-1.0, 1.0).acos();
            let exact_dev = 2.0 * (n * (0.5 * alpha).sin()).asin() - alpha;
            assert!(
                (got_dev - exact_dev).abs() < 1e-10,
                "prism deviation at {lam}nm: got={got_dev}, exact={exact_dev}"
            );
            assert!(
                got_dev > prev_dev,
                "shorter wavelength must bend more (dispersion)"
            );
            prev_dev = got_dev;
        }
    }
}
