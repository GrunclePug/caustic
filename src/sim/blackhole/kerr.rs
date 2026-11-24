//! Exact Kerr spacetime geometry in Outgoing Cartesian Kerr-Schild and
//! Boyer-Lindquist coordinates, covariant ZAMO camera tetrad, and exact
//! cross-chart coordinate maps.
//!
//! Conventions (geometric units `G = c = M = 1`, signature `(-, +, +, +)`):
//! - Signed spin `a in [-0.998, +0.998]` about `+z`.
//! - Accretion disk orbits in `+phi` with angular velocity `Omega = 1 / (r^(3/2) + a)`,
//!   so `a >= 0` is prograde and `a < 0` is retrograde.
//! - Outgoing Kerr-Schild coordinates `(t_out, x, y, z)` satisfy:
//!   `dt_out = dt_BL - (2r / Delta) dr`, `dphi_out = dphi_BL - (a / Delta) dr`,
//!   `x + i y = (r - i a) sin(theta) exp(i phi_out)`, `z = r cos(theta)`,
//!   so `atan2(y, x) = phi_out - atan(a / r)`.
//! - Metric: `g_{mu nu} = eta_{mu nu} + 2 H l_mu l_nu`, `g^{mu nu} = eta^{mu nu} - 2 H l^mu l^nu`,
//!   `H = r^3 / (r^4 + a^2 z^2)`,
//!   `l_mu = (1, -(r x - a y)/(r^2 + a^2), -(r y + a x)/(r^2 + a^2), -z/r)`.

use glam::{DMat4, DVec3, DVec4};

/// Minimum allowed camera distance outside the outer horizon `r_+`.
pub const MIN_CAMERA_HORIZON_MARGIN: f64 = 0.2;
/// Maximum supported camera distance for direct ray initialization.
pub const MAX_CAMERA_RADIUS: f64 = 200.0;
/// Maximum allowed spin magnitude `|a|`.
pub const MAX_SPIN: f64 = 0.998;

/// Kerr spacetime parameters (geometric units `G = c = M = 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KerrParams {
    /// Dimensionless spin `a = J / M^2 in [-0.998, +0.998]`.
    pub spin: f64,
    /// Outer radius of the equatorial thin accretion disk (in units of `M`).
    pub disk_outer: f64,
    /// Escape radius for background sky rays (in units of `M`).
    pub r_escape: f64,
}

impl Default for KerrParams {
    fn default() -> Self {
        Self {
            spin: 0.95,
            disk_outer: 16.0,
            r_escape: 1000.0,
        }
    }
}

impl KerrParams {
    /// Create validated Kerr parameters, clamping spin to `[-MAX_SPIN, MAX_SPIN]`.
    pub fn new(spin: f64, disk_outer: f64, r_escape: f64) -> Self {
        Self {
            spin: spin.clamp(-MAX_SPIN, MAX_SPIN),
            disk_outer,
            r_escape,
        }
    }

    /// Outer event horizon radius `r_+ = 1 + sqrt((1 - a)(1 + a))`.
    pub fn r_plus(&self) -> f64 {
        r_plus(self.spin)
    }

    /// Inner Cauchy horizon radius `r_- = 1 - sqrt((1 - a)(1 + a))`.
    pub fn r_minus(&self) -> f64 {
        r_minus(self.spin)
    }

    /// Exact Kerr equatorial ISCO radius `r_ISCO(a)` (Bardeen-Press-Teukolsky 1972).
    pub fn r_isco(&self) -> f64 {
        r_isco(self.spin)
    }
}

/// Outer event horizon radius `r_+ = 1 + sqrt((1 - a)(1 + a))`.
pub fn r_plus(a: f64) -> f64 {
    1.0 + ((1.0 - a) * (1.0 + a)).max(0.0).sqrt()
}

/// Inner Cauchy horizon radius `r_- = 1 - sqrt((1 - a)(1 + a))`.
pub fn r_minus(a: f64) -> f64 {
    1.0 - ((1.0 - a) * (1.0 + a)).max(0.0).sqrt()
}

/// Factored horizon function `Delta(r) = (r - r_+)(r - r_-)`.
pub fn delta_factored(r: f64, a: f64) -> f64 {
    (r - r_plus(a)) * (r - r_minus(a))
}

/// Exact Kerr equatorial ISCO radius (Bardeen-Press-Teukolsky 1972).
///
/// Positive `a` gives prograde (`6 -> 1`), negative `a` gives retrograde (`6 -> 9`).
pub fn r_isco(a: f64) -> f64 {
    let a_abs = a.abs();
    let one_minus_a2 = ((1.0 - a_abs) * (1.0 + a_abs)).max(0.0);
    let z1 = 1.0 + one_minus_a2.cbrt() * ((1.0 + a_abs).cbrt() + (1.0 - a_abs).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    let term = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
    if a >= 0.0 {
        3.0 + z2 - term
    } else {
        3.0 + z2 + term
    }
}

/// Circular equatorial disk angular velocity `Omega(r) = 1 / (r^(3/2) + a)`.
pub fn disk_omega(r: f64, a: f64) -> f64 {
    1.0 / (r * r.sqrt() + a)
}

/// Time component `u^t` of the circular equatorial geodesic 4-velocity at radius `r`.
///
/// Valid for `r` outside the circular photon orbit (`r^(3/2) - 3 r^(1/2) + 2 a > 0`).
pub fn disk_u_t(r: f64, a: f64) -> f64 {
    let r_sqrt = r.sqrt();
    let r32 = r * r_sqrt;
    let denom = (r32 * (r32 - 3.0 * r_sqrt + 2.0 * a)).max(1e-30).sqrt();
    (r32 + a) / denom
}

/// Ergosphere-safe relativistic redshift factor for a circular equatorial emitter.
///
/// Assumes photon 4-momentum is normalized to camera-frame frequency `nu_obs = -(p_mu u^mu_obs) = 1`,
/// with conserved Killing quantities `E = -p_t` and `L_z = p_phi`.
pub fn disk_redshift_g(r_emit: f64, a: f64, energy: f64, lz: f64) -> f64 {
    let omega = disk_omega(r_emit, a);
    let u_t = disk_u_t(r_emit, a);
    let nu_emit = u_t * (energy - omega * lz);
    if nu_emit > 1e-12 { 1.0 / nu_emit } else { 0.0 }
}

/// Oblate-spheroidal Boyer-Lindquist radius `r` from Cartesian Kerr-Schild `(x, y, z)`.
pub fn ks_radius(pos: DVec3, a: f64) -> f64 {
    let a2 = a * a;
    let b = pos.length_squared() - a2;
    let disc = (b * b + 4.0 * a2 * pos.z * pos.z).sqrt();
    let r2 = 0.5 * (b + disc);
    r2.max(1e-30).sqrt()
}

/// Integrated outgoing Kerr-Schild azimuthal twist `psi(r) = int_inf^r (a / Delta) dr`.
///
/// `phi_out = phi_BL - psi(r)` for `r > r_+`.
pub fn outgoing_ks_phi_twist(r: f64, a: f64) -> f64 {
    if a.abs() < 1e-15 {
        return 0.0;
    }
    let rp = r_plus(a);
    let rm = r_minus(a);
    let diff = rp - rm;
    if diff.abs() < 1e-12 {
        return 0.0;
    }
    (a / diff) * ((r - rp) / (r - rm)).ln()
}

/// Convert Boyer-Lindquist `(r, theta, phi_bl)` to Outgoing Cartesian KS `(x, y, z)`.
pub fn bl_to_outgoing_ks_pos(r: f64, theta: f64, phi_bl: f64, a: f64) -> DVec3 {
    let phi_out = phi_bl - outgoing_ks_phi_twist(r, a);
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let (sin_p, cos_p) = phi_out.sin_cos();
    DVec3::new(
        (r * cos_p + a * sin_p) * sin_t,
        (r * sin_p - a * cos_p) * sin_t,
        r * cos_t,
    )
}

/// Convert Outgoing Cartesian KS `(x, y, z)` to Boyer-Lindquist `(r, theta, phi_bl)`.
pub fn outgoing_ks_to_bl_pos(pos: DVec3, a: f64) -> (f64, f64, f64) {
    let r = ks_radius(pos, a);
    let cos_t = (pos.z / r).clamp(-1.0, 1.0);
    let theta = cos_t.acos();
    let phi_out = pos.y.atan2(pos.x) + (a / r).atan();
    let phi_bl = phi_out + outgoing_ks_phi_twist(r, a);
    (r, theta, phi_bl)
}

/// Outgoing Kerr-Schild scalar `H` and spatial null covector `l = (l_x, l_y, l_z)` (with `l_t = 1`).
pub fn outgoing_ks_hl(pos: DVec3, a: f64) -> (f64, f64, DVec3) {
    let r = ks_radius(pos, a);
    let r2 = r * r;
    let a2 = a * a;
    let d = r2 + a2;
    let h = (r2 * r) / (r2 * r2 + a2 * pos.z * pos.z);
    let l = DVec3::new(
        -(r * pos.x - a * pos.y) / d,
        -(r * pos.y + a * pos.x) / d,
        -pos.z / r,
    );
    (r, h, l)
}

/// Covariant metric `g_{mu nu}` in Outgoing Cartesian Kerr-Schild coordinates.
///
/// Columns/rows indexed as `(t, x, y, z)`.
pub fn outgoing_ks_metric(pos: DVec3, a: f64) -> DMat4 {
    let (_, h, l) = outgoing_ks_hl(pos, a);
    let lv = [1.0, l.x, l.y, l.z];
    let eta = [-1.0, 1.0, 1.0, 1.0];
    let mut cols = [DVec4::ZERO; 4];
    for c in 0..4 {
        let mut col = [0.0; 4];
        for r in 0..4 {
            let diag = if r == c { eta[r] } else { 0.0 };
            col[r] = diag + 2.0 * h * lv[r] * lv[c];
        }
        cols[c] = DVec4::from_array(col);
    }
    DMat4::from_cols(cols[0], cols[1], cols[2], cols[3])
}

/// Contravariant inverse metric `g^{mu nu}` in Outgoing Cartesian Kerr-Schild coordinates.
pub fn outgoing_ks_inv_metric(pos: DVec3, a: f64) -> DMat4 {
    let (_, h, l) = outgoing_ks_hl(pos, a);
    let l_up = [-1.0, l.x, l.y, l.z];
    let eta = [-1.0, 1.0, 1.0, 1.0];
    let mut cols = [DVec4::ZERO; 4];
    for c in 0..4 {
        let mut col = [0.0; 4];
        for r in 0..4 {
            let diag = if r == c { eta[r] } else { 0.0 };
            col[r] = diag - 2.0 * h * l_up[r] * l_up[c];
        }
        cols[c] = DVec4::from_array(col);
    }
    DMat4::from_cols(cols[0], cols[1], cols[2], cols[3])
}

/// Metric inner product `g_{mu nu} u^mu v^nu`.
pub fn metric_dot(g: &DMat4, u: DVec4, v: DVec4) -> f64 {
    u.dot(*g * v)
}

/// Outgoing Cartesian KS null geodesic Hamiltonian `H_geo = 0.5 * g^{mu nu} p_mu p_nu`.
///
/// `energy = -p_t`, `p = (p_x, p_y, p_z)`.
pub fn outgoing_ks_hamiltonian(pos: DVec3, p: DVec3, energy: f64, a: f64) -> f64 {
    let (_, h, l) = outgoing_ks_hl(pos, a);
    let k = energy + l.dot(p);
    0.5 * (-energy * energy + p.length_squared()) - h * k * k
}

/// Future-directed Hamiltonian RHS `(dx/dlambda, dp/dlambda)` in Outgoing Cartesian KS.
///
/// Backward camera tracing integrates with step `-h`.
pub fn outgoing_ks_rhs(pos: DVec3, p: DVec3, energy: f64, a: f64) -> (DVec3, DVec3) {
    let r = ks_radius(pos, a);
    let r2 = r * r;
    let a2 = a * a;
    let d = r2 + a2;
    let denom = r2 * r2 + a2 * pos.z * pos.z; // r^2 * Sigma
    let h = (r2 * r) / denom;
    let l = DVec3::new(
        -(r * pos.x - a * pos.y) / d,
        -(r * pos.y + a * pos.x) / d,
        -pos.z / r,
    );

    let k = energy + l.dot(p);
    let dx = p - (2.0 * h * k) * l;

    // grad r = (r / (r^4 + a^2 z^2)) * (r^2 x, r^2 y, (r^2 + a^2) z)
    let grad_r_scale = r / denom;
    let grad_r = DVec3::new(
        grad_r_scale * r2 * pos.x,
        grad_r_scale * r2 * pos.y,
        grad_r_scale * d * pos.z,
    );

    // grad H = (dH/dr) grad_r + (0, 0, dH/dz|_r)
    let denom2 = denom * denom;
    let dh_dr = r2 * (3.0 * a2 * pos.z * pos.z - r2 * r2) / denom2;
    let dh_dz_fixed_r = -2.0 * a2 * r2 * r * pos.z / denom2;
    let grad_h = dh_dr * grad_r + DVec3::new(0.0, 0.0, dh_dz_fixed_r);

    // p . (d l / d x^i) = (p . d l / d x^i|_r) + (p . d l / dr) * grad_r_i
    let p_dl_dr =
        -(p.x * (pos.x + 2.0 * r * l.x) + p.y * (pos.y + 2.0 * r * l.y)) / d - (p.z * l.z) / r;
    let p_dl_dx_fixed = DVec3::new(-(r * p.x + a * p.y) / d, (a * p.x - r * p.y) / d, -p.z / r);
    let p_grad_l = p_dl_dx_fixed + p_dl_dr * grad_r;

    let dp = (k * k) * grad_h + (2.0 * h * k) * p_grad_l;
    (dx, dp)
}

/// Covariant ZAMO (Zero Angular Momentum Observer) orthonormal tetrad in Outgoing KS.
///
/// All basis vectors are contravariant 4-vectors `(t, x, y, z)` satisfying
/// `g_{mu nu} e^mu_{(a)} e^nu_{(b)} = eta_{(a)(b)}`.
#[derive(Debug, Clone, Copy)]
pub struct ZamoTetrad {
    /// Camera position in Outgoing Cartesian KS.
    pub pos: DVec3,
    /// Boyer-Lindquist `(r, theta, phi_bl)` of the camera.
    pub bl_pos: (f64, f64, f64),
    /// Timelike future-directed ZAMO 4-velocity `e_{(0)} = u_obs`.
    pub e_t: DVec4,
    /// Outward radial unit vector `e_{(r)}`.
    pub e_r: DVec4,
    /// Polar southward unit vector `e_{(theta)}` (or Cartesian fallback on axis).
    pub e_theta: DVec4,
    /// Azimuthal `+phi` unit vector `e_{(phi)}` (or Cartesian fallback on axis).
    pub e_phi: DVec4,
    /// Local ZAMO frame-dragging angular velocity `omega = -g_{t phi} / g_{phi phi}`.
    pub omega: f64,
    /// Lapse function `alpha = 1 / u^t`.
    pub alpha: f64,
}

impl ZamoTetrad {
    /// Construct the covariant ZAMO tetrad at Boyer-Lindquist `(r, theta, phi_bl)`.
    ///
    /// Built from Killing vectors `xi_t = d_t`, `xi_phi = x d_y - y d_x` and raised
    /// gradients `(dr)^sharp`, `(dtheta)^sharp` so it matches the analytic Bardeen ZAMO
    /// frame in Boyer-Lindquist, with a smooth Cartesian fallback on the spin axis.
    pub fn from_bl(r: f64, theta: f64, phi_bl: f64, a: f64) -> Self {
        let pos = bl_to_outgoing_ks_pos(r, theta, phi_bl, a);
        let g = outgoing_ks_metric(pos, a);
        let g_inv = outgoing_ks_inv_metric(pos, a);

        let xi_t = DVec4::new(1.0, 0.0, 0.0, 0.0);
        let xi_phi = DVec4::new(0.0, -pos.y, pos.x, 0.0);
        let rho = pos.x.hypot(pos.y);

        // ZAMO 4-velocity u = (xi_t + omega * xi_phi) / alpha
        let (omega, u_raw) = if rho > 1e-10 {
            let g_tphi = metric_dot(&g, xi_t, xi_phi);
            let g_phiphi = metric_dot(&g, xi_phi, xi_phi);
            let omega = -g_tphi / g_phiphi;
            (omega, xi_t + omega * xi_phi)
        } else {
            (0.0, xi_t)
        };
        let u_norm2 = -metric_dot(&g, u_raw, u_raw);
        let alpha = u_norm2.max(1e-30).sqrt();
        let e_t = u_raw / alpha;

        // Raised gradient of r: (dr)^sharp = g^{mu nu} d_nu r
        let r2 = r * r;
        let a2 = a * a;
        let d = r2 + a2;
        let denom = r2 * r2 + a2 * pos.z * pos.z;
        let grad_r_cov = DVec4::new(
            0.0,
            (r / denom) * r2 * pos.x,
            (r / denom) * r2 * pos.y,
            (r / denom) * d * pos.z,
        );
        let e_r_raw = g_inv * grad_r_cov;
        // Project out any tiny floating-point component along e_t.
        let e_r_orth = e_r_raw + metric_dot(&g, e_r_raw, e_t) * e_t;
        let e_r = e_r_orth / metric_dot(&g, e_r_orth, e_r_orth).max(1e-30).sqrt();

        let (e_theta, e_phi) = if rho > 1e-8 * r {
            // Azimuthal Killing direction e_(phi)
            let e_phi_orth =
                xi_phi + metric_dot(&g, xi_phi, e_t) * e_t - metric_dot(&g, xi_phi, e_r) * e_r;
            let e_phi = e_phi_orth / metric_dot(&g, e_phi_orth, e_phi_orth).max(1e-30).sqrt();

            // Covariant polar gradient d(cos theta) = d(z / r) = (dz - (z/r) dr) / r
            // Since dtheta = -d(cos theta) / sin(theta), pointing toward +theta (southward):
            let cos_t = pos.z / r;
            let grad_cos_t_cov = DVec4::new(
                0.0,
                -cos_t * grad_r_cov.y / r,
                -cos_t * grad_r_cov.z / r,
                (1.0 - cos_t * grad_r_cov.w) / r,
            );
            let e_theta_raw = -(g_inv * grad_cos_t_cov);
            let e_theta_orth = e_theta_raw + metric_dot(&g, e_theta_raw, e_t) * e_t
                - metric_dot(&g, e_theta_raw, e_r) * e_r
                - metric_dot(&g, e_theta_raw, e_phi) * e_phi;
            let e_theta =
                e_theta_orth / metric_dot(&g, e_theta_orth, e_theta_orth).max(1e-30).sqrt();
            (e_theta, e_phi)
        } else {
            // Exact polar-axis Cartesian fallback matching phi_bl azimuth orientation:
            // At north pole (z > 0): +theta along (cos phi_out, sin phi_out, 0),
            // +phi along (-sin phi_out, cos phi_out, 0).
            // At south pole (z < 0): +theta along (-cos phi_out, -sin phi_out, 0),
            // +phi along (-sin phi_out, cos phi_out, 0).
            let phi_out = phi_bl - outgoing_ks_phi_twist(r, a);
            let (sp, cp) = phi_out.sin_cos();
            let sgn_z = if pos.z >= 0.0 { 1.0 } else { -1.0 };
            // Account for Cartesian rotation atan(a/r) so axis basis matches phi_bl smoothly:
            let psi_cart = phi_out - (a / r).atan();
            let (spc, cpc) = psi_cart.sin_cos();
            let _ = (sp, cp);
            let raw_theta = DVec4::new(0.0, sgn_z * cpc, sgn_z * spc, 0.0);
            let raw_phi = DVec4::new(0.0, -spc, cpc, 0.0);

            let e_theta_orth = raw_theta + metric_dot(&g, raw_theta, e_t) * e_t
                - metric_dot(&g, raw_theta, e_r) * e_r;
            let e_theta =
                e_theta_orth / metric_dot(&g, e_theta_orth, e_theta_orth).max(1e-30).sqrt();

            let e_phi_orth = raw_phi + metric_dot(&g, raw_phi, e_t) * e_t
                - metric_dot(&g, raw_phi, e_r) * e_r
                - metric_dot(&g, raw_phi, e_theta) * e_theta;
            let e_phi = e_phi_orth / metric_dot(&g, e_phi_orth, e_phi_orth).max(1e-30).sqrt();
            (e_theta, e_phi)
        };

        Self {
            pos,
            bl_pos: (r, theta, phi_bl),
            e_t,
            e_r,
            e_theta,
            e_phi,
            omega,
            alpha,
        }
    }

    /// Initialize a future-directed null photon covector `p_mu = (-E, p_x, p_y, p_z)` in
    /// Outgoing Cartesian KS from a unit local ZAMO sky-look direction `local_dir = (d_r, d_theta, d_phi)`.
    ///
    /// A photon arriving from `+local_dir` travels forward in time along `-local_dir`, so:
    /// `p^mu = e_t^mu - d_r e_r^mu - d_theta e_theta^mu - d_phi e_phi^mu`,
    /// normalized to camera-frame frequency `nu_obs = -(p_mu u^mu_obs) = 1`.
    pub fn init_ray_covector(&self, local_dir: DVec3, a: f64) -> (f64, DVec3) {
        let d = local_dir.normalize();
        let p_up = self.e_t - d.x * self.e_r - d.y * self.e_theta - d.z * self.e_phi;
        let g = outgoing_ks_metric(self.pos, a);
        let p_down = g * p_up;
        let energy = -p_down.x;
        let p_spatial = DVec3::new(p_down.y, p_down.z, p_down.w);
        (energy, p_spatial)
    }

    /// Compute the exact Boyer-Lindquist conserved quantities `(E, L_z, Q)` and initial
    /// Mino-time derivatives `(dr/dlambda_M, du/dlambda_M)` for the same local ZAMO ray direction.
    ///
    /// Uses the analytic Bardeen ZAMO tetrad in Boyer-Lindquist so the CPU oracle is
    /// completely independent of the Cartesian Kerr-Schild coordinate chart.
    pub fn init_bl_ray(&self, local_dir: DVec3, a: f64) -> BlInitialRay {
        let d = local_dir.normalize();
        let (r, theta, phi_bl) = self.bl_pos;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let r2 = r * r;
        let a2 = a * a;
        let sigma = r2 + a2 * cos_t * cos_t;
        let delta = delta_factored(r, a);
        let big_a = (r2 + a2) * (r2 + a2) - a2 * delta * sin_t * sin_t;

        // Analytic Bardeen ZAMO tetrad components in BL:
        // e_(0) = sqrt(A / (Sigma * Delta)) (d_t + omega d_phi), omega = 2 a r / A
        // e_(r) = sqrt(Delta / Sigma) d_r
        // e_(theta) = (1 / sqrt(Sigma)) d_theta
        // e_(phi) = sqrt(Sigma / A) / sin(theta) d_phi
        let alpha_bl = (sigma * delta / big_a).sqrt();
        let omega_bl = (2.0 * a * r) / big_a;
        let sqrt_g_phiphi = (big_a / sigma).sqrt() * sin_t;

        // Photon arriving from +d travels forward in time along -d:
        // p^mu = e_(0)^mu - d_r e_(r)^mu - d_theta e_(theta)^mu - d_phi e_(phi)^mu
        // Conserved Killing covectors:
        // L_z = p_phi = g_{phi mu} p^mu = -d_phi * sqrt(g_{phi phi})
        // E = -p_t = alpha_bl + omega_bl * L_z
        let lz = -d.z * sqrt_g_phiphi;
        let energy = alpha_bl + omega_bl * lz;

        // Future-directed Mino-time derivatives (dlambda_M = dlambda / Sigma):
        // dr/dlambda_M = Sigma * p^r = -d_r * sqrt(Sigma * Delta)
        // dtheta/dlambda_M = Sigma * p^theta = -d_theta * sqrt(Sigma)
        // du/dlambda_M = -sin(theta) * dtheta/dlambda_M = +d_theta * sin(theta) * sqrt(Sigma)
        let dr_dlm = -d.x * (sigma * delta).sqrt();
        let p_theta = -d.y * sigma.sqrt();
        let du_dlm = -sin_t * p_theta;

        // Carter constant Q = p_theta^2 + cos^2(theta) * (L_z^2 / sin^2(theta) - a^2 E^2)
        // Note L_z / sin(theta) = -d_phi * sqrt(A / Sigma) is finite even at sin(theta) = 0!
        let lz_over_sin = -d.z * (big_a / sigma).sqrt();
        let q =
            p_theta * p_theta + cos_t * cos_t * (lz_over_sin * lz_over_sin - a2 * energy * energy);

        BlInitialRay {
            r,
            u: cos_t,
            phi_bl,
            dr_dlm,
            du_dlm,
            energy,
            lz,
            q,
        }
    }
}

/// Initial Boyer-Lindquist Mino-time ray state for the independent CPU `f64` oracle.
#[derive(Debug, Clone, Copy)]
pub struct BlInitialRay {
    pub r: f64,
    pub u: f64,
    pub phi_bl: f64,
    pub dr_dlm: f64,
    pub du_dlm: f64,
    pub energy: f64,
    pub lz: f64,
    pub q: f64,
}

/// Camera configuration in Boyer-Lindquist spherical coordinates around the black hole.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Boyer-Lindquist radial coordinate `r_obs` (in `M`).
    pub radius: f64,
    /// Polar inclination `theta_obs in [0, pi]` (`0` = north pole face-on, `pi/2` = equatorial edge-on).
    pub inclination: f64,
    /// Azimuth `phi_obs` in radians.
    pub azimuth: f64,
    /// Vertical field of view in radians.
    pub vfov: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            radius: 32.0,
            inclination: 82.0_f64.to_radians(),
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        }
    }
}

impl Camera {
    /// Construct the local ZAMO camera direction `(d_r, d_theta, d_phi)` for normalized
    /// screen coordinates `(ndc_x, ndc_y)` in `[-1, 1]` (`+x` right, `+y` up).
    pub fn pixel_local_dir(&self, ndc_x: f64, ndc_y: f64, aspect: f64) -> DVec3 {
        let tan_half = (0.5 * self.vfov).tan();
        // Forward is -e_r, right is +e_phi, up is -e_theta.
        DVec3::new(-1.0, -ndc_y * tan_half, ndc_x * aspect * tan_half).normalize()
    }

    /// Build the exact covariant ZAMO tetrad for this camera and spin `a`.
    pub fn zamo_tetrad(&self, a: f64) -> ZamoTetrad {
        ZamoTetrad::from_bl(self.radius, self.inclination, self.azimuth, a)
    }
}

/// Convert Boyer-Lindquist tangent vector `(dr/dl, du/dl, dphi_bl/dl)` at `(r, u, phi_bl)`
/// into Outgoing Cartesian Kerr-Schild spatial velocity `dx_ks / dl`.
pub fn bl_velocity_to_outgoing_ks_dir(
    r: f64,
    u: f64,
    phi_bl: f64,
    dr: f64,
    du: f64,
    dphi_bl: f64,
    a: f64,
) -> DVec3 {
    let sin_t = (1.0 - u * u).max(0.0).sqrt();
    let phi_out = phi_bl - outgoing_ks_phi_twist(r, a);
    let (sp, cp) = phi_out.sin_cos();
    let delta = delta_factored(r, a);
    let dphi_out = dphi_bl - (a / delta) * dr;

    // d(sin_t) = -(u / sin_t) * du
    // x = (r cp + a sp) sin_t, y = (r sp - a cp) sin_t, z = r u
    let rx_part = r * cp + a * sp;
    let ry_part = r * sp - a * cp;

    let d_rx_part = cp * dr + (-r * sp + a * cp) * dphi_out;
    let d_ry_part = sp * dr + (r * cp + a * sp) * dphi_out;

    let (dx, dy) = if sin_t > 1e-12 {
        let d_sin_t = -(u / sin_t) * du;
        (
            d_rx_part * sin_t + rx_part * d_sin_t,
            d_ry_part * sin_t + ry_part * d_sin_t,
        )
    } else {
        // Near axis: sin_t -> 0, d_sin_t = sgn * |dtheta|. Fallback to radial unit direction.
        (d_rx_part * sin_t, d_ry_part * sin_t)
    };
    let dz = dr * u + r * du;
    DVec3::new(dx, dy, dz)
}

/// Exact Boyer-Lindquist metric matrix `g_BL` at `(r, theta)` for pullback verification.
pub fn boyer_lindquist_metric(r: f64, theta: f64, a: f64) -> DMat4 {
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let sin2 = sin_t * sin_t;
    let r2 = r * r;
    let a2 = a * a;
    let sigma = r2 + a2 * cos_t * cos_t;
    let delta = delta_factored(r, a);

    let g_tt = -(1.0 - 2.0 * r / sigma);
    let g_tphi = -2.0 * a * r * sin2 / sigma;
    let g_rr = sigma / delta;
    let g_thth = sigma;
    let g_phiphi = (r2 + a2 + 2.0 * a2 * r * sin2 / sigma) * sin2;

    DMat4::from_cols(
        DVec4::new(g_tt, 0.0, 0.0, g_tphi),
        DVec4::new(0.0, g_rr, 0.0, 0.0),
        DVec4::new(0.0, 0.0, g_thth, 0.0),
        DVec4::new(g_tphi, 0.0, 0.0, g_phiphi),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizons_and_isco_values() {
        assert!((r_plus(0.0) - 2.0).abs() < 1e-15);
        assert!((r_minus(0.0) - 0.0).abs() < 1e-15);
        assert!((r_isco(0.0) - 6.0).abs() < 1e-13);

        // Extremal prograde/retrograde limits
        assert!((r_plus(0.998) - (1.0 + (1.0 - 0.998_f64 * 0.998).sqrt())).abs() < 1e-15);
        assert!(r_isco(0.998) > 1.2 && r_isco(0.998) < 1.25);
        assert!(r_isco(-0.998) > 8.99 && r_isco(-0.998) < 9.0);
    }

    #[test]
    fn outgoing_ks_metric_pullback_equals_boyer_lindquist() {
        // Pull back the Outgoing Cartesian KS metric through the exact map
        // (t_BL, r, theta, phi_BL) -> (t_out, x, y, z) and verify J^T g_KS J == g_BL to 1e-12,
        // plus frame dragging omega > 0 for a > 0.
        let cases = [
            (0.0, 5.0, 1.1, 0.4),
            (0.7, 4.2, 0.9, 1.3),
            (0.998, 2.5, 1.4, -0.7),
            (-0.998, 3.1, 0.6, 2.1),
        ];
        for (a, r, theta, phi_bl) in cases {
            let pos = bl_to_outgoing_ks_pos(r, theta, phi_bl, a);
            let (r_back, th_back, phi_back) = outgoing_ks_to_bl_pos(pos, a);
            assert!((r_back - r).abs() < 1e-12, "r mismatch for a={a}");
            assert!((th_back - theta).abs() < 1e-12, "theta mismatch for a={a}");
            let dphi = (phi_back - phi_bl + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            assert!(dphi.abs() < 1e-12, "phi mismatch {dphi} for a={a}");

            // Analytic Jacobian J^mu_a = d x_KS^mu / d x_BL^a
            let delta = delta_factored(r, a);
            let phi_out = phi_bl - outgoing_ks_phi_twist(r, a);
            let (sp, cp) = phi_out.sin_cos();
            let (st, ct) = theta.sin_cos();

            // Column 0: d/dt_BL = (1, 0, 0, 0)
            let col_t = DVec4::new(1.0, 0.0, 0.0, 0.0);
            // Column 3: d/dphi_BL = (0, dx/dphi_out, dy/dphi_out, 0)
            let dx_dphi = (-r * sp + a * cp) * st;
            let dy_dphi = (r * cp + a * sp) * st;
            let col_phi = DVec4::new(0.0, dx_dphi, dy_dphi, 0.0);
            // Column 1: d/dr = (-2r/Delta, dx/dr|_phi_out - (a/Delta)*dx/dphi_out, ..., cos theta)
            let col_r = DVec4::new(
                -2.0 * r / delta,
                cp * st - (a / delta) * dx_dphi,
                sp * st - (a / delta) * dy_dphi,
                ct,
            );
            // Column 2: d/dtheta
            let col_th = DVec4::new(0.0, (r * cp + a * sp) * ct, (r * sp - a * cp) * ct, -r * st);
            let jac = DMat4::from_cols(col_t, col_r, col_th, col_phi);
            let g_ks = outgoing_ks_metric(pos, a);
            let g_pullback = jac.transpose() * g_ks * jac;
            let g_bl = boyer_lindquist_metric(r, theta, a);

            let max_err = (g_pullback - g_bl)
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_err < 1e-12,
                "KS->BL pullback error {max_err} for a={a}, r={r}, theta={theta}"
            );

            // Inverse metric check: g * g_inv == I
            let g_inv = outgoing_ks_inv_metric(pos, a);
            let prod = g_ks * g_inv - DMat4::IDENTITY;
            let inv_err = prod
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            assert!(inv_err < 1e-12, "g * g_inv error {inv_err}");

            // Frame dragging sign: omega > 0 for a > 0, omega < 0 for a < 0
            let tetrad = ZamoTetrad::from_bl(r, theta, phi_bl, a);
            if a > 0.0 {
                assert!(tetrad.omega > 0.0, "expected omega > 0 for a={a}");
            } else if a < 0.0 {
                assert!(tetrad.omega < 0.0, "expected omega < 0 for a={a}");
            }
        }
    }

    #[test]
    fn zamo_tetrad_orthonormality_including_ergosphere_and_axis() {
        let cases = [
            // Standard equatorial & inclined views
            (0.0, 20.0, 1.4, 0.3),
            (0.9, 15.0, 1.2, -0.5),
            // Deep inside ergosphere (at theta=pi/2, r_ergo = 2.0, r_+ = 1.063 for a=0.998)
            (0.998, 1.3, std::f64::consts::FRAC_PI_2, 0.7),
            (-0.998, 1.4, 1.3, 1.1),
            // Exact north pole and near-pole axis fallback
            (0.0, 20.0, 0.0, 0.0),
            (0.95, 12.0, 0.0, 0.4),
            (0.95, 12.0, 1e-5, 0.4),
            (0.95, 12.0, std::f64::consts::PI, 0.0),
        ];
        let eta = [-1.0, 1.0, 1.0, 1.0];
        for (a, r, theta, phi) in cases {
            let tet = ZamoTetrad::from_bl(r, theta, phi, a);
            let g = outgoing_ks_metric(tet.pos, a);
            let basis = [tet.e_t, tet.e_r, tet.e_theta, tet.e_phi];
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { eta[i] } else { 0.0 };
                    let got = metric_dot(&g, basis[i], basis[j]);
                    assert!(
                        (got - expected).abs() < 1e-11,
                        "tetrad dot ({i},{j})={got} != {expected} at a={a}, r={r}, th={theta}"
                    );
                }
            }
            // Ray initialization produces null momentum with camera-frame frequency == 1,
            // and agrees with independent BL Bardeen ZAMO on E and L_z!
            let dir = DVec3::new(-0.8, 0.3, -0.5).normalize();
            let (e_ks, p_ks) = tet.init_ray_covector(dir, a);
            let h_null = outgoing_ks_hamiltonian(tet.pos, p_ks, e_ks, a);
            assert!(h_null.abs() < 1e-12, "null condition error {h_null}");

            let p_down = DVec4::new(-e_ks, p_ks.x, p_ks.y, p_ks.z);
            let nu_obs = -p_down.dot(tet.e_t);
            assert!((nu_obs - 1.0).abs() < 1e-12, "nu_obs={nu_obs}");

            let bl_ray = tet.init_bl_ray(dir, a);
            let lz_ks = tet.pos.x * p_ks.y - tet.pos.y * p_ks.x;
            assert!(
                (e_ks - bl_ray.energy).abs() < 1e-11,
                "E mismatch KS={e_ks} vs BL={} at a={a}, r={r}, th={theta}",
                bl_ray.energy
            );
            assert!(
                (lz_ks - bl_ray.lz).abs() < 1e-11,
                "Lz mismatch KS={lz_ks} vs BL={} at a={a}, r={r}, th={theta}",
                bl_ray.lz
            );
        }
    }

    #[test]
    fn outgoing_ks_hamiltonian_rhs_matches_finite_differences() {
        let a = 0.93;
        let pos = DVec3::new(3.2, -1.7, 1.4);
        let p = DVec3::new(-0.4, 0.65, -0.25);
        let energy = 1.1;
        let (dx, dp) = outgoing_ks_rhs(pos, p, energy, a);
        let eps = 1e-6;

        for axis in 0..3 {
            let mut dp_plus = p;
            let mut dp_minus = p;
            dp_plus[axis] += eps;
            dp_minus[axis] -= eps;
            let fd_dx = (outgoing_ks_hamiltonian(pos, dp_plus, energy, a)
                - outgoing_ks_hamiltonian(pos, dp_minus, energy, a))
                / (2.0 * eps);
            assert!(
                (dx[axis] - fd_dx).abs() < 1e-9,
                "dx[{axis}] analytic={} vs fd={fd_dx}",
                dx[axis]
            );

            let mut x_plus = pos;
            let mut x_minus = pos;
            x_plus[axis] += eps;
            x_minus[axis] -= eps;
            let fd_dp = -(outgoing_ks_hamiltonian(x_plus, p, energy, a)
                - outgoing_ks_hamiltonian(x_minus, p, energy, a))
                / (2.0 * eps);
            assert!(
                (dp[axis] - fd_dp).abs() < 1e-9,
                "dp[{axis}] analytic={} vs fd={fd_dp}",
                dp[axis]
            );
        }
    }
}
