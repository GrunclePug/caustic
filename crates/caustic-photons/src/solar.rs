//! Scientifically accurate true-ratio Solar System model with `f64` Keplerian orbits
//! and KSP/KSA-style floating-origin camera-relative GPU compilation.
//!
//! All body radii, orbital semi-major axes, eccentricities, periods, Bond albedos,
//! and ring bounds use exact NASA fact-sheet ratios (`1 AU = 149,597,870.7 km`).
//! Supports both `1:1 Real` and `1:10 (KSP/KSA)` linear system scale—preserving every
//! physical geometric ratio identically while letting users zoom seamlessly across
//! 7 orders of magnitude (`40 AU` down to `4,000 km`).

use crate::scene::{
    EMITTER_DISK, EMITTER_SOLAR_FAR, GpuEmitter, MAT_ATMOSPHERE, MAT_DIFFUSE, MAT_SPLITTER,
    OptCircle, OptSegment, SPEC_BLACKBODY,
};

/// Astronomical Unit in kilometers (IAU 2012 exact definition).
pub const AU_KM: f64 = 149_597_870.7;
/// Speed of light in km/s (exact).
pub const C_KM_S: f64 = 299_792.458;
/// Light travel time for 1 AU in seconds (`~499.0048 s = 8m 19s`).
pub const LIGHT_TIME_AU_S: f64 = AU_KM / C_KM_S;
/// Julian year in seconds (`365.25 * 86400`).
pub const YEAR_S: f64 = 31_557_600.0;
/// Exact solar mean radius in AU (`696,340 km / AU`).
pub const SUN_RADIUS_AU: f64 = 696_340.0 / AU_KM;

/// Uniform system scale factor (both radii AND distances scale identically -> exact ratios!).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarScale {
    /// Exact 1:1 real-world Solar System (`1 AU = 149.6M km`, light-time `8m 19s`).
    Real1To1,
    /// KSP/KSA-style 1/10th linear scale (`1 AU = 14.96M km`, light-time `49.9s`, identical ratios!).
    Scaled1To10,
}

impl SolarScale {
    /// Linear scale factor applied to physical lengths (km) and light-travel times.
    pub fn factor(self) -> f64 {
        match self {
            Self::Real1To1 => 1.0,
            Self::Scaled1To10 => 0.1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Real1To1 => "1:1 Real",
            Self::Scaled1To10 => "1:10 (KSP/KSA)",
        }
    }
}

/// Camera focus mode in the Solar System.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarFocus {
    /// Full Solar System overview out to Neptune (~36 AU).
    System,
    /// Inner rocky planets overview (~1.85 AU).
    Inner,
    /// Lock camera onto a specific body by index in [`SolarSystem::bodies`].
    Body(usize),
}

/// One astronomical body with exact NASA physical & Keplerian orbital parameters.
#[derive(Debug, Clone, Copy)]
pub struct SolarBody {
    pub name: &'static str,
    /// True mean radius in AU (ratio to `1 AU` is exact!).
    pub radius_au: f64,
    /// Orbital semi-major axis in AU (`0.0` for the Sun).
    pub semi_major_au: f64,
    /// Orbital eccentricity `e in [0, 1)`.
    pub eccentricity: f64,
    /// Orbital period in real seconds (`0.0` for the Sun).
    pub period_s: f64,
    /// Argument of periapsis in radians (orientation of ellipse).
    pub arg_periapsis: f64,
    /// Initial mean anomaly at `t = 0` in radians.
    pub mean_anomaly_0: f64,
    /// Optional parent body index (`None` = Sun, `Some(idx)` = planet for moons).
    pub parent: Option<usize>,
    /// Realistic visible RGB reflectance (scaled to Bond albedo).
    pub albedo: [f32; 3],
    /// Marker tint in zoomed-out KSP/KSA map view.
    pub map_color: [u8; 3],
    /// Effective optical atmosphere thickness as a fraction of `radius_au` (`0.0` = airless).
    pub atmo_thickness: f64,
    /// Dimensionless Rayleigh optical depth per planet radius at `550 nm`.
    pub atmo_tau_per_r: f64,
    /// Atmospheric limb refractive index delta `n - 1` (e.g. `0.0022` for Earth's `1.12 deg` refraction).
    pub atmo_n_delta: f64,
    /// Optional planetary ring inner/outer radii in multiples of `radius_au` (`0.0` = none).
    pub ring_inner_mult: f64,
    pub ring_outer_mult: f64,
    /// Ring tilt angle in orbital slice (radians).
    pub ring_angle: f64,
}

/// Canonical indices in [`SolarSystem::bodies`].
pub const BODY_SUN: usize = 0;
pub const BODY_MERCURY: usize = 1;
pub const BODY_VENUS: usize = 2;
pub const BODY_EARTH: usize = 3;
pub const BODY_MOON: usize = 4;
pub const BODY_MARS: usize = 5;
pub const BODY_JUPITER: usize = 6;
pub const BODY_IO: usize = 7;
pub const BODY_EUROPA: usize = 8;
pub const BODY_SATURN: usize = 9;
pub const BODY_URANUS: usize = 10;
pub const BODY_NEPTUNE: usize = 11;

/// Solve Kepler's equation `E - e * sin(E) = M` to `< 1e-14` in `f64`.
#[inline]
pub fn solve_kepler(mean_anomaly: f64, e: f64) -> f64 {
    let m = mean_anomaly.rem_euclid(std::f64::consts::TAU);
    let mut ea = if e > 0.8 {
        std::f64::consts::PI
    } else {
        m + e * m.sin()
    };
    for _ in 0..8 {
        let f = ea - e * ea.sin() - m;
        let fp = 1.0 - e * ea.cos();
        let delta = f / fp;
        ea -= delta;
        if delta.abs() < 1e-14 {
            break;
        }
    }
    ea
}

/// Compiled floating-origin GPU snapshot + HUD telemetry for the Solar System.
#[derive(Debug, Clone)]
pub struct SolarGpuFrame {
    pub emitters: Vec<GpuEmitter>,
    pub segments: Vec<OptSegment>,
    pub circles: Vec<OptCircle>,
    /// True if using far-field frustum-sampled sunlight (Sun outside viewport).
    pub far_sun_mode: bool,
    /// Current `view_half_height` in AU.
    pub view_half_height_au: f64,
    /// Current `view_center` in AU (`f64`).
    pub view_center_au: [f64; 2],
    /// Distance from camera center to Sun in AU.
    pub sun_dist_au: f64,
    /// Exact angular diameter of the Sun as seen from camera center (degrees).
    pub sun_angular_diam_deg: f64,
    /// Light travel time from Sun to camera center (seconds, scaled by [`SolarScale`]).
    pub light_time_s: f64,
    /// Screen-space normalized GPU positions (`x in [-aspect, aspect], y in [-1, 1]`) and GPU radii for all bodies.
    pub body_gpu_pos: Vec<[f64; 2]>,
    pub body_gpu_rad: Vec<f64>,
}

/// Complete double-precision Solar System state.
#[derive(Debug, Clone)]
pub struct SolarSystem {
    pub bodies: Vec<SolarBody>,
    pub scale: SolarScale,
    pub focus: SolarFocus,
    /// Simulated elapsed time in real seconds.
    pub sim_time_s: f64,
    /// Time warp multiplier (simulated seconds per real second; e.g. `1.0` = real time, `86400.0` = 1 day/s).
    pub time_warp: f64,
    /// Vertical half-height of camera view in AU (`0.00003..=45.0`).
    pub view_half_height_au: f64,
    /// Manual pan offset from the focused target in AU.
    pub pan_offset_au: [f64; 2],
    /// Cached world positions in AU for all bodies at `sim_time_s`.
    positions_au: Vec<[f64; 2]>,
}

impl Default for SolarSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SolarSystem {
    /// Construct our exact-ratio Solar System starting framed on an iconic Earth + Moon Solar Eclipse!
    pub fn new() -> Self {
        let km = |r_km: f64| r_km / AU_KM;
        let bodies = vec![
            SolarBody {
                name: "Sun",
                radius_au: SUN_RADIUS_AU,
                semi_major_au: 0.0,
                eccentricity: 0.0,
                period_s: 0.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 0.0,
                parent: None,
                albedo: [1.0, 0.96, 0.86],
                map_color: [255, 224, 150],
                atmo_thickness: 0.0,
                atmo_tau_per_r: 0.0,
                atmo_n_delta: 0.0,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Mercury",
                radius_au: km(2_439.7),
                semi_major_au: 0.387_098,
                eccentricity: 0.205_63,
                period_s: 0.240_846_7 * YEAR_S,
                arg_periapsis: 0.508,
                mean_anomaly_0: 2.1,
                parent: None,
                albedo: [0.22, 0.21, 0.20],
                map_color: [168, 162, 156],
                atmo_thickness: 0.0,
                atmo_tau_per_r: 0.0,
                atmo_n_delta: 0.0,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Venus",
                radius_au: km(6_051.8),
                semi_major_au: 0.723_332,
                eccentricity: 0.006_77,
                period_s: 0.615_197_26 * YEAR_S,
                arg_periapsis: 0.958,
                mean_anomaly_0: -2.15,
                parent: None,
                albedo: [0.84, 0.76, 0.56],
                map_color: [232, 200, 138],
                atmo_thickness: 0.032,
                atmo_tau_per_r: 9.5,
                atmo_n_delta: 0.0035,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Earth",
                radius_au: km(6_371.0),
                semi_major_au: 1.000_000,
                eccentricity: 0.016_71,
                period_s: YEAR_S,
                arg_periapsis: 1.796,
                // Starts along +X from the Sun so horizontal 16:9 widescreen eclipses frame naturally.
                mean_anomaly_0: std::f64::consts::TAU - 1.796,
                parent: None,
                albedo: [0.28, 0.46, 0.72],
                map_color: [108, 174, 245],
                // Real optical Rayleigh scattering limb + ~1.1 deg sunset refraction into lunar umbra.
                atmo_thickness: 0.034,
                atmo_tau_per_r: 6.2,
                atmo_n_delta: 0.0024,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Moon",
                radius_au: km(1_737.4),
                semi_major_au: km(384_400.0),
                eccentricity: 0.054_9,
                period_s: 27.321_661 * 86_400.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 0.35,
                parent: Some(BODY_EARTH),
                albedo: [0.34, 0.33, 0.31],
                map_color: [195, 194, 190],
                atmo_thickness: 0.0,
                atmo_tau_per_r: 0.0,
                atmo_n_delta: 0.0,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Mars",
                radius_au: km(3_389.5),
                semi_major_au: 1.523_679,
                eccentricity: 0.093_4,
                period_s: 1.880_847_6 * YEAR_S,
                arg_periapsis: 5.000,
                mean_anomaly_0: 2.05,
                parent: None,
                albedo: [0.62, 0.27, 0.16],
                map_color: [224, 118, 82],
                atmo_thickness: 0.018,
                atmo_tau_per_r: 2.2,
                atmo_n_delta: 0.0005,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Jupiter",
                radius_au: km(69_911.0),
                semi_major_au: 5.203_363,
                eccentricity: 0.048_9,
                period_s: 11.862_615 * YEAR_S,
                arg_periapsis: 4.779,
                // Near +X at t=0 so Io transits stream horizontally across Jupiter's sunlit clouds.
                mean_anomaly_0: 1.62,
                parent: None,
                albedo: [0.72, 0.60, 0.46],
                map_color: [218, 184, 142],
                atmo_thickness: 0.012,
                atmo_tau_per_r: 3.0,
                atmo_n_delta: 0.0008,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Io",
                radius_au: km(1_821.6),
                semi_major_au: km(421_700.0),
                eccentricity: 0.004_1,
                period_s: 1.769_138 * 86_400.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 0.8,
                parent: Some(BODY_JUPITER),
                albedo: [0.82, 0.72, 0.34],
                map_color: [232, 206, 108],
                atmo_thickness: 0.0,
                atmo_tau_per_r: 0.0,
                atmo_n_delta: 0.0,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Europa",
                radius_au: km(1_560.8),
                semi_major_au: km(671_100.0),
                eccentricity: 0.009,
                period_s: 3.551_181 * 86_400.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 2.3,
                parent: Some(BODY_JUPITER),
                albedo: [0.76, 0.72, 0.66],
                map_color: [210, 204, 192],
                atmo_thickness: 0.0,
                atmo_tau_per_r: 0.0,
                atmo_n_delta: 0.0,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Saturn",
                radius_au: km(58_232.0),
                semi_major_au: 9.537_070,
                eccentricity: 0.056_5,
                period_s: 29.447_498 * YEAR_S,
                arg_periapsis: 5.923,
                mean_anomaly_0: 2.60,
                parent: None,
                albedo: [0.74, 0.66, 0.48],
                map_color: [232, 212, 162],
                atmo_thickness: 0.012,
                atmo_tau_per_r: 2.6,
                atmo_n_delta: 0.0007,
                // Real C/B/A ring span: 74,500 km to 136,775 km (1.28 to 2.35 R_saturn).
                ring_inner_mult: 1.28,
                ring_outer_mult: 2.35,
                ring_angle: 0.62,
            },
            SolarBody {
                name: "Uranus",
                radius_au: km(25_362.0),
                semi_major_au: 19.191_264,
                eccentricity: 0.047_2,
                period_s: 84.016_846 * YEAR_S,
                arg_periapsis: 1.688,
                mean_anomaly_0: 1.85,
                parent: None,
                albedo: [0.44, 0.72, 0.78],
                map_color: [148, 214, 228],
                atmo_thickness: 0.018,
                atmo_tau_per_r: 4.5,
                atmo_n_delta: 0.0008,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Neptune",
                radius_au: km(24_622.0),
                semi_major_au: 30.068_963,
                eccentricity: 0.008_6,
                period_s: 164.791_32 * YEAR_S,
                arg_periapsis: 4.769,
                mean_anomaly_0: -0.35,
                parent: None,
                albedo: [0.24, 0.44, 0.84],
                map_color: [98, 142, 238],
                atmo_thickness: 0.018,
                atmo_tau_per_r: 5.5,
                atmo_n_delta: 0.0008,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
        ];
        let n = bodies.len();
        let mut s = Self {
            bodies,
            scale: SolarScale::Real1To1,
            focus: SolarFocus::Body(BODY_EARTH),
            sim_time_s: 0.0,
            time_warp: 3_600.0, // 1 hour/sec (visible lunar motion!)
            view_half_height_au: 0.000_25,
            pan_offset_au: [0.0, 0.0],
            positions_au: vec![[0.0, 0.0]; n],
        };
        s.update_positions();
        // Start with a clean, awe-inspiring Earth + Moon Lunar Eclipse alignment so the
        // first glance immediately showcases true-scale shadows, blue sky, and red umbra!
        s.stage_lunar_eclipse();
        s
    }

    /// Recompute exact `f64` Keplerian positions for all bodies at `self.sim_time_s`.
    pub fn update_positions(&mut self) {
        for i in 0..self.bodies.len() {
            let b = self.bodies[i];
            if b.semi_major_au <= 0.0 || b.period_s <= 0.0 {
                self.positions_au[i] = [0.0, 0.0];
                continue;
            }
            let m = b.mean_anomaly_0 + std::f64::consts::TAU * (self.sim_time_s / b.period_s);
            let ea = solve_kepler(m, b.eccentricity);
            let x_orb = b.semi_major_au * (ea.cos() - b.eccentricity);
            let y_orb = b.semi_major_au * (1.0 - b.eccentricity * b.eccentricity).sqrt() * ea.sin();
            let cw = b.arg_periapsis.cos();
            let sw = b.arg_periapsis.sin();
            let rx = x_orb * cw - y_orb * sw;
            let ry = x_orb * sw + y_orb * cw;
            let parent_pos = b.parent.map(|p| self.positions_au[p]).unwrap_or([0.0, 0.0]);
            self.positions_au[i] = [parent_pos[0] + rx, parent_pos[1] + ry];
        }
    }

    /// Advance orbital clock by `dt_real_s * time_warp` seconds.
    pub fn step_time(&mut self, dt_real_s: f64) -> bool {
        if self.time_warp.abs() < 1e-9 || dt_real_s <= 0.0 {
            return false;
        }
        self.sim_time_s += dt_real_s * self.time_warp;
        self.update_positions();
        true
    }

    /// Get cached `f64` position in AU for body `idx`.
    pub fn body_pos_au(&self, idx: usize) -> [f64; 2] {
        self.positions_au.get(idx).copied().unwrap_or([0.0, 0.0])
    }

    /// Current camera center in AU (`f64`).
    pub fn view_center_au(&self) -> [f64; 2] {
        let base = match self.focus {
            SolarFocus::System | SolarFocus::Inner => [0.0, 0.0],
            SolarFocus::Body(idx) => self.body_pos_au(idx),
        };
        [
            base[0] + self.pan_offset_au[0],
            base[1] + self.pan_offset_au[1],
        ]
    }

    /// Focus camera smoothly onto a system overview or specific body with ideal framing.
    pub fn set_focus(&mut self, focus: SolarFocus) {
        self.focus = focus;
        self.pan_offset_au = [0.0, 0.0];
        self.view_half_height_au = match focus {
            SolarFocus::System => 33.5,
            SolarFocus::Inner => 1.95,
            SolarFocus::Body(BODY_SUN) => SUN_RADIUS_AU * 4.5,
            SolarFocus::Body(BODY_SATURN) => self.bodies[BODY_SATURN].radius_au * 4.5,
            SolarFocus::Body(BODY_JUPITER) => self.bodies[BODY_JUPITER].radius_au * 5.0,
            SolarFocus::Body(idx) => {
                let r = self.bodies.get(idx).map(|b| b.radius_au).unwrap_or(0.0001);
                (r * 4.2).max(0.000_035)
            }
        };
        // Set a sensible orbital speed for each zoom scale so planets/moons visibly move without flying off.
        self.time_warp = match focus {
            SolarFocus::System => 30.0 * 86_400.0, // 30 days/s
            SolarFocus::Inner => 5.0 * 86_400.0,   // 5 days/s
            SolarFocus::Body(BODY_EARTH | BODY_MOON) => 3_600.0, // 1 hr/s
            SolarFocus::Body(BODY_JUPITER | BODY_IO | BODY_EUROPA) => 1_800.0, // 30 min/s
            _ => 86_400.0,                         // 1 day/s
        };
    }

    /// Stage an exact true-ratio **Total Solar Eclipse** (Moon at perigee directly between Sun and Earth!).
    pub fn stage_solar_eclipse(&mut self) {
        self.sim_time_s = 0.0;
        self.update_positions();
        let earth_pos = self.body_pos_au(BODY_EARTH);
        let sun_to_earth_ang = earth_pos[1].atan2(earth_pos[0]);
        // Place Moon's perigee directly sunward of Earth (`sun_to_earth_ang + PI`) at `M = 0` (exact perigee = 363,300 km!).
        self.bodies[BODY_MOON].arg_periapsis = sun_to_earth_ang + std::f64::consts::PI;
        self.bodies[BODY_MOON].mean_anomaly_0 = 0.0;
        self.update_positions();

        let moon_pos = self.body_pos_au(BODY_MOON);
        // Frame Moon + tapering lunar umbra + Earth horizontally across the widescreen canvas.
        self.focus = SolarFocus::Body(BODY_EARTH);
        self.pan_offset_au = [
            0.40 * (moon_pos[0] - earth_pos[0]),
            0.40 * (moon_pos[1] - earth_pos[1]),
        ];
        self.view_half_height_au = 0.001_05;
        self.time_warp = 900.0; // 15 min/s so you can watch the umbra sweep across Earth!
    }

    /// Stage an exact true-ratio **Total Lunar Eclipse** (Moon behind Earth inside Earth's copper-red umbra!).
    pub fn stage_lunar_eclipse(&mut self) {
        self.sim_time_s = 0.0;
        self.update_positions();
        let earth_pos = self.body_pos_au(BODY_EARTH);
        let sun_to_earth_ang = earth_pos[1].atan2(earth_pos[0]);
        // Place Moon directly anti-sunward of Earth inside Earth's shadow cone!
        self.bodies[BODY_MOON].arg_periapsis = sun_to_earth_ang;
        self.bodies[BODY_MOON].mean_anomaly_0 = 0.0;
        self.update_positions();

        let moon_pos = self.body_pos_au(BODY_MOON);
        self.focus = SolarFocus::Body(BODY_EARTH);
        self.pan_offset_au = [
            0.70 * (moon_pos[0] - earth_pos[0]),
            0.70 * (moon_pos[1] - earth_pos[1]),
        ];
        self.view_half_height_au = 0.001_18;
        self.time_warp = 900.0;
    }

    /// Stage **Saturn & its true-scale C/B/A icy ring system**.
    pub fn stage_saturn_rings(&mut self) {
        self.set_focus(SolarFocus::Body(BODY_SATURN));
        self.view_half_height_au = self.bodies[BODY_SATURN].radius_au * 3.8;
    }

    /// Stage **Io transiting directly sunward of Jupiter**, casting a sharp moon shadow onto Jupiter's clouds!
    pub fn stage_jupiter_io_transit(&mut self) {
        self.sim_time_s = 0.0;
        self.update_positions();
        let jup_pos = self.body_pos_au(BODY_JUPITER);
        let ang = jup_pos[1].atan2(jup_pos[0]);
        self.bodies[BODY_IO].arg_periapsis = ang + std::f64::consts::PI;
        self.bodies[BODY_IO].mean_anomaly_0 = 0.02;
        self.update_positions();

        let io_pos = self.body_pos_au(BODY_IO);
        self.focus = SolarFocus::Body(BODY_JUPITER);
        self.pan_offset_au = [
            0.42 * (io_pos[0] - jup_pos[0]),
            0.42 * (io_pos[1] - jup_pos[1]),
        ];
        self.view_half_height_au = 0.001_35;
        self.time_warp = 600.0;
    }

    /// Sample points along body `idx`'s exact Keplerian ellipse in AU (for crisp KSP/KSA orbit guides).
    pub fn sample_orbit_ellipse_au(&self, idx: usize, num_pts: usize) -> Vec<[f64; 2]> {
        let Some(b) = self.bodies.get(idx) else {
            return Vec::new();
        };
        if b.semi_major_au <= 0.0 {
            return Vec::new();
        }
        let parent_pos = b.parent.map(|p| self.body_pos_au(p)).unwrap_or([0.0, 0.0]);
        let cw = b.arg_periapsis.cos();
        let sw = b.arg_periapsis.sin();
        let b_semi = b.semi_major_au * (1.0 - b.eccentricity * b.eccentricity).sqrt();
        let mut pts = Vec::with_capacity(num_pts + 1);
        for k in 0..=num_pts {
            let ea = std::f64::consts::TAU * (k as f64) / (num_pts as f64);
            let xo = b.semi_major_au * (ea.cos() - b.eccentricity);
            let yo = b_semi * ea.sin();
            pts.push([
                parent_pos[0] + xo * cw - yo * sw,
                parent_pos[1] + xo * sw + yo * cw,
            ]);
        }
        pts
    }

    /// Compile the `f64` Solar System into camera-centered normalized GPU primitives (`view_half_height = 1.0`).
    ///
    /// Automatically switches between direct `EMITTER_DISK` (when the Sun is inside/near the frame)
    /// and frustum importance-sampled `EMITTER_SOLAR_FAR` (when zoomed into a planet/moon up to
    /// 30 AU away, preserving exact solar angular diameter and sub-pixel FP32 precision!).
    pub fn compile_to_gpu(&self, aspect: f64) -> SolarGpuFrame {
        let center = self.view_center_au();
        let half_h = self.view_half_height_au.max(1e-6);
        let inv_s = 1.0 / half_h;
        let half_w_gpu = aspect.max(0.5);
        let screen_diag_gpu = (half_w_gpu * half_w_gpu + 1.0).sqrt();

        let mut body_gpu_pos = Vec::with_capacity(self.bodies.len());
        let mut body_gpu_rad = Vec::with_capacity(self.bodies.len());
        for (i, b) in self.bodies.iter().enumerate() {
            let p = self.positions_au[i];
            body_gpu_pos.push([(p[0] - center[0]) * inv_s, (p[1] - center[1]) * inv_s]);
            body_gpu_rad.push(b.radius_au * inv_s);
        }

        let sun_dx = center[0] - self.positions_au[BODY_SUN][0];
        let sun_dy = center[1] - self.positions_au[BODY_SUN][1];
        let sun_dist_au = (sun_dx * sun_dx + sun_dy * sun_dy).sqrt();
        let sun_dist_gpu = sun_dist_au * inv_s;
        let sun_rad_gpu = SUN_RADIUS_AU * inv_s;

        // Exact angular radius of the Sun as seen from camera center.
        let alpha_sun = if sun_dist_au > SUN_RADIUS_AU {
            (SUN_RADIUS_AU / sun_dist_au).clamp(0.0, 1.0).asin()
        } else {
            std::f64::consts::FRAC_PI_2
        };

        let far_sun_mode = sun_dist_gpu > screen_diag_gpu * 3.2 + sun_rad_gpu;

        let mut circles = Vec::new();
        let mut segments = Vec::new();
        let sun_to_cam_ang = sun_dy.atan2(sun_dx);
        let sdir_x = sun_to_cam_ang.cos();
        let sdir_y = sun_to_cam_ang.sin();
        // Tight projected upstream spawn plane just past the viewport boundary (fast wavefront entry
        // in Pulse mode, automatically pushed further back whenever an upstream occluder is present).
        let mut d_spawn_gpu = (half_w_gpu * sdir_x.abs() + sdir_y.abs()) * 1.25;

        // Include all bodies visible in the viewport OR lying upstream along the Sun corridor
        // (so Earth still eclipses the Moon even when zoomed tightly onto the Moon!).
        for (i, b) in self.bodies.iter().enumerate().skip(1) {
            let gp = body_gpu_pos[i];
            let gr = body_gpu_rad[i];
            let dist_from_cam = (gp[0] * gp[0] + gp[1] * gp[1]).sqrt();

            // Check if body lies in the Sun-to-viewport light corridor.
            let proj_along_sun = -(gp[0] * sdir_x + gp[1] * sdir_y); // positive = upstream toward Sun
            let perp_dist = (gp[0] * -sdir_y + gp[1] * sdir_x).abs();
            let max_corridor_half =
                screen_diag_gpu * 1.35 + proj_along_sun.max(0.0) * alpha_sun.tan() + gr * 2.5;
            let in_upstream_corridor = far_sun_mode
                && proj_along_sun > -screen_diag_gpu * 1.5
                && proj_along_sun < sun_dist_gpu - sun_rad_gpu
                && perp_dist <= max_corridor_half;

            let in_view_neighborhood = dist_from_cam < screen_diag_gpu * 3.5 + gr * 3.0;
            if !in_view_neighborhood && !in_upstream_corridor {
                continue;
            }
            // Skip sub-microscopic dust specks when zoomed out to 35 AU so GPU intersection stays crisp
            // (bodies remain clickable via KSP/KSA HUD markers and appear seamlessly as you zoom in).
            if gr < 1e-4 {
                continue;
            }

            if in_upstream_corridor && proj_along_sun + gr * 2.0 > d_spawn_gpu {
                d_spawn_gpu = (proj_along_sun + gr * 2.5).min(sun_dist_gpu - sun_rad_gpu - 0.1);
            }

            // Solid planet body (`MAT_DIFFUSE`).
            circles.push(OptCircle {
                geom: [gp[0] as f32, gp[1] as f32, gr as f32, 0.0],
                clip_arc: [0.0, 0.0, -2.0, 0.0],
                material: [b.albedo[0], b.albedo[1], b.albedo[2], 0.0],
                meta: [MAT_DIFFUSE, 0, 0, 0],
            });

            // Optional Rayleigh atmosphere shell (`MAT_ATMOSPHERE`) with zoom-invariant optical depth!
            if b.atmo_thickness > 0.0 && b.atmo_tau_per_r > 0.0 && gr > 0.002 {
                let r_atmo = gr * (1.0 + b.atmo_thickness);
                let gpu_density = b.atmo_tau_per_r / gr;
                circles.push(OptCircle {
                    geom: [gp[0] as f32, gp[1] as f32, r_atmo as f32, 0.0],
                    clip_arc: [0.0, 0.0, -2.0, 0.0],
                    material: [gpu_density as f32, b.atmo_n_delta as f32, gr as f32, 0.0],
                    meta: [MAT_ATMOSPHERE, 0, 0, 0],
                });
            }

            // Optional icy planetary rings (`MAT_SPLITTER`, subkind = 1).
            if b.ring_outer_mult > b.ring_inner_mult && gr > 0.001 {
                let r_in = gr * b.ring_inner_mult;
                let r_out = gr * b.ring_outer_mult;
                let ca = b.ring_angle.cos();
                let sa = b.ring_angle.sin();
                for sign in [-1.0_f64, 1.0_f64] {
                    let p0 = [gp[0] + sign * r_in * ca, gp[1] + sign * r_in * sa];
                    let p1 = [gp[0] + sign * r_out * ca, gp[1] + sign * r_out * sa];
                    segments.push(OptSegment {
                        endpoints: [p0[0] as f32, p0[1] as f32, p1[0] as f32, p1[1] as f32],
                        material: [0.88, 0.80, 0.64, 0.42],
                        meta: [MAT_SPLITTER, 1, 0, 0],
                    });
                }
            }
        }

        let emitters = if far_sun_mode {
            // Exact projected half-span of the rectangular viewport perpendicular to sunlight,
            // concentrating ~100% of GPU photons through the visible frame!
            let exact_span = half_w_gpu * sdir_y.abs() + sdir_x.abs();
            let span_half = exact_span * 1.06 + alpha_sun.tan() * screen_diag_gpu;
            vec![GpuEmitter {
                geom: [0.0, 0.0, sun_to_cam_ang as f32, span_half as f32],
                spec: [5778.0, 1.0, 1.0, alpha_sun as f32],
                info: [
                    EMITTER_SOLAR_FAR,
                    SPEC_BLACKBODY,
                    (d_spawn_gpu as f32).to_bits(),
                    0,
                ],
            }]
        } else {
            vec![GpuEmitter {
                geom: [
                    body_gpu_pos[BODY_SUN][0] as f32,
                    body_gpu_pos[BODY_SUN][1] as f32,
                    0.0,
                    sun_rad_gpu as f32,
                ],
                spec: [5778.0, 1.0, 1.0, 0.0],
                info: [EMITTER_DISK, SPEC_BLACKBODY, 0, 0],
            }]
        };

        SolarGpuFrame {
            emitters,
            segments,
            circles,
            far_sun_mode,
            view_half_height_au: half_h,
            view_center_au: center,
            sun_dist_au,
            sun_angular_diam_deg: (2.0 * alpha_sun).to_degrees(),
            light_time_s: sun_dist_au * LIGHT_TIME_AU_S * self.scale.factor(),
            body_gpu_pos,
            body_gpu_rad,
        }
    }
}

/// Format an astronomical distance (`au`) scaled by [`SolarScale`] cleanly for HUD telemetry.
pub fn format_distance_au(au: f64, scale: SolarScale) -> String {
    let km = au * AU_KM * scale.factor();
    if au >= 0.25 && scale == SolarScale::Real1To1 {
        format!("{au:.2} AU")
    } else if km >= 10_000_000.0 {
        format!("{:.2}M km", km * 1e-6)
    } else if km >= 100_000.0 {
        format!("{:.0}k km", km * 1e-3)
    } else {
        format!("{km:.0} km")
    }
}

/// Format a light-travel duration in seconds cleanly (`s`, `m s`, or `h m`).
pub fn format_duration_s(sec: f64) -> String {
    if sec < 60.0 {
        format!("{sec:.1}s")
    } else if sec < 3600.0 {
        let m = (sec / 60.0).floor() as u64;
        let s = (sec % 60.0).round() as u64;
        format!("{m}m {s:02}s")
    } else {
        let h = (sec / 3600.0).floor() as u64;
        let m = ((sec % 3600.0) / 60.0).round() as u64;
        format!("{h}h {m:02}m")
    }
}

/// Format orbital time warp rate cleanly (`1x`, `15m/s`, `1d/s`, etc.).
pub fn format_warp(warp: f64) -> String {
    if warp < 60.0 {
        format!("{warp:.0}x")
    } else if warp < 3600.0 {
        format!("{:.0}m/s", warp / 60.0)
    } else if warp < 86_400.0 {
        format!("{:.1}h/s", warp / 3600.0)
    } else {
        format!("{:.1}d/s", warp / 86_400.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_nasa_ratios_and_kepler_third_law() {
        let sys = SolarSystem::new();

        // Sun-to-Earth radius ratio ~ 109.3
        let sun_earth_r = sys.bodies[BODY_SUN].radius_au / sys.bodies[BODY_EARTH].radius_au;
        assert!(
            (sun_earth_r - 109.3).abs() < 0.2,
            "Sun/Earth radius ratio={sun_earth_r:.2}"
        );

        // Earth-Moon distance / Earth radius ~ 60.34
        let moon_dist_r = sys.bodies[BODY_MOON].semi_major_au / sys.bodies[BODY_EARTH].radius_au;
        assert!(
            (moon_dist_r - 60.34).abs() < 0.1,
            "Moon distance / R_earth={moon_dist_r:.2}"
        );

        // Total vs Annular eclipse physics: Moon angular radius > Sun at perigee, < Sun at apogee!
        let alpha_sun = SUN_RADIUS_AU; // at 1 AU
        let moon = sys.bodies[BODY_MOON];
        let alpha_moon_perigee = moon.radius_au / (moon.semi_major_au * (1.0 - moon.eccentricity));
        let alpha_moon_apogee = moon.radius_au / (moon.semi_major_au * (1.0 + moon.eccentricity));
        assert!(
            alpha_moon_perigee > alpha_sun,
            "Moon at perigee must exceed Sun angular radius for total eclipse"
        );
        assert!(
            alpha_moon_apogee < alpha_sun,
            "Moon at apogee must be smaller than Sun angular radius for annular eclipse"
        );

        // Kepler's Third Law: T_yr^2 / a_AU^3 == 1.0 to < 0.5% for all 8 planets!
        for idx in [
            BODY_MERCURY,
            BODY_VENUS,
            BODY_EARTH,
            BODY_MARS,
            BODY_JUPITER,
            BODY_SATURN,
            BODY_URANUS,
            BODY_NEPTUNE,
        ] {
            let b = sys.bodies[idx];
            let t_yr = b.period_s / YEAR_S;
            let ratio = (t_yr * t_yr) / b.semi_major_au.powi(3);
            assert!(
                (ratio - 1.0).abs() < 5e-3,
                "{} failed Kepler 3rd Law: T^2/a^3 = {ratio:.5}",
                b.name
            );
        }
    }

    #[test]
    fn kepler_solver_and_floating_origin_eclipse_geometry() {
        for e in [0.0, 0.0549, 0.2056] {
            for k in 0..16 {
                let m = std::f64::consts::TAU * (f64::from(k) / 16.0);
                let ea = solve_kepler(m, e);
                let resid = (ea - e * ea.sin() - m).abs();
                assert!(resid < 1e-13, "Kepler residual {resid:.2e} at e={e}");
            }
        }

        let mut sys = SolarSystem::new();
        sys.stage_solar_eclipse();
        let frame = sys.compile_to_gpu(16.0 / 9.0);
        assert!(
            frame.far_sun_mode,
            "Earth eclipse closeup must use far-sun importance sampling"
        );
        assert!(
            (frame.sun_angular_diam_deg - 0.533).abs() < 0.02,
            "Sun angular diameter at Earth should be ~0.533 deg, got {:.4}",
            frame.sun_angular_diam_deg
        );
        // Both Earth (diffuse + atmosphere) and Moon (diffuse) must be compiled into GPU circles!
        assert!(
            frame.circles.len() >= 3,
            "expected Earth + Earth atmo + Moon circles, got {}",
            frame.circles.len()
        );
        let d_spawn = f32::from_bits(frame.emitters[0].info[2]);
        assert!(
            d_spawn > 1.5 && d_spawn.is_finite(),
            "invalid upstream spawn distance d_spawn={d_spawn}"
        );
    }
}
