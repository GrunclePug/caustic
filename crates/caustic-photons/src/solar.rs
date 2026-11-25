//! Scientifically accurate true-ratio Solar System model with `f64` Keplerian orbits
//! and KSP/KSA-style floating-origin camera-relative GPU compilation.
//!
//! All body radii, orbital semi-major axes, eccentricities, periods, Bond albedos,
//! and ring bounds use exact NASA fact-sheet ratios (`1 AU = 149,597,870.7 km`).
//! Supports both `Real (1:1)` exact NASA scale (letting users zoom seamlessly across
//! 7 orders of magnitude from `40 AU` down to `4,000 km`) and an exaggerated `Orrery`
//! cartoon scale where the Sun, all 8 planets, 3 moons, atmospheres, and Saturn's rings
//! fit together on one screen.

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

/// System scale mode: exact 1:1 NASA true-ratio Solar System vs exaggerated Orrery (cartoon) scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarScale {
    /// Exact 1:1 real-world NASA Solar System (`1 AU = 149.6M km`, light-time `8m 19s`).
    Real1To1,
    /// Exaggerated cartoon / orrery scale where the Sun, all 8 planets, moons, rings,
    /// thick Rayleigh atmospheres, and big dramatic shadows fit together on one screen.
    Orrery,
}

impl SolarScale {
    pub fn label(self) -> &'static str {
        match self {
            Self::Real1To1 => "Real (1:1)",
            Self::Orrery => "Orrery",
        }
    }

    pub fn is_orrery(self) -> bool {
        self == Self::Orrery
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

/// Spectral class for primary or companion stars (drives real Planck blackbody photon spectra).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarClass {
    /// Cool M-class Red Dwarf (`3,100 K`): warm ember-red light, suppressed blue Rayleigh scattering.
    RedDwarf,
    /// G2V yellow-white main-sequence star (`5,778 K`, our Sun).
    Sun,
    /// Hot B/O-class Blue Giant (`12,500 K`): intense violet/blue flux, blazing Rayleigh skies.
    BlueGiant,
}

impl StarClass {
    pub const ALL: [Self; 3] = [Self::RedDwarf, Self::Sun, Self::BlueGiant];

    pub fn label(self) -> &'static str {
        match self {
            Self::RedDwarf => "Red Dwarf",
            Self::Sun => "Sun",
            Self::BlueGiant => "Blue Giant",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::RedDwarf => "Red",
            Self::Sun => "Sun",
            Self::BlueGiant => "Blue",
        }
    }

    pub fn temp_k(self) -> f32 {
        match self {
            Self::RedDwarf => 3_100.0,
            Self::Sun => 5_778.0,
            Self::BlueGiant => 12_500.0,
        }
    }

    pub fn radius_mult(self) -> f64 {
        match self {
            Self::RedDwarf => 0.78,
            Self::Sun => 1.0,
            Self::BlueGiant => 1.24,
        }
    }

    pub fn map_color(self) -> [u8; 3] {
        match self {
            Self::RedDwarf => [255, 118, 72],
            Self::Sun => [255, 224, 150],
            Self::BlueGiant => [146, 206, 255],
        }
    }

    pub fn contrasting(self) -> Self {
        match self {
            Self::BlueGiant => Self::RedDwarf,
            Self::Sun | Self::RedDwarf => Self::BlueGiant,
        }
    }
}

/// Optional orbiting & draggable second star for Binary Star dual-colored shadows.
#[derive(Debug, Clone, Copy)]
pub struct CompanionStar {
    pub enabled: bool,
    pub class: StarClass,
    pub semi_major_au: f64,
    pub period_s: f64,
    pub phase_0: f64,
    pub radius_au: f64,
    pub power: f32,
    pub pos_au: [f64; 2],
}

impl Default for CompanionStar {
    fn default() -> Self {
        Self {
            enabled: false,
            class: StarClass::BlueGiant,
            semi_major_au: 3.35,
            period_s: 28.0,
            phase_0: -0.58,
            radius_au: 0.72,
            power: 1.0,
            pos_au: [2.8, -1.84],
        }
    }
}

/// Optional draggable Schwarzschild black hole dropped into the Solar System for real-time
/// isotropic General-Relativistic gravitational lensing of starlight and planet shadows.
#[derive(Debug, Clone, Copy)]
pub struct RogueBlackHole {
    pub enabled: bool,
    pub pos_au: [f64; 2],
    /// Geometric mass `M` in AU (isotropic horizon `r_H = 0.5 M`, photon sphere `r_ph = 1.866 M`,
    /// critical capture impact parameter `b_crit = 3*sqrt(3)*M = 5.196 M`).
    pub mass_au: f64,
    /// User-adjustable mass multiplier (`0.3..=2.5`).
    pub mass_scale: f32,
}

impl Default for RogueBlackHole {
    fn default() -> Self {
        Self {
            enabled: false,
            pos_au: [-3.4, 2.2],
            mass_au: 0.26,
            mass_scale: 1.0,
        }
    }
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
    /// Optional companion star `(gpu_pos, gpu_radius, class)` when enabled.
    pub companion_gpu: Option<([f64; 2], f64, StarClass)>,
    /// Optional rogue black hole `(gpu_pos, gpu_mass_M)` when enabled.
    pub black_hole_gpu: Option<([f64; 2], f64)>,
}

/// Complete double-precision Solar System state.
#[derive(Debug, Clone)]
pub struct SolarSystem {
    pub bodies: Vec<SolarBody>,
    pub scale: SolarScale,
    pub focus: SolarFocus,
    /// Spectral class of the central primary star (`RedDwarf`, `Sun`, `BlueGiant`).
    pub sun_class: StarClass,
    /// Optional second orbiting/draggable companion star for binary dual-color shadows.
    pub companion: CompanionStar,
    /// Optional draggable rogue black hole for live gravitational lensing inside Space.
    pub black_hole: RogueBlackHole,
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
    /// Construct the default exaggerated `Orrery` Solar System in full `System` overview.
    pub fn new() -> Self {
        let bodies = Self::orrery_bodies();
        let n = bodies.len();
        let mut s = Self {
            bodies,
            scale: SolarScale::Orrery,
            focus: SolarFocus::System,
            sun_class: StarClass::Sun,
            companion: CompanionStar::default(),
            black_hole: RogueBlackHole::default(),
            sim_time_s: 0.0,
            time_warp: 1.0,
            view_half_height_au: 8.9,
            pan_offset_au: [0.0, 0.0],
            positions_au: vec![[0.0, 0.0]; n],
        };
        s.update_positions();
        s
    }

    /// Construct the exact 1:1 NASA true-ratio Solar System framed on a Total Lunar Eclipse.
    pub fn new_real() -> Self {
        let bodies = Self::real_bodies();
        let n = bodies.len();
        let mut s = Self {
            bodies,
            scale: SolarScale::Real1To1,
            focus: SolarFocus::Body(BODY_EARTH),
            sun_class: StarClass::Sun,
            companion: CompanionStar::default(),
            black_hole: RogueBlackHole::default(),
            sim_time_s: 0.0,
            time_warp: 3_600.0,
            view_half_height_au: 0.000_25,
            pan_offset_au: [0.0, 0.0],
            positions_au: vec![[0.0, 0.0]; n],
        };
        s.update_positions();
        s.stage_lunar_eclipse();
        s
    }

    /// Exact NASA 1:1 true-ratio Solar System catalog (units in AU and real seconds).
    fn real_bodies() -> Vec<SolarBody> {
        let km = |r_km: f64| r_km / AU_KM;
        vec![
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
        ]
    }

    /// Exaggerated cartoon / orrery Solar System catalog where the Sun, all 8 planets, 3 moons,
    /// thick Rayleigh atmospheres, and Saturn's icy rings all shine and cast shadows on one screen.
    fn orrery_bodies() -> Vec<SolarBody> {
        vec![
            SolarBody {
                name: "Sun",
                radius_au: 1.05,
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
                radius_au: 0.16,
                semi_major_au: 2.05,
                eccentricity: 0.0,
                period_s: 14.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 2.25,
                parent: None,
                albedo: [0.36, 0.34, 0.32],
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
                radius_au: 0.28,
                semi_major_au: 3.10,
                eccentricity: 0.0,
                period_s: 24.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: -0.75,
                parent: None,
                albedo: [0.88, 0.78, 0.56],
                map_color: [232, 200, 138],
                atmo_thickness: 0.34,
                atmo_tau_per_r: 1.35,
                atmo_n_delta: 0.032,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Earth",
                radius_au: 0.38,
                semi_major_au: 4.65,
                eccentricity: 0.0,
                period_s: 38.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 0.28,
                parent: None,
                albedo: [0.28, 0.52, 0.80],
                map_color: [108, 174, 245],
                // Lush visible Rayleigh sky + vivid sunset red refraction into lunar umbra!
                atmo_thickness: 0.48,
                atmo_tau_per_r: 1.30,
                atmo_n_delta: 0.042,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Moon",
                radius_au: 0.13,
                semi_major_au: 0.96,
                eccentricity: 0.0,
                period_s: 6.5,
                arg_periapsis: 0.0,
                mean_anomaly_0: 0.50,
                parent: Some(BODY_EARTH),
                albedo: [0.58, 0.57, 0.54],
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
                radius_au: 0.23,
                semi_major_au: 6.15,
                eccentricity: 0.0,
                period_s: 56.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: -1.95,
                parent: None,
                albedo: [0.80, 0.36, 0.22],
                map_color: [224, 118, 82],
                atmo_thickness: 0.22,
                atmo_tau_per_r: 0.55,
                atmo_n_delta: 0.012,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Jupiter",
                radius_au: 0.60,
                semi_major_au: 8.05,
                eccentricity: 0.0,
                period_s: 90.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 2.82,
                parent: None,
                albedo: [0.78, 0.64, 0.48],
                map_color: [218, 184, 142],
                atmo_thickness: 0.14,
                atmo_tau_per_r: 0.65,
                atmo_n_delta: 0.015,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Io",
                radius_au: 0.10,
                semi_major_au: 1.02,
                eccentricity: 0.0,
                period_s: 4.8,
                arg_periapsis: 0.0,
                mean_anomaly_0: -0.35,
                parent: Some(BODY_JUPITER),
                albedo: [0.86, 0.76, 0.36],
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
                radius_au: 0.09,
                semi_major_au: 1.42,
                eccentricity: 0.0,
                period_s: 8.4,
                arg_periapsis: 0.0,
                mean_anomaly_0: 1.9,
                parent: Some(BODY_JUPITER),
                albedo: [0.78, 0.75, 0.70],
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
                radius_au: 0.50,
                semi_major_au: 10.10,
                eccentricity: 0.0,
                period_s: 135.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: -2.55,
                parent: None,
                albedo: [0.80, 0.72, 0.52],
                map_color: [232, 212, 162],
                atmo_thickness: 0.14,
                atmo_tau_per_r: 0.55,
                atmo_n_delta: 0.014,
                ring_inner_mult: 1.28,
                ring_outer_mult: 2.35,
                ring_angle: -0.62,
            },
            SolarBody {
                name: "Uranus",
                radius_au: 0.34,
                semi_major_au: 11.95,
                eccentricity: 0.0,
                period_s: 185.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: 2.52,
                parent: None,
                albedo: [0.46, 0.76, 0.82],
                map_color: [148, 214, 228],
                atmo_thickness: 0.22,
                atmo_tau_per_r: 0.95,
                atmo_n_delta: 0.015,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
            SolarBody {
                name: "Neptune",
                radius_au: 0.33,
                semi_major_au: 13.75,
                eccentricity: 0.0,
                period_s: 240.0,
                arg_periapsis: 0.0,
                mean_anomaly_0: -2.96,
                parent: None,
                albedo: [0.26, 0.48, 0.88],
                map_color: [98, 142, 238],
                atmo_thickness: 0.24,
                atmo_tau_per_r: 1.15,
                atmo_n_delta: 0.015,
                ring_inner_mult: 0.0,
                ring_outer_mult: 0.0,
                ring_angle: 0.0,
            },
        ]
    }

    /// Switch cleanly between `Real (1:1)` exact NASA scale and exaggerated `Orrery` cartoon scale.
    pub fn set_scale(&mut self, scale: SolarScale) {
        if self.scale == scale {
            return;
        }
        let prev_focus = self.focus;
        let had_comp = self.companion.enabled;
        let had_bh = self.black_hole.enabled;
        self.scale = scale;
        self.sim_time_s = 0.0;
        self.bodies = match scale {
            SolarScale::Real1To1 => Self::real_bodies(),
            SolarScale::Orrery => Self::orrery_bodies(),
        };
        self.update_positions();
        match scale {
            SolarScale::Orrery => self.set_focus(SolarFocus::System),
            SolarScale::Real1To1 => {
                if prev_focus == SolarFocus::System {
                    self.stage_lunar_eclipse();
                } else {
                    self.set_focus(prev_focus);
                }
            }
        }
        if had_comp {
            self.configure_companion_defaults();
        }
        if had_bh {
            self.place_black_hole_in_view();
        }
    }

    /// Configure scale-appropriate orbital parameters for the companion star.
    fn configure_companion_defaults(&mut self) {
        match self.scale {
            SolarScale::Orrery => {
                self.companion.semi_major_au = 3.75;
                self.companion.period_s = 28.0;
                self.companion.phase_0 = 1.12;
                self.companion.radius_au = 0.76;
            }
            SolarScale::Real1To1 => {
                self.companion.semi_major_au = 0.58;
                self.companion.period_s = 0.44 * YEAR_S;
                self.companion.phase_0 = 1.35;
                self.companion.radius_au = SUN_RADIUS_AU * 1.05;
            }
        }
        self.update_positions();
    }

    /// Place the rogue black hole cleanly inside the current camera view at an immediately dramatic scale.
    fn place_black_hole_in_view(&mut self) {
        let center = self.view_center_au();
        let half_h = self.view_half_height_au.max(1e-6);
        match self.scale {
            SolarScale::Orrery if matches!(self.focus, SolarFocus::System | SolarFocus::Inner) => {
                self.black_hole.pos_au = [-4.1, 1.45];
                self.black_hole.mass_au = 0.36;
            }
            _ => {
                self.black_hole.pos_au = [center[0] - half_h * 0.48, center[1] + half_h * 0.16];
                self.black_hole.mass_au = half_h * 0.038;
            }
        }
    }

    /// Toggle the second orbiting star ON/OFF (choosing a contrasting spectral color by default).
    pub fn toggle_companion(&mut self) {
        self.companion.enabled = !self.companion.enabled;
        if self.companion.enabled {
            self.companion.class = self.sun_class.contrasting();
            self.configure_companion_defaults();
        }
    }

    /// Move the companion star interactively to `pos_au`, updating its orbit radius & phase so it
    /// continues orbiting smoothly from the dropped location.
    pub fn set_companion_pos_au(&mut self, pos_au: [f64; 2]) {
        let r = pos_au[0].hypot(pos_au[1]).max(0.05);
        let ang = pos_au[1].atan2(pos_au[0]);
        self.companion.semi_major_au = r;
        let omega_t = if self.companion.period_s.abs() > 1e-6 {
            std::f64::consts::TAU * (self.sim_time_s / self.companion.period_s)
        } else {
            0.0
        };
        self.companion.phase_0 = (ang - omega_t).rem_euclid(std::f64::consts::TAU);
        self.companion.pos_au = pos_au;
    }

    /// Toggle the drop-in Schwarzschild black hole ON/OFF right inside the current viewport.
    pub fn toggle_black_hole(&mut self) {
        self.black_hole.enabled = !self.black_hole.enabled;
        if self.black_hole.enabled {
            self.black_hole.mass_scale = 1.0;
            self.place_black_hole_in_view();
        }
    }

    /// Stage a dramatic **Binary Star** dual-colored shadow showcase (Red/Amber primary + Blue Giant companion).
    pub fn stage_binary_shadows(&mut self) {
        self.sim_time_s = 0.0;
        self.sun_class = StarClass::RedDwarf;
        self.companion.enabled = true;
        self.companion.class = StarClass::BlueGiant;
        self.black_hole.enabled = false;
        match self.scale {
            SolarScale::Orrery => {
                self.set_focus(SolarFocus::System);
                self.companion.semi_major_au = 3.75;
                self.companion.period_s = 28.0;
                self.companion.phase_0 = 1.12;
                self.companion.radius_au = 0.78;
            }
            SolarScale::Real1To1 => {
                self.set_focus(SolarFocus::Body(BODY_EARTH));
                self.view_half_height_au = self.bodies[BODY_EARTH].radius_au * 4.2;
                self.companion.semi_major_au = 0.65;
                self.companion.period_s = 0.52 * YEAR_S;
                self.companion.phase_0 = 1.38;
                self.companion.radius_au = SUN_RADIUS_AU * 1.1;
            }
        }
        self.update_positions();
    }

    /// Stage a dramatic **Rogue Black Hole Gravitational Lensing** showcase right in front of starlight.
    pub fn stage_black_hole_lensing(&mut self) {
        self.sim_time_s = 0.0;
        self.sun_class = StarClass::Sun;
        self.companion.enabled = false;
        self.black_hole.enabled = true;
        self.black_hole.mass_scale = 1.15;
        match self.scale {
            SolarScale::Orrery => {
                self.set_focus(SolarFocus::System);
                self.black_hole.pos_au = [-4.1, 1.45];
                self.black_hole.mass_au = 0.36;
            }
            SolarScale::Real1To1 => {
                self.set_focus(SolarFocus::Body(BODY_EARTH));
                self.view_half_height_au = self.bodies[BODY_EARTH].radius_au * 5.2;
                let e = self.body_pos_au(BODY_EARTH);
                let h = self.view_half_height_au;
                self.black_hole.pos_au = [e[0] - h * 0.55, e[1] + h * 0.14];
                self.black_hole.mass_au = h * 0.036;
            }
        }
        self.update_positions();
    }

    /// Sample points along the companion star's circular orbit (for HUD guide).
    pub fn sample_companion_orbit_au(&self, num_pts: usize) -> Vec<[f64; 2]> {
        if !self.companion.enabled || self.companion.semi_major_au <= 0.0 {
            return Vec::new();
        }
        let r = self.companion.semi_major_au;
        (0..=num_pts)
            .map(|k| {
                let a = std::f64::consts::TAU * (k as f64) / (num_pts as f64);
                [r * a.cos(), r * a.sin()]
            })
            .collect()
    }

    /// Allowed camera `view_half_height_au` range for scroll zooming in the active scale mode.
    pub fn zoom_bounds(&self) -> (f64, f64) {
        match self.scale {
            SolarScale::Real1To1 => (0.000_025, 45.0),
            SolarScale::Orrery => (0.65, 24.0),
        }
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
        if self.companion.enabled && self.companion.semi_major_au > 0.0 {
            let ang = self.companion.phase_0
                + if self.companion.period_s.abs() > 1e-6 {
                    std::f64::consts::TAU * (self.sim_time_s / self.companion.period_s)
                } else {
                    0.0
                };
            self.companion.pos_au = [
                self.companion.semi_major_au * ang.cos(),
                self.companion.semi_major_au * ang.sin(),
            ];
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
        let sun_r = self.bodies[BODY_SUN].radius_au;
        match self.scale {
            SolarScale::Real1To1 => {
                self.view_half_height_au = match focus {
                    SolarFocus::System => 33.5,
                    SolarFocus::Inner => 1.95,
                    SolarFocus::Body(BODY_SUN) => sun_r * 4.5,
                    SolarFocus::Body(BODY_SATURN) => self.bodies[BODY_SATURN].radius_au * 4.5,
                    SolarFocus::Body(BODY_JUPITER) => self.bodies[BODY_JUPITER].radius_au * 5.0,
                    SolarFocus::Body(idx) => {
                        let r = self.bodies.get(idx).map(|b| b.radius_au).unwrap_or(0.0001);
                        (r * 4.2).max(0.000_035)
                    }
                };
                self.time_warp = match focus {
                    SolarFocus::System => 30.0 * 86_400.0,
                    SolarFocus::Inner => 5.0 * 86_400.0,
                    SolarFocus::Body(BODY_EARTH | BODY_MOON) => 3_600.0,
                    SolarFocus::Body(BODY_JUPITER | BODY_IO | BODY_EUROPA) => 1_800.0,
                    _ => 86_400.0,
                };
            }
            SolarScale::Orrery => {
                self.view_half_height_au = match focus {
                    SolarFocus::System => 8.9,
                    SolarFocus::Inner => 6.4,
                    SolarFocus::Body(BODY_SUN) => sun_r * 4.5,
                    SolarFocus::Body(BODY_SATURN) => self.bodies[BODY_SATURN].radius_au * 4.2,
                    SolarFocus::Body(BODY_JUPITER) => self.bodies[BODY_JUPITER].radius_au * 4.5,
                    SolarFocus::Body(idx) => {
                        let r = self.bodies.get(idx).map(|b| b.radius_au).unwrap_or(0.3);
                        (r * 5.2).max(1.25)
                    }
                };
                self.time_warp = match focus {
                    SolarFocus::System | SolarFocus::Inner => 1.0,
                    _ => 0.5,
                };
            }
        }
    }

    /// Stage a **Total Solar Eclipse** (Moon directly between Sun and Earth).
    pub fn stage_solar_eclipse(&mut self) {
        self.sim_time_s = 0.0;
        self.update_positions();
        let earth_pos = self.body_pos_au(BODY_EARTH);
        let sun_to_earth_ang = earth_pos[1].atan2(earth_pos[0]);
        self.bodies[BODY_MOON].arg_periapsis = sun_to_earth_ang + std::f64::consts::PI;
        self.bodies[BODY_MOON].mean_anomaly_0 = 0.0;
        self.update_positions();

        let moon_pos = self.body_pos_au(BODY_MOON);
        self.focus = SolarFocus::Body(BODY_EARTH);
        match self.scale {
            SolarScale::Real1To1 => {
                self.pan_offset_au = [
                    0.40 * (moon_pos[0] - earth_pos[0]),
                    0.40 * (moon_pos[1] - earth_pos[1]),
                ];
                self.view_half_height_au = 0.001_05;
                self.time_warp = 900.0;
            }
            SolarScale::Orrery => {
                self.pan_offset_au = [
                    0.35 * (moon_pos[0] - earth_pos[0]),
                    0.35 * (moon_pos[1] - earth_pos[1]),
                ];
                self.view_half_height_au = 2.0;
                self.time_warp = 0.25;
            }
        }
    }

    /// Stage a **Total Lunar Eclipse** (Moon directly behind Earth inside Earth's copper-red umbra).
    pub fn stage_lunar_eclipse(&mut self) {
        self.sim_time_s = 0.0;
        self.update_positions();
        let earth_pos = self.body_pos_au(BODY_EARTH);
        let sun_to_earth_ang = earth_pos[1].atan2(earth_pos[0]);
        self.bodies[BODY_MOON].arg_periapsis = sun_to_earth_ang;
        self.bodies[BODY_MOON].mean_anomaly_0 = 0.0;
        self.update_positions();

        let moon_pos = self.body_pos_au(BODY_MOON);
        self.focus = SolarFocus::Body(BODY_EARTH);
        match self.scale {
            SolarScale::Real1To1 => {
                self.pan_offset_au = [
                    0.70 * (moon_pos[0] - earth_pos[0]),
                    0.70 * (moon_pos[1] - earth_pos[1]),
                ];
                self.view_half_height_au = 0.001_18;
                self.time_warp = 900.0;
            }
            SolarScale::Orrery => {
                self.pan_offset_au = [
                    0.45 * (moon_pos[0] - earth_pos[0]),
                    0.45 * (moon_pos[1] - earth_pos[1]),
                ];
                self.view_half_height_au = 2.1;
                self.time_warp = 0.25;
            }
        }
    }

    /// Stage **Saturn & its icy ring system**.
    pub fn stage_saturn_rings(&mut self) {
        self.set_focus(SolarFocus::Body(BODY_SATURN));
        self.view_half_height_au = self.bodies[BODY_SATURN].radius_au * 3.8;
    }

    /// Stage **Io transiting directly sunward of Jupiter**, casting a moon shadow onto Jupiter.
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
        match self.scale {
            SolarScale::Real1To1 => {
                self.view_half_height_au = 0.001_35;
                self.time_warp = 600.0;
            }
            SolarScale::Orrery => {
                self.view_half_height_au = 2.5;
                self.time_warp = 0.25;
            }
        }
    }

    /// Sample points along body `idx`'s exact Keplerian ellipse (for orbit guides).
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
    /// and frustum importance-sampled `EMITTER_SOLAR_FAR` (when zoomed into a distant planet/moon,
    /// preserving exact solar angular diameter and sub-pixel FP32 precision).
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

        let sun_r = self.bodies[BODY_SUN].radius_au * self.sun_class.radius_mult();
        let sun_dx = center[0] - self.positions_au[BODY_SUN][0];
        let sun_dy = center[1] - self.positions_au[BODY_SUN][1];
        let sun_dist_au = (sun_dx * sun_dx + sun_dy * sun_dy).sqrt();
        let sun_dist_gpu = sun_dist_au * inv_s;
        let sun_rad_gpu = sun_r * inv_s;
        body_gpu_rad[BODY_SUN] = sun_rad_gpu;

        // Exact angular radius of the Sun as seen from camera center.
        let alpha_sun = if sun_dist_au > sun_r {
            (sun_r / sun_dist_au).clamp(0.0, 1.0).asin()
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
            // (bodies remain clickable via HUD markers and appear seamlessly as you zoom in).
            if gr < 1e-4 {
                continue;
            }

            if in_upstream_corridor && proj_along_sun + gr * 2.0 > d_spawn_gpu {
                d_spawn_gpu = (proj_along_sun + gr * 2.5).min(sun_dist_gpu - sun_rad_gpu - 0.1);
            }

            // Solid planet body (`MAT_DIFFUSE`) tagged with its 2K equirectangular texture layer (`i + 1`)
            // and live axial rotation angle (`material.w`).
            let spin_angle = body_spin_angle(i, self.sim_time_s, self.scale);
            circles.push(OptCircle {
                geom: [gp[0] as f32, gp[1] as f32, gr as f32, 0.0],
                clip_arc: [0.0, 0.0, -2.0, 0.0],
                material: [b.albedo[0], b.albedo[1], b.albedo[2], spin_angle],
                meta: [MAT_DIFFUSE, (i as u32) + 1, 0, 0],
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

        let mut emitters = if far_sun_mode {
            // Exact projected half-span of the rectangular viewport perpendicular to sunlight,
            // concentrating ~100% of GPU photons through the visible frame!
            let exact_span = half_w_gpu * sdir_y.abs() + sdir_x.abs();
            let span_half = exact_span * 1.06 + alpha_sun.tan() * screen_diag_gpu;
            vec![GpuEmitter {
                geom: [0.0, 0.0, sun_to_cam_ang as f32, span_half as f32],
                spec: [self.sun_class.temp_k(), 1.0, 1.0, alpha_sun as f32],
                info: [
                    EMITTER_SOLAR_FAR,
                    SPEC_BLACKBODY,
                    (d_spawn_gpu as f32).to_bits(),
                    0,
                ],
            }]
        } else {
            let sun_spin = body_spin_angle(BODY_SUN, self.sim_time_s, self.scale);
            vec![GpuEmitter {
                geom: [
                    body_gpu_pos[BODY_SUN][0] as f32,
                    body_gpu_pos[BODY_SUN][1] as f32,
                    0.0,
                    sun_rad_gpu as f32,
                ],
                spec: [self.sun_class.temp_k(), 1.0, 1.0, sun_spin],
                info: [EMITTER_DISK, SPEC_BLACKBODY, 1, 0],
            }]
        };

        let companion_gpu = if self.companion.enabled {
            let c_pos = self.companion.pos_au;
            let c_rad_au = self.companion.radius_au * self.companion.class.radius_mult();
            let c_gp = [
                (c_pos[0] - center[0]) * inv_s,
                (c_pos[1] - center[1]) * inv_s,
            ];
            let c_gr = c_rad_au * inv_s;
            let c_dx = center[0] - c_pos[0];
            let c_dy = center[1] - c_pos[1];
            let c_dist_au = (c_dx * c_dx + c_dy * c_dy).sqrt();
            let c_dist_gpu = c_dist_au * inv_s;
            let c_far = c_dist_gpu > screen_diag_gpu * 3.2 + c_gr;
            if c_far {
                let c_ang = c_dy.atan2(c_dx);
                let cx = c_ang.cos();
                let cy = c_ang.sin();
                let c_alpha = if c_dist_au > c_rad_au {
                    (c_rad_au / c_dist_au).clamp(0.0, 1.0).asin()
                } else {
                    std::f64::consts::FRAC_PI_2
                };
                let c_span =
                    (half_w_gpu * cy.abs() + cx.abs()) * 1.06 + c_alpha.tan() * screen_diag_gpu;
                let c_spawn = (half_w_gpu * cx.abs() + cy.abs()) * 1.3;
                emitters.push(GpuEmitter {
                    geom: [0.0, 0.0, c_ang as f32, c_span as f32],
                    spec: [
                        self.companion.class.temp_k(),
                        self.companion.power,
                        1.0,
                        c_alpha as f32,
                    ],
                    info: [
                        EMITTER_SOLAR_FAR,
                        SPEC_BLACKBODY,
                        (c_spawn as f32).to_bits(),
                        0,
                    ],
                });
            } else {
                let c_spin = body_spin_angle(BODY_SUN, self.sim_time_s * 1.3 + 21.0, self.scale);
                emitters.push(GpuEmitter {
                    geom: [c_gp[0] as f32, c_gp[1] as f32, 0.0, c_gr as f32],
                    spec: [
                        self.companion.class.temp_k(),
                        self.companion.power,
                        1.0,
                        c_spin,
                    ],
                    info: [EMITTER_DISK, SPEC_BLACKBODY, 1, 0],
                });
            }
            // Normalize cumulative probability weights across primary + companion stars.
            let total_w: f32 = emitters.iter().map(|e| e.spec[1].max(0.05)).sum::<f32>();
            let mut acc = 0.0_f32;
            let n_em = emitters.len();
            for (idx, em) in emitters.iter_mut().enumerate() {
                acc += em.spec[1].max(0.05);
                em.spec[2] = if idx + 1 == n_em { 1.0 } else { acc / total_w };
            }
            Some((c_gp, c_gr, self.companion.class))
        } else {
            None
        };

        let black_hole_gpu = if self.black_hole.enabled {
            let bp = self.black_hole.pos_au;
            let b_gp = [(bp[0] - center[0]) * inv_s, (bp[1] - center[1]) * inv_s];
            let b_m_gpu = (self.black_hole.mass_au * f64::from(self.black_hole.mass_scale)) * inv_s;
            Some((b_gp, b_m_gpu))
        } else {
            None
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
            light_time_s: sun_dist_au * LIGHT_TIME_AU_S,
            body_gpu_pos,
            body_gpu_rad,
            companion_gpu,
            black_hole_gpu,
        }
    }
}

/// Compute visually expressive axial spin longitude (radians) for body `idx`.
///
/// Initial offsets orient each body's signature features (Earth's continents, Jupiter's Great Red Spot,
/// Moon's near-side Maria, Io's Pele plume, Neptune's Great Dark Spot) toward the sunlit face.
fn body_spin_angle(idx: usize, sim_time_s: f64, scale: SolarScale) -> f32 {
    let (offset, real_period_s, orrery_period_s) = match idx {
        BODY_SUN => (0.2, 25.38 * 86_400.0, 70.0),
        BODY_MERCURY => (0.4, 58.65 * 86_400.0, 48.0),
        BODY_VENUS => (0.6, -243.02 * 86_400.0, -52.0),
        BODY_EARTH => (0.45, 86_164.1, 34.0),
        BODY_MOON => (0.45, 27.321_661 * 86_400.0, 42.0),
        BODY_MARS => (0.25, 88_642.7, 36.0),
        BODY_JUPITER => (-0.52, 35_730.0, 26.0),
        BODY_IO => (-1.15, 1.769_138 * 86_400.0, 32.0),
        BODY_EUROPA => (0.3, 3.551_181 * 86_400.0, 38.0),
        BODY_SATURN => (0.2, 38_362.0, 28.0),
        BODY_URANUS => (0.1, -62_064.0, -40.0),
        BODY_NEPTUNE => (0.25, 57_996.0, 36.0),
        _ => (0.0, 86_400.0, 40.0),
    };
    let period: f64 = match scale {
        SolarScale::Real1To1 => real_period_s,
        SolarScale::Orrery => orrery_period_s,
    };
    let turns = if period.abs() > 1e-6 {
        (sim_time_s / period).rem_euclid(1.0)
    } else {
        0.0
    };
    (offset + std::f64::consts::TAU * turns).rem_euclid(std::f64::consts::TAU) as f32
}

/// Format an astronomical distance (`au`) cleanly for Real 1:1 HUD telemetry.
pub fn format_distance_au(au: f64) -> String {
    let km = au * AU_KM;
    if au >= 0.25 {
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

/// Format orbital time warp rate cleanly for either `Real (1:1)` or `Orrery` scale.
pub fn format_warp(warp: f64, scale: SolarScale) -> String {
    if scale.is_orrery() {
        if warp < 1.0 {
            format!("{warp:.2}x")
        } else {
            format!("{warp:.1}x")
        }
    } else if warp < 60.0 {
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
        let default_sys = SolarSystem::new();
        assert_eq!(default_sys.scale, SolarScale::Orrery);
        assert_eq!(default_sys.focus, SolarFocus::System);

        let sys = SolarSystem::new_real();

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

        let mut sys = SolarSystem::new_real();
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

        // Verify Real (1:1) Mercury genuinely places the Sun at a Keplerian focus (off-center by a*e).
        let merc = sys.bodies[BODY_MERCURY];
        let merc_pts = sys.sample_orbit_ellipse_au(BODY_MERCURY, 128);
        let (min_x, max_x) = merc_pts
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p[0]), hi.max(p[0]))
            });
        let (min_y, max_y) = merc_pts
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p[1]), hi.max(p[1]))
            });
        let merc_mid_dist = (0.5 * (min_x + max_x)).hypot(0.5 * (min_y + max_y));
        assert!(
            (merc_mid_dist - merc.semi_major_au * merc.eccentricity).abs() < 2e-3,
            "Real Mercury orbit center should be offset by a*e ({:.4}), got {:.4}",
            merc.semi_major_au * merc.eccentricity,
            merc_mid_dist
        );

        // Verify exaggerated Orrery mode renders all planets, moons, atmospheres, and Saturn's rings in one view,
        // with clean circular planet orbits centered dead-on the Sun!
        sys.set_scale(SolarScale::Orrery);
        let orr = sys.compile_to_gpu(16.0 / 9.0);
        assert!(
            !orr.far_sun_mode,
            "Orrery system view must place Sun directly on screen"
        );
        assert_eq!(
            orr.body_gpu_pos[BODY_SUN],
            [0.0, 0.0],
            "Orrery Sun must sit dead-center at origin"
        );
        for (i, b) in sys.bodies.iter().enumerate().skip(1) {
            if b.parent.is_some() {
                continue;
            }
            let pts = sys.sample_orbit_ellipse_au(i, 64);
            let (lo_x, hi_x) = pts
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p[0]), hi.max(p[0]))
                });
            let (lo_y, hi_y) = pts
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p[1]), hi.max(p[1]))
                });
            assert!(
                (lo_x + hi_x).abs() < 1e-12 && (lo_y + hi_y).abs() < 1e-12,
                "Orrery {} orbit ring should be centered on Sun, got mid=({:.3e}, {:.3e})",
                b.name,
                0.5 * (lo_x + hi_x),
                0.5 * (lo_y + hi_y)
            );
        }
        assert!(
            orr.circles.len() >= 15,
            "expected all 8 planets + 3 moons + atmospheres in Orrery view, got {}",
            orr.circles.len()
        );
        assert_eq!(
            orr.segments.len(),
            2,
            "expected Saturn's 2 ring segments in Orrery view"
        );
    }

    #[test]
    fn editable_space_binary_stars_and_isotropic_schwarzschild_lensing() {
        // 1. Verify Binary Star staging compiles 2 emitters with valid CDF & distinct star temps.
        let mut sys = SolarSystem::new();
        sys.stage_binary_shadows();
        let bin = sys.compile_to_gpu(16.0 / 9.0);
        assert_eq!(bin.emitters.len(), 2, "binary mode must emit 2 stars");
        assert!((bin.emitters[0].spec[0] - StarClass::RedDwarf.temp_k()).abs() < 1.0);
        assert!((bin.emitters[1].spec[0] - StarClass::BlueGiant.temp_k()).abs() < 1.0);
        assert!(bin.emitters[0].spec[2] > 0.1 && bin.emitters[0].spec[2] < 0.9);
        assert!((bin.emitters[1].spec[2] - 1.0).abs() < 1e-6);
        assert!(bin.companion_gpu.is_some());

        // 2. Verify Rogue Black Hole staging compiles valid GPU mass & position in both Orrery and Real.
        sys.stage_black_hole_lensing();
        let bh_orr = sys.compile_to_gpu(16.0 / 9.0);
        let (_, m_orr) = bh_orr.black_hole_gpu.expect("Orrery black hole active");
        assert!(m_orr > 0.01 && m_orr < 0.2);

        sys.set_scale(SolarScale::Real1To1);
        sys.stage_black_hole_lensing();
        let bh_real = sys.compile_to_gpu(16.0 / 9.0);
        let (_, m_real) = bh_real.black_hole_gpu.expect("Real 1:1 black hole active");
        assert!(m_real > 0.01 && m_real < 0.2);

        // 3. Verify our exact isotropic Schwarzschild index n(r) = (1+u)^3 / (1-u) reproduces
        // Einstein's exact general-relativistic light deflection Delta_phi = 4M / b + b_crit = 3*sqrt(3)*M!
        let bh_n = |r: f64, m: f64| -> f64 {
            let u = (0.5 * m / r).min(0.92);
            (1.0 + u).powi(3) / (1.0 - u)
        };
        let bh_acc = |p: [f64; 2], m: f64| -> [f64; 2] {
            let r2 = (p[0] * p[0] + p[1] * p[1]).max((0.54 * m).powi(2));
            let r = r2.sqrt();
            let u = (0.5 * m / r).min(0.92);
            let coeff = -2.0 * u * (1.0 + u).powi(5) * (2.0 - u) / ((1.0 - u).powi(3) * r2);
            [p[0] * coeff, p[1] * coeff]
        };
        let trace_deflection = |b: f64| -> Option<f64> {
            let m = 1.0_f64;
            let mut pos = [-400.0, b];
            let mut dir = [1.0, 0.0];
            for _ in 0..8000 {
                let r = pos[0].hypot(pos[1]);
                if r <= 0.62 * m {
                    return None;
                }
                let n0 = bh_n(r, m);
                let p0 = [dir[0] * n0, dir[1] * n0];
                let h = (0.025 * r / n0).clamp(0.004, 1.5);
                let k1x = p0;
                let k1p = bh_acc(pos, m);
                let k2x = [p0[0] + 0.5 * h * k1p[0], p0[1] + 0.5 * h * k1p[1]];
                let k2p = bh_acc([pos[0] + 0.5 * h * k1x[0], pos[1] + 0.5 * h * k1x[1]], m);
                let k3x = [p0[0] + 0.5 * h * k2p[0], p0[1] + 0.5 * h * k2p[1]];
                let k3p = bh_acc([pos[0] + 0.5 * h * k2x[0], pos[1] + 0.5 * h * k2x[1]], m);
                let k4x = [p0[0] + h * k3p[0], p0[1] + h * k3p[1]];
                let k4p = bh_acc([pos[0] + h * k3x[0], pos[1] + h * k3x[1]], m);
                pos[0] += (h / 6.0) * (k1x[0] + 2.0 * k2x[0] + 2.0 * k3x[0] + k4x[0]);
                pos[1] += (h / 6.0) * (k1x[1] + 2.0 * k2x[1] + 2.0 * k3x[1] + k4x[1]);
                let np = [
                    p0[0] + (h / 6.0) * (k1p[0] + 2.0 * k2p[0] + 2.0 * k3p[0] + k4p[0]),
                    p0[1] + (h / 6.0) * (k1p[1] + 2.0 * k2p[1] + 2.0 * k3p[1] + k4p[1]),
                ];
                let nl = np[0].hypot(np[1]);
                dir = [np[0] / nl, np[1] / nl];
                if pos[0] > 400.0 {
                    return Some((-dir[1]).atan2(dir[0]));
                }
            }
            None
        };
        // Exact capture threshold b_crit = 3*sqrt(3) = 5.196 M
        assert!(trace_deflection(4.8).is_none(), "b=4.8M must be captured");
        let d40 = trace_deflection(40.0).expect("b=40M must escape");
        let einstein_40 = 4.0 / 40.0 + 15.0 * std::f64::consts::PI / (4.0 * 40.0 * 40.0);
        let rel_err = (d40 - einstein_40).abs() / einstein_40;
        assert!(
            rel_err < 0.015,
            "isotropic Schwarzschild deflection at b=40M: got {d40:.6}, expected {einstein_40:.6} (rel_err={rel_err:.4})"
        );
    }
}
