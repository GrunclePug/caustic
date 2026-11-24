//! Built-in optical, GRIN, solar-system, sandbox, and relativistic photon scenes plus
//! multi-emitter and editable optical bench primitives.

use bytemuck::{Pod, Zeroable};

/// Physics backend selector shared with `photon_step.wgsl` and `photon_resolve.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PhysicsKind {
    Optics = 0,
    Kerr = 1,
    GrinLuneburg = 2,
    GrinBlackHole = 3,
    GrinMirage = 4,
}

/// High-level simulation category for UI grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneCategory {
    Optics,
    Grin,
    Relativity,
}

/// Built-in forward photon simulation presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonSceneId {
    /// Dispersive glass prism splitting a white beam into a rainbow + concave mirror.
    Prism,
    /// Circular mirror cavity producing the classic nephroid caustic envelope.
    Cardioid,
    /// Thick dispersive biconvex glass lens showing spherical & chromatic aberration caustics.
    Lens,
    /// Orrery-scale solar system: 5778K Sun, orbiting planets, shadows, eclipses, and Rayleigh atmospheres.
    SolarSystem,
    /// Clean blank-slate photon playground for combining multiple emitters, optics, walls, and planets.
    Sandbox,
    /// Analytical Luneburg GRIN sphere focusing parallel light with zero spherical aberration.
    GrinLuneburg,
    /// Dispersive inverse-square optical black hole capturing violet while red escapes.
    GrinBlackHole,
    /// Heated-road thermal atmospheric gradient bending shallow rays into a mirage caustic.
    GrinMirage,
    /// Collimated relativistic photon beam grazing a rotating Kerr black hole (Ingoing KS).
    KerrBeam,
    /// Isotropic point beacon near a spinning Kerr black hole producing relativistic caustics.
    KerrPulsar,
}

impl PhotonSceneId {
    /// All built-in forward photon presets in UI display order.
    pub const ALL: [Self; 10] = [
        Self::Prism,
        Self::Cardioid,
        Self::Lens,
        Self::SolarSystem,
        Self::Sandbox,
        Self::GrinLuneburg,
        Self::GrinBlackHole,
        Self::GrinMirage,
        Self::KerrBeam,
        Self::KerrPulsar,
    ];

    /// CLI identifier slug.
    pub fn name(self) -> &'static str {
        match self {
            Self::Prism => "prism",
            Self::Cardioid => "cardioid",
            Self::Lens => "lens",
            Self::SolarSystem => "solar-system",
            Self::Sandbox => "sandbox",
            Self::GrinLuneburg => "luneburg",
            Self::GrinBlackHole => "optical-bh",
            Self::GrinMirage => "mirage",
            Self::KerrBeam => "kerr-beam",
            Self::KerrPulsar => "kerr-pulsar",
        }
    }

    /// Human-readable full name for UI headers.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Prism => "Dispersive Prism",
            Self::Cardioid => "Nephroid Cavity",
            Self::Lens => "Chromatic Lens",
            Self::SolarSystem => "Solar System",
            Self::Sandbox => "Photon Sandbox",
            Self::GrinLuneburg => "Luneburg Lens",
            Self::GrinBlackHole => "Optical Black Hole",
            Self::GrinMirage => "Thermal Mirage",
            Self::KerrBeam => "Kerr Lensing Beam",
            Self::KerrPulsar => "Kerr Pulsar",
        }
    }

    /// Compact pill label for the top switcher bar.
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Prism => "Prism",
            Self::Cardioid => "Cavity",
            Self::Lens => "Lens",
            Self::SolarSystem => "Solar",
            Self::Sandbox => "Sandbox",
            Self::GrinLuneburg => "Luneburg",
            Self::GrinBlackHole => "Optical BH",
            Self::GrinMirage => "Mirage",
            Self::KerrBeam => "Kerr Beam",
            Self::KerrPulsar => "Kerr Pulsar",
        }
    }

    pub fn category(self) -> SceneCategory {
        match self {
            Self::Prism | Self::Cardioid | Self::Lens | Self::SolarSystem | Self::Sandbox => {
                SceneCategory::Optics
            }
            Self::GrinLuneburg | Self::GrinBlackHole | Self::GrinMirage => SceneCategory::Grin,
            Self::KerrBeam | Self::KerrPulsar => SceneCategory::Relativity,
        }
    }

    pub fn physics_kind(self) -> PhysicsKind {
        match self {
            Self::Prism | Self::Cardioid | Self::Lens | Self::SolarSystem | Self::Sandbox => {
                PhysicsKind::Optics
            }
            Self::GrinLuneburg => PhysicsKind::GrinLuneburg,
            Self::GrinBlackHole => PhysicsKind::GrinBlackHole,
            Self::GrinMirage => PhysicsKind::GrinMirage,
            Self::KerrBeam | Self::KerrPulsar => PhysicsKind::Kerr,
        }
    }

    pub fn is_kerr(self) -> bool {
        self.physics_kind() == PhysicsKind::Kerr
    }

    pub fn is_grin(self) -> bool {
        self.category() == SceneCategory::Grin
    }

    pub fn is_optics(self) -> bool {
        self.category() == SceneCategory::Optics
    }

    pub fn is_solar_system(self) -> bool {
        self == Self::SolarSystem
    }
}

/// Accumulation vs real-time pulse flight mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PhotonMode {
    /// Continuous long-exposure HDR accumulation (instant full-path tracing + convergence).
    Accumulate = 0,
    /// Synchronized flying photon pulses with smooth trail persistence (watch light move at c).
    Pulse = 1,
}

/// Emitter geometry types shared with `photon_step.wgsl`.
pub const EMITTER_BEAM: u32 = 0;
pub const EMITTER_POINT: u32 = 1;
pub const EMITTER_SPOT: u32 = 2;
pub const EMITTER_DISK: u32 = 3;

/// Emitter spectrum modes shared with `photon_step.wgsl`.
pub const SPEC_WHITE_FLAT: u32 = 0;
pub const SPEC_BLACKBODY: u32 = 1;
pub const SPEC_LASER: u32 = 2;

/// Maximum simultaneous emitters uploaded to the GPU.
pub const MAX_EMITTERS: usize = 16;

/// Emitter geometric shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterKind {
    /// Collimated parallel beam of half-width `spread`.
    Beam,
    /// 360-degree omnidirectional point light.
    Point,
    /// Directional cone / flashlight of half-angle `spread` radians.
    Spot,
    /// Physical circular emitting body (e.g. a star / Sun) of radius `spread` casting true umbra/penumbra.
    Disk,
}

impl EmitterKind {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Beam => EMITTER_BEAM,
            Self::Point => EMITTER_POINT,
            Self::Spot => EMITTER_SPOT,
            Self::Disk => EMITTER_DISK,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Beam => "Beam",
            Self::Point => "Point",
            Self::Spot => "Spotlight",
            Self::Disk => "Star Disk",
        }
    }
}

/// Physical emission spectrum of a photon emitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitterSpectrum {
    /// Equal-energy visible white (`380..780 nm`, ideal for prism rainbow dispersion).
    WhiteFlat,
    /// True Planck blackbody thermal spectrum at `temp_k` Kelvin (`5778 K` = Sun).
    Blackbody(f32),
    /// Monochromatic laser at `wavelength_nm` (`380..780 nm`).
    Laser(f32),
}

impl EmitterSpectrum {
    pub fn label(self) -> &'static str {
        match self {
            Self::WhiteFlat => "White",
            Self::Blackbody(_) => "Blackbody",
            Self::Laser(_) => "Laser",
        }
    }
}

/// Interactive photon emitter on the playground canvas.
#[derive(Debug, Clone, Copy)]
pub struct PhotonEmitter {
    pub id: u32,
    pub kind: EmitterKind,
    pub spectrum: EmitterSpectrum,
    pub pos: [f32; 2],
    pub angle: f32,
    /// Beam half-width (`Beam`), cone half-angle in rad (`Spot`), or disk radius (`Disk`).
    pub spread: f32,
    /// Relative luminous power multiplier.
    pub power: f32,
}

/// GPU-packed emitter record (`48 bytes`, matches WGSL `struct GpuEmitter`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEmitter {
    /// `xy = pos`, `z = angle`, `w = spread`
    pub geom: [f32; 4],
    /// `x = spec_param` (temp_k or wavelength_nm), `y = power`, `z = cdf_hi` (cumulative weight in [0,1]), `w = reserved`
    pub spec: [f32; 4],
    /// `x = emitter_kind`, `y = spectrum_kind`, `zw = reserved`
    pub info: [u32; 4],
}

/// Real dispersive optical glass / medium presets with Cauchy `(A, B)` parameters (`um^2`).
///
/// `Water`, `CrownBk7`, `FlintSf11`, and `Diamond` match published refractive dispersion;
/// `StudioFlint` uses exaggerated dispersion for wide screen-filling rainbows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlassMaterial {
    Water,
    CrownBk7,
    FlintSf11,
    Diamond,
    StudioFlint,
}

impl GlassMaterial {
    pub const ALL: [Self; 5] = [
        Self::Water,
        Self::CrownBk7,
        Self::FlintSf11,
        Self::Diamond,
        Self::StudioFlint,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Water => "Water",
            Self::CrownBk7 => "BK7 Glass",
            Self::FlintSf11 => "SF11 Flint",
            Self::Diamond => "Diamond",
            Self::StudioFlint => "High-Disp",
        }
    }

    /// Return `(cauchy_a, cauchy_b * dispersion_scale)`.
    pub fn cauchy(self, dispersion_scale: f32) -> (f32, f32) {
        let (a, b) = match self {
            Self::Water => (1.324, 0.0031),
            Self::CrownBk7 => (1.5046, 0.0042),
            Self::FlintSf11 => (1.7376, 0.0132),
            Self::Diamond => (2.378, 0.0145),
            Self::StudioFlint => (1.460, 0.0420),
        };
        (a, b * dispersion_scale.clamp(0.0, 4.0))
    }
}

/// Editable optical & astronomical element kinds for the interactive playground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementKind {
    /// Equilateral refractive prism with circumradius `radius`.
    Prism { radius: f32 },
    /// Thick symmetric biconvex glass lens (intersection of two circles of `radius` separated by `2 * half_offset`).
    Lens { radius: f32, half_offset: f32 },
    /// Flat specular mirror of total length `2 * half_len`.
    FlatMirror { half_len: f32 },
    /// Circular concave/convex mirror arc of `radius` covering `[-arc_half_angle, +arc_half_angle]` around `angle`.
    CurvedMirror { radius: f32, arc_half_angle: f32 },
    /// 50/50 partially silvered beam splitter of total length `2 * half_len`.
    BeamSplitter { half_len: f32 },
    /// Matte diffuse colored surface (Lambertian scattering + spectral albedo).
    MatteWall { half_len: f32, color: [f32; 3] },
    /// Solid planet / moon with realistic diffuse albedo, optional Rayleigh atmosphere, optional ring, and optional orbit.
    Planet {
        radius: f32,
        albedo: [f32; 3],
        /// Atmosphere thickness in multiples of `radius` (`0.0` = airless rock like Mercury/Moon).
        atmo_thickness: f32,
        /// Rayleigh scattering optical density at 550 nm (scales as `(550 / lambda)^4`).
        atmo_density: f32,
        /// Optional translucent planetary ring outer radius (`0.0` = no ring).
        ring_outer: f32,
        /// Orbital radius around origin (or parent planet when `parent_idx` is `Some`); `0.0` = stationary.
        orbit_radius: f32,
        /// Angular orbital speed in rad/s.
        orbit_speed: f32,
        /// Current orbital angle in radians.
        orbit_phase: f32,
        /// Optional parent element index (for moons orbiting a planet).
        parent_idx: Option<usize>,
    },
    /// Matte black optical stop / slit edge of total length `2 * half_len`.
    Absorber { half_len: f32 },
}

impl ElementKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Prism { .. } => "Prism",
            Self::Lens { .. } => "Biconvex Lens",
            Self::FlatMirror { .. } => "Flat Mirror",
            Self::CurvedMirror { .. } => "Curved Mirror",
            Self::BeamSplitter { .. } => "Beam Splitter",
            Self::MatteWall { .. } => "Matte Wall",
            Self::Planet { .. } => "Planet",
            Self::Absorber { .. } => "Beam Stop",
        }
    }

    pub fn is_glass(self) -> bool {
        matches!(self, Self::Prism { .. } | Self::Lens { .. })
    }

    pub fn size(self) -> f32 {
        match self {
            Self::Prism { radius }
            | Self::Lens { radius, .. }
            | Self::CurvedMirror { radius, .. }
            | Self::Planet { radius, .. } => radius,
            Self::FlatMirror { half_len }
            | Self::BeamSplitter { half_len }
            | Self::MatteWall { half_len, .. }
            | Self::Absorber { half_len } => half_len,
        }
    }

    pub fn set_size(&mut self, value: f32) {
        let v = value.clamp(0.15, 8.0);
        match self {
            Self::Prism { radius }
            | Self::CurvedMirror { radius, .. }
            | Self::Planet { radius, .. } => *radius = v,
            Self::Lens {
                radius,
                half_offset,
            } => {
                let ratio = (*half_offset / (*radius).max(0.1)).clamp(0.35, 0.90);
                *radius = v;
                *half_offset = v * ratio;
            }
            Self::FlatMirror { half_len }
            | Self::BeamSplitter { half_len }
            | Self::MatteWall { half_len, .. }
            | Self::Absorber { half_len } => *half_len = v,
        }
    }
}

/// Editable optical or astronomical element on the playground canvas.
#[derive(Debug, Clone, Copy)]
pub struct OpticalElement {
    pub id: u32,
    pub name: &'static str,
    pub kind: ElementKind,
    pub pos: [f32; 2],
    pub angle: f32,
    pub material: GlassMaterial,
}

impl OpticalElement {
    /// World-space grab handle position (midpoint of arc for curved mirrors, `pos` otherwise).
    pub fn handle_pos(&self) -> [f32; 2] {
        match self.kind {
            ElementKind::CurvedMirror { radius, .. } => [
                self.pos[0] + radius * self.angle.cos(),
                self.pos[1] + radius * self.angle.sin(),
            ],
            _ => self.pos,
        }
    }

    /// Move the element so its grab handle tracks `new_handle`.
    pub fn move_handle_to(&mut self, new_handle: [f32; 2]) {
        match self.kind {
            ElementKind::CurvedMirror { radius, .. } => {
                self.pos = [
                    new_handle[0] - radius * self.angle.cos(),
                    new_handle[1] - radius * self.angle.sin(),
                ];
            }
            _ => self.pos = new_handle,
        }
    }
}

/// Material kinds shared with `photon_step.wgsl` and `photon_resolve.wgsl`.
pub const MAT_MIRROR: u32 = 1;
pub const MAT_GLASS: u32 = 2;
pub const MAT_ABSORBER: u32 = 3;
pub const MAT_DIFFUSE: u32 = 4;
pub const MAT_ATMOSPHERE: u32 = 5;
pub const MAT_SPLITTER: u32 = 6;

/// 2D line segment optical element (`48 bytes`, matches WGSL `struct OptSegment`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OptSegment {
    /// `xy = p0`, `zw = p1`.
    pub endpoints: [f32; 4],
    /// Glass: `x = cauchy_a, y = cauchy_b`. Diffuse/Splitter: `xyz = spectral albedo RGB, w = reflectance/transmittance`.
    pub material: [f32; 4],
    /// `x = mat_kind`, `yzw = reserved`.
    pub meta: [u32; 4],
}

/// 2D circular arc / lens / planet / atmosphere surface (`64 bytes`, matches WGSL `struct OptCircle`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OptCircle {
    /// `xy = center`, `z = radius`, `w = clip_radius` (if > 0 for lens).
    pub geom: [f32; 4],
    /// `xy = clip_center`, `z = arc_cos_half` (if > -1.5), `w = arc_angle_center`.
    pub clip_arc: [f32; 4],
    /// Glass: `x = cauchy_a, y = cauchy_b`. Diffuse: `xyz = albedo RGB`. Atmosphere: `x = rayleigh_density, y = limb_refract, z = planet_radius`.
    pub material: [f32; 4],
    /// `x = mat_kind`, `y = invert_normal` (1 for inward-facing cavity mirror), `zw = reserved`.
    pub meta: [u32; 4],
}

/// Complete scene state driving both GPU shaders and interactive controls.
#[derive(Debug, Clone)]
pub struct PhotonScene {
    pub id: PhotonSceneId,
    /// World half-height in scene units (`M` for Kerr scenes, world units for optics/GRIN/solar).
    pub view_half_height: f32,
    /// World center `(x, y)`.
    pub view_center: [f32; 2],
    /// Simultaneous editable photon emitters (at least 1).
    pub emitters: Vec<PhotonEmitter>,
    next_emitter_id: u32,
    /// Kerr spin parameter `a in [-0.998, +0.998]` (used when `id.is_kerr()`).
    pub spin: f32,
    /// Dispersion scale multiplier (`1.0` default; adjustable live in optical & GRIN scenes).
    pub dispersion_scale: f32,
    /// GRIN field strength multiplier (`1.0` default).
    pub grin_strength: f32,
    /// Orbital animation speed multiplier (for `SolarSystem`, `0.0` = paused).
    pub orbit_speed: f32,
    /// High-level editable optical & planetary elements (compiled into `segments` and `circles`).
    pub elements: Vec<OpticalElement>,
    next_element_id: u32,
    /// Compiled line segments uploaded to GPU.
    pub segments: Vec<OptSegment>,
    /// Compiled circular arcs / lenses / planets / atmospheres uploaded to GPU.
    pub circles: Vec<OptCircle>,
}

impl PhotonScene {
    /// Construct one of the built-in presets with balanced defaults.
    pub fn preset(id: PhotonSceneId) -> Self {
        let mut s = Self {
            id,
            view_half_height: 5.0,
            view_center: [0.0, 0.0],
            emitters: Vec::new(),
            next_emitter_id: 1,
            spin: 0.95,
            dispersion_scale: 1.0,
            grin_strength: 1.0,
            orbit_speed: 0.35,
            elements: Vec::new(),
            next_element_id: 1,
            segments: Vec::new(),
            circles: Vec::new(),
        };
        s.rebuild_geometry();
        s
    }

    /// Add a new photon emitter and return its index.
    pub fn add_emitter(
        &mut self,
        kind: EmitterKind,
        spectrum: EmitterSpectrum,
        pos: [f32; 2],
        angle: f32,
        spread: f32,
        power: f32,
    ) -> usize {
        if self.emitters.len() >= MAX_EMITTERS {
            return self.emitters.len() - 1;
        }
        let id = self.next_emitter_id;
        self.next_emitter_id = self.next_emitter_id.wrapping_add(1);
        self.emitters.push(PhotonEmitter {
            id,
            kind,
            spectrum,
            pos,
            angle,
            spread,
            power: power.max(0.05),
        });
        self.emitters.len() - 1
    }

    /// Remove an emitter by index (keeps at least one emitter in non-sandbox scenes).
    pub fn remove_emitter(&mut self, idx: usize) {
        if self.emitters.len() > 1 && idx < self.emitters.len() {
            self.emitters.remove(idx);
        }
    }

    /// Hit-test emitters in world space.
    pub fn hit_test_emitter(&self, world: [f32; 2], pick_margin: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, em) in self.emitters.iter().enumerate() {
            let d = ((world[0] - em.pos[0]).powi(2) + (world[1] - em.pos[1]).powi(2)).sqrt();
            let r = match em.kind {
                EmitterKind::Disk => em.spread.max(pick_margin),
                _ => pick_margin,
            };
            if d <= r && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Pack all emitters into GPU records with cumulative power distribution for multi-emitter sampling.
    pub fn pack_gpu_emitters(&self) -> Vec<GpuEmitter> {
        let total_power: f32 = self
            .emitters
            .iter()
            .take(MAX_EMITTERS)
            .map(|e| e.power.max(0.01))
            .sum::<f32>()
            .max(1e-5);
        let mut acc = 0.0_f32;
        let n = self.emitters.len().min(MAX_EMITTERS);
        let mut out = Vec::with_capacity(n);
        for (i, em) in self.emitters.iter().take(n).enumerate() {
            acc += em.power.max(0.01) / total_power;
            let cdf_hi = if i + 1 == n { 1.0 } else { acc.min(1.0) };
            let (spec_kind, spec_param) = match em.spectrum {
                EmitterSpectrum::WhiteFlat => (SPEC_WHITE_FLAT, 550.0),
                EmitterSpectrum::Blackbody(t) => (SPEC_BLACKBODY, t.clamp(1500.0, 25000.0)),
                EmitterSpectrum::Laser(nm) => (SPEC_LASER, nm.clamp(380.0, 780.0)),
            };
            out.push(GpuEmitter {
                geom: [em.pos[0], em.pos[1], em.angle, em.spread],
                spec: [spec_param, em.power.max(0.01), cdf_hi, 0.0],
                info: [em.kind.as_u32(), spec_kind, 0, 0],
            });
        }
        out
    }

    /// Advance orbital mechanics for orbiting planets/moons by `dt` seconds.
    /// Returns `true` if any orbiting body moved.
    pub fn step_orbits(&mut self, dt: f32) -> bool {
        if self.orbit_speed.abs() < 1e-5 || dt <= 0.0 {
            return false;
        }
        let mut moved = false;
        let sun_pos = self.emitters.first().map(|e| e.pos).unwrap_or([0.0, 0.0]);
        // Pass 1: primary planets orbiting the Sun/origin.
        for i in 0..self.elements.len() {
            if let ElementKind::Planet {
                orbit_radius,
                orbit_speed,
                orbit_phase,
                parent_idx: None,
                ..
            } = &mut self.elements[i].kind
                && *orbit_radius > 0.0
                && orbit_speed.abs() > 0.0
            {
                *orbit_phase = (*orbit_phase + *orbit_speed * self.orbit_speed * dt)
                    .rem_euclid(std::f32::consts::TAU);
                let p = *orbit_phase;
                let r = *orbit_radius;
                self.elements[i].pos = [sun_pos[0] + r * p.cos(), sun_pos[1] + r * p.sin()];
                moved = true;
            }
        }
        // Pass 2: moons orbiting a parent planet.
        for i in 0..self.elements.len() {
            let parent_pos = if let ElementKind::Planet {
                parent_idx: Some(pidx),
                ..
            } = self.elements[i].kind
            {
                self.elements.get(pidx).map(|e| e.pos)
            } else {
                None
            };
            if let Some(center) = parent_pos
                && let ElementKind::Planet {
                    orbit_radius,
                    orbit_speed,
                    orbit_phase,
                    ..
                } = &mut self.elements[i].kind
                && *orbit_radius > 0.0
            {
                *orbit_phase = (*orbit_phase + *orbit_speed * self.orbit_speed * dt)
                    .rem_euclid(std::f32::consts::TAU);
                let p = *orbit_phase;
                let r = *orbit_radius;
                self.elements[i].pos = [center[0] + r * p.cos(), center[1] + r * p.sin()];
                moved = true;
            }
        }
        if moved {
            self.compile_elements();
        }
        moved
    }

    /// Add a new editable optical/planetary element, compile GPU primitives, and return its index.
    pub fn add_element(
        &mut self,
        name: &'static str,
        kind: ElementKind,
        pos: [f32; 2],
        angle: f32,
        material: GlassMaterial,
    ) -> usize {
        let id = self.next_element_id;
        self.next_element_id = self.next_element_id.wrapping_add(1);
        self.elements.push(OpticalElement {
            id,
            name,
            kind,
            pos,
            angle,
            material,
        });
        self.compile_elements();
        self.elements.len() - 1
    }

    /// Remove an editable optical element by index and recompile GPU primitives.
    pub fn remove_element(&mut self, idx: usize) {
        if idx < self.elements.len() {
            self.elements.remove(idx);
            // Clear stale parent_idx references if a planet with moons was removed.
            for el in &mut self.elements {
                if let ElementKind::Planet { parent_idx, .. } = &mut el.kind
                    && let Some(p) = *parent_idx
                {
                    if p == idx {
                        *parent_idx = None;
                    } else if p > idx {
                        *parent_idx = Some(p - 1);
                    }
                }
            }
            self.compile_elements();
        }
    }

    /// Hit-test editable optical elements in world space; returns the closest matching element index.
    pub fn hit_test_element(&self, world: [f32; 2], pick_margin: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, el) in self.elements.iter().enumerate() {
            let d = distance_to_element(el, world);
            if d <= pick_margin && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// Compile `self.elements` into GPU `segments` and `circles` without touching emitters or camera.
    pub fn compile_elements(&mut self) {
        self.segments.clear();
        self.circles.clear();
        let disp = self.dispersion_scale.clamp(0.0, 4.0);

        for el in &self.elements {
            let (ca, cb) = el.material.cauchy(disp);
            let ca_cos = el.angle.cos();
            let ca_sin = el.angle.sin();
            let rot = |lx: f32, ly: f32| -> [f32; 2] {
                [
                    el.pos[0] + lx * ca_cos - ly * ca_sin,
                    el.pos[1] + lx * ca_sin + ly * ca_cos,
                ]
            };

            match el.kind {
                ElementKind::Prism { radius } => {
                    let v0 = rot(0.0, radius);
                    let v1 = rot(-radius * 0.866_025_4, -0.5 * radius);
                    let v2 = rot(radius * 0.866_025_4, -0.5 * radius);
                    for (p0, p1) in [(v0, v1), (v1, v2), (v2, v0)] {
                        self.segments.push(OptSegment {
                            endpoints: [p0[0], p0[1], p1[0], p1[1]],
                            material: [ca, cb, 0.0, 0.0],
                            meta: [MAT_GLASS, 0, 0, 0],
                        });
                    }
                }
                ElementKind::Lens {
                    radius,
                    half_offset,
                } => {
                    let c_left = rot(half_offset, 0.0);
                    let c_right = rot(-half_offset, 0.0);
                    self.circles.push(OptCircle {
                        geom: [c_left[0], c_left[1], radius, radius],
                        clip_arc: [c_right[0], c_right[1], -2.0, 0.0],
                        material: [ca, cb, 0.0, 0.0],
                        meta: [MAT_GLASS, 0, 0, 0],
                    });
                    self.circles.push(OptCircle {
                        geom: [c_right[0], c_right[1], radius, radius],
                        clip_arc: [c_left[0], c_left[1], -2.0, 0.0],
                        material: [ca, cb, 0.0, 0.0],
                        meta: [MAT_GLASS, 0, 0, 0],
                    });
                }
                ElementKind::FlatMirror { half_len } => {
                    let p0 = rot(0.0, -half_len);
                    let p1 = rot(0.0, half_len);
                    self.segments.push(OptSegment {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        material: [1.0, 0.0, 0.0, 0.0],
                        meta: [MAT_MIRROR, 0, 0, 0],
                    });
                }
                ElementKind::CurvedMirror {
                    radius,
                    arc_half_angle,
                } => {
                    self.circles.push(OptCircle {
                        geom: [el.pos[0], el.pos[1], radius, 0.0],
                        clip_arc: [0.0, 0.0, arc_half_angle.cos(), el.angle],
                        material: [1.0, 0.0, 0.0, 0.0],
                        meta: [MAT_MIRROR, 1, 0, 0],
                    });
                }
                ElementKind::BeamSplitter { half_len } => {
                    let p0 = rot(0.0, -half_len);
                    let p1 = rot(0.0, half_len);
                    // 50% specular reflection, 50% transmission
                    self.segments.push(OptSegment {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        material: [0.95, 0.95, 0.98, 0.50],
                        meta: [MAT_SPLITTER, 0, 0, 0],
                    });
                }
                ElementKind::MatteWall { half_len, color } => {
                    let p0 = rot(0.0, -half_len);
                    let p1 = rot(0.0, half_len);
                    self.segments.push(OptSegment {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        material: [color[0], color[1], color[2], 0.0],
                        meta: [MAT_DIFFUSE, 0, 0, 0],
                    });
                }
                ElementKind::Planet {
                    radius,
                    albedo,
                    atmo_thickness,
                    atmo_density,
                    ring_outer,
                    ..
                } => {
                    // Solid planetary body (diffuse Lambertian reflector).
                    self.circles.push(OptCircle {
                        geom: [el.pos[0], el.pos[1], radius, 0.0],
                        clip_arc: [0.0, 0.0, -2.0, 0.0],
                        material: [albedo[0], albedo[1], albedo[2], 0.0],
                        meta: [MAT_DIFFUSE, 0, 0, 0],
                    });
                    // Optional Rayleigh atmosphere shell.
                    if atmo_thickness > 0.02 && atmo_density > 0.01 {
                        let r_atmo = radius * (1.0 + atmo_thickness);
                        self.circles.push(OptCircle {
                            geom: [el.pos[0], el.pos[1], r_atmo, 0.0],
                            clip_arc: [0.0, 0.0, -2.0, 0.0],
                            material: [atmo_density, 0.04, radius, 0.0],
                            meta: [MAT_ATMOSPHERE, 0, 0, 0],
                        });
                    }
                    // Optional translucent icy planetary ring (two segments flanking the planet).
                    if ring_outer > radius * 1.15 {
                        let r_in = radius * 1.25;
                        for sign in [-1.0_f32, 1.0_f32] {
                            let p0 = rot(sign * r_in, 0.0);
                            let p1 = rot(sign * ring_outer, 0.0);
                            // Translucent icy ring: diffuse/specular partial scattering (`w = 0.35` reflect/scatter).
                            self.segments.push(OptSegment {
                                endpoints: [p0[0], p0[1], p1[0], p1[1]],
                                material: [0.88, 0.80, 0.64, 0.38],
                                meta: [MAT_SPLITTER, 1, 0, 0],
                            });
                        }
                    }
                }
                ElementKind::Absorber { half_len } => {
                    let p0 = rot(0.0, -half_len);
                    let p1 = rot(0.0, half_len);
                    self.segments.push(OptSegment {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        material: [1.0, 0.0, 0.0, 0.0],
                        meta: [MAT_ABSORBER, 0, 0, 0],
                    });
                }
            }
        }
    }

    /// Alias kept for callers updating material dispersion without moving emitters.
    pub fn rebuild_primitives(&mut self) {
        self.compile_elements();
    }

    /// Clear all elements (keeps emitters) for blank-canvas building.
    pub fn clear_elements(&mut self) {
        self.elements.clear();
        self.compile_elements();
    }

    /// Reset emitters, camera framing, and analytical geometry to the preset defaults.
    pub fn rebuild_geometry(&mut self) {
        self.elements.clear();
        self.emitters.clear();
        self.next_element_id = 1;
        self.next_emitter_id = 1;
        self.grin_strength = 1.0;

        match self.id {
            PhotonSceneId::Prism => {
                self.view_half_height = 4.6;
                self.view_center = [0.6, -0.2];
                self.dispersion_scale = 1.0;
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-6.2, -1.55],
                    0.48,
                    0.22,
                    1.0,
                );
                self.add_element(
                    "Prism",
                    ElementKind::Prism { radius: 2.6 },
                    [-0.8, 0.0],
                    0.0,
                    GlassMaterial::StudioFlint,
                );
                self.add_element(
                    "Concave Mirror",
                    ElementKind::CurvedMirror {
                        radius: 4.4,
                        arc_half_angle: 0.82_f32.acos(),
                    },
                    [2.2, -0.4],
                    -0.22,
                    GlassMaterial::StudioFlint,
                );
            }
            PhotonSceneId::Cardioid => {
                self.view_half_height = 4.8;
                self.view_center = [0.0, 0.0];
                self.dispersion_scale = 1.0;
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-7.2, 0.0],
                    0.0,
                    3.55,
                    1.0,
                );
                self.add_element(
                    "Cavity Mirror",
                    ElementKind::CurvedMirror {
                        radius: 3.8,
                        arc_half_angle: (-0.25_f32).acos(),
                    },
                    [0.0, 0.0],
                    0.0,
                    GlassMaterial::StudioFlint,
                );
            }
            PhotonSceneId::Lens => {
                self.view_half_height = 4.8;
                self.view_center = [1.0, 0.0];
                self.dispersion_scale = 1.0;
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-6.5, 0.0],
                    0.0,
                    2.6,
                    1.0,
                );
                self.add_element(
                    "Biconvex Lens",
                    ElementKind::Lens {
                        radius: 4.6,
                        half_offset: 3.3,
                    },
                    [-1.5, 0.0],
                    0.0,
                    GlassMaterial::StudioFlint,
                );
            }
            PhotonSceneId::SolarSystem => {
                // Orrery-scale Solar System: true 5778 K Planck solar disk casting soft umbra/penumbra
                // shadows onto Mercury, Venus (dense golden atmosphere), Earth (Rayleigh blue sky +
                // orbiting Moon for real solar & blood-moon lunar eclipses!), Mars, Jupiter, and Saturn.
                self.view_half_height = 10.5;
                self.view_center = [0.0, 0.0];
                self.dispersion_scale = 1.0;
                self.orbit_speed = 0.25;

                // Physical 5778 K Sun disk at the center.
                self.add_emitter(
                    EmitterKind::Disk,
                    EmitterSpectrum::Blackbody(5778.0),
                    [0.0, 0.0],
                    0.0,
                    1.15,
                    1.0,
                );

                let place = |r: f32, ang: f32| [r * ang.cos(), r * ang.sin()];

                // Mercury
                let merc_r = 2.45;
                let merc_a = 2.35;
                self.add_element(
                    "Mercury",
                    ElementKind::Planet {
                        radius: 0.24,
                        albedo: [0.52, 0.50, 0.48],
                        atmo_thickness: 0.0,
                        atmo_density: 0.0,
                        ring_outer: 0.0,
                        orbit_radius: merc_r,
                        orbit_speed: 1.55,
                        orbit_phase: merc_a,
                        parent_idx: None,
                    },
                    place(merc_r, merc_a),
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Venus (thick golden scattering cloud atmosphere)
                let ven_r = 3.75;
                let ven_a = -0.85;
                self.add_element(
                    "Venus",
                    ElementKind::Planet {
                        radius: 0.42,
                        albedo: [0.88, 0.78, 0.54],
                        atmo_thickness: 0.42,
                        atmo_density: 2.2,
                        ring_outer: 0.0,
                        orbit_radius: ven_r,
                        orbit_speed: 0.98,
                        orbit_phase: ven_a,
                        parent_idx: None,
                    },
                    place(ven_r, ven_a),
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Earth (Rayleigh sky-blue atmosphere!)
                let earth_r = 5.45;
                let earth_a = 0.42;
                let earth_idx = self.add_element(
                    "Earth",
                    ElementKind::Planet {
                        radius: 0.50,
                        albedo: [0.30, 0.52, 0.78],
                        atmo_thickness: 0.48,
                        atmo_density: 2.1,
                        ring_outer: 0.0,
                        orbit_radius: earth_r,
                        orbit_speed: 0.65,
                        orbit_phase: earth_a,
                        parent_idx: None,
                    },
                    place(earth_r, earth_a),
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Moon orbiting Earth (positioned near eclipse alignment!)
                let ep = place(earth_r, earth_a);
                let moon_r = 1.15;
                let moon_a = earth_a + 0.12;
                self.add_element(
                    "Moon",
                    ElementKind::Planet {
                        radius: 0.18,
                        albedo: [0.68, 0.67, 0.64],
                        atmo_thickness: 0.0,
                        atmo_density: 0.0,
                        ring_outer: 0.0,
                        orbit_radius: moon_r,
                        orbit_speed: 2.8,
                        orbit_phase: moon_a,
                        parent_idx: Some(earth_idx),
                    },
                    [ep[0] + moon_r * moon_a.cos(), ep[1] + moon_r * moon_a.sin()],
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Mars (dusty iron-oxide red albedo + thin atmosphere)
                let mars_r = 7.35;
                let mars_a = 2.72;
                self.add_element(
                    "Mars",
                    ElementKind::Planet {
                        radius: 0.34,
                        albedo: [0.84, 0.36, 0.22],
                        atmo_thickness: 0.24,
                        atmo_density: 0.65,
                        ring_outer: 0.0,
                        orbit_radius: mars_r,
                        orbit_speed: 0.44,
                        orbit_phase: mars_a,
                        parent_idx: None,
                    },
                    place(mars_r, mars_a),
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Jupiter (large bright gas giant casting a dramatic long umbra/penumbra shadow)
                let jup_r = 9.5;
                let jup_a = -2.18;
                self.add_element(
                    "Jupiter",
                    ElementKind::Planet {
                        radius: 0.88,
                        albedo: [0.85, 0.72, 0.56],
                        atmo_thickness: 0.18,
                        atmo_density: 0.9,
                        ring_outer: 0.0,
                        orbit_radius: jup_r,
                        orbit_speed: 0.24,
                        orbit_phase: jup_a,
                        parent_idx: None,
                    },
                    place(jup_r, jup_a),
                    0.0,
                    GlassMaterial::CrownBk7,
                );

                // Saturn (golden gas giant with translucent icy rings)
                let sat_r = 12.2;
                let sat_a = -0.22;
                self.add_element(
                    "Saturn",
                    ElementKind::Planet {
                        radius: 0.72,
                        albedo: [0.88, 0.79, 0.58],
                        atmo_thickness: 0.16,
                        atmo_density: 0.7,
                        ring_outer: 1.55,
                        orbit_radius: sat_r,
                        orbit_speed: 0.15,
                        orbit_phase: sat_a,
                        parent_idx: None,
                    },
                    place(sat_r, sat_a),
                    0.48,
                    GlassMaterial::CrownBk7,
                );
            }
            PhotonSceneId::Sandbox => {
                // Multi-emitter playground starter: a 532nm green laser + white spotlight + beam splitter
                // + diamond prism + Rayleigh planet so users immediately see what they can mix!
                self.view_half_height = 5.4;
                self.view_center = [0.0, 0.0];
                self.dispersion_scale = 1.0;

                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-7.2, 1.4],
                    -0.14,
                    0.24,
                    1.0,
                );
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::Laser(532.0),
                    [-7.2, -2.0],
                    0.22,
                    0.08,
                    0.55,
                );
                self.add_element(
                    "Beam Splitter",
                    ElementKind::BeamSplitter { half_len: 1.9 },
                    [-2.4, 0.4],
                    0.68,
                    GlassMaterial::CrownBk7,
                );
                self.add_element(
                    "Diamond Prism",
                    ElementKind::Prism { radius: 1.85 },
                    [1.6, 0.2],
                    0.15,
                    GlassMaterial::Diamond,
                );
                self.add_element(
                    "Planet",
                    ElementKind::Planet {
                        radius: 0.85,
                        albedo: [0.35, 0.58, 0.82],
                        atmo_thickness: 0.45,
                        atmo_density: 2.0,
                        ring_outer: 0.0,
                        orbit_radius: 0.0,
                        orbit_speed: 0.0,
                        orbit_phase: 0.0,
                        parent_idx: None,
                    },
                    [-1.6, -2.8],
                    0.0,
                    GlassMaterial::CrownBk7,
                );
            }
            PhotonSceneId::GrinLuneburg => {
                self.view_half_height = 5.0;
                self.view_center = [0.8, 0.0];
                self.dispersion_scale = 0.45;
                self.grin_strength = 1.0;
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-6.2, 0.0],
                    0.0,
                    2.95,
                    1.0,
                );
            }
            PhotonSceneId::GrinBlackHole => {
                self.view_half_height = 6.5;
                self.view_center = [1.4, 0.0];
                self.dispersion_scale = 1.0;
                self.grin_strength = 1.0;
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-8.4, 0.0],
                    0.0,
                    3.8,
                    1.0,
                );
            }
            PhotonSceneId::GrinMirage => {
                self.view_half_height = 4.2;
                self.view_center = [2.2, 0.2];
                self.dispersion_scale = 1.4;
                self.grin_strength = 1.15;
                self.add_emitter(
                    EmitterKind::Spot,
                    EmitterSpectrum::Blackbody(5778.0),
                    [-4.6, 0.45],
                    -0.22,
                    0.26,
                    1.0,
                );
            }
            PhotonSceneId::KerrBeam => {
                self.view_half_height = 14.0;
                self.view_center = [2.4, 0.0];
                self.add_emitter(
                    EmitterKind::Beam,
                    EmitterSpectrum::WhiteFlat,
                    [-18.5, 0.0],
                    0.0,
                    9.0,
                    1.0,
                );
            }
            PhotonSceneId::KerrPulsar => {
                self.view_half_height = 13.5;
                self.view_center = [1.2, 0.0];
                self.add_emitter(
                    EmitterKind::Point,
                    EmitterSpectrum::WhiteFlat,
                    [-8.5, 0.0],
                    0.0,
                    std::f32::consts::PI,
                    1.0,
                );
            }
        }
        self.compile_elements();
    }
}

fn dist_point_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let len2 = (abx * abx + aby * aby).max(1e-10);
    let t = (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len2).clamp(0.0, 1.0);
    let dx = p[0] - (a[0] + abx * t);
    let dy = p[1] - (a[1] + aby * t);
    (dx * dx + dy * dy).sqrt()
}

fn distance_to_element(el: &OpticalElement, world: [f32; 2]) -> f32 {
    let dx = world[0] - el.pos[0];
    let dy = world[1] - el.pos[1];
    let c = el.angle.cos();
    let s = el.angle.sin();
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;

    match el.kind {
        ElementKind::Prism { radius } => {
            let v0 = [0.0, radius];
            let v1 = [-radius * 0.866_025_4, -0.5 * radius];
            let v2 = [radius * 0.866_025_4, -0.5 * radius];
            let cross = |a: [f32; 2], b: [f32; 2]| {
                (b[0] - a[0]) * (ly - a[1]) - (b[1] - a[1]) * (lx - a[0])
            };
            if cross(v0, v1) >= 0.0 && cross(v1, v2) >= 0.0 && cross(v2, v0) >= 0.0 {
                return 0.0;
            }
            dist_point_segment([lx, ly], v0, v1)
                .min(dist_point_segment([lx, ly], v1, v2))
                .min(dist_point_segment([lx, ly], v2, v0))
        }
        ElementKind::Lens {
            radius,
            half_offset,
        } => {
            let d1 = ((lx - half_offset).powi(2) + ly * ly).sqrt() - radius;
            let d2 = ((lx + half_offset).powi(2) + ly * ly).sqrt() - radius;
            d1.max(d2).max(0.0)
        }
        ElementKind::FlatMirror { half_len }
        | ElementKind::BeamSplitter { half_len }
        | ElementKind::MatteWall { half_len, .. }
        | ElementKind::Absorber { half_len } => {
            dist_point_segment([lx, ly], [0.0, -half_len], [0.0, half_len])
        }
        ElementKind::CurvedMirror {
            radius,
            arc_half_angle,
        } => {
            let r = (lx * lx + ly * ly).sqrt();
            let ang = ly.atan2(lx).abs();
            if ang <= arc_half_angle {
                (r - radius).abs()
            } else {
                let e0 = [radius * arc_half_angle.cos(), radius * arc_half_angle.sin()];
                let e1 = [e0[0], -e0[1]];
                let d0 = ((lx - e0[0]).powi(2) + (ly - e0[1]).powi(2)).sqrt();
                let d1 = ((lx - e1[0]).powi(2) + (ly - e1[1]).powi(2)).sqrt();
                d0.min(d1)
            }
        }
        ElementKind::Planet { radius, .. } => {
            let r = (lx * lx + ly * ly).sqrt();
            (r - radius).max(0.0)
        }
    }
}
