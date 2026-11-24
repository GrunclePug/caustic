//! Built-in optical, GRIN, and relativistic photon scenes plus editable optical bench primitives.

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
    /// Dispersive flint-glass prism splitting a white beam into a rainbow + concave mirror.
    Prism,
    /// Circular mirror cavity producing the classic nephroid caustic envelope.
    Cardioid,
    /// Thick dispersive biconvex glass lens showing spherical & chromatic aberration caustics.
    Lens,
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
    pub const ALL: [Self; 8] = [
        Self::Prism,
        Self::Cardioid,
        Self::Lens,
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
            Self::GrinLuneburg => "Luneburg",
            Self::GrinBlackHole => "Optical BH",
            Self::GrinMirage => "Mirage",
            Self::KerrBeam => "Kerr Beam",
            Self::KerrPulsar => "Kerr Pulsar",
        }
    }

    pub fn category(self) -> SceneCategory {
        match self {
            Self::Prism | Self::Cardioid | Self::Lens => SceneCategory::Optics,
            Self::GrinLuneburg | Self::GrinBlackHole | Self::GrinMirage => SceneCategory::Grin,
            Self::KerrBeam | Self::KerrPulsar => SceneCategory::Relativity,
        }
    }

    pub fn physics_kind(self) -> PhysicsKind {
        match self {
            Self::Prism | Self::Cardioid | Self::Lens => PhysicsKind::Optics,
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

/// Real dispersive optical glass / medium presets with Cauchy `(A, B)` parameters (`um^2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlassMaterial {
    Water,
    CrownBk7,
    FlintSf11,
    StudioFlint,
    Diamond,
}

impl GlassMaterial {
    pub const ALL: [Self; 5] = [
        Self::Water,
        Self::CrownBk7,
        Self::FlintSf11,
        Self::StudioFlint,
        Self::Diamond,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Water => "Water",
            Self::CrownBk7 => "Crown BK7",
            Self::FlintSf11 => "Flint SF11",
            Self::StudioFlint => "Dense Flint",
            Self::Diamond => "Diamond",
        }
    }

    /// Return `(cauchy_a, cauchy_b * dispersion_scale)`.
    pub fn cauchy(self, dispersion_scale: f32) -> (f32, f32) {
        let (a, b) = match self {
            Self::Water => (1.324, 0.0055),
            Self::CrownBk7 => (1.505, 0.0110),
            Self::FlintSf11 => (1.560, 0.0280),
            Self::StudioFlint => (1.460, 0.0420),
            Self::Diamond => (2.378, 0.0240),
        };
        (a, b * dispersion_scale.clamp(0.0, 4.0))
    }
}

/// Editable optical element kinds for the interactive optical bench.
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
            Self::Absorber { .. } => "Beam Stop",
        }
    }

    pub fn is_glass(self) -> bool {
        matches!(self, Self::Prism { .. } | Self::Lens { .. })
    }

    /// Primary editable size parameter for inspector slider.
    pub fn size(self) -> f32 {
        match self {
            Self::Prism { radius } | Self::CurvedMirror { radius, .. } => radius,
            Self::Lens { radius, .. } => radius,
            Self::FlatMirror { half_len } | Self::Absorber { half_len } => half_len,
        }
    }

    /// Update primary size parameter while preserving sensible geometry ratios.
    pub fn set_size(&mut self, value: f32) {
        let v = value.clamp(0.6, 8.0);
        match self {
            Self::Prism { radius } => *radius = v,
            Self::Lens {
                radius,
                half_offset,
            } => {
                let ratio = (*half_offset / (*radius).max(0.1)).clamp(0.35, 0.90);
                *radius = v;
                *half_offset = v * ratio;
            }
            Self::FlatMirror { half_len } | Self::Absorber { half_len } => *half_len = v,
            Self::CurvedMirror { radius, .. } => *radius = v,
        }
    }
}

/// Editable optical element on the interactive optical bench.
#[derive(Debug, Clone, Copy)]
pub struct OpticalElement {
    pub id: u32,
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

/// Emitter shapes shared with `photon_step.wgsl`.
pub const EMITTER_BEAM: u32 = 0;
pub const EMITTER_POINT: u32 = 1;

/// 2D line segment optical element (`48 bytes`, matches WGSL `struct OptSegment`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OptSegment {
    /// `xy = p0`, `zw = p1` (outward normal is `(p1.y - p0.y, -(p1.x - p0.x))` normalized;
    /// for CCW polygons that points outward from the glass interior).
    pub endpoints: [f32; 4],
    /// `x = cauchy_a`, `y = cauchy_b`, `z = reserved`, `w = reserved`.
    pub material: [f32; 4],
    /// `x = mat_kind`, `yzw = reserved`.
    pub meta: [u32; 4],
}

/// 2D circular arc / lens surface (`64 bytes`, matches WGSL `struct OptCircle`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OptCircle {
    /// `xy = center`, `z = radius`, `w = clip_radius` (if > 0, hit must lie inside clip circle!).
    pub geom: [f32; 4],
    /// `xy = clip_center`, `z = arc_cos_half` (if > -1.5, restrict to arc around `arc_dir`), `w = arc_angle_center`.
    pub clip_arc: [f32; 4],
    /// `x = cauchy_a`, `y = cauchy_b`, `z = reserved`, `w = reserved`.
    pub material: [f32; 4],
    /// `x = mat_kind`, `y = invert_normal` (1 for inward-facing cavity mirror), `zw = reserved`.
    pub meta: [u32; 4],
}

/// Complete scene state driving both GPU shaders and interactive controls.
#[derive(Debug, Clone)]
pub struct PhotonScene {
    pub id: PhotonSceneId,
    /// World half-height in scene units (`M` for Kerr scenes, meters/cm for optical/GRIN scenes).
    pub view_half_height: f32,
    /// World center `(x, y)`.
    pub view_center: [f32; 2],
    /// Emitter position `(x, y)` in world coordinates.
    pub emitter_pos: [f32; 2],
    /// Emitter beam aim angle in radians.
    pub emitter_angle: f32,
    /// Emitter beam half-width (for `EMITTER_BEAM`) or cone half-angle (for `EMITTER_POINT`).
    pub emitter_spread: f32,
    /// Emitter type (`EMITTER_BEAM` or `EMITTER_POINT`).
    pub emitter_kind: u32,
    /// Kerr spin parameter `a in [-0.998, +0.998]` (used when `id.is_kerr()`).
    pub spin: f32,
    /// Dispersion scale multiplier (`1.0` default; adjustable live in optical & GRIN scenes).
    pub dispersion_scale: f32,
    /// GRIN field strength multiplier (`1.0` default; tunes Luneburg focus, optical BH capture radius, or mirage gradient).
    pub grin_strength: f32,
    /// High-level editable optical elements (compiled into `segments` and `circles`).
    pub elements: Vec<OpticalElement>,
    next_element_id: u32,
    /// Compiled line segments uploaded to GPU.
    pub segments: Vec<OptSegment>,
    /// Compiled circular arcs / lenses uploaded to GPU.
    pub circles: Vec<OptCircle>,
}

impl PhotonScene {
    /// Construct one of the 8 built-in presets with balanced defaults.
    pub fn preset(id: PhotonSceneId) -> Self {
        let mut s = Self {
            id,
            view_half_height: 5.0,
            view_center: [0.0, 0.0],
            emitter_pos: [-6.5, 0.0],
            emitter_angle: 0.0,
            emitter_spread: 0.5,
            emitter_kind: EMITTER_BEAM,
            spin: 0.95,
            dispersion_scale: 1.0,
            grin_strength: 1.0,
            elements: Vec::new(),
            next_element_id: 1,
            segments: Vec::new(),
            circles: Vec::new(),
        };
        s.rebuild_geometry();
        s
    }

    /// Add a new editable optical element, compile GPU primitives, and return its index.
    pub fn add_element(
        &mut self,
        kind: ElementKind,
        pos: [f32; 2],
        angle: f32,
        material: GlassMaterial,
    ) -> usize {
        let id = self.next_element_id;
        self.next_element_id = self.next_element_id.wrapping_add(1);
        self.elements.push(OpticalElement {
            id,
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

    /// Compile `self.elements` into GPU `segments` and `circles` without touching emitter or camera.
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
                    // Equilateral triangle in local frame, CCW vertices.
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

    /// Alias kept for callers updating material dispersion without moving the emitter.
    pub fn rebuild_primitives(&mut self) {
        self.compile_elements();
    }

    /// Reset emitter, camera framing, and analytical geometry to the preset defaults.
    pub fn rebuild_geometry(&mut self) {
        self.elements.clear();
        self.next_element_id = 1;
        self.grin_strength = 1.0;

        match self.id {
            PhotonSceneId::Prism => {
                self.view_half_height = 4.6;
                self.view_center = [0.6, -0.2];
                self.emitter_pos = [-6.2, -1.55];
                self.emitter_angle = 0.48;
                self.emitter_spread = 0.22;
                self.emitter_kind = EMITTER_BEAM;
                self.dispersion_scale = 1.0;
                self.add_element(
                    ElementKind::Prism { radius: 2.6 },
                    [-0.8, 0.0],
                    0.0,
                    GlassMaterial::StudioFlint,
                );
                self.add_element(
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
                self.emitter_pos = [-7.2, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 3.55;
                self.emitter_kind = EMITTER_BEAM;
                self.dispersion_scale = 1.0;
                self.add_element(
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
                self.emitter_pos = [-6.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 2.6;
                self.emitter_kind = EMITTER_BEAM;
                self.dispersion_scale = 1.0;
                self.add_element(
                    ElementKind::Lens {
                        radius: 4.6,
                        half_offset: 3.3,
                    },
                    [-1.5, 0.0],
                    0.0,
                    GlassMaterial::StudioFlint,
                );
            }
            PhotonSceneId::GrinLuneburg => {
                self.view_half_height = 5.0;
                self.view_center = [0.8, 0.0];
                self.emitter_pos = [-6.2, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 2.95;
                self.emitter_kind = EMITTER_BEAM;
                self.dispersion_scale = 0.45;
                self.grin_strength = 1.0;
            }
            PhotonSceneId::GrinBlackHole => {
                self.view_half_height = 6.5;
                self.view_center = [1.4, 0.0];
                self.emitter_pos = [-8.4, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 3.8;
                self.emitter_kind = EMITTER_BEAM;
                self.dispersion_scale = 1.0;
                self.grin_strength = 1.0;
            }
            PhotonSceneId::GrinMirage => {
                self.view_half_height = 4.2;
                self.view_center = [2.2, 0.2];
                self.emitter_pos = [-4.6, 0.45];
                self.emitter_angle = -0.22;
                self.emitter_spread = 0.26;
                self.emitter_kind = EMITTER_POINT;
                self.dispersion_scale = 1.4;
                self.grin_strength = 1.15;
            }
            PhotonSceneId::KerrBeam => {
                self.view_half_height = 14.0;
                self.view_center = [2.4, 0.0];
                self.emitter_pos = [-18.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 9.0;
                self.emitter_kind = EMITTER_BEAM;
            }
            PhotonSceneId::KerrPulsar => {
                self.view_half_height = 13.5;
                self.view_center = [1.2, 0.0];
                self.emitter_pos = [-8.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = std::f32::consts::PI;
                self.emitter_kind = EMITTER_POINT;
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
    // Transform cursor into element local coordinates.
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;

    match el.kind {
        ElementKind::Prism { radius } => {
            let v0 = [0.0, radius];
            let v1 = [-radius * 0.866_025_4, -0.5 * radius];
            let v2 = [radius * 0.866_025_4, -0.5 * radius];
            // Check inside CCW triangle.
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
        ElementKind::FlatMirror { half_len } | ElementKind::Absorber { half_len } => {
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
    }
}
