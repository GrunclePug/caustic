//! Built-in optical and relativistic photon scenes and GPU scene-primitive packing.

use bytemuck::{Pod, Zeroable};

/// Built-in forward photon simulation presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonSceneId {
    /// Dispersive flint-glass prism splitting a white beam into a rainbow + angled mirror.
    Prism,
    /// Circular mirror cavity producing the classic nephroid / cardioid caustic envelope.
    Cardioid,
    /// Thick dispersive biconvex glass lens showing spherical & chromatic aberration caustics.
    Lens,
    /// Collimated relativistic photon beam grazing a rotating Kerr black hole (Ingoing KS).
    KerrBeam,
    /// Isotropic point beacon near a spinning Kerr black hole producing relativistic wavefront caustics.
    KerrPulsar,
}

impl PhotonSceneId {
    pub fn name(self) -> &'static str {
        match self {
            Self::Prism => "prism",
            Self::Cardioid => "cardioid",
            Self::Lens => "lens",
            Self::KerrBeam => "kerr-beam",
            Self::KerrPulsar => "kerr-pulsar",
        }
    }

    pub fn is_kerr(self) -> bool {
        matches!(self, Self::KerrBeam | Self::KerrPulsar)
    }
}

/// Accumulation vs real-time pulse flight mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PhotonMode {
    /// Continuous long-exposure HDR accumulation (razor-sharp caustics & rainbows).
    Accumulate = 0,
    /// Synchronized flying photon pulses with smooth trail persistence (watch light move at c).
    Pulse = 1,
}

/// Material kinds shared with `photon_step.wgsl` and `photon_resolve.wgsl`.
pub const MAT_MIRROR: u32 = 1;
pub const MAT_GLASS: u32 = 2;
pub const MAT_ABSORBER: u32 = 3;

/// Emitter shapes shared with `photon_step.wgsl`.
pub const EMITTER_BEAM: u32 = 0;
pub const EMITTER_POINT: u32 = 1;

/// 2D line segment optical element (`32 bytes`, matches WGSL `struct OptSegment`).
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

/// 2D circular arc / lens surface (`48 bytes`, matches WGSL `struct OptCircle`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OptCircle {
    /// `xy = center`, `z = radius`, `w = clip_radius` (if > 0, hit must lie inside clip circle!).
    pub geom: [f32; 4],
    /// `xy = clip_center`, `z = arc_cos_half` (if > -1, restrict to arc around `arc_dir`), `w = arc_angle_center`.
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
    /// World half-height in scene units (`M` for Kerr scenes, meters/cm for optical scenes).
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
    /// Dispersion scale multiplier (`1.0` default; adjustable live in optical scenes).
    pub dispersion_scale: f32,
    /// Line segments in scene.
    pub segments: Vec<OptSegment>,
    /// Circular arcs / lenses in scene.
    pub circles: Vec<OptCircle>,
}

impl PhotonScene {
    /// Construct one of the 5 built-in presets with balanced defaults.
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
            segments: Vec::new(),
            circles: Vec::new(),
        };
        s.rebuild_geometry();
        s
    }

    /// Rebuild analytical scene geometry (called on preset switch or dispersion change).
    pub fn rebuild_geometry(&mut self) {
        self.segments.clear();
        self.circles.clear();
        let disp = self.dispersion_scale.clamp(0.1, 4.0);

        match self.id {
            PhotonSceneId::Prism => {
                self.view_half_height = 4.6;
                self.view_center = [0.6, -0.2];
                self.emitter_pos = [-6.2, -1.55];
                self.emitter_angle = 0.48;
                self.emitter_spread = 0.22;
                self.emitter_kind = EMITTER_BEAM;

                // Equilateral flint glass prism centered near (-0.8, 0.0), CCW vertices.
                // Incidence angle ~57.5 deg keeps the entire 380..780 nm spectrum below TIR
                // on the exit face while fanning a vivid ~21 deg rainbow across the right half.
                let cx = -0.8_f32;
                let cy = 0.0_f32;
                let r = 2.6_f32;
                let v0 = [cx, cy + r];
                let v1 = [cx - r * 0.866_025_4, cy - 0.5 * r];
                let v2 = [cx + r * 0.866_025_4, cy - 0.5 * r];
                let a_ior = 1.46;
                let b_ior = 0.042 * disp;
                for (p0, p1) in [(v0, v1), (v1, v2), (v2, v0)] {
                    self.segments.push(OptSegment {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        material: [a_ior, b_ior, 0.0, 0.0],
                        meta: [MAT_GLASS, 0, 0, 0],
                    });
                }

                // Concave mirror on the right catching the full dispersed rainbow and focusing
                // it back into a reflected chromatic caustic.
                self.circles.push(OptCircle {
                    geom: [2.2, -0.4, 4.4, 0.0],
                    clip_arc: [0.0, 0.0, 0.82, -0.22],
                    material: [1.0, 0.0, 0.0, 0.0],
                    meta: [MAT_MIRROR, 1, 0, 0],
                });
            }
            PhotonSceneId::Cardioid => {
                // Classic circular coffee-cup mirror producing the exact nephroid caustic!
                self.view_half_height = 4.8;
                self.view_center = [0.0, 0.0];
                self.emitter_pos = [-7.2, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 3.55;
                self.emitter_kind = EMITTER_BEAM;

                // Open semicircular mirror centered at origin, radius 3.8, facing left (-x)
                // so a parallel beam from the left enters freely and reflects into the nephroid!
                self.circles.push(OptCircle {
                    geom: [0.0, 0.0, 3.8, 0.0],
                    clip_arc: [0.0, 0.0, -0.25, 0.0], // arc covering ~210 deg around +x (right side)
                    material: [1.0, 0.0, 0.0, 0.0],
                    meta: [MAT_MIRROR, 1, 0, 0],
                });
            }
            PhotonSceneId::Lens => {
                // Thick dispersive biconvex glass lens (exact intersection of two circles!)
                self.view_half_height = 4.8;
                self.view_center = [1.0, 0.0];
                self.emitter_pos = [-6.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 2.6;
                self.emitter_kind = EMITTER_BEAM;

                let r_lens = 4.6_f32;
                let offset = 3.3_f32;
                let cx_l = -1.5_f32;
                let a_ior = 1.52;
                let b_ior = 0.042 * disp;
                // Left convex surface: circle centered at (cx_l + offset, 0), clipped to inside right circle.
                self.circles.push(OptCircle {
                    geom: [cx_l + offset, 0.0, r_lens, r_lens],
                    clip_arc: [cx_l - offset, 0.0, -2.0, 0.0],
                    material: [a_ior, b_ior, 0.0, 0.0],
                    meta: [MAT_GLASS, 0, 0, 0],
                });
                // Right convex surface: circle centered at (cx_l - offset, 0), clipped to inside left circle.
                self.circles.push(OptCircle {
                    geom: [cx_l - offset, 0.0, r_lens, r_lens],
                    clip_arc: [cx_l + offset, 0.0, -2.0, 0.0],
                    material: [a_ior, b_ior, 0.0, 0.0],
                    meta: [MAT_GLASS, 0, 0, 0],
                });
            }
            PhotonSceneId::KerrBeam => {
                // Collimated relativistic laser beam grazing a spinning Kerr black hole (Ingoing KS)!
                self.view_half_height = 14.0;
                self.view_center = [2.4, 0.0];
                self.emitter_pos = [-18.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = 9.0;
                self.emitter_kind = EMITTER_BEAM;
            }
            PhotonSceneId::KerrPulsar => {
                // Isotropic point beacon emitting 360-degree relativistic photon wavefronts!
                self.view_half_height = 13.5;
                self.view_center = [1.2, 0.0];
                self.emitter_pos = [-8.5, 0.0];
                self.emitter_angle = 0.0;
                self.emitter_spread = std::f32::consts::PI;
                self.emitter_kind = EMITTER_POINT;
            }
        }
    }
}
