//! Unified minimal dark-glass interactive photon playground & relativity studio for `caustic`.
//!
//! Hosts both the 3D Outgoing Kerr-Schild black hole tracer and all 10 forward spectral
//! photon modes (dispersive optics, Solar System orrery, multi-emitter sandbox, continuous
//! GRIN eikonal optics, and equatorial Ingoing Kerr-Schild relativity) behind a minimal
//! `egui` 0.37 / `egui-wgpu` overlay (`H` or `Tab` hides all UI cleanly).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use egui::{
    Align2, Color32, CornerRadius, FontId, Frame, Id, LayerId, Margin, Pos2, RichText, Stroke,
    TextStyle, Vec2, ViewportId,
};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use caustic_blackhole::gpu_tracer::{BlackHolePipeline, FrameView, RenderSettings};
use caustic_blackhole::kerr::{
    Camera, KerrParams, MAX_CAMERA_RADIUS, MAX_SPIN, MIN_CAMERA_HORIZON_MARGIN, r_plus,
};
use caustic_core::gpu::GpuContext;
use caustic_core::post::{DebugView, readback_rgba8_texture, write_rgba8_png};
use caustic_core::spectrum::cie_1931_xyz;
use caustic_photons::gpu_sim::{PhotonPipeline, PhotonSettings};
use caustic_photons::scene::{
    ElementKind, EmitterKind, EmitterSpectrum, GlassMaterial, PhotonMode, PhotonScene,
    PhotonSceneId, SceneCategory,
};
use caustic_photons::solar::{
    BODY_EARTH, BODY_EUROPA, BODY_IO, BODY_JUPITER, BODY_MARS, BODY_MERCURY, BODY_MOON,
    BODY_NEPTUNE, BODY_SATURN, BODY_SUN, BODY_URANUS, BODY_VENUS, SUN_RADIUS_AU, SolarFocus,
    SolarScale, format_distance_au, format_duration_s, format_warp,
};

/// Stop stepping new photon frames in `Accumulate` mode once converged to save GPU/power.
const MAX_CONVERGED_FRAMES: u64 = 256;

/// Active engine / named mode inside the unified studio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioMode {
    BlackHole3D,
    Photons(PhotonSceneId),
}

impl StudioMode {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::BlackHole3D => "Kerr Black Hole",
            Self::Photons(id) => id.display_name(),
        }
    }

    pub fn category_badge(self) -> &'static str {
        match self {
            Self::BlackHole3D => "3D RELATIVITY",
            Self::Photons(PhotonSceneId::SolarSystem) => "SOLAR SYSTEM",
            Self::Photons(PhotonSceneId::Sandbox) => "PLAYGROUND",
            Self::Photons(id) => match id.category() {
                SceneCategory::Optics => "OPTICAL BENCH",
                SceneCategory::Grin => "GRIN OPTICS",
                SceneCategory::Relativity => "2D RELATIVITY",
            },
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::BlackHole3D => "blackhole-3d",
            Self::Photons(id) => id.name(),
        }
    }
}

/// Initial launch configuration for [`run_studio`].
pub struct StudioConfig {
    pub initial_mode: StudioMode,
    pub camera: Camera,
    pub bh_params: KerrParams,
    pub bh_settings: RenderSettings,
    pub photon_scene: PhotonScene,
    pub photon_settings: PhotonSettings,
    pub width: u32,
    pub height: u32,
    pub force_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selection {
    Emitter(usize),
    Element(usize),
}

#[derive(Debug, Clone, Copy)]
enum CanvasDrag {
    Orbit3D,
    SolarPan,
    EmitterMove { idx: usize },
    EmitterAim { idx: usize },
    ElementMove { idx: usize, offset: [f32; 2] },
    ElementRotate { idx: usize },
}

struct StudioGfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    bh_pipeline: BlackHolePipeline,
    photon_pipeline: PhotonPipeline,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: EguiRenderer,
}

struct StudioApp {
    mode: StudioMode,
    // 3D Kerr black hole state
    camera: Camera,
    bh_params: KerrParams,
    bh_settings: RenderSettings,
    init_camera: Camera,
    init_bh_params: KerrParams,
    init_bh_settings: RenderSettings,
    bh_trace_dirty: bool,
    bh_shade_dirty: bool,
    bh_full_res: bool,
    bh_last_input: Instant,

    // 2D Forward photon playground state
    photon_scene: PhotonScene,
    photon_settings: PhotonSettings,
    selection: Selection,
    orbit_paused: bool,
    photon_last_edit: Instant,
    photon_was_dragging: bool,

    // Shared window / UI / input state
    init_width: u32,
    init_height: u32,
    force_fallback: bool,
    gfx: Option<StudioGfx>,
    show_ui: bool,
    show_inspector: bool,
    drag: Option<CanvasDrag>,
    last_cursor: Option<(f64, f64)>,
    modifiers: ModifiersState,
    last_frame_instant: Instant,
    smoothed_fps: f32,
    smoothed_ms: f32,
    last_title_update: Instant,
    shot_counter: u32,
    status_toast: Option<(String, Instant)>,
    init_error: Option<anyhow::Error>,
}

impl StudioApp {
    fn new(cfg: StudioConfig) -> Self {
        let now = Instant::now();
        Self {
            mode: cfg.initial_mode,
            camera: cfg.camera,
            bh_params: cfg.bh_params,
            bh_settings: cfg.bh_settings,
            init_camera: cfg.camera,
            init_bh_params: cfg.bh_params,
            init_bh_settings: cfg.bh_settings,
            bh_trace_dirty: true,
            bh_shade_dirty: true,
            bh_full_res: true,
            bh_last_input: now,
            photon_scene: cfg.photon_scene,
            photon_settings: cfg.photon_settings,
            selection: Selection::Emitter(0),
            orbit_paused: false,
            photon_last_edit: now - Duration::from_secs(1),
            photon_was_dragging: false,
            init_width: cfg.width,
            init_height: cfg.height,
            force_fallback: cfg.force_fallback,
            gfx: None,
            show_ui: true,
            show_inspector: true,
            drag: None,
            last_cursor: None,
            modifiers: ModifiersState::empty(),
            last_frame_instant: now,
            smoothed_fps: 60.0,
            smoothed_ms: 16.0,
            last_title_update: now,
            shot_counter: 0,
            status_toast: None,
            init_error: None,
        }
    }

    fn active_emitter_idx(&self) -> usize {
        match self.selection {
            Selection::Emitter(i) if i < self.photon_scene.emitters.len() => i,
            _ => 0,
        }
    }

    fn viewport_aspect(&self) -> f64 {
        self.gfx
            .as_ref()
            .map(|g| {
                f64::from(g.surface_config.width.max(1)) / f64::from(g.surface_config.height.max(1))
            })
            .unwrap_or(16.0 / 9.0)
    }

    fn sync_solar_scene(&mut self, clear_accum: bool, blend: bool) {
        if !self.photon_scene.id.is_solar_system() {
            return;
        }
        let aspect = self.viewport_aspect();
        self.photon_scene.sync_solar_to_gpu(aspect);
        // When zoomed out to map overview (System / Inner), the Sun sits still at the origin and
        // sub-pixel planets move only on the KSP/KSA HUD overlay, so keep accumulating smoothly!
        let static_gpu_map = !clear_accum
            && self.photon_scene.circles.is_empty()
            && self.photon_scene.segments.is_empty()
            && self
                .photon_scene
                .last_solar_frame
                .as_ref()
                .is_some_and(|sf| !sf.far_sun_mode);
        if let Some(gfx) = &mut self.gfx {
            gfx.photon_pipeline
                .upload_scene(&gfx.gpu, &self.photon_scene);
            if self.photon_settings.mode == PhotonMode::Accumulate && !static_gpu_map {
                if clear_accum {
                    gfx.photon_pipeline.clear_accumulation();
                } else if blend {
                    gfx.photon_pipeline.blend_accumulation();
                }
            }
        }
    }

    fn switch_mode(&mut self, next: StudioMode) {
        self.mode = next;
        self.drag = None;
        self.selection = Selection::Emitter(0);
        self.orbit_paused = false;
        match next {
            StudioMode::BlackHole3D => {
                self.bh_trace_dirty = true;
                self.bh_shade_dirty = true;
                self.bh_full_res = true;
            }
            StudioMode::Photons(id) => {
                self.photon_scene = PhotonScene::preset(id);
                if id.is_solar_system() {
                    let aspect = self.viewport_aspect();
                    self.photon_scene.sync_solar_to_gpu(aspect);
                }
                if let Some(gfx) = &mut self.gfx {
                    gfx.photon_pipeline
                        .upload_scene(&gfx.gpu, &self.photon_scene);
                    gfx.photon_pipeline.reset_accumulation();
                }
            }
        }
    }

    fn mark_photon_edited(&mut self, reupload_geometry: bool, interactive: bool) {
        if reupload_geometry {
            self.photon_scene.compile_elements();
        }
        if let Some(gfx) = &mut self.gfx {
            if reupload_geometry {
                gfx.photon_pipeline
                    .upload_scene(&gfx.gpu, &self.photon_scene);
            }
            if self.photon_settings.mode == PhotonMode::Accumulate {
                gfx.photon_pipeline.clear_accumulation();
            }
        }
        if interactive {
            self.photon_last_edit = Instant::now();
        }
    }

    fn cursor_to_world(&self, px: f64, py: f64) -> Option<[f32; 2]> {
        let gfx = self.gfx.as_ref()?;
        let w = f64::from(gfx.surface_config.width.max(1));
        let h = f64::from(gfx.surface_config.height.max(1));
        let aspect = (w / h) as f32;
        let half_h = self.photon_scene.view_half_height;
        let half_w = half_h * aspect;
        let u = (px / w) as f32;
        let v = (py / h) as f32;
        let wx = self.photon_scene.view_center[0] + (u * 2.0 - 1.0) * half_w;
        let wy = self.photon_scene.view_center[1] + (1.0 - v * 2.0) * half_h;
        Some([wx, wy])
    }

    fn world_to_egui(
        world: [f32; 2],
        scene: &PhotonScene,
        width_px: u32,
        height_px: u32,
        ppp: f32,
    ) -> Pos2 {
        let fw = width_px.max(1) as f32;
        let fh = height_px.max(1) as f32;
        let aspect = fw / fh;
        let half_h = scene.view_half_height;
        let half_w = half_h * aspect;
        let u = (world[0] - scene.view_center[0]) / half_w * 0.5 + 0.5;
        let v = 0.5 - (world[1] - scene.view_center[1]) / half_h * 0.5;
        Pos2::new((u * fw) / ppp, (v * fh) / ppp)
    }

    fn save_screenshot(&mut self) {
        let Some(gfx) = &mut self.gfx else {
            return;
        };
        self.shot_counter += 1;
        let path = PathBuf::from(format!(
            "caustic-{}-{:03}.png",
            self.mode.slug(),
            self.shot_counter
        ));
        let w = gfx.surface_config.width.max(1);
        let h = gfx.surface_config.height.max(1);
        let tex = gfx.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("studio-shot-tex"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("studio-shot-enc"),
            });
        match self.mode {
            StudioMode::BlackHole3D => {
                gfx.bh_pipeline.ensure_size(&gfx.gpu, w, h);
                gfx.bh_pipeline.encode_frame(
                    &gfx.gpu,
                    &mut enc,
                    &view,
                    FrameView {
                        camera: &self.camera,
                        params: &self.bh_params,
                        settings: &self.bh_settings,
                        display_aspect: f64::from(w) / f64::from(h),
                    },
                    false,
                );
            }
            StudioMode::Photons(_) => {
                gfx.photon_pipeline.ensure_size(&gfx.gpu, w, h);
                gfx.photon_pipeline
                    .encode_present_only(&gfx.gpu, &mut enc, &view);
            }
        }
        gfx.gpu.queue.submit(Some(enc.finish()));
        match readback_rgba8_texture(&gfx.gpu, &tex, w, h)
            .and_then(|rgba| write_rgba8_png(&path, w, h, &rgba))
        {
            Ok(()) => {
                let msg = format!("Saved {}", path.display());
                log::info!("{msg}");
                self.status_toast = Some((msg, Instant::now()));
            }
            Err(err) => {
                let msg = format!("Screenshot failed: {err}");
                log::error!("{msg}");
                self.status_toast = Some((msg, Instant::now()));
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::KeyH | KeyCode::Tab => {
                self.show_ui = !self.show_ui;
            }
            KeyCode::KeyP
                if matches!(self.mode, StudioMode::Photons(PhotonSceneId::SolarSystem)) =>
            {
                self.orbit_paused = !self.orbit_paused;
                if self.orbit_paused {
                    self.mark_photon_edited(false, false);
                }
            }
            KeyCode::Digit0 | KeyCode::Backquote => self.switch_mode(StudioMode::BlackHole3D),
            KeyCode::Digit1 => self.switch_mode(StudioMode::Photons(PhotonSceneId::Prism)),
            KeyCode::Digit2 => self.switch_mode(StudioMode::Photons(PhotonSceneId::Cardioid)),
            KeyCode::Digit3 => self.switch_mode(StudioMode::Photons(PhotonSceneId::Lens)),
            KeyCode::Digit4 => self.switch_mode(StudioMode::Photons(PhotonSceneId::SolarSystem)),
            KeyCode::Digit5 => self.switch_mode(StudioMode::Photons(PhotonSceneId::Sandbox)),
            KeyCode::Digit6 => self.switch_mode(StudioMode::Photons(PhotonSceneId::GrinLuneburg)),
            KeyCode::Digit7 => self.switch_mode(StudioMode::Photons(PhotonSceneId::GrinBlackHole)),
            KeyCode::Digit8 => self.switch_mode(StudioMode::Photons(PhotonSceneId::GrinMirage)),
            KeyCode::Digit9 => self.switch_mode(StudioMode::Photons(PhotonSceneId::KerrBeam)),
            KeyCode::KeyS => self.save_screenshot(),
            KeyCode::KeyR => match self.mode {
                StudioMode::BlackHole3D => {
                    self.camera = self.init_camera;
                    self.bh_params = self.init_bh_params;
                    self.bh_settings = self.init_bh_settings;
                    self.bh_trace_dirty = true;
                    self.bh_shade_dirty = true;
                    self.bh_full_res = true;
                }
                StudioMode::Photons(id) => self.switch_mode(StudioMode::Photons(id)),
            },
            KeyCode::Equal => match self.mode {
                StudioMode::BlackHole3D => {
                    self.bh_settings.exposure = (self.bh_settings.exposure * 1.2).min(16.0);
                    self.bh_shade_dirty = true;
                }
                StudioMode::Photons(_) => {
                    self.photon_settings.exposure = (self.photon_settings.exposure * 1.2).min(16.0);
                    self.mark_photon_edited(false, false);
                }
            },
            KeyCode::Minus => match self.mode {
                StudioMode::BlackHole3D => {
                    self.bh_settings.exposure = (self.bh_settings.exposure / 1.2).max(0.05);
                    self.bh_shade_dirty = true;
                }
                StudioMode::Photons(_) => {
                    self.photon_settings.exposure = (self.photon_settings.exposure / 1.2).max(0.05);
                    self.mark_photon_edited(false, false);
                }
            },
            KeyCode::KeyM if matches!(self.mode, StudioMode::Photons(_)) => {
                self.photon_settings.mode = match self.photon_settings.mode {
                    PhotonMode::Accumulate => PhotonMode::Pulse,
                    PhotonMode::Pulse => PhotonMode::Accumulate,
                };
                if let Some(gfx) = &mut self.gfx {
                    gfx.photon_pipeline.reset_accumulation();
                }
            }
            KeyCode::Space if matches!(self.mode, StudioMode::Photons(_)) => {
                if let Some(gfx) = &mut self.gfx {
                    match self.photon_settings.mode {
                        PhotonMode::Pulse => gfx.photon_pipeline.trigger_pulse(),
                        PhotonMode::Accumulate => gfx.photon_pipeline.clear_accumulation(),
                    }
                }
            }
            KeyCode::KeyQ if matches!(self.mode, StudioMode::Photons(_)) => match self.selection {
                Selection::Element(idx) if idx < self.photon_scene.elements.len() => {
                    self.photon_scene.elements[idx].angle += 0.08;
                    self.mark_photon_edited(true, true);
                }
                Selection::Emitter(idx) if idx < self.photon_scene.emitters.len() => {
                    self.photon_scene.emitters[idx].angle += 0.06;
                    self.mark_photon_edited(false, true);
                }
                _ => {}
            },
            KeyCode::KeyE if matches!(self.mode, StudioMode::Photons(_)) => match self.selection {
                Selection::Element(idx) if idx < self.photon_scene.elements.len() => {
                    self.photon_scene.elements[idx].angle -= 0.08;
                    self.mark_photon_edited(true, true);
                }
                Selection::Emitter(idx) if idx < self.photon_scene.emitters.len() => {
                    self.photon_scene.emitters[idx].angle -= 0.06;
                    self.mark_photon_edited(false, true);
                }
                _ => {}
            },
            KeyCode::Delete | KeyCode::Backspace if matches!(self.mode, StudioMode::Photons(_)) => {
                match self.selection {
                    Selection::Element(idx) if idx < self.photon_scene.elements.len() => {
                        self.photon_scene.remove_element(idx);
                        self.selection = Selection::Emitter(0);
                        self.mark_photon_edited(true, false);
                    }
                    Selection::Emitter(idx) if self.photon_scene.emitters.len() > 1 => {
                        self.photon_scene.remove_emitter(idx);
                        self.selection = Selection::Emitter(0);
                        self.mark_photon_edited(false, false);
                    }
                    _ => {}
                }
            }
            KeyCode::BracketRight => match self.mode {
                StudioMode::BlackHole3D => {
                    self.bh_params.spin = (self.bh_params.spin + 0.05).clamp(-MAX_SPIN, MAX_SPIN);
                    self.bh_trace_dirty = true;
                }
                StudioMode::Photons(id) => {
                    if id.is_solar_system() {
                        let (lo, hi) = if self.photon_scene.solar.scale.is_orrery() {
                            (0.1, 16.0)
                        } else {
                            (1.0, 5_000_000.0)
                        };
                        self.photon_scene.solar.time_warp =
                            (self.photon_scene.solar.time_warp * 2.0).clamp(lo, hi);
                    } else if id.is_kerr() {
                        self.photon_scene.spin =
                            (self.photon_scene.spin + 0.05).clamp(-0.998, 0.998);
                        self.mark_photon_edited(false, true);
                    } else {
                        self.photon_scene.dispersion_scale =
                            (self.photon_scene.dispersion_scale + 0.15).min(3.0);
                        self.mark_photon_edited(id.is_optics(), true);
                    }
                }
            },
            KeyCode::BracketLeft => match self.mode {
                StudioMode::BlackHole3D => {
                    self.bh_params.spin = (self.bh_params.spin - 0.05).clamp(-MAX_SPIN, MAX_SPIN);
                    self.bh_trace_dirty = true;
                }
                StudioMode::Photons(id) => {
                    if id.is_solar_system() {
                        let (lo, hi) = if self.photon_scene.solar.scale.is_orrery() {
                            (0.1, 16.0)
                        } else {
                            (1.0, 5_000_000.0)
                        };
                        self.photon_scene.solar.time_warp =
                            (self.photon_scene.solar.time_warp * 0.5).clamp(lo, hi);
                    } else if id.is_kerr() {
                        self.photon_scene.spin =
                            (self.photon_scene.spin - 0.05).clamp(-0.998, 0.998);
                        self.mark_photon_edited(false, true);
                    } else {
                        self.photon_scene.dispersion_scale =
                            (self.photon_scene.dispersion_scale - 0.15).max(0.0);
                        self.mark_photon_edited(id.is_optics(), true);
                    }
                }
            },
            KeyCode::KeyD if self.mode == StudioMode::BlackHole3D => {
                self.bh_settings.disk_enabled = !self.bh_settings.disk_enabled;
                self.bh_trace_dirty = true;
            }
            KeyCode::KeyC if matches!(self.mode, StudioMode::Photons(_)) => {
                self.mark_photon_edited(false, false);
            }
            _ => {}
        }
    }

    fn build_ui(
        &mut self,
        ui: &mut egui::Ui,
        width_px: u32,
        height_px: u32,
        ppp: f32,
        accum_frames: u64,
    ) -> UiActions {
        let mut actions = UiActions::default();
        let ctx = ui.ctx().clone();

        if !self.show_ui {
            return actions;
        }

        // 1. Subtle world-space handles & KSP/KSA Solar System orbit guides / reticles.
        if let StudioMode::Photons(scene_id) = self.mode {
            let painter = ctx.layer_painter(LayerId::background());
            if scene_id.is_solar_system() {
                if let Some(sf) = &self.photon_scene.last_solar_frame {
                    let solar = &self.photon_scene.solar;
                    let half_h_au = sf.view_half_height_au.max(1e-9);
                    let inv_s = 1.0 / half_h_au;
                    let c_au = sf.view_center_au;
                    let aspect = (width_px.max(1) as f64) / (height_px.max(1) as f64);
                    let px_per_gpu = (height_px.max(1) as f32) * 0.5 / ppp;

                    // Crisp Keplerian orbit ellipses (planets in map view, moons when zoomed near host).
                    for (i, b) in solar.bodies.iter().enumerate().skip(1) {
                        let a_gpu = b.semi_major_au * inv_s;
                        let a_px = (a_gpu as f32) * px_per_gpu;
                        if !(12.0..=12_000.0).contains(&a_px) {
                            continue;
                        }
                        let is_focused = solar.focus == SolarFocus::Body(i);
                        let alpha = if is_focused { 155 } else { 82 };
                        let col = Color32::from_rgba_unmultiplied(
                            b.map_color[0],
                            b.map_color[1],
                            b.map_color[2],
                            alpha,
                        );
                        let pts = solar.sample_orbit_ellipse_au(i, 96);
                        let screen_pts: Vec<Pos2> = pts
                            .into_iter()
                            .map(|p| {
                                Self::world_to_egui(
                                    [
                                        ((p[0] - c_au[0]) * inv_s) as f32,
                                        ((p[1] - c_au[1]) * inv_s) as f32,
                                    ],
                                    &self.photon_scene,
                                    width_px,
                                    height_px,
                                    ppp,
                                )
                            })
                            .collect();
                        let stroke_w = if is_focused { 1.5 } else { 1.15 };
                        for seg in screen_pts.windows(2) {
                            painter.line_segment([seg[0], seg[1]], Stroke::new(stroke_w, col));
                        }
                    }

                    // KSP/KSA clickable body reticles & clean shadowed labels.
                    for (i, b) in solar.bodies.iter().enumerate() {
                        let Some(&gp) = sf.body_gpu_pos.get(i) else {
                            continue;
                        };
                        if gp[0].abs() > aspect * 1.15 || gp[1].abs() > 1.15 {
                            continue;
                        }
                        // Hide moon markers when zoomed way out so they don't overlap parent planets.
                        if b.parent.is_some() {
                            let orbit_px = ((b.semi_major_au * inv_s) as f32) * px_per_gpu;
                            if orbit_px < 14.0 {
                                continue;
                            }
                        }
                        let pos = Self::world_to_egui(
                            [gp[0] as f32, gp[1] as f32],
                            &self.photon_scene,
                            width_px,
                            height_px,
                            ppp,
                        );
                        let r_px = (sf.body_gpu_rad[i] as f32) * px_per_gpu;
                        let is_focused = solar.focus == SolarFocus::Body(i);
                        let tint =
                            Color32::from_rgb(b.map_color[0], b.map_color[1], b.map_color[2]);

                        if r_px < 8.5 && i != BODY_SUN {
                            painter.circle_filled(
                                pos,
                                6.8,
                                Color32::from_rgba_unmultiplied(8, 10, 15, 145),
                            );
                            painter.circle_filled(pos, if is_focused { 3.2 } else { 2.5 }, tint);
                            painter.circle_stroke(
                                pos,
                                if is_focused { 8.0 } else { 6.5 },
                                Stroke::new(
                                    if is_focused { 1.6 } else { 1.25 },
                                    if is_focused {
                                        Color32::from_rgba_unmultiplied(136, 198, 255, 245)
                                    } else {
                                        Color32::from_rgba_unmultiplied(
                                            b.map_color[0],
                                            b.map_color[1],
                                            b.map_color[2],
                                            220,
                                        )
                                    },
                                ),
                            );
                        } else if is_focused && r_px < 90.0 {
                            painter.circle_stroke(
                                pos,
                                r_px + 5.0,
                                Stroke::new(
                                    1.1,
                                    Color32::from_rgba_unmultiplied(136, 198, 255, 130),
                                ),
                            );
                        }

                        let lbl_pos = pos + Vec2::new(0.0, r_px.max(7.0) + 5.0);
                        painter.text(
                            lbl_pos + Vec2::new(1.0, 1.0),
                            Align2::CENTER_TOP,
                            b.name,
                            FontId::proportional(10.0),
                            Color32::from_rgba_unmultiplied(0, 0, 0, 220),
                        );
                        painter.text(
                            lbl_pos,
                            Align2::CENTER_TOP,
                            b.name,
                            FontId::proportional(10.0),
                            if is_focused {
                                Color32::from_rgb(245, 249, 255)
                            } else {
                                Color32::from_rgba_unmultiplied(222, 232, 246, 205)
                            },
                        );
                    }
                }
            } else {
                for (i, em) in self.photon_scene.emitters.iter().enumerate() {
                    let selected = self.selection == Selection::Emitter(i);
                    let em_pos =
                        Self::world_to_egui(em.pos, &self.photon_scene, width_px, height_px, ppp);
                    let dot_col = spectrum_preview_color(em.spectrum);

                    if em.kind == EmitterKind::Disk {
                        if selected {
                            painter.circle_stroke(
                                em_pos,
                                9.0,
                                Stroke::new(
                                    1.4,
                                    Color32::from_rgba_unmultiplied(128, 190, 255, 190),
                                ),
                            );
                        }
                    } else {
                        let aim = Vec2::new(em.angle.cos(), -em.angle.sin());
                        painter.circle_filled(em_pos, if selected { 4.0 } else { 3.0 }, dot_col);
                        painter.circle_stroke(
                            em_pos,
                            if selected { 7.5 } else { 5.5 },
                            Stroke::new(
                                if selected { 1.5 } else { 1.0 },
                                if selected {
                                    Color32::from_rgba_unmultiplied(128, 190, 255, 220)
                                } else {
                                    Color32::from_rgba_unmultiplied(210, 220, 235, 120)
                                },
                            ),
                        );
                        if matches!(em.kind, EmitterKind::Beam | EmitterKind::Spot) {
                            painter.line_segment(
                                [em_pos + aim * 7.5, em_pos + aim * 17.0],
                                Stroke::new(
                                    1.4,
                                    if selected {
                                        Color32::from_rgba_unmultiplied(128, 190, 255, 210)
                                    } else {
                                        Color32::from_rgba_unmultiplied(210, 220, 235, 110)
                                    },
                                ),
                            );
                        }
                    }
                }

                if let Selection::Element(idx) = self.selection
                    && let Some(el) = self.photon_scene.elements.get(idx)
                {
                    let h_pos = Self::world_to_egui(
                        el.handle_pos(),
                        &self.photon_scene,
                        width_px,
                        height_px,
                        ppp,
                    );
                    let dir = Vec2::new(el.angle.cos(), -el.angle.sin());
                    painter.circle_stroke(
                        h_pos,
                        9.0,
                        Stroke::new(1.6, Color32::from_rgb(122, 186, 255)),
                    );
                    painter.circle_filled(
                        h_pos,
                        3.0,
                        Color32::from_rgba_unmultiplied(122, 186, 255, 220),
                    );
                    if !matches!(el.kind, ElementKind::Planet { ring_outer, .. } if ring_outer <= 0.0)
                    {
                        painter.line_segment(
                            [h_pos + dir * 9.0, h_pos + dir * 20.0],
                            Stroke::new(1.4, Color32::from_rgba_unmultiplied(122, 186, 255, 200)),
                        );
                    }
                }
            }
        }

        let glass_frame = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(11, 13, 18, 214))
            .stroke(Stroke::new(
                1.0,
                Color32::from_rgba_unmultiplied(255, 255, 255, 22),
            ))
            .corner_radius(CornerRadius::same(11))
            .inner_margin(Margin::symmetric(10, 6));

        // 2. Top floating pill bar: brand + all named modes + live telemetry + inspector toggle.
        egui::Area::new(Id::unique("caustic_top_bar"))
            .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 12.0))
            .fade_in(false)
            .show(&ctx, |ui| {
                glass_frame.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        ui.label(
                            RichText::new("caustic")
                                .strong()
                                .size(12.5)
                                .color(Color32::from_rgb(236, 241, 249)),
                        );
                        pill_divider(ui);

                        if mode_chip(ui, "Black Hole 3D", self.mode == StudioMode::BlackHole3D) {
                            actions.switch_mode = Some(StudioMode::BlackHole3D);
                        }
                        pill_divider(ui);

                        for (i, id) in PhotonSceneId::ALL.iter().copied().enumerate() {
                            if i == 5 || i == 8 {
                                pill_divider(ui);
                            }
                            let active = self.mode == StudioMode::Photons(id);
                            if mode_chip(ui, id.short_name(), active) {
                                actions.switch_mode = Some(StudioMode::Photons(id));
                            }
                        }

                        pill_divider(ui);
                        ui.label(
                            RichText::new(format!(
                                "{:.0} fps · {:.1} ms",
                                self.smoothed_fps, self.smoothed_ms
                            ))
                            .monospace()
                            .size(10.5)
                            .color(Color32::from_rgb(138, 150, 170)),
                        );
                        pill_divider(ui);
                        let toggle_lbl = if self.show_inspector {
                            "Hide"
                        } else {
                            "Controls"
                        };
                        if mode_chip(ui, toggle_lbl, self.show_inspector) {
                            self.show_inspector = !self.show_inspector;
                        }
                    });
                });
            });

        // 3. Collapsible right floating inspector.
        if self.show_inspector {
            let panel_frame = Frame::new()
                .fill(Color32::from_rgba_unmultiplied(11, 13, 18, 218))
                .stroke(Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(255, 255, 255, 22),
                ))
                .corner_radius(CornerRadius::same(12))
                .inner_margin(Margin::symmetric(14, 12));

            let max_panel_h = ((height_px as f32) / ppp - 96.0).max(260.0);

            egui::Area::new(Id::unique("caustic_inspector"))
                .anchor(Align2::RIGHT_TOP, Vec2::new(-14.0, 56.0))
                .fade_in(false)
                .show(&ctx, |ui| {
                    panel_frame.show(ui, |ui| {
                        ui.set_width(268.0);
                        egui::ScrollArea::vertical()
                            .max_height(max_panel_h)
                            .min_scrolled_height(max_panel_h)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing.y = 7.0;

                                // Active mode header
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(self.mode.display_name())
                                            .strong()
                                            .size(14.0)
                                            .color(Color32::from_rgb(240, 245, 252)),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                RichText::new(self.mode.category_badge())
                                                    .size(9.5)
                                                    .color(Color32::from_rgb(120, 176, 255)),
                                            );
                                        },
                                    );
                                });
                                subtle_rule(ui);

                                match self.mode {
                                    StudioMode::BlackHole3D => {
                                        let mut spin = self.bh_params.spin as f32;
                                        if labeled_slider(
                                            ui,
                                            "Spin a",
                                            &mut spin,
                                            -0.998..=0.998,
                                            false,
                                        ) {
                                            self.bh_params.spin = f64::from(spin);
                                            actions.bh_trace_dirty = true;
                                        }
                                        let mut disk = self.bh_settings.disk_enabled;
                                        if ui.checkbox(&mut disk, "Accretion Disk").changed() {
                                            self.bh_settings.disk_enabled = disk;
                                            actions.bh_trace_dirty = true;
                                        }
                                        if self.bh_settings.disk_enabled
                                            && labeled_slider(
                                                ui,
                                                "Peak Temp (K)",
                                                &mut self.bh_settings.disk_peak_temp_k,
                                                2500.0..=12000.0,
                                                false,
                                            )
                                        {
                                            actions.bh_shade_dirty = true;
                                        }
                                        if labeled_slider(
                                            ui,
                                            "Exposure",
                                            &mut self.bh_settings.exposure,
                                            0.15..=6.0,
                                            true,
                                        ) {
                                            actions.bh_shade_dirty = true;
                                        }

                                        section_label(ui, "VIEW");
                                        ui.horizontal_wrapped(|ui| {
                                            for (lbl, dv) in [
                                                ("Beauty", DebugView::Beauty),
                                                ("Steps", DebugView::StepHeatmap),
                                                ("Redshift", DebugView::Redshift),
                                                ("Hits", DebugView::HitClass),
                                            ] {
                                                if mode_chip(
                                                    ui,
                                                    lbl,
                                                    self.bh_settings.debug_view == dv,
                                                ) {
                                                    self.bh_settings.debug_view = dv;
                                                    actions.bh_shade_dirty = true;
                                                }
                                            }
                                        });
                                    }
                                    StudioMode::Photons(id) => {
                                        // Solar System controls: Real (1:1) exact NASA ratios vs exaggerated Orrery scale
                                        if id.is_solar_system() {
                                            ui.spacing_mut().item_spacing.y = 4.5;
                                            let is_orrery =
                                                self.photon_scene.solar.scale.is_orrery();
                                            ui.horizontal(|ui| {
                                                for sc in [SolarScale::Real1To1, SolarScale::Orrery]
                                                {
                                                    if mode_chip(
                                                        ui,
                                                        sc.label(),
                                                        self.photon_scene.solar.scale == sc,
                                                    ) && self.photon_scene.solar.scale != sc
                                                    {
                                                        self.photon_scene.solar.set_scale(sc);
                                                        actions.solar_changed = true;
                                                    }
                                                }
                                            });

                                            section_label(ui, "EVENTS");
                                            ui.horizontal_wrapped(|ui| {
                                                ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
                                                if ui.small_button("Lunar Eclipse").clicked() {
                                                    self.photon_scene.solar.stage_lunar_eclipse();
                                                    actions.solar_changed = true;
                                                }
                                                if ui.small_button("Solar Eclipse").clicked() {
                                                    self.photon_scene.solar.stage_solar_eclipse();
                                                    actions.solar_changed = true;
                                                }
                                                if ui.small_button("Saturn Rings").clicked() {
                                                    self.photon_scene.solar.stage_saturn_rings();
                                                    actions.solar_changed = true;
                                                }
                                                if ui.small_button("Io Transit").clicked() {
                                                    self.photon_scene
                                                        .solar
                                                        .stage_jupiter_io_transit();
                                                    actions.solar_changed = true;
                                                }
                                            });

                                            section_label(ui, "FOCUS");
                                            ui.horizontal_wrapped(|ui| {
                                                ui.spacing_mut().item_spacing = Vec2::new(2.5, 2.5);
                                                ui.spacing_mut().button_padding =
                                                    Vec2::new(5.0, 2.2);
                                                let targets = [
                                                    ("System", SolarFocus::System),
                                                    ("Inner", SolarFocus::Inner),
                                                    ("Sun", SolarFocus::Body(BODY_SUN)),
                                                    ("Mercury", SolarFocus::Body(BODY_MERCURY)),
                                                    ("Venus", SolarFocus::Body(BODY_VENUS)),
                                                    ("Earth", SolarFocus::Body(BODY_EARTH)),
                                                    ("Moon", SolarFocus::Body(BODY_MOON)),
                                                    ("Mars", SolarFocus::Body(BODY_MARS)),
                                                    ("Jupiter", SolarFocus::Body(BODY_JUPITER)),
                                                    ("Io", SolarFocus::Body(BODY_IO)),
                                                    ("Europa", SolarFocus::Body(BODY_EUROPA)),
                                                    ("Saturn", SolarFocus::Body(BODY_SATURN)),
                                                    ("Uranus", SolarFocus::Body(BODY_URANUS)),
                                                    ("Neptune", SolarFocus::Body(BODY_NEPTUNE)),
                                                ];
                                                for (lbl, tgt) in targets {
                                                    let active =
                                                        self.photon_scene.solar.focus == tgt;
                                                    if mode_chip(ui, lbl, active) {
                                                        self.photon_scene.solar.set_focus(tgt);
                                                        actions.solar_changed = true;
                                                    }
                                                }
                                            });

                                            section_label(ui, "TIME WARP");
                                            ui.horizontal(|ui| {
                                                let lbl = if self.orbit_paused {
                                                    "Resume [P]"
                                                } else {
                                                    "Pause [P]"
                                                };
                                                if mode_chip(ui, lbl, !self.orbit_paused) {
                                                    self.orbit_paused = !self.orbit_paused;
                                                    if self.orbit_paused {
                                                        actions.solar_changed = true;
                                                    }
                                                }
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format_warp(
                                                                self.photon_scene.solar.time_warp,
                                                                self.photon_scene.solar.scale,
                                                            ))
                                                            .monospace()
                                                            .size(10.5)
                                                            .color(Color32::from_rgb(
                                                                138, 196, 255,
                                                            )),
                                                        );
                                                    },
                                                );
                                            });
                                            ui.horizontal_wrapped(|ui| {
                                                ui.spacing_mut().item_spacing = Vec2::new(2.5, 2.5);
                                                ui.spacing_mut().button_padding =
                                                    Vec2::new(5.0, 2.2);
                                                let presets: &[(&str, f64)] = if is_orrery {
                                                    &[
                                                        ("0.25x", 0.25),
                                                        ("0.5x", 0.5),
                                                        ("1x", 1.0),
                                                        ("2x", 2.0),
                                                        ("5x", 5.0),
                                                    ]
                                                } else {
                                                    &[
                                                        ("1x", 1.0),
                                                        ("1m", 60.0),
                                                        ("15m", 900.0),
                                                        ("1h", 3_600.0),
                                                        ("1d", 86_400.0),
                                                        ("30d", 30.0 * 86_400.0),
                                                    ]
                                                };
                                                for &(lbl, wv) in presets {
                                                    let sel = (self.photon_scene.solar.time_warp
                                                        - wv)
                                                        .abs()
                                                        / wv.max(0.01)
                                                        < 0.12;
                                                    if mode_chip(ui, lbl, sel) {
                                                        self.photon_scene.solar.time_warp = wv;
                                                    }
                                                }
                                            });

                                            if let Some(sf) = &self.photon_scene.last_solar_frame {
                                                card_frame().show(ui, |ui| {
                                                    ui.spacing_mut().item_spacing.y = 2.5;
                                                    let rows: Vec<(&str, String)> = if is_orrery {
                                                        vec![
                                                            ("Scale", "Orrery (Cartoon)".into()),
                                                            (
                                                                "Bodies",
                                                                "Sun + 8 Planets + 3 Moons".into(),
                                                            ),
                                                        ]
                                                    } else {
                                                        vec![
                                                            (
                                                                "View Height",
                                                                format_distance_au(
                                                                    2.0 * sf.view_half_height_au,
                                                                ),
                                                            ),
                                                            (
                                                                "Sun Dist",
                                                                format_distance_au(sf.sun_dist_au),
                                                            ),
                                                            (
                                                                "Sun Angular Ø",
                                                                if sf.sun_dist_au
                                                                    <= SUN_RADIUS_AU * 1.05
                                                                {
                                                                    "—".to_string()
                                                                } else {
                                                                    format!(
                                                                        "{:.4}°",
                                                                        sf.sun_angular_diam_deg
                                                                    )
                                                                },
                                                            ),
                                                            (
                                                                "Light Time",
                                                                format_duration_s(sf.light_time_s),
                                                            ),
                                                        ]
                                                    };
                                                    for (k, v) in rows {
                                                        ui.horizontal(|ui| {
                                                            ui.label(
                                                                RichText::new(k).size(10.0).color(
                                                                    Color32::from_rgb(
                                                                        146, 158, 178,
                                                                    ),
                                                                ),
                                                            );
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    ui.label(
                                                                        RichText::new(v)
                                                                            .monospace()
                                                                            .size(10.0)
                                                                            .color(
                                                                                Color32::from_rgb(
                                                                                    228, 236, 248,
                                                                                ),
                                                                            ),
                                                                    );
                                                                },
                                                            );
                                                        });
                                                    }
                                                });
                                            }
                                            subtle_rule(ui);
                                        }

                                        // Transport mode: Steady vs Wavefront
                                        section_label(ui, "TRANSPORT");
                                        ui.horizontal(|ui| {
                                            if mode_chip(
                                                ui,
                                                "Steady",
                                                self.photon_settings.mode == PhotonMode::Accumulate,
                                            ) && self.photon_settings.mode
                                                != PhotonMode::Accumulate
                                            {
                                                self.photon_settings.mode = PhotonMode::Accumulate;
                                                actions.photon_reset_accum = true;
                                            }
                                            if mode_chip(
                                                ui,
                                                "Wavefront",
                                                self.photon_settings.mode == PhotonMode::Pulse,
                                            ) && self.photon_settings.mode != PhotonMode::Pulse
                                            {
                                                self.photon_settings.mode = PhotonMode::Pulse;
                                                actions.photon_reset_accum = true;
                                            }
                                            if self.photon_settings.mode == PhotonMode::Pulse {
                                                if ui.button("Pulse").clicked() {
                                                    actions.fire_pulse = true;
                                                }
                                            } else {
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        let pct = ((accum_frames
                                                            .min(MAX_CONVERGED_FRAMES)
                                                            * 100)
                                                            / MAX_CONVERGED_FRAMES)
                                                            as u32;
                                                        ui.label(
                                                            RichText::new(format!("{pct}%"))
                                                                .monospace()
                                                                .size(10.5)
                                                                .color(Color32::from_rgb(
                                                                    122, 134, 152,
                                                                )),
                                                        );
                                                    },
                                                );
                                            }
                                        });

                                        if self.photon_settings.mode == PhotonMode::Pulse {
                                            labeled_slider(
                                                ui,
                                                "Trail",
                                                &mut self.photon_settings.pulse_decay,
                                                0.65..=0.96,
                                                false,
                                            );
                                        }

                                        if id.is_solar_system() {
                                            if labeled_slider(
                                                ui,
                                                "Exposure",
                                                &mut self.photon_settings.exposure,
                                                0.15..=4.0,
                                                true,
                                            ) {
                                                actions.photon_edited = true;
                                            }
                                        } else {
                                            // Multi-emitter controls
                                            subtle_rule(ui);
                                            section_label(ui, "EMITTERS");
                                            ui.horizontal_wrapped(|ui| {
                                                let spawn_pos = [
                                                    self.photon_scene.view_center[0]
                                                        - self.photon_scene.view_half_height * 0.8,
                                                    self.photon_scene.view_center[1],
                                                ];
                                                if ui.small_button("+ Beam").clicked() {
                                                    let idx = self.photon_scene.add_emitter(
                                                        EmitterKind::Beam,
                                                        EmitterSpectrum::WhiteFlat,
                                                        spawn_pos,
                                                        0.0,
                                                        0.28,
                                                        1.0,
                                                    );
                                                    self.selection = Selection::Emitter(idx);
                                                    actions.photon_edited = true;
                                                }
                                                if ui.small_button("+ Laser").clicked() {
                                                    let idx = self.photon_scene.add_emitter(
                                                        EmitterKind::Beam,
                                                        EmitterSpectrum::Laser(532.0),
                                                        [spawn_pos[0], spawn_pos[1] - 1.2],
                                                        0.15,
                                                        0.08,
                                                        0.6,
                                                    );
                                                    self.selection = Selection::Emitter(idx);
                                                    actions.photon_edited = true;
                                                }
                                                if ui.small_button("+ Star").clicked() {
                                                    let idx = self.photon_scene.add_emitter(
                                                        EmitterKind::Disk,
                                                        EmitterSpectrum::Blackbody(5778.0),
                                                        self.photon_scene.view_center,
                                                        0.0,
                                                        0.65,
                                                        1.0,
                                                    );
                                                    self.selection = Selection::Emitter(idx);
                                                    actions.photon_edited = true;
                                                }
                                                if ui.small_button("+ Spot").clicked() {
                                                    let idx = self.photon_scene.add_emitter(
                                                        EmitterKind::Spot,
                                                        EmitterSpectrum::Blackbody(5778.0),
                                                        spawn_pos,
                                                        0.0,
                                                        0.35,
                                                        1.0,
                                                    );
                                                    self.selection = Selection::Emitter(idx);
                                                    actions.photon_edited = true;
                                                }
                                            });

                                            // Selected emitter editor card
                                            if let Selection::Emitter(eidx) = self.selection
                                                && eidx < self.photon_scene.emitters.len()
                                            {
                                                let mut del_em = false;
                                                let can_del = self.photon_scene.emitters.len() > 1;
                                                let em = &mut self.photon_scene.emitters[eidx];
                                                let swatch = spectrum_preview_color(em.spectrum);

                                                card_frame().show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        let (r, _) = ui.allocate_exact_size(
                                                            Vec2::splat(9.0),
                                                            egui::Sense::hover(),
                                                        );
                                                        ui.painter().circle_filled(
                                                            r.center(),
                                                            4.5,
                                                            swatch,
                                                        );
                                                        ui.label(
                                                        RichText::new(format!(
                                                            "Emitter #{} ({})",
                                                            eidx + 1,
                                                            em.kind.label()
                                                        ))
                                                        .strong()
                                                        .size(11.5)
                                                        .color(Color32::from_rgb(224, 234, 248)),
                                                    );
                                                        if can_del {
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if ui
                                                                        .small_button("Delete")
                                                                        .clicked()
                                                                    {
                                                                        del_em = true;
                                                                    }
                                                                },
                                                            );
                                                        }
                                                    });

                                                    // Shape chips
                                                    ui.horizontal_wrapped(|ui| {
                                                        for k in [
                                                            EmitterKind::Beam,
                                                            EmitterKind::Point,
                                                            EmitterKind::Spot,
                                                            EmitterKind::Disk,
                                                        ] {
                                                            if mode_chip(
                                                                ui,
                                                                k.label(),
                                                                em.kind == k,
                                                            ) && em.kind != k
                                                            {
                                                                em.kind = k;
                                                                em.spread = match k {
                                                                    EmitterKind::Beam => {
                                                                        if id.is_kerr() {
                                                                            8.0
                                                                        } else {
                                                                            0.45
                                                                        }
                                                                    }
                                                                    EmitterKind::Point => {
                                                                        std::f32::consts::PI
                                                                    }
                                                                    EmitterKind::Spot => 0.35,
                                                                    EmitterKind::Disk => 0.85,
                                                                };
                                                                actions.photon_edited = true;
                                                            }
                                                        }
                                                    });

                                                    // Spectrum chips (hidden on Kerr where frequency shift owns color)
                                                    if !id.is_kerr() {
                                                        ui.horizontal_wrapped(|ui| {
                                                            let is_white = matches!(
                                                                em.spectrum,
                                                                EmitterSpectrum::WhiteFlat
                                                            );
                                                            let is_bb = matches!(
                                                                em.spectrum,
                                                                EmitterSpectrum::Blackbody(_)
                                                            );
                                                            let is_laser = matches!(
                                                                em.spectrum,
                                                                EmitterSpectrum::Laser(_)
                                                            );
                                                            if mode_chip(ui, "White", is_white)
                                                                && !is_white
                                                            {
                                                                em.spectrum =
                                                                    EmitterSpectrum::WhiteFlat;
                                                                actions.photon_edited = true;
                                                            }
                                                            if mode_chip(ui, "Star / Sun", is_bb)
                                                                && !is_bb
                                                            {
                                                                em.spectrum =
                                                                    EmitterSpectrum::Blackbody(
                                                                        5778.0,
                                                                    );
                                                                actions.photon_edited = true;
                                                            }
                                                            if mode_chip(ui, "Laser", is_laser)
                                                                && !is_laser
                                                            {
                                                                em.spectrum =
                                                                    EmitterSpectrum::Laser(532.0);
                                                                actions.photon_edited = true;
                                                            }
                                                        });

                                                        match &mut em.spectrum {
                                                            EmitterSpectrum::Blackbody(temp) => {
                                                                if labeled_slider(
                                                                    ui,
                                                                    "Temp (K)",
                                                                    temp,
                                                                    2000.0..=15000.0,
                                                                    true,
                                                                ) {
                                                                    actions.photon_edited = true;
                                                                }
                                                            }
                                                            EmitterSpectrum::Laser(nm) => {
                                                                if labeled_slider(
                                                                    ui,
                                                                    "Color (nm)",
                                                                    nm,
                                                                    400.0..=700.0,
                                                                    false,
                                                                ) {
                                                                    actions.photon_edited = true;
                                                                }
                                                            }
                                                            EmitterSpectrum::WhiteFlat => {}
                                                        }
                                                    }

                                                    match em.kind {
                                                        EmitterKind::Beam => {
                                                            let max_w = if id.is_kerr() {
                                                                16.0
                                                            } else {
                                                                4.8
                                                            };
                                                            if labeled_slider(
                                                                ui,
                                                                "Width",
                                                                &mut em.spread,
                                                                0.03..=max_w,
                                                                false,
                                                            ) {
                                                                actions.photon_edited = true;
                                                            }
                                                        }
                                                        EmitterKind::Spot => {
                                                            let mut deg = em.spread.to_degrees();
                                                            if labeled_slider(
                                                                ui,
                                                                "Cone",
                                                                &mut deg,
                                                                2.0..=170.0,
                                                                false,
                                                            ) {
                                                                em.spread = deg.to_radians();
                                                                actions.photon_edited = true;
                                                            }
                                                        }
                                                        EmitterKind::Disk => {
                                                            if labeled_slider(
                                                                ui,
                                                                "Radius",
                                                                &mut em.spread,
                                                                0.15..=2.5,
                                                                false,
                                                            ) {
                                                                actions.photon_edited = true;
                                                            }
                                                        }
                                                        EmitterKind::Point => {}
                                                    }

                                                    if can_del
                                                        && labeled_slider(
                                                            ui,
                                                            "Power",
                                                            &mut em.power,
                                                            0.1..=3.0,
                                                            false,
                                                        )
                                                    {
                                                        actions.photon_edited = true;
                                                    }
                                                });

                                                if del_em {
                                                    self.photon_scene.remove_emitter(eidx);
                                                    self.selection = Selection::Emitter(0);
                                                    actions.photon_edited = true;
                                                }
                                            }

                                            // Physics controls
                                            subtle_rule(ui);
                                            section_label(ui, "PHYSICS");
                                            if id.is_kerr() {
                                                if labeled_slider(
                                                    ui,
                                                    "Kerr Spin a",
                                                    &mut self.photon_scene.spin,
                                                    -0.998..=0.998,
                                                    false,
                                                ) {
                                                    actions.photon_edited = true;
                                                }
                                            } else {
                                                if !id.is_solar_system()
                                                    && labeled_slider(
                                                        ui,
                                                        "Dispersion",
                                                        &mut self.photon_scene.dispersion_scale,
                                                        0.0..=3.0,
                                                        false,
                                                    )
                                                {
                                                    actions.photon_edited = true;
                                                    actions.photon_reupload = id.is_optics();
                                                }
                                                if id.is_grin()
                                                    && labeled_slider(
                                                        ui,
                                                        "Strength",
                                                        &mut self.photon_scene.grin_strength,
                                                        0.25..=2.2,
                                                        false,
                                                    )
                                                {
                                                    actions.photon_edited = true;
                                                }
                                            }

                                            if labeled_slider(
                                                ui,
                                                "Exposure",
                                                &mut self.photon_settings.exposure,
                                                0.15..=4.0,
                                                true,
                                            ) {
                                                actions.photon_edited = true;
                                            }

                                            // Interactive Playground / Optical Bench
                                            if id.is_optics() {
                                                subtle_rule(ui);
                                                section_label(ui, "OBJECTS");
                                                ui.horizontal_wrapped(|ui| {
                                                    let c = self.photon_scene.view_center;
                                                    if ui.small_button("+ Prism").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Prism",
                                                            ElementKind::Prism { radius: 1.8 },
                                                            c,
                                                            0.0,
                                                            GlassMaterial::StudioFlint,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Lens").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Lens",
                                                            ElementKind::Lens {
                                                                radius: 3.4,
                                                                half_offset: 2.4,
                                                            },
                                                            c,
                                                            0.0,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Mirror").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Mirror",
                                                            ElementKind::FlatMirror {
                                                                half_len: 1.8,
                                                            },
                                                            c,
                                                            0.35,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Arc").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Curved Mirror",
                                                            ElementKind::CurvedMirror {
                                                                radius: 3.2,
                                                                arc_half_angle: 0.75,
                                                            },
                                                            [c[0] - 2.0, c[1]],
                                                            0.0,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Splitter").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Beam Splitter",
                                                            ElementKind::BeamSplitter {
                                                                half_len: 1.8,
                                                            },
                                                            c,
                                                            0.785,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Wall").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Matte Wall",
                                                            ElementKind::MatteWall {
                                                                half_len: 2.0,
                                                                color: [0.85, 0.85, 0.88],
                                                            },
                                                            [c[0] + 2.5, c[1]],
                                                            0.0,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Planet").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Planet",
                                                            ElementKind::Planet {
                                                                radius: 0.75,
                                                                albedo: [0.32, 0.55, 0.80],
                                                                atmo_thickness: 0.45,
                                                                atmo_density: 2.0,
                                                                ring_outer: 0.0,
                                                                orbit_radius: 0.0,
                                                                orbit_speed: 0.0,
                                                                orbit_phase: 0.0,
                                                                parent_idx: None,
                                                            },
                                                            [c[0] + 1.5, c[1]],
                                                            0.0,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if ui.small_button("+ Stop").clicked() {
                                                        let idx = self.photon_scene.add_element(
                                                            "Beam Stop",
                                                            ElementKind::Absorber { half_len: 1.4 },
                                                            c,
                                                            0.0,
                                                            GlassMaterial::CrownBk7,
                                                        );
                                                        self.selection = Selection::Element(idx);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                    if !self.photon_scene.elements.is_empty()
                                                        && ui.small_button("Clear").clicked()
                                                    {
                                                        self.photon_scene.clear_elements();
                                                        self.selection = Selection::Emitter(0);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                });

                                                // Selected object inspector card
                                                if let Selection::Element(idx) = self.selection
                                                    && idx < self.photon_scene.elements.len()
                                                {
                                                    let mut delete_it = false;
                                                    let el_name =
                                                        self.photon_scene.elements[idx].name;
                                                    let is_glass = self.photon_scene.elements[idx]
                                                        .kind
                                                        .is_glass();

                                                    card_frame().show(ui, |ui| {
                                                        ui.horizontal(|ui| {
                                                            ui.label(
                                                                RichText::new(el_name)
                                                                    .strong()
                                                                    .size(11.5)
                                                                    .color(Color32::from_rgb(
                                                                        224, 234, 248,
                                                                    )),
                                                            );
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if ui
                                                                        .small_button("Delete")
                                                                        .clicked()
                                                                    {
                                                                        delete_it = true;
                                                                    }
                                                                },
                                                            );
                                                        });

                                                        if is_glass {
                                                            ui.horizontal_wrapped(|ui| {
                                                                for mat in GlassMaterial::ALL {
                                                                    let sel = self
                                                                        .photon_scene
                                                                        .elements[idx]
                                                                        .material
                                                                        == mat;
                                                                    if mode_chip(
                                                                        ui,
                                                                        mat.label(),
                                                                        sel,
                                                                    ) {
                                                                        self.photon_scene
                                                                            .elements[idx]
                                                                            .material = mat;
                                                                        actions.photon_reupload =
                                                                            true;
                                                                        actions.photon_edited =
                                                                            true;
                                                                    }
                                                                }
                                                            });
                                                        }

                                                        match &mut self.photon_scene.elements[idx]
                                                            .kind
                                                        {
                                                            ElementKind::Planet {
                                                                albedo,
                                                                atmo_thickness,
                                                                atmo_density,
                                                                ring_outer,
                                                                radius,
                                                                ..
                                                            } => {
                                                                ui.horizontal_wrapped(|ui| {
                                                                    for (
                                                                        lbl,
                                                                        col,
                                                                        atmo_t,
                                                                        atmo_d,
                                                                    ) in [
                                                                        (
                                                                            "Earth",
                                                                            [0.30, 0.52, 0.78],
                                                                            0.48,
                                                                            2.1,
                                                                        ),
                                                                        (
                                                                            "Mars",
                                                                            [0.84, 0.36, 0.22],
                                                                            0.24,
                                                                            0.65,
                                                                        ),
                                                                        (
                                                                            "Venus",
                                                                            [0.88, 0.78, 0.54],
                                                                            0.42,
                                                                            2.2,
                                                                        ),
                                                                        (
                                                                            "Moon",
                                                                            [0.68, 0.67, 0.64],
                                                                            0.0,
                                                                            0.0,
                                                                        ),
                                                                    ] {
                                                                        if ui
                                                                            .small_button(lbl)
                                                                            .clicked()
                                                                        {
                                                                            *albedo = col;
                                                                            *atmo_thickness =
                                                                                atmo_t;
                                                                            *atmo_density = atmo_d;
                                                                            actions
                                                                                .photon_reupload =
                                                                                true;
                                                                            actions.photon_edited =
                                                                                true;
                                                                        }
                                                                    }
                                                                });
                                                                if labeled_slider(
                                                                    ui,
                                                                    "Atmosphere",
                                                                    atmo_density,
                                                                    0.0..=3.5,
                                                                    false,
                                                                ) {
                                                                    if *atmo_density > 0.02
                                                                        && *atmo_thickness < 0.1
                                                                    {
                                                                        *atmo_thickness = 0.45;
                                                                    }
                                                                    actions.photon_reupload = true;
                                                                    actions.photon_edited = true;
                                                                }
                                                                let mut has_ring =
                                                                    *ring_outer > *radius * 1.15;
                                                                if ui
                                                                    .checkbox(
                                                                        &mut has_ring,
                                                                        "Icy Rings",
                                                                    )
                                                                    .changed()
                                                                {
                                                                    *ring_outer = if has_ring {
                                                                        *radius * 2.15
                                                                    } else {
                                                                        0.0
                                                                    };
                                                                    actions.photon_reupload = true;
                                                                    actions.photon_edited = true;
                                                                }
                                                            }
                                                            ElementKind::MatteWall {
                                                                color,
                                                                ..
                                                            } => {
                                                                ui.horizontal_wrapped(|ui| {
                                                                    for (lbl, col) in [
                                                                        (
                                                                            "White",
                                                                            [0.88, 0.88, 0.90],
                                                                        ),
                                                                        ("Red", [0.88, 0.20, 0.16]),
                                                                        (
                                                                            "Green",
                                                                            [0.20, 0.82, 0.28],
                                                                        ),
                                                                        (
                                                                            "Blue",
                                                                            [0.20, 0.44, 0.90],
                                                                        ),
                                                                        (
                                                                            "Gold",
                                                                            [0.92, 0.72, 0.22],
                                                                        ),
                                                                    ] {
                                                                        if ui
                                                                            .small_button(lbl)
                                                                            .clicked()
                                                                        {
                                                                            *color = col;
                                                                            actions
                                                                                .photon_reupload =
                                                                                true;
                                                                            actions.photon_edited =
                                                                                true;
                                                                        }
                                                                    }
                                                                });
                                                            }
                                                            _ => {}
                                                        }

                                                        let mut deg = self.photon_scene.elements
                                                            [idx]
                                                            .angle
                                                            .to_degrees()
                                                            .rem_euclid(360.0);
                                                        if deg > 180.0 {
                                                            deg -= 360.0;
                                                        }
                                                        if labeled_slider(
                                                            ui,
                                                            "Angle",
                                                            &mut deg,
                                                            -180.0..=180.0,
                                                            false,
                                                        ) {
                                                            self.photon_scene.elements[idx].angle =
                                                                deg.to_radians();
                                                            actions.photon_reupload = true;
                                                            actions.photon_edited = true;
                                                        }

                                                        let mut sz = self.photon_scene.elements
                                                            [idx]
                                                            .kind
                                                            .size();
                                                        if labeled_slider(
                                                            ui,
                                                            "Size",
                                                            &mut sz,
                                                            0.2..=6.5,
                                                            false,
                                                        ) {
                                                            self.photon_scene.elements[idx]
                                                                .kind
                                                                .set_size(sz);
                                                            actions.photon_reupload = true;
                                                            actions.photon_edited = true;
                                                        }
                                                    });

                                                    if delete_it {
                                                        self.photon_scene.remove_element(idx);
                                                        self.selection = Selection::Emitter(0);
                                                        actions.photon_reupload = true;
                                                        actions.photon_edited = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                subtle_rule(ui);
                                ui.horizontal(|ui| {
                                    if ui.button("Reset").clicked() {
                                        actions.reset = true;
                                    }
                                    if ui.button("Save PNG").clicked() {
                                        actions.save_png = true;
                                    }
                                });
                            });
                    });
                });
        }

        // 4. Bottom-left subtle gesture hint pill & temporary screenshot toast.
        let hint_text = match self.mode {
            StudioMode::BlackHole3D => "LMB Orbit · Scroll Zoom · H Hide UI",
            StudioMode::Photons(PhotonSceneId::SolarSystem) => {
                "Click Body Focus · LMB Pan · Scroll Zoom · P Pause · M Wavefront · H Hide UI"
            }
            StudioMode::Photons(id) if id.is_optics() => {
                "LMB Drag Emitter / Object · Shift+LMB Rotate · RMB Aim · Scroll Zoom · H Hide UI"
            }
            StudioMode::Photons(_) => {
                "LMB Move Emitter · RMB Aim · Space Pulse · Scroll Zoom · H Hide UI"
            }
        };
        let hint_frame = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(11, 13, 18, 175))
            .stroke(Stroke::new(
                1.0,
                Color32::from_rgba_unmultiplied(255, 255, 255, 16),
            ))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 5));
        egui::Area::new(Id::unique("caustic_hint"))
            .anchor(Align2::LEFT_BOTTOM, Vec2::new(14.0, -12.0))
            .interactable(false)
            .fade_in(false)
            .show(&ctx, |ui| {
                hint_frame.show(ui, |ui| {
                    if let Some((msg, t)) = &self.status_toast
                        && t.elapsed() < Duration::from_secs(3)
                    {
                        ui.label(
                            RichText::new(msg)
                                .size(11.0)
                                .color(Color32::from_rgb(138, 198, 255)),
                        );
                    }
                    ui.label(
                        RichText::new(hint_text)
                            .size(10.5)
                            .color(Color32::from_rgba_unmultiplied(186, 198, 216, 190)),
                    );
                });
            });

        actions
    }

    fn render_frame(&mut self) {
        if self.gfx.is_none() {
            return;
        }

        let now = Instant::now();
        let dt_sec = (now - self.last_frame_instant)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        let dt_ms = dt_sec * 1000.0;
        self.last_frame_instant = now;
        if dt_ms > 0.05 && dt_ms < 2000.0 {
            self.smoothed_ms = self.smoothed_ms * 0.90 + dt_ms * 0.10;
            self.smoothed_fps = 1000.0 / self.smoothed_ms.max(0.1);
        }

        // Advance true-ratio f64 Keplerian Solar System orbital clock when running.
        if matches!(self.mode, StudioMode::Photons(PhotonSceneId::SolarSystem))
            && !self.orbit_paused
            && self.drag.is_none()
            && self.photon_scene.solar.step_time(f64::from(dt_sec))
        {
            self.sync_solar_scene(false, true);
        }

        if self.mode == StudioMode::BlackHole3D
            && !self.bh_full_res
            && self.drag.is_none()
            && self.bh_last_input.elapsed() >= Duration::from_millis(120)
        {
            self.bh_full_res = true;
            self.bh_trace_dirty = true;
        }

        let (w, h, ppp, accum_frames, raw_input) = {
            let gfx = self.gfx.as_mut().expect("gfx checked");
            let w = gfx.surface_config.width.max(1);
            let h = gfx.surface_config.height.max(1);
            let ppp = gfx.window.scale_factor() as f32;
            let raw = gfx.egui_state.take_egui_input(&gfx.window);
            (w, h, ppp, gfx.photon_pipeline.accum_frames(), raw)
        };

        let ctx = self.gfx.as_ref().expect("gfx").egui_ctx.clone();
        let mut ui_actions = UiActions::default();
        let full_output = ctx.run_ui(raw_input, |ui| {
            ui_actions = self.build_ui(ui, w, h, ppp, accum_frames);
        });

        if let Some(next_mode) = ui_actions.switch_mode {
            self.switch_mode(next_mode);
        }
        if ui_actions.reset {
            match self.mode {
                StudioMode::BlackHole3D => {
                    self.camera = self.init_camera;
                    self.bh_params = self.init_bh_params;
                    self.bh_settings = self.init_bh_settings;
                    self.bh_trace_dirty = true;
                    self.bh_shade_dirty = true;
                    self.bh_full_res = true;
                }
                StudioMode::Photons(id) => self.switch_mode(StudioMode::Photons(id)),
            }
        }
        if ui_actions.bh_trace_dirty {
            self.bh_trace_dirty = true;
            self.bh_full_res = false;
            self.bh_last_input = Instant::now();
        }
        if ui_actions.bh_shade_dirty {
            self.bh_shade_dirty = true;
        }
        if ui_actions.solar_changed {
            self.sync_solar_scene(true, false);
            self.photon_last_edit = Instant::now();
        }
        if ui_actions.photon_reset_accum
            && let Some(gfx) = &mut self.gfx
        {
            gfx.photon_pipeline.reset_accumulation();
        }
        if ui_actions.fire_pulse
            && let Some(gfx) = &mut self.gfx
        {
            gfx.photon_pipeline.trigger_pulse();
        }
        if ui_actions.photon_edited || ui_actions.photon_reupload {
            self.mark_photon_edited(ui_actions.photon_reupload, true);
        }
        if ui_actions.save_png {
            self.save_screenshot();
        }

        if matches!(self.mode, StudioMode::Photons(_)) {
            let active_now =
                self.drag.is_some() || self.photon_last_edit.elapsed() < Duration::from_millis(120);
            if self.photon_was_dragging
                && !active_now
                && let Some(gfx) = &mut self.gfx
                && self.photon_settings.mode == PhotonMode::Accumulate
            {
                gfx.photon_pipeline.clear_accumulation();
            }
            self.photon_was_dragging = active_now;
            self.photon_settings.interactive_drag = active_now;
        }

        let Some(gfx) = &mut self.gfx else {
            return;
        };
        gfx.egui_state
            .handle_platform_output(&gfx.window, full_output.platform_output);

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.surface_config);
                self.bh_trace_dirty = true;
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("studio-frame-enc"),
            });

        match self.mode {
            StudioMode::BlackHole3D => {
                let (tw, th) = if self.bh_full_res {
                    (w, h)
                } else {
                    ((w / 2).max(1), (h / 2).max(1))
                };
                gfx.bh_pipeline.ensure_size(&gfx.gpu, tw, th);
                gfx.bh_pipeline.encode_frame(
                    &gfx.gpu,
                    &mut enc,
                    &view,
                    FrameView {
                        camera: &self.camera,
                        params: &self.bh_params,
                        settings: &self.bh_settings,
                        display_aspect: f64::from(w) / f64::from(h),
                    },
                    false,
                );
                self.bh_trace_dirty = false;
                self.bh_shade_dirty = false;
            }
            StudioMode::Photons(_) => {
                gfx.photon_pipeline.ensure_size(&gfx.gpu, w, h);
                let converged = self.photon_settings.mode == PhotonMode::Accumulate
                    && !self.photon_settings.interactive_drag
                    && gfx.photon_pipeline.is_converged(MAX_CONVERGED_FRAMES);
                if converged {
                    gfx.photon_pipeline
                        .encode_present_only(&gfx.gpu, &mut enc, &view);
                } else {
                    gfx.photon_pipeline.encode_frame(
                        &gfx.gpu,
                        &mut enc,
                        &view,
                        &self.photon_scene,
                        &self.photon_settings,
                        false,
                    );
                }
            }
        }

        let screen_desc = ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                gfx.egui_renderer
                    .update_texture(&gfx.gpu.device, &gfx.gpu.queue, *id, delta);
            }
        }
        let paint_jobs = gfx
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let extra_cmds = gfx.egui_renderer.update_buffers(
            &gfx.gpu.device,
            &gfx.gpu.queue,
            &mut enc,
            &paint_jobs,
            &screen_desc,
        );
        {
            let mut rpass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("studio-egui-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            gfx.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_desc);
        }
        for id in &full_output.textures_delta.free {
            gfx.egui_renderer.free_texture(id);
        }

        let mut submits = extra_cmds;
        submits.push(enc.finish());
        gfx.gpu.queue.submit(submits);
        gfx.gpu.queue.present(frame);

        if self.last_title_update.elapsed() >= Duration::from_millis(400) {
            gfx.window.set_title(&format!(
                "caustic · {} · {:.0} fps",
                self.mode.display_name(),
                self.smoothed_fps
            ));
            self.last_title_update = Instant::now();
        }
    }
}

#[derive(Default)]
struct UiActions {
    switch_mode: Option<StudioMode>,
    bh_trace_dirty: bool,
    bh_shade_dirty: bool,
    solar_changed: bool,
    photon_edited: bool,
    photon_reupload: bool,
    photon_reset_accum: bool,
    fire_pulse: bool,
    reset: bool,
    save_png: bool,
}

impl ApplicationHandler for StudioApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let init_res = (|| -> Result<StudioGfx> {
            let attrs = Window::default_attributes()
                .with_title("caustic")
                .with_inner_size(LogicalSize::new(self.init_width, self.init_height));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .context("create studio window")?,
            );

            let display = event_loop.owned_display_handle();
            let mut inst_desc =
                wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display));
            inst_desc.backends = wgpu::Backends::VULKAN;
            let instance = wgpu::Instance::new(inst_desc);
            let surface = instance
                .create_surface(Arc::clone(&window))
                .context("create wgpu surface")?;
            let gpu = GpuContext::new_for_surface(instance, &surface, self.force_fallback)?;

            let size = window.inner_size();
            let w = size.width.max(1);
            let h = size.height.max(1);
            let caps = surface.get_capabilities(&gpu.adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .unwrap_or(caps.formats[0]);
            let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            };
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: w,
                height: h,
                present_mode,
                desired_maximum_frame_latency: 1,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
            };
            surface.configure(&gpu.device, &surface_config);

            if self.photon_scene.id.is_solar_system() {
                self.photon_scene
                    .sync_solar_to_gpu(f64::from(w) / f64::from(h));
            }
            let bh_pipeline = BlackHolePipeline::new(&gpu, w, h, format);
            let mut photon_pipeline =
                PhotonPipeline::new(&gpu, w, h, self.photon_settings.num_photons, format);
            photon_pipeline.upload_scene(&gpu, &self.photon_scene);
            photon_pipeline.reset_accumulation();

            let egui_ctx = egui::Context::default();
            apply_minimal_theme(&egui_ctx);
            let egui_state = egui_winit::State::new(
                egui_ctx.clone(),
                ViewportId::ROOT,
                event_loop,
                Some(window.scale_factor() as f32),
                None,
                None,
            );
            let egui_renderer = EguiRenderer::new(&gpu.device, format, RendererOptions::default());

            Ok(StudioGfx {
                window,
                surface,
                surface_config,
                gpu,
                bh_pipeline,
                photon_pipeline,
                egui_ctx,
                egui_state,
                egui_renderer,
            })
        })();

        match init_res {
            Ok(gfx) => self.gfx = Some(gfx),
            Err(err) => {
                self.init_error = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut ui_consumed = false;
        let mut ui_wants_pointer = false;
        let mut ui_wants_keyboard = false;
        if let Some(gfx) = &mut self.gfx {
            let resp = gfx.egui_state.on_window_event(&gfx.window, &event);
            ui_consumed = resp.consumed;
            ui_wants_pointer =
                gfx.egui_ctx.egui_wants_pointer_input() || gfx.egui_ctx.is_pointer_over_egui();
            ui_wants_keyboard = gfx.egui_ctx.egui_wants_keyboard_input();
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::Resized(new_size) => {
                let mut resized_solar = false;
                if let Some(gfx) = &mut self.gfx
                    && new_size.width > 0
                    && new_size.height > 0
                {
                    gfx.surface_config.width = new_size.width;
                    gfx.surface_config.height = new_size.height;
                    gfx.surface.configure(&gfx.gpu.device, &gfx.surface_config);
                    self.bh_trace_dirty = true;
                    self.bh_full_res = true;
                    resized_solar = self.photon_scene.id.is_solar_system();
                }
                if resized_solar {
                    self.sync_solar_scene(true, false);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                if !down {
                    self.drag = None;
                    return;
                }
                if ui_consumed || ui_wants_pointer {
                    return;
                }
                match (self.mode, button) {
                    (StudioMode::BlackHole3D, MouseButton::Left) => {
                        self.drag = Some(CanvasDrag::Orbit3D);
                    }
                    (StudioMode::Photons(PhotonSceneId::SolarSystem), MouseButton::Left) => {
                        // Click a planet/moon reticle to focus onto it, or drag anywhere to pan in f64 AU.
                        let mut clicked_body: Option<usize> = None;
                        if let Some((cx, cy)) = self.last_cursor
                            && let Some(gfx) = &self.gfx
                            && let Some(sf) = &self.photon_scene.last_solar_frame
                        {
                            let w = f64::from(gfx.surface_config.width.max(1));
                            let h = f64::from(gfx.surface_config.height.max(1));
                            let aspect = w / h;
                            let inv_s = 1.0 / sf.view_half_height_au.max(1e-9);
                            let mut best_d = 18.0_f64;
                            for (i, b) in self.photon_scene.solar.bodies.iter().enumerate() {
                                if b.parent.is_some()
                                    && (b.semi_major_au * inv_s) * (h * 0.5) < 14.0
                                {
                                    continue;
                                }
                                let gp = sf.body_gpu_pos[i];
                                let bx = (gp[0] / aspect * 0.5 + 0.5) * w;
                                let by = (0.5 - gp[1] * 0.5) * h;
                                let d = ((cx - bx).powi(2) + (cy - by).powi(2)).sqrt();
                                let pick = (sf.body_gpu_rad[i] * h * 0.5).max(16.0);
                                if d <= pick && d < best_d.max(pick) {
                                    best_d = d;
                                    clicked_body = Some(i);
                                }
                            }
                        }
                        if let Some(idx) = clicked_body {
                            self.photon_scene.solar.set_focus(SolarFocus::Body(idx));
                            self.sync_solar_scene(true, false);
                            self.photon_last_edit = Instant::now();
                        } else {
                            self.drag = Some(CanvasDrag::SolarPan);
                        }
                    }
                    (StudioMode::Photons(PhotonSceneId::SolarSystem), MouseButton::Right) => {}
                    (StudioMode::Photons(id), MouseButton::Left) => {
                        if let Some((cx, cy)) = self.last_cursor
                            && let Some(world) = self.cursor_to_world(cx, cy)
                        {
                            let px_world = self
                                .gfx
                                .as_ref()
                                .map(|g| {
                                    (2.0 * self.photon_scene.view_half_height)
                                        / (g.surface_config.height.max(1) as f32)
                                })
                                .unwrap_or(0.015);
                            let pick_r = px_world * 18.0;

                            if let Some(eidx) = self.photon_scene.hit_test_emitter(world, pick_r) {
                                self.selection = Selection::Emitter(eidx);
                                self.drag = Some(CanvasDrag::EmitterMove { idx: eidx });
                            } else if id.is_optics()
                                && let Some(idx) = self.photon_scene.hit_test_element(world, pick_r)
                            {
                                self.selection = Selection::Element(idx);
                                if self.modifiers.shift_key() {
                                    self.drag = Some(CanvasDrag::ElementRotate { idx });
                                } else {
                                    let h = self.photon_scene.elements[idx].handle_pos();
                                    self.drag = Some(CanvasDrag::ElementMove {
                                        idx,
                                        offset: [h[0] - world[0], h[1] - world[1]],
                                    });
                                }
                            } else {
                                let eidx = self.active_emitter_idx();
                                self.selection = Selection::Emitter(eidx);
                                self.drag = Some(CanvasDrag::EmitterMove { idx: eidx });
                                if let Some(em) = self.photon_scene.emitters.get_mut(eidx) {
                                    em.pos = world;
                                    self.mark_photon_edited(false, true);
                                }
                            }
                        }
                    }
                    (StudioMode::Photons(_), MouseButton::Right) => {
                        let eidx = self.active_emitter_idx();
                        self.selection = Selection::Emitter(eidx);
                        self.drag = Some(CanvasDrag::EmitterAim { idx: eidx });
                        if let Some((cx, cy)) = self.last_cursor
                            && let Some(world) = self.cursor_to_world(cx, cy)
                            && let Some(em) = self.photon_scene.emitters.get_mut(eidx)
                        {
                            let dx = world[0] - em.pos[0];
                            let dy = world[1] - em.pos[1];
                            if dx * dx + dy * dy > 1e-6 {
                                em.angle = dy.atan2(dx);
                                self.mark_photon_edited(false, true);
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let prev = self.last_cursor;
                self.last_cursor = Some((position.x, position.y));
                let Some(drag) = self.drag else {
                    return;
                };
                match drag {
                    CanvasDrag::Orbit3D => {
                        if let Some((px, py)) = prev {
                            let dx = position.x - px;
                            let dy = position.y - py;
                            self.camera.azimuth = (self.camera.azimuth - dx * 0.006)
                                .rem_euclid(std::f64::consts::TAU);
                            self.camera.inclination = (self.camera.inclination - dy * 0.006)
                                .clamp(0.01, std::f64::consts::PI - 0.01);
                            self.bh_trace_dirty = true;
                            self.bh_full_res = false;
                            self.bh_last_input = Instant::now();
                        }
                    }
                    CanvasDrag::SolarPan => {
                        if let Some((px, py)) = prev
                            && let Some(gfx) = &self.gfx
                        {
                            let h = f64::from(gfx.surface_config.height.max(1));
                            let au_per_px = (2.0 * self.photon_scene.solar.view_half_height_au) / h;
                            self.photon_scene.solar.pan_offset_au[0] -=
                                (position.x - px) * au_per_px;
                            self.photon_scene.solar.pan_offset_au[1] +=
                                (position.y - py) * au_per_px;
                            self.sync_solar_scene(true, false);
                            self.photon_last_edit = Instant::now();
                        }
                    }
                    CanvasDrag::EmitterMove { idx } => {
                        if let Some(world) = self.cursor_to_world(position.x, position.y)
                            && let Some(em) = self.photon_scene.emitters.get_mut(idx)
                        {
                            em.pos = world;
                            self.mark_photon_edited(false, true);
                        }
                    }
                    CanvasDrag::EmitterAim { idx } => {
                        if let Some(world) = self.cursor_to_world(position.x, position.y)
                            && let Some(em) = self.photon_scene.emitters.get_mut(idx)
                        {
                            let dx = world[0] - em.pos[0];
                            let dy = world[1] - em.pos[1];
                            if dx * dx + dy * dy > 1e-6 {
                                em.angle = dy.atan2(dx);
                                self.mark_photon_edited(false, true);
                            }
                        }
                    }
                    CanvasDrag::ElementMove { idx, offset } => {
                        if idx < self.photon_scene.elements.len()
                            && let Some(world) = self.cursor_to_world(position.x, position.y)
                        {
                            let next = [world[0] + offset[0], world[1] + offset[1]];
                            self.photon_scene.elements[idx].move_handle_to(next);
                            // If dragging a planet, update its orbital radius/phase to match!
                            if let ElementKind::Planet {
                                orbit_radius,
                                orbit_phase,
                                parent_idx: None,
                                ..
                            } = &mut self.photon_scene.elements[idx].kind
                                && *orbit_radius > 0.0
                            {
                                let sun = self
                                    .photon_scene
                                    .emitters
                                    .first()
                                    .map(|e| e.pos)
                                    .unwrap_or([0.0, 0.0]);
                                let dx = next[0] - sun[0];
                                let dy = next[1] - sun[1];
                                *orbit_radius = (dx * dx + dy * dy).sqrt().max(1.5);
                                *orbit_phase = dy.atan2(dx);
                            }
                            self.mark_photon_edited(true, true);
                        }
                    }
                    CanvasDrag::ElementRotate { idx } => {
                        if idx < self.photon_scene.elements.len()
                            && let Some(world) = self.cursor_to_world(position.x, position.y)
                        {
                            let center = self.photon_scene.elements[idx].pos;
                            let dx = world[0] - center[0];
                            let dy = world[1] - center[1];
                            if dx * dx + dy * dy > 1e-6 {
                                self.photon_scene.elements[idx].angle = dy.atan2(dx);
                                self.mark_photon_edited(true, true);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if ui_consumed || ui_wants_pointer {
                    return;
                }
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                match self.mode {
                    StudioMode::BlackHole3D => {
                        let min_r = r_plus(self.bh_params.spin) + MIN_CAMERA_HORIZON_MARGIN;
                        let factor = (-f64::from(scroll) * 0.1).exp();
                        self.camera.radius =
                            (self.camera.radius * factor).clamp(min_r, MAX_CAMERA_RADIUS);
                        self.bh_trace_dirty = true;
                        self.bh_full_res = false;
                        self.bh_last_input = Instant::now();
                    }
                    StudioMode::Photons(PhotonSceneId::SolarSystem) => {
                        let (z_min, z_max) = self.photon_scene.solar.zoom_bounds();
                        let factor = (-f64::from(scroll) * 0.14).exp();
                        self.photon_scene.solar.view_half_height_au =
                            (self.photon_scene.solar.view_half_height_au * factor)
                                .clamp(z_min, z_max);
                        self.sync_solar_scene(true, false);
                        self.photon_last_edit = Instant::now();
                    }
                    StudioMode::Photons(_) => {
                        self.photon_scene.view_half_height = (self.photon_scene.view_half_height
                            * (-scroll * 0.08).exp())
                        .clamp(1.0, 60.0);
                        self.mark_photon_edited(false, true);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !ui_wants_keyboard
                    && event.state == ElementState::Pressed
                    && let PhysicalKey::Code(code) = event.physical_key
                {
                    self.handle_key(code, event_loop);
                }
            }
            WindowEvent::RedrawRequested => self.render_frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }
}

fn spectrum_preview_color(spec: EmitterSpectrum) -> Color32 {
    match spec {
        EmitterSpectrum::WhiteFlat => Color32::from_rgb(242, 246, 255),
        EmitterSpectrum::Blackbody(temp) => {
            let t = ((temp - 2400.0) / 9600.0).clamp(0.0, 1.0);
            if t < 0.35 {
                let f = t / 0.35;
                Color32::from_rgb(255, (130.0 + 112.0 * f) as u8, (55.0 + 150.0 * f) as u8)
            } else {
                let f = (t - 0.35) / 0.65;
                Color32::from_rgb(
                    (255.0 - 75.0 * f) as u8,
                    (242.0 - 20.0 * f) as u8,
                    (205.0 + 50.0 * f) as u8,
                )
            }
        }
        EmitterSpectrum::Laser(nm) => {
            let [x, y, z] = cie_1931_xyz(f64::from(nm));
            let r = (3.2404542 * x - 1.5371385 * y - 0.4985314 * z).max(0.0);
            let g = (-0.9692660 * x + 1.8760108 * y + 0.0415560 * z).max(0.0);
            let b = (0.0556434 * x - 0.2040259 * y + 1.0572252 * z).max(0.0);
            let m = r.max(g).max(b).max(1e-6);
            Color32::from_rgb(
                ((r / m).powf(1.0 / 2.2) * 255.0) as u8,
                ((g / m).powf(1.0 / 2.2) * 255.0) as u8,
                ((b / m).powf(1.0 / 2.2) * 255.0) as u8,
            )
        }
    }
}

fn card_frame() -> Frame {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 7))
        .stroke(Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(122, 186, 255, 65),
        ))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(8))
}

fn apply_minimal_theme(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = Vec2::new(5.0, 6.0);
    style.spacing.button_padding = Vec2::new(6.5, 3.8);
    style.spacing.slider_width = 112.0;
    style.visuals = egui::Visuals::dark();
    style.visuals.slider_trailing_fill = true;
    style.visuals.override_text_color = Some(Color32::from_rgb(212, 220, 232));
    style.visuals.panel_fill = Color32::from_rgba_unmultiplied(11, 13, 18, 214);
    style.visuals.window_fill = Color32::from_rgba_unmultiplied(11, 13, 18, 218);
    style.visuals.window_stroke =
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 22));
    style.visuals.widgets.noninteractive.bg_fill =
        Color32::from_rgba_unmultiplied(255, 255, 255, 8);
    style.visuals.widgets.noninteractive.bg_stroke =
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 16));
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgba_unmultiplied(255, 255, 255, 10);
    style.visuals.widgets.inactive.weak_bg_fill =
        Color32::from_rgba_unmultiplied(255, 255, 255, 10);
    style.visuals.widgets.inactive.bg_stroke =
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 16));
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(186, 196, 212));
    style.visuals.widgets.inactive.corner_radius = CornerRadius::same(7);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgba_unmultiplied(255, 255, 255, 20);
    style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgba_unmultiplied(255, 255, 255, 20);
    style.visuals.widgets.hovered.bg_stroke =
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 180, 255, 100));
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(242, 246, 252));
    style.visuals.widgets.hovered.corner_radius = CornerRadius::same(7);
    style.visuals.widgets.active.bg_fill = Color32::from_rgba_unmultiplied(112, 174, 255, 44);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(122, 186, 255));
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.active.corner_radius = CornerRadius::same(7);
    style.visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(112, 174, 255, 50);
    style.visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(138, 196, 255));
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(11.5));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(11.0));
    style
        .text_styles
        .insert(TextStyle::Small, FontId::proportional(10.0));
    ctx.set_global_style(style);
}

fn mode_chip(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let fill = if active {
        Color32::from_rgba_unmultiplied(112, 174, 255, 44)
    } else {
        Color32::TRANSPARENT
    };
    let stroke = if active {
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(134, 192, 255, 150))
    } else {
        Stroke::NONE
    };
    let text = RichText::new(label).size(11.0).color(if active {
        Color32::from_rgb(242, 246, 252)
    } else {
        Color32::from_rgb(164, 174, 192)
    });
    ui.add(
        egui::Button::new(text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(CornerRadius::same(7)),
    )
    .clicked()
}

fn section_label(ui: &mut egui::Ui, label: &str) {
    ui.label(
        RichText::new(label)
            .size(9.5)
            .color(Color32::from_rgb(116, 128, 148)),
    );
}

fn pill_divider(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(5.0, 14.0), egui::Sense::hover());
    ui.painter().line_segment(
        [
            Pos2::new(rect.center().x, rect.top() + 2.0),
            Pos2::new(rect.center().x, rect.bottom() - 2.0),
        ],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 22)),
    );
}

fn subtle_rule(ui: &mut egui::Ui) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 2.0), egui::Sense::hover());
    ui.painter().line_segment(
        [
            Pos2::new(rect.left(), rect.center().y),
            Pos2::new(rect.right(), rect.center().y),
        ],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 16)),
    );
}

fn labeled_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .size(11.5)
                .color(Color32::from_rgb(176, 186, 204)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let resp = ui.add(
                egui::Slider::new(value, range)
                    .logarithmic(logarithmic)
                    .max_decimals(2),
            );
            changed = resp.changed();
        });
    });
    changed
}

/// Launch the unified minimal interactive `caustic` studio window.
pub fn run_studio(cfg: StudioConfig) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    let mut app = StudioApp::new(cfg);
    event_loop
        .run_app(&mut app)
        .context("run winit event loop")?;
    if let Some(err) = app.init_error {
        return Err(err);
    }
    Ok(())
}
