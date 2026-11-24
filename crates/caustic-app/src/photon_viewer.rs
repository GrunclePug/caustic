//! Interactive real-time windowed viewer for forward spectral photon simulations.
//!
//! Supports live mouse emitter dragging, beam aiming, scene hotkeys (`1`-`5`),
//! `Accumulate` vs `Pulse` mode toggle (`M`), manual wavefront pulses (`Space`),
//! and spin/dispersion adjustment (`[`/`]`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use caustic_core::gpu::GpuContext;
use caustic_core::post::{readback_rgba8_texture, write_rgba8_png};
use caustic_photons::gpu_sim::{PhotonPipeline, PhotonSettings};
use caustic_photons::scene::{PhotonMode, PhotonScene, PhotonSceneId};

struct PhotonGfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    pipeline: PhotonPipeline,
}

struct PhotonViewerApp {
    scene: PhotonScene,
    settings: PhotonSettings,
    init_width: u32,
    init_height: u32,
    force_fallback: bool,

    gfx: Option<PhotonGfx>,
    left_drag: bool,
    right_drag: bool,
    last_cursor: Option<(f64, f64)>,
    last_title_update: Instant,
    fps_frames: u32,
    fps_accum_ms: f64,
    shot_counter: u32,
    init_error: Option<anyhow::Error>,
}

impl PhotonViewerApp {
    fn new(
        scene: PhotonScene,
        settings: PhotonSettings,
        width: u32,
        height: u32,
        force_fallback: bool,
    ) -> Self {
        Self {
            scene,
            settings,
            init_width: width,
            init_height: height,
            force_fallback,
            gfx: None,
            left_drag: false,
            right_drag: false,
            last_cursor: None,
            last_title_update: Instant::now(),
            fps_frames: 0,
            fps_accum_ms: 0.0,
            shot_counter: 0,
            init_error: None,
        }
    }

    fn cursor_to_world(&self, px: f64, py: f64) -> Option<[f32; 2]> {
        let gfx = self.gfx.as_ref()?;
        let w = f64::from(gfx.surface_config.width.max(1));
        let h = f64::from(gfx.surface_config.height.max(1));
        let aspect = (w / h) as f32;
        let half_h = self.scene.view_half_height;
        let half_w = half_h * aspect;
        let u = (px / w) as f32;
        let v = (py / h) as f32;
        let wx = self.scene.view_center[0] + (u * 2.0 - 1.0) * half_w;
        let wy = self.scene.view_center[1] + (1.0 - v * 2.0) * half_h;
        Some([wx, wy])
    }

    fn switch_scene(&mut self, id: PhotonSceneId) {
        self.scene = PhotonScene::preset(id);
        if let Some(gfx) = &mut self.gfx {
            gfx.pipeline.upload_scene(&gfx.gpu, &self.scene);
            gfx.pipeline.reset_accumulation();
        }
    }

    fn save_screenshot(&mut self) {
        let Some(gfx) = &mut self.gfx else {
            return;
        };
        self.shot_counter += 1;
        let path = PathBuf::from(format!(
            "caustic-photons-{}-{:03}.png",
            self.scene.id.name(),
            self.shot_counter
        ));
        let w = gfx.surface_config.width.max(1);
        let h = gfx.surface_config.height.max(1);
        let tex = gfx.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("photon-shot-tex"),
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
                label: Some("photon-shot-enc"),
            });
        gfx.pipeline.encode_frame(
            &gfx.gpu,
            &mut enc,
            &view,
            &self.scene,
            &self.settings,
            false,
        );
        gfx.gpu.queue.submit(Some(enc.finish()));
        match readback_rgba8_texture(&gfx.gpu, &tex, w, h)
            .and_then(|rgba| write_rgba8_png(&path, w, h, &rgba))
        {
            Ok(()) => log::info!("Saved {}", path.display()),
            Err(err) => log::error!("Screenshot failed: {err:#}"),
        }
    }

    fn render_frame(&mut self) {
        let Some(gfx) = &mut self.gfx else {
            return;
        };
        let w = gfx.surface_config.width.max(1);
        let h = gfx.surface_config.height.max(1);
        gfx.pipeline.ensure_size(&gfx.gpu, w, h);

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let t0 = Instant::now();
        let mut enc = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("photon-viewer-enc"),
            });
        gfx.pipeline.encode_frame(
            &gfx.gpu,
            &mut enc,
            &view,
            &self.scene,
            &self.settings,
            false,
        );
        gfx.gpu.queue.submit(Some(enc.finish()));
        gfx.gpu.queue.present(frame);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        self.fps_frames += 1;
        self.fps_accum_ms += ms;
        if self.last_title_update.elapsed().as_millis() >= 250 {
            let avg_ms = self.fps_accum_ms / f64::from(self.fps_frames.max(1));
            let mode_str = match self.settings.mode {
                PhotonMode::Accumulate => "Accumulate",
                PhotonMode::Pulse => "Pulse",
            };
            let extra = if self.scene.id.is_kerr() {
                format!("a={:+.3}", self.scene.spin)
            } else {
                format!("disp={:.2}x", self.scene.dispersion_scale)
            };
            let dispatched = gfx
                .pipeline
                .dispatched_photons(&self.scene, self.settings.mode);
            let count_str = match self.settings.mode {
                PhotonMode::Accumulate => format!("{:.1}K paths/f", f64::from(dispatched) * 1e-3),
                PhotonMode::Pulse => format!("{:.2}M photons", f64::from(dispatched) * 1e-6),
            };
            let title = format!(
                "caustic photons | {} ({mode_str}) | {count_str} | {extra} | exp={:.2} | f={} | {:.1}ms",
                self.scene.id.name(),
                self.settings.exposure,
                gfx.pipeline.accum_frames(),
                avg_ms,
            );
            gfx.window.set_title(&title);
            self.last_title_update = Instant::now();
            self.fps_frames = 0;
            self.fps_accum_ms = 0.0;
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Digit1 => self.switch_scene(PhotonSceneId::Prism),
            KeyCode::Digit2 => self.switch_scene(PhotonSceneId::Cardioid),
            KeyCode::Digit3 => self.switch_scene(PhotonSceneId::Lens),
            KeyCode::Digit4 => self.switch_scene(PhotonSceneId::KerrBeam),
            KeyCode::Digit5 => self.switch_scene(PhotonSceneId::KerrPulsar),
            KeyCode::KeyM => {
                self.settings.mode = match self.settings.mode {
                    PhotonMode::Accumulate => PhotonMode::Pulse,
                    PhotonMode::Pulse => PhotonMode::Accumulate,
                };
                if let Some(gfx) = &mut self.gfx {
                    gfx.pipeline.reset_accumulation();
                }
            }
            KeyCode::Space => {
                if let Some(gfx) = &mut self.gfx {
                    match self.settings.mode {
                        PhotonMode::Pulse => gfx.pipeline.trigger_pulse(),
                        PhotonMode::Accumulate => gfx.pipeline.reset_accumulation(),
                    }
                }
            }
            KeyCode::KeyQ => {
                self.scene.emitter_angle += 0.06;
                if let Some(gfx) = &mut self.gfx
                    && self.settings.mode == PhotonMode::Accumulate
                {
                    gfx.pipeline.clear_accumulation();
                }
            }
            KeyCode::KeyE => {
                self.scene.emitter_angle -= 0.06;
                if let Some(gfx) = &mut self.gfx
                    && self.settings.mode == PhotonMode::Accumulate
                {
                    gfx.pipeline.clear_accumulation();
                }
            }
            KeyCode::Comma => {
                self.scene.emitter_spread = (self.scene.emitter_spread / 1.15).max(0.03);
                if let Some(gfx) = &mut self.gfx
                    && self.settings.mode == PhotonMode::Accumulate
                {
                    gfx.pipeline.clear_accumulation();
                }
            }
            KeyCode::Period => {
                let max_spread = if self.scene.id.is_kerr() { 20.0 } else { 6.0 };
                self.scene.emitter_spread = (self.scene.emitter_spread * 1.15).min(max_spread);
                if let Some(gfx) = &mut self.gfx
                    && self.settings.mode == PhotonMode::Accumulate
                {
                    gfx.pipeline.clear_accumulation();
                }
            }
            KeyCode::KeyC => {
                if let Some(gfx) = &mut self.gfx {
                    gfx.pipeline.clear_accumulation();
                }
            }
            KeyCode::BracketRight => {
                if self.scene.id.is_kerr() {
                    self.scene.spin = (self.scene.spin + 0.05).clamp(-0.998, 0.998);
                } else {
                    self.scene.dispersion_scale = (self.scene.dispersion_scale * 1.15).min(4.0);
                    self.scene.rebuild_primitives();
                }
                if let Some(gfx) = &mut self.gfx {
                    gfx.pipeline.upload_scene(&gfx.gpu, &self.scene);
                    if self.settings.mode == PhotonMode::Accumulate {
                        gfx.pipeline.clear_accumulation();
                    }
                }
            }
            KeyCode::BracketLeft => {
                if self.scene.id.is_kerr() {
                    self.scene.spin = (self.scene.spin - 0.05).clamp(-0.998, 0.998);
                } else {
                    self.scene.dispersion_scale = (self.scene.dispersion_scale / 1.15).max(0.1);
                    self.scene.rebuild_primitives();
                }
                if let Some(gfx) = &mut self.gfx {
                    gfx.pipeline.upload_scene(&gfx.gpu, &self.scene);
                    if self.settings.mode == PhotonMode::Accumulate {
                        gfx.pipeline.clear_accumulation();
                    }
                }
            }
            KeyCode::Equal => {
                self.settings.exposure = (self.settings.exposure * 1.25).min(64.0);
            }
            KeyCode::Minus => {
                self.settings.exposure = (self.settings.exposure / 1.25).max(0.02);
            }
            KeyCode::KeyR => self.switch_scene(self.scene.id),
            KeyCode::KeyS => self.save_screenshot(),
            _ => {}
        }
    }
}

impl ApplicationHandler for PhotonViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let init_res = (|| -> Result<PhotonGfx> {
            let attrs = Window::default_attributes()
                .with_title("caustic photons")
                .with_inner_size(LogicalSize::new(self.init_width, self.init_height));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .context("create photon viewer window")?,
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

            let mut pipeline = PhotonPipeline::new(&gpu, w, h, self.settings.num_photons, format);
            pipeline.upload_scene(&gpu, &self.scene);
            pipeline.reset_accumulation();

            Ok(PhotonGfx {
                window,
                surface,
                surface_config,
                gpu,
                pipeline,
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
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(gfx) = &mut self.gfx
                    && new_size.width > 0
                    && new_size.height > 0
                {
                    gfx.surface_config.width = new_size.width;
                    gfx.surface_config.height = new_size.height;
                    gfx.surface.configure(&gfx.gpu.device, &gfx.surface_config);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.left_drag = down;
                        if down
                            && let Some((cx, cy)) = self.last_cursor
                            && let Some(world) = self.cursor_to_world(cx, cy)
                        {
                            self.scene.emitter_pos = world;
                            if let Some(gfx) = &mut self.gfx
                                && self.settings.mode == PhotonMode::Accumulate
                            {
                                gfx.pipeline.clear_accumulation();
                            }
                        }
                    }
                    MouseButton::Right => {
                        self.right_drag = down;
                        if down
                            && let Some((cx, cy)) = self.last_cursor
                            && let Some(world) = self.cursor_to_world(cx, cy)
                        {
                            let dx = world[0] - self.scene.emitter_pos[0];
                            let dy = world[1] - self.scene.emitter_pos[1];
                            if dx * dx + dy * dy > 1e-6 {
                                self.scene.emitter_angle = dy.atan2(dx);
                                if let Some(gfx) = &mut self.gfx
                                    && self.settings.mode == PhotonMode::Accumulate
                                {
                                    gfx.pipeline.clear_accumulation();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor = Some((position.x, position.y));
                if let Some(world) = self.cursor_to_world(position.x, position.y) {
                    if self.left_drag {
                        self.scene.emitter_pos = world;
                        if let Some(gfx) = &mut self.gfx
                            && self.settings.mode == PhotonMode::Accumulate
                        {
                            gfx.pipeline.clear_accumulation();
                        }
                    } else if self.right_drag {
                        let dx = world[0] - self.scene.emitter_pos[0];
                        let dy = world[1] - self.scene.emitter_pos[1];
                        if dx * dx + dy * dy > 1e-6 {
                            self.scene.emitter_angle = dy.atan2(dx);
                            if let Some(gfx) = &mut self.gfx
                                && self.settings.mode == PhotonMode::Accumulate
                            {
                                gfx.pipeline.clear_accumulation();
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                self.scene.view_half_height =
                    (self.scene.view_half_height * (-scroll * 0.08).exp()).clamp(1.0, 60.0);
                if let Some(gfx) = &mut self.gfx {
                    gfx.pipeline.clear_accumulation();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed
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

/// Launch the interactive real-time forward photon viewer.
pub fn run_photon_viewer(
    scene: PhotonScene,
    settings: PhotonSettings,
    width: u32,
    height: u32,
    force_fallback: bool,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    let mut app = PhotonViewerApp::new(scene, settings, width, height, force_fallback);
    event_loop
        .run_app(&mut app)
        .context("run winit event loop")?;
    if let Some(err) = app.init_error {
        return Err(err);
    }
    Ok(())
}
