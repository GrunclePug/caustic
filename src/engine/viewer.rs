//! Interactive windowed `winit` + `wgpu` orbit viewer for the Kerr black hole.
//!
//! Uses half-resolution geodesic tracing while dragging/zooming and automatically
//! refines to full native resolution when the camera settles, keeping Wayland
//! presentation smooth with `desired_maximum_frame_latency = 1`.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::engine::gpu::GpuContext;
use crate::engine::post::{DebugView, write_rgba8_png};
use crate::sim::blackhole::gpu_tracer::{BlackHolePipeline, FrameView, RenderSettings};
use crate::sim::blackhole::kerr::{
    Camera, KerrParams, MAX_CAMERA_RADIUS, MAX_SPIN, MIN_CAMERA_HORIZON_MARGIN, r_plus,
};

const SETTLE_DELAY: Duration = Duration::from_millis(120);

struct ViewerGpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    pipeline: BlackHolePipeline,
}

struct ViewerApp {
    initial_camera: Camera,
    initial_params: KerrParams,
    initial_settings: RenderSettings,
    camera: Camera,
    params: KerrParams,
    settings: RenderSettings,
    init_width: u32,
    init_height: u32,
    force_fallback: bool,

    gfx: Option<ViewerGpu>,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    last_interaction: Instant,
    half_res_active: bool,
    pending_input_time: Option<Instant>,
    drag_latencies_ms: VecDeque<f64>,
    screenshot_counter: u32,
    init_error: Option<anyhow::Error>,
}

impl ViewerApp {
    fn new(
        camera: Camera,
        params: KerrParams,
        settings: RenderSettings,
        width: u32,
        height: u32,
        force_fallback: bool,
    ) -> Self {
        Self {
            initial_camera: camera,
            initial_params: params,
            initial_settings: settings,
            camera,
            params,
            settings,
            init_width: width,
            init_height: height,
            force_fallback,
            gfx: None,
            dragging: false,
            last_cursor: None,
            last_interaction: Instant::now() - SETTLE_DELAY,
            half_res_active: false,
            pending_input_time: None,
            drag_latencies_ms: VecDeque::with_capacity(128),
            screenshot_counter: 0,
            init_error: None,
        }
    }

    fn mark_interacting(&mut self) {
        let now = Instant::now();
        self.last_interaction = now;
        self.half_res_active = true;
        if self.pending_input_time.is_none() {
            self.pending_input_time = Some(now);
        }
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }

    fn mark_dirty_full(&mut self) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }

    fn clamp_camera(&mut self) {
        let min_r = r_plus(self.params.spin) + MIN_CAMERA_HORIZON_MARGIN;
        self.camera.radius = self.camera.radius.clamp(min_r, MAX_CAMERA_RADIUS);
        self.camera.inclination = self
            .camera
            .inclination
            .clamp(0.002, std::f64::consts::PI - 0.002);
    }

    fn p95_drag_latency(&self) -> f64 {
        if self.drag_latencies_ms.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.drag_latencies_ms.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let idx = ((sorted.len() as f64) * 0.95) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn update_title(&self, frame_ms: f64, traced: bool, trace_w: u32, trace_h: u32) {
        let Some(gfx) = &self.gfx else {
            return;
        };
        let mode = match self.settings.debug_view {
            DebugView::Beauty => "Beauty",
            DebugView::StepHeatmap => "Steps",
            DebugView::Redshift => "Redshift",
            DebugView::HitClass => "HitClass",
        };
        let state = if traced { "trace" } else { "cached" };
        let title = format!(
            "caustic | a={:+.3} r={:.1}M inc={:.1}deg | {trace_w}x{trace_h} ({state}) {frame_ms:.1}ms p95-drag={:.1}ms | {mode}",
            self.params.spin,
            self.camera.radius,
            self.camera.inclination.to_degrees(),
            self.p95_drag_latency(),
        );
        gfx.window.set_title(&title);
    }

    fn save_screenshot(&mut self) {
        let Some(gfx) = &mut self.gfx else {
            return;
        };
        self.screenshot_counter += 1;
        let path = PathBuf::from(format!("caustic-shot-{:03}.png", self.screenshot_counter));
        let w = gfx.surface_config.width;
        let h = gfx.surface_config.height;
        gfx.pipeline.ensure_size(&gfx.gpu, w, h);
        match gfx.pipeline.render_headless(
            &gfx.gpu,
            &self.camera,
            &self.params,
            &self.settings,
            false,
        ) {
            Ok((rgba, _)) => match write_rgba8_png(&path, w, h, &rgba) {
                Ok(()) => log::info!("Saved screenshot {}", path.display()),
                Err(err) => log::error!("Failed to write {}: {err:#}", path.display()),
            },
            Err(err) => log::error!("Failed to capture screenshot: {err:#}"),
        }
    }

    fn render_frame(&mut self) {
        let now = Instant::now();
        let settling = !self.dragging && now.duration_since(self.last_interaction) >= SETTLE_DELAY;
        if settling {
            self.half_res_active = false;
        }

        let Some(gfx) = &mut self.gfx else {
            return;
        };
        let win_w = gfx.surface_config.width.max(1);
        let win_h = gfx.surface_config.height.max(1);
        let (trace_w, trace_h) = if self.half_res_active {
            ((win_w / 2).max(1), (win_h / 2).max(1))
        } else {
            (win_w, win_h)
        };
        gfx.pipeline.ensure_size(&gfx.gpu, trace_w, trace_h);

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.surface_config);
                gfx.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let aspect = f64::from(win_w) / f64::from(win_h);

        let start = Instant::now();
        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewer-frame-encoder"),
            });
        let traced = gfx.pipeline.encode_frame(
            &gfx.gpu,
            &mut encoder,
            &view,
            FrameView {
                camera: &self.camera,
                params: &self.params,
                settings: &self.settings,
                display_aspect: aspect,
            },
            false,
        );
        gfx.gpu.queue.submit(Some(encoder.finish()));
        gfx.gpu.queue.present(frame);
        let frame_ms = start.elapsed().as_secs_f64() * 1000.0;

        if let Some(t_in) = self.pending_input_time.take() {
            let lat_ms = t_in.elapsed().as_secs_f64() * 1000.0;
            if self.drag_latencies_ms.len() >= 120 {
                self.drag_latencies_ms.pop_front();
            }
            self.drag_latencies_ms.push_back(lat_ms);
        }

        self.update_title(frame_ms, traced, trace_w, trace_h);
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Digit1 => {
                self.settings.debug_view = DebugView::Beauty;
                self.mark_dirty_full();
            }
            KeyCode::Digit2 => {
                self.settings.debug_view = DebugView::StepHeatmap;
                self.mark_dirty_full();
            }
            KeyCode::Digit3 => {
                self.settings.debug_view = DebugView::Redshift;
                self.mark_dirty_full();
            }
            KeyCode::Digit4 => {
                self.settings.debug_view = DebugView::HitClass;
                self.mark_dirty_full();
            }
            KeyCode::KeyD => {
                self.settings.disk_enabled = !self.settings.disk_enabled;
                self.mark_interacting();
            }
            KeyCode::BracketRight => {
                self.params.spin = (self.params.spin + 0.05).clamp(-MAX_SPIN, MAX_SPIN);
                self.clamp_camera();
                self.mark_interacting();
            }
            KeyCode::BracketLeft => {
                self.params.spin = (self.params.spin - 0.05).clamp(-MAX_SPIN, MAX_SPIN);
                self.clamp_camera();
                self.mark_interacting();
            }
            KeyCode::Equal => {
                self.settings.exposure = (self.settings.exposure * 1.25).min(64.0);
                self.mark_dirty_full();
            }
            KeyCode::Minus => {
                self.settings.exposure = (self.settings.exposure / 1.25).max(0.02);
                self.mark_dirty_full();
            }
            KeyCode::KeyT => {
                self.settings.disk_peak_temp_k =
                    (self.settings.disk_peak_temp_k * 1.15).min(30_000.0);
                self.mark_dirty_full();
            }
            KeyCode::KeyG => {
                self.settings.disk_peak_temp_k =
                    (self.settings.disk_peak_temp_k / 1.15).max(1_500.0);
                self.mark_dirty_full();
            }
            KeyCode::KeyR => {
                self.camera = self.initial_camera;
                self.params = self.initial_params;
                self.settings = self.initial_settings;
                self.mark_interacting();
            }
            KeyCode::KeyS => self.save_screenshot(),
            _ => {}
        }
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let init_res = (|| -> Result<ViewerGpu> {
            let attrs = Window::default_attributes()
                .with_title("caustic")
                .with_inner_size(LogicalSize::new(self.init_width, self.init_height));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .context("create winit window")?,
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

            let pipeline = BlackHolePipeline::new(&gpu, w, h, format);
            window.request_redraw();
            Ok(ViewerGpu {
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
                    gfx.window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                    self.last_interaction = Instant::now();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some((lx, ly)) = self.last_cursor {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        self.camera.azimuth -= dx * 0.005;
                        self.camera.inclination -= dy * 0.005;
                        self.clamp_camera();
                        self.mark_interacting();
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                self.camera.radius *= (-scroll * 0.08).exp();
                self.clamp_camera();
                self.mark_interacting();
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
        if self.half_res_active && !self.dragging {
            let settle_at = self.last_interaction + SETTLE_DELAY;
            if Instant::now() >= settle_at {
                if let Some(gfx) = &self.gfx {
                    gfx.window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(settle_at));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// Launch the interactive windowed Kerr black hole viewer.
pub fn run_viewer(
    camera: Camera,
    params: KerrParams,
    settings: RenderSettings,
    width: u32,
    height: u32,
    force_fallback: bool,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    let mut app = ViewerApp::new(camera, params, settings, width, height, force_fallback);
    event_loop
        .run_app(&mut app)
        .context("run winit event loop")?;
    if let Some(err) = app.init_error {
        return Err(err);
    }
    Ok(())
}
