//! GPU forward spectral photon simulation pipeline (Pass 1 step+splat, Pass 2 resolve+overlay,
//! Pass 3 shared `caustic-core` AgX tone mapping).

use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use caustic_core::gpu::{GpuContext, GpuTimers};
use caustic_core::post::{AgxPresentPass, readback_rgba8_texture};
use caustic_core::spectrum::build_spectral_xyz_lut;
use wgpu::util::DeviceExt;

use crate::ingoing_kerr::r_plus;
use crate::scene::{
    GpuEmitter, MAX_EMITTERS, OptCircle, OptSegment, PhotonMode, PhotonScene, PhysicsKind,
};

/// Maximum segments / circles supported in scene storage buffers.
const MAX_SEGMENTS: usize = 128;
const MAX_CIRCLES: usize = 128;
/// Fixed-point multiplier for `atomic<u32>` spectral CIE XYZ deposits.
const FIXED_SCALE: f32 = 1024.0;

/// Matches `struct SimUniforms` in `photon_step.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SimUniforms {
    view: [f32; 4],
    emitter: [f32; 4],
    physics: [f32; 4],
    extra: [f32; 4],
    dims: [u32; 4],
    flags: [u32; 4],
    counts: [u32; 4],
}

/// Matches `struct ResolveUniforms` in `photon_resolve.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ResolveUniforms {
    view: [f32; 4],
    params: [f32; 4],
    kerr: [f32; 4],
    dims: [u32; 4],
    counts: [u32; 4],
}

/// Matches `struct Photon` in `photon_step.wgsl` (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuPhoton {
    pos: [f32; 2],
    p: [f32; 2],
    lambda_nm: f32,
    energy: f32,
    rng: u32,
    flags: u32,
}

/// Runtime settings for the forward photon simulator.
#[derive(Debug, Clone, Copy)]
pub struct PhotonSettings {
    /// Number of simultaneously simulated forward photon packets.
    pub num_photons: u32,
    /// Integration substeps per frame.
    pub substeps: u32,
    /// Coordinate-time / spatial step per substep.
    pub dt_substep: f32,
    /// Accumulate (long-exposure caustic) vs Pulse (live wavefront flight).
    pub mode: PhotonMode,
    /// Exponential persistence decay per frame in `Pulse` mode (`0.85..0.98`).
    pub pulse_decay: f32,
    /// Automatic pulse re-trigger interval in frames (`Pulse` mode; 0 = manual only).
    pub pulse_period_frames: u32,
    /// Exposure multiplier.
    pub exposure: f32,
    /// Deterministic PRNG seed.
    pub seed: u32,
    /// When true (during active mouse/slider dragging), uses a leaner per-frame batch and
    /// wider stochastic splat stride for locked 200+ FPS live responsiveness.
    pub interactive_drag: bool,
}

impl Default for PhotonSettings {
    fn default() -> Self {
        Self {
            num_photons: 1_000_000,
            substeps: 4,
            dt_substep: 0.08,
            mode: PhotonMode::Accumulate,
            pulse_decay: 0.87,
            pulse_period_frames: 110,
            exposure: 1.0,
            seed: 42,
            interactive_drag: false,
        }
    }
}

/// Performance telemetry for one stepped photon frame.
#[derive(Debug, Clone, Copy)]
pub struct PhotonTelemetry {
    pub width: u32,
    pub height: u32,
    pub mode: PhotonMode,
    pub dispatched_photons: u32,
    pub substeps: u32,
    pub pass1_step_ms: Option<f64>,
    pub pass2_resolve_ms: Option<f64>,
    pub pass3_agx_ms: Option<f64>,
    pub wall_ms: f64,
    pub photon_throughput_per_sec: f64,
    pub accum_frames: u64,
}

struct SizedBuffers {
    width: u32,
    height: u32,
    _accum_xyz_buf: wgpu::Buffer,
    _hdr_accum_buf: wgpu::Buffer,
    _hdr_tex: wgpu::Texture,
    step_bg: wgpu::BindGroup,
    resolve_bg: wgpu::BindGroup,
    agx_bg: wgpu::BindGroup,
}

/// 3-pass forward spectral photon pipeline.
pub struct PhotonPipeline {
    step_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    agx_pass: AgxPresentPass,
    step_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    sim_uniform_buf: wgpu::Buffer,
    resolve_uniform_buf: wgpu::Buffer,
    photons_buf: wgpu::Buffer,
    num_photons: u32,
    spectral_lut_buf: wgpu::Buffer,
    segments_buf: wgpu::Buffer,
    circles_buf: wgpu::Buffer,
    emitters_buf: wgpu::Buffer,
    sized: SizedBuffers,
    timers: Option<GpuTimers>,
    frame_idx: u32,
    accum_frames: u64,
    /// `0` = normal accumulate/pulse, `1` = hard clear, `2` = temporal EMA blend (smooth live animation).
    pending_clear_mode: u32,
    pending_pulse: bool,
}

impl PhotonPipeline {
    /// Create all pipelines, photon state buffers, spectral CIE LUT, and resolution buffers.
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        num_photons: u32,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let step_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("photon_step.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/photon_step.wgsl").into()),
            });
        let resolve_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("photon_resolve.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/photon_resolve.wgsl").into(),
                ),
            });

        let step_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("photon-step-bgl"),
                entries: &[
                    bgl_buf(0, wgpu::BufferBindingType::Uniform),
                    bgl_buf(1, wgpu::BufferBindingType::Storage { read_only: false }),
                    bgl_buf(2, wgpu::BufferBindingType::Storage { read_only: false }),
                    bgl_buf(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_buf(4, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_buf(5, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_buf(6, wgpu::BufferBindingType::Storage { read_only: true }),
                ],
            });

        let resolve_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("photon-resolve-bgl"),
                entries: &[
                    bgl_buf(0, wgpu::BufferBindingType::Uniform),
                    bgl_buf(1, wgpu::BufferBindingType::Storage { read_only: false }),
                    bgl_buf(2, wgpu::BufferBindingType::Storage { read_only: false }),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    bgl_buf(4, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_buf(5, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_buf(6, wgpu::BufferBindingType::Storage { read_only: true }),
                ],
            });

        let step_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("photon-step-layout"),
                bind_group_layouts: &[Some(&step_bgl)],
                immediate_size: 0,
            });
        let step_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("photon-step-pipeline"),
                layout: Some(&step_layout),
                module: &step_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let resolve_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("photon-resolve-layout"),
                bind_group_layouts: &[Some(&resolve_bgl)],
                immediate_size: 0,
            });
        let resolve_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("photon-resolve-pipeline"),
                    layout: Some(&resolve_layout),
                    module: &resolve_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let agx_pass = AgxPresentPass::new(gpu, target_format);

        let sim_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-sim-uniforms"),
            size: std::mem::size_of::<SimUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let resolve_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-resolve-uniforms"),
            size: std::mem::size_of::<ResolveUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count = num_photons.max(256);
        let photons_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-state-buf"),
            size: u64::from(count) * (std::mem::size_of::<GpuPhoton>() as u64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let spectral_lut = build_spectral_xyz_lut();
        let spectral_lut_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spectral-xyz-lut"),
                contents: bytemuck::cast_slice(&spectral_lut),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let segments_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-segments"),
            size: (MAX_SEGMENTS * std::mem::size_of::<OptSegment>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let circles_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-circles"),
            size: (MAX_CIRCLES * std::mem::size_of::<OptCircle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let emitters_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-emitters"),
            size: (MAX_EMITTERS * std::mem::size_of::<GpuEmitter>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sized = Self::create_sized(
            gpu,
            width.max(1),
            height.max(1),
            &step_bgl,
            &resolve_bgl,
            &agx_pass,
            &sim_uniform_buf,
            &resolve_uniform_buf,
            &photons_buf,
            &spectral_lut_buf,
            &segments_buf,
            &circles_buf,
            &emitters_buf,
        );
        let timers = GpuTimers::new(gpu);

        Self {
            step_pipeline,
            resolve_pipeline,
            agx_pass,
            step_bgl,
            resolve_bgl,
            sim_uniform_buf,
            resolve_uniform_buf,
            photons_buf,
            num_photons: count,
            spectral_lut_buf,
            segments_buf,
            circles_buf,
            emitters_buf,
            sized,
            timers,
            frame_idx: 0,
            accum_frames: 0,
            pending_clear_mode: 1,
            pending_pulse: true,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "internal helper constructing resolution-dependent bind groups once"
    )]
    fn create_sized(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        step_bgl: &wgpu::BindGroupLayout,
        resolve_bgl: &wgpu::BindGroupLayout,
        agx_pass: &AgxPresentPass,
        sim_uniform_buf: &wgpu::Buffer,
        resolve_uniform_buf: &wgpu::Buffer,
        photons_buf: &wgpu::Buffer,
        spectral_lut_buf: &wgpu::Buffer,
        segments_buf: &wgpu::Buffer,
        circles_buf: &wgpu::Buffer,
        emitters_buf: &wgpu::Buffer,
    ) -> SizedBuffers {
        let num_px = u64::from(width) * u64::from(height);
        let accum_xyz_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-accum-xyz-atomic"),
            size: num_px * 12,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let hdr_accum_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon-hdr-accum"),
            size: num_px * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let hdr_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("photon-hdr-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let step_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photon-step-bg"),
            layout: step_bgl,
            entries: &[
                bg_buf(0, sim_uniform_buf),
                bg_buf(1, photons_buf),
                bg_buf(2, &accum_xyz_buf),
                bg_buf(3, spectral_lut_buf),
                bg_buf(4, segments_buf),
                bg_buf(5, circles_buf),
                bg_buf(6, emitters_buf),
            ],
        });
        let resolve_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photon-resolve-bg"),
            layout: resolve_bgl,
            entries: &[
                bg_buf(0, resolve_uniform_buf),
                bg_buf(1, &accum_xyz_buf),
                bg_buf(2, &hdr_accum_buf),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                bg_buf(4, segments_buf),
                bg_buf(5, circles_buf),
                bg_buf(6, emitters_buf),
            ],
        });
        let agx_bg = agx_pass.create_bind_group(gpu, &hdr_view);

        SizedBuffers {
            width,
            height,
            _accum_xyz_buf: accum_xyz_buf,
            _hdr_accum_buf: hdr_accum_buf,
            _hdr_tex: hdr_tex,
            step_bg,
            resolve_bg,
            agx_bg,
        }
    }

    /// Resize screen buffers if needed (also clears accumulation).
    pub fn ensure_size(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.sized.width == w && self.sized.height == h {
            return;
        }
        self.sized = Self::create_sized(
            gpu,
            w,
            h,
            &self.step_bgl,
            &self.resolve_bgl,
            &self.agx_pass,
            &self.sim_uniform_buf,
            &self.resolve_uniform_buf,
            &self.photons_buf,
            &self.spectral_lut_buf,
            &self.segments_buf,
            &self.circles_buf,
            &self.emitters_buf,
        );
        self.reset_accumulation();
    }

    /// Clear accumulated HDR exposure and reset deterministic sequence/pulse state.
    pub fn reset_accumulation(&mut self) {
        self.frame_idx = 0;
        self.accum_frames = 0;
        self.pending_clear_mode = 1;
        self.pending_pulse = true;
    }

    /// Clear accumulated HDR exposure for live interactive emitter/camera edits without
    /// restarting Pulse wavefronts or resetting PRNG temporal decorrelation.
    pub fn clear_accumulation(&mut self) {
        self.accum_frames = 0;
        self.pending_clear_mode = 1;
    }

    /// Smoothly blend the next Steady frame with the previous HDR average via temporal EMA
    /// so smoothly animating scenes (like orbiting planets) stay noise-free without ghosting.
    pub fn blend_accumulation(&mut self) {
        if self.pending_clear_mode == 0 {
            self.pending_clear_mode = 2;
        }
    }

    /// Active photons dispatched per frame (full end-to-end trajectories in Accumulate mode;
    /// full persistent pool in Pulse mode).
    pub fn dispatched_photons(&self, scene: &PhotonScene, settings: &PhotonSettings) -> u32 {
        match settings.mode {
            PhotonMode::Pulse => self.num_photons,
            PhotonMode::Accumulate => {
                let phys = scene.id.physics_kind();
                let (div, cap) = match (phys, settings.interactive_drag) {
                    (PhysicsKind::Optics, true) => (40, 24_576),
                    (PhysicsKind::Optics, false) => (20, 49_152),
                    (_, true) => (64, 16_384),
                    (_, false) => (40, 24_576),
                };
                (self.num_photons / div)
                    .clamp(256, cap)
                    .min(self.num_photons)
            }
        }
    }

    /// Re-present the already-converged HDR buffer through AgX without stepping new photons
    /// (~0.01 ms GPU cost when an Accumulate scene has finished settling).
    pub fn encode_present_only(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
        self.agx_pass
            .encode(gpu, encoder, &self.sized.agx_bg, target_view, 1.0, None);
    }

    /// Fire a fresh synchronized photon pulse (keeping existing fading trails in Pulse mode).
    pub fn trigger_pulse(&mut self) {
        self.pending_pulse = true;
    }

    /// Upload scene primitives (segments and circles) to the GPU.
    pub fn upload_scene(&self, gpu: &GpuContext, scene: &PhotonScene) {
        let n_seg = scene.segments.len().min(MAX_SEGMENTS);
        if n_seg > 0 {
            gpu.queue.write_buffer(
                &self.segments_buf,
                0,
                bytemuck::cast_slice(&scene.segments[..n_seg]),
            );
        }
        let n_cir = scene.circles.len().min(MAX_CIRCLES);
        if n_cir > 0 {
            gpu.queue.write_buffer(
                &self.circles_buf,
                0,
                bytemuck::cast_slice(&scene.circles[..n_cir]),
            );
        }
    }

    /// Encode one complete simulation & render frame (Step -> Resolve -> AgX) into `target_view`.
    pub fn encode_frame(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        scene: &PhotonScene,
        settings: &PhotonSettings,
        record_timestamps: bool,
    ) {
        let w = self.sized.width;
        let h = self.sized.height;
        let aspect = (w as f32) / (h as f32);
        let phys = scene.id.physics_kind();
        let is_kerr = phys == PhysicsKind::Kerr;
        let spin = scene.spin.clamp(-0.998, 0.998);
        let rp = r_plus(f64::from(spin)) as f32;

        let auto_pulse = settings.mode == PhotonMode::Pulse
            && settings.pulse_period_frames > 0
            && self.frame_idx > 0
            && self.frame_idx.is_multiple_of(settings.pulse_period_frames);
        let pulse_trigger = self.pending_pulse || auto_pulse;
        let clear_mode = self.pending_clear_mode;
        self.pending_pulse = false;
        self.pending_clear_mode = 0;

        let prev_accum_frames = self.accum_frames.max(1);
        if clear_mode != 0 {
            self.accum_frames = 1;
        } else {
            self.accum_frames += 1;
        }

        // Tune step scale so optical/GRIN scenes (~10 units wide) and Kerr scenes (~28M wide)
        // traverse at an intuitive real-time visual speed in Pulse mode.
        let dt_substep = if is_kerr {
            settings.dt_substep * 2.2
        } else {
            settings.dt_substep
        };

        // Unbiased stochastic strided splatting cuts atomic contention by ~3-4x on narrow beams
        // and high-DPI displays while keeping expected irradiance identical.
        let splat_stride = match (settings.mode, settings.interactive_drag, phys) {
            (PhotonMode::Pulse, _, _) => 1.5,
            (PhotonMode::Accumulate, true, _) => 4.0,
            (PhotonMode::Accumulate, false, PhysicsKind::Optics) => 2.5,
            (PhotonMode::Accumulate, false, _) => 2.0,
        };

        let packed_emitters = scene.pack_gpu_emitters();
        let num_emitters = packed_emitters.len().min(MAX_EMITTERS);
        if num_emitters > 0 {
            gpu.queue.write_buffer(
                &self.emitters_buf,
                0,
                bytemuck::cast_slice(&packed_emitters[..num_emitters]),
            );
        }
        let total_power: f32 = packed_emitters
            .iter()
            .map(|e| e.spec[1])
            .sum::<f32>()
            .max(0.01);

        let active_photons = self.dispatched_photons(scene, settings);
        let sim_u = SimUniforms {
            view: [
                scene.view_center[0],
                scene.view_center[1],
                scene.view_half_height,
                aspect,
            ],
            emitter: [total_power, 0.0, 0.0, 0.0],
            physics: [dt_substep, spin, rp, FIXED_SCALE],
            extra: [
                splat_stride,
                scene.grin_strength.clamp(0.1, 3.0),
                scene.dispersion_scale.clamp(0.0, 4.0),
                0.0,
            ],
            dims: [w, h, active_photons, settings.substeps.max(1)],
            flags: [
                phys as u32,
                settings.mode as u32,
                num_emitters as u32,
                u32::from(pulse_trigger),
            ],
            counts: [
                scene.segments.len().min(MAX_SEGMENTS) as u32,
                scene.circles.len().min(MAX_CIRCLES) as u32,
                self.frame_idx,
                settings.seed,
            ],
        };
        gpu.queue
            .write_buffer(&self.sim_uniform_buf, 0, bytemuck::bytes_of(&sim_u));

        // Exact step-size-invariant arc-length splatting scales per-pixel density linearly with
        // (active_photons / height), so multiplying by (height / active_photons) keeps visual
        // exposure consistent across all resolutions, drag/settle budgets, and photon counts.
        let norm = (h as f32) / (active_photons as f32).max(1.0);
        let gain_k = match (settings.mode, phys, scene.id.is_solar_system()) {
            (PhotonMode::Accumulate, PhysicsKind::Optics, true) => 0.42,
            (PhotonMode::Accumulate, PhysicsKind::Optics, false) => 0.16 * total_power.sqrt(),
            (PhotonMode::Accumulate, PhysicsKind::Kerr, _) => 0.14,
            (PhotonMode::Accumulate, _, _) => 0.22,
            (PhotonMode::Pulse, PhysicsKind::Kerr, _) => 0.42,
            (PhotonMode::Pulse, _, true) => 1.25,
            (PhotonMode::Pulse, _, false) => 0.70,
        };
        let base_gain = gain_k * norm * settings.exposure;
        // For EMA blend mode (`clear_mode == 2`), pass previous `1 / prev_accum_frames` so shader
        // normalizes the previous accumulated sum before mixing with the fresh single frame.
        let accum_norm = if clear_mode == 2 {
            1.0 / (prev_accum_frames as f32)
        } else {
            1.0 / (self.accum_frames as f32).max(1.0)
        };

        let resolve_u = ResolveUniforms {
            view: sim_u.view,
            params: [
                1.0 / FIXED_SCALE,
                settings.pulse_decay,
                base_gain,
                accum_norm,
            ],
            kerr: [
                spin,
                rp,
                scene.grin_strength.clamp(0.1, 3.0),
                scene.dispersion_scale.clamp(0.0, 4.0),
            ],
            dims: [w, h, phys as u32, settings.mode as u32],
            counts: [
                scene.segments.len().min(MAX_SEGMENTS) as u32,
                scene.circles.len().min(MAX_CIRCLES) as u32,
                clear_mode,
                num_emitters as u32,
            ],
        };
        gpu.queue
            .write_buffer(&self.resolve_uniform_buf, 0, bytemuck::bytes_of(&resolve_u));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("photon-pass1-step"),
                timestamp_writes: if record_timestamps {
                    self.timers.as_ref().map(|t| t.compute_writes(0))
                } else {
                    None
                },
            });
            pass.set_pipeline(&self.step_pipeline);
            pass.set_bind_group(0, &self.sized.step_bg, &[]);
            pass.dispatch_workgroups(active_photons.div_ceil(256), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("photon-pass2-resolve"),
                timestamp_writes: if record_timestamps {
                    self.timers.as_ref().map(|t| t.compute_writes(1))
                } else {
                    None
                },
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &self.sized.resolve_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }

        self.agx_pass.encode(
            gpu,
            encoder,
            &self.sized.agx_bg,
            target_view,
            1.0,
            if record_timestamps {
                self.timers.as_ref().map(|t| t.render_writes(2))
            } else {
                None
            },
        );

        if record_timestamps && let Some(timers) = &self.timers {
            timers.resolve(encoder, GpuTimers::SLOTS);
        }

        self.frame_idx = self.frame_idx.wrapping_add(1);
    }

    /// Current accumulated exposure frame count.
    pub fn accum_frames(&self) -> u64 {
        self.accum_frames
    }

    /// Step `frames` frames headlessly and read back the final RGBA8 image + last-frame telemetry.
    pub fn render_headless(
        &mut self,
        gpu: &GpuContext,
        scene: &PhotonScene,
        settings: &PhotonSettings,
        frames: u32,
    ) -> Result<(Vec<u8>, PhotonTelemetry)> {
        let w = self.sized.width;
        let h = self.sized.height;
        self.upload_scene(gpu, scene);
        self.reset_accumulation();

        let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("photon-headless-out"),
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
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let n = frames.max(1);
        let mut last_wall_ms = 0.0_f64;
        for f in 0..n {
            let record = f + 1 == n;
            let t0 = Instant::now();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("photon-headless-enc"),
                });
            self.encode_frame(gpu, &mut enc, &out_view, scene, settings, record);
            gpu.queue.submit(Some(enc.finish()));
            if record {
                gpu.wait_idle()?;
                last_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let pass_ms = if let Some(timers) = &self.timers {
            Some(timers.read_pass_ms(gpu, 3)?)
        } else {
            None
        };
        let rgba = readback_rgba8_texture(gpu, &out_tex, w, h)?;

        let p1 = pass_ms.as_ref().map(|v| v[0]);
        let step_sec = p1.unwrap_or(last_wall_ms) * 1e-3;
        let dispatched = self.dispatched_photons(scene, settings);
        let work_units = match settings.mode {
            PhotonMode::Accumulate => f64::from(dispatched),
            PhotonMode::Pulse => f64::from(dispatched) * f64::from(settings.substeps.max(1)),
        };

        Ok((
            rgba,
            PhotonTelemetry {
                width: w,
                height: h,
                mode: settings.mode,
                dispatched_photons: dispatched,
                substeps: settings.substeps.max(1),
                pass1_step_ms: p1,
                pass2_resolve_ms: pass_ms.as_ref().map(|v| v[1]),
                pass3_agx_ms: pass_ms.as_ref().map(|v| v[2]),
                wall_ms: last_wall_ms,
                photon_throughput_per_sec: if step_sec > 0.0 {
                    work_units / step_sec
                } else {
                    0.0
                },
                accum_frames: self.accum_frames,
            },
        ))
    }

    /// Benchmark forward photon simulation over `measure_iters` timed frames.
    pub fn benchmark(
        &mut self,
        gpu: &GpuContext,
        scene: &PhotonScene,
        settings: &PhotonSettings,
        warmup_iters: usize,
        measure_iters: usize,
    ) -> Result<PhotonTelemetry> {
        let w = self.sized.width;
        let h = self.sized.height;
        self.upload_scene(gpu, scene);
        self.reset_accumulation();

        let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("photon-bench-out"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        for _ in 0..warmup_iters {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("photon-bench-warmup"),
                });
            self.encode_frame(gpu, &mut enc, &out_view, scene, settings, false);
            gpu.queue.submit(Some(enc.finish()));
            gpu.wait_idle()?;
        }

        let mut p1s = Vec::with_capacity(measure_iters);
        let mut p2s = Vec::with_capacity(measure_iters);
        let mut p3s = Vec::with_capacity(measure_iters);
        let mut walls = Vec::with_capacity(measure_iters);

        for _ in 0..measure_iters.max(1) {
            let t0 = Instant::now();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("photon-bench-measure"),
                });
            self.encode_frame(gpu, &mut enc, &out_view, scene, settings, true);
            gpu.queue.submit(Some(enc.finish()));
            gpu.wait_idle()?;
            walls.push(t0.elapsed().as_secs_f64() * 1000.0);

            if let Some(timers) = &self.timers {
                let ms = timers.read_pass_ms(gpu, 3)?;
                p1s.push(ms[0]);
                p2s.push(ms[1]);
                p3s.push(ms[2]);
            }
        }

        let med = |v: &mut [f64]| -> f64 {
            v.sort_by(|a, b| a.total_cmp(b));
            let m = v.len() / 2;
            if v.len() % 2 == 1 {
                v[m]
            } else {
                0.5 * (v[m - 1] + v[m])
            }
        };

        let med_wall = med(&mut walls);
        let med_p1 = (!p1s.is_empty()).then(|| med(&mut p1s));
        let med_p2 = (!p2s.is_empty()).then(|| med(&mut p2s));
        let med_p3 = (!p3s.is_empty()).then(|| med(&mut p3s));
        let step_sec = med_p1.unwrap_or(med_wall) * 1e-3;
        let dispatched = self.dispatched_photons(scene, settings);
        let work_units = match settings.mode {
            PhotonMode::Accumulate => f64::from(dispatched),
            PhotonMode::Pulse => f64::from(dispatched) * f64::from(settings.substeps.max(1)),
        };

        Ok(PhotonTelemetry {
            width: w,
            height: h,
            mode: settings.mode,
            dispatched_photons: dispatched,
            substeps: settings.substeps.max(1),
            pass1_step_ms: med_p1,
            pass2_resolve_ms: med_p2,
            pass3_agx_ms: med_p3,
            wall_ms: med_wall,
            photon_throughput_per_sec: if step_sec > 0.0 {
                work_units / step_sec
            } else {
                0.0
            },
            accum_frames: self.accum_frames,
        })
    }
}

fn bgl_buf(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_buf(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
