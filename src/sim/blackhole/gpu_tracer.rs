//! GPU two-pass cached Ray-Buffer Kerr black hole pipeline + Pass 3 AgX presentation.
//!
//! - Pass 1 (`kerr_trace.wgsl`): Outgoing Cartesian KS RK4 -> split 4K-safe SoA Ray Buffers
//!   (`ray_hits`: 16 B/px + `ray_skies`: 16 B/px).
//! - Pass 2 (`kerr_shade.wgsl`): Page-Thorne disk LUT + log-XYZ CIE blackbody LUT +
//!   Jacobian-filtered sky -> `Rgba16Float` linear HDR target.
//! - Pass 3 (`post_agx.wgsl`): Exposure + AgX tone mapping + debug overlays -> target texture.

use std::time::Instant;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::engine::gpu::{GpuContext, GpuTimers};
use crate::engine::post::{DebugView, PostUniforms, readback_rgba8_texture};

use super::disk::{
    BLACKBODY_LUT_LEN, BLACKBODY_T_MAX, BLACKBODY_T_MIN, BlackbodyLut, DISK_LUT_LEN,
    build_disk_temp_lut,
};
use super::kerr::{Camera, KerrParams};
use super::oracle::RayStatus;

/// Matches `struct TraceUniforms` in `src/shaders/kerr_trace.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TraceUniforms {
    pub cam_pos: [f32; 4],
    pub tet_et: [f32; 4],
    pub tet_er: [f32; 4],
    pub tet_eth: [f32; 4],
    pub tet_ephi: [f32; 4],
    pub params0: [f32; 4],
    pub params1: [f32; 4],
    pub dims: [u32; 4],
}

/// Matches `struct RayHit` in `src/shaders/kerr_trace.wgsl` (16 bytes, native `u32` flags).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRayHit {
    pub r_hit: f32,
    pub phi_hit: f32,
    pub redshift_g: f32,
    pub flags: u32,
}

impl GpuRayHit {
    pub fn status(self) -> RayStatus {
        match self.flags & 3 {
            0 => RayStatus::Captured,
            1 => RayStatus::DiskHit,
            2 => RayStatus::Escaped,
            _ => RayStatus::BudgetExhausted,
        }
    }

    pub fn order(self) -> u32 {
        (self.flags >> 2) & 3
    }

    pub fn steps(self) -> u32 {
        self.flags >> 4
    }
}

/// Matches `struct ShadeUniforms` in `src/shaders/kerr_shade.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadeUniforms {
    pub disk_params: [f32; 4],
    pub lut_params: [f32; 4],
    pub dims: [u32; 4],
}

/// Integrator and rendering settings for the black hole tracer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Near-field radial step factor `eta_near` (inside `r <= 5M`).
    pub eta_near: f32,
    /// Far-field radial step factor `eta_far` (outside `r >= 16M`).
    pub eta_far: f32,
    /// Hard RK4 step budget per ray.
    pub max_steps: u32,
    /// Enable the equatorial Page-Thorne accretion disk.
    pub disk_enabled: bool,
    /// Peak accretion disk temperature scale in Kelvin (visible-band rendering parameter).
    pub disk_peak_temp_k: f32,
    /// Radiance exposure multiplier for Pass 2 & Pass 3.
    pub exposure: f32,
    /// Sky brightness scale.
    pub sky_intensity: f32,
    /// Active output / debug view mode.
    pub debug_view: DebugView,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            eta_near: 0.008,
            eta_far: 0.075,
            max_steps: 2048,
            disk_enabled: true,
            disk_peak_temp_k: 5_500.0,
            exposure: 1.8,
            sky_intensity: 0.35,
            debug_view: DebugView::Beauty,
        }
    }
}

/// Cache key tracking when Pass 1 (geodesic trace) must be re-dispatched.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TraceCacheKey {
    width: u32,
    height: u32,
    radius_bits: u64,
    inclination_bits: u64,
    azimuth_bits: u64,
    vfov_bits: u64,
    spin_bits: u64,
    disk_outer_bits: u64,
    r_escape_bits: u64,
    eta_near_bits: u32,
    eta_far_bits: u32,
    max_steps: u32,
    disk_enabled: bool,
}

impl TraceCacheKey {
    fn from_state(
        width: u32,
        height: u32,
        camera: &Camera,
        params: &KerrParams,
        settings: &RenderSettings,
    ) -> Self {
        Self {
            width,
            height,
            radius_bits: camera.radius.to_bits(),
            inclination_bits: camera.inclination.to_bits(),
            azimuth_bits: camera.azimuth.to_bits(),
            vfov_bits: camera.vfov.to_bits(),
            spin_bits: params.spin.to_bits(),
            disk_outer_bits: params.disk_outer.to_bits(),
            r_escape_bits: params.r_escape.to_bits(),
            eta_near_bits: settings.eta_near.to_bits(),
            eta_far_bits: settings.eta_far.to_bits(),
            max_steps: settings.max_steps,
            disk_enabled: settings.disk_enabled,
        }
    }
}

/// Timing and ray statistics returned by a trace/render or benchmark run.
#[derive(Debug, Clone)]
pub struct FrameTelemetry {
    pub width: u32,
    pub height: u32,
    pub pass1_trace_ms: Option<f64>,
    pub pass2_shade_ms: Option<f64>,
    pub pass3_post_ms: Option<f64>,
    pub wall_ms: f64,
    pub rays_per_sec: f64,
    pub mean_steps: f64,
    pub p95_steps: u32,
    pub max_steps_seen: u32,
    pub simt_efficiency: f64,
    pub captured_count: u64,
    pub disk_hit_count: u64,
    pub escaped_count: u64,
    pub budget_exhausted_count: u64,
}

/// Resolution-dependent GPU buffers & bind groups (rebuilt on resize).
struct SizedResources {
    width: u32,
    height: u32,
    ray_hits_buf: wgpu::Buffer,
    ray_skies_buf: wgpu::Buffer,
    _hdr_tex: wgpu::Texture,
    trace_bg: wgpu::BindGroup,
    shade_bg: wgpu::BindGroup,
    post_bg: wgpu::BindGroup,
}

/// Per-frame scene parameters passed to [`BlackHolePipeline::encode_frame`].
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    pub camera: &'a Camera,
    pub params: &'a KerrParams,
    pub settings: &'a RenderSettings,
    pub display_aspect: f64,
}

/// Complete 3-pass Kerr black hole GPU pipeline.
pub struct BlackHolePipeline {
    trace_pipeline: wgpu::ComputePipeline,
    shade_pipeline: wgpu::ComputePipeline,
    post_pipeline: wgpu::RenderPipeline,
    trace_bgl: wgpu::BindGroupLayout,
    shade_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    trace_uniform_buf: wgpu::Buffer,
    shade_uniform_buf: wgpu::Buffer,
    post_uniform_buf: wgpu::Buffer,
    disk_lut_buf: wgpu::Buffer,
    blackbody_lut_buf: wgpu::Buffer,
    hdr_sampler: wgpu::Sampler,
    sized: SizedResources,
    timers: Option<GpuTimers>,
    last_trace_key: Option<TraceCacheKey>,
    last_disk_lut_key: Option<(u64, u64)>,
}

impl BlackHolePipeline {
    /// Create all pipelines, static LUT buffers, and initial resolution resources.
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let trace_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kerr_trace.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/kerr_trace.wgsl").into(),
                ),
            });
        let shade_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kerr_shade.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/kerr_shade.wgsl").into(),
                ),
            });
        let post_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post_agx.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/post_agx.wgsl").into(),
                ),
            });

        let trace_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kerr-trace-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shade_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kerr-shade-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let post_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post-agx-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let trace_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("kerr-trace-layout"),
                bind_group_layouts: &[Some(&trace_bgl)],
                immediate_size: 0,
            });
        let trace_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kerr-trace-pipeline"),
                layout: Some(&trace_layout),
                module: &trace_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let shade_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("kerr-shade-layout"),
                bind_group_layouts: &[Some(&shade_bgl)],
                immediate_size: 0,
            });
        let shade_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kerr-shade-pipeline"),
                layout: Some(&shade_layout),
                module: &shade_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let post_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post-agx-layout"),
                bind_group_layouts: &[Some(&post_bgl)],
                immediate_size: 0,
            });
        let post_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post-agx-pipeline"),
                layout: Some(&post_layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let trace_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trace-uniforms"),
            size: std::mem::size_of::<TraceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shade_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade-uniforms"),
            size: std::mem::size_of::<ShadeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post-uniforms"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_disk_lut = vec![0.0_f32; DISK_LUT_LEN];
        let disk_lut_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("page-thorne-lut"),
                contents: bytemuck::cast_slice(&init_disk_lut),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let bb_lut = BlackbodyLut::build();
        let blackbody_lut_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cie-blackbody-lut"),
                contents: bytemuck::cast_slice(&bb_lut.entries),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let hdr_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr-linear-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sized = Self::create_sized(
            gpu,
            width.max(1),
            height.max(1),
            &trace_bgl,
            &shade_bgl,
            &post_bgl,
            &trace_uniform_buf,
            &shade_uniform_buf,
            &post_uniform_buf,
            &disk_lut_buf,
            &blackbody_lut_buf,
            &hdr_sampler,
        );
        let timers = GpuTimers::new(gpu);

        Self {
            trace_pipeline,
            shade_pipeline,
            post_pipeline,
            trace_bgl,
            shade_bgl,
            post_bgl,
            trace_uniform_buf,
            shade_uniform_buf,
            post_uniform_buf,
            disk_lut_buf,
            blackbody_lut_buf,
            hdr_sampler,
            sized,
            timers,
            last_trace_key: None,
            last_disk_lut_key: None,
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
        trace_bgl: &wgpu::BindGroupLayout,
        shade_bgl: &wgpu::BindGroupLayout,
        post_bgl: &wgpu::BindGroupLayout,
        trace_uniform_buf: &wgpu::Buffer,
        shade_uniform_buf: &wgpu::Buffer,
        post_uniform_buf: &wgpu::Buffer,
        disk_lut_buf: &wgpu::Buffer,
        blackbody_lut_buf: &wgpu::Buffer,
        hdr_sampler: &wgpu::Sampler,
    ) -> SizedResources {
        let num_pixels = u64::from(width) * u64::from(height);
        let hit_bytes = num_pixels * (std::mem::size_of::<GpuRayHit>() as u64);
        let sky_bytes = num_pixels * 16;

        let ray_hits_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-hits-soa"),
            size: hit_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ray_skies_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-skies-soa"),
            size: sky_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hdr_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr-target"),
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

        let trace_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr-trace-bg"),
            layout: trace_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: trace_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ray_hits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ray_skies_buf.as_entire_binding(),
                },
            ],
        });

        let shade_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr-shade-bg"),
            layout: shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shade_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ray_hits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ray_skies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: disk_lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: blackbody_lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
            ],
        });

        let post_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post-agx-bg"),
            layout: post_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: post_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(hdr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ray_hits_buf.as_entire_binding(),
                },
            ],
        });

        SizedResources {
            width,
            height,
            ray_hits_buf,
            ray_skies_buf,
            _hdr_tex: hdr_tex,
            trace_bg,
            shade_bg,
            post_bg,
        }
    }

    /// Resize the internal Ray Buffers and HDR texture if dimensions changed.
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
            &self.trace_bgl,
            &self.shade_bgl,
            &self.post_bgl,
            &self.trace_uniform_buf,
            &self.shade_uniform_buf,
            &self.post_uniform_buf,
            &self.disk_lut_buf,
            &self.blackbody_lut_buf,
            &self.hdr_sampler,
        );
        self.last_trace_key = None;
    }

    /// Force Pass 1 to re-run on the next render call.
    pub fn invalidate_cache(&mut self) {
        self.last_trace_key = None;
    }

    /// Upload Page-Thorne LUT if spin or outer disk radius changed.
    fn sync_disk_lut(&mut self, gpu: &GpuContext, params: &KerrParams) {
        let key = (params.spin.to_bits(), params.disk_outer.to_bits());
        if self.last_disk_lut_key == Some(key) {
            return;
        }
        let lut = build_disk_temp_lut(params.spin, params.disk_outer);
        gpu.queue
            .write_buffer(&self.disk_lut_buf, 0, bytemuck::cast_slice(&lut));
        self.last_disk_lut_key = Some(key);
    }

    fn write_uniforms(
        &self,
        gpu: &GpuContext,
        camera: &Camera,
        params: &KerrParams,
        settings: &RenderSettings,
        display_aspect: f64,
    ) {
        let tet = camera.zamo_tetrad(params.spin);
        let trace_u = TraceUniforms {
            cam_pos: [
                tet.pos.x as f32,
                tet.pos.y as f32,
                tet.pos.z as f32,
                params.spin as f32,
            ],
            tet_et: [
                tet.e_t.x as f32,
                tet.e_t.y as f32,
                tet.e_t.z as f32,
                tet.e_t.w as f32,
            ],
            tet_er: [
                tet.e_r.x as f32,
                tet.e_r.y as f32,
                tet.e_r.z as f32,
                tet.e_r.w as f32,
            ],
            tet_eth: [
                tet.e_theta.x as f32,
                tet.e_theta.y as f32,
                tet.e_theta.z as f32,
                tet.e_theta.w as f32,
            ],
            tet_ephi: [
                tet.e_phi.x as f32,
                tet.e_phi.y as f32,
                tet.e_phi.z as f32,
                tet.e_phi.w as f32,
            ],
            params0: [
                params.r_plus() as f32,
                params.r_isco() as f32,
                params.disk_outer as f32,
                params.r_escape as f32,
            ],
            params1: [
                (0.5 * camera.vfov).tan() as f32,
                display_aspect as f32,
                settings.eta_near,
                settings.eta_far,
            ],
            dims: [
                self.sized.width,
                self.sized.height,
                settings.max_steps,
                u32::from(settings.disk_enabled),
            ],
        };
        gpu.queue
            .write_buffer(&self.trace_uniform_buf, 0, bytemuck::bytes_of(&trace_u));

        let shade_u = ShadeUniforms {
            disk_params: [
                params.r_isco() as f32,
                params.disk_outer as f32,
                settings.disk_peak_temp_k,
                settings.exposure,
            ],
            lut_params: [
                BLACKBODY_T_MIN.ln() as f32,
                BLACKBODY_T_MAX.ln() as f32,
                BLACKBODY_T_MIN as f32,
                settings.sky_intensity,
            ],
            dims: [
                self.sized.width,
                self.sized.height,
                DISK_LUT_LEN as u32,
                BLACKBODY_LUT_LEN as u32,
            ],
        };
        gpu.queue
            .write_buffer(&self.shade_uniform_buf, 0, bytemuck::bytes_of(&shade_u));

        let post_u = PostUniforms {
            params: [1.0, (settings.max_steps as f32) * 0.5, 0.0, 0.0],
            dims: [
                self.sized.width,
                self.sized.height,
                settings.debug_view as u32,
                0,
            ],
        };
        gpu.queue
            .write_buffer(&self.post_uniform_buf, 0, bytemuck::bytes_of(&post_u));
    }

    /// Encode Pass 1 (only when cache invalidated), Pass 2 (shading), and Pass 3 (post -> `target_view`).
    ///
    /// Returns `true` if Pass 1 (geodesic trace) was dispatched on this frame.
    pub fn encode_frame(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        view: FrameView<'_>,
        record_timestamps: bool,
    ) -> bool {
        self.sync_disk_lut(gpu, view.params);
        self.write_uniforms(
            gpu,
            view.camera,
            view.params,
            view.settings,
            view.display_aspect,
        );

        let key = TraceCacheKey::from_state(
            self.sized.width,
            self.sized.height,
            view.camera,
            view.params,
            view.settings,
        );
        let need_trace = self.last_trace_key != Some(key);
        let wg_x = self.sized.width.div_ceil(8);
        let wg_y = self.sized.height.div_ceil(8);

        if need_trace {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass1-kerr-trace"),
                timestamp_writes: if record_timestamps {
                    self.timers.as_ref().map(|t| t.compute_writes(0))
                } else {
                    None
                },
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.sized.trace_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            self.last_trace_key = Some(key);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass2-kerr-shade"),
                timestamp_writes: if record_timestamps {
                    self.timers.as_ref().map(|t| t.compute_writes(1))
                } else {
                    None
                },
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &self.sized.shade_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass3-post-agx"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: if record_timestamps {
                    self.timers.as_ref().map(|t| t.render_writes(2))
                } else {
                    None
                },
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &self.sized.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if record_timestamps && let Some(timers) = &self.timers {
            timers.resolve(encoder, GpuTimers::SLOTS);
        }

        need_trace
    }

    /// Run Pass 1 headlessly and read back the linear `ray_hits` and `ray_skies` buffers
    /// for cross-formulation physics verification against the CPU `f64` oracle.
    pub fn trace_and_readback(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        params: &KerrParams,
        settings: &RenderSettings,
    ) -> Result<(Vec<GpuRayHit>, Vec<[f32; 4]>)> {
        let aspect = f64::from(self.sized.width) / f64::from(self.sized.height);
        self.write_uniforms(gpu, camera, params, settings, aspect);
        self.last_trace_key = None;

        let num_pixels = usize::try_from(self.sized.width).expect("width fits usize")
            * usize::try_from(self.sized.height).expect("height fits usize");
        let hit_bytes = (num_pixels * std::mem::size_of::<GpuRayHit>()) as u64;
        let sky_bytes = (num_pixels * 16) as u64;

        let hit_stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-hits-readback"),
            size: hit_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let sky_stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-skies-readback"),
            size: sky_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trace-readback-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass1-verify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.sized.trace_bg, &[]);
            pass.dispatch_workgroups(
                self.sized.width.div_ceil(8),
                self.sized.height.div_ceil(8),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.sized.ray_hits_buf, 0, &hit_stage, 0, hit_bytes);
        encoder.copy_buffer_to_buffer(&self.sized.ray_skies_buf, 0, &sky_stage, 0, sky_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let hits = map_read_vec::<GpuRayHit>(gpu, &hit_stage)?;
        let skies = map_read_vec::<[f32; 4]>(gpu, &sky_stage)?;
        Ok((hits, skies))
    }

    /// Render a complete headless frame (Pass 1 + Pass 2 + Pass 3), returning RGBA8 pixels
    /// and GPU/ray telemetry.
    pub fn render_headless(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        params: &KerrParams,
        settings: &RenderSettings,
        collect_ray_stats: bool,
    ) -> Result<(Vec<u8>, FrameTelemetry)> {
        let w = self.sized.width;
        let h = self.sized.height;
        let aspect = f64::from(w) / f64::from(h);

        let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-out-rgba8"),
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

        self.invalidate_cache();
        let start = Instant::now();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-frame-encoder"),
            });
        self.encode_frame(
            gpu,
            &mut encoder,
            &out_view,
            FrameView {
                camera,
                params,
                settings,
                display_aspect: aspect,
            },
            true,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.wait_idle()?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

        let pass_ms = if let Some(timers) = &self.timers {
            Some(timers.read_pass_ms(gpu, 3)?)
        } else {
            None
        };

        let rgba = readback_rgba8_texture(gpu, &out_tex, w, h)?;

        let (mean_steps, p95_steps, max_steps_seen, simt_eff, counts) = if collect_ray_stats {
            let num_pixels = u64::from(w) * u64::from(h);
            let hit_bytes = num_pixels * (std::mem::size_of::<GpuRayHit>() as u64);
            let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stats-readback"),
                size: hit_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("stats-copy"),
                });
            enc.copy_buffer_to_buffer(&self.sized.ray_hits_buf, 0, &stage, 0, hit_bytes);
            gpu.queue.submit(Some(enc.finish()));
            let hits = map_read_vec::<GpuRayHit>(gpu, &stage)?;
            compute_ray_stats(&hits, w, h)
        } else {
            (0.0, 0, 0, 0.0, [0; 4])
        };

        let pass1_ms = pass_ms.as_ref().map(|v| v[0]);
        let trace_sec = pass1_ms.unwrap_or(wall_ms) * 1e-3;
        let rays_per_sec = if trace_sec > 0.0 {
            (f64::from(w) * f64::from(h)) / trace_sec
        } else {
            0.0
        };

        let telemetry = FrameTelemetry {
            width: w,
            height: h,
            pass1_trace_ms: pass1_ms,
            pass2_shade_ms: pass_ms.as_ref().map(|v| v[1]),
            pass3_post_ms: pass_ms.as_ref().map(|v| v[2]),
            wall_ms,
            rays_per_sec,
            mean_steps,
            p95_steps,
            max_steps_seen,
            simt_efficiency: simt_eff,
            captured_count: counts[0],
            disk_hit_count: counts[1],
            escaped_count: counts[2],
            budget_exhausted_count: counts[3],
        };

        Ok((rgba, telemetry))
    }

    /// Run a multi-iteration headless GPU benchmark (Pass 1 + Pass 2 + Pass 3), returning
    /// median pass timings plus exact ray statistics.
    pub fn benchmark(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        params: &KerrParams,
        settings: &RenderSettings,
        warmup_iters: usize,
        measure_iters: usize,
    ) -> Result<FrameTelemetry> {
        let w = self.sized.width;
        let h = self.sized.height;
        let aspect = f64::from(w) / f64::from(h);

        let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench-out-rgba8"),
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
            self.invalidate_cache();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench-warmup"),
                });
            self.encode_frame(
                gpu,
                &mut enc,
                &out_view,
                FrameView {
                    camera,
                    params,
                    settings,
                    display_aspect: aspect,
                },
                false,
            );
            gpu.queue.submit(Some(enc.finish()));
            gpu.wait_idle()?;
        }

        let mut p1_samples = Vec::with_capacity(measure_iters);
        let mut p2_samples = Vec::with_capacity(measure_iters);
        let mut p3_samples = Vec::with_capacity(measure_iters);
        let mut wall_samples = Vec::with_capacity(measure_iters);

        for _ in 0..measure_iters.max(1) {
            self.invalidate_cache();
            let start = Instant::now();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench-measure"),
                });
            self.encode_frame(
                gpu,
                &mut enc,
                &out_view,
                FrameView {
                    camera,
                    params,
                    settings,
                    display_aspect: aspect,
                },
                true,
            );
            gpu.queue.submit(Some(enc.finish()));
            gpu.wait_idle()?;
            wall_samples.push(start.elapsed().as_secs_f64() * 1000.0);

            if let Some(timers) = &self.timers {
                let ms = timers.read_pass_ms(gpu, 3)?;
                p1_samples.push(ms[0]);
                p2_samples.push(ms[1]);
                p3_samples.push(ms[2]);
            }
        }

        // Read back ray hits once for step & classification telemetry.
        let num_pixels = u64::from(w) * u64::from(h);
        let hit_bytes = num_pixels * (std::mem::size_of::<GpuRayHit>() as u64);
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-stats-readback"),
            size: hit_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench-stats-copy"),
            });
        enc.copy_buffer_to_buffer(&self.sized.ray_hits_buf, 0, &stage, 0, hit_bytes);
        gpu.queue.submit(Some(enc.finish()));
        let hits = map_read_vec::<GpuRayHit>(gpu, &stage)?;
        let (mean_steps, p95_steps, max_steps_seen, simt_eff, counts) =
            compute_ray_stats(&hits, w, h);

        let med_wall = median(&mut wall_samples);
        let med_p1 = (!p1_samples.is_empty()).then(|| median(&mut p1_samples));
        let med_p2 = (!p2_samples.is_empty()).then(|| median(&mut p2_samples));
        let med_p3 = (!p3_samples.is_empty()).then(|| median(&mut p3_samples));
        let trace_sec = med_p1.unwrap_or(med_wall) * 1e-3;

        Ok(FrameTelemetry {
            width: w,
            height: h,
            pass1_trace_ms: med_p1,
            pass2_shade_ms: med_p2,
            pass3_post_ms: med_p3,
            wall_ms: med_wall,
            rays_per_sec: if trace_sec > 0.0 {
                (f64::from(w) * f64::from(h)) / trace_sec
            } else {
                0.0
            },
            mean_steps,
            p95_steps,
            max_steps_seen,
            simt_efficiency: simt_eff,
            captured_count: counts[0],
            disk_hit_count: counts[1],
            escaped_count: counts[2],
            budget_exhausted_count: counts[3],
        })
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

fn map_read_vec<T: Pod>(gpu: &GpuContext, buf: &wgpu::Buffer) -> Result<Vec<T>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.wait_idle()?;
    rx.recv()
        .context("receive buffer map result")?
        .context("map readback buffer")?;
    let view = slice
        .get_mapped_range()
        .context("get readback mapped range")?;
    let out = bytemuck::cast_slice::<u8, T>(&view).to_vec();
    drop(view);
    buf.unmap();
    Ok(out)
}

fn compute_ray_stats(
    hits: &[GpuRayHit],
    width: u32,
    height: u32,
) -> (f64, u32, u32, f64, [u64; 4]) {
    let mut counts = [0_u64; 4];
    let mut steps_vec = Vec::with_capacity(hits.len());
    let mut sum_steps = 0_u64;
    let mut max_steps = 0_u32;

    for hit in hits {
        let s = hit.steps();
        sum_steps += u64::from(s);
        max_steps = max_steps.max(s);
        steps_vec.push(s);
        counts[(hit.flags & 3) as usize] += 1;
    }

    steps_vec.sort_unstable();
    let p95 = steps_vec
        .get(((steps_vec.len() as f64) * 0.95) as usize)
        .copied()
        .unwrap_or(0);
    let mean_steps = if hits.is_empty() {
        0.0
    } else {
        (sum_steps as f64) / (hits.len() as f64)
    };

    // Estimate 8x8 tile SIMT efficiency: sum(steps) / sum(tile_size * tile_max_steps).
    let w = width as usize;
    let h = height as usize;
    let mut tile_work_denom = 0_u64;
    for ty in (0..h).step_by(8) {
        let y_end = (ty + 8).min(h);
        for tx in (0..w).step_by(8) {
            let x_end = (tx + 8).min(w);
            let mut tile_max = 0_u32;
            let mut tile_px = 0_u64;
            for y in ty..y_end {
                for x in tx..x_end {
                    tile_max = tile_max.max(hits[y * w + x].steps());
                    tile_px += 1;
                }
            }
            tile_work_denom += tile_px * u64::from(tile_max);
        }
    }
    let simt_eff = if tile_work_denom > 0 {
        (sum_steps as f64) / (tile_work_denom as f64)
    } else {
        0.0
    };

    (mean_steps, p95, max_steps, simt_eff, counts)
}
