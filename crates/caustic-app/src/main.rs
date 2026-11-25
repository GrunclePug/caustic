//! Caustic CLI and unified interactive studio: general-relativistic Kerr black hole
//! tracer and real-time forward spectral photon / GRIN optics simulator.

mod studio;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glam::DVec3;

use caustic_blackhole::gpu_tracer::{BlackHolePipeline, FrameTelemetry, RenderSettings};
use caustic_blackhole::kerr::{
    Camera, KerrParams, MAX_CAMERA_RADIUS, MAX_SPIN, MIN_CAMERA_HORIZON_MARGIN, r_isco, r_plus,
};
use caustic_blackhole::oracle::{RayStatus, trace_oracle_pixel};
use caustic_core::gpu::GpuContext;
use caustic_core::post::{DebugView, write_rgba8_png};
use caustic_photons::gpu_sim::{PhotonPipeline, PhotonSettings, PhotonTelemetry};
use caustic_photons::scene::{PhotonMode, PhotonScene, PhotonSceneId};
use caustic_photons::solar::{
    BODY_EARTH, BODY_IO, BODY_JUPITER, BODY_MARS, BODY_MERCURY, BODY_MOON, BODY_NEPTUNE, BODY_SUN,
    BODY_URANUS, BODY_VENUS, SolarFocus, SolarScale,
};

use crate::studio::{StudioConfig, StudioMode, run_studio};

#[derive(Debug, Parser)]
#[command(
    name = "caustic",
    version,
    about = "GPU-accelerated general-relativistic Kerr black hole, GRIN optics, and spectral photon simulator"
)]
struct Cli {
    /// Force software Vulkan adapter (lavapipe) instead of hardware GPU.
    #[arg(long, global = true)]
    software: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DebugViewArg {
    Beauty,
    Steps,
    Redshift,
    HitClass,
}

impl From<DebugViewArg> for DebugView {
    fn from(value: DebugViewArg) -> Self {
        match value {
            DebugViewArg::Beauty => Self::Beauty,
            DebugViewArg::Steps => Self::StepHeatmap,
            DebugViewArg::Redshift => Self::Redshift,
            DebugViewArg::HitClass => Self::HitClass,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PhotonSceneArg {
    Prism,
    Cardioid,
    Lens,
    SolarSystem,
    Sandbox,
    Luneburg,
    OpticalBh,
    Mirage,
    KerrBeam,
    KerrPulsar,
}

impl From<PhotonSceneArg> for PhotonSceneId {
    fn from(value: PhotonSceneArg) -> Self {
        match value {
            PhotonSceneArg::Prism => Self::Prism,
            PhotonSceneArg::Cardioid => Self::Cardioid,
            PhotonSceneArg::Lens => Self::Lens,
            PhotonSceneArg::SolarSystem => Self::SolarSystem,
            PhotonSceneArg::Sandbox => Self::Sandbox,
            PhotonSceneArg::Luneburg => Self::GrinLuneburg,
            PhotonSceneArg::OpticalBh => Self::GrinBlackHole,
            PhotonSceneArg::Mirage => Self::GrinMirage,
            PhotonSceneArg::KerrBeam => Self::KerrBeam,
            PhotonSceneArg::KerrPulsar => Self::KerrPulsar,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PhotonModeArg {
    Accumulate,
    Pulse,
}

impl From<PhotonModeArg> for PhotonMode {
    fn from(value: PhotonModeArg) -> Self {
        match value {
            PhotonModeArg::Accumulate => Self::Accumulate,
            PhotonModeArg::Pulse => Self::Pulse,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SolarFocusArg {
    LunarEclipse,
    SolarEclipse,
    Saturn,
    IoTransit,
    System,
    Inner,
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Io,
    Uranus,
    Neptune,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SolarScaleArg {
    Real,
    #[value(alias = "cartoon")]
    Orrery,
}

impl From<SolarScaleArg> for SolarScale {
    fn from(value: SolarScaleArg) -> Self {
        match value {
            SolarScaleArg::Real => Self::Real1To1,
            SolarScaleArg::Orrery => Self::Orrery,
        }
    }
}

#[derive(Debug, Clone, Args)]
struct SceneArgs {
    /// Dimensionless Kerr spin a in [-0.998, +0.998] (0 = Schwarzschild).
    #[arg(long, default_value_t = 0.95, allow_hyphen_values = true)]
    spin: f64,

    /// Camera Boyer-Lindquist radius in units of M.
    #[arg(long, default_value_t = 32.0)]
    radius: f64,

    /// Camera polar inclination in degrees (0 = north pole face-on, 90 = equatorial edge-on).
    #[arg(long, default_value_t = 82.0)]
    inclination: f64,

    /// Camera azimuth in degrees.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    azimuth: f64,

    /// Vertical field of view in degrees.
    #[arg(long, default_value_t = 45.0)]
    fov: f64,

    /// Accretion disk outer radius in units of M.
    #[arg(long, default_value_t = 16.0)]
    disk_outer: f64,

    /// Disable the equatorial accretion disk (pure shadow + lensed sky).
    #[arg(long)]
    no_disk: bool,

    /// Peak accretion disk blackbody temperature scale in Kelvin (visible-band parameter).
    #[arg(long, default_value_t = 5500.0)]
    temp: f32,

    /// Exposure multiplier.
    #[arg(long, default_value_t = 1.8)]
    exposure: f32,

    /// Near-field RK4 radial step factor eta_near (r <= 3.5M).
    #[arg(long, default_value_t = 0.008)]
    eta_near: f32,

    /// Far-field RK4 radial step factor eta_far (r >= 14M).
    #[arg(long, default_value_t = 0.075)]
    eta_far: f32,

    /// Maximum RK4 steps per ray.
    #[arg(long, default_value_t = 2048)]
    max_steps: u32,

    /// Output visualization mode.
    #[arg(long, value_enum, default_value_t = DebugViewArg::Beauty)]
    debug: DebugViewArg,
}

impl SceneArgs {
    fn to_parts(&self) -> Result<(Camera, KerrParams, RenderSettings)> {
        anyhow::ensure!(
            self.spin.abs() <= MAX_SPIN,
            "spin {} out of supported range [-{MAX_SPIN}, +{MAX_SPIN}]",
            self.spin
        );
        let min_r = r_plus(self.spin) + MIN_CAMERA_HORIZON_MARGIN;
        anyhow::ensure!(
            self.radius >= min_r && self.radius <= MAX_CAMERA_RADIUS,
            "camera radius {} out of supported range [{min_r:.3}, {MAX_CAMERA_RADIUS}]",
            self.radius
        );
        let params = KerrParams::new(self.spin, self.disk_outer, 1000.0);
        let camera = Camera {
            radius: self.radius,
            inclination: self.inclination.clamp(0.0, 180.0).to_radians(),
            azimuth: self.azimuth.to_radians(),
            vfov: self.fov.clamp(5.0, 140.0).to_radians(),
        };
        let settings = RenderSettings {
            eta_near: self.eta_near,
            eta_far: self.eta_far,
            max_steps: self.max_steps,
            disk_enabled: !self.no_disk,
            disk_peak_temp_k: self.temp,
            exposure: self.exposure,
            debug_view: self.debug.into(),
            ..RenderSettings::default()
        };
        Ok((camera, params, settings))
    }
}

#[derive(Debug, Clone, Args)]
struct PhotonArgs {
    /// Forward photon scene preset.
    #[arg(long, value_enum, default_value_t = PhotonSceneArg::Prism)]
    scene: PhotonSceneArg,

    /// Accumulation (long-exposure caustics) or live Pulse wavefront flight.
    #[arg(long, value_enum, default_value_t = PhotonModeArg::Accumulate)]
    mode: PhotonModeArg,

    /// Number of simultaneously simulated forward photon packets.
    #[arg(long, default_value_t = 1_000_000)]
    photons: u32,

    /// Simulation substeps per frame.
    #[arg(long, default_value_t = 4)]
    substeps: u32,

    /// Kerr spin a in [-0.998, +0.998] (for kerr-beam / kerr-pulsar).
    #[arg(long, default_value_t = 0.95, allow_hyphen_values = true)]
    spin: f32,

    /// Glass / GRIN dispersion scale multiplier (leave unset to keep preset default).
    #[arg(long)]
    dispersion: Option<f32>,

    /// GRIN field strength multiplier (for luneburg / optical-bh / mirage).
    #[arg(long, default_value_t = 1.0)]
    strength: f32,

    /// Exposure multiplier.
    #[arg(long, default_value_t = 1.0)]
    exposure: f32,

    /// Deterministic PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Solar System camera focus or staged astronomical event (`--scene solar-system`).
    #[arg(long, value_enum)]
    solar_focus: Option<SolarFocusArg>,

    /// Solar System scale mode (`orrery` / `cartoon` = exaggerated overview, `real` = exact 1:1 NASA ratios).
    #[arg(long, value_enum, default_value_t = SolarScaleArg::Orrery)]
    solar_scale: SolarScaleArg,
}

impl PhotonArgs {
    fn to_parts(&self, aspect: f64) -> (PhotonScene, PhotonSettings) {
        let mut scene = PhotonScene::preset(self.scene.into());
        scene.spin = self.spin.clamp(-0.998, 0.998);
        scene.grin_strength = self.strength.clamp(0.1, 3.0);
        if let Some(d) = self.dispersion {
            scene.dispersion_scale = d.clamp(0.0, 4.0);
        }
        if scene.id.is_solar_system() {
            scene.solar.set_scale(self.solar_scale.into());
            if let Some(focus) = self.solar_focus {
                match focus {
                    SolarFocusArg::LunarEclipse => scene.solar.stage_lunar_eclipse(),
                    SolarFocusArg::SolarEclipse => scene.solar.stage_solar_eclipse(),
                    SolarFocusArg::Saturn => scene.solar.stage_saturn_rings(),
                    SolarFocusArg::IoTransit => scene.solar.stage_jupiter_io_transit(),
                    SolarFocusArg::System => scene.solar.set_focus(SolarFocus::System),
                    SolarFocusArg::Inner => scene.solar.set_focus(SolarFocus::Inner),
                    SolarFocusArg::Sun => scene.solar.set_focus(SolarFocus::Body(BODY_SUN)),
                    SolarFocusArg::Mercury => scene.solar.set_focus(SolarFocus::Body(BODY_MERCURY)),
                    SolarFocusArg::Venus => scene.solar.set_focus(SolarFocus::Body(BODY_VENUS)),
                    SolarFocusArg::Earth => scene.solar.set_focus(SolarFocus::Body(BODY_EARTH)),
                    SolarFocusArg::Moon => scene.solar.set_focus(SolarFocus::Body(BODY_MOON)),
                    SolarFocusArg::Mars => scene.solar.set_focus(SolarFocus::Body(BODY_MARS)),
                    SolarFocusArg::Jupiter => scene.solar.set_focus(SolarFocus::Body(BODY_JUPITER)),
                    SolarFocusArg::Io => scene.solar.set_focus(SolarFocus::Body(BODY_IO)),
                    SolarFocusArg::Uranus => scene.solar.set_focus(SolarFocus::Body(BODY_URANUS)),
                    SolarFocusArg::Neptune => scene.solar.set_focus(SolarFocus::Body(BODY_NEPTUNE)),
                }
            }
            scene.sync_solar_to_gpu(aspect);
        } else if scene.id.is_optics() {
            scene.compile_elements();
        }
        let settings = PhotonSettings {
            num_photons: self.photons.max(256),
            substeps: self.substeps.max(1),
            mode: self.mode.into(),
            exposure: self.exposure,
            seed: self.seed,
            ..PhotonSettings::default()
        };
        (scene, settings)
    }
}

#[derive(Debug, Subcommand)]
enum PhotonCommand {
    /// Launch the interactive real-time forward photon viewer.
    View {
        #[command(flatten)]
        args: PhotonArgs,
        /// Initial window width in pixels.
        #[arg(long, default_value_t = 1280)]
        width: u32,
        /// Initial window height in pixels.
        #[arg(long, default_value_t = 720)]
        height: u32,
    },
    /// Step the forward photon simulation headlessly and write a PNG screenshot.
    Render {
        #[command(flatten)]
        args: PhotonArgs,
        /// Output image width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,
        /// Output image height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,
        /// Number of frames to step before capturing.
        #[arg(long, default_value_t = 120)]
        frames: u32,
        /// Output PNG file path.
        #[arg(short, long, default_value = "photons.png")]
        output: PathBuf,
    },
    /// Benchmark forward photon simulation throughput on the active GPU.
    Bench {
        #[command(flatten)]
        args: PhotonArgs,
        /// Output image width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,
        /// Output image height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,
        /// Number of timed iterations.
        #[arg(long, default_value_t = 25)]
        iters: usize,
    },
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Launch the interactive windowed Kerr black hole orbit viewer.
    View {
        #[command(flatten)]
        scene: SceneArgs,
        /// Initial window width in pixels.
        #[arg(long, default_value_t = 1280)]
        width: u32,
        /// Initial window height in pixels.
        #[arg(long, default_value_t = 720)]
        height: u32,
    },
    /// Render a single deterministic Kerr black hole frame headlessly to PNG.
    Render {
        #[command(flatten)]
        scene: SceneArgs,
        /// Output image width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,
        /// Output image height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,
        /// Output PNG file path.
        #[arg(short, long, default_value = "render.png")]
        output: PathBuf,
    },
    /// Run the headless Kerr black hole GPU benchmark gate across 720p, 1440p, and 4K.
    Bench {
        #[command(flatten)]
        scene: SceneArgs,
        /// Number of timed iterations per resolution.
        #[arg(long, default_value_t = 9)]
        iters: usize,
    },
    /// Run cross-formulation Kerr GPU vs CPU f64 oracle accuracy verification across views.
    Verify {
        /// Grid width (default 512 for hardware GPU gate, use 64 for fast lavapipe check).
        #[arg(long, default_value_t = 512)]
        width: u32,
        /// Grid height.
        #[arg(long, default_value_t = 512)]
        height: u32,
        /// Near-field RK4 radial step factor eta_near.
        #[arg(long, default_value_t = 0.008)]
        eta_near: f32,
        /// Far-field RK4 radial step factor eta_far.
        #[arg(long, default_value_t = 0.075)]
        eta_far: f32,
        /// Maximum RK4 steps per ray.
        #[arg(long, default_value_t = 2048)]
        max_steps: u32,
    },
    /// Forward spectral photon simulator (dispersive optics & relativistic Kerr wavefronts).
    Photons {
        #[command(subcommand)]
        command: PhotonCommand,
    },
}

fn print_telemetry(label: &str, t: &FrameTelemetry) {
    let total = u64::from(t.width) * u64::from(t.height);
    let exh_pct = if total > 0 {
        (t.budget_exhausted_count as f64) * 100.0 / (total as f64)
    } else {
        0.0
    };
    let p1 = t
        .pass1_trace_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    let p2 = t
        .pass2_shade_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    let p3 = t
        .pass3_post_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    println!(
        "{label} ({}x{}): Pass1={p1}, Pass2={p2}, Pass3={p3}, wall={:.2} ms, {:.1} Mrays/s | steps mean={:.1} p95={} max={} SIMT={:.1}% | hit={} cap={} sky={} exh={} ({exh_pct:.3}%)",
        t.width,
        t.height,
        t.wall_ms,
        t.rays_per_sec * 1e-6,
        t.mean_steps,
        t.p95_steps,
        t.max_steps_seen,
        t.simt_efficiency * 100.0,
        t.disk_hit_count,
        t.captured_count,
        t.escaped_count,
        t.budget_exhausted_count,
    );
}

fn print_photon_telemetry(label: &str, t: &PhotonTelemetry) {
    let p1 = t
        .pass1_step_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    let p2 = t
        .pass2_resolve_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    let p3 = t
        .pass3_agx_ms
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a".into());
    match t.mode {
        PhotonMode::Accumulate => println!(
            "{label} ({}x{}, {:.1}K full paths/f): Step={p1}, Resolve={p2}, AgX={p3}, wall={:.2} ms, {:.2} M full paths/s | frames={}",
            t.width,
            t.height,
            f64::from(t.dispatched_photons) * 1e-3,
            t.wall_ms,
            t.photon_throughput_per_sec * 1e-6,
            t.accum_frames,
        ),
        PhotonMode::Pulse => println!(
            "{label} ({}x{}, {:.2}M photons x{} substeps): Step={p1}, Resolve={p2}, AgX={p3}, wall={:.2} ms, {:.2} B photon-steps/s | frames={}",
            t.width,
            t.height,
            f64::from(t.dispatched_photons) * 1e-6,
            t.substeps,
            t.wall_ms,
            t.photon_throughput_per_sec * 1e-9,
            t.accum_frames,
        ),
    }
}

fn angle_wrap(a: f64, b: f64) -> f64 {
    (a - b + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn run_verify_view(
    gpu: &GpuContext,
    label: &str,
    camera: Camera,
    params: KerrParams,
    settings: RenderSettings,
    width: u32,
    height: u32,
) -> Result<()> {
    let t0 = Instant::now();
    let mut pipeline =
        BlackHolePipeline::new(gpu, width, height, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (gpu_hits, gpu_skies) = pipeline.trace_and_readback(gpu, &camera, &params, &settings)?;
    let aspect = f64::from(width) / f64::from(height);
    let w = width as usize;
    let h = height as usize;
    let total = w * h;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(h.max(1));
    let rows_per_chunk = h.div_ceil(threads);
    let mut cpu_results = Vec::with_capacity(total);
    let chunks: Vec<_> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let y0 = t * rows_per_chunk;
            let y1 = ((t + 1) * rows_per_chunk).min(h);
            if y0 >= y1 {
                continue;
            }
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity((y1 - y0) * w);
                for py in y0..y1 {
                    let ndc_y = 1.0 - ((py as f64 + 0.5) / (h as f64)) * 2.0;
                    for px in 0..w {
                        let ndc_x = ((px as f64 + 0.5) / (w as f64)) * 2.0 - 1.0;
                        local.push(trace_oracle_pixel(
                            &camera,
                            params,
                            ndc_x,
                            ndc_y,
                            aspect,
                            settings.disk_enabled,
                        ));
                    }
                }
                local
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("oracle worker panicked"))
            .collect()
    });
    for mut c in chunks {
        cpu_results.append(&mut c);
    }

    let r_in = r_isco(params.spin);
    let r_out = params.disk_outer;
    let mut mismatches = 0_usize;
    let mut budget_exhausted = 0_usize;
    let mut interior = 0_usize;
    let mut max_r_rel = 0.0_f64;
    let mut max_phi = 0.0_f64;
    let mut max_g_rel = 0.0_f64;
    let mut max_dir = 0.0_f64;

    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let gh = gpu_hits[idx];
            let gs = gh.status();
            let cr = cpu_results[idx];
            if gs == RayStatus::BudgetExhausted {
                budget_exhausted += 1;
            }

            let mut near_boundary = cr.status == RayStatus::BudgetExhausted;
            if cr.status == RayStatus::DiskHit
                && ((cr.r_hit - r_in).abs() < 0.08 * r_in
                    || (r_out - cr.r_hit).abs() < 0.05 * r_out)
            {
                near_boundary = true;
            }
            if !near_boundary {
                let y0 = py.saturating_sub(2);
                let y1 = (py + 2).min(h - 1);
                let x0 = px.saturating_sub(2);
                let x1 = (px + 2).min(w - 1);
                'nb: for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let nr = cpu_results[ny * w + nx];
                        if nr.status != cr.status || nr.order != cr.order {
                            near_boundary = true;
                            break 'nb;
                        }
                    }
                }
            }
            if near_boundary {
                continue;
            }
            interior += 1;
            if gs != cr.status {
                mismatches += 1;
                continue;
            }
            match cr.status {
                RayStatus::DiskHit => {
                    max_r_rel = max_r_rel.max((f64::from(gh.r_hit) - cr.r_hit).abs() / cr.r_hit);
                    max_phi = max_phi.max(angle_wrap(f64::from(gh.phi_hit), cr.phi_ks_cart).abs());
                    max_g_rel = max_g_rel.max(
                        (f64::from(gh.redshift_g) - cr.redshift_g).abs() / cr.redshift_g.max(1e-6),
                    );
                }
                RayStatus::Escaped => {
                    let gd = DVec3::new(
                        f64::from(gpu_skies[idx][0]),
                        f64::from(gpu_skies[idx][1]),
                        f64::from(gpu_skies[idx][2]),
                    )
                    .normalize();
                    max_dir = max_dir.max(gd.dot(cr.escape_dir_ks).clamp(-1.0, 1.0).acos());
                }
                RayStatus::Captured | RayStatus::BudgetExhausted => {}
            }
        }
    }

    let exh_pct = (budget_exhausted as f64) * 100.0 / (total as f64);
    println!(
        "[{label} {width}x{height} a={:+.3}] ({:.2}s) interior={interior}/{total} mismatches={mismatches} exh={budget_exhausted} ({exh_pct:.4}%) max_r_rel={max_r_rel:.2e} max_phi={max_phi:.2e} max_g_rel={max_g_rel:.2e} max_dir={max_dir:.2e}",
        params.spin,
        t0.elapsed().as_secs_f64()
    );

    anyhow::ensure!(
        mismatches == 0,
        "[{label}] {mismatches} interior classification mismatches"
    );
    anyhow::ensure!(
        exh_pct < 0.1,
        "[{label}] budget exhaustion {exh_pct:.4}% >= 0.1%"
    );
    anyhow::ensure!(
        max_r_rel < 1e-3,
        "[{label}] max_r_rel {max_r_rel:.2e} >= 1e-3"
    );
    anyhow::ensure!(max_phi < 2e-3, "[{label}] max_phi {max_phi:.2e} >= 2e-3");
    anyhow::ensure!(
        max_g_rel < 1e-3,
        "[{label}] max_g_rel {max_g_rel:.2e} >= 1e-3"
    );
    anyhow::ensure!(max_dir < 1e-3, "[{label}] max_dir {max_dir:.2e} >= 1e-3");
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let Some(command) = cli.command else {
        // Bare `caustic` opens the unified studio in Space -> Solar System (Orrery system view).
        let mut photon_scene = PhotonScene::preset(PhotonSceneId::SolarSystem);
        photon_scene.sync_solar_to_gpu(1440.0 / 860.0);
        return run_studio(StudioConfig {
            initial_mode: StudioMode::Photons(PhotonSceneId::SolarSystem),
            camera: Camera::default(),
            bh_params: KerrParams::new(0.95, 16.0, 1000.0),
            bh_settings: RenderSettings::default(),
            photon_scene,
            photon_settings: PhotonSettings::default(),
            width: 1440,
            height: 860,
            force_fallback: cli.software,
        });
    };

    match command {
        Command::View {
            scene,
            width,
            height,
        } => {
            let (camera, params, settings) = scene.to_parts()?;
            run_studio(StudioConfig {
                initial_mode: StudioMode::BlackHole3D,
                camera,
                bh_params: params,
                bh_settings: settings,
                photon_scene: PhotonScene::preset(PhotonSceneId::Prism),
                photon_settings: PhotonSettings::default(),
                width,
                height,
                force_fallback: cli.software,
            })
        }
        Command::Render {
            scene,
            width,
            height,
            output,
        } => {
            let (camera, params, settings) = scene.to_parts()?;
            let gpu = GpuContext::new_headless(cli.software)?;
            let mut pipeline =
                BlackHolePipeline::new(&gpu, width, height, wgpu::TextureFormat::Rgba8UnormSrgb);
            let (rgba, telemetry) =
                pipeline.render_headless(&gpu, &camera, &params, &settings, true)?;
            write_rgba8_png(&output, width, height, &rgba)
                .with_context(|| format!("write PNG to {}", output.display()))?;
            println!(
                "Adapter: {} ({:?})",
                gpu.adapter_info.name, gpu.adapter_info.device_type
            );
            print_telemetry("Render", &telemetry);
            println!("Saved {}", output.display());
            Ok(())
        }
        Command::Bench { scene, iters } => {
            let (camera, params, settings) = scene.to_parts()?;
            let gpu = GpuContext::new_headless(cli.software)?;
            println!(
                "Adapter: {} ({:?}), spin={:+.3}, r_obs={:.1}M, inc={:.1} deg, eta=({:.3}, {:.3})",
                gpu.adapter_info.name,
                gpu.adapter_info.device_type,
                params.spin,
                camera.radius,
                camera.inclination.to_degrees(),
                settings.eta_near,
                settings.eta_far
            );
            for (label, w, h) in [
                ("720p-drag", 1280, 720),
                ("1440p-full", 2560, 1440),
                ("4K-settle", 3840, 2160),
            ] {
                let mut pipeline =
                    BlackHolePipeline::new(&gpu, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);
                let tel = pipeline.benchmark(&gpu, &camera, &params, &settings, 2, iters)?;
                print_telemetry(label, &tel);
            }
            Ok(())
        }
        Command::Verify {
            width,
            height,
            eta_near,
            eta_far,
            max_steps,
        } => {
            let gpu = GpuContext::new_headless(cli.software)?;
            println!(
                "Verifying on adapter: {} ({:?}) at {width}x{height}",
                gpu.adapter_info.name, gpu.adapter_info.device_type
            );
            let base_settings = RenderSettings {
                eta_near,
                eta_far,
                max_steps,
                ..RenderSettings::default()
            };
            run_verify_view(
                &gpu,
                "prograde-extremal",
                Camera {
                    radius: 25.0,
                    inclination: 82.0_f64.to_radians(),
                    azimuth: 0.0,
                    vfov: 45.0_f64.to_radians(),
                },
                KerrParams::new(0.998, 20.0, 1000.0),
                base_settings,
                width,
                height,
            )?;
            run_verify_view(
                &gpu,
                "retrograde-extremal",
                Camera {
                    radius: 25.0,
                    inclination: 82.0_f64.to_radians(),
                    azimuth: 0.0,
                    vfov: 45.0_f64.to_radians(),
                },
                KerrParams::new(-0.998, 20.0, 1000.0),
                base_settings,
                width,
                height,
            )?;
            run_verify_view(
                &gpu,
                "near-polar-kerr",
                Camera {
                    radius: 25.0,
                    inclination: 0.01,
                    azimuth: 0.4,
                    vfov: 45.0_f64.to_radians(),
                },
                KerrParams::new(0.9, 20.0, 1000.0),
                base_settings,
                width,
                height,
            )?;
            run_verify_view(
                &gpu,
                "edge-on-shadow",
                Camera {
                    radius: 25.0,
                    inclination: std::f64::consts::FRAC_PI_2,
                    azimuth: 0.0,
                    vfov: 45.0_f64.to_radians(),
                },
                KerrParams::new(0.998, 20.0, 1000.0),
                RenderSettings {
                    disk_enabled: false,
                    ..base_settings
                },
                width,
                height,
            )?;
            println!("ALL VERIFICATION GATES PASSED.");
            Ok(())
        }
        Command::Photons { command } => match command {
            PhotonCommand::View {
                args,
                width,
                height,
            } => {
                let (scene, settings) =
                    args.to_parts(f64::from(width.max(1)) / f64::from(height.max(1)));
                let initial_mode = StudioMode::Photons(scene.id);
                run_studio(StudioConfig {
                    initial_mode,
                    camera: Camera::default(),
                    bh_params: KerrParams::new(0.95, 16.0, 1000.0),
                    bh_settings: RenderSettings::default(),
                    photon_scene: scene,
                    photon_settings: settings,
                    width,
                    height,
                    force_fallback: cli.software,
                })
            }
            PhotonCommand::Render {
                args,
                width,
                height,
                frames,
                output,
            } => {
                let (scene, settings) =
                    args.to_parts(f64::from(width.max(1)) / f64::from(height.max(1)));
                let gpu = GpuContext::new_headless(cli.software)?;
                let mut pipeline = PhotonPipeline::new(
                    &gpu,
                    width,
                    height,
                    settings.num_photons,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                );
                let (rgba, tel) = pipeline.render_headless(&gpu, &scene, &settings, frames)?;
                write_rgba8_png(&output, width, height, &rgba)
                    .with_context(|| format!("write PNG to {}", output.display()))?;
                println!(
                    "Adapter: {} ({:?})",
                    gpu.adapter_info.name, gpu.adapter_info.device_type
                );
                print_photon_telemetry(scene.id.name(), &tel);
                println!("Saved {}", output.display());
                Ok(())
            }
            PhotonCommand::Bench {
                args,
                width,
                height,
                iters,
            } => {
                let (scene, settings) =
                    args.to_parts(f64::from(width.max(1)) / f64::from(height.max(1)));
                let gpu = GpuContext::new_headless(cli.software)?;
                println!(
                    "Adapter: {} ({:?}), scene={}, mode={:?}",
                    gpu.adapter_info.name,
                    gpu.adapter_info.device_type,
                    scene.id.name(),
                    settings.mode
                );
                let mut pipeline = PhotonPipeline::new(
                    &gpu,
                    width,
                    height,
                    settings.num_photons,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                );
                let tel = pipeline.benchmark(&gpu, &scene, &settings, 5, iters)?;
                print_photon_telemetry(scene.id.name(), &tel);
                Ok(())
            }
        },
    }
}
