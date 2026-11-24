//! Caustic CLI: headless rendering, RTX 4080 / lavapipe benchmarking, cross-formulation
//! GR physics verification, and interactive windowed orbit viewer.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glam::DVec3;

use caustic::engine::gpu::GpuContext;
use caustic::engine::post::{DebugView, write_rgba8_png};
use caustic::engine::viewer::run_viewer;
use caustic::sim::blackhole::gpu_tracer::{BlackHolePipeline, FrameTelemetry, RenderSettings};
use caustic::sim::blackhole::kerr::{
    Camera, KerrParams, MAX_CAMERA_RADIUS, MAX_SPIN, MIN_CAMERA_HORIZON_MARGIN, r_isco, r_plus,
};
use caustic::sim::blackhole::oracle::{RayStatus, trace_oracle_pixel};

#[derive(Debug, Parser)]
#[command(
    name = "caustic",
    version,
    about = "GPU-accelerated general-relativistic Kerr black hole and physics simulator"
)]
struct Cli {
    /// Force software Vulkan adapter (lavapipe) instead of hardware GPU.
    #[arg(long, global = true)]
    software: bool,

    #[command(subcommand)]
    command: Command,
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

#[derive(Debug, Subcommand)]
enum Command {
    /// Launch the interactive windowed orbit viewer.
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
    /// Render a single deterministic frame headlessly to a PNG file and print telemetry.
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
    /// Run the headless GPU benchmark gate across 720p, 1440p, and 4K.
    Bench {
        #[command(flatten)]
        scene: SceneArgs,
        /// Number of timed iterations per resolution.
        #[arg(long, default_value_t = 9)]
        iters: usize,
    },
    /// Run cross-formulation GPU vs CPU f64 oracle accuracy verification across views.
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

    // Parallelize CPU f64 oracle across hardware threads using std::thread::scope!
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

    match cli.command {
        Command::View {
            scene,
            width,
            height,
        } => {
            let (camera, params, settings) = scene.to_parts()?;
            run_viewer(camera, params, settings, width, height, cli.software)
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
    }
}
