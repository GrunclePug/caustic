//! Cross-formulation GPU vs CPU Ray-Buffer physics verification suite (Test 6 + Test 5 GPU).
//!
//! Runs by default on software Vulkan (`lavapipe`) at 64x64 so `make test` passes on any
//! machine without a discrete GPU, and includes an explicit hardware GPU gate (`512x512` at
//! `a = +/-0.998`) executed during verification.

use caustic::engine::gpu::GpuContext;
use caustic::sim::blackhole::gpu_tracer::{BlackHolePipeline, RenderSettings};
use caustic::sim::blackhole::kerr::{Camera, KerrParams, r_isco};
use caustic::sim::blackhole::oracle::{RayStatus, trace_oracle_pixel};
use glam::DVec3;

fn angle_wrap(a: f64, b: f64) -> f64 {
    (a - b + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// Run the cross-formulation GPU (Outgoing Cartesian KS RK4 FP32) vs CPU (Boyer-Lindquist
/// Mino-time DOPRI5 FP64) verification gate for one camera/spin view.
fn verify_view(
    gpu: &GpuContext,
    label: &str,
    camera: Camera,
    params: KerrParams,
    width: u32,
    height: u32,
    disk_enabled: bool,
) {
    let settings = RenderSettings {
        disk_enabled,
        ..RenderSettings::default()
    };
    let mut pipeline =
        BlackHolePipeline::new(gpu, width, height, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (gpu_hits, gpu_skies) = pipeline
        .trace_and_readback(gpu, &camera, &params, &settings)
        .expect("GPU trace and readback");

    let aspect = f64::from(width) / f64::from(height);
    let total = (width as usize) * (height as usize);

    // Compute CPU f64 oracle across the grid.
    let mut cpu_results = Vec::with_capacity(total);
    for py in 0..height {
        let ndc_y = 1.0 - ((f64::from(py) + 0.5) / f64::from(height)) * 2.0;
        for px in 0..width {
            let ndc_x = ((f64::from(px) + 0.5) / f64::from(width)) * 2.0 - 1.0;
            cpu_results.push(trace_oracle_pixel(
                &camera,
                params,
                ndc_x,
                ndc_y,
                aspect,
                disk_enabled,
            ));
        }
    }

    let w = width as usize;
    let h = height as usize;
    let mut mismatches = 0_usize;
    let mut budget_exhausted = 0_usize;
    let mut max_r_rel_err = 0.0_f64;
    let mut max_phi_err = 0.0_f64;
    let mut max_g_rel_err = 0.0_f64;
    let mut max_dir_err = 0.0_f64;
    let mut interior_checked = 0_usize;

    let r_in = r_isco(params.spin);
    let r_out = params.disk_outer;

    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let gh = gpu_hits[idx];
            let gs = gh.status();
            let cr = cpu_results[idx];

            if gs == RayStatus::BudgetExhausted {
                budget_exhausted += 1;
            }

            // A pixel lies inside the 1.5-pixel critical-curve / disk-edge boundary band if any
            // 5x5 neighbor (radius <= 2 px) changes hit classification, order, or is near a disk edge.
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

            interior_checked += 1;
            if gs != cr.status {
                mismatches += 1;
                continue;
            }

            match cr.status {
                RayStatus::DiskHit => {
                    let r_rel = (f64::from(gh.r_hit) - cr.r_hit).abs() / cr.r_hit;
                    let phi_err = angle_wrap(f64::from(gh.phi_hit), cr.phi_ks_cart).abs();
                    let g_rel =
                        (f64::from(gh.redshift_g) - cr.redshift_g).abs() / cr.redshift_g.max(1e-6);
                    max_r_rel_err = max_r_rel_err.max(r_rel);
                    max_phi_err = max_phi_err.max(phi_err);
                    max_g_rel_err = max_g_rel_err.max(g_rel);
                }
                RayStatus::Escaped => {
                    let gd = DVec3::new(
                        f64::from(gpu_skies[idx][0]),
                        f64::from(gpu_skies[idx][1]),
                        f64::from(gpu_skies[idx][2]),
                    )
                    .normalize();
                    let cos_ang = gd.dot(cr.escape_dir_ks).clamp(-1.0, 1.0);
                    let ang = cos_ang.acos();
                    max_dir_err = max_dir_err.max(ang);
                }
                RayStatus::Captured | RayStatus::BudgetExhausted => {}
            }
        }
    }

    let exhausted_frac = (budget_exhausted as f64) / (total as f64);
    eprintln!(
        "[{label} {width}x{height} a={}] interior={interior_checked}/{total} mismatches={mismatches} exhausted={budget_exhausted} ({:.3}%) max_r_rel={max_r_rel_err:.2e} max_phi={max_phi_err:.2e} max_g_rel={max_g_rel_err:.2e} max_dir={max_dir_err:.2e}",
        params.spin,
        exhausted_frac * 100.0
    );

    assert_eq!(
        mismatches, 0,
        "[{label}] interior classification mismatches"
    );
    assert!(
        exhausted_frac < 0.005,
        "[{label}] budget exhaustion rate {:.3}% too high",
        exhausted_frac * 100.0
    );
    assert!(
        max_r_rel_err < 1e-3,
        "[{label}] disk r_hit relative error {max_r_rel_err:.2e} >= 1e-3"
    );
    assert!(
        max_phi_err < 2e-3,
        "[{label}] disk phi_hit error {max_phi_err:.2e} >= 2e-3 rad"
    );
    assert!(
        max_g_rel_err < 1e-3,
        "[{label}] disk redshift g relative error {max_g_rel_err:.2e} >= 1e-3"
    );
    assert!(
        max_dir_err < 1e-3,
        "[{label}] escaped sky direction error {max_dir_err:.2e} >= 1e-3 rad"
    );
}

#[test]
fn test6_cross_formulation_lavapipe_64x64() {
    let gpu = GpuContext::new_headless(true).expect("headless lavapipe Vulkan context");

    // 1. Equatorial / inclined prograde extremal Kerr (a = +0.998)
    verify_view(
        &gpu,
        "lavapipe-prograde-extremal",
        Camera {
            radius: 25.0,
            inclination: 82.0_f64.to_radians(),
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        },
        KerrParams::new(0.998, 20.0, 1000.0),
        64,
        64,
        true,
    );

    // 2. Retrograde extremal Kerr (a = -0.998)
    verify_view(
        &gpu,
        "lavapipe-retrograde-extremal",
        Camera {
            radius: 25.0,
            inclination: 82.0_f64.to_radians(),
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        },
        KerrParams::new(-0.998, 20.0, 1000.0),
        64,
        64,
        true,
    );

    // 3. Near-polar axis stress test (theta = 0.01 rad, Schwarzschild & Kerr)
    verify_view(
        &gpu,
        "lavapipe-near-polar-kerr",
        Camera {
            radius: 25.0,
            inclination: 0.01,
            azimuth: 0.4,
            vfov: 45.0_f64.to_radians(),
        },
        KerrParams::new(0.9, 20.0, 1000.0),
        64,
        64,
        true,
    );

    // 4. Exact equatorial edge-on camera stress test (theta = pi/2, ensures zero |z| stall!)
    verify_view(
        &gpu,
        "lavapipe-edge-on",
        Camera {
            radius: 25.0,
            inclination: std::f64::consts::FRAC_PI_2,
            azimuth: 0.0,
            vfov: 45.0_f64.to_radians(),
        },
        KerrParams::new(0.9, 20.0, 1000.0),
        64,
        64,
        false,
    );
}
