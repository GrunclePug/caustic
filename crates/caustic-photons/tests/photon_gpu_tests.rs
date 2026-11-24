//! Headless `lavapipe` GPU smoke and determinism tests for `caustic-photons`
//! covering both dispersive flat-space optics (`prism`) and curved Ingoing Kerr-Schild (`kerr-beam`).

use caustic_core::gpu::GpuContext;
use caustic_photons::gpu_sim::{PhotonPipeline, PhotonSettings};
use caustic_photons::scene::{PhotonMode, PhotonScene, PhotonSceneId};

#[test]
fn lavapipe_prism_and_kerr_beam_render_deterministically() {
    let gpu = GpuContext::new_headless(true).expect("headless lavapipe Vulkan context");

    for scene_id in [PhotonSceneId::Prism, PhotonSceneId::KerrBeam] {
        let scene = PhotonScene::preset(scene_id);
        let settings = PhotonSettings {
            num_photons: 16_384,
            substeps: 4,
            mode: PhotonMode::Accumulate,
            seed: 1337,
            ..PhotonSettings::default()
        };

        let mut p1 = PhotonPipeline::new(
            &gpu,
            128,
            72,
            settings.num_photons,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let (img_a, _) = p1
            .render_headless(&gpu, &scene, &settings, 12)
            .expect("first lavapipe photon render");

        let mut p2 = PhotonPipeline::new(
            &gpu,
            128,
            72,
            settings.num_photons,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let (img_b, _) = p2
            .render_headless(&gpu, &scene, &settings, 12)
            .expect("second lavapipe photon render");

        // Verify non-trivial spectral photon deposition happened (bright pixels exist).
        let lit_pixels = img_a
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 40 || px[1] > 40 || px[2] > 40)
            .count();
        assert!(
            lit_pixels > 100,
            "scene {} expected > 100 lit pixels, got {lit_pixels}",
            scene_id.name()
        );

        // Verify deterministic seeded replay on the same adapter (fixed-point atomicAdd is commutative!).
        assert_eq!(
            img_a,
            img_b,
            "scene {} output differed across identical seeded runs",
            scene_id.name()
        );

        // Verify a SINGLE frame (as during continuous live emitter dragging/aiming)
        // traces complete end-to-end trajectories all the way into the right half of the frame!
        let (img_single, _) = p1
            .render_headless(&gpu, &scene, &settings, 1)
            .expect("single-frame live-drag lavapipe photon render");
        let mut right_half_lit = 0_usize;
        for py in 0..72_usize {
            for px in 64..128_usize {
                let i = (py * 128 + px) * 4;
                if img_single[i] > 35 || img_single[i + 1] > 35 || img_single[i + 2] > 35 {
                    right_half_lit += 1;
                }
            }
        }
        assert!(
            right_half_lit > 40,
            "scene {} single-frame live drag expected > 40 lit pixels in right half, got {right_half_lit}",
            scene_id.name()
        );
    }
}
