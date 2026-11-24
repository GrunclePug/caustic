# caustic

GPU-accelerated physics simulation engine in pure safe Rust, featuring a verified rotating Kerr black hole tracer and a real-time forward spectral photon simulator.

![Rotating Kerr black hole rendered by caustic](docs/kerr-blackhole.png)

![Dispersive prism and concave mirror spectral caustic](docs/photons-prism.png)

![Relativistic Shapiro-delayed photon wavefront wrapping a spinning Kerr black hole](docs/photons-kerr-pulse.png)

## Overview

`caustic` is a modular GPU physics engine built on `wgpu` (Vulkan) and WGSL compute shaders with zero `unsafe` code. It ships two coupled simulation engines sharing a common `wgpu` runtime, CIE 1931 spectral colorimetry core, and AgX filmic tone mapper:

* **caustic-core**: Shared Vulkan/WGSL GPU context, timestamp profiling, CIE 1931 2-degree spectral/blackbody LUTs, and Troy Sobotka AgX Punchy presentation pass.
* **caustic-blackhole**: Backward-traced 3D rotating Kerr black hole renderer verified against an independent Boyer-Lindquist CPU `f64` oracle.
* **caustic-photons**: Real-time forward-in-time spectral (`380..780 nm`) photon packet simulator covering both dispersive flat-space optics and curved equatorial Ingoing Kerr-Schild spacetime.
* **caustic-app**: Unified CLI and interactive `winit` viewers for both simulation engines.

## Features

* **Exact rotating Kerr geodesics**: Integrates future-directed null 4-momentum backward in Outgoing Cartesian Kerr-Schild coordinates with classical RK4, avoiding polar singularities and crossing the past outer horizon smoothly across `a in [-0.998, +0.998]`.
* **Covariant ZAMO camera & ergosphere safety**: Builds the Zero Angular Momentum Observer tetrad directly from Kerr Killing vectors in `f64`, normalizing initial momentum by camera-frame frequency so rays inside the ergosphere (`E <= 0`) stay finite.
* **Relativistic Page-Thorne accretion disk**: Precomputes the exact Page-Thorne (1974) radial temperature profile and a log-spaced CIE 1931 log-XYZ blackbody LUT, evaluating `B_nu(g * T_emit)` for exact Doppler beaming and gravitational redshift with no `g^4` double-counting.
* **Forward dispersive spectral photon transport**: Propagates 1M+ physical photon packets forward in time with Cauchy refractive dispersion `n(lambda)`, exact unpolarized Fresnel reflection/refraction, total internal reflection, and dispersive group velocity `c / n(lambda)`.
* **Curved-spacetime photon flight & Shapiro delay**: Integrates real-time forward photon packets in equatorial Ingoing Cartesian Kerr-Schild coordinates, visualizing gravitational frequency shift, frame-dragging asymmetry, and swallowtail wavefront caustics.
* **Physical CIE 1931 XYZ atomic spectral splatting**: Accumulates strictly non-negative fixed-point CIE XYZ values across arbitrary wavelengths before converting to Rec.709 RGB and AgX HDR, allowing rainbows to recombine physically into pure white light without `f32` atomics.
* **Independent cross-formulation verification**: Verifies GPU FP32 black hole outputs against an independent CPU `f64` inverse-radius Boyer-Lindquist Mino-time DOPRI5 oracle, plus optical Fresnel/prism and forward Kerr deflection tests on `lavapipe`.
* **Pure safe Rust**: Compiles with `unsafe_code = "forbid"`, embedded WGSL shaders, and no C/C++ build toolchain.

## Installation

### Requirements

* Rust `1.100` or newer (edition 2024)
* GNU Make
* A Vulkan 1.2+ driver (NVIDIA, AMD, Intel, or Mesa `lavapipe` for software headless verification)

### Build

```sh
git clone https://github.com/grunclepug/caustic
cd caustic
make build
sudo make install
```

## Usage

### Rotating Kerr Black Hole

Render a headless 1080p Kerr black hole PNG and print GPU pass timings:

```sh
caustic render --spin 0.998 --radius 20 --inclination 82 --output kerr.png
```

Launch the interactive 3D orbit viewer:

```sh
caustic view --spin 0.95 --radius 22 --inclination 82
```

Run the headless GPU benchmark gate across 720p, 1440p, and 4K, or run cross-formulation verification against the CPU `f64` oracle:

```sh
caustic bench --spin 0.998
caustic verify --width 512 --height 512
caustic verify --software --width 64 --height 64
```

### Forward Spectral Photon Simulator

Launch the interactive forward photon viewer (`prism`, `cardioid`, `lens`, `kerr-beam`, `kerr-pulsar`):

```sh
caustic photons view --scene prism
caustic photons view --scene kerr-beam --mode pulse --spin 0.98
```

Render headless caustics or mid-flight wavefront snapshots:

```sh
caustic photons render --scene prism --frames 150 --output prism.png
caustic photons render --scene lens --frames 150 --output lens.png
caustic photons render --scene kerr-beam --mode pulse --frames 46 --output kerr-pulse.png
```

Benchmark forward photon throughput (billion photon-steps/sec):

```sh
caustic photons bench --scene prism --photons 1000000
caustic photons bench --scene kerr-beam --photons 1000000
```

### Interactive controls (Black Hole Viewer)

| Key / Input | Action |
| --- | --- |
| Left drag | Orbit camera azimuth and inclination (half-res during drag, full-res on settle) |
| Scroll wheel | Zoom radial distance |
| `[` / `]` | Decrease / increase black hole spin `a` |
| `d` | Toggle equatorial accretion disk |
| `-` / `=` | Decrease / increase exposure |
| `g` / `t` | Decrease / increase disk peak temperature scale |
| `1` / `2` / `3` / `4` | Switch view: Beauty (AgX), Step heatmap, Redshift `g`, Hit class |
| `s` | Save full-resolution PNG screenshot |
| `r` | Reset camera and parameters |
| `Esc` | Quit |

### Interactive controls (Forward Photon Viewer)

| Key / Input | Action |
| --- | --- |
| Left drag | Move photon emitter position live |
| Right drag | Aim emitter beam angle |
| Scroll wheel | Zoom view in / out |
| `Space` | Fire a synchronized photon pulse (Pulse mode) or reset exposure (Accumulate mode) |
| `m` | Toggle Accumulate (long-exposure caustic) vs Pulse (flying wavefront) mode |
| `1` / `2` / `3` / `4` / `5` | Switch preset: Prism, Cardioid, Lens, Kerr Beam, Kerr Pulsar |
| `[` / `]` | Decrease / increase Kerr spin `a` (Kerr scenes) or glass dispersion (optical scenes) |
| `,` / `.` | Narrow / widen emitter beam width |
| `-` / `=` | Decrease / increase exposure |
| `c` | Clear accumulation buffer |
| `s` | Save PNG screenshot |
| `r` | Reset current preset |
| `Esc` | Quit |

### Configuration (Black Hole)

| Flag | Default | Description |
| --- | --- | --- |
| `--spin` | `0.95` | Dimensionless Kerr spin `a` in `[-0.998, +0.998]` (`0` = Schwarzschild) |
| `--radius` | `32.0` | Camera Boyer-Lindquist distance in units of `M` |
| `--inclination` | `82.0` | Camera polar angle in degrees (`0` = face-on, `90` = edge-on) |
| `--azimuth` | `0.0` | Camera azimuth in degrees |
| `--fov` | `45.0` | Vertical field of view in degrees |
| `--disk-outer` | `16.0` | Accretion disk outer radius in units of `M` |
| `--no-disk` | `false` | Disable the accretion disk (shadow and lensed sky only) |
| `--temp` | `5500.0` | Peak disk blackbody temperature scale in Kelvin (visible-band parameter) |
| `--exposure` | `1.8` | Radiance exposure multiplier |
| `--eta-near` | `0.008` | Near-field RK4 radial step factor (`r <= 3.5M`) |
| `--eta-far` | `0.075` | Far-field RK4 radial step factor (`r >= 14M`) |
| `--max-steps` | `2048` | Maximum RK4 steps per ray |
| `--debug` | `beauty` | View mode: `beauty`, `steps`, `redshift`, or `hit-class` |
| `--software` | `false` | Force software Vulkan adapter (`lavapipe`) |

### Configuration (Forward Photons)

| Flag | Default | Description |
| --- | --- | --- |
| `--scene` | `prism` | Preset scene: `prism`, `cardioid`, `lens`, `kerr-beam`, or `kerr-pulsar` |
| `--mode` | `accumulate` | Mode: `accumulate` (long-exposure caustics) or `pulse` (live wavefront flight) |
| `--photons` | `1000000` | Number of simultaneously simulated forward photon packets |
| `--substeps` | `4` | Integration substeps per frame |
| `--dt` | `0.08` | Step size per substep |
| `--spin` | `0.95` | Kerr black hole spin `a` in `[-0.998, +0.998]` (for Kerr scenes) |
| `--dispersion` | `1.0` | Glass Cauchy dispersion multiplier (for optical scenes) |
| `--exposure` | `1.0` | Radiance exposure multiplier |
| `--seed` | `42` | Deterministic PRNG seed |
| `--software` | `false` | Force software Vulkan adapter (`lavapipe`) |

## Development

```sh
make run ARGS="view --spin 0.95"
make run ARGS="photons view --scene prism"
make test
make lint
make fmt
make build
make bench
make photons-bench
make render
make clean
```

## Contributing

Fork the repository and open a pull request.

## License

MIT. See [LICENSE](LICENSE).

## Author

Chad Humphries | [Website](https://grunclepug.com/) | [GitHub](https://github.com/GrunclePug) | [More projects](https://github.com/GrunclePug?tab=repositories)
