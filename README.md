# caustic

GPU-accelerated photon playground and general-relativistic physics studio in pure safe Rust, unifying a verified 3D rotating Kerr black hole tracer, an interactive multi-emitter optical sandbox, an animated Solar System orrery with Rayleigh atmospheres, and continuous graded-index (GRIN) optics.

![Solar System photon simulation with 5778K sunlight, shadows, eclipses, and Rayleigh atmospheres](docs/solar-system.png)

![Multi-emitter photon sandbox with green laser, white beam, beam splitter, diamond prism, and planetary atmosphere](docs/sandbox.png)

![Dispersive prism and concave mirror spectral caustic](docs/photons-prism.png)

![Rotating Kerr black hole rendered by caustic](docs/kerr-blackhole.png)

## Overview

`caustic` is a modular GPU physics engine built on `wgpu` (Vulkan) and WGSL compute shaders with zero `unsafe` code. Launching `caustic` opens a unified dark-glass interactive studio where every named mode sits one click away in a minimal floating pill bar (`H` or `Tab` hides all UI for a full-bleed canvas):

* **caustic-core**: Shared Vulkan/WGSL GPU context, timestamp profiling, CIE 1931 2-degree spectral/blackbody LUTs, and Troy Sobotka AgX Punchy presentation pass.
* **caustic-blackhole**: Backward-traced 3D rotating Kerr black hole renderer verified against an independent Boyer-Lindquist CPU `f64` oracle.
* **caustic-photons**: Real-time forward-in-time spectral (`380..780 nm`) photon playground supporting multiple physical emitters, dispersive optics, diffuse surfaces, Rayleigh planetary atmospheres, continuous GRIN eikonal fields, and curved equatorial Ingoing Kerr-Schild spacetime.
* **caustic-app**: Unified minimal `egui` studio and headless CLI for rendering, benchmarking, and physics verification.

## Features

* **Multi-emitter physical light sources**: Combine up to 16 simultaneous emitters (`Beam`, `Point`, `Spotlight`, `Star Disk`) with equal-energy white, true Planck blackbody temperatures (`2000-15000 K`, including `5778 K` sunlight), or monochromatic lasers (`400-700 nm`).
* **Solar System photon orrery**: Simulates a physical `5778 K` solar disk casting true soft umbra/penumbra shadows across orbiting Mercury, Venus, Earth and Moon, Mars, Jupiter, and ringed Saturn, with Beer-Lambert Rayleigh (`1 / lambda^4`) scattering creating blue daytime skies and blood-red sunset / lunar-eclipse shadows.
* **Interactive photon sandbox & real glass catalog**: Click to move, `Shift+drag` to rotate, add (`Prism`, `Lens`, `Mirror`, `Arc`, `Beam Splitter`, `Matte Wall`, `Planet`, `Stop`), or delete elements live using published refractive dispersion (`Water`, `BK7 Glass`, `SF11 Flint`, `Diamond`, plus studio `High-Disp`), exact unpolarized Fresnel reflection/refraction, TIR, and Lambertian albedo scattering.
* **Continuous GRIN eikonal optics**: Integrates Fermat's eikonal Hamiltonian `H = 1/2 (|p|^2 - n(x, lambda)^2) = 0` via RK4 across an aberration-free **Luneburg Lens**, a chromatic inverse-square **Optical Black Hole**, and a **Thermal Mirage** caustic.
* **Exact rotating Kerr geodesics (3D & 2D)**: Traces 3D Outgoing Kerr-Schild geodesics with a Page-Thorne (1974) accretion disk and 2D forward Ingoing Kerr-Schild photon wavefronts with Shapiro delay and frame-dragging asymmetry across `a in [-0.998, +0.998]`.
* **High-FPS stochastic strided spectral splatting**: Accumulates strictly non-negative fixed-point CIE 1931 XYZ with unbiased strided arc-length splatting, smooth temporal EMA for animated orbits, and automatic idle convergence.
* **Independent verification**: Tests GPU black hole outputs against a CPU `f64` Boyer-Lindquist DOPRI5 oracle, alongside analytical unit tests for Fresnel/prism refraction, Planck blackbody peaks, Rayleigh `1 / lambda^4` scaling, GRIN lensing/deflection, and headless `lavapipe` determinism.
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

### Unified Interactive Studio

Launch the unified studio (all 11 named modes accessible from the top floating pill bar):

```sh
caustic
caustic view --spin 0.95 --radius 22 --inclination 82
caustic photons view --scene solar-system
caustic photons view --scene sandbox
```

Built-in named modes:

* **3D Relativity**: `Kerr Black Hole` (`0`)
* **Optical Playground & Solar Orrery**: `Dispersive Prism` (`1`), `Nephroid Cavity` (`2`), `Chromatic Lens` (`3`), `Solar System` (`4`), `Photon Sandbox` (`5`)
* **GRIN Optics**: `Luneburg Lens` (`6`), `Optical Black Hole` (`7`), `Thermal Mirage` (`8`)
* **2D Relativity**: `Kerr Lensing Beam` (`9`), `Kerr Pulsar`

### Headless Rendering, Benchmarks & Verification

```sh
caustic render --spin 0.998 --radius 20 --inclination 82 --output kerr.png
caustic bench --spin 0.998
caustic verify --width 512 --height 512

caustic photons render --scene solar-system --frames 80 --output solar.png
caustic photons render --scene solar-system --mode pulse --frames 24 --output solar-pulse.png
caustic photons render --scene sandbox --frames 60 --output sandbox.png
caustic photons bench --scene solar-system --width 1920 --height 1080
```

### Interactive controls

| Key / Input | Action |
| --- | --- |
| `0` / `1`-`9` | Switch named mode (`0` = Kerr Black Hole 3D, `1`-`9` = forward photon presets) |
| `H` / `Tab` | Toggle minimal floating HUD visibility |
| `P` | Pause / resume planetary orbits in Solar System mode |
| Left drag | Orbit camera (3D Black Hole), drag selected emitter, or move optical element / planet |
| `Shift` + Left drag | Rotate selected optical element |
| Right drag | Aim active photon emitter beam angle |
| Scroll wheel | Zoom camera distance or 2D field of view |
| `M` | Toggle Steady accumulation vs Wavefront pulse mode |
| `Space` | Fire synchronized photon wavefront pulse (Wavefront mode) |
| `[` / `]` | Decrease / increase Kerr spin `a` or dispersion scale |
| `Q` / `E` | Rotate selected object or emitter angle |
| `Delete` / `Backspace` | Delete selected object or extra emitter |
| `-` / `=` | Decrease / increase exposure |
| `S` | Save full-resolution PNG screenshot |
| `R` | Reset current mode |
| `Esc` | Quit |

### Configuration (Forward Photons, Solar System & GRIN)

| Flag | Default | Description |
| --- | --- | --- |
| `--scene` | `prism` | Preset: `prism`, `cardioid`, `lens`, `solar-system`, `sandbox`, `luneburg`, `optical-bh`, `mirage`, `kerr-beam`, `kerr-pulsar` |
| `--mode` | `accumulate` | Transport mode: `accumulate` (Steady) or `pulse` (Wavefront) |
| `--photons` | `1000000` | Photon pool size (persistent count in Wavefront; scaled per-frame batch in Steady) |
| `--substeps` | `4` | Integration substeps per frame in Wavefront mode |
| `--spin` | `0.95` | Kerr black hole spin `a` in `[-0.998, +0.998]` |
| `--dispersion` | preset | Dispersion multiplier (`0.0` = achromatic) |
| `--strength` | `1.0` | GRIN refractive field strength multiplier |
| `--exposure` | `1.0` | Radiance exposure multiplier |
| `--seed` | `42` | Deterministic PRNG seed |
| `--software` | `false` | Force software Vulkan adapter (`lavapipe`) |

## Development

```sh
make run
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
