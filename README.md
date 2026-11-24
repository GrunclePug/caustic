# caustic

GPU-accelerated physics simulation engine in pure safe Rust, starting with physically accurate rotating Kerr black holes.

![Rotating Kerr black hole rendered by caustic](docs/kerr-blackhole.png)

## Overview

`caustic` is a performance-focused GPU simulation runtime built on `wgpu` (Vulkan) and WGSL compute shaders with zero `unsafe` code. Version 0.1 ships a general-relativistic rotating Kerr black hole tracer verified against an independent Boyer-Lindquist CPU `f64` oracle, designed to grow into a modular workspace covering particles, cellular automata, and photon transport.

## Features

* **Exact rotating Kerr geodesics**: Integrates future-directed null 4-momentum backward in Outgoing Cartesian Kerr-Schild coordinates with classical RK4, avoiding polar singularities and crossing the past outer horizon smoothly across `a in [-0.998, +0.998]`.
* **Covariant ZAMO camera & ergosphere safety**: Builds the Zero Angular Momentum Observer tetrad directly from Kerr Killing vectors in `f64`, normalizing initial momentum by camera-frame frequency so rays inside the ergosphere (`E <= 0`) stay finite.
* **Relativistic Page-Thorne accretion disk**: Precomputes the exact Page-Thorne (1974) radial temperature profile and a log-spaced CIE 1931 log-XYZ blackbody LUT, evaluating `B_nu(g * T_emit)` for exact Doppler beaming and gravitational redshift with no `g^4` double-counting.
* **Two-pass cached 4K ray buffers**: Separates geodesic tracing (two 16 B/px SoA buffers under 128 MiB each at 4K) from shading and AgX filmic tone mapping, with Jacobian-filtered lensed starfields and interactive half-resolution drag / full-resolution settle.
* **Independent cross-formulation verification**: Tests GPU FP32 Outgoing Kerr-Schild outputs directly against an independent CPU `f64` inverse-radius Boyer-Lindquist Mino-time Dormand-Prince RK45 oracle, including headless `lavapipe` verification.
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

Render a headless 1080p Kerr black hole PNG and print GPU pass timings:

```sh
caustic render --spin 0.998 --radius 20 --inclination 82 --output kerr.png
```

Launch the interactive orbit viewer:

```sh
caustic view --spin 0.95 --radius 22 --inclination 82
```

Run the headless GPU benchmark gate across 720p, 1440p, and 4K:

```sh
caustic bench --spin 0.998
```

Run cross-formulation GPU vs CPU `f64` oracle verification:

```sh
caustic verify --width 512 --height 512
caustic verify --software --width 64 --height 64
```

### Interactive controls

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

### Configuration

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

## Development

```sh
make run ARGS="view --spin 0.95"
make test
make lint
make fmt
make build
make bench
make render
make clean
```

## Contributing

Fork the repository and open a pull request.

## License

MIT. See [LICENSE](LICENSE).

## Author

Chad Humphries | [Website](https://grunclepug.com/) | [GitHub](https://github.com/GrunclePug) | [More projects](https://github.com/GrunclePug?tab=repositories)
