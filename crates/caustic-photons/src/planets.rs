//! Embedded high-definition 2K (`2048x1024`) equirectangular planetary texture atlas
//! and Cassini radial ring profile for the Solar System and sandbox planets.
//!
//! Textures are decoded in parallel once per process (via [`OnceLock`]) with gamma-correct
//! linear mipmaps so zoomed-out Orrery views never shimmer while closeups stay razor-sharp.
//! See `assets/planets/CREDITS.md` for full NASA / Solar System Scope CC BY 4.0 & PD credits.

use std::io::Cursor;
use std::sync::OnceLock;

use caustic_core::gpu::GpuContext;
use zune_jpeg::JpegDecoder;

/// Equirectangular texture width for every planet layer.
pub const PLANET_TEX_WIDTH: u32 = 2048;
/// Equirectangular texture height for every planet layer.
pub const PLANET_TEX_HEIGHT: u32 = 1024;
/// Number of layers in the planetary `texture_2d_array<f32>` (matches `BODY_SUN..=BODY_NEPTUNE`).
pub const PLANET_TEX_LAYERS: u32 = 12;
/// Full mipmap chain (`2048x1024` down to `2x1`).
pub const PLANET_MIP_LEVELS: u32 = 11;
/// Radial sample count for Saturn's Cassini ring profile texture (`256x1`).
pub const RING_TEX_WIDTH: u32 = 256;

/// Embedded 2K JPEG buffers ordered identically to `solar::BODY_SUN..=solar::BODY_NEPTUNE`.
const PLANET_JPEGS: [&[u8]; PLANET_TEX_LAYERS as usize] = [
    include_bytes!("../assets/planets/sun.jpg"),
    include_bytes!("../assets/planets/mercury.jpg"),
    include_bytes!("../assets/planets/venus.jpg"),
    include_bytes!("../assets/planets/earth.jpg"),
    include_bytes!("../assets/planets/moon.jpg"),
    include_bytes!("../assets/planets/mars.jpg"),
    include_bytes!("../assets/planets/jupiter.jpg"),
    include_bytes!("../assets/planets/io.jpg"),
    include_bytes!("../assets/planets/europa.jpg"),
    include_bytes!("../assets/planets/saturn.jpg"),
    include_bytes!("../assets/planets/uranus.jpg"),
    include_bytes!("../assets/planets/neptune.jpg"),
];

const SATURN_RING_PNG: &[u8] = include_bytes!("../assets/planets/saturn_ring.png");

struct MipLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

struct PlanetAtlasCpu {
    /// `PLANET_TEX_LAYERS` layers, each containing `PLANET_MIP_LEVELS` RGBA8 sRGB mipmaps.
    layers: Vec<Vec<MipLevel>>,
    /// `RING_TEX_WIDTH x 1` RGBA8 sRGB radial profile for Saturn's rings.
    saturn_ring_rgba: Vec<u8>,
}

static CACHED_ATLAS: OnceLock<PlanetAtlasCpu> = OnceLock::new();

fn srgb_to_linear_lut() -> [f32; 256] {
    let mut lut = [0.0_f32; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let c = (i as f32) / 255.0;
        *slot = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }
    lut
}

#[inline]
fn linear_to_srgb_u8(x: f32) -> u8 {
    let c = x.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

fn decode_layer_with_mips(jpg_bytes: &[u8], srgb_lut: &[f32; 256]) -> Vec<MipLevel> {
    let mut decoder = JpegDecoder::new(Cursor::new(jpg_bytes));
    let rgb = decoder
        .decode()
        .expect("embedded planet JPEG must decode cleanly");
    let info = decoder.info().expect("decoded planet JPEG must have info");
    let w = u32::from(info.width);
    let h = u32::from(info.height);
    assert_eq!(
        (w, h),
        (PLANET_TEX_WIDTH, PLANET_TEX_HEIGHT),
        "embedded planet map dimensions must match {PLANET_TEX_WIDTH}x{PLANET_TEX_HEIGHT}"
    );

    let px_count = (w as usize) * (h as usize);
    let mut base_rgba = Vec::with_capacity(px_count * 4);
    let mut cur_lin = Vec::with_capacity(px_count * 3);
    for &[r, g, b] in rgb.as_chunks::<3>().0 {
        base_rgba.extend_from_slice(&[r, g, b, 255]);
        cur_lin.push(srgb_lut[usize::from(r)]);
        cur_lin.push(srgb_lut[usize::from(g)]);
        cur_lin.push(srgb_lut[usize::from(b)]);
    }

    let mut mips = Vec::with_capacity(PLANET_MIP_LEVELS as usize);
    mips.push(MipLevel {
        width: w,
        height: h,
        rgba: base_rgba,
    });

    let mut cw = w as usize;
    let mut ch = h as usize;
    for _ in 1..PLANET_MIP_LEVELS {
        let nw = (cw / 2).max(1);
        let nh = (ch / 2).max(1);
        let mut next_lin = Vec::with_capacity(nw * nh * 3);
        let mut next_rgba = Vec::with_capacity(nw * nh * 4);

        for y in 0..nh {
            let y0 = (y * 2).min(ch - 1);
            let y1 = (y * 2 + 1).min(ch - 1);
            for x in 0..nw {
                let x0 = (x * 2).min(cw - 1);
                let x1 = (x * 2 + 1).min(cw - 1);
                let i00 = (y0 * cw + x0) * 3;
                let i10 = (y0 * cw + x1) * 3;
                let i01 = (y1 * cw + x0) * 3;
                let i11 = (y1 * cw + x1) * 3;
                let r = 0.25 * (cur_lin[i00] + cur_lin[i10] + cur_lin[i01] + cur_lin[i11]);
                let g = 0.25
                    * (cur_lin[i00 + 1] + cur_lin[i10 + 1] + cur_lin[i01 + 1] + cur_lin[i11 + 1]);
                let b = 0.25
                    * (cur_lin[i00 + 2] + cur_lin[i10 + 2] + cur_lin[i01 + 2] + cur_lin[i11 + 2]);
                next_lin.push(r);
                next_lin.push(g);
                next_lin.push(b);
                next_rgba.extend_from_slice(&[
                    linear_to_srgb_u8(r),
                    linear_to_srgb_u8(g),
                    linear_to_srgb_u8(b),
                    255,
                ]);
            }
        }

        mips.push(MipLevel {
            width: nw as u32,
            height: nh as u32,
            rgba: next_rgba,
        });
        cur_lin = next_lin;
        cw = nw;
        ch = nh;
    }

    mips
}

fn decode_saturn_ring() -> Vec<u8> {
    let decoder = png::Decoder::new(Cursor::new(SATURN_RING_PNG));
    let mut reader = decoder
        .read_info()
        .expect("embedded Saturn ring PNG header");
    let mut buf = vec![0_u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .expect("embedded Saturn ring PNG frame");
    assert_eq!(info.width, RING_TEX_WIDTH);
    assert_eq!(info.height, 1);
    buf.truncate(info.buffer_size());
    buf
}

fn cached_atlas() -> &'static PlanetAtlasCpu {
    CACHED_ATLAS.get_or_init(|| {
        let srgb_lut = srgb_to_linear_lut();
        let layers = std::thread::scope(|scope| {
            let handles: Vec<_> = PLANET_JPEGS
                .iter()
                .map(|bytes| scope.spawn(|| decode_layer_with_mips(bytes, &srgb_lut)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("planet texture decode worker panicked"))
                .collect()
        });
        let saturn_ring_rgba = decode_saturn_ring();
        PlanetAtlasCpu {
            layers,
            saturn_ring_rgba,
        }
    })
}

/// GPU-resident 2K planetary texture array, Cassini ring profile, and trilinear sampler.
pub struct PlanetTextures {
    _planet_tex: wgpu::Texture,
    pub planet_view: wgpu::TextureView,
    _ring_tex: wgpu::Texture,
    pub ring_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl PlanetTextures {
    /// Upload all 12 mipmapped 2K planet maps and Saturn's ring profile to the GPU.
    pub fn upload(gpu: &GpuContext) -> Self {
        let atlas = cached_atlas();

        let planet_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet-equirect-array-2k"),
            size: wgpu::Extent3d {
                width: PLANET_TEX_WIDTH,
                height: PLANET_TEX_HEIGHT,
                depth_or_array_layers: PLANET_TEX_LAYERS,
            },
            mip_level_count: PLANET_MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer_idx, mips) in atlas.layers.iter().enumerate() {
            for (mip_idx, mip) in mips.iter().enumerate() {
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &planet_tex,
                        mip_level: mip_idx as u32,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_idx as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &mip.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(mip.width * 4),
                        rows_per_image: Some(mip.height),
                    },
                    wgpu::Extent3d {
                        width: mip.width,
                        height: mip.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        let planet_view = planet_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("planet-equirect-array-view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let ring_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("saturn-ring-profile"),
            size: wgpu::Extent3d {
                width: RING_TEX_WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ring_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.saturn_ring_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(RING_TEX_WIDTH * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: RING_TEX_WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let ring_view = ring_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("planet-trilinear-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            _planet_tex: planet_tex,
            planet_view,
            _ring_tex: ring_tex,
            ring_view,
            sampler,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_planet_textures_and_mips_decode_cleanly() {
        let atlas = cached_atlas();
        assert_eq!(atlas.layers.len(), PLANET_TEX_LAYERS as usize);
        for (i, mips) in atlas.layers.iter().enumerate() {
            assert_eq!(mips.len(), PLANET_MIP_LEVELS as usize, "layer {i}");
            assert_eq!(mips[0].width, PLANET_TEX_WIDTH);
            assert_eq!(mips[0].height, PLANET_TEX_HEIGHT);
            assert_eq!(
                mips[0].rgba.len(),
                (PLANET_TEX_WIDTH * PLANET_TEX_HEIGHT * 4) as usize
            );
            assert_eq!(mips.last().unwrap().width, 2);
            assert_eq!(mips.last().unwrap().height, 1);
        }
        assert_eq!(atlas.saturn_ring_rgba.len(), (RING_TEX_WIDTH * 4) as usize);
    }
}
