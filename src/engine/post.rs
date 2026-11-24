//! Pass 3: Exposure, AgX filmic tone mapping, debug overlays, and headless PNG export.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};

use super::gpu::GpuContext;

/// Debug visualization modes for Pass 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum DebugView {
    /// Physical beauty render (Exposure + AgX tone mapping).
    #[default]
    Beauty = 0,
    /// RK4 integration step-count heatmap (turbo colormap).
    StepHeatmap = 1,
    /// Relativistic redshift `g`-factor map (blue = blueshift, red = redshift).
    Redshift = 2,
    /// 4-way ray hit classification (black=Captured, gold/cyan/white=DiskHit n=0/1/2, blue=Escaped, magenta=BudgetExhausted).
    HitClass = 3,
}

/// GPU uniform block for `src/shaders/post_agx.wgsl` (`struct PostUniforms`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PostUniforms {
    /// x = exposure multiplier, y = max_steps_scale (for heatmap), z = reserved, w = reserved.
    pub params: [f32; 4],
    /// x = width, y = height, z = debug_mode, w = reserved.
    pub dims: [u32; 4],
}

/// Write an RGBA8 image buffer to a PNG file atomically via a temporary sibling file.
pub fn write_rgba8_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    let tmp_path = path.with_extension("png.tmp");
    {
        let file =
            File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().context("write PNG header")?;
        writer.write_image_data(rgba).context("write PNG data")?;
    }
    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("rename {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

/// Read back an `Rgba8UnormSrgb` (or `Rgba8Unorm`) texture to tightly packed RGBA8 bytes.
pub fn readback_rgba8_texture(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let bytes_per_pixel = 4_u32;
    let unpadded_bpr = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let buf_size = u64::from(padded_bpr) * u64::from(height);

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("png-readback-staging"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("png-readback-encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.wait_idle()?;
    rx.recv()
        .context("receive texture map result")?
        .context("map texture readback buffer")?;

    let mapped = slice
        .get_mapped_range()
        .context("get texture mapped range")?;
    let mut out = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height as usize {
        let start = row * (padded_bpr as usize);
        let end = start + (unpadded_bpr as usize);
        out.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
}
