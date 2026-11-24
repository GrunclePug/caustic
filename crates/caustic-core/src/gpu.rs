//! Pure-safe `wgpu` Vulkan context supporting both headless (RTX 4080 + lavapipe)
//! and windowed `winit` surfaces, plus optional GPU timestamp profiling.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use anyhow::{Context as _, Result};

/// Minimal zero-dependency `block_on` for synchronous `wgpu` initialization.
pub fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(future);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => thread::park(),
        }
    }
}

/// Shared GPU device, queue, and adapter capability metadata.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub has_timestamps: bool,
    pub timestamp_period_ns: f32,
}

impl GpuContext {
    /// Create a headless Vulkan `wgpu` context.
    ///
    /// When `force_fallback` is `true`, selects a software Vulkan adapter (`lavapipe`).
    pub fn new_headless(force_fallback: bool) -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        Self::from_instance(instance, None, force_fallback)
    }

    /// Create a Vulkan `wgpu` context compatible with an existing window surface.
    pub fn new_for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        force_fallback: bool,
    ) -> Result<Self> {
        Self::from_instance(instance, Some(surface), force_fallback)
    }

    fn from_instance(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        force_fallback: bool,
    ) -> Result<Self> {
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: if force_fallback {
                wgpu::PowerPreference::LowPower
            } else {
                wgpu::PowerPreference::HighPerformance
            },
            force_fallback_adapter: force_fallback,
            compatible_surface,
            apply_limit_buckets: false,
        }))
        .with_context(|| {
            if force_fallback {
                "request software Vulkan adapter (lavapipe)"
            } else {
                "request Vulkan GPU adapter"
            }
        })?;

        let adapter_info = adapter.get_info();
        let supported = adapter.features();
        let has_timestamps = supported.contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if has_timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        // Request adapter limits up to 512 MiB storage buffers so 4K ray buffers never fail.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.min(512 * 1024 * 1024),
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(512 * 1024 * 1024),
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d.min(8192),
            ..Default::default()
        };

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("caustic-device"),
            required_features,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("create wgpu device")?;

        let timestamp_period_ns = queue.get_timestamp_period();
        log::info!(
            "GPU adapter: {} ({:?}, {:?}), timestamps={}",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
            has_timestamps
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
            has_timestamps,
            timestamp_period_ns,
        })
    }

    /// Block until all submitted GPU work on this device has finished.
    pub fn wait_idle(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .context("poll GPU device")?;
        Ok(())
    }
}

/// Optional GPU timestamp query helper recording up to 3 pass pairs (6 timestamps).
pub struct GpuTimers {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    period_ns: f32,
}

impl GpuTimers {
    /// Number of timestamp slots (3 passes x start/end).
    pub const SLOTS: u32 = 6;

    /// Allocate timestamp resources if supported by `gpu`.
    pub fn new(gpu: &GpuContext) -> Option<Self> {
        if !gpu.has_timestamps {
            return None;
        }
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::SLOTS,
        });
        let size = u64::from(Self::SLOTS) * 8;
        let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            readback_buf,
            period_ns: gpu.timestamp_period_ns,
        })
    }

    /// Compute-pass timestamp writes for pass `pass_idx in 0..3`.
    pub fn compute_writes(&self, pass_idx: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_idx * 2),
            end_of_pass_write_index: Some(pass_idx * 2 + 1),
        }
    }

    /// Render-pass timestamp writes for pass `pass_idx in 0..3`.
    pub fn render_writes(&self, pass_idx: u32) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_idx * 2),
            end_of_pass_write_index: Some(pass_idx * 2 + 1),
        }
    }

    /// Resolve all recorded timestamps into the staging buffer.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let c = count.min(Self::SLOTS);
        encoder.resolve_query_set(&self.query_set, 0..c, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.readback_buf,
            0,
            u64::from(c) * 8,
        );
    }

    /// Synchronously read back pass durations in milliseconds after queue submission.
    pub fn read_pass_ms(&self, gpu: &GpuContext, num_passes: usize) -> Result<Vec<f64>> {
        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        gpu.wait_idle()?;
        rx.recv()
            .context("receive timestamp map result")?
            .context("map timestamp readback buffer")?;

        let view = slice
            .get_mapped_range()
            .context("get timestamp mapped range")?;
        let raw: &[u64] = bytemuck::cast_slice(&view);
        let mut out = Vec::with_capacity(num_passes);
        for i in 0..num_passes {
            let dt = raw[i * 2 + 1].saturating_sub(raw[i * 2]);
            out.push((dt as f64) * f64::from(self.period_ns) * 1e-6);
        }
        drop(view);
        self.readback_buf.unmap();
        Ok(out)
    }
}
