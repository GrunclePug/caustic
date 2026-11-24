//! Rotating Kerr black hole simulation module: exact GR geometry, CPU `f64` oracle,
//! Page-Thorne accretion disk LUTs, and 3-pass GPU ray-buffer pipeline.

pub mod disk;
pub mod gpu_tracer;
pub mod kerr;
pub mod oracle;
