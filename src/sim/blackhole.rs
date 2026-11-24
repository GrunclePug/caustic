//! Rotating Kerr black hole simulation: exact GR geometry, CPU `f64` oracle,
//! Page-Thorne accretion disk & CIE blackbody LUTs, and GPU ray-buffer pipeline.

pub mod disk;
pub mod gpu_tracer;
pub mod kerr;
pub mod oracle;
