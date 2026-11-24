//! Real-time forward spectral photon packet simulation module.
//!
//! Emits millions of wavelength-tagged (`380..780 nm`) photons forward in time on the GPU
//! through either:
//! - Flat-space dispersive optics (mirrors, Cauchy glass prisms, thick biconvex lenses,
//!   exact unpolarized Fresnel reflection/refraction and total internal reflection), or
//! - Curved Ingoing Cartesian Kerr-Schild equatorial spacetime (`a in [-0.998, +0.998]`)
//!   with real gravitational frequency shift, frame dragging, and relativistic caustics.

pub mod gpu_sim;
pub mod ingoing_kerr;
pub mod optics;
pub mod scene;
