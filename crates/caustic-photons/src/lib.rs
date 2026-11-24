//! Real-time forward spectral photon packet simulation module.
//!
//! Emits millions of wavelength-tagged (`380..780 nm`) photons forward in time on the GPU
//! through:
//! - Flat-space dispersive optics (editable optical bench with prisms, lenses, mirrors,
//!   blockers, real glass Cauchy dispersion, exact unpolarized Fresnel and TIR),
//! - Continuous graded-index (GRIN) eikonal optics (Luneburg lens, chromatic optical
//!   black hole, and thermal mirage), and
//! - Curved Ingoing Cartesian Kerr-Schild equatorial spacetime (`a in [-0.998, +0.998]`)
//!   with real gravitational frequency shift, frame dragging, and relativistic caustics.

pub mod gpu_sim;
pub mod grin;
pub mod ingoing_kerr;
pub mod optics;
pub mod scene;
pub mod solar;
