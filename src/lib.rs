//! Caustic: GPU-accelerated physics simulation engine.
//!
//! v0.1 implements a general-relativistic rotating Kerr black hole tracer in
//! Outgoing Cartesian Kerr-Schild coordinates verified against an independent
//! Boyer-Lindquist Mino-time CPU `f64` oracle.

pub mod engine;
pub mod sim;
