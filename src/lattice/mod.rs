//! Integer-lattice primitives for lattice-based cryptography.
//!
//! This module provides the algebraic building blocks for schemes
//! over `Z/qZ`:
//!
//! - [`Zq`]: modular integers with `Add` / `Sub` / `Mul` / `Neg`.
//! - [`ZqVec`]: vectors with linear combinations and squared L2 norm
//!   (in the balanced signed-representative interpretation).
//! - [`ZqMatrix`]: row-major matrices with matrix-vector multiplication.
//!
//! These types are the substrate for the upcoming BF11-style linearly
//! homomorphic signature scheme that will let a recoding gossip relay
//! produce authenticated pieces without the source's key.

mod matrix;
mod vector;
mod zq;

pub use matrix::ZqMatrix;
pub use vector::ZqVec;
pub use zq::Zq;
