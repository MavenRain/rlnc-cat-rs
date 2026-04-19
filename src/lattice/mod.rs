//! Integer-lattice primitives for lattice-based cryptography.
//!
//! This module provides the algebraic building blocks for schemes
//! over `Z/qZ`:
//!
//! - [`Zq`]: modular integers with `Add` / `Sub` / `Mul` / `Neg`.
//! - [`ZqVec`]: vectors with linear combinations and squared L2 norm
//!   (in the balanced signed-representative interpretation).
//! - [`ZqMatrix`]: row-major matrices with matrix-vector multiplication.
//! - [`ZVec`] / [`ZMatrix`]: integer counterparts without modular
//!   reduction, with `reduce_mod` projections into `Z/qZ`.
//! - [`discrete_gaussian`]: rejection sampler for `D_{Z, σ, c}`.
//! - [`PreimageContext`]: Babai nearest-plane and Klein Gaussian
//!   preimage sampling over a lattice with a known basis.
//!
//! These types are the substrate for the BFKW09-style linearly
//! homomorphic signature scheme in [`crate::lhs`], which lets a
//! recoding gossip relay produce authenticated pieces without the
//! source's key.

mod gaussian;
mod matrix;
mod preimage;
mod signed;
mod vector;
mod zq;

pub use gaussian::discrete_gaussian;
pub use matrix::ZqMatrix;
pub use preimage::PreimageContext;
pub use signed::{ZMatrix, ZVec};
pub use vector::ZqVec;
pub use zq::Zq;
