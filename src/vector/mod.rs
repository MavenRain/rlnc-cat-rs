//! Linear algebra over GF(2^8).
//!
//! Provides immutable vectors ([`GfVec`]), matrices ([`GfMatrix`]),
//! and Gaussian elimination ([`gauss`]).

pub mod gauss;
mod matrix;
mod vec;

pub use matrix::GfMatrix;
pub use vec::GfVec;
