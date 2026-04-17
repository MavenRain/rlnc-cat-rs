//! Finite field arithmetic over GF(2^8).
//!
//! The field uses irreducible polynomial x^8 + x^4 + x^3 + x + 1
//! with primitive element 3.

mod element;
mod mac;
mod tables;

pub use element::Gf256;
pub use mac::mac;
