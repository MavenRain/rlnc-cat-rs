//! BF11-style linearly homomorphic signatures over `Z/QZ`.
//!
//! This module implements a Boneh-Freeman 2011 / BFKW09-flavoured
//! lattice-based signature scheme whose signatures compose under
//! `Zq`-linear combinations.  A recoding gossip relay can therefore
//! emit fresh authenticated pieces *without the secret key*, which is
//! the property RLNC needs from any authenticator that tolerates
//! on-the-fly linear mixing.
//!
//! # Construction
//!
//! Micciancio-Peikert 2012 gadget-based trapdoor.  The public matrix
//! is `A = [A_0 | G - A_0·R] ∈ Zq^{n × m}` where `G = I_n ⊗ gadget` is
//! the gadget matrix and `R ∈ {-1,0,1}^{m0 × n·k_gadget}` is a secret
//! ternary matrix.  A signature on `h ∈ Zq^n` is
//!
//! ```text
//! σ = [R·w ; w]    where w = G⁻¹(h) ∈ {0,1}^{n·k_gadget}.
//! ```
//!
//! Correctness follows from `A · σ = A_0·R·w + (G - A_0·R)·w = G·w = h`.
//! Linear homomorphism: `c·σ_1 + σ_2` signs `c·h_1 + h_2` (combined
//! over `Z` with balanced-signed lifts of the `Zq` coefficients).
//!
//! # Public surface
//!
//! - [`LhsParams`]: dimensional constants and norm bound.
//! - [`keygen`]: sample a fresh keypair.
//! - [`PublicKey`] / [`SecretKey`]: opaque key types.
//! - [`sign`] / [`combine`]: produce and compose signatures.
//! - [`verify`]: check `A·σ ≡ target (mod Q)` and `||σ||² ≤ β²`.
//!
//! # Status
//!
//! This is a correctness-oriented v0.1.  It uses `f64` nowhere, is not
//! constant-time, and sets conservative-but-small parameters suitable
//! for unit tests and gossip-layer integration.  The discrete Gaussian
//! sampling over short bases (required for BFKW09 proper) lives in
//! [`crate::lattice`]; this module uses the deterministic gadget
//! preimage which is sound but not statistically close to the claimed
//! distribution.

mod gadget;
mod keys;
mod params;
mod sign;
mod verify;

pub use keys::{PublicKey, SecretKey, keygen};
pub use params::LhsParams;
pub use sign::{Signature, combine, sign};
pub use verify::verify;
