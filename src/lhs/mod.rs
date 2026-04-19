//! BFKW09-style linearly homomorphic signatures over `Z/QZ`.
//!
//! This module implements a Boneh-Freeman-Katz-Waters 2009 / Gentry-
//! Peikert lattice-based signature scheme whose signatures compose
//! under `Zq`-linear combinations.  A recoding gossip relay can
//! therefore emit fresh authenticated pieces *without the secret key*,
//! which is the property RLNC needs from any authenticator that
//! tolerates on-the-fly linear mixing.
//!
//! # Construction
//!
//! Micciancio-Peikert 2012 gadget-based trapdoor.  The public matrix
//! is `A = [A_0 | G - A_0·R] ∈ Zq^{n × m}` where `G = I_n ⊗ gadget` is
//! the gadget matrix and `R ∈ {-1,0,1}^{m0 × n·k_gadget}` is a secret
//! ternary matrix.  A signature on `h ∈ Zq^n` is
//!
//! ```text
//! σ = [R·w ; w]    where w is a short Klein/Gaussian preimage of h.
//! ```
//!
//! `w` is drawn from the discrete Gaussian `D_{w_0 + Λ^⊥_q(G), σ_g}`,
//! where `w_0 = bits(h)` is the deterministic gadget preimage and the
//! Gaussian offset lives in the per-block gadget kernel.  Correctness
//! follows from `A · σ = A_0·R·w + (G - A_0·R)·w = G·w ≡ h (mod Q)`.
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
//! This is a correctness-oriented v0.2 that upgrades the deterministic
//! gadget preimage to Klein/Gaussian sampling over the kernel lattice
//! `Λ^⊥_q(G)`, achieving BFKW09 statistical closeness to the ideal
//! discrete Gaussian.  The Klein sampler uses `f64` Gram-Schmidt and is
//! therefore not constant-time; this module is suitable for unit tests
//! and gossip-layer integration, not production side-channel-resistant
//! deployments.

mod authenticator;
mod gadget;
mod keys;
mod params;
mod sign;
mod verify;

pub use authenticator::{Commitment, LatticeHomomorphicAuthenticator};
pub use keys::{PublicKey, SecretKey, keygen};
pub use params::LhsParams;
pub use sign::{Signature, combine, sign};
pub use verify::verify;
