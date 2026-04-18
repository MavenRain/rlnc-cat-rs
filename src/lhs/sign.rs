//! Sign and Combine operations for the LHS scheme.
//!
//! The signature on hash target `h_i ∈ Zq^n` is the integer vector
//! `σ_i = [R·w ; w] ∈ Z^m`, where `w = G⁻¹(h_i) ∈ {0,1}^{n·k_gadget}`
//! is the bit-decomposition preimage of `h_i` under the gadget matrix.
//! The gadget trapdoor identity gives
//!
//! ```text
//! A · σ_i = A_0·(R·w) + A_1·w
//!         = A_0·R·w + (G - A_0·R)·w
//!         = G·w = h_i  (mod Q).
//! ```
//!
//! Linear homomorphism follows: a `Zq`-linear combination of
//! signatures (lifted to `Z` via the balanced signed representative)
//! is itself a valid signature on the matching combination of hash
//! targets, and the composition is performed by [`combine`] without
//! the secret key.

use crate::error::Error;
use crate::lattice::{ZVec, Zq};
use crate::lhs::gadget::gadget_preimage;
use crate::lhs::keys::{PublicKey, SecretKey};

/// A BF11-style signature: the integer vector `σ ∈ Z^m` whose
/// length equals `params.m() = m0 + n·k_gadget` for a well-formed key.
///
/// Signatures live over `Z`, not `Z/QZ`: the norm bound that the
/// verifier checks is measured in the integer interpretation, so a
/// fresh signature remains short while a "signature" cooked up from a
/// lattice vector modulo `Q` can be arbitrarily long.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct Signature {
    sigma: ZVec,
}

impl Signature {
    /// Wrap an owned integer vector as a signature.
    pub fn new(sigma: ZVec) -> Self {
        Self { sigma }
    }

    /// Borrow of the underlying integer vector.
    pub fn sigma(&self) -> &ZVec {
        &self.sigma
    }

    /// The length of `σ`.  A well-formed signature under parameters
    /// `params` has `len() == params.m()`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sigma.len()
    }

    /// Whether the underlying vector has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sigma.is_empty()
    }

    /// Squared L2 norm `||σ||²`.  Verification compares this against
    /// `params.sig_norm_bound_sq()`.
    #[must_use]
    pub fn squared_l2_norm(&self) -> u128 {
        self.sigma.squared_l2_norm()
    }
}

/// Sign the public key's `index`-th hash target.
///
/// Produces `σ = [R·w ; w]` where `w = G⁻¹(h_index)` is the
/// bit-decomposition preimage under the gadget matrix.  By the gadget
/// trapdoor identity, `A · σ ≡ h_index (mod Q)`, and the norm is
/// bounded by the ternary/bit structure of `R` and `w`.
///
/// # Errors
///
/// - [`Error::DimensionMismatch`] if `index >= params.k_pieces()` or if
///   the trapdoor matrix shape does not match the gadget width
///   (unreachable when `pk` and `sk` come from the same [`keygen`]
///   call).
///
/// [`keygen`]: crate::lhs::keys::keygen
pub fn sign<const Q: u32>(
    pk: &PublicKey<Q>,
    sk: &SecretKey<Q>,
    index: usize,
) -> Result<Signature, Error> {
    let targets = pk.hash_targets();
    targets.get(index).map_or_else(
        || {
            Err(Error::DimensionMismatch {
                expected: targets.len(),
                actual: index,
            })
        },
        |h| {
            let w_zvec = ZVec::new(gadget_preimage::<Q>(h));
            sk.r().mul_vec(&w_zvec).map(|r_w| {
                let entries: Vec<i64> = r_w
                    .entries()
                    .iter()
                    .copied()
                    .chain(w_zvec.entries().iter().copied())
                    .collect();
                Signature::new(ZVec::new(entries))
            })
        },
    )
}

/// Combine existing signatures under `Zq` coefficients.
///
/// Given pairs `(c_i, σ_i)`, returns the signature `σ = Σ c_i · σ_i`
/// computed over `Z`, with each coefficient lifted to its balanced
/// signed representative in `(-Q/2, Q/2]` before multiplication.  The
/// combined signature signs `Σ c_i · h_i (mod Q)`; lifting to balanced
/// representatives keeps the result short without changing the mod-`Q`
/// identity.
///
/// An empty input yields an empty signature; dimension-mismatched
/// inputs propagate the underlying error.
///
/// # Errors
///
/// - [`Error::DimensionMismatch`] if the signatures have different
///   lengths (from [`ZVec::linear_combine`]).
pub fn combine<const Q: u32>(
    pairs: &[(Zq<Q>, Signature)],
) -> Result<Signature, Error> {
    let scalars: Vec<i64> = pairs.iter().map(|(c, _)| c.signed_repr()).collect();
    let vecs: Vec<ZVec> = pairs.iter().map(|(_, s)| s.sigma().clone()).collect();
    ZVec::linear_combine(&scalars, &vecs).map(Signature::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ZqVec;
    use crate::lhs::keys::keygen;
    use crate::lhs::params::LhsParams;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter_rng() -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(0));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n)
                .map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8)
                .collect())
        }
    }

    fn unreachable_params<const Q: u32>() -> LhsParams<Q> {
        LhsParams::<Q>::new(1, 1, 1, 1)
            .ok()
            .unwrap_or_else(unreachable_params)
    }

    #[test]
    fn sign_produces_signature_verifying_against_hash_target() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        let all_ok = keys.is_some_and(|(pk, sk)| {
            (0..params.k_pieces()).all(|i| {
                sign(&pk, &sk, i).ok().is_some_and(|sig| {
                    let sigma_zq: ZqVec<97> = sig.sigma().reduce_mod();
                    pk.a().mul_vec(&sigma_zq).ok().as_ref()
                        == pk.hash_targets().get(i)
                })
            })
        });
        assert!(all_ok);
    }

    #[test]
    fn sign_rejects_out_of_bounds_index() {
        let params = LhsParams::<97>::new(2, 2, 2, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let errored = keys.is_some_and(|(pk, sk)| sign(&pk, &sk, 2).is_err());
        assert!(errored);
    }

    #[test]
    fn sign_produces_signature_with_length_m() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        let len = keys.and_then(|(pk, sk)| sign(&pk, &sk, 0).ok().map(|sig| sig.len()));
        assert_eq!(len, Some(params.m()));
    }

    #[test]
    fn combine_singleton_is_scaling_over_z() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                let c = Zq::<97>::new(3);
                let combined = combine(&[(c, sig.clone())]).ok();
                let scaled: Vec<i64> = sig
                    .sigma()
                    .entries()
                    .iter()
                    .map(|&x| c.signed_repr() * x)
                    .collect();
                combined.map(|s| s.sigma().entries().to_vec()) == Some(scaled)
            })
        });
        assert!(ok);
    }

    #[test]
    fn combine_uses_balanced_signed_lift() {
        // Coefficient 95 in Zq<97> should lift to -2, not 95.
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                let c = Zq::<97>::new(95);
                let combined = combine(&[(c, sig.clone())]).ok();
                let scaled: Vec<i64> = sig
                    .sigma()
                    .entries()
                    .iter()
                    .map(|&x| (-2_i64) * x)
                    .collect();
                combined.map(|s| s.sigma().entries().to_vec()) == Some(scaled)
            })
        });
        assert!(ok);
    }

    #[test]
    fn combine_is_linear_over_signatures() {
        let params = LhsParams::<97>::new(2, 2, 2, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|s0| {
                sign(&pk, &sk, 1).ok().is_some_and(|s1| {
                    let c0 = Zq::<97>::new(2);
                    let c1 = Zq::<97>::new(95);
                    let combined = combine(&[(c0, s0.clone()), (c1, s1.clone())]).ok();
                    let manual: Vec<i64> = (0..s0.len())
                        .map(|j| {
                            c0.signed_repr() * s0.sigma().entries()[j]
                                + c1.signed_repr() * s1.sigma().entries()[j]
                        })
                        .collect();
                    combined.map(|s| s.sigma().entries().to_vec()) == Some(manual)
                })
            })
        });
        assert!(ok);
    }

    #[test]
    fn combined_signature_verifies_against_combined_target() {
        // A · (c0·σ0 + c1·σ1) ≡ c0·h0 + c1·h1 (mod Q).
        let params = LhsParams::<97>::new(2, 2, 2, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|s0| {
                sign(&pk, &sk, 1).ok().is_some_and(|s1| {
                    let c0 = Zq::<97>::new(3);
                    let c1 = Zq::<97>::new(5);
                    combine(&[(c0, s0), (c1, s1)]).ok().is_some_and(|sig| {
                        let sigma_zq: ZqVec<97> = sig.sigma().reduce_mod();
                        let lhs = pk.a().mul_vec(&sigma_zq).ok();
                        let rhs = ZqVec::linear_combine(
                            &[c0, c1],
                            &[
                                pk.hash_targets()[0].clone(),
                                pk.hash_targets()[1].clone(),
                            ],
                        )
                        .ok();
                        lhs == rhs
                    })
                })
            })
        });
        assert!(ok);
    }

    #[test]
    fn combine_propagates_length_mismatch() {
        let long = Signature::new(ZVec::new(vec![1, 2, 3]));
        let short = Signature::new(ZVec::new(vec![1]));
        assert!(
            combine::<97>(&[(Zq::new(1), long), (Zq::new(1), short)]).is_err()
        );
    }

    #[test]
    fn signature_norm_reports_sum_of_squares() {
        let sig = Signature::new(ZVec::new(vec![3, -4, 0]));
        assert_eq!(sig.squared_l2_norm(), 9 + 16);
    }
}
