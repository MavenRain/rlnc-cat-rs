//! Sign and Combine operations for the LHS scheme.
//!
//! The signature on hash target `h ∈ Zq^n` is the integer vector
//! `σ = [R·w ; w] ∈ Z^m`, where `w` is a short preimage of `h` under
//! the gadget matrix: `G · w ≡ h (mod Q)`.  The gadget trapdoor
//! identity gives
//!
//! ```text
//! A · σ = A_0·(R·w) + A_1·w
//!       = A_0·R·w + (G - A_0·R)·w
//!       = G·w ≡ h  (mod Q).
//! ```
//!
//! `w` is produced in two stages:
//!
//! 1. Compute the deterministic bit-decomposition `w_0 = bits(h)`.
//!    This satisfies `G · w_0 = h` exactly over `Z`.
//! 2. For each `n`-wise gadget block, draw a Gaussian offset
//!    `v ∈ Λ^⊥_q(g)` centered at `-w_0` with width `σ_g` via Klein's
//!    algorithm, and set the block to `w_0 + v`.  Adding a kernel
//!    element preserves `G · w ≡ h (mod Q)`; the combined draw is
//!    statistically close to the canonical discrete Gaussian on the
//!    coset `w_0 + Λ^⊥_q(g)` (BFKW09 / Gentry-Peikert).
//!
//! Linear homomorphism follows: a `Zq`-linear combination of
//! signatures (lifted to `Z` via the balanced signed representative)
//! is itself a valid signature on the matching combination of hash
//! targets, and the composition is performed by [`combine`] without
//! the secret key.

use crate::error::Error;
use crate::lattice::{PreimageContext, ZMatrix, ZVec, Zq, ZqVec};
use crate::lhs::gadget::gadget_preimage;
use crate::lhs::keys::{PublicKey, SecretKey};
use crate::lhs::params::LhsParams;

/// A BFKW09-style signature: the integer vector `σ ∈ Z^m` whose
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

/// Sign the hash target `target ∈ Zq^n`.
///
/// Produces `σ = [R·w ; w]` where `w = w_0 + v`, `w_0` is the
/// deterministic bit-decomposition preimage under `G`, and `v` is a
/// Klein-sampled Gaussian vector from the gadget kernel `Λ^⊥_q(G)`
/// centered at `-w_0` with width `params.sigma_g()`.  The resulting
/// `w` remains in the coset `w_0 + Λ^⊥_q(G)` so `G · w ≡ target (mod Q)`
/// by construction; shortness is inherited from the Gaussian bound on
/// `w` together with the ternary structure of `R`.
///
/// # Errors
///
/// - [`Error::DimensionMismatch`] if `target.len() != pk.params().n()`,
///   or if the trapdoor matrix shape does not match the gadget width
///   (unreachable when `pk` and `sk` come from the same [`keygen`]
///   call).
/// - [`Error::RandomGenerationFailed`] propagated from Klein's
///   discrete-Gaussian rejection sampler.
///
/// [`keygen`]: crate::lhs::keys::keygen
pub fn sign<const Q: u32, F>(
    pk: &PublicKey<Q>,
    sk: &SecretKey<Q>,
    target: &ZqVec<Q>,
    rng: &F,
) -> Result<Signature, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    let n = pk.params().n();
    let k = LhsParams::<Q>::k_gadget();
    let sigma_g = pk.params().sigma_g();
    (target.len() == n)
        .then_some(())
        .ok_or(Error::DimensionMismatch {
            expected: n,
            actual: target.len(),
        })
        .and_then(|()| {
            let w_0 = gadget_preimage::<Q>(target);
            randomized_preimage(&w_0, sk.gadget_kernel_ctx(), sigma_g, n, k, rng)
        })
        .and_then(|w| {
            let w_zvec = ZVec::new(w);
            sk.r().mul_vec(&w_zvec).map(|r_w| {
                let entries: Vec<i64> = r_w
                    .entries()
                    .iter()
                    .copied()
                    .chain(w_zvec.entries().iter().copied())
                    .collect();
                Signature::new(ZVec::new(entries))
            })
        })
}

/// Apply per-block Klein sampling on top of the deterministic
/// bit-decomposition `w_0`.  The gadget kernel `Λ^⊥_q(G)` is the
/// `n`-fold direct sum of `Λ^⊥_q(g)`, so one shared `k × k` context
/// suffices and each block is sampled independently.
fn randomized_preimage<F>(
    w_0: &[i64],
    ctx: &PreimageContext,
    sigma_g: f64,
    n: usize,
    k: usize,
    rng: &F,
) -> Result<Vec<i64>, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    (0..n)
        .map(|i| {
            let start = i * k;
            let end = start + k;
            randomize_block(&w_0[start..end], ctx, sigma_g, rng)
        })
        .collect::<Result<Vec<Vec<i64>>, Error>>()
        .map(|blocks| blocks.into_iter().flatten().collect())
}

fn randomize_block<F>(
    w_0_block: &[i64],
    ctx: &PreimageContext,
    sigma_g: f64,
    rng: &F,
) -> Result<Vec<i64>, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    let target_f: Vec<f64> = w_0_block.iter().map(|&x| negate_as_f64(x)).collect();
    ctx.gaussian_preimage(&target_f, sigma_g, rng).map(|zs| {
        let v = lattice_point(zs.entries(), ctx.basis());
        w_0_block
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| a + b)
            .collect()
    })
}

/// `i64 → f64` followed by negation.  The cast is precision-lossy for
/// `|x| > 2^53`, but Klein's per-coordinate budget is `±12·σ_g` — a
/// few dozen at most for this scheme — so the magnitudes involved are
/// well within the f64-exact integer range.
fn negate_as_f64(x: i64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let xf = x as f64;
    -xf
}

/// Compute the lattice point `Σ zs[i] · basis[i]` over `Z`.  The
/// rows of `basis` are the lattice generators, so this is a left
/// multiplication of the row vector `zs` by the matrix.
fn lattice_point(zs: &[i64], basis: &ZMatrix) -> Vec<i64> {
    (0..basis.cols())
        .map(|c| {
            zs.iter().enumerate().fold(0_i64, |acc, (i, &z)| {
                let bic = basis
                    .row(i)
                    .and_then(|r| r.get(c).copied())
                    .unwrap_or(0);
                acc + z * bic
            })
        })
        .collect()
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
        counter_rng_with_seed(0)
    }

    fn counter_rng_with_seed(
        seed: usize,
    ) -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(seed));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n)
                .map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8)
                .collect())
        }
    }

    fn unreachable_params<const Q: u32>() -> LhsParams<Q> {
        LhsParams::<Q>::new(1, 1, 1, 1, 1.0)
            .ok()
            .unwrap_or_else(unreachable_params)
    }

    /// Deterministic length-2 target derived from an index; every index
    /// yields a distinct vector so sign/combine tests can operate on
    /// independent targets without pulling in an RNG.
    fn fresh_target<const Q: u32>(i: usize) -> ZqVec<Q> {
        let seed = u32::try_from(i).unwrap_or(0);
        ZqVec::new(vec![Zq::new(seed.wrapping_add(1)), Zq::new(seed.wrapping_mul(3).wrapping_add(7))])
    }

    #[test]
    fn sign_produces_signature_verifying_against_target() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        let all_ok = keys.is_some_and(|(pk, sk)| {
            (0..params.k_pieces()).all(|i| {
                let target = fresh_target::<97>(i);
                sign(&pk, &sk, &target, &counter_rng()).ok().is_some_and(|sig| {
                    let sigma_zq: ZqVec<97> = sig.sigma().reduce_mod();
                    pk.a().mul_vec(&sigma_zq).ok() == Some(target)
                })
            })
        });
        assert!(all_ok);
    }

    #[test]
    fn sign_rejects_wrong_target_length() {
        let params = LhsParams::<97>::new(2, 2, 2, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let errored = keys.is_some_and(|(pk, sk)| {
            let bad = ZqVec::<97>::new(vec![Zq::new(1), Zq::new(2), Zq::new(3)]);
            sign(&pk, &sk, &bad, &counter_rng()).is_err()
        });
        assert!(errored);
    }

    #[test]
    fn sign_produces_signature_with_length_m() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        let target = fresh_target::<97>(0);
        let len = keys.and_then(|(pk, sk)| {
            sign(&pk, &sk, &target, &counter_rng()).ok().map(|sig| sig.len())
        });
        assert_eq!(len, Some(params.m()));
    }

    fn blake3_rng(seed: u64) -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(0));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            let mut payload = Vec::with_capacity(24);
            payload.extend_from_slice(&seed.to_le_bytes());
            payload.extend_from_slice(&u64::try_from(i).unwrap_or(u64::MAX).to_le_bytes());
            payload.extend_from_slice(&u64::try_from(n).unwrap_or(u64::MAX).to_le_bytes());
            let hash = blake3::hash(&payload);
            let bytes = hash.as_bytes();
            Ok((0..n).map(|j| bytes[j % 32]).collect())
        }
    }

    #[test]
    fn sign_is_randomized_across_rng_seeds() {
        // Same (pk, sk, target) with distinct RNG streams should yield
        // distinct σ: the Klein offset over the gadget kernel is the
        // source of randomness.  If this ever collides, the Klein
        // sampler is effectively deterministic and the BFKW09
        // distribution guarantee is broken.  Uses a BLAKE3-based
        // counter RNG so the seed diffuses into every byte rather
        // than only shifting the counter's starting position.
        let params = LhsParams::<97>::new(2, 2, 1, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let target = fresh_target::<97>(0);
        let differ = keys.is_some_and(|(pk, sk)| {
            let sig_a = sign(&pk, &sk, &target, &blake3_rng(0xA1)).ok();
            let sig_b = sign(&pk, &sk, &target, &blake3_rng(0xB2)).ok();
            sig_a.zip(sig_b).is_some_and(|(a, b)| a != b)
        });
        assert!(differ);
    }

    #[test]
    fn combine_singleton_is_scaling_over_z() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let target = fresh_target::<97>(0);
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, &target, &counter_rng()).ok().is_some_and(|sig| {
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
        let params = LhsParams::<97>::new(2, 2, 1, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let target = fresh_target::<97>(0);
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, &target, &counter_rng()).ok().is_some_and(|sig| {
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
        let params = LhsParams::<97>::new(2, 2, 2, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let t0 = fresh_target::<97>(0);
        let t1 = fresh_target::<97>(1);
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, &t0, &counter_rng()).ok().is_some_and(|s0| {
                sign(&pk, &sk, &t1, &counter_rng()).ok().is_some_and(|s1| {
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
        let params = LhsParams::<97>::new(2, 2, 2, 10_000_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let t0 = fresh_target::<97>(0);
        let t1 = fresh_target::<97>(1);
        let ok = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, &t0, &counter_rng()).ok().is_some_and(|s0| {
                sign(&pk, &sk, &t1, &counter_rng()).ok().is_some_and(|s1| {
                    let c0 = Zq::<97>::new(3);
                    let c1 = Zq::<97>::new(5);
                    combine(&[(c0, s0), (c1, s1)]).ok().is_some_and(|sig| {
                        let sigma_zq: ZqVec<97> = sig.sigma().reduce_mod();
                        let lhs = pk.a().mul_vec(&sigma_zq).ok();
                        let rhs = ZqVec::linear_combine(
                            &[c0, c1],
                            &[t0.clone(), t1.clone()],
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

    #[test]
    fn signature_empirical_variance_scales_with_sigma_g() {
        // BFKW09 statistical-closeness check: σ = R·z + w₀ where w₀ is
        // deterministic and z ~ D_{Λ⊥(G), σ_g}, so Var(σ[0]) across RNG
        // seeds scales with σ_g².  Doubling σ_g should at minimum
        // double the empirical variance (expected: ≈ 4× the variance).
        let collect_entry0 = |sigma_g: f64| -> Option<Vec<f64>> {
            LhsParams::<97>::new(2, 2, 1, 100_000_000, sigma_g)
                .ok()
                .and_then(|params| keygen(params, &counter_rng()).ok())
                .and_then(|(pk, sk)| {
                    let target = fresh_target::<97>(0);
                    (0..300)
                        .map(|seed| {
                            sign(&pk, &sk, &target, &blake3_rng(seed)).ok().map(|s| {
                                #[allow(clippy::cast_precision_loss)]
                                let v = s.sigma().entries()[0] as f64;
                                v
                            })
                        })
                        .collect::<Option<Vec<f64>>>()
                })
        };
        let sample_variance = |data: &[f64]| {
            #[allow(clippy::cast_precision_loss)]
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
        };
        let ok = collect_entry0(3.0)
            .zip(collect_entry0(6.0))
            .is_some_and(|(a, b)| {
                let va = sample_variance(&a);
                let vb = sample_variance(&b);
                vb > 2.0 * va
            });
        assert!(ok);
    }
}
