//! Verification for the LHS scheme.
//!
//! The verifier accepts `σ ∈ Z^m` as a signature on `target ∈ Zq^n`
//! iff both of the following hold:
//!
//! 1. `||σ||² ≤ params.sig_norm_bound_sq()` (shortness);
//! 2. `A · σ ≡ target (mod Q)` (algebraic correctness).
//!
//! Dropping (1) admits the trivial "forgery" `σ = G⁻¹(target)` which
//! is algebraically correct but has no SIS-related hardness backing;
//! dropping (2) admits any short vector unrelated to `target`.  Both
//! checks are therefore necessary.
//!
//! The norm check is evaluated in the `Z` interpretation of `σ`; the
//! algebraic check reduces `σ` entrywise modulo `Q` before applying
//! `A`.  Any mismatch (including dimension mismatches) returns
//! [`Error::AuthenticatorRejected`] so callers can treat the verifier
//! as a single boolean oracle.

use crate::error::Error;
use crate::lattice::ZqVec;
use crate::lhs::keys::PublicKey;
use crate::lhs::sign::Signature;

/// Verify that `sig` is a valid LHS signature on `target` under `pk`.
///
/// Returns `Ok(())` when the signature passes both the norm bound and
/// the algebraic identity; otherwise returns
/// [`Error::AuthenticatorRejected`].
///
/// # Errors
///
/// - [`Error::AuthenticatorRejected`] if `sig.len() != params.m()`,
///   if `target.len() != params.n()`,
///   if `||σ||² > params.sig_norm_bound_sq()`,
///   or if `A · σ ≢ target (mod Q)`.
pub fn verify<const Q: u32>(
    pk: &PublicKey<Q>,
    sig: &Signature,
    target: &ZqVec<Q>,
) -> Result<(), Error> {
    check_dimensions(sig, target, pk)
        .and_then(|()| check_norm_bound(sig, pk))
        .and_then(|()| check_algebraic_identity(pk, sig, target))
}

fn check_dimensions<const Q: u32>(
    sig: &Signature,
    target: &ZqVec<Q>,
    pk: &PublicKey<Q>,
) -> Result<(), Error> {
    let params = pk.params();
    (sig.len() == params.m() && target.len() == params.n())
        .then_some(())
        .ok_or(Error::AuthenticatorRejected)
}

fn check_norm_bound<const Q: u32>(sig: &Signature, pk: &PublicKey<Q>) -> Result<(), Error> {
    (sig.squared_l2_norm() <= pk.params().sig_norm_bound_sq())
        .then_some(())
        .ok_or(Error::AuthenticatorRejected)
}

fn check_algebraic_identity<const Q: u32>(
    pk: &PublicKey<Q>,
    sig: &Signature,
    target: &ZqVec<Q>,
) -> Result<(), Error> {
    let sigma_zq: ZqVec<Q> = sig.sigma().reduce_mod();
    pk.a()
        .mul_vec(&sigma_zq)
        .map_err(|_| Error::AuthenticatorRejected)
        .and_then(|computed| {
            (computed == *target)
                .then_some(())
                .ok_or(Error::AuthenticatorRejected)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{ZVec, Zq};
    use crate::lhs::keys::keygen;
    use crate::lhs::params::LhsParams;
    use crate::lhs::sign::{combine, sign};
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
    fn fresh_signature_verifies_against_its_target() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            (0..pk.params().k_pieces()).all(|i| {
                sign(&pk, &sk, i)
                    .ok()
                    .is_some_and(|sig| verify(&pk, &sig, &pk.hash_targets()[i]).is_ok())
            })
        });
        assert!(ok);
    }

    #[test]
    fn combined_signature_verifies_against_combined_target() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            let sigs: Option<Vec<Signature>> = (0..pk.params().k_pieces())
                .map(|i| sign(&pk, &sk, i).ok())
                .collect();
            sigs.is_some_and(|sigs| {
                let coeffs = vec![Zq::<97>::new(2), Zq::<97>::new(3), Zq::<97>::new(1)];
                let pairs: Vec<(Zq<97>, Signature)> =
                    coeffs.iter().copied().zip(sigs).collect();
                let combined = combine(&pairs).ok();
                let target =
                    ZqVec::<97>::linear_combine(&coeffs, pk.hash_targets()).ok();
                combined
                    .and_then(|sig| target.map(|t| verify(&pk, &sig, &t)))
                    .is_some_and(|r| r.is_ok())
            })
        });
        assert!(ok);
    }

    #[test]
    fn wrong_target_is_rejected() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let rejected = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                let wrong = ZqVec::<97>::new(vec![Zq::new(1), Zq::new(1)]);
                verify(&pk, &sig, &wrong).is_err()
            })
        });
        assert!(rejected);
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let rejected = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                // Add 1 to entry 0: norm grows slightly, algebra breaks.
                let tampered_entries: Vec<i64> = sig
                    .sigma()
                    .entries()
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| if i == 0 { x + 1 } else { x })
                    .collect();
                let tampered = Signature::new(ZVec::new(tampered_entries));
                verify(&pk, &tampered, &pk.hash_targets()[0]).is_err()
            })
        });
        assert!(rejected);
    }

    #[test]
    fn algebraically_valid_but_norm_excessive_is_rejected() {
        // σ' = σ + 2Q·e_0 preserves A·σ' ≡ A·σ (mod Q) but bloats norm
        // beyond the bound.  Isolates the norm check as the reason for
        // rejection (algebra still passes).
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let rejected = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                let bloated_entries: Vec<i64> = sig
                    .sigma()
                    .entries()
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| if i == 0 { x + 2 * 97 } else { x })
                    .collect();
                let bloated = Signature::new(ZVec::new(bloated_entries));
                verify(&pk, &bloated, &pk.hash_targets()[0]).is_err()
            })
        });
        assert!(rejected);
    }

    #[test]
    fn short_signature_is_rejected() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let rejected = keys.is_some_and(|(pk, _)| {
            let short = Signature::new(ZVec::new(vec![0_i64; 3]));
            verify(&pk, &short, &pk.hash_targets()[0]).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn wrong_target_length_is_rejected() {
        let params = LhsParams::<97>::new(2, 2, 1, 10_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let rejected = keys.is_some_and(|(pk, sk)| {
            sign(&pk, &sk, 0).ok().is_some_and(|sig| {
                let wrong = ZqVec::<97>::zeros(5);
                verify(&pk, &sig, &wrong).is_err()
            })
        });
        assert!(rejected);
    }
}
