//! Linearly homomorphic signature [`Authenticator`] for RLNC gossip.
//!
//! Wires the BF11-style scheme from [`crate::lhs`] into the gossip
//! layer's [`Authenticator`] interface.  The authenticator publishes
//!
//! - the public key `pk = (A, h_1, …, h_k)`,
//! - the per-piece signatures `σ_1, …, σ_k` on the hash targets,
//! - a BLAKE3 commitment fingerprint over `(pk, σ_1, …, σ_k)` that
//!   identifies the generation.
//!
//! A tag for a coded piece with coding vector `cv ∈ GF(2^8)^k` is the
//! linear combination `σ = Σ cv_i · σ_i` computed by [`combine`] in
//! the lifted integer scheme.  The combination is *public*: any party
//! holding `(pk, σ_1, …, σ_k)` can tag any coded piece without seeing
//! `sk`, so a recoding relay can authenticate freshly mixed pieces.
//!
//! Verification recomputes the per-piece target
//! `target = Σ cv_i · h_i (mod Q)` and delegates to [`verify`], which
//! enforces both the algebraic identity `A·σ ≡ target (mod Q)` and
//! the shortness bound `||σ||² ≤ β²`.  Cross-generation replay is
//! blocked by comparing the wire commitment against the one cached at
//! construction.
//!
//! # BFKW09 broadcast model
//!
//! The signatures `σ_1, …, σ_k` form the **public transcript** of the
//! generation.  They are sampled once with `sk`, bound into the
//! commitment, and `sk` is never required again at tagging or
//! verification time.  Relays run [`Self::from_public_artifacts`] to
//! rebuild the authenticator from the broadcast tuple.
//!
//! # v0.1 binding limitation
//!
//! [`keygen`](crate::lhs::keygen) samples the hash targets `h_i`
//! uniformly at random.  They are not derived from any particular
//! [`OriginalData`], so this authenticator binds to the keypair and
//! per-generation signatures but not to the file contents: any
//! [`OriginalData`] with the matching piece count will pass.  A
//! production variant should derive `h_i = H(generation_metadata || i)`
//! before signing so the commitment is tied to the data.

use crate::auth::Authenticator;
use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;
use crate::lattice::{ZVec, Zq, ZqVec};
use crate::lhs::keys::{PublicKey, SecretKey};
use crate::lhs::sign::{Signature, combine, sign};
use crate::lhs::verify::verify;

/// Per-generation commitment: a BLAKE3 fingerprint of
/// `(pk, hash_targets, σ_originals)`.
///
/// Equality is constant-time via [`blake3::Hash`]'s built-in ct-eq,
/// so verifiers can compare wire commitments against the cached one
/// without leaking a timing side channel.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Commitment([u8; 32]);

impl From<[u8; 32]> for Commitment {
    fn from(bytes: [u8; 32]) -> Self {
        Commitment(bytes)
    }
}

impl From<Commitment> for [u8; 32] {
    fn from(c: Commitment) -> Self {
        c.0
    }
}

impl Commitment {
    /// Borrow the commitment bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl PartialEq for Commitment {
    fn eq(&self, other: &Self) -> bool {
        blake3::Hash::from_bytes(self.0) == blake3::Hash::from_bytes(other.0)
    }
}

impl Eq for Commitment {}

/// Linearly homomorphic signature [`Authenticator`] over `Z/QZ`.
///
/// Holds the public artefacts of one generation: the public key, the
/// per-piece signatures on its hash targets, and the commitment
/// fingerprint that binds them together.  All three are needed to
/// tag and verify coded pieces.
///
/// # Construction
///
/// - [`Self::new`] at the source side: signs every hash target once
///   from `(pk, sk)` and caches `sk`-free artefacts thereafter.
/// - [`Self::from_public_artifacts`] at relay and verifier sides:
///   rebuilds from the broadcast tuple `(pk, σ_1, …, σ_k)` without
///   access to `sk`.
#[derive(Clone, Debug)]
#[must_use]
pub struct LatticeHomomorphicAuthenticator<const Q: u32> {
    pk: PublicKey<Q>,
    signed_originals: Vec<Signature>,
    commitment: Commitment,
}

impl<const Q: u32> LatticeHomomorphicAuthenticator<Q> {
    /// Build an authenticator at the source side.
    ///
    /// Signs each of the `params.k_pieces()` hash targets with `sk`
    /// and caches the resulting `σ_i`.  After this call, tagging and
    /// verification use only `pk` and the cached signatures.
    ///
    /// # Errors
    ///
    /// - Propagates any error returned by [`sign`] (unreachable for
    ///   keys that came from a successful [`keygen`] call).
    ///
    /// [`keygen`]: crate::lhs::keygen
    pub fn new(pk: PublicKey<Q>, sk: &SecretKey<Q>) -> Result<Self, Error> {
        let k = pk.params().k_pieces();
        (0..k)
            .map(|i| sign(&pk, sk, i))
            .collect::<Result<Vec<Signature>, Error>>()
            .map(|signed_originals| {
                let commitment = derive_commitment(&pk, &signed_originals);
                Self {
                    pk,
                    signed_originals,
                    commitment,
                }
            })
    }

    /// Build an authenticator from the public broadcast tuple.
    ///
    /// Relays and verifiers use this constructor: they hold
    /// `(pk, σ_1, …, σ_k)` but no secret key, and can still tag and
    /// verify pieces because the homomorphic combine is a public
    /// computation.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if
    ///   `signed_originals.len() != pk.params().k_pieces()`.
    pub fn from_public_artifacts(
        pk: PublicKey<Q>,
        signed_originals: Vec<Signature>,
    ) -> Result<Self, Error> {
        let expected = pk.params().k_pieces();
        (signed_originals.len() == expected)
            .then_some(())
            .ok_or(Error::DimensionMismatch {
                expected,
                actual: signed_originals.len(),
            })
            .map(|()| {
                let commitment = derive_commitment(&pk, &signed_originals);
                Self {
                    pk,
                    signed_originals,
                    commitment,
                }
            })
    }

    /// Borrow the public key.
    pub fn public_key(&self) -> &PublicKey<Q> {
        &self.pk
    }

    /// Borrow the signed originals `σ_1, …, σ_k`.
    pub fn signed_originals(&self) -> &[Signature] {
        &self.signed_originals
    }

    /// Borrow the commitment fingerprint.
    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
}

impl<const Q: u32> Authenticator for LatticeHomomorphicAuthenticator<Q> {
    type Commitment = Commitment;
    type Tag = Signature;

    fn commit(&self, _original: &OriginalData) -> Commitment {
        self.commitment
    }

    fn tag(&self, _commitment: &Commitment, piece: &CodedPiece) -> Signature {
        lifted_pairs::<Q>(piece, &self.signed_originals)
            .and_then(|pairs| combine::<Q>(&pairs).ok())
            .unwrap_or_else(|| Signature::new(ZVec::new(Vec::new())))
    }

    fn verify(
        &self,
        commitment: &Commitment,
        piece: &CodedPiece,
        tag: &Signature,
    ) -> Result<(), Error> {
        (commitment == &self.commitment)
            .then_some(())
            .ok_or(Error::AuthenticatorRejected)
            .and_then(|()| {
                derive_target::<Q>(piece, self.pk.hash_targets())
                    .map_err(|_| Error::AuthenticatorRejected)
            })
            .and_then(|target| verify(&self.pk, tag, &target))
    }
}

fn lifted_pairs<const Q: u32>(
    piece: &CodedPiece,
    signed_originals: &[Signature],
) -> Option<Vec<(Zq<Q>, Signature)>> {
    let cv_bytes = piece.coding_vector().as_vec().to_bytes();
    (cv_bytes.len() == signed_originals.len()).then(|| {
        cv_bytes
            .iter()
            .zip(signed_originals.iter())
            .map(|(&b, sig)| (Zq::<Q>::new(u32::from(b)), sig.clone()))
            .collect()
    })
}

fn derive_target<const Q: u32>(
    piece: &CodedPiece,
    hash_targets: &[ZqVec<Q>],
) -> Result<ZqVec<Q>, Error> {
    let cv_bytes = piece.coding_vector().as_vec().to_bytes();
    (cv_bytes.len() == hash_targets.len())
        .then_some(())
        .ok_or(Error::DimensionMismatch {
            expected: hash_targets.len(),
            actual: cv_bytes.len(),
        })
        .and_then(|()| {
            let scalars: Vec<Zq<Q>> = cv_bytes
                .iter()
                .map(|&b| Zq::<Q>::new(u32::from(b)))
                .collect();
            ZqVec::<Q>::linear_combine(&scalars, hash_targets)
        })
}

fn derive_commitment<const Q: u32>(
    pk: &PublicKey<Q>,
    signed_originals: &[Signature],
) -> Commitment {
    let a = pk.a();
    let a_bytes = (0..a.rows())
        .flat_map(|i| (0..a.cols()).map(move |j| (i, j)))
        .flat_map(|(i, j)| a.entry(i, j).unwrap_or_else(Zq::zero).value().to_le_bytes());
    let target_bytes = pk
        .hash_targets()
        .iter()
        .flat_map(ZqVec::entries)
        .flat_map(|z| z.value().to_le_bytes());
    let sig_bytes = signed_originals
        .iter()
        .flat_map(|sig| sig.sigma().entries().iter().copied())
        .flat_map(i64::to_le_bytes);
    let payload: Vec<u8> = a_bytes.chain(target_bytes).chain(sig_bytes).collect();
    Commitment::from(*blake3::hash(&payload).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lhs::keys::keygen;
    use crate::lhs::params::LhsParams;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter_rng(seed: usize) -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
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
        LhsParams::<Q>::new(1, 1, 1, 1)
            .ok()
            .unwrap_or_else(unreachable_params)
    }

    fn sample_auth_with_seed(seed: usize) -> Option<LatticeHomomorphicAuthenticator<97>> {
        let params = LhsParams::<97>::new(2, 2, 4, 100_000_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng(seed)).ok();
        keys.and_then(|(pk, sk)| LatticeHomomorphicAuthenticator::new(pk, &sk).ok())
    }

    fn sample_auth() -> Option<LatticeHomomorphicAuthenticator<97>> {
        sample_auth_with_seed(0)
    }

    fn sample_piece<const Q: u32>(
        auth: &LatticeHomomorphicAuthenticator<Q>,
        coding_bytes: &[u8],
    ) -> CodedPiece {
        use crate::coding::piece::CodingVector;
        use crate::vector::GfVec;
        CodedPiece::new(
            CodingVector::from_bytes(coding_bytes),
            GfVec::from_bytes(&vec![0u8; auth.signed_originals.len()]),
        )
    }

    #[test]
    fn new_produces_k_signed_originals() {
        let auth = sample_auth();
        assert_eq!(auth.map(|a| a.signed_originals().len()), Some(4));
    }

    #[test]
    fn commitment_is_deterministic_for_fixed_artifacts() {
        let auth = sample_auth();
        let recloned = auth.as_ref().map(|a| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                a.public_key().clone(),
                a.signed_originals().to_vec(),
            )
            .ok()
        });
        let eq = auth.and_then(|a| {
            recloned
                .flatten()
                .map(|r| r.commitment() == a.commitment())
        });
        assert_eq!(eq, Some(true));
    }

    #[test]
    fn from_public_artifacts_rejects_wrong_count() {
        let auth = sample_auth();
        let built = auth.map(|a| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                a.public_key().clone(),
                a.signed_originals().iter().take(2).cloned().collect(),
            )
        });
        assert!(built.is_some_and(|r| r.is_err()));
    }

    #[test]
    fn fresh_tag_verifies() {
        let auth = sample_auth();
        let ok = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 2, 3, 4]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            a.verify(&commitment, &piece, &tag).is_ok()
        });
        assert!(ok);
    }

    #[test]
    fn tag_on_standard_basis_matches_signed_original() {
        // cv = e_i (a single 1, rest 0) lifts to [0, ..., 1, ..., 0] in Zq;
        // the combined signature is exactly signed_originals[i].
        let auth = sample_auth();
        let ok = auth.is_some_and(|a| {
            (0..a.signed_originals().len()).all(|i| {
                let cv: Vec<u8> = (0..a.signed_originals().len())
                    .map(|j| u8::from(j == i))
                    .collect();
                let piece = sample_piece(&a, &cv);
                let commitment = *a.commitment();
                let tag = a.tag(&commitment, &piece);
                tag == a.signed_originals()[i]
            })
        });
        assert!(ok);
    }

    #[test]
    fn wrong_commitment_rejects_tag() {
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 0, 0, 0]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            let bogus = Commitment::from([0xFF; 32]);
            a.verify(&bogus, &piece, &tag).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn tag_from_foreign_generation_rejects() {
        // Two independent authenticators (distinct RNG seeds) have
        // independent signed_originals and hash_targets, so a tag
        // produced by one fails verification by the other even if the
        // commitment is forged to match.
        let auth_a = sample_auth_with_seed(0);
        let auth_b = sample_auth_with_seed(10_000);
        let rejected = auth_a.zip(auth_b).is_some_and(|(a, b)| {
            let piece = sample_piece(&a, &[1, 2, 3, 4]);
            let commitment_a = *a.commitment();
            let tag_a = a.tag(&commitment_a, &piece);
            let commitment_b = *b.commitment();
            b.verify(&commitment_b, &piece, &tag_a).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn wrong_coding_vector_length_rejects() {
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 2, 3]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            a.verify(&commitment, &piece, &tag).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn mismatched_tag_and_coding_vector_rejects() {
        // Tag was generated for cv_a, but the wire piece carries cv_b.
        // derive_target(cv_b) mismatches A·tag, so verify rejects.
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece_a = sample_piece(&a, &[1, 0, 0, 0]);
            let piece_b = sample_piece(&a, &[0, 1, 0, 0]);
            let commitment = *a.commitment();
            let tag_a = a.tag(&commitment, &piece_a);
            a.verify(&commitment, &piece_b, &tag_a).is_err()
        });
        assert!(rejected);
    }
}
